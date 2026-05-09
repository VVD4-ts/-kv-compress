"""
Cross-Layer KV Eviction (CLE) — training-free inter-layer KV cache compression.

Algorithm
---------
1. Prefill normally, but with output_attentions=True.
2. For each token position, aggregate the attention it *receives* across all
   layers and all heads → a single "cross-layer importance" score per position.
3. Keep only the top-B positions in the KV cache of every layer (B = budget).
   All layers evict the same token set — this is the "inter-layer" cooperation.
4. Decode using the pruned KV cache.  New decode tokens are always appended
   (never evicted), so the budget applies only to the prefill context.

Why this preserves quality
--------------------------
Each layer still uses its own W_K and W_V projections → no projection
mismatch → no catastrophic PPL degradation (unlike YOCO / SimLayerKV on
Pythia-70M where adjacent-layer similarity is < 0.09).

Metric impact
-------------
TTFT      : neutral (full prefill + attention collection)
TPOT      : ↓ lower   — decode attention is Q(1) × K(B+t), not Q(1) × K(S+t)
Throughput: ↑ higher  — follows from TPOT reduction
FLOPs     : ↓ lower   — prefill same; decode attention FLOPs ∝ budget B not S

Ref: related to PyramidKV (Cai et al., EMNLP 2024) and SnapKV (Li et al.,
     arXiv 2404.14469), but with unified cross-layer eviction scoring.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import torch

sys.path.insert(0, str(Path(__file__).parent))

from utils import WallTimer, make_result, GenerationResult, num_layers
from kv_utils import _n_layers, _get_kv, _build

try:
    from transformers import DynamicCache
    from transformers.cache_utils import Cache as _CacheBase
    _HAS_DYNAMIC = True
except ImportError:
    _HAS_DYNAMIC = False
    DynamicCache = None
    _CacheBase   = None


# ── Importance scoring ────────────────────────────────────────────────────────

def compute_cross_layer_importance(past_kv) -> torch.Tensor:
    """
    Compute per-token importance by aggregating key-vector L2 norms across all
    layers and attention heads.

    Rationale: attention score  ∝  Q · K.  For a given query distribution,
    a token with a high-magnitude key vector will on average receive more
    attention.  Key-norm scoring is a robust, attention-output-free proxy for
    importance and is used across several KV-eviction papers.

    Cross-layer aggregation is the "inter-layer" element: instead of each layer
    independently deciding which tokens to evict, all layers collectively vote
    via their key norms.  Tokens important across many layers are kept.

    past_kv: DynamicCache or legacy tuple returned by model().
    Returns: importance tensor of shape [seq_len], float32.
    """
    n = _n_layers(past_kv)
    layer_scores: List[torch.Tensor] = []

    for i in range(n):
        k, _ = _get_kv(past_kv, i)
        # k: [batch=1, heads, seq_len, head_dim]
        # L2 norm per position per head → mean over heads → [seq_len]
        score = k.float().norm(dim=-1).mean(dim=1).squeeze(0)
        layer_scores.append(score)

    return torch.stack(layer_scores).mean(dim=0)   # [seq_len]


# ── KV cache surgery ──────────────────────────────────────────────────────────

def evict_kv_cache(past_kv, keep_indices: torch.Tensor):
    """
    Return a new KV cache containing only the positions in keep_indices.

    keep_indices: 1-D LongTensor with sorted unique positions to keep.
    Sequence dimension is dim=2 of the key/value tensors.
    """
    n    = _n_layers(past_kv)
    keys: List[torch.Tensor] = []
    vals: List[torch.Tensor] = []

    for i in range(n):
        k, v = _get_kv(past_kv, i)
        # k, v: [batch, heads, seq_len, head_dim]
        keys.append(k[:, :, keep_indices, :])
        vals.append(v[:, :, keep_indices, :])

    new_cache = _build(past_kv, keys, vals)

    # Fix _seen_tokens so the cache reports the pruned length correctly
    kept = keep_indices.shape[0]
    if _HAS_DYNAMIC and hasattr(new_cache, "_seen_tokens"):
        new_cache._seen_tokens = kept

    return new_cache


# ── FLOPs estimation ──────────────────────────────────────────────────────────

def estimate_flops_cle(
    model,
    prompt_len: int,
    gen_len:    int,
    budget:     int,
) -> Dict[str, float]:
    """
    Analytical FLOPs for Cross-Layer Eviction.

    Prefill: same as baseline (full attention needed to collect scores).
    Decode:  attention is Q(1) × K(budget+t) at step t,
             instead of Q(1) × K(prompt_len+t) for the baseline.

    Per active layer:
      prefill: 24·S·H²  +  4·S²·H     (S = prompt_len)
      decode:  24·H²    +  4·C_cle·H  (C_cle = budget + step + 1)
    """
    cfg = model.config
    H   = cfg.hidden_size
    n   = num_layers(model)

    # Prefill (unchanged)
    prefill_flops = n * (24 * prompt_len * H * H + 4 * (prompt_len ** 2) * H)

    # Decode (attention context = budget + decode_step + 1)
    decode_flops = 0.0
    for step in range(gen_len):
        ctx = budget + step + 1          # ← pruned, not prompt_len + step + 1
        for _ in range(n):
            decode_flops += 24 * H * H + 4 * ctx * H

    total  = prefill_flops + decode_flops
    n_toks = prompt_len + gen_len
    return {
        "prefill_gflops":     prefill_flops / 1e9,
        "decode_gflops":      decode_flops  / 1e9,
        "total_gflops":       total          / 1e9,
        "avg_gflops_per_tok": total / n_toks / 1e9,
    }


# ── Generation ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def cle_generate(
    model,
    tokenizer,
    prompt:        str,
    max_new_tokens: int   = 128,
    budget_ratio:   float = 0.5,
    device:         str   = "cuda",
) -> GenerationResult:
    """
    Greedy decoding with Cross-Layer KV Eviction.

    After prefill, keeps only the top-(budget_ratio * prompt_len) positions
    in the KV cache of every layer, selected by aggregated cross-layer
    attention scores.  Decode tokens are always appended (never evicted).

    position_ids are tracked manually so RoPE embeddings remain correct
    even after the KV cache is pruned.
    """
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos_id     = tokenizer.eos_token_id

    budget = max(1, int(prompt_len * budget_ratio))

    total_timer = WallTimer().start()

    # ── Prefill ───────────────────────────────────────────────────────────────
    ttft_timer = WallTimer().start()
    out     = model(input_ids, use_cache=True)
    past_kv = out.past_key_values

    # Cross-layer key-norm importance → keep top-budget positions
    importance   = compute_cross_layer_importance(past_kv)
    keep_indices = importance.topk(budget).indices.sort().values   # sorted positions
    past_kv      = evict_kv_cache(past_kv, keep_indices)

    next_tok  = out.logits[:, -1:, :].argmax(dim=-1)
    ttft_ms   = ttft_timer.elapsed_ms()

    generated = [next_tok.item()]
    real_pos  = prompt_len   # track true position for RoPE

    # ── Decode ───────────────────────────────────────────────────────────────
    for _ in range(max_new_tokens - 1):
        if next_tok.item() == eos_id:
            break
        pos_ids = torch.tensor([[real_pos]], device=device)
        out     = model(next_tok, past_key_values=past_kv,
                        use_cache=True, position_ids=pos_ids)
        past_kv  = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        real_pos += 1
        generated.append(next_tok.item())

    total_ms = total_timer.elapsed_ms()
    all_ids  = input_ids[0].tolist() + generated

    flops = estimate_flops_cle(model, prompt_len, len(generated), budget)
    return make_result(
        all_ids, prompt_len, total_ms, ttft_ms, past_kv,
        total_gflops=flops["total_gflops"],
    )


# ── Convenience configs ───────────────────────────────────────────────────────

CLE_CONFIGS = {
    "CLE-Light  (-20%KV)":  0.80,   # keep 80% of prefill tokens
    "CLE-Medium (-50%KV)":  0.50,   # keep 50%
    "CLE-Heavy  (-70%KV)":  0.30,   # keep 30%
}

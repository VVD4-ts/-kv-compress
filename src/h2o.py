"""
H2O – Heavy Hitter Oracle KV cache eviction.

Idea
----
Zhang et al. (NeurIPS 2023) show that a small subset of tokens ("heavy
hitters") accumulate the majority of attention mass across all layers and
steps.  Keeping only these heavy hitters + the most recent tokens preserves
generation quality with a fixed KV budget.

Algorithm (per decode step)
---------------------------
1. Run one forward pass with output_attentions=True.
2. Add the new attention weights to a per-layer cumulative score tensor.
3. If the KV cache exceeds `budget`, evict the lowest-scoring non-sink,
   non-recent positions.

Training-free guarantee
-----------------------
No weight updates.  Only the KV cache is pruned between forward passes.

Memory
------
Cache bounded by `budget` tokens per layer (constant after warm-up).
Effective KV reduction ≈ (T − budget) / T for long sequences.

Reference
---------
Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference
of Large Language Models", NeurIPS 2023.
https://arxiv.org/abs/2306.14048
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import WallTimer, make_result, GenerationResult
from kv_utils import _n_layers, _get_kv, _build, kv_cache_length


# ── Attention score accumulation ──────────────────────────────────────────────

def accumulate_h2o_scores(
    scores: List[torch.Tensor],
    attentions,                          # tuple of [1, heads, q_len, kv_len] per layer
) -> List[torch.Tensor]:
    """
    Add the latest attention weights to the running per-layer H2O scores.

    scores[i] is a 1-D tensor of length kv_len giving the cumulative attention
    mass received by each cached position at layer i.
    """
    for i, attn in enumerate(attentions):
        if attn is None:
            continue
        # attn: [1, heads, q_len, kv_len] – take mean over heads and queries
        new_score = attn[0].float().mean(dim=0).mean(dim=0)   # [kv_len]
        if i >= len(scores):
            scores.append(new_score.clone())
        else:
            cur = scores[i]
            kv_len = new_score.shape[0]
            if kv_len > cur.shape[0]:
                # Cache grew (new token appended) – extend with the new position's score
                extra = new_score[cur.shape[0]:]
                scores[i] = torch.cat([cur + new_score[:cur.shape[0]], extra])
            else:
                scores[i] = cur[:kv_len] + new_score
    return scores


# ── Eviction ──────────────────────────────────────────────────────────────────

def apply_h2o(
    past_kv,
    scores:   List[torch.Tensor],
    budget:   int,
    n_recent: int,
    n_sink:   int,
) -> tuple:
    """
    Evict lowest-scoring non-sink, non-recent positions from every layer.

    Returns (new_past_kv, new_scores) with lengths trimmed to `budget`.
    """
    seq_len = kv_cache_length(past_kv)
    if seq_len <= budget:
        return past_kv, scores

    # Aggregate importance across layers
    if scores:
        agg = torch.stack([s[:seq_len] for s in scores]).mean(dim=0)   # [seq_len]
    else:
        agg = torch.zeros(seq_len)

    # Protect sink tokens and recent tokens
    agg[:n_sink]   = float("inf")
    agg[-n_recent:] = float("inf")

    keep_indices = agg.topk(budget).indices.sort().values

    n    = _n_layers(past_kv)
    keys: List[torch.Tensor] = []
    vals: List[torch.Tensor] = []
    for i in range(n):
        k, v = _get_kv(past_kv, i)
        keys.append(k[:, :, keep_indices, :])
        vals.append(v[:, :, keep_indices, :])

    new_past_kv = _build(past_kv, keys, vals)
    new_scores  = [s[keep_indices] if i < len(scores) else s
                   for i, s in enumerate(scores)]
    return new_past_kv, new_scores


# ── Generation ────────────────────────────────────────────────────────────────

def _set_eager(model):
    """Force eager attention so output_attentions=True works on all backends."""
    try:
        for m in model.modules():
            if hasattr(m, "_attn_implementation"):
                m._attn_implementation = "eager"
    except Exception:
        pass


@torch.inference_mode()
def h2o_generate(
    model,
    tokenizer,
    prompt:         str,
    max_new_tokens: int = 128,
    budget:         int = 256,
    n_recent:       int = 32,
    n_sink:         int = 4,
    device:         str = "cuda",
) -> GenerationResult:
    """
    Greedy decoding with H2O dynamic KV eviction.

    Accumulates per-layer attention scores, evicts lowest-scoring positions
    whenever the cache exceeds `budget`.  Sink tokens (first n_sink) and
    the most recent n_recent tokens are always kept.
    """
    _set_eager(model)

    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos_id     = tokenizer.eos_token_id

    total_timer = WallTimer().start()
    ttft_timer  = WallTimer().start()

    out     = model(input_ids, use_cache=True, output_attentions=True)
    past_kv = out.past_key_values
    scores: List[torch.Tensor] = []
    if out.attentions:
        scores = accumulate_h2o_scores(scores, out.attentions)

    # Evict if prefill already exceeds budget
    past_kv, scores = apply_h2o(past_kv, scores, budget, n_recent, n_sink)

    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    ttft_ms  = ttft_timer.elapsed_ms()

    generated = [next_tok.item()]
    real_pos  = prompt_len

    for _ in range(max_new_tokens - 1):
        if next_tok.item() == eos_id:
            break
        pos_ids = torch.tensor([[real_pos]], device=device)
        out     = model(next_tok, past_key_values=past_kv,
                        use_cache=True, position_ids=pos_ids,
                        output_attentions=True)
        past_kv = out.past_key_values

        if out.attentions:
            scores = accumulate_h2o_scores(scores, out.attentions)

        past_kv, scores = apply_h2o(past_kv, scores, budget, n_recent, n_sink)

        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        real_pos += 1
        generated.append(next_tok.item())

    total_ms = total_timer.elapsed_ms()
    all_ids  = input_ids[0].tolist() + generated
    return make_result(all_ids, prompt_len, total_ms, ttft_ms, past_kv)


# ── Configs ───────────────────────────────────────────────────────────────────

H2O_CONFIGS = {
    "H2O(budget=128)": {"budget": 128, "n_recent": 32, "n_sink": 4},
    "H2O(budget=256)": {"budget": 256, "n_recent": 32, "n_sink": 4},
}

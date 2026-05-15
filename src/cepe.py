"""
Context Expansion with Parallel Encoding (CEPE) — KV compression variant.

Training-free approximation
---------------------------
Without a dedicated encoder, we approximate the "compressed context" idea by
average-pooling the accumulated KV cache:

  - The most recent `keep_recent` token positions are kept verbatim (local
    attention remains exact).
  - Older positions are grouped into non-overlapping blocks of `pool_size`
    and each block is replaced by its per-head mean key and value tensor.

This yields a compressed prefix that grows at rate 1/pool_size instead of
linearly, bounding memory use for long-context generation.

Why this is better than StreamingLLM
--------------------------------------
StreamingLLM discards all old tokens beyond the sink window — the model
completely forgets them.  CEPE pooling retains a compressed summary of every
old token: the block-averaged key/value still carries semantic signal (e.g.,
topic, entity mentions) that pure truncation loses.

Compatibility with PLD
----------------------
PLD n-gram search should be restricted to the recent `keep_recent` tokens
(whose token IDs are still known exactly).  Pooled old tokens have no
corresponding token IDs, so they cannot serve as n-gram candidates.

Reference
---------
Yen et al., "Long-Context Language Modeling with Parallel Context Encoding",
ACL 2024.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent))

from utils import WallTimer, make_result, GenerationResult, num_layers
from kv_utils import _n_layers, _get_kv, _build, kv_cache_length

try:
    from transformers import DynamicCache
    _HAS_DYNAMIC = True
except ImportError:
    _HAS_DYNAMIC = False
    DynamicCache = None


# ── Core pooling operation ────────────────────────────────────────────────────

def apply_cepe_pooling(past_kv, keep_recent: int, pool_size: int):
    """
    Compress old KV positions via average pooling, keep recent positions exact.

    Layout after pooling:
        [ pooled_block_0 | pooled_block_1 | ... | exact_recent_tokens ]

    Positions 0 .. (seq_len - keep_recent - 1) are pooled into blocks of
    pool_size (last incomplete block is also averaged).
    Positions (seq_len - keep_recent) .. (seq_len - 1) are kept unchanged.

    Returns new cache with length:
        ceil(old_len / pool_size) + keep_recent   (≤ old_len)
    """
    cur_len = kv_cache_length(past_kv)
    old_len = cur_len - keep_recent          # number of positions to pool
    if old_len <= pool_size:                 # nothing to compress yet
        return past_kv

    n = _n_layers(past_kv)
    new_keys: List[torch.Tensor] = []
    new_vals: List[torch.Tensor] = []

    for i in range(n):
        k, v = _get_kv(past_kv, i)
        # k: [1, heads, seq_len, head_dim]
        old_k = k[:, :, :old_len, :]        # positions to pool
        rec_k = k[:, :, old_len:, :]        # recent positions to keep

        # Split into blocks of pool_size, average each block
        n_full   = old_len // pool_size
        pooled_blocks_k = []
        pooled_blocks_v = []
        old_v = v[:, :, :old_len, :]
        rec_v = v[:, :, old_len:, :]

        for b in range(n_full):
            start = b * pool_size
            end   = start + pool_size
            pooled_blocks_k.append(old_k[:, :, start:end, :].mean(dim=2, keepdim=True))
            pooled_blocks_v.append(old_v[:, :, start:end, :].mean(dim=2, keepdim=True))

        # Remaining partial block
        remainder = old_len - n_full * pool_size
        if remainder > 0:
            pooled_blocks_k.append(old_k[:, :, -remainder:, :].mean(dim=2, keepdim=True))
            pooled_blocks_v.append(old_v[:, :, -remainder:, :].mean(dim=2, keepdim=True))

        pooled_k = torch.cat(pooled_blocks_k, dim=2)   # [1, heads, n_blocks, head_dim]
        pooled_v = torch.cat(pooled_blocks_v, dim=2)

        new_keys.append(torch.cat([pooled_k, rec_k], dim=2))
        new_vals.append(torch.cat([pooled_v, rec_v], dim=2))

    new_cache = _build(past_kv, new_keys, new_vals)
    new_len   = new_keys[0].shape[2]
    if _HAS_DYNAMIC and hasattr(new_cache, "_seen_tokens"):
        new_cache._seen_tokens = new_len
    return new_cache


# ── FLOPs estimation ─────────────────────────────────────────────────────────

def estimate_flops_cepe(
    model,
    prompt_len:   int,
    gen_len:      int,
    keep_recent:  int,
    pool_size:    int,
) -> Dict[str, float]:
    """
    Analytical FLOPs for CEPE pooling.

    After warm-up the effective context length is:
        ceil(prompt_len / pool_size) + keep_recent + decode_step
    which grows much slower than prompt_len + decode_step.
    """
    cfg = model.config
    H   = cfg.hidden_size
    n   = num_layers(model)

    # Prefill: full attention
    prefill_flops = n * (24 * prompt_len * H * H + 4 * (prompt_len ** 2) * H)

    # Effective pooled prefix length after first compression
    import math
    pooled_prefix = math.ceil(prompt_len / pool_size)

    decode_flops = 0.0
    for step in range(gen_len):
        ctx = pooled_prefix + keep_recent + step + 1
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
def cepe_generate(
    model,
    tokenizer,
    prompt:          str,
    max_new_tokens:  int = 128,
    keep_recent:     int = 256,
    pool_size:       int = 4,
    compress_every:  int = 1,        # apply pooling every N decode steps
    device:          str = "cuda",
) -> GenerationResult:
    """
    Greedy decoding with CEPE-style KV pooling.

    Old token KV entries are block-averaged every `compress_every` steps.
    The most recent `keep_recent` positions are always kept verbatim.
    """
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos_id     = tokenizer.eos_token_id

    total_timer = WallTimer().start()

    # ── Prefill ──────────────────────────────────────────────────────────────
    ttft_timer = WallTimer().start()
    out     = model(input_ids, use_cache=True)
    past_kv = out.past_key_values
    # Compress the prompt KV right after prefill
    past_kv  = apply_cepe_pooling(past_kv, keep_recent, pool_size)
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    ttft_ms  = ttft_timer.elapsed_ms()

    generated = [next_tok.item()]

    # ── Decode ───────────────────────────────────────────────────────────────
    for step in range(max_new_tokens - 1):
        if next_tok.item() == eos_id:
            break
        out     = model(next_tok, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_tok.item())

        if (step + 1) % compress_every == 0:
            past_kv = apply_cepe_pooling(past_kv, keep_recent, pool_size)

    total_ms = total_timer.elapsed_ms()
    all_ids  = input_ids[0].tolist() + generated

    flops = estimate_flops_cepe(model, prompt_len, len(generated),
                                keep_recent, pool_size)
    return make_result(
        all_ids, prompt_len, total_ms, ttft_ms, past_kv,
        total_gflops=flops["total_gflops"],
    )


# ── Configs ───────────────────────────────────────────────────────────────────

CEPE_CONFIGS = {
    "CEPE(r=256,p=4)":  {"keep_recent": 256, "pool_size": 4,  "compress_every": 1},
    "CEPE(r=128,p=4)":  {"keep_recent": 128, "pool_size": 4,  "compress_every": 1},
    "CEPE(r=256,p=8)":  {"keep_recent": 256, "pool_size": 8,  "compress_every": 1},
    "CEPE(r=128,p=8)":  {"keep_recent": 128, "pool_size": 8,  "compress_every": 1},
}

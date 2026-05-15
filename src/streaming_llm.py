"""
StreamingLLM — attention-sink + sliding-window KV cache management.

Keeps the first `n_sink` token positions (attention sinks) plus the most
recent `window` positions, evicting everything in between.  This bounds
KV cache size to (n_sink + window) regardless of sequence length.

Ref: Xiao et al., "Efficient Streaming Language Models with Attention
     Sinks", ICLR 2024.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

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


# ── Core trim operation ──────────────────────────────────────────────────────

def _trim_to_streaming_window(past_kv, n_sink: int, window: int):
    """
    Trim KV cache to [0:n_sink] ∪ [-(window):].

    If the cache is already within (n_sink + window) entries, return as-is.
    """
    cur_len = kv_cache_length(past_kv)
    max_len = n_sink + window
    if cur_len <= max_len:
        return past_kv

    n = _n_layers(past_kv)
    keys: List[torch.Tensor] = []
    vals: List[torch.Tensor] = []

    for i in range(n):
        k, v = _get_kv(past_kv, i)
        k_new = torch.cat([k[:, :, :n_sink, :], k[:, :, -window:, :]], dim=2)
        v_new = torch.cat([v[:, :, :n_sink, :], v[:, :, -window:, :]], dim=2)
        keys.append(k_new)
        vals.append(v_new)

    new_cache = _build(past_kv, keys, vals)
    if _HAS_DYNAMIC and hasattr(new_cache, "_seen_tokens"):
        new_cache._seen_tokens = max_len
    return new_cache


# ── FLOPs estimation ────────────────────────────────────────────────────────

def estimate_flops_streaming(
    model,
    prompt_len: int,
    gen_len:    int,
    n_sink:     int,
    window:     int,
) -> Dict[str, float]:
    """
    Analytical FLOPs for StreamingLLM.

    Prefill: full attention (same as baseline).
    Decode:  attention context is capped at (n_sink + window + t') where
             t' = min(step, 0) once the window is full.  Effectively,
             context = min(n_sink + window, prompt_len + step + 1).
    """
    cfg = model.config
    H   = cfg.hidden_size
    n   = num_layers(model)

    prefill_flops = n * (24 * prompt_len * H * H + 4 * (prompt_len ** 2) * H)

    max_ctx = n_sink + window
    decode_flops = 0.0
    for step in range(gen_len):
        ctx = min(max_ctx, prompt_len + step + 1)
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


# ── Generation ──────────────────────────────────────────────────────────────

@torch.inference_mode()
def streaming_generate(
    model,
    tokenizer,
    prompt:         str,
    max_new_tokens: int = 128,
    window:         int = 256,
    n_sink:         int = 4,
    device:         str = "cuda",
) -> GenerationResult:
    """Greedy decoding with StreamingLLM KV cache management."""
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos_id     = tokenizer.eos_token_id

    total_timer = WallTimer().start()

    # ── Prefill ──────────────────────────────────────────────────────────────
    ttft_timer = WallTimer().start()
    out     = model(input_ids, use_cache=True)
    past_kv = out.past_key_values
    past_kv = _trim_to_streaming_window(past_kv, n_sink, window)
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    ttft_ms  = ttft_timer.elapsed_ms()

    generated = [next_tok.item()]

    # ── Decode ───────────────────────────────────────────────────────────────
    for _ in range(max_new_tokens - 1):
        if next_tok.item() == eos_id:
            break
        out     = model(next_tok, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        past_kv = _trim_to_streaming_window(past_kv, n_sink, window)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_tok.item())

    total_ms = total_timer.elapsed_ms()
    all_ids  = input_ids[0].tolist() + generated

    flops = estimate_flops_streaming(model, prompt_len, len(generated),
                                     n_sink, window)
    return make_result(
        all_ids, prompt_len, total_ms, ttft_ms, past_kv,
        total_gflops=flops["total_gflops"],
    )


# ── Configs ─────────────────────────────────────────────────────────────────

STREAMING_CONFIGS = {
    "StreamingLLM(W=128)": {"window": 128, "n_sink": 4},
    "StreamingLLM(W=256)": {"window": 256, "n_sink": 4},
}

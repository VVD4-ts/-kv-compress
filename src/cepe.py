"""
Context Expansion with Parallel Encoding (CEPE) – KV compression variant.

Idea
----
CEPE introduces a parallel encoder that compresses a long context into a
compact KV representation, allowing the main decoder to efficiently attend to
much longer histories than its native context window.

Training-free approximation
---------------------------
Without a dedicated encoder, we approximate the "compressed context" idea by
average-pooling the accumulated KV cache:
  • The most recent `keep_recent` token positions are kept verbatim (local
    attention remains exact).
  • Older positions are grouped into blocks of `pool_size` and each block is
    replaced by its per-head average key and value.

This yields a compressed prefix that grows at rate 1/pool_size instead of
linearly, bounding memory use for long-context generation.

Memory saving after warm-up: old-context shrinks by factor `pool_size`.

Reference
---------
Yen et al., "Long-Context Language Modeling with Parallel Context Encoding",
ACL 2024.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import WallTimer, make_result, GenerationResult
from kv_utils import _n_layers, _get_kv, _build, kv_cache_length


# ── KV pooling ────────────────────────────────────────────────────────────────

def apply_cepe(past_kv, pool_size: int, keep_recent: int):
    """
    Pool old KV entries into blocks of `pool_size`, keep last `keep_recent` verbatim.

    Sequence layout after compression:
      [ pooled_block_0 | pooled_block_1 | ... | recent_0 | ... | recent_{keep_recent-1} ]

    Positions that don't fill a complete block are kept as-is (between pooled
    region and recent region).
    """
    n       = _n_layers(past_kv)
    seq_len = _get_kv(past_kv, 0)[0].shape[-2]

    old_len = seq_len - keep_recent
    if old_len <= pool_size:          # nothing old enough to pool
        return past_kv

    n_blocks = old_len // pool_size   # complete blocks
    usable   = n_blocks * pool_size   # tokens consumed by blocks

    keys: List[torch.Tensor] = []
    vals: List[torch.Tensor] = []

    for i in range(n):
        k, v = _get_kv(past_kv, i)   # [1, heads, seq_len, head_dim]
        heads, _, hdim = k.shape[1], k.shape[2], k.shape[3]

        # Pool old region into blocks
        k_pool = (k[:, :, :usable, :]
                  .reshape(1, heads, n_blocks, pool_size, hdim)
                  .mean(dim=3))                          # [1, heads, n_blocks, hdim]
        v_pool = (v[:, :, :usable, :]
                  .reshape(1, heads, n_blocks, pool_size, hdim)
                  .mean(dim=3))

        # Remainder (not enough for a full block)
        k_rem = k[:, :, usable : seq_len - keep_recent, :]
        v_rem = v[:, :, usable : seq_len - keep_recent, :]

        # Recent tokens kept verbatim
        k_rec = k[:, :, -keep_recent:, :]
        v_rec = v[:, :, -keep_recent:, :]

        keys.append(torch.cat([k_pool, k_rem, k_rec], dim=2))
        vals.append(torch.cat([v_pool, v_rem, v_rec], dim=2))

    return _build(past_kv, keys, vals)


# ── Generation ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def cepe_generate(
    model,
    tokenizer,
    prompt:         str,
    max_new_tokens: int = 128,
    pool_size:      int = 4,
    keep_recent:    int = 64,
    compress_every: int = 32,
    device:         str = "cuda",
) -> GenerationResult:
    """
    Greedy decoding with CEPE-style KV pooling.

    Every `compress_every` decode steps, average-pool the old portion of the
    KV cache into blocks of `pool_size`, keeping the last `keep_recent` tokens
    verbatim.  position_ids are tracked to keep RoPE correct after compression.
    """
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos_id     = tokenizer.eos_token_id

    total_timer = WallTimer().start()
    ttft_timer  = WallTimer().start()

    out      = model(input_ids, use_cache=True)
    past_kv  = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    ttft_ms  = ttft_timer.elapsed_ms()

    generated = [next_tok.item()]
    real_pos  = prompt_len
    step      = 0

    for _ in range(max_new_tokens - 1):
        if next_tok.item() == eos_id:
            break
        pos_ids = torch.tensor([[real_pos]], device=device)
        out     = model(next_tok, past_key_values=past_kv,
                        use_cache=True, position_ids=pos_ids)
        past_kv  = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        real_pos += 1
        step     += 1
        generated.append(next_tok.item())

        if step % compress_every == 0:
            past_kv = apply_cepe(past_kv, pool_size, keep_recent)

    total_ms = total_timer.elapsed_ms()
    all_ids  = input_ids[0].tolist() + generated
    return make_result(all_ids, prompt_len, total_ms, ttft_ms, past_kv)


# ── Configs ───────────────────────────────────────────────────────────────────

CEPE_CONFIGS = {
    "CEPE(pool=2)": {"pool_size": 2, "keep_recent": 64, "compress_every": 32},
    "CEPE(pool=4)": {"pool_size": 4, "keep_recent": 64, "compress_every": 32},
}

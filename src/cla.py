"""
Cross-Layer Attention (CLA) inference.

Idea
----
Adjacent transformer layers often learn highly correlated key-value
projections.  CLA exploits this by having every `group_size` layers share a
single set of K/V tensors – only the first ("reference") layer in each group
computes K and V; the remaining layers reuse them.

Training-free approximation
---------------------------
We cannot retrain the model, so we apply sharing *between* forward passes:
after each step we overwrite the KV cache entries of all sharing layers with
the reference layer's entries.  At the next step the model attends its own
(per-layer) Q to the reference layer's accumulated K/V history.

Memory saving: (group_size − 1) / group_size
  group_size=2 → 50 % reduction
  group_size=3 → 67 % reduction

Reference
---------
Brandon et al., "Reducing Transformer Key-Value Cache Size with
Cross-Layer Attention", arXiv 2405.12981, 2024.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import WallTimer, make_result, GenerationResult
from kv_utils import _n_layers, _get_kv, _build


# ── CLA KV sharing ────────────────────────────────────────────────────────────

def apply_cla(past_kv, group_size: int):
    """
    Force all layers in each group to share the reference layer's KV cache.

    Group assignment:
      layer i  →  reference = (i // group_size) * group_size

    All layers in a group point to the same K/V tensors (same Python object),
    so kv_size_mb() counts the memory only once per group.
    """
    n = _n_layers(past_kv)

    keys: List[torch.Tensor] = []
    vals: List[torch.Tensor] = []

    for i in range(n):
        ref = (i // group_size) * group_size
        k, v = _get_kv(past_kv, ref)
        keys.append(k)
        vals.append(v)

    return _build(past_kv, keys, vals)


# ── Generation ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def cla_generate(
    model,
    tokenizer,
    prompt:         str,
    max_new_tokens: int = 128,
    group_size:     int = 2,
    device:         str = "cuda",
) -> GenerationResult:
    """
    Greedy decoding with Cross-Layer Attention KV sharing.

    After each forward pass, overwrites non-reference layers' KV entries with
    the reference layer's entries so the next step sees the shared cache.
    """
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos_id     = tokenizer.eos_token_id

    total_timer = WallTimer().start()
    ttft_timer  = WallTimer().start()

    out      = model(input_ids, use_cache=True)
    past_kv  = apply_cla(out.past_key_values, group_size)
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    ttft_ms  = ttft_timer.elapsed_ms()

    generated = [next_tok.item()]
    real_pos  = prompt_len

    for _ in range(max_new_tokens - 1):
        if next_tok.item() == eos_id:
            break
        pos_ids = torch.tensor([[real_pos]], device=device)
        out     = model(next_tok, past_key_values=past_kv,
                        use_cache=True, position_ids=pos_ids)
        past_kv  = apply_cla(out.past_key_values, group_size)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        real_pos += 1
        generated.append(next_tok.item())

    total_ms = total_timer.elapsed_ms()
    all_ids  = input_ids[0].tolist() + generated
    return make_result(all_ids, prompt_len, total_ms, ttft_ms, past_kv)


# ── Configs ───────────────────────────────────────────────────────────────────

CLA_CONFIGS = {
    "CLA(g=2, -50%KV)": {"group_size": 2},
    "CLA(g=3, -67%KV)": {"group_size": 3},
}

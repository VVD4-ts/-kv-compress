"""
Layer-Condensed KV Cache (LCKV) inference.

Idea
----
Not all transformer layers contribute equally to the KV cache.  LCKV
designates a small subset of layers as *condensed* (anchor) layers that
maintain their own K/V entries.  All other layers are assigned to the nearest
condensed layer and reuse its KV cache instead of computing their own.

Training-free approximation
---------------------------
After each forward pass we overwrite each non-condensed layer's KV entry with
the entry of its nearest condensed layer.  On the next step those layers
attend their own Q to the condensed layer's accumulated K/V history.

Memory saving: (n_layers - n_condensed) / n_layers
  ratio=0.50 -> 50 % reduction (same as CLA-2)
  ratio=0.25 -> 75 % reduction (aggressive)

Reference
---------
Zhang et al., "Layer-Condensed KV Cache for Efficient Inference of Large
Language Models", arXiv 2405.10637, 2024.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import WallTimer, make_result, GenerationResult, num_layers
from kv_utils import _n_layers, _get_kv, _build


# ---- Condensed-layer selection -----------------------------------------------

def default_condensed_layers(n_layers: int, condensed_ratio: float = 0.5) -> List[int]:
    """
    Return evenly spaced condensed layer indices.

    With condensed_ratio=0.5 and n_layers=6: [0, 2, 4]  (every other layer).
    With condensed_ratio=0.25 and n_layers=8: [0, 3, 6] (every 3rd layer).
    """
    n_condensed = max(1, round(n_layers * condensed_ratio))
    if n_condensed >= n_layers:
        return list(range(n_layers))
    step = n_layers / n_condensed
    return sorted(set(min(int(i * step), n_layers - 1) for i in range(n_condensed)))


# ---- LCKV KV sharing ---------------------------------------------------------

def apply_lckv(past_kv, condensed_layers: List[int]):
    """
    Redirect each non-condensed layer to its nearest condensed layer's KV.

    Tie-breaking: prefer the lower-index condensed layer.
    """
    n             = _n_layers(past_kv)
    condensed_set = set(condensed_layers)

    source: List[int] = []
    for i in range(n):
        if i in condensed_set:
            source.append(i)
        else:
            nearest = min(condensed_layers, key=lambda c: (abs(i - c), c))
            source.append(nearest)

    keys: List[torch.Tensor] = []
    vals: List[torch.Tensor] = []
    for i in range(n):
        k, v = _get_kv(past_kv, source[i])
        keys.append(k)
        vals.append(v)

    return _build(past_kv, keys, vals)


# ---- Generation --------------------------------------------------------------

@torch.inference_mode()
def lckv_generate(
    model,
    tokenizer,
    prompt:           str,
    max_new_tokens:   int                = 128,
    condensed_ratio:  float              = 0.5,
    condensed_layers: Optional[List[int]] = None,
    device:           str                = "cuda",
) -> GenerationResult:
    """
    Greedy decoding with Layer-Condensed KV sharing.

    If condensed_layers is None, evenly-spaced layers are selected based on
    condensed_ratio.  After each forward pass, non-condensed layers' KV
    entries are replaced with their nearest condensed layer's entries.
    """
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos_id     = tokenizer.eos_token_id

    n_lay = num_layers(model)
    if condensed_layers is None:
        condensed_layers = default_condensed_layers(n_lay, condensed_ratio)

    total_timer = WallTimer().start()
    ttft_timer  = WallTimer().start()

    out      = model(input_ids, use_cache=True)
    past_kv  = apply_lckv(out.past_key_values, condensed_layers)
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
        past_kv  = apply_lckv(out.past_key_values, condensed_layers)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        real_pos += 1
        generated.append(next_tok.item())

    total_ms = total_timer.elapsed_ms()
    all_ids  = input_ids[0].tolist() + generated
    return make_result(all_ids, prompt_len, total_ms, ttft_ms, past_kv)


# ---- Configs -----------------------------------------------------------------

LCKV_CONFIGS = {
    "LCKV(ratio=0.50, -50%KV)": {"condensed_ratio": 0.50},
    "LCKV(ratio=0.25, -75%KV)": {"condensed_ratio": 0.25},
}

"""
SimLayerKV — Training-free inter-layer KV cache compression.

Algorithm
---------
1. Profile: run a calibration corpus through the model; for each adjacent
   layer pair (i, i+1) compute the cosine similarity of their key tensors.
2. Identify lazy layers: layer i+1 is "lazy" if sim(layer_i, layer_{i+1}) is
   above a threshold — its KV pattern is close enough to reuse layer_i's KV.
3. Inference: after every forward pass, replace each lazy layer's KV with the
   nearest preceding active layer's KV (via kv_utils.apply_sim_layer_kv).

FLOPs analysis (analytical, theoretical)
-----------------------------------------
Lazy layers skip K and V projections in an ideal implementation.
Savings per lazy layer per decode step: 4 * H²  (2 mat-muls of dim H×H).
We report the theoretical FLOPs as standard in the literature.

Memory
------
kv_size_mb() deduplicates by data_ptr(), so shared KV tensors are
counted only once — same mechanism as YOCO.

Reference
---------
Zhang et al., "SimLayerKV: A Simple Framework for Layer-Level KV Cache
Reduction", arXiv 2405.13527, 2024.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from utils import WallTimer, make_result, GenerationResult, num_layers
from kv_utils import _n_layers, _get_kv, apply_sim_layer_kv


# ── Layer-similarity profiling ────────────────────────────────────────────────

@torch.inference_mode()
def profile_layer_similarity(
    model,
    tokenizer,
    texts: List[str],
    device: str = "cuda",
    max_tokens: int = 256,
) -> List[float]:
    """
    Compute average cosine similarity between adjacent layer key tensors.

    Returns a list of length (n_layers - 1):
        result[i]  =  cosine_sim( keys_layer_i,  keys_layer_{i+1} )
    averaged over all calibration texts, attention heads, and positions.
    Higher means more similar → layer i+1 is a good candidate to be lazy.
    """
    n = num_layers(model)
    sim_sums   = [0.0] * (n - 1)
    n_samples  = 0

    for text in texts:
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        if ids.shape[1] > max_tokens:
            ids = ids[:, :max_tokens]
        if ids.shape[1] < 2:
            continue

        out     = model(ids, use_cache=True)
        past_kv = out.past_key_values
        n_act   = _n_layers(past_kv)

        for i in range(min(n - 1, n_act - 1)):
            k_i,   _ = _get_kv(past_kv, i)
            k_ip1, _ = _get_kv(past_kv, i + 1)

            # Flatten → (S * heads, head_dim), compute per-position cos sim
            flat_i   = k_i.reshape(-1,   k_i.shape[-1]).float()
            flat_ip1 = k_ip1.reshape(-1, k_ip1.shape[-1]).float()
            sim = F.cosine_similarity(flat_i, flat_ip1, dim=-1).mean().item()
            sim_sums[i] += sim

        n_samples += 1

    if n_samples == 0:
        return [0.0] * (n - 1)
    return [s / n_samples for s in sim_sums]


def identify_lazy_layers(
    similarities: List[float],
    threshold: float = 0.90,
    min_active: int = 2,
) -> Set[int]:
    """
    Select which layers are "lazy" based on adjacent-layer similarity.

    Rules
    -----
    * Layer 0 is always active (never lazy).
    * Layer i+1 is a candidate if similarities[i] >= threshold.
    * A lazy layer i cannot itself be the source for layer i+1 being lazy
      (we need an active predecessor to provide valid KV).
    * At least min_active layers remain active.

    Returns a set of lazy layer indices (0-based).
    """
    n      = len(similarities) + 1
    lazy:  Set[int] = set()
    active = n   # all start active

    # Process candidate pairs ordered by descending similarity
    for layer_idx, sim in sorted(
        enumerate(similarities), key=lambda x: -x[1]
    ):
        candidate = layer_idx + 1         # layer (layer_idx+1) would be lazy
        if sim < threshold:
            break
        if active - 1 < min_active:
            break
        # Predecessor (layer_idx) must itself be active to serve as source
        if layer_idx in lazy:
            continue
        lazy.add(candidate)
        active -= 1

    return lazy


# ── FLOPs estimation (analytical, theoretical) ───────────────────────────────

def estimate_flops(
    model,
    prompt_len: int,
    gen_len:    int,
    lazy_layers: Set[int],
) -> Dict[str, float]:
    """
    Analytical FLOPs estimate for GPT-NeoX / Pythia architecture.

    Convention: 1 multiply-add = 2 FLOPs (standard ML convention).

    Per active layer, prefill of S tokens:
        Q proj:    2·S·H²
        K proj:    2·S·H²   ← skipped for lazy
        V proj:    2·S·H²   ← skipped for lazy
        QK^T:      2·S²·H   (full causal, upper-tri zeroed → ≈ S²·H)
        AV:        2·S²·H
        Out proj:  2·S·H²
        FFN:       2·(2·S·H·4H) = 16·S·H²
    Total active  =  24·S·H²  +  4·S²·H
    Total lazy    =  20·S·H²  +  4·S²·H   (save 4·S·H²)

    Per layer, one decode step (context length C = prompt + step + 1):
        Q proj:    2·H²
        K proj:    2·H²   ← skipped for lazy
        V proj:    2·H²   ← skipped for lazy
        QK^T:      2·C·H
        AV:        2·C·H
        Out proj:  2·H²
        FFN:       16·H²
    Total active  =  24·H²  +  4·C·H
    Total lazy    =  20·H²  +  4·C·H   (save 4·H²)
    """
    cfg  = model.config
    H    = cfg.hidden_size
    n    = num_layers(model)

    # ── Prefill ──
    prefill_flops = 0.0
    for i in range(n):
        is_lazy = i in lazy_layers
        base = (20 if is_lazy else 24) * prompt_len * H * H
        attn = 4 * (prompt_len ** 2) * H
        prefill_flops += base + attn

    # ── Decode ──
    decode_flops = 0.0
    for step in range(gen_len):
        ctx = prompt_len + step + 1
        for i in range(n):
            is_lazy = i in lazy_layers
            base = (20 if is_lazy else 24) * H * H
            attn = 4 * ctx * H
            decode_flops += base + attn

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
def sim_layer_kv_generate(
    model,
    tokenizer,
    prompt:         str,
    max_new_tokens: int       = 128,
    lazy_layers:    Set[int]  = None,
    device:         str       = "cuda",
) -> GenerationResult:
    """
    Greedy decoding with SimLayerKV KV sharing.

    After each forward pass (prefill + every decode step) lazy layers'
    KV entries are replaced with their nearest active predecessor's KV.
    Timing metrics (TTFT, TPOT, Throughput) are measured.
    FLOPs are computed analytically.
    """
    if lazy_layers is None:
        lazy_layers = set()

    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos_id     = tokenizer.eos_token_id

    total_timer = WallTimer().start()

    # ── Prefill ──────────────────────────────────────────────────────────────
    ttft_timer = WallTimer().start()
    out     = model(input_ids, use_cache=True)
    past_kv = apply_sim_layer_kv(out.past_key_values, lazy_layers)
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    ttft_ms  = ttft_timer.elapsed_ms()

    generated = [next_tok.item()]

    # ── Decode ───────────────────────────────────────────────────────────────
    for _ in range(max_new_tokens - 1):
        if next_tok.item() == eos_id:
            break
        out     = model(next_tok, past_key_values=past_kv, use_cache=True)
        past_kv = apply_sim_layer_kv(out.past_key_values, lazy_layers)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_tok.item())

    total_ms = total_timer.elapsed_ms()
    all_ids  = input_ids[0].tolist() + generated

    # Theoretical FLOPs
    flops = estimate_flops(model, prompt_len, len(generated), lazy_layers)

    return make_result(
        all_ids, prompt_len, total_ms, ttft_ms, past_kv,
        total_gflops=flops["total_gflops"],
    )


# ── Convenience: fixed configs for Pythia-70M (6 layers) ─────────────────────

def slk_configs(n_layers: int) -> Dict[str, Set[int]]:
    """
    Return a dict of named SimLayerKV configurations.

    Keys:   "SLK-Light"  (last 1 lazy),
            "SLK-Medium" (last 2 lazy),
            "SLK-Heavy"  (last 3 lazy)
    """
    configs = {}
    thirds  = max(1, n_layers // 6)    # ~17 % intervals
    for name, n_lazy in [
        ("SLK-Light",  1),
        ("SLK-Medium", 2),
        ("SLK-Heavy",  3),
    ]:
        if n_lazy < n_layers:
            lazy = set(range(n_layers - n_lazy, n_layers))
            configs[name] = lazy
    return configs

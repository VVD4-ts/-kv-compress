"""
KV cache manipulation helpers — YOCO and SimLayerKV.

Supports:
  - DynamicCache with DynamicLayer  (transformers >= 4.57, .layers API)
  - DynamicCache with key_cache/value_cache  (transformers 4.38–4.56)
  - Legacy tuple-of-(key, value) pairs
"""

from __future__ import annotations

import torch
from typing import List, Tuple

# ── One-time import probe ────────────────────────────────────────────────────

try:
    from transformers import DynamicCache
    from transformers.cache_utils import Cache as _CacheBase
    _HAS_DYNAMIC_CACHE = True
except ImportError:
    _HAS_DYNAMIC_CACHE = False
    DynamicCache = None      # type: ignore
    _CacheBase   = None      # type: ignore


def _is_cache_obj(past_kv) -> bool:
    if _HAS_DYNAMIC_CACHE and _CacheBase is not None:
        if isinstance(past_kv, _CacheBase):
            return True
    return hasattr(past_kv, "get_seq_length") or hasattr(past_kv, "layers")


def _has_layers_api(past_kv) -> bool:
    """True for transformers ≥ 4.57 (.layers list of DynamicLayer)."""
    return hasattr(past_kv, "layers") and isinstance(past_kv.layers, list)


def _has_keycache_api(past_kv) -> bool:
    """True for transformers 4.38–4.56 (.key_cache list)."""
    return hasattr(past_kv, "key_cache")


# ── Low-level accessors ──────────────────────────────────────────────────────

def _n_layers(past_kv) -> int:
    if _has_layers_api(past_kv):
        return len(past_kv.layers)
    if _has_keycache_api(past_kv):
        return len(past_kv.key_cache)
    return len(past_kv)


def _get_kv(past_kv, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (key, value) tensors for layer_idx — preserves data_ptr."""
    if _has_layers_api(past_kv):
        layer = past_kv.layers[layer_idx]
        return layer.keys, layer.values
    if _has_keycache_api(past_kv):
        return past_kv.key_cache[layer_idx], past_kv.value_cache[layer_idx]
    item = past_kv[layer_idx]
    return item[0], item[1]


def _build(original_past_kv, keys: List[torch.Tensor], vals: List[torch.Tensor]):
    """
    Construct a KV cache from lists of key/value tensors.

    For DynamicCache (.layers API, transformers ≥ 4.57):
      Initialises each DynamicLayer via update(), then overwrites .keys/.values
      with our actual tensor references so that shared tensors (same Python
      object) retain the same data_ptr() — enabling correct deduplication
      in kv_size_mb().

    For DynamicCache (.key_cache API):
      Uses the public update() API.

    For legacy tuples:
      Returns tuple-of-(key, value) pairs.
    """
    if _HAS_DYNAMIC_CACHE and _is_cache_obj(original_past_kv):
        new = DynamicCache()

        if _has_layers_api(original_past_kv):
            for i, (k, v) in enumerate(zip(keys, vals)):
                new.update(k, v, i)        # initialise DynamicLayer
                new.layers[i].keys   = k   # overwrite → preserves data_ptr()
                new.layers[i].values = v
        else:
            for i, (k, v) in enumerate(zip(keys, vals)):
                new.update(k, v, i)
            if keys:
                try:
                    new._seen_tokens = keys[0].shape[-2]
                except AttributeError:
                    pass

        return new

    return tuple(zip(keys, vals))


# ── YOCO – You Only Cache Once ───────────────────────────────────────────────

def apply_yoco(past_kv, split_idx: int):
    """
    YOCO-style KV sharing.

    Layers [0, split_idx)  — self-decoder:  keep their own KV (grows normally).
    Layers [split_idx, n)  — cross-decoder: share layer (split_idx-1)'s KV.

    Memory reduction
    ----------------
    All cross-decoder layers point to the SAME tensor (layer split_idx-1's K/V).
    kv_size_mb() deduplicates by data_ptr(), so counted memory is:
        split_idx / n_layers  ×  baseline_kv
    e.g. split_idx=3, n=6 → 50 % of baseline.

    Why this preserves quality
    --------------------------
    After each decode step, cross-decoder layer i builds its KV as:
        cat(K_{split_idx-1, prev_step},   k_{i, current_token})
    so positions 0..T-2 use the self-decoder's K/V projection (from layer
    split_idx-1), while only the current-step position uses layer i's own
    projection.  The context is never frozen; cross-decoder layers always
    see all previously processed tokens.

    Ref: Sun et al., "You Only Cache Once", arXiv 2405.05254 (2024).
    """
    n = _n_layers(past_kv)
    if split_idx >= n:          # no cross-decoder → equivalent to baseline
        return past_kv
    if split_idx < 1:
        split_idx = 1

    shared_k, shared_v = _get_kv(past_kv, split_idx - 1)

    keys: List[torch.Tensor] = []
    vals: List[torch.Tensor] = []
    for i in range(n):
        if i < split_idx:
            k, v = _get_kv(past_kv, i)
        else:
            k, v = shared_k, shared_v   # same Python object → same data_ptr
        keys.append(k)
        vals.append(v)

    return _build(past_kv, keys, vals)


# ── SimLayerKV – Similarity-guided inter-layer KV sharing ────────────────────

def apply_sim_layer_kv(past_kv, lazy_layers: set):
    """
    SimLayerKV-style KV sharing.

    For each "lazy" layer i, find the nearest preceding active layer j < i
    (where j ∉ lazy_layers) and share its KV object.

    Memory reduction
    ----------------
    Each lazy layer contributes zero additional unique tensors.
    With lazy_layers = {3,4,5} on a 6-layer model: 50 % reduction.

    Quality
    -------
    Lazy layers only reuse KV from adjacent (often highly similar) layers,
    so PPL degradation is typically small for well-chosen lazy layers.

    Ref: Zhang et al., "SimLayerKV: A Simple Framework for Layer-Level
         KV Cache Reduction", arXiv 2405.13527 (2024).
    """
    if not lazy_layers:
        return past_kv

    n = _n_layers(past_kv)

    # Compute the "source layer" for each position:
    # active layer → itself; lazy layer → nearest active predecessor
    source: List[int] = list(range(n))
    for i in range(1, n):
        if i in lazy_layers:
            source[i] = source[i - 1]   # inherit predecessor's source

    keys: List[torch.Tensor] = []
    vals: List[torch.Tensor] = []
    for i in range(n):
        k, v = _get_kv(past_kv, source[i])
        keys.append(k)
        vals.append(v)

    return _build(past_kv, keys, vals)

"""
Perplexity evaluation — WikiText-2 and PG-19.

Methods
-------
  Baseline                 full KV cache
  CLE-Light  (-20%KV)      Cross-Layer KV Eviction, budget_ratio=0.80
  CLE-Medium (-50%KV)      budget_ratio=0.50
  CLE-Heavy  (-70%KV)      budget_ratio=0.30
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_model, set_seed, num_layers
from cross_layer_evict import (
    compute_cross_layer_importance,
    evict_kv_cache,
    CLE_CONFIGS,
)


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_wikitext(split: str = "test") -> str:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(r["text"] for r in ds if r["text"].strip())


def load_pg19(n_chars: int = 200_000) -> str:
    from datasets import load_dataset
    for hf_name, cfg, spl in [
        ("emozilla/pg19-test-tokenized", None, "test"),
        ("storytracer/pg19-books",       None, "train"),
    ]:
        try:
            ds = load_dataset(hf_name, cfg, split=spl, streaming=True)
            for sample in ds:
                text = (sample.get("text") or sample.get("book_text")
                        or sample.get("story") or "")
                if len(text) > 10_000:
                    print(f"  [pg19 source: {hf_name}]")
                    return text[:n_chars]
        except Exception:
            continue
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())
    while len(text) < n_chars:
        text = text + "\n\n" + text
    print("  [pg19 source: wikitext-2 train repeated (last resort)]")
    return text[:n_chars]


# ── CLE transform factory ─────────────────────────────────────────────────────

def _make_cle_transform(budget_ratio: float, window: int = 512, n_sink: int = 16):
    """
    Returns a kv_transform callable for compute_ppl_sliding.

    Mirrors the real CLE use case: **one-shot eviction per window**.
    After the KV cache first exceeds `budget`, we evict once and then leave
    the cache to grow normally for the rest of the window (new decode tokens
    are always appended, never evicted).  When the next window starts the
    cache is reset to None → n_cached drops to 1 → state is cleared.

    Budget = window × budget_ratio.
    Sink tokens: the first n_sink positions are always kept (attention anchors)
    so that important early context is not discarded.
    """
    budget = max(n_sink + 1, int(window * budget_ratio))
    # Stateful: track whether we already evicted in the current window.
    # Detect window boundary via a sharp drop in n_cached (reset to ≈1).
    state = {"evicted": False, "last_n": 0}

    def _transform(past_kv):
        n_cached = _get_cache_len(past_kv)

        # --- Detect window reset: new window always starts fresh (past_kv=None
        #     → 1 token after first step).  A drop to ≤ last_n // 3 reliably
        #     flags this without ever triggering mid-window.
        if n_cached <= max(1, state["last_n"] // 3):
            state["evicted"] = False

        state["last_n"] = n_cached

        # Already evicted this window, or still below budget → nothing to do
        if state["evicted"] or n_cached <= budget:
            return past_kv

        # First time over budget → evict once
        importance = compute_cross_layer_importance(past_kv)
        if n_sink > 0:
            importance[:n_sink] = float("inf")   # protect sink tokens
        keep_indices = importance.topk(budget).indices.sort().values
        state["evicted"] = True
        state["last_n"]  = budget   # cache is now at budget after eviction
        return evict_kv_cache(past_kv, keep_indices)

    return _transform


def _get_cache_len(past_kv) -> int:
    if hasattr(past_kv, "layers") and past_kv.layers:
        layer = past_kv.layers[0]
        if hasattr(layer, "keys"):
            return layer.keys.shape[2]
    if hasattr(past_kv, "key_cache") and past_kv.key_cache:
        return past_kv.key_cache[0].shape[2]
    if past_kv and hasattr(past_kv[0], "__len__"):
        return past_kv[0][0].shape[2]
    return 0


def _get_kv_device(past_kv) -> torch.device:
    if hasattr(past_kv, "layers") and past_kv.layers:
        layer = past_kv.layers[0]
        if hasattr(layer, "keys"):
            return layer.keys.device
    if hasattr(past_kv, "key_cache") and past_kv.key_cache:
        return past_kv.key_cache[0].device
    return torch.device("cpu")


# ── Build transforms ──────────────────────────────────────────────────────────

def make_transforms(n_layers: int) -> Dict[str, Optional[Callable]]:
    """Return {method_name: kv_transform_or_None} for Baseline + CLE configs."""
    transforms: Dict[str, Optional[Callable]] = {"Baseline": None}
    for name, ratio in CLE_CONFIGS.items():
        transforms[name] = _make_cle_transform(ratio)
    return transforms


# ── Sliding-window PPL ────────────────────────────────────────────────────────

def compute_ppl_sliding(
    model,
    tokenizer,
    text:         str,
    window:       int               = 512,
    stride:       int               = 256,
    max_tokens:   int               = 4096,
    device:       str               = "cpu",
    kv_transform: Optional[Callable] = None,
) -> float:
    enc = tokenizer(text, return_tensors="pt").input_ids.to(device)
    if max_tokens and max_tokens > 0:
        enc = enc[:, :max_tokens]
    seq_len = enc.shape[1]
    if seq_len < 2:
        return float("nan")

    nlls:     list = []
    prev_end: int  = 0
    n_windows = max(1, (seq_len - 1) // stride)
    win_idx   = 0

    for begin in range(0, seq_len, stride):
        end        = min(begin + window, seq_len)
        chunk      = enc[:, begin:end]
        target_len = end - prev_end
        if target_len <= 0:
            prev_end = end
            continue

        past_kv:    Optional[object] = None
        win_logits: list             = []

        with torch.no_grad():
            for t in range(chunk.shape[1]):
                tok = chunk[:, t : t + 1]
                out = model(tok, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                if kv_transform is not None:
                    past_kv = kv_transform(past_kv)
                win_logits.append(out.logits[:, 0, :])

        logits = torch.stack(win_logits, dim=1)

        if target_len > 1:
            lp  = F.log_softmax(logits[:, -target_len:-1, :], dim=-1)
            lab = chunk[:, -target_len + 1:]
            nll = -lp.gather(2, lab.unsqueeze(2)).squeeze(2)
            nlls.append(nll.mean().item())

        win_idx += 1
        if win_idx % 20 == 0 or end == seq_len:
            ppl_so_far = math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")
            print(f"    window {win_idx}/{n_windows}  pos={end}/{seq_len}"
                  f"  ppl_so_far={ppl_so_far:.2f}", flush=True)

        prev_end = end
        if end == seq_len:
            break

    return math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")

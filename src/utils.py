"""
Shared utilities: model loading, timing, metrics, seed.
"""

import time
import random
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class GenerationResult:
    token_ids:          List[int]
    prompt_len:         int
    num_generated:      int
    elapsed_ms:         float    # wall time for the whole call
    ttft_ms:            float    # time to first token (prefill)
    tpot_ms:            float    # average time per output token (decode phase)
    throughput:         float    # new tokens / second
    kv_size_mb:         float    # final KV cache footprint
    total_gflops:       float = 0.0   # theoretical total FLOPs (analytical)
    avg_gflops_per_tok: float = 0.0   # total_gflops / (prompt_len + num_generated)
    n_proposed:         int   = 0     # speculative decoding: total proposed tokens
    n_accepted:         int   = 0     # speculative decoding: total accepted tokens

    def speedup_vs(self, baseline: "GenerationResult") -> float:
        if self.tpot_ms == 0:
            return float("inf")
        return baseline.tpot_ms / self.tpot_ms


def load_model(
    model_name: str = "EleutherAI/pythia-70m",
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model = model.to(device).eval()
    return model, tokenizer


class WallTimer:
    """Accurate wall-clock timer. Synchronises CUDA before reading time."""

    def __init__(self):
        self._t0: Optional[float] = None

    def start(self) -> "WallTimer":
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def elapsed_ms(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - self._t0) * 1_000.0


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kv_size_mb(past_key_values) -> float:
    """
    Return the *unique* KV cache memory in megabytes.
    Tracks each tensor by data_ptr() to avoid double-counting shared tensors.
    """
    if past_key_values is None:
        return 0.0

    seen: set = set()
    total_bytes_ref = [0]

    def _add(t: torch.Tensor):
        ptr = t.data_ptr()
        if ptr not in seen:
            seen.add(ptr)
            total_bytes_ref[0] += t.numel() * t.element_size()

    if hasattr(past_key_values, "layers") and isinstance(past_key_values.layers, list):
        for layer in past_key_values.layers:
            if hasattr(layer, "keys") and isinstance(layer.keys, torch.Tensor):
                _add(layer.keys)
            if hasattr(layer, "values") and isinstance(layer.values, torch.Tensor):
                _add(layer.values)
    elif hasattr(past_key_values, "key_cache"):
        for k, v in zip(past_key_values.key_cache, past_key_values.value_cache):
            if isinstance(k, torch.Tensor):
                _add(k); _add(v)
    else:
        for layer_kv in past_key_values:
            if layer_kv is None:
                continue
            k, v = layer_kv
            _add(k); _add(v)

    return total_bytes_ref[0] / (1024 ** 2)


def num_layers(model) -> int:
    """Return the number of transformer decoder layers."""
    if hasattr(model, "gpt_neox"):
        return len(model.gpt_neox.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer"):
        return len(model.transformer.h)
    raise ValueError(f"Unsupported architecture: {type(model)}")


def make_result(
    token_ids: List[int],
    prompt_len: int,
    total_ms: float,
    ttft_ms: float,
    past_kv,
    total_gflops: float = 0.0,
    n_proposed: int = 0,
    n_accepted: int = 0,
) -> GenerationResult:
    num_gen    = len(token_ids) - prompt_len
    decode_ms  = total_ms - ttft_ms
    tpot_ms    = decode_ms / max(num_gen - 1, 1)
    throughput = num_gen / (total_ms / 1_000.0) if total_ms > 0 else 0.0
    total_toks = prompt_len + num_gen
    avg_gflops = total_gflops / total_toks if total_toks > 0 else 0.0
    return GenerationResult(
        token_ids=token_ids,
        prompt_len=prompt_len,
        num_generated=num_gen,
        elapsed_ms=total_ms,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        throughput=throughput,
        kv_size_mb=kv_size_mb(past_kv),
        total_gflops=total_gflops,
        avg_gflops_per_tok=avg_gflops,
        n_proposed=n_proposed,
        n_accepted=n_accepted,
    )

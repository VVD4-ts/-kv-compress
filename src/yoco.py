"""
YOCO (You Only Cache Once) — training-free inference approximation.

Cross-decoder layers share the self-decoder's last-layer KV.
FLOPs equal baseline (all projections still computed; savings are memory-only).
"""

import torch
from utils import WallTimer, make_result, GenerationResult, num_layers
from kv_utils import apply_yoco
from sim_layer_kv import estimate_flops


@torch.inference_mode()
def yoco_generate(
    model,
    tokenizer,
    prompt:         str,
    max_new_tokens: int   = 128,
    split_ratio:    float = 0.5,
    split_idx:      int   = None,   # explicit layer index; overrides split_ratio
    device:         str   = "cuda",
) -> GenerationResult:
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos_id     = tokenizer.eos_token_id

    n = num_layers(model)
    if split_idx is None:
        split_idx = max(1, min(n - 1, round(n * split_ratio)))
    else:
        split_idx = max(1, min(n - 1, split_idx))

    total_timer = WallTimer().start()

    ttft_timer = WallTimer().start()
    out     = model(input_ids, use_cache=True)
    past_kv = apply_yoco(out.past_key_values, split_idx)
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    ttft_ms  = ttft_timer.elapsed_ms()

    generated = [next_tok.item()]

    for _ in range(max_new_tokens - 1):
        if next_tok.item() == eos_id:
            break
        out     = model(next_tok, past_key_values=past_kv, use_cache=True)
        past_kv = apply_yoco(out.past_key_values, split_idx)
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_tok.item())

    total_ms = total_timer.elapsed_ms()
    all_ids  = input_ids[0].tolist() + generated

    # YOCO does not skip projections → FLOPs = baseline (lazy_layers=set())
    flops = estimate_flops(model, prompt_len, len(generated), lazy_layers=set())
    return make_result(
        all_ids, prompt_len, total_ms, ttft_ms, past_kv,
        total_gflops=flops["total_gflops"],
    )

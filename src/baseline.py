"""
Baseline: standard greedy autoregressive decoding with full KV cache.
"""

import torch
from utils import WallTimer, make_result, GenerationResult, num_layers
from sim_layer_kv import estimate_flops


@torch.inference_mode()
def greedy_generate(
    model,
    tokenizer,
    prompt:         str,
    max_new_tokens: int = 128,
    device:         str = "cuda",
) -> GenerationResult:
    """Standard greedy decoding with KV cache reuse."""
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos_id     = tokenizer.eos_token_id

    total_timer = WallTimer().start()

    # ── Prefill ──────────────────────────────────────────────────────────────
    ttft_timer = WallTimer().start()
    out     = model(input_ids, use_cache=True)
    past_kv = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    ttft_ms  = ttft_timer.elapsed_ms()

    generated = [next_tok.item()]

    # ── Decode ───────────────────────────────────────────────────────────────
    for _ in range(max_new_tokens - 1):
        if next_tok.item() == eos_id:
            break
        out     = model(next_tok, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_tok.item())

    total_ms = total_timer.elapsed_ms()
    all_ids  = input_ids[0].tolist() + generated

    flops = estimate_flops(model, prompt_len, len(generated), lazy_layers=set())
    return make_result(
        all_ids, prompt_len, total_ms, ttft_ms, past_kv,
        total_gflops=flops["total_gflops"],
    )

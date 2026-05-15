"""
Prompt Lookup Decoding (PLD) — training-free speculative decoding.

Instead of a separate draft model, PLD searches the existing generated
sequence for n-gram matches and uses the successor tokens as speculative
candidates.  A single verification forward pass accepts a prefix of
matching tokens plus one bonus token from the model's own prediction.

Best case:  K+1 tokens per step (K candidates all accepted).
Worst case: 1 token per step (no match or all rejected).

Ref: Saxena, "Prompt Lookup Decoding", 2023.
     https://github.com/apoorvumang/prompt-lookup-decoding
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).parent))

from utils import WallTimer, make_result, GenerationResult, num_layers
from kv_utils import kv_cache_length, trim_kv_cache
from sim_layer_kv import estimate_flops


# ── N-gram candidate search ─────────────────────────────────────────────────

def find_candidate_pred_tokens(
    input_ids:       torch.Tensor,   # [1, S]
    max_ngram_size:  int = 3,
    num_pred_tokens: int = 5,
) -> torch.Tensor:
    """
    Search input_ids for an n-gram match with the last n tokens,
    then return up to num_pred_tokens successor tokens.

    Tries n-gram sizes from max_ngram_size down to 1, returning the
    longest match found.  Returns an empty tensor if no match exists.
    """
    seq_len = input_ids.size(1)
    for ngram_size in range(max_ngram_size, 0, -1):
        if seq_len < ngram_size + 1:
            continue
        query = input_ids[0, -ngram_size:]                    # [ngram_size]
        # Slide over all valid starting positions (exclude the last ngram_size tokens)
        for start in range(seq_len - ngram_size - 1, -1, -1):
            window = input_ids[0, start : start + ngram_size]
            if torch.equal(query, window):
                end = min(start + ngram_size + num_pred_tokens, seq_len)
                candidates = input_ids[0, start + ngram_size : end]
                if candidates.numel() > 0:
                    return candidates
    return torch.tensor([], dtype=input_ids.dtype, device=input_ids.device)


# ── PLD generation ──────────────────────────────────────────────────────────

@torch.inference_mode()
def pld_generate(
    model,
    tokenizer,
    prompt:          str,
    max_new_tokens:  int = 128,
    K:               int = 5,
    max_ngram_size:  int = 3,
    device:          str = "cuda",
) -> GenerationResult:
    """
    Greedy decoding with Prompt Lookup Decoding (PLD).

    K: maximum number of speculative candidate tokens per step.
    """
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos_id     = tokenizer.eos_token_id

    total_timer = WallTimer().start()

    # ── Prefill ──────────────────────────────────────────────────────────────
    ttft_timer = WallTimer().start()
    out     = model(input_ids, use_cache=True)
    past_kv = out.past_key_values
    anchor  = out.logits[:, -1:, :].argmax(dim=-1)   # [1, 1]
    ttft_ms = ttft_timer.elapsed_ms()

    generated = torch.cat([input_ids, anchor], dim=-1)
    n_proposed_total = 0
    n_accepted_total = 0
    n_steps          = 1
    eos_hit          = False

    # ── PLD decode ───────────────────────────────────────────────────────────
    while generated.size(1) - prompt_len < max_new_tokens:
        candidates = find_candidate_pred_tokens(generated, max_ngram_size, K)
        num_cand   = int(candidates.numel())

        fwd_input = (anchor if num_cand == 0
                     else torch.cat([anchor, candidates.unsqueeze(0)], dim=-1))

        cache_len_before = kv_cache_length(past_kv)

        out     = model(fwd_input, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        target_preds = out.logits[0].argmax(dim=-1)
        n_steps += 1

        if num_cand == 0:
            n_accepted = 0
            bonus      = target_preds[0:1].unsqueeze(0)
        else:
            matches    = (target_preds[:num_cand] == candidates).int()
            n_accepted = int(matches.cumprod(dim=0).sum().item())
            bonus      = target_preds[n_accepted : n_accepted + 1].unsqueeze(0)
            target_len = cache_len_before + 1 + n_accepted
            past_kv    = trim_kv_cache(past_kv, target_len)

        if n_accepted > 0:
            accepted  = candidates[:n_accepted].unsqueeze(0)
            generated = torch.cat([generated, accepted, bonus], dim=-1)
        else:
            generated = torch.cat([generated, bonus], dim=-1)

        anchor = bonus
        n_proposed_total += num_cand
        n_accepted_total += n_accepted

        if eos_id is not None:
            if eos_id in generated[0, -(n_accepted + 1):].tolist():
                eos_hit = True
                break

    total_ms = total_timer.elapsed_ms()
    if not eos_hit:
        generated = generated[:, : prompt_len + max_new_tokens]

    all_ids = generated[0].tolist()
    flops = estimate_flops(model, prompt_len, len(all_ids) - prompt_len,
                           lazy_layers=set())
    return make_result(
        all_ids, prompt_len, total_ms, ttft_ms, past_kv,
        total_gflops=flops["total_gflops"],
        n_proposed=n_proposed_total,
        n_accepted=n_accepted_total,
    )


# ── Configs ─────────────────────────────────────────────────────────────────

PLD_CONFIGS = {
    "PLD(K=5)":  {"K": 5,  "max_ngram_size": 3},
    "PLD(K=10)": {"K": 10, "max_ngram_size": 3},
}

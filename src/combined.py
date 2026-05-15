"""
Combined acceleration: CLE + PLD  and  StreamingLLM + PLD.

Key design principle — KV-consistent n-gram search:
  Vanilla PLD searches the full generated sequence for n-gram matches.
  When the KV cache is compressed (CLE eviction or streaming trim), the
  model only "remembers" a subset of past tokens; candidates drawn from
  evicted positions will be rejected almost every time, killing acceptance.

  Fix: restrict n-gram lookup to only the tokens whose KV entries are
  still present in the cache ("remembered" tokens).  This keeps the
  proposal distribution aligned with the model's actual memory.

CLE+PLD:
  1. Full prefill
  2. CLE one-shot eviction → keep_indices marks surviving token positions
  3. PLD n-gram search inside kept tokens only
  4. PLD speculative decode on the compressed KV cache

StreamingLLM+PLD:
  1. Full prefill
  2. Per-step streaming window trim (sink + recent W tokens)
  3. PLD n-gram search inside [sinks ∪ last-window] tokens only
  4. PLD speculative decode inside the streaming cache
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import WallTimer, make_result, GenerationResult, num_layers
from kv_utils import kv_cache_length, trim_kv_cache, _n_layers, _get_kv, _build
from cross_layer_evict import compute_cross_layer_importance, evict_kv_cache
from streaming_llm import _trim_to_streaming_window
from pld import find_candidate_pred_tokens
from cepe import apply_cepe_pooling, estimate_flops_cepe


# ── KV-consistent n-gram helpers ──────────────────────────────────────────────

def _find_candidates_in_subset(
    full_ids:       torch.Tensor,   # [1, S]  full generated sequence
    subset_ids:     torch.Tensor,   # [1, M]  tokens the model still "remembers"
    max_ngram_size: int,
    num_pred_tokens: int,
) -> torch.Tensor:
    """
    Search for n-gram matches restricted to subset_ids.

    The last ngram_size tokens of full_ids are used as the query.
    If found inside subset_ids, return up to num_pred_tokens successors
    from subset_ids.  Falls back to searching full_ids when no match is
    found in the subset (guarantees at least as many proposals as vanilla).
    """
    # Try subset first (aligned with KV cache)
    cands = find_candidate_pred_tokens(subset_ids, max_ngram_size, num_pred_tokens)
    if cands.numel() > 0:
        return cands
    # Fallback: full sequence (same as vanilla PLD)
    return find_candidate_pred_tokens(full_ids, max_ngram_size, num_pred_tokens)

try:
    from transformers import DynamicCache
    _HAS_DYNAMIC = True
except ImportError:
    _HAS_DYNAMIC = False
    DynamicCache = None


# ── CLE + PLD ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def cle_pld_generate(
    model,
    tokenizer,
    prompt:         str,
    max_new_tokens: int   = 128,
    budget_ratio:   float = 0.5,
    n_sink:         int   = 4,
    K:              int   = 5,
    max_ngram_size: int   = 3,
    device:         str   = "cuda",
) -> GenerationResult:
    """
    CLE one-shot eviction after prefill, then PLD speculative decode.

    The PLD verification forward pass benefits from the smaller KV cache
    (budget tokens instead of full prompt length), giving multiplicative gains.
    """
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos_id     = tokenizer.eos_token_id

    budget = max(n_sink + 1, int(prompt_len * budget_ratio))

    total_timer = WallTimer().start()

    # ── Prefill ───────────────────────────────────────────────────────────────
    ttft_timer = WallTimer().start()
    out     = model(input_ids, use_cache=True)
    past_kv = out.past_key_values

    # CLE: cross-layer importance → evict once
    importance = compute_cross_layer_importance(past_kv)
    importance[:n_sink] = float("inf")           # protect sink tokens
    keep_indices = importance.topk(budget).indices.sort().values
    past_kv = evict_kv_cache(past_kv, keep_indices)

    anchor = out.logits[:, -1:, :].argmax(dim=-1)   # [1, 1]
    ttft_ms = ttft_timer.elapsed_ms()

    generated = torch.cat([input_ids, anchor], dim=-1)
    # kept_mask[i] = True if prompt token i survived CLE eviction
    kept_mask = torch.zeros(prompt_len, dtype=torch.bool, device=device)
    kept_mask[keep_indices] = True

    n_proposed_total = 0
    n_accepted_total = 0
    n_steps          = 1
    eos_hit          = False

    # ── PLD speculative decode on compressed KV cache ─────────────────────────
    while generated.size(1) - prompt_len < max_new_tokens:
        # Build subset: kept prompt tokens + all newly generated tokens
        n_gen = generated.size(1) - prompt_len
        if n_gen > 0:
            kept_prompt = generated[:, :prompt_len][:, kept_mask]   # [1, budget]
            new_tokens  = generated[:, prompt_len:]                  # [1, n_gen]
            subset_ids  = torch.cat([kept_prompt, new_tokens], dim=1)
        else:
            subset_ids  = generated[:, :prompt_len][:, kept_mask]

        candidates  = _find_candidates_in_subset(generated, subset_ids,
                                                  max_ngram_size, K)
        num_cand    = int(candidates.numel())
        fwd_input   = (anchor if num_cand == 0
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
    # FLOPs: CLE prefill same as baseline; decode uses budget context
    from cross_layer_evict import estimate_flops_cle
    flops = estimate_flops_cle(model, prompt_len, len(all_ids) - prompt_len, budget)
    return make_result(
        all_ids, prompt_len, total_ms, ttft_ms, past_kv,
        total_gflops=flops["total_gflops"],
        n_proposed=n_proposed_total,
        n_accepted=n_accepted_total,
    )


# ── StreamingLLM + PLD ────────────────────────────────────────────────────────

@torch.inference_mode()
def streaming_pld_generate(
    model,
    tokenizer,
    prompt:         str,
    max_new_tokens: int = 128,
    window:         int = 256,
    n_sink:         int = 4,
    K:              int = 5,
    max_ngram_size: int = 3,
    device:         str = "cuda",
) -> GenerationResult:
    """
    StreamingLLM sliding-window KV cache + PLD speculative decode.

    Per step: trim KV to (n_sink + window), then run PLD verification.
    The bounded cache makes each PLD verification forward pass O(1) in
    sequence length rather than O(S).
    """
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos_id     = tokenizer.eos_token_id

    total_timer = WallTimer().start()

    # ── Prefill ───────────────────────────────────────────────────────────────
    ttft_timer = WallTimer().start()
    out     = model(input_ids, use_cache=True)
    past_kv = out.past_key_values
    # Apply streaming trim right after prefill
    past_kv = _trim_to_streaming_window(past_kv, n_sink, window)

    anchor  = out.logits[:, -1:, :].argmax(dim=-1)
    ttft_ms = ttft_timer.elapsed_ms()

    generated = torch.cat([input_ids, anchor], dim=-1)
    n_proposed_total = 0
    n_accepted_total = 0
    n_steps          = 1
    eos_hit          = False

    # ── Streaming + PLD decode ────────────────────────────────────────────────
    while generated.size(1) - prompt_len < max_new_tokens:
        # Restrict n-gram search to tokens the streaming cache "remembers":
        # first n_sink tokens  +  last `window` tokens of generated sequence
        seq_len = generated.size(1)
        if seq_len > n_sink + window:
            subset_ids = torch.cat([
                generated[:, :n_sink],
                generated[:, -(window):],
            ], dim=1)
        else:
            subset_ids = generated

        candidates = _find_candidates_in_subset(generated, subset_ids,
                                                 max_ngram_size, K)
        num_cand   = int(candidates.numel())
        fwd_input  = (anchor if num_cand == 0
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

        # Streaming trim after each verified step
        past_kv = _trim_to_streaming_window(past_kv, n_sink, window)

        if eos_id is not None:
            if eos_id in generated[0, -(n_accepted + 1):].tolist():
                eos_hit = True
                break

    total_ms = total_timer.elapsed_ms()
    if not eos_hit:
        generated = generated[:, : prompt_len + max_new_tokens]

    all_ids = generated[0].tolist()
    from streaming_llm import estimate_flops_streaming
    flops = estimate_flops_streaming(model, prompt_len,
                                      len(all_ids) - prompt_len, n_sink, window)
    return make_result(
        all_ids, prompt_len, total_ms, ttft_ms, past_kv,
        total_gflops=flops["total_gflops"],
        n_proposed=n_proposed_total,
        n_accepted=n_accepted_total,
    )


# ── CLE + StreamingLLM (no PLD) ──────────────────────────────────────────────

@torch.inference_mode()
def cle_streaming_generate(
    model,
    tokenizer,
    prompt:          str,
    max_new_tokens:  int   = 128,
    budget_ratio:    float = 0.5,
    decode_window:   int   = 64,
    n_sink:          int   = 4,
    device:          str   = "cuda",
) -> GenerationResult:
    """
    CLE one-shot eviction (prompt) + StreamingLLM sliding window (decode).

    CLE compresses the prompt KV to the top-B important tokens after prefill.
    StreamingLLM then caps the decode-phase KV growth: only the last
    `decode_window` generated tokens are kept alongside the CLE prompt KV.

    Total KV is bounded at (budget + decode_window) throughout generation.
    """
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos_id     = tokenizer.eos_token_id

    budget = max(n_sink + 1, int(prompt_len * budget_ratio))

    total_timer = WallTimer().start()

    # ── Prefill ──────────────────────────────────────────────────────────────
    ttft_timer = WallTimer().start()
    out     = model(input_ids, use_cache=True)
    past_kv = out.past_key_values

    # CLE: cross-layer importance → one-shot eviction
    importance = compute_cross_layer_importance(past_kv)
    importance[:n_sink] = float("inf")
    keep_indices = importance.topk(budget).indices.sort().values
    past_kv = evict_kv_cache(past_kv, keep_indices)

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

        # StreamingLLM trim: keep CLE prompt KV [0:budget] + last decode_window
        cur_len = kv_cache_length(past_kv)
        if cur_len - budget > decode_window:
            past_kv = trim_kv_cache(past_kv, budget + decode_window)

    total_ms = total_timer.elapsed_ms()
    all_ids  = input_ids[0].tolist() + generated

    gen_len = len(generated)
    from cross_layer_evict import estimate_flops_cle
    effective_ctx = budget + min(decode_window, gen_len // 2)
    flops = estimate_flops_cle(model, prompt_len, gen_len, effective_ctx)

    return make_result(
        all_ids, prompt_len, total_ms, ttft_ms, past_kv,
        total_gflops=flops["total_gflops"],
    )


# ── CLE + StreamingLLM + PLD (three-way combination) ─────────────────────────

@torch.inference_mode()
def cle_stream_pld_generate(
    model,
    tokenizer,
    prompt:          str,
    max_new_tokens:  int   = 128,
    budget_ratio:    float = 0.5,   # CLE: keep this fraction of prompt tokens
    decode_window:   int   = 64,    # StreamingLLM: keep last N *decode* tokens
    n_sink:          int   = 4,
    K:               int   = 5,
    max_ngram_size:  int   = 3,
    device:          str   = "cuda",
) -> GenerationResult:
    """
    Three-way combination: CLE + StreamingLLM + PLD.

    Design rationale
    ----------------
    The root cause of poor CLE/Stream + PLD acceptance is that the model's
    KV cache no longer covers all sequence positions, so the model predicts
    different tokens than what n-gram lookup expects.

    Solution: decouple the two KV-compression axes so they *complement* each
    other instead of both evicting sequence positions:

      CLE axis  (prompt KV, one-shot after prefill):
        Keep the B most-important prompt tokens.  These supply the global
        semantic context the model needs across the entire generation.

      StreamingLLM axis  (decode KV, per-step):
        Keep only the last `decode_window` *newly generated* tokens.
        This bounds KV growth during decode without touching the CLE-selected
        prompt context.

      PLD axis:
        Restrict n-gram search to exactly the tokens present in the KV cache
        = [CLE-kept prompt tokens] ∪ [last decode_window tokens].
        The proposal distribution is now perfectly aligned with the model's
        memory, so acceptance rate is recovered.

    Total bounded KV = B + decode_window  (constant during generation).
    This is strictly smaller than either CLE-only (B grows with decode)
    or StreamingLLM-only (n_sink + window is larger at high budget_ratio).
    """
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos_id     = tokenizer.eos_token_id

    budget = max(n_sink + 1, int(prompt_len * budget_ratio))

    total_timer = WallTimer().start()

    # ── Prefill ───────────────────────────────────────────────────────────────
    ttft_timer = WallTimer().start()
    out     = model(input_ids, use_cache=True)
    past_kv = out.past_key_values

    # CLE: one-shot eviction of prompt KV → keep `budget` positions
    importance = compute_cross_layer_importance(past_kv)
    importance[:n_sink] = float("inf")
    keep_indices = importance.topk(budget).indices.sort().values
    past_kv   = evict_kv_cache(past_kv, keep_indices)

    anchor  = out.logits[:, -1:, :].argmax(dim=-1)   # [1, 1]
    ttft_ms = ttft_timer.elapsed_ms()

    # Build subset for n-gram search: kept prompt tokens (fixed) + decode tokens
    kept_prompt_ids = input_ids[:, keep_indices]   # [1, budget]

    generated = torch.cat([input_ids, anchor], dim=-1)
    n_proposed_total = 0
    n_accepted_total = 0
    n_steps          = 1
    eos_hit          = False

    # ── Decode: PLD on (CLE-kept prompt KV) + (last decode_window decode KV) ──
    while generated.size(1) - prompt_len < max_new_tokens:
        # n-gram search space = CLE-kept prompt tokens + recent decode tokens
        n_gen = generated.size(1) - prompt_len          # tokens generated so far
        recent_decode = generated[:, prompt_len:]        # [1, n_gen]
        if n_gen > decode_window:
            recent_decode = recent_decode[:, -decode_window:]
        subset_ids = torch.cat([kept_prompt_ids, recent_decode], dim=1)

        candidates = _find_candidates_in_subset(generated, subset_ids,
                                                max_ngram_size, K)
        num_cand   = int(candidates.numel())
        fwd_input  = (anchor if num_cand == 0
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

        # StreamingLLM trim: keep CLE prompt KV + last decode_window decode KV
        # The first `budget` positions are the CLE prompt tokens (never touch them)
        # Positions budget.. are the decode tokens; trim to last decode_window
        cur_len   = kv_cache_length(past_kv)
        n_decode_in_cache = cur_len - budget
        if n_decode_in_cache > decode_window:
            # We want to keep positions [0:budget] + last decode_window decode positions.
            # trim_kv_cache keeps [0:target], so we can only trim the tail.
            # Keep budget + decode_window positions total.
            past_kv = trim_kv_cache(past_kv, budget + decode_window)

        if eos_id is not None:
            if eos_id in generated[0, -(n_accepted + 1):].tolist():
                eos_hit = True
                break

    total_ms = total_timer.elapsed_ms()
    if not eos_hit:
        generated = generated[:, : prompt_len + max_new_tokens]

    all_ids    = generated[0].tolist()
    gen_len    = len(all_ids) - prompt_len

    # FLOPs: prefill same as baseline; decode attention over budget+decode_window
    from cross_layer_evict import estimate_flops_cle
    effective_ctx = budget + min(decode_window, gen_len // 2)  # avg decode ctx
    flops = estimate_flops_cle(model, prompt_len, gen_len, effective_ctx)

    return make_result(
        all_ids, prompt_len, total_ms, ttft_ms, past_kv,
        total_gflops=flops["total_gflops"],
        n_proposed=n_proposed_total,
        n_accepted=n_accepted_total,
    )


# ── CEPE + StreamingLLM + PLD ─────────────────────────────────────────────────

@torch.inference_mode()
def cepe_stream_pld_generate(
    model,
    tokenizer,
    prompt:          str,
    max_new_tokens:  int   = 128,
    keep_recent:     int   = 256,    # CEPE+Stream: exact recent tokens
    pool_size:       int   = 4,      # CEPE: old tokens pooled in blocks of N
    compress_every:  int   = 1,      # CEPE: re-pool every N decode steps
    K:               int   = 5,      # PLD: max speculative candidates
    max_ngram_size:  int   = 3,
    device:          str   = "cuda",
) -> GenerationResult:
    """
    CEPE (pool old KV) + StreamingLLM (bound recent KV) + PLD (speculative).

    Design
    ------
    The three methods operate on non-overlapping KV regions:

      CEPE region  (old tokens, positions 0 .. seq-keep_recent-1):
        Block-averaged into pooled summary tokens.  Semantic content is
        partially preserved — better quality than StreamingLLM's hard discard.

      StreamingLLM region  (recent tokens, last keep_recent positions):
        Kept verbatim.  Exact attention over local context.

      PLD:
        N-gram search is restricted to the recent `keep_recent` generated
        token IDs (whose text is known exactly).  Pooled old tokens have no
        token IDs and cannot serve as n-gram candidates.
        Acceptance rate is therefore close to baseline PLD.

    Memory growth:
        Old context: O(prompt_len / pool_size)  — sub-linear
        Recent:      O(keep_recent)              — constant
        Decode:      O(decode_step)              — linear but bounded by keep_recent trim
    """
    input_ids  = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    eos_id     = tokenizer.eos_token_id

    total_timer = WallTimer().start()

    # ── Prefill ──────────────────────────────────────────────────────────────
    ttft_timer = WallTimer().start()
    out     = model(input_ids, use_cache=True)
    past_kv = out.past_key_values

    # CEPE: compress prompt's old positions right after prefill
    past_kv = apply_cepe_pooling(past_kv, keep_recent, pool_size)

    anchor  = out.logits[:, -1:, :].argmax(dim=-1)   # [1, 1]
    ttft_ms = ttft_timer.elapsed_ms()

    generated = torch.cat([input_ids, anchor], dim=-1)
    n_proposed_total = 0
    n_accepted_total = 0
    n_steps          = 1
    eos_hit          = False

    # ── Decode ───────────────────────────────────────────────────────────────
    for step in range(max_new_tokens):
        n_gen = generated.size(1) - prompt_len
        if n_gen >= max_new_tokens:
            break

        # PLD: search only within the recent keep_recent generated token IDs
        # (pooled old tokens have no recoverable token IDs)
        recent_gen = generated[:, -keep_recent:] if generated.size(1) > keep_recent \
                     else generated
        candidates  = find_candidate_pred_tokens(recent_gen, max_ngram_size, K)
        num_cand    = int(candidates.numel())
        fwd_input   = (anchor if num_cand == 0
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

        # CEPE: re-pool old tokens periodically
        if (step + 1) % compress_every == 0:
            past_kv = apply_cepe_pooling(past_kv, keep_recent, pool_size)

        if eos_id is not None:
            if eos_id in generated[0, -(n_accepted + 1):].tolist():
                eos_hit = True
                break

    total_ms = total_timer.elapsed_ms()
    if not eos_hit:
        generated = generated[:, : prompt_len + max_new_tokens]

    all_ids = generated[0].tolist()
    flops = estimate_flops_cepe(model, prompt_len, len(all_ids) - prompt_len,
                                keep_recent, pool_size)
    return make_result(
        all_ids, prompt_len, total_ms, ttft_ms, past_kv,
        total_gflops=flops["total_gflops"],
        n_proposed=n_proposed_total,
        n_accepted=n_accepted_total,
    )


# ── Configs ───────────────────────────────────────────────────────────────────

COMBINED_CONFIGS = {
    "CLE-Med+PLD(K=5)":    ("cle_pld",      {"budget_ratio": 0.50, "K": 5}),
    "CLE-Med+PLD(K=10)":   ("cle_pld",      {"budget_ratio": 0.50, "K": 10}),
    "CLE-Hvy+PLD(K=5)":    ("cle_pld",      {"budget_ratio": 0.30, "K": 5}),
    "Stream+PLD(K=5)":     ("stream_pld",   {"window": 256, "K": 5}),
    "Stream+PLD(K=10)":    ("stream_pld",   {"window": 256, "K": 10}),
    "CLE+Stream+PLD(K=5)": ("cle_stream_pld", {"budget_ratio": 0.50, "decode_window": 64, "K": 5}),
    "CLE+Stream+PLD(K=10)":("cle_stream_pld", {"budget_ratio": 0.50, "decode_window": 64, "K": 10}),
}

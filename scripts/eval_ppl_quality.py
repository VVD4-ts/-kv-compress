"""
PPL quality comparison: CEPE vs StreamingLLM vs Baseline
Long-text evaluation with 100,000 tokens (PG-19).

Key question: at the same KV budget, does CEPE preserve quality
better than StreamingLLM (which hard-discards old tokens)?
"""

from __future__ import annotations

import argparse
import math
import sys
import json
from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import load_model, set_seed
from cepe import apply_cepe_pooling
from streaming_llm import _trim_to_streaming_window
from cross_layer_evict import compute_cross_layer_importance, evict_kv_cache


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_long_text(n_tokens: int, tokenizer) -> torch.Tensor:
    """Load PG-19 (fallback: repeated WikiText-2) up to n_tokens tokens."""
    from datasets import load_dataset

    print(f"  Loading text (~{n_tokens} tokens)...")
    for hf_name, spl in [
        ("emozilla/pg19-test-tokenized", "test"),
        ("storytracer/pg19-books", "train"),
    ]:
        try:
            ds = load_dataset(hf_name, split=spl, streaming=True)
            chunks = []
            total  = 0
            for sample in ds:
                text = (sample.get("text") or sample.get("book_text") or "")
                if len(text) < 1000:
                    continue
                ids = tokenizer(text, return_tensors="pt").input_ids[0]
                chunks.append(ids)
                total += ids.shape[0]
                if total >= n_tokens:
                    break
            if total >= n_tokens // 2:
                all_ids = torch.cat(chunks)[:n_tokens]
                print(f"  [source: {hf_name}, {all_ids.shape[0]} tokens]")
                return all_ids
        except Exception:
            continue

    # Fallback: repeated WikiText-2
    print("  [source: wikitext-2 repeated]")
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())
    while len(text) < n_tokens * 6:
        text = text + "\n\n" + text
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    return ids[:n_tokens]


# ── PPL with sliding window + KV transform ────────────────────────────────────

def compute_ppl(
    model,
    input_ids:    torch.Tensor,   # [seq_len]
    window:       int   = 1024,
    stride:       int   = 512,
    device:       str   = "cuda",
    kv_transform: Optional[Callable] = None,
    label:        str   = "",
) -> float:
    """
    Sliding-window PPL.  kv_transform(past_kv) is called after each
    token's forward pass to apply compression (CEPE pooling,
    StreamingLLM trim, etc.).
    """
    seq_len   = input_ids.shape[0]
    n_windows = max(1, (seq_len - 1) // stride)
    nlls      = []
    prev_end  = 0

    for win_idx, begin in enumerate(range(0, seq_len, stride)):
        end        = min(begin + window, seq_len)
        chunk      = input_ids[begin:end].unsqueeze(0).to(device)  # [1, chunk_len]
        target_len = end - prev_end
        if target_len <= 0:
            prev_end = end
            continue

        past_kv:    Optional[object] = None
        win_logits: list             = []

        with torch.no_grad():
            for t in range(chunk.shape[1]):
                tok = chunk[:, t:t+1]
                out = model(tok, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                if kv_transform is not None:
                    past_kv = kv_transform(past_kv)
                win_logits.append(out.logits[:, 0, :])

        logits = torch.stack(win_logits, dim=1)          # [1, chunk, vocab]

        if target_len > 1:
            lp  = F.log_softmax(logits[:, -target_len:-1, :], dim=-1)
            lab = chunk[:, -target_len+1:]
            nll = -lp.gather(2, lab.unsqueeze(2)).squeeze(2)
            nlls.append(nll.mean().item())

        prev_end = end
        if (win_idx + 1) % 10 == 0 or end == seq_len:
            ppl = math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")
            print(f"    [{label}] window {win_idx+1}/{n_windows}  "
                  f"pos={end}/{seq_len}  ppl={ppl:.2f}", flush=True)

        if end == seq_len:
            break

    return math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="EleutherAI/pythia-70m")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-tokens",   type=int, default=100_000)
    p.add_argument("--window",     type=int, default=1024)
    p.add_argument("--stride",     type=int, default=512)
    p.add_argument("--out-dir",    default="results")
    args = p.parse_args()

    set_seed(42)
    model, tokenizer = load_model(args.model, device=args.device)
    model.eval()

    input_ids = load_long_text(args.n_tokens, tokenizer)
    print(f"  Evaluating on {input_ids.shape[0]} tokens  "
          f"(window={args.window}, stride={args.stride})\n")

    results = {}

    # ── Configurations ────────────────────────────────────────────────────────
    # Each entry: (label, kv_transform_or_None)
    # All CEPE/Stream configs chosen to produce ~same effective KV budget
    # for fair quality comparison.
    #
    # Approximate effective KV at ctx=1024:
    #   StreamingLLM(W=256): 4+256 = 260 tokens
    #   CEPE(r=256,p=4):     ceil(768/4)+256 = 192+256 = 448 tokens
    #   CEPE(r=256,p=8):     ceil(768/8)+256 = 96+256  = 352 tokens
    #   CEPE(r=128,p=8):     ceil(896/8)+128 = 112+128 = 240 tokens

    def make_stream(n_sink, window):
        def _t(kv): return _trim_to_streaming_window(kv, n_sink, window)
        return _t

    def make_cepe(keep_recent, pool_size):
        def _t(kv): return apply_cepe_pooling(kv, keep_recent, pool_size)
        return _t

    def make_cle(budget_ratio):
        state = {"done": False}
        def _t(kv):
            if state["done"]:
                return kv
            imp = compute_cross_layer_importance(kv)
            n   = imp.shape[0]
            budget = max(1, int(n * budget_ratio))
            keep = imp.topk(budget).indices.sort().values
            state["done"] = True
            return evict_kv_cache(kv, keep)
        return _t

    configs = [
        # ── Baseline ──────────────────────────────────────────────────────────
        ("Baseline",                None),

        # ── StreamingLLM (hard discard) ───────────────────────────────────────
        ("StreamingLLM(sink=4,W=64)",  make_stream(4, 64)),
        ("StreamingLLM(sink=4,W=128)", make_stream(4, 128)),
        ("StreamingLLM(sink=4,W=256)", make_stream(4, 256)),
        ("StreamingLLM(sink=4,W=512)", make_stream(4, 512)),

        # ── CEPE (pool old, keep recent exact) ────────────────────────────────
        ("CEPE(r=64,p=8)",   make_cepe(64,  8)),
        ("CEPE(r=128,p=8)",  make_cepe(128, 8)),
        ("CEPE(r=256,p=8)",  make_cepe(256, 8)),
        ("CEPE(r=512,p=8)",  make_cepe(512, 8)),
        ("CEPE(r=128,p=4)",  make_cepe(128, 4)),
        ("CEPE(r=256,p=4)",  make_cepe(256, 4)),
    ]

    for label, transform in configs:
        print(f"\n── {label} ──")
        ppl = compute_ppl(
            model, input_ids,
            window=args.window,
            stride=args.stride,
            device=args.device,
            kv_transform=transform,
            label=label,
        )
        results[label] = round(ppl, 3)
        print(f"  → PPL = {ppl:.3f}")

    # ── Print summary ─────────────────────────────────────────────────────────
    base_ppl = results.get("Baseline", 1.0)
    print(f"\n{'═'*60}")
    print(f"  PPL Summary  ({input_ids.shape[0]} tokens, PG-19/WikiText-2)")
    print(f"{'═'*60}")
    print(f"  {'Method':<30} {'PPL':>8}  {'ΔPPL':>8}")
    print(f"  {'─'*50}")
    for label, ppl in results.items():
        delta = ppl - base_ppl
        marker = " ◀ CEPE" if "CEPE" in label else (" ◀ Stream" if "Stream" in label else "")
        print(f"  {label:<30} {ppl:>8.3f}  {delta:>+8.3f}{marker}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out_dir) / "ppl_quality_comparison.json"
    Path(args.out_dir).mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "n_tokens": int(input_ids.shape[0]),
            "window": args.window,
            "stride": args.stride,
            "results": results
        }, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()

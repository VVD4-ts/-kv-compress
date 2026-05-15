"""
CLE + StreamingLLM combination evaluation.

Runs two configs on top of existing eval_full_results.json:
  ① CLE-Heavy(0.30) + Stream(W=64)   → very aggressive, KV ≈ 32% of baseline
  ② CLE-Medium(0.50) + Stream(W=128) → balanced,        KV ≈ 54% of baseline

Saves merged results to results/eval_combined_results.json and prints a
unified summary table alongside the single-method baselines.
"""

from __future__ import annotations
import json, math, sys, argparse
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils              import load_model, set_seed, kv_size_mb, make_result, WallTimer, num_layers
from kv_utils           import kv_cache_length
from cross_layer_evict  import compute_cross_layer_importance, evict_kv_cache
from streaming_llm      import _trim_to_streaming_window
from combined           import cle_streaming_generate


PROGRESS_PATH = Path("results/eval_combined_progress.json")


# ─────────────────────────────────────────────────────────────────────────────
# PPL transform: CLE once per sliding-window + StreamingLLM every step
# ─────────────────────────────────────────────────────────────────────────────

def make_cle_stream_transform(budget_ratio: float, n_sink: int, window: int):
    """
    Combined KV transform for sliding-window PPL evaluation.

    Detects window resets (KV length drops) to re-arm CLE eviction each window.
    - CLE  : one-shot importance-based eviction to `budget_ratio` of current KV
    - Stream: per-step sink+window trim applied after CLE
    """
    state = {"cle_done": False, "last_len": 0}

    def transform(kv):
        cur_len = kv_cache_length(kv)

        # New sliding window detected → re-arm CLE
        if cur_len < state["last_len"]:
            state["cle_done"] = False
        state["last_len"] = cur_len

        # CLE: one-shot eviction (once per window)
        if not state["cle_done"]:
            imp    = compute_cross_layer_importance(kv)
            n      = imp.shape[0]
            budget = max(n_sink + 1, int(n * budget_ratio))
            if n > budget:
                keep = imp.topk(budget).indices.sort().values
                kv   = evict_kv_cache(kv, keep)
            state["cle_done"] = True

        # StreamingLLM: always trim to sink + window
        kv = _trim_to_streaming_window(kv, n_sink, window)
        return kv

    return transform


# ─────────────────────────────────────────────────────────────────────────────
# PPL (sliding window)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ppl(model, input_ids, window, stride, device,
                kv_transform=None, label="") -> float:
    seq_len = input_ids.shape[0]
    n_win   = max(1, (seq_len - 1) // stride)
    nlls, prev_end = [], 0

    for wi, begin in enumerate(range(0, seq_len, stride)):
        end   = min(begin + window, seq_len)
        chunk = input_ids[begin:end].unsqueeze(0).to(device)
        tlen  = end - prev_end
        if tlen <= 0:
            prev_end = end
            continue

        past_kv, logits_list = None, []
        with torch.no_grad():
            for t in range(chunk.shape[1]):
                out     = model(chunk[:, t:t+1], past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                if kv_transform:
                    past_kv = kv_transform(past_kv)
                logits_list.append(out.logits[:, 0, :])

        logits = torch.stack(logits_list, dim=1)
        if tlen > 1:
            lp  = F.log_softmax(logits[:, -tlen:-1, :], dim=-1)
            lab = chunk[:, -tlen+1:]
            nll = -lp.gather(2, lab.unsqueeze(2)).squeeze(2)
            nlls.append(nll.mean().item())

        prev_end = end
        if (wi + 1) % 20 == 0 or end == seq_len:
            ppl = math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")
            print(f"    [{label}] {wi+1}/{n_win}  pos={end}/{seq_len}  ppl={ppl:.2f}",
                  flush=True)
        if end == seq_len:
            break

    return math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Speed benchmark
# ─────────────────────────────────────────────────────────────────────────────

def bench_speed(fn, model, tokenizer, prompt, max_new_tokens, device,
                n_warmup=1, n_runs=3, **kw) -> dict:
    results = []
    for i in range(n_warmup + n_runs):
        set_seed(42)
        r = fn(model=model, tokenizer=tokenizer, prompt=prompt,
               max_new_tokens=max_new_tokens, device=device, **kw)
        if i >= n_warmup:
            results.append(r)
    return {
        "ttft_ms":          round(mean(r.ttft_ms    for r in results), 3),
        "tpot_ms":          round(mean(r.tpot_ms    for r in results), 3),
        "throughput":       round(mean(r.throughput  for r in results), 2),
        "kv_mb":            round(mean(r.kv_size_mb  for r in results), 3),
        "total_gflops":     round(results[0].total_gflops, 4),
        "avg_gflops_per_tok": round(results[0].avg_gflops_per_tok, 6),
        "num_gen":          results[0].num_generated,
    }


def save_progress(data: dict):
    PROGRESS_PATH.parent.mkdir(exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="EleutherAI/pythia-70m")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ppl-tokens",     type=int, default=100_000)
    p.add_argument("--speed-ctx",      type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--ppl-window",     type=int, default=1024)
    p.add_argument("--ppl-stride",     type=int, default=512)
    p.add_argument("--n-runs",         type=int, default=3)
    p.add_argument("--baseline-json",  default="results/eval_full_results.json")
    p.add_argument("--out-dir",        default="results")
    args = p.parse_args()

    set_seed(42)
    model, tokenizer = load_model(args.model, device=args.device)
    print(f"\nModel: {args.model}  Device={args.device}")

    # Resume if progress exists
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            progress = json.load(f)
        print(f"  [Resuming: PPL done={list(progress['ppl'].keys())}]")
    else:
        progress = {"status": "running", "ppl": {}, "speed": {}}
    save_progress(progress)

    # ── Load PG-19 tokens ─────────────────────────────────────────────────────
    from datasets import load_dataset
    print(f"  Loading ~{args.ppl_tokens} tokens from PG-19 …")
    ppl_ids = None
    for hf_name, spl in [("emozilla/pg19-test-tokenized", "test"),
                          ("storytracer/pg19-books", "train")]:
        try:
            ds     = load_dataset(hf_name, split=spl, streaming=True)
            chunks, total = [], 0
            for s in ds:
                txt = s.get("text") or s.get("book_text") or ""
                if len(txt) < 1000: continue
                ids = tokenizer(txt, return_tensors="pt").input_ids[0]
                chunks.append(ids); total += len(ids)
                if total >= args.ppl_tokens: break
            if total >= args.ppl_tokens // 2:
                ppl_ids = torch.cat(chunks)[:args.ppl_tokens]
                print(f"  [source: {hf_name}, {ppl_ids.shape[0]} tokens]")
                break
        except Exception:
            continue
    if ppl_ids is None:
        ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(r["text"] for r in ds if r["text"].strip())
        while len(text) < args.ppl_tokens * 6: text = text + "\n\n" + text
        ppl_ids = tokenizer(text, return_tensors="pt").input_ids[0][:args.ppl_tokens]
        print("  [source: wikitext-2 repeated]")

    speed_prompt = tokenizer.decode(ppl_ids[:args.speed_ctx], skip_special_tokens=True)
    print(f"  Speed prompt: {tokenizer(speed_prompt, return_tensors='pt').input_ids.shape[1]} tokens\n")

    # ── Two combination configs ───────────────────────────────────────────────
    #
    #   ① CLE-Heavy + Stream(W=64)   budget=0.30, decode_window=64
    #      → KV = 0.30×prompt + 64 decode  ≈ 1228+64 = 1292 tokens  (~69% reduction)
    #
    #   ② CLE-Medium + Stream(W=128) budget=0.50, decode_window=128
    #      → KV = 0.50×prompt + 128 decode ≈ 2048+128 = 2176 tokens (~47% reduction)
    #
    ppl_configs = [
        ("CLE-Hvy+Stream(W=64)",
         make_cle_stream_transform(0.30, 4, 64)),
        ("CLE-Med+Stream(W=128)",
         make_cle_stream_transform(0.50, 4, 128)),
    ]
    speed_configs = [
        ("CLE-Hvy+Stream(W=64)",
         cle_streaming_generate,
         {"budget_ratio": 0.30, "decode_window": 64,  "n_sink": 4}),
        ("CLE-Med+Stream(W=128)",
         cle_streaming_generate,
         {"budget_ratio": 0.50, "decode_window": 128, "n_sink": 4}),
    ]

    # ── PPL ───────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  PPL Evaluation  ({ppl_ids.shape[0]} tokens)")
    print(f"{'═'*60}")

    for label, transform in ppl_configs:
        if label in progress["ppl"]:
            print(f"\n── {label} ── (already done: PPL={progress['ppl'][label]})")
            continue
        print(f"\n── {label} ──")
        ppl = compute_ppl(model, ppl_ids,
                          window=args.ppl_window, stride=args.ppl_stride,
                          device=args.device, kv_transform=transform, label=label)
        progress["ppl"][label] = round(ppl, 3)
        save_progress(progress)
        print(f"  → PPL = {ppl:.3f}")

    # ── Speed ─────────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Speed Benchmark  (ctx={args.speed_ctx}, gen={args.max_new_tokens})")
    print(f"{'═'*60}")

    for label, fn, kw in speed_configs:
        if label in progress["speed"] and "error" not in progress["speed"][label]:
            s = progress["speed"][label]
            print(f"\n── {label} ── (already done: TPOT={s.get('tpot_ms','?')}ms)")
            continue
        print(f"\n── {label} ──", flush=True)
        try:
            s = bench_speed(fn, model, tokenizer, speed_prompt,
                            args.max_new_tokens, args.device,
                            n_warmup=1, n_runs=args.n_runs, **kw)
            progress["speed"][label] = s
            save_progress(progress)
            print(f"  TTFT={s['ttft_ms']:.1f}ms  TPOT={s['tpot_ms']:.2f}ms  "
                  f"Tput={s['throughput']:.1f}tok/s  KV={s['kv_mb']:.2f}MB  "
                  f"GFLOPs={s['total_gflops']:.3f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            progress["speed"][label] = {"error": str(e)}
            save_progress(progress)

    progress["status"] = "done"
    save_progress(progress)

    # ── Load baseline results and merge ───────────────────────────────────────
    baseline = {}
    base_path = Path(args.baseline_json)
    if base_path.exists():
        with open(base_path) as f:
            bd = json.load(f)
        for lbl, ppl in bd["ppl"].items():
            baseline[lbl] = {"ppl": ppl, "speed": bd["speed"].get(lbl, {})}

    # Merge combination results in
    for lbl, ppl in progress["ppl"].items():
        baseline[lbl] = {"ppl": ppl, "speed": progress["speed"].get(lbl, {})}

    # ── Print unified summary ─────────────────────────────────────────────────
    base_ppl  = baseline.get("Baseline", {}).get("ppl", 1)
    base_tpot = baseline.get("Baseline", {}).get("speed", {}).get("tpot_ms", 1)
    base_kv   = baseline.get("Baseline", {}).get("speed", {}).get("kv_mb", 1)

    ORDER = [
        "Baseline",
        "CLE-Medium(-50%KV)", "CLE-Heavy(-70%KV)",
        "CLE-Med+Stream(W=128)", "CLE-Hvy+Stream(W=64)",
        "StreamingLLM(W=128)", "StreamingLLM(W=256)", "StreamingLLM(W=512)",
        "YOCO(split=4,-33%KV)", "YOCO(split=5,-17%KV)",
        "PLD(K=5)", "PLD(K=10)",
    ]
    rows = [l for l in ORDER if l in baseline]
    rows += [l for l in baseline if l not in rows]

    print(f"\n{'═'*100}")
    print(f"  Unified Summary (100k-token PG-19 PPL  +  4096-token Speed)")
    print(f"{'═'*100}")
    print(f"  {'Method':<28} {'PPL':>8} {'ΔPPL':>7} | "
          f"{'TTFT':>7} {'TPOT':>7} {'Tok/s':>7} {'KV↓%':>6} {'GFLOPs':>8} {'Spd':>6}")
    print(f"  {'─'*96}")

    for lbl in rows:
        d    = baseline[lbl]
        ppl  = d["ppl"]
        dppl = ppl - base_ppl
        sp   = d["speed"]
        ttft = sp.get("ttft_ms", float("nan"))
        tpot = sp.get("tpot_ms", float("nan"))
        tput = sp.get("throughput", float("nan"))
        kv   = sp.get("kv_mb", float("nan"))
        fl   = sp.get("total_gflops", float("nan"))
        spd  = base_tpot / tpot if tpot and tpot > 0 else float("nan")
        kvrd = (1 - kv / base_kv) * 100 if kv and base_kv else float("nan")
        star = " ◀" if "CLE" in lbl and "Stream" in lbl else ""
        print(f"  {lbl:<28} {ppl:>8.2f} {dppl:>+7.2f} | "
              f"{ttft:>7.1f} {tpot:>7.2f} {tput:>7.1f} "
              f"{kvrd:>5.1f}% {fl:>8.3f} {spd:>5.2f}x{star}")

    # Save merged JSON
    out = Path(args.out_dir) / "eval_combined_results.json"
    with open(out, "w") as f:
        json.dump({"baseline": baseline, "combined": progress}, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

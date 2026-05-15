"""
Benchmark: CEPE + StreamingLLM + PLD vs single methods.

Key comparison:
  - Baseline (full KV)
  - StreamingLLM (hard discard of old tokens)
  - CEPE alone (pool old tokens, keep recent exact)
  - PLD alone (speculative on full KV)
  - CEPE + Stream + PLD (three-way: pool + bound + speculative)

Reports: TTFT, TPOT, Throughput, KV MB, GFLOPs, Acceptance Rate
"""

import sys, os, json
from pathlib import Path
from statistics import mean, stdev

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import load_model, set_seed, num_layers
from baseline import greedy_generate
from streaming_llm import streaming_generate
from cepe import cepe_generate, CEPE_CONFIGS
from pld import pld_generate
from combined import cepe_stream_pld_generate

CONTEXT_LENGTHS = [512, 1024, 2048]
MAX_NEW_TOKENS  = 200
N_WARMUP = 1
N_RUNS   = 3


def load_prompt(n_tokens, tokenizer):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())
    ids  = tokenizer(text, return_tensors="pt").input_ids[0]
    if ids.shape[0] < n_tokens:
        ids = ids.repeat((n_tokens // ids.shape[0]) + 1)
    return tokenizer.decode(ids[:n_tokens], skip_special_tokens=True)


def bench(fn, n_warmup, n_runs, **kw):
    results = []
    for i in range(n_warmup + n_runs):
        set_seed(42)
        r = fn(**kw)
        if i >= n_warmup:
            results.append(r)
    acc_rates = [
        r.n_accepted / r.n_proposed
        if getattr(r, "n_proposed", 0) > 0 else None
        for r in results
    ]
    acc = next((a for a in acc_rates if a is not None), None)
    return {
        "ttft_ms":    mean(r.ttft_ms    for r in results),
        "tpot_ms":    mean(r.tpot_ms    for r in results),
        "throughput": mean(r.throughput  for r in results),
        "kv_mb":      mean(r.kv_size_mb for r in results),
        "total_gflops":       results[0].total_gflops,
        "avg_gflops_per_tok": results[0].avg_gflops_per_tok,
        "num_gen":    results[0].num_generated,
        "acc_rate":   round(acc, 4) if acc is not None else None,
        "tpot_std":   stdev(r.tpot_ms for r in results) if n_runs > 1 else 0,
    }


def fmt_row(name, s, base_tpot, base_kv, base_fl):
    sp    = base_tpot / s["tpot_ms"] if s["tpot_ms"] > 0 else 0
    kv_rd = (1 - s["kv_mb"] / base_kv) * 100 if base_kv > 0 else 0
    fl_rd = (1 - s["total_gflops"] / base_fl) * 100 if base_fl > 0 else 0
    acc   = f"{s['acc_rate']*100:.1f}%" if s["acc_rate"] is not None else "  ---"
    return (f"{name:<28} {s['ttft_ms']:>7.1f} {s['tpot_ms']:>7.2f} "
            f"{s['throughput']:>7.1f} {s['kv_mb']:>7.2f} "
            f"{sp:>6.2f}x {kv_rd:>5.1f}% "
            f"{s['total_gflops']:>8.3f} {fl_rd:>5.1f}%  {acc}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="EleutherAI/pythia-70m")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    model, tokenizer = load_model(args.model, device=args.device)
    n_lay = num_layers(model)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model}  Layers={n_lay}  Params={n_par/1e6:.1f}M  Device={args.device}")

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []

    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n{'═'*100}")
        print(f"  Context = {ctx_len} tokens  |  Generate {MAX_NEW_TOKENS} tokens")
        print(f"{'═'*100}")

        prompt = load_prompt(ctx_len, tokenizer)
        actual = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        kw  = dict(model=model, tokenizer=tokenizer, prompt=prompt,
                   max_new_tokens=MAX_NEW_TOKENS, device=args.device)
        bkw = dict(n_warmup=N_WARMUP, n_runs=N_RUNS)

        hdr = (f"{'Method':<28} {'TTFT':>7} {'TPOT':>7} {'Tok/s':>7} "
               f"{'KV MB':>7} {'Spd':>6} {'KV↓%':>6} "
               f"{'GFLOPs':>8} {'FL↓%':>5}  {'AccR':>6}")
        sep = "─" * len(hdr)
        print(f"\n{hdr}\n{sep}")

        rows = {}

        # ── Baseline ─────────────────────────────────────────────────────────
        s = bench(greedy_generate, **bkw, **kw)
        rows["Baseline"] = s
        base_tpot, base_kv, base_fl = s["tpot_ms"], s["kv_mb"], s["total_gflops"]
        print(fmt_row("Baseline", s, base_tpot, base_kv, base_fl))
        print(sep)

        # ── StreamingLLM ─────────────────────────────────────────────────────
        for window in [128, 256]:
            name = f"StreamingLLM(W={window})"
            s = bench(streaming_generate, **bkw, **kw, window=window, n_sink=4)
            rows[name] = s
            print(fmt_row(name, s, base_tpot, base_kv, base_fl))
        print(sep)

        # ── CEPE alone ───────────────────────────────────────────────────────
        for name, cfg in CEPE_CONFIGS.items():
            s = bench(cepe_generate, **bkw, **kw, **cfg)
            rows[name] = s
            print(fmt_row(name, s, base_tpot, base_kv, base_fl))
        print(sep)

        # ── PLD alone ────────────────────────────────────────────────────────
        for K in [5, 10]:
            name = f"PLD(K={K})"
            s = bench(pld_generate, **bkw, **kw, K=K)
            rows[name] = s
            print(fmt_row(name, s, base_tpot, base_kv, base_fl))
        print(sep)

        # ── CEPE + Stream + PLD ──────────────────────────────────────────────
        combos = [
            ("CEPE+Str+PLD(r=256,p=4,K=5)",  {"keep_recent": 256, "pool_size": 4, "K": 5}),
            ("CEPE+Str+PLD(r=256,p=4,K=10)", {"keep_recent": 256, "pool_size": 4, "K": 10}),
            ("CEPE+Str+PLD(r=128,p=4,K=5)",  {"keep_recent": 128, "pool_size": 4, "K": 5}),
            ("CEPE+Str+PLD(r=128,p=4,K=10)", {"keep_recent": 128, "pool_size": 4, "K": 10}),
            ("CEPE+Str+PLD(r=256,p=8,K=5)",  {"keep_recent": 256, "pool_size": 8, "K": 5}),
            ("CEPE+Str+PLD(r=128,p=8,K=10)", {"keep_recent": 128, "pool_size": 8, "K": 10}),
        ]
        for name, extra in combos:
            s = bench(cepe_stream_pld_generate, **bkw, **kw, **extra)
            rows[name] = s
            print(fmt_row(name, s, base_tpot, base_kv, base_fl))
        print(sep)

        for method, s in rows.items():
            sp    = base_tpot / s["tpot_ms"] if s["tpot_ms"] > 0 else 0
            kv_rd = (1 - s["kv_mb"] / base_kv) * 100 if base_kv > 0 else 0
            fl_rd = (1 - s["total_gflops"] / base_fl) * 100 if base_fl > 0 else 0
            all_rows.append({
                "context_length": ctx_len, "actual_ctx": actual,
                "method": method, **s,
                "speedup": round(sp, 4),
                "kv_reduction_%": round(kv_rd, 2),
                "flops_reduction_%": round(fl_rd, 2),
            })

    out_path = Path(args.out_dir) / "cepe_combined_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n[saved] {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print("  Summary  (ctx=1024, relative to Baseline)")
    print(f"{'═'*72}")
    ctx_rows = {r["method"]: r for r in all_rows if r["context_length"] == 1024}
    if not ctx_rows:
        ctx_rows = {r["method"]: r for r in all_rows
                    if r["context_length"] == max(r2["context_length"] for r2 in all_rows)}
    bt = ctx_rows.get("Baseline", {}).get("tpot_ms", 1)
    bk = ctx_rows.get("Baseline", {}).get("kv_mb",   1)
    print(f"  {'Method':<28} {'TPOT Spd':>9} {'KV↓%':>7} {'AccR':>7}")
    print(f"  {'─'*56}")
    for name, r in ctx_rows.items():
        sp    = bt / r["tpot_ms"] if r["tpot_ms"] > 0 else 0
        kv_rd = (1 - r["kv_mb"] / bk) * 100 if bk > 0 else 0
        acc   = f"{r['acc_rate']*100:.1f}%" if r["acc_rate"] else "---"
        print(f"  {name:<28} {sp:>8.2f}x {kv_rd:>6.1f}% {acc:>7}")


if __name__ == "__main__":
    main()

"""
Cross-Layer KV Eviction (CLE) — Full Evaluation.

Steps
-----
  1. Performance benchmark  (TTFT / TPOT / Throughput / FLOPs)
  2. PPL evaluation         (WikiText-2 + PG-19)
  3. Context-length sweep
  4. Figure generation

Usage
-----
  python scripts/run_all.py
  python scripts/run_all.py --device cuda --n-runs 5
  python scripts/run_all.py --skip-ppl --skip-sweep
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import load_model, set_seed, GenerationResult, num_layers
from baseline import greedy_generate
from cross_layer_evict import cle_generate, estimate_flops_cle, CLE_CONFIGS
from eval_ppl import load_wikitext, load_pg19, compute_ppl_sliding, make_transforms

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   Cross-Layer KV Eviction — Full Evaluation                  ║
║   Pythia-70M  |  WikiText-2 + PG-19  |  Training-free        ║
╚══════════════════════════════════════════════════════════════╝
"""

BENCH_PROMPT = (
    "The transformer architecture revolutionized natural language processing. "
    "The key-value cache is central to efficient autoregressive decoding: "
    "it stores previously computed key and value tensors so that each new "
    "token only needs one forward pass through the full model. However, as "
    "the context window grows, the KV cache becomes the primary memory "
    "bottleneck. Cross-Layer KV Eviction addresses this by identifying which "
    "tokens are collectively important across all transformer layers and "
    "retaining only those in the cache. Unlike methods that share KV tensors "
    "across layers, this approach preserves each layer's own projections, "
    "avoiding quality degradation while still reducing memory and compute."
)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",          default="EleutherAI/pythia-70m")
    p.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-new-tokens", type=int,   default=128)
    p.add_argument("--n-runs",         type=int,   default=3)
    p.add_argument("--n-warmup",       type=int,   default=1)
    p.add_argument("--ppl-max-tokens", type=int,   default=4096)
    p.add_argument("--ppl-window",     type=int,   default=512)
    p.add_argument("--ppl-stride",     type=int,   default=256)
    p.add_argument("--skip-ppl",       action="store_true")
    p.add_argument("--skip-sweep",     action="store_true")
    p.add_argument("--skip-plots",     action="store_true")
    p.add_argument("--out-dir",        default="results")
    return p.parse_args()


# ── Benchmarking ──────────────────────────────────────────────────────────────

def bench_method(name, fn, n_warmup, n_runs, **kwargs) -> Dict:
    results: List[GenerationResult] = []
    for i in range(n_warmup + n_runs):
        set_seed(42)
        r = fn(**kwargs)
        if i >= n_warmup:
            results.append(r)
    return {
        "method":             name,
        "num_generated":      results[0].num_generated,
        "ttft_ms":            mean(r.ttft_ms    for r in results),
        "tpot_ms":            mean(r.tpot_ms    for r in results),
        "throughput":         mean(r.throughput for r in results),
        "kv_mb":              mean(r.kv_size_mb for r in results),
        "total_gflops":       results[0].total_gflops,
        "avg_gflops_per_tok": results[0].avg_gflops_per_tok,
        "ttft_std":           stdev(r.ttft_ms    for r in results) if n_runs > 1 else 0,
        "tput_std":           stdev(r.throughput for r in results) if n_runs > 1 else 0,
    }


def print_perf_table(rows: List[Dict]):
    base = rows[0]
    hdr  = (f"{'Method':<26} {'Tok':>5} {'TTFT':>7} {'TPOT':>7} "
            f"{'Tok/s':>7} {'KV MB':>7} {'Spd':>6} {'KV↓%':>6} "
            f"{'GFLOPs':>9} {'FL↓%':>6} {'GF/tok':>8}")
    sep  = "─" * len(hdr)
    print(f"\n{sep}\n{hdr}\n{sep}")
    for r in rows:
        sp    = base["tpot_ms"] / r["tpot_ms"] if r["tpot_ms"] > 0 else 0
        kv_rd = (1 - r["kv_mb"]         / base["kv_mb"])         * 100 if base["kv_mb"]         > 0 else 0
        fl_rd = (1 - r["total_gflops"]  / base["total_gflops"])  * 100 if base["total_gflops"]  > 0 else 0
        print(f"{r['method']:<26} {r['num_generated']:>5} "
              f"{r['ttft_ms']:>7.1f} {r['tpot_ms']:>7.2f} "
              f"{r['throughput']:>7.1f} {r['kv_mb']:>7.3f} "
              f"{sp:>6.2f}x {kv_rd:>5.1f}% "
              f"{r['total_gflops']:>9.3f} {fl_rd:>5.1f}% "
              f"{r['avg_gflops_per_tok']:>8.5f}")
    print(sep)


# ── PPL evaluation ────────────────────────────────────────────────────────────

def run_ppl(model, tokenizer, n_lay, text, dataset_name, args) -> List[Dict]:
    transforms = make_transforms(n_lay)
    results    = []
    baseline_ppl = None
    for method, tf in transforms.items():
        ppl = compute_ppl_sliding(
            model, tokenizer, text,
            window=args.ppl_window, stride=args.ppl_stride,
            max_tokens=args.ppl_max_tokens,
            device=args.device,
            kv_transform=tf,
        )
        if baseline_ppl is None:
            baseline_ppl = ppl
        delta = ppl - baseline_ppl
        print(f"  {method:<36} PPL={ppl:>8.2f}   Δ={delta:>+8.2f}")
        results.append({"method": method, "ppl": round(ppl, 4)})
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print(BANNER)
    print(f"Model : {args.model}")
    print(f"Device: {args.device}")

    model, tokenizer = load_model(args.model, device=args.device)
    n_lay   = num_layers(model)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Layers: {n_lay}   Params: {n_param / 1e6:.1f}M\n")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Performance benchmark ──────────────────────────────────────────────
    print("═" * 64)
    print("  1 / 3   Performance Benchmark (fixed prompt, 128 new tokens)")
    print("═" * 64)

    shared_kw = dict(
        model=model, tokenizer=tokenizer,
        prompt=BENCH_PROMPT,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    run_kw = dict(n_warmup=args.n_warmup, n_runs=args.n_runs)

    method_list = [("Baseline", greedy_generate, {})]
    for cfg_name, ratio in CLE_CONFIGS.items():
        method_list.append((cfg_name, cle_generate, {"budget_ratio": ratio}))

    perf_rows: List[Dict] = []
    for name, fn, extra in method_list:
        print(f"  {name} …", end=" ", flush=True)
        r = bench_method(name, fn, **run_kw, **shared_kw, **extra)
        print(f"TPOT={r['tpot_ms']:.2f} ms  KV={r['kv_mb']:.3f} MB  "
              f"GFLOPs={r['total_gflops']:.3f}")
        perf_rows.append(r)

    print_perf_table(perf_rows)

    # ── 2. PPL evaluation ─────────────────────────────────────────────────────
    ppl_wt2:  List[Dict] = []
    ppl_pg19: List[Dict] = []

    if not args.skip_ppl:
        print("\n" + "═" * 64)
        print("  2 / 3   Perplexity Evaluation")
        print("═" * 64)

        print(f"\n  ── WikiText-2 (≤{args.ppl_max_tokens} tokens,"
              f" window={args.ppl_window}, stride={args.ppl_stride}) ──")
        wt2_text = load_wikitext("test")
        ppl_wt2  = run_ppl(model, tokenizer, n_lay, wt2_text, "WikiText-2", args)

        print(f"\n  ── PG-19 (≤{args.ppl_max_tokens} tokens) ──")
        try:
            pg19_text = load_pg19()
            ppl_pg19  = run_ppl(model, tokenizer, n_lay, pg19_text, "PG-19", args)
        except Exception as e:
            print(f"  PG-19 load failed ({e}); skipping.")

    # ── 3. Context-length sweep ───────────────────────────────────────────────
    if not args.skip_sweep:
        print("\n" + "═" * 64)
        print("  3 / 3   Context-Length Sweep")
        print("═" * 64)
        sweep_script = Path(__file__).parent / "seq_len_sweep.py"
        cmd = [sys.executable, str(sweep_script),
               "--model",    args.model,
               "--device",   args.device,
               "--n-runs",   str(args.n_runs),
               "--n-warmup", str(args.n_warmup),
               "--out-dir",  args.out_dir]
        subprocess.run(cmd, check=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "model":          args.model,
        "max_new_tokens": args.max_new_tokens,
        "performance":    perf_rows,
        "ppl_wikitext2":  ppl_wt2,
        "ppl_pg19":       ppl_pg19,
    }
    out_path = Path(args.out_dir) / "results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_path}")

    # ── Figures ───────────────────────────────────────────────────────────────
    if not args.skip_plots:
        sweep_json = Path(args.out_dir) / "seq_len_sweep.json"
        if sweep_json.exists():
            print("\nGenerating figures …")
            plot_script = Path(__file__).parent / "plot_results.py"
            subprocess.run([sys.executable, str(plot_script),
                            "--out-dir", args.out_dir], check=True)

    # ── Summaries ─────────────────────────────────────────────────────────────
    for ds_name, rows in [("WikiText-2", ppl_wt2), ("PG-19", ppl_pg19)]:
        if not rows:
            continue
        base = rows[0]["ppl"]
        print(f"\n{'═'*52}")
        print(f"  PPL Summary — {ds_name}")
        print(f"{'═'*52}")
        print(f"  {'Method':<36} {'PPL':>8} {'ΔPPL':>8}")
        print(f"  {'─'*54}")
        for r in rows:
            print(f"  {r['method']:<36} {r['ppl']:>8.2f} {r['ppl']-base:>+8.2f}")

    if perf_rows:
        base_fl = perf_rows[0]["total_gflops"]
        print(f"\n{'═'*52}")
        print(f"  FLOPs Summary")
        print(f"{'═'*52}")
        print(f"  {'Method':<26} {'GFLOPs':>9} {'FLOPs↓%':>8} {'GF/tok':>9}")
        print(f"  {'─'*54}")
        for r in perf_rows:
            fl_rd = (1 - r["total_gflops"] / base_fl) * 100 if base_fl > 0 else 0
            print(f"  {r['method']:<26} {r['total_gflops']:>9.3f} "
                  f"{fl_rd:>7.1f}% {r['avg_gflops_per_tok']:>9.5f}")


if __name__ == "__main__":
    main()

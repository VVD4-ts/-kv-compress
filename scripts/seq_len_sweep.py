"""
Context-length sweep: Baseline vs CLE variants at 128/256/512/1024 tokens.
Reports TTFT, TPOT, Throughput, KV MB, Total GFLOPs, Avg GFLOPs/token.
"""

import json
import os
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import load_model, set_seed, GenerationResult, num_layers
from baseline import greedy_generate
from cross_layer_evict import cle_generate, CLE_CONFIGS

CONTEXT_LENGTHS = [128, 256, 512, 1024]
MAX_NEW_TOKENS  = 100
N_WARMUP        = 1
N_RUNS          = 3


def load_test_text() -> str:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(r["text"] for r in ds if r["text"].strip())


def truncate_to_tokens(text: str, n_tokens: int, tokenizer) -> str:
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if len(ids) < n_tokens:
        raise ValueError(f"Only {len(ids)} tokens, need {n_tokens}")
    return tokenizer.decode(ids[:n_tokens], skip_special_tokens=True)


def bench_one(fn, n_warmup: int, n_runs: int, **kwargs) -> Dict:
    results: List[GenerationResult] = []
    for i in range(n_warmup + n_runs):
        set_seed(42)
        r = fn(**kwargs)
        if i >= n_warmup:
            results.append(r)
    return {
        "ttft_ms":            mean(r.ttft_ms    for r in results),
        "tpot_ms":            mean(r.tpot_ms    for r in results),
        "throughput":         mean(r.throughput for r in results),
        "kv_mb":              mean(r.kv_size_mb for r in results),
        "total_gflops":       results[0].total_gflops,
        "avg_gflops_per_tok": results[0].avg_gflops_per_tok,
        "num_gen":            results[0].num_generated,
        "ttft_std":           stdev(r.ttft_ms    for r in results) if n_runs > 1 else 0,
        "tput_std":           stdev(r.throughput for r in results) if n_runs > 1 else 0,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="EleutherAI/pythia-70m")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-runs",   type=int, default=N_RUNS)
    p.add_argument("--n-warmup", type=int, default=N_WARMUP)
    p.add_argument("--out-dir",  default="results")
    args = p.parse_args()

    print(f"Model: {args.model}   Device: {args.device}")
    model, tokenizer = load_model(args.model, device=args.device)

    print("Loading WikiText-2 test …")
    test_text    = load_test_text()
    total_tokens = tokenizer(test_text, return_tensors="pt").input_ids.shape[1]
    print(f"  Total tokens: {total_tokens}")

    method_registry = [("Baseline", greedy_generate, {})]
    for cfg_name, ratio in CLE_CONFIGS.items():
        method_registry.append((cfg_name, cle_generate, {"budget_ratio": ratio}))

    all_rows: List[Dict] = []
    run_kw = {"n_warmup": args.n_warmup, "n_runs": args.n_runs}

    for ctx_len in CONTEXT_LENGTHS:
        if ctx_len >= total_tokens:
            continue
        try:
            prompt = truncate_to_tokens(test_text, ctx_len, tokenizer)
        except ValueError as e:
            print(f"  Skipping ctx={ctx_len}: {e}")
            continue

        actual_len = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        print(f"\n── ctx_len={ctx_len} (actual={actual_len}) ──────────────")
        print(f"{'Method':<26} {'TTFT':>7} {'TPOT':>7} {'Tok/s':>7} "
              f"{'KV MB':>7} {'GFLOPs':>9} {'GF/tok':>8} {'FL↓%':>6}")
        print("─" * 84)

        baseline_tput = None
        baseline_kv   = None
        baseline_fl   = None

        for name, fn, extra in method_registry:
            stats = bench_one(
                fn, prompt=prompt,
                model=model, tokenizer=tokenizer,
                max_new_tokens=MAX_NEW_TOKENS,
                device=args.device,
                **run_kw, **extra,
            )
            if baseline_tput is None:
                baseline_tput = stats["throughput"]
                baseline_kv   = stats["kv_mb"]
                baseline_fl   = stats["total_gflops"]

            speedup = stats["throughput"] / baseline_tput if baseline_tput else 0
            kv_red  = (1 - stats["kv_mb"]        / baseline_kv) * 100 if baseline_kv else 0
            fl_red  = (1 - stats["total_gflops"] / baseline_fl) * 100 if baseline_fl else 0

            print(f"{name:<26} {stats['ttft_ms']:>7.1f} {stats['tpot_ms']:>7.2f} "
                  f"{stats['throughput']:>7.1f} {stats['kv_mb']:>7.3f} "
                  f"{stats['total_gflops']:>9.3f} {stats['avg_gflops_per_tok']:>8.5f} "
                  f"{fl_red:>5.1f}%")

            all_rows.append({
                "context_length":     ctx_len,
                "method":             name,
                **stats,
                "speedup":            round(speedup, 4),
                "kv_reduction_%":     round(kv_red, 2),
                "flops_reduction_%":  round(fl_red, 2),
            })

        print("─" * 84)

    os.makedirs(args.out_dir, exist_ok=True)

    json_path = Path(args.out_dir) / "seq_len_sweep.json"
    json_path.write_text(json.dumps(all_rows, indent=2))
    print(f"\n[saved] {json_path}")

    import csv
    csv_path = Path(args.out_dir) / "seq_len_sweep.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()

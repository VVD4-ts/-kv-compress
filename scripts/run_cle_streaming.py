"""
Benchmark: CLE + StreamingLLM (no PLD).

Compares baseline, CLE-only, StreamingLLM-only, and CLE+StreamingLLM
across multiple context lengths and compression configs.
"""

import sys, os, json
from pathlib import Path
from statistics import mean, stdev

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import load_model, set_seed, num_layers
from baseline import greedy_generate
from cross_layer_evict import cle_generate, CLE_CONFIGS
from streaming_llm import streaming_generate
from combined import cle_streaming_generate

CONTEXT_LENGTHS = [512, 1024, 2048]
MAX_NEW_TOKENS  = 200
N_WARMUP = 1
N_RUNS   = 3


def load_prompt(n_tokens, tokenizer):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
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
    return {
        "ttft_ms":    mean(r.ttft_ms    for r in results),
        "tpot_ms":    mean(r.tpot_ms    for r in results),
        "throughput": mean(r.throughput  for r in results),
        "kv_mb":      mean(r.kv_size_mb for r in results),
        "total_gflops":       results[0].total_gflops,
        "avg_gflops_per_tok": results[0].avg_gflops_per_tok,
        "num_gen":    results[0].num_generated,
        "tpot_std":   stdev(r.tpot_ms for r in results) if n_runs > 1 else 0,
    }


def fmt_row(name, s, base_tpot, base_kv, base_fl):
    sp    = base_tpot / s["tpot_ms"] if s["tpot_ms"] > 0 else 0
    kv_rd = (1 - s["kv_mb"] / base_kv) * 100 if base_kv > 0 else 0
    fl_rd = (1 - s["total_gflops"] / base_fl) * 100 if base_fl > 0 else 0
    return (f"{name:<30} {s['ttft_ms']:>7.1f} {s['tpot_ms']:>7.2f} "
            f"{s['throughput']:>7.1f} {s['kv_mb']:>7.2f} "
            f"{sp:>6.2f}x {kv_rd:>5.1f}% "
            f"{s['total_gflops']:>8.3f} {fl_rd:>5.1f}%")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="EleutherAI/pythia-70m")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    model, tokenizer = load_model(args.model, device=args.device)
    n_lay = num_layers(model)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {args.model}  Layers={n_lay}  Params={n_par/1e6:.1f}M  Device={args.device}")

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []

    for ctx_len in CONTEXT_LENGTHS:
        print(f"\n{'═'*95}")
        print(f"  Context length = {ctx_len} tokens  |  Generating {MAX_NEW_TOKENS} tokens")
        print(f"{'═'*95}")

        prompt = load_prompt(ctx_len, tokenizer)
        actual = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
        kw = dict(model=model, tokenizer=tokenizer, prompt=prompt,
                  max_new_tokens=MAX_NEW_TOKENS, device=args.device)
        bkw = dict(n_warmup=N_WARMUP, n_runs=N_RUNS)

        hdr = (f"{'Method':<30} {'TTFT':>7} {'TPOT':>7} {'Tok/s':>7} "
               f"{'KV MB':>7} {'Spd':>6} {'KV↓%':>6} "
               f"{'GFLOPs':>8} {'FL↓%':>5}")
        sep = "─" * len(hdr)
        print(f"\n{hdr}\n{sep}")

        rows = {}

        # Baseline
        s = bench(greedy_generate, **bkw, **kw)
        rows["Baseline"] = s
        base_tpot, base_kv, base_fl = s["tpot_ms"], s["kv_mb"], s["total_gflops"]
        print(fmt_row("Baseline", s, base_tpot, base_kv, base_fl))
        print(sep)

        # CLE only
        for name, ratio in CLE_CONFIGS.items():
            s = bench(cle_generate, **bkw, **kw, budget_ratio=ratio)
            rows[name] = s
            print(fmt_row(name, s, base_tpot, base_kv, base_fl))
        print(sep)

        # StreamingLLM only
        for window in [128, 256]:
            name = f"StreamingLLM(W={window})"
            s = bench(streaming_generate, **bkw, **kw, window=window, n_sink=4)
            rows[name] = s
            print(fmt_row(name, s, base_tpot, base_kv, base_fl))
        print(sep)

        # CLE + StreamingLLM
        combos = [
            ("CLE50%+Stream(dw=128)", {"budget_ratio": 0.50, "decode_window": 128, "n_sink": 4}),
            ("CLE50%+Stream(dw=64)",  {"budget_ratio": 0.50, "decode_window": 64,  "n_sink": 4}),
            ("CLE50%+Stream(dw=32)",  {"budget_ratio": 0.50, "decode_window": 32,  "n_sink": 4}),
            ("CLE30%+Stream(dw=128)", {"budget_ratio": 0.30, "decode_window": 128, "n_sink": 4}),
            ("CLE30%+Stream(dw=64)",  {"budget_ratio": 0.30, "decode_window": 64,  "n_sink": 4}),
            ("CLE30%+Stream(dw=32)",  {"budget_ratio": 0.30, "decode_window": 32,  "n_sink": 4}),
            ("CLE80%+Stream(dw=64)",  {"budget_ratio": 0.80, "decode_window": 64,  "n_sink": 4}),
        ]
        for name, extra in combos:
            s = bench(cle_streaming_generate, **bkw, **kw, **extra)
            rows[name] = s
            print(fmt_row(name, s, base_tpot, base_kv, base_fl))
        print(sep)

        for method, s in rows.items():
            sp    = base_tpot / s["tpot_ms"] if s["tpot_ms"] > 0 else 0
            kv_rd = (1 - s["kv_mb"] / base_kv) * 100 if base_kv > 0 else 0
            fl_rd = (1 - s["total_gflops"] / base_fl) * 100 if base_fl > 0 else 0
            all_rows.append({
                "context_length": ctx_len,
                "actual_ctx":     actual,
                "method":         method,
                **s,
                "speedup":        round(sp, 4),
                "kv_reduction_%": round(kv_rd, 2),
                "flops_reduction_%": round(fl_rd, 2),
            })

    out_path = Path(args.out_dir) / "cle_streaming_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()

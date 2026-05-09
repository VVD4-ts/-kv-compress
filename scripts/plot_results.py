"""
Generate publication-quality figures for SimLayerKV KV cache compression.

Figures
-------
  fig1_ppl_wt2        PPL bar chart — WikiText-2
  fig1_ppl_pg19       PPL bar chart — PG-19
  fig2_tradeoff_*     PPL vs KV-reduction% scatter
  fig3_kv_vs_seqlen   KV cache size vs context length
  fig4_tput_vs_seqlen Throughput vs context length
  fig5_speedup        Throughput speedup vs context length
  fig6_flops          Theoretical GFLOPs vs context length
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Colour / style ────────────────────────────────────────────────────────────

# SimLayerKV: green shades (light→heavy)
# YOCO:       blue shades
# Baseline:   dark grey

_PALETTE = {
    "Baseline":   "#333333",
    "CLE-Light":  "#52C41A",   # green light
    "CLE-Medium": "#237804",   # green medium
    "CLE-Heavy":  "#092B00",   # green dark
}

_MARKERS = {
    "Baseline":   "o",
    "CLE-Light":  "s",
    "CLE-Medium": "^",
    "CLE-Heavy":  "D",
}


def _method_style(name: str):
    """Return (color, marker, linestyle) for a method name."""
    nl = name.lower()
    if name == "Baseline":
        return _PALETTE["Baseline"], _MARKERS["Baseline"], "--"
    if "light" in nl:
        return _PALETTE["CLE-Light"],  _MARKERS["CLE-Light"],  "-"
    if "medium" in nl:
        return _PALETTE["CLE-Medium"], _MARKERS["CLE-Medium"], "-"
    if "heavy" in nl:
        return _PALETTE["CLE-Heavy"],  _MARKERS["CLE-Heavy"],  "-"
    # fallback
    import hashlib
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    colors = ["#FA541C", "#722ED1", "#13C2C2", "#EB2F96"]
    return colors[h % len(colors)], "x", "-"


LINEWIDTH  = 1.8
MARKERSIZE = 7
FONTSIZE   = 11

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       FONTSIZE,
    "axes.titlesize":  FONTSIZE,
    "axes.labelsize":  FONTSIZE,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":      150,
    "savefig.bbox":    "tight",
    "savefig.dpi":     300,
})


def _save(fig, out_dir: Path, name: str):
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        print(f"  [saved] {p}")
    plt.close(fig)


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_sweep(path: Path) -> List[Dict]:
    return json.loads(path.read_text())


def group_by_method(rows: List[Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for r in rows:
        m = r["method"]
        if m not in out:
            out[m] = {k: [] for k in
                      ("ctx", "kv_mb", "throughput", "speedup",
                       "ttft_ms", "tpot_ms", "total_gflops", "avg_gflops_per_tok")}
        out[m]["ctx"].append(r["context_length"])
        for k in ("kv_mb", "throughput", "speedup", "ttft_ms", "tpot_ms",
                  "total_gflops", "avg_gflops_per_tok"):
            out[m][k].append(r.get(k, 0))
    return out


# ── Figure 1 — PPL bar chart ──────────────────────────────────────────────────

def plot_ppl_bars(ppl_data: List[Dict], out_dir: Path, dataset_name: str):
    if not ppl_data:
        return
    methods = [r["method"] for r in ppl_data]
    ppls    = [r["ppl"]    for r in ppl_data]

    fig, ax = plt.subplots(figsize=(max(7, len(methods) * 1.2), 4.5))
    colors  = [_method_style(m)[0] for m in methods]
    bars    = ax.bar(range(len(methods)), ppls, color=colors,
                     edgecolor="white", linewidth=0.8, width=0.65)

    baseline_ppl = ppl_data[0]["ppl"]
    ax.axhline(baseline_ppl, color="#333333", linestyle="--",
               linewidth=1.2, alpha=0.7, label=f"Baseline ({baseline_ppl:.1f})")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Perplexity  (↓ better)")
    ax.set_title(f"Cross-Layer KV Eviction — Perplexity on {dataset_name}\n(Pythia-70M, training-free)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{ppl:.1f}", ha="center", va="bottom", fontsize=8)

    suffix = dataset_name.lower().replace("-", "").replace(" ", "_")
    _save(fig, out_dir, f"fig1_ppl_{suffix}")


# ── Figure 2 — PPL vs KV-reduction tradeoff ──────────────────────────────────

def plot_tradeoff(ppl_data: List[Dict], sweep_rows: List[Dict],
                  out_dir: Path, dataset_name: str):
    if not ppl_data or not sweep_rows:
        return
    ctx_target  = max(r["context_length"] for r in sweep_rows)
    baseline_kv = None
    kv_at_ctx   = {}
    for r in sweep_rows:
        if r["context_length"] == ctx_target:
            if r["method"] == "Baseline":
                baseline_kv = r["kv_mb"]
            kv_at_ctx[r["method"]] = r["kv_mb"]

    ppl_map = {r["method"]: r["ppl"] for r in ppl_data}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for name, ppl in ppl_map.items():
        kv = kv_at_ctx.get(name)
        kv_red = (1 - kv / baseline_kv) * 100 if (kv and baseline_kv) else 0.0
        color, marker, _ = _method_style(name)
        ax.scatter(kv_red, ppl, color=color, marker=marker, s=120, zorder=3, label=name)
        ax.annotate(name, (kv_red, ppl),
                    textcoords="offset points", xytext=(6, 3),
                    fontsize=7.5, color=color)

    ax.set_xlabel("KV Cache Reduction (%)")
    ax.set_ylabel("Perplexity  (↓ better)")
    ax.set_title(f"Quality–Memory Tradeoff — {dataset_name}\n"
                 f"(Pythia-70M, ctx={ctx_target})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    suffix = dataset_name.lower().replace("-", "").replace(" ", "_")
    _save(fig, out_dir, f"fig2_tradeoff_{suffix}")


# ── Figure 3 — KV size vs context length ─────────────────────────────────────

def plot_kv_vs_seqlen(by_method: Dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, data in by_method.items():
        c, m, ls = _method_style(name)
        ax.plot(data["ctx"], data["kv_mb"],
                color=c, marker=m, linestyle=ls,
                linewidth=LINEWIDTH, markersize=MARKERSIZE, label=name)
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("KV Cache Size (MB)")
    ax.set_title("KV Cache Memory vs. Context Length\n(Pythia-70M)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out_dir, "fig3_kv_vs_seqlen")


# ── Figure 4 — Throughput vs context length ───────────────────────────────────

def plot_tput_vs_seqlen(by_method: Dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, data in by_method.items():
        c, m, ls = _method_style(name)
        ax.plot(data["ctx"], data["throughput"],
                color=c, marker=m, linestyle=ls,
                linewidth=LINEWIDTH, markersize=MARKERSIZE, label=name)
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Throughput (tokens / sec)")
    ax.set_title("Generation Throughput vs. Context Length\n(Pythia-70M)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out_dir, "fig4_tput_vs_seqlen")


# ── Figure 5 — Speedup vs context length ─────────────────────────────────────

def plot_speedup_vs_seqlen(by_method: Dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, data in by_method.items():
        if name == "Baseline":
            continue
        c, m, ls = _method_style(name)
        ax.plot(data["ctx"], data["speedup"],
                color=c, marker=m, linestyle=ls,
                linewidth=LINEWIDTH, markersize=MARKERSIZE, label=name)
    ax.axhline(1.0, color="#333333", linestyle="--",
               linewidth=1.2, label="Baseline (1.0×)")
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Throughput Speedup (×)")
    ax.set_title("Throughput Speedup vs. Context Length\n(Pythia-70M)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out_dir, "fig5_speedup_vs_seqlen")


# ── Figure 6 — Theoretical FLOPs vs context length ───────────────────────────

def plot_flops_vs_seqlen(by_method: Dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, data in by_method.items():
        if not data.get("total_gflops") or all(v == 0 for v in data["total_gflops"]):
            continue
        c, m, ls = _method_style(name)
        ax.plot(data["ctx"], data["total_gflops"],
                color=c, marker=m, linestyle=ls,
                linewidth=LINEWIDTH, markersize=MARKERSIZE, label=name)
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Theoretical FLOPs (GFLOPs)")
    ax.set_title("Theoretical FLOPs vs. Context Length\n(Pythia-70M, analytical)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out_dir, "fig6_flops_vs_seqlen")


# ── Figure 7 — Avg GFLOPs/token vs context length ────────────────────────────

def plot_flops_per_tok(by_method: Dict, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, data in by_method.items():
        if not data.get("avg_gflops_per_tok") or all(v == 0 for v in data["avg_gflops_per_tok"]):
            continue
        c, m, ls = _method_style(name)
        ax.plot(data["ctx"], data["avg_gflops_per_tok"],
                color=c, marker=m, linestyle=ls,
                linewidth=LINEWIDTH, markersize=MARKERSIZE, label=name)
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Avg FLOPs per Token (GFLOPs)")
    ax.set_title("Average FLOPs per Token vs. Context Length\n(Pythia-70M, analytical)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
    _save(fig, out_dir, "fig7_flops_per_tok")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sweep_path   = out_dir / "seq_len_sweep.json"
    results_path = out_dir / "results.json"

    if not sweep_path.exists():
        print(f"No sweep data at {sweep_path}. Run seq_len_sweep.py first.")
        sys.exit(1)

    sweep_rows = load_sweep(sweep_path)
    by_method  = group_by_method(sweep_rows)

    print("Generating figures …")

    plot_kv_vs_seqlen(by_method, fig_dir)
    plot_tput_vs_seqlen(by_method, fig_dir)
    plot_speedup_vs_seqlen(by_method, fig_dir)
    plot_flops_vs_seqlen(by_method, fig_dir)
    plot_flops_per_tok(by_method, fig_dir)

    if results_path.exists():
        results  = json.loads(results_path.read_text())
        ppl_wt2  = results.get("ppl_wikitext2", [])
        ppl_pg19 = results.get("ppl_pg19", [])
        if ppl_wt2:
            plot_ppl_bars(ppl_wt2,  fig_dir, "WikiText-2")
            plot_tradeoff(ppl_wt2,  sweep_rows, fig_dir, "WikiText-2")
        if ppl_pg19:
            plot_ppl_bars(ppl_pg19, fig_dir, "PG-19")
            plot_tradeoff(ppl_pg19, sweep_rows, fig_dir, "PG-19")

    print(f"\nAll figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()

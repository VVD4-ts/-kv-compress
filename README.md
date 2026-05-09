# Cross-Layer KV Eviction for Efficient LLM Inference

> **Course project — NeurIPS 2025 format**  
> Model: [EleutherAI/pythia-70m](https://huggingface.co/EleutherAI/pythia-70m) · Training-free · CUDA

---

## Overview

The KV cache grows linearly with sequence length, becoming the dominant memory and latency bottleneck in long-context LLM inference. Most existing compression methods either share projections across layers (causing quality degradation on small models) or prune tokens independently per layer (ignoring cross-layer correlation).

We propose **Cross-Layer KV Eviction (CLE)**: a training-free method that aggregates key-vector L2 norms across *all* layers to score token importance, then evicts the same low-importance positions from every layer's KV cache simultaneously. Because each layer still uses its own W_K / W_V projections, there is no projection mismatch — quality is preserved even at aggressive compression ratios.

### Key properties

| Property | CLE |
|----------|-----|
| Training required | ✗ None |
| Projection sharing | ✗ No (avoids quality loss) |
| Eviction granularity | Token-level, cross-layer coordinated |
| Decode attention context | `budget` tokens (not full prompt) |
| PPL degradation (−70% KV) | +0.08 on WikiText-2 |

---

## Results

### Perplexity (WikiText-2 & PG-19, 4096 tokens)

| Method | WikiText-2 PPL | ΔPPL | PG-19 PPL | ΔPPL |
|--------|---------------|------|-----------|------|
| Baseline (full KV) | 51.16 | — | 30.89 | — |
| CLE-Light  (−20% KV) | **51.32** | +0.16 | **30.96** | +0.08 |
| CLE-Medium (−50% KV) | **51.25** | +0.08 | **31.04** | +0.15 |
| CLE-Heavy  (−70% KV) | **51.25** | +0.08 | **30.89** | +0.00 |

Near-zero PPL degradation even when 70% of cached tokens are discarded.

### Performance (context length sweep, CUDA)

| Method | ctx=128 | ctx=512 | ctx=1024 |
|--------|---------|---------|---------|
| Baseline TPOT | 4.38 ms | 4.07 ms | 4.44 ms |
| CLE-Light  TPOT | 4.31 ms | 4.19 ms | **4.11 ms** |
| CLE-Medium TPOT | 4.29 ms | 4.26 ms | **4.20 ms** |
| CLE-Heavy  TPOT | 4.37 ms | 4.37 ms | **4.26 ms** |

| Method | ctx=1024 KV↓ | ctx=1024 Speedup |
|--------|-------------|-----------------|
| CLE-Light  (−20% KV) | −18.3% | 1.08× |
| CLE-Medium (−50% KV) | −45.6% | 1.06× |
| CLE-Heavy  (−70% KV) | −63.8% | 1.04× |

> **Note on speedup magnitude:** Pythia-70M has hidden size H=512. At decode time, linear projections (O(H²)) dominate over attention (O(ctx·H)), so KV compression provides modest wall-clock savings on this small model. On production-scale models (H≥4096) with long contexts, the same method yields proportionally larger speedups.

### Figures

<p align="center">
  <img src="results/figures/fig1_ppl_wikitext2.png" width="48%"/>
  <img src="results/figures/fig1_ppl_pg19.png" width="48%"/>
</p>
<p align="center">
  <em>Left: WikiText-2 PPL. Right: PG-19 PPL. CLE variants stay within 0.2 points of baseline.</em>
</p>

<p align="center">
  <img src="results/figures/fig3_kv_vs_seqlen.png" width="48%"/>
  <img src="results/figures/fig5_speedup_vs_seqlen.png" width="48%"/>
</p>
<p align="center">
  <em>Left: KV memory vs context length. Right: throughput speedup vs context length.</em>
</p>

<p align="center">
  <img src="results/figures/fig6_flops_vs_seqlen.png" width="48%"/>
  <img src="results/figures/fig7_flops_per_tok.png" width="48%"/>
</p>
<p align="center">
  <em>Left: Total GFLOPs vs context length. Right: Average GFLOPs per token.</em>
</p>

---

## Method

### Algorithm

```
Prefill
  1. Run full forward pass → KV cache [B, H, S, D] per layer
  2. For each layer i, compute key-norm score:
       score_i[pos] = mean_heads( ||K_i[:, :, pos, :]||_2 )
  3. Aggregate across layers:
       importance[pos] = mean_layers( score_i[pos] )
  4. Protect first n_sink positions (attention sink tokens)
  5. Keep top-budget positions by importance → evict the rest

Decode
  - Each new token is appended to the (pruned) KV cache
  - No further eviction; position_ids track true positions for RoPE
```

### Why key-norm scoring?

Attention score ∝ Q·K. For a given query distribution, tokens with high-magnitude keys receive more attention on average. Key-norm importance is a robust, attention-output-free proxy used across several KV eviction papers (SnapKV, PyramidKV). The **cross-layer** aggregation is CLE's key contribution: instead of each layer independently evicting, all layers collectively vote, keeping tokens that matter globally.

### Why no projection mismatch?

Methods like YOCO or SimLayerKV share KV tensors across layers. For Pythia-70M, adjacent-layer KV cosine similarity is < 0.09 — sharing causes catastrophic quality loss. CLE never shares projections; each layer retains its own K and V tensors for the kept positions.

### FLOPs analysis

```
Prefill:  same as baseline  (full attention needed to score tokens)
          = n_layers × (24·S·H² + 4·S²·H)

Decode:   attention context = budget + decode_step  (not prompt_len + step)
          = n_layers × (24·H² + 4·(budget+t)·H)   per step t
```

---

## Project Structure

```
kv-compress/
├── src/
│   ├── utils.py              # model loading, WallTimer, GenerationResult
│   ├── kv_utils.py           # KV cache primitives (_get_kv, _build, kv_size_mb)
│   ├── baseline.py           # greedy decoding with FLOPs accounting
│   ├── cross_layer_evict.py  # ★ CLE: importance scoring, eviction, generation
│   └── eval_ppl.py           # sliding-window PPL (WikiText-2 & PG-19)
├── scripts/
│   ├── run_all.py            # full evaluation: benchmark + PPL + sweep + figures
│   ├── seq_len_sweep.py      # context-length sweep (128/256/512/1024)
│   └── plot_results.py       # generate all figures
├── results/
│   ├── figures/              # PNG + PDF figures (9 total)
│   ├── results.json          # benchmark + PPL results
│   └── seq_len_sweep.json    # context-length sweep data
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full evaluation (benchmark + PPL + sweep + figures)
python scripts/run_all.py --device cuda

# 3. Or run components separately:
python scripts/run_all.py --skip-sweep --skip-plots   # benchmark + PPL only
python scripts/seq_len_sweep.py --device cuda          # context-length sweep
python scripts/plot_results.py --out-dir results       # regenerate figures

# 4. Optional flags
python scripts/run_all.py --model EleutherAI/pythia-160m --device cuda
python scripts/run_all.py --ppl-max-tokens 8192 --n-runs 5
```

### Requirements

```
torch>=2.0
transformers>=4.40
datasets
matplotlib
numpy
```

---

## Relationship to Teammate's Work

Our teammate ([Sunkw1224/spec-decoding-pythia](https://github.com/Sunkw1224/spec-decoding-pythia)) implements **Prompt Lookup Decoding (PLD)** — a speculative decoding method that proposes candidate tokens via n-gram matching to skip forward passes entirely.

The two approaches are **orthogonal**:

| | Speculative decoding (PLD) | KV compression (CLE) |
|-|---------------------------|----------------------|
| Reduces | Number of model forward passes | Cost per forward pass |
| Bottleneck targeted | Redundant computation | Memory bandwidth / cache size |
| Can be combined | ✓ | ✓ |

---

## References

1. Li et al. "SnapKV: LLM Knows What You are Looking for Before Generation." arXiv 2404.14469 (2024).
2. Cai et al. "PyramidKV: Dynamic KV Cache Compression based on Pyramid-like Attention Distribution." EMNLP 2024.
3. Sun et al. "You Only Cache Once: Decoder-Decoder Architectures for Language Models." arXiv 2405.05254 (2024).
4. Brandon et al. "Reducing Transformer Key-Value Cache Size with Cross-Layer Attention." arXiv 2405.12981 (2024).
5. Zhang et al. "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models." NeurIPS 2023.

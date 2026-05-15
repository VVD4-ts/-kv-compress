"""
Comprehensive evaluation: YOCO, CLE, PLD, StreamingLLM
  - PPL  : sliding-window on 100k-token PG-19
  - Speed: TTFT / TPOT / Throughput / FLOPs  on 4096-token prompt

Progress is written to results/eval_full_progress.json after every method.
"""

from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
from statistics import mean
from typing import Callable, Optional

import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils        import load_model, set_seed, kv_size_mb, make_result, WallTimer, num_layers
from baseline     import greedy_generate
from streaming_llm import streaming_generate, _trim_to_streaming_window, estimate_flops_streaming
from cross_layer_evict import cle_generate, compute_cross_layer_importance, evict_kv_cache, estimate_flops_cle, CLE_CONFIGS
from kv_utils        import kv_cache_length
from pld          import pld_generate
from yoco         import yoco_generate

PROGRESS_PATH = Path("results/eval_full_progress.json")


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_pg19_tokens(n_tokens: int, tokenizer) -> torch.Tensor:
    from datasets import load_dataset
    print(f"  Loading ~{n_tokens} tokens from PG-19 …")
    for hf_name, spl in [("emozilla/pg19-test-tokenized", "test"),
                          ("storytracer/pg19-books", "train")]:
        try:
            ds = load_dataset(hf_name, split=spl, streaming=True)
            chunks, total = [], 0
            for s in ds:
                txt = s.get("text") or s.get("book_text") or ""
                if len(txt) < 1000: continue
                ids = tokenizer(txt, return_tensors="pt").input_ids[0]
                chunks.append(ids); total += len(ids)
                if total >= n_tokens: break
            if total >= n_tokens // 2:
                ids = torch.cat(chunks)[:n_tokens]
                print(f"  [pg19: {hf_name}, {ids.shape[0]} tokens]")
                return ids
        except Exception: continue
    # fallback
    from datasets import load_dataset
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())
    while len(text) < n_tokens * 6: text = text + "\n\n" + text
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    print("  [pg19 fallback: wikitext-2 repeated]")
    return ids[:n_tokens]


def load_prompt_str(n_tokens: int, tokenizer) -> str:
    ids = load_pg19_tokens(n_tokens, tokenizer)
    return tokenizer.decode(ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# PPL evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ppl(model, input_ids, window, stride, device,
                kv_transform=None, label="") -> float:
    seq_len = input_ids.shape[0]
    n_win   = max(1, (seq_len - 1) // stride)
    nlls, prev_end = [], 0
    for wi, begin in enumerate(range(0, seq_len, stride)):
        end    = min(begin + window, seq_len)
        chunk  = input_ids[begin:end].unsqueeze(0).to(device)
        tlen   = end - prev_end
        if tlen <= 0: prev_end = end; continue
        past_kv, logits_list = None, []
        with torch.no_grad():
            for t in range(chunk.shape[1]):
                out = model(chunk[:, t:t+1], past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                if kv_transform: past_kv = kv_transform(past_kv)
                logits_list.append(out.logits[:, 0, :])
        logits = torch.stack(logits_list, dim=1)
        if tlen > 1:
            lp  = F.log_softmax(logits[:, -tlen:-1, :], dim=-1)
            lab = chunk[:, -tlen+1:]
            nll = -lp.gather(2, lab.unsqueeze(2)).squeeze(2)
            nlls.append(nll.mean().item())
        prev_end = end
        if (wi + 1) % 20 == 0 or end == seq_len:
            ppl = math.exp(sum(nlls)/len(nlls)) if nlls else float("nan")
            print(f"    [{label}] {wi+1}/{n_win}  pos={end}/{seq_len}  ppl={ppl:.2f}", flush=True)
        if end == seq_len: break
    return math.exp(sum(nlls)/len(nlls)) if nlls else float("nan")


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
        if i >= n_warmup: results.append(r)
    acc_list = [r.n_accepted / r.n_proposed
                if getattr(r, "n_proposed", 0) > 0 else None
                for r in results]
    acc = next((a for a in acc_list if a is not None), None)
    return {
        "ttft_ms":    round(mean(r.ttft_ms    for r in results), 3),
        "tpot_ms":    round(mean(r.tpot_ms    for r in results), 3),
        "throughput": round(mean(r.throughput  for r in results), 2),
        "kv_mb":      round(mean(r.kv_size_mb  for r in results), 3),
        "total_gflops":       round(results[0].total_gflops, 4),
        "avg_gflops_per_tok": round(results[0].avg_gflops_per_tok, 6),
        "num_gen":    results[0].num_generated,
        "acc_rate":   round(acc, 4) if acc is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Progress persistence
# ─────────────────────────────────────────────────────────────────────────────

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
    p.add_argument("--out-dir",        default="results")
    args = p.parse_args()

    set_seed(42)
    model, tokenizer = load_model(args.model, device=args.device)
    n_lay = num_layers(model)
    print(f"\nModel: {args.model}  Layers={n_lay}  Device={args.device}")

    # Resume from existing progress if available
    if PROGRESS_PATH.exists():
        try:
            with open(PROGRESS_PATH) as f:
                progress = json.load(f)
            progress["status"] = "running"
            print(f"  [Resuming from {PROGRESS_PATH}  "
                  f"PPL done: {list(progress['ppl'].keys())}]")
        except Exception:
            progress = {"model": args.model, "device": args.device,
                        "ppl_tokens": args.ppl_tokens, "speed_ctx": args.speed_ctx,
                        "status": "running", "ppl": {}, "speed": {}}
    else:
        progress = {"model": args.model, "device": args.device,
                    "ppl_tokens": args.ppl_tokens, "speed_ctx": args.speed_ctx,
                    "status": "running", "ppl": {}, "speed": {}}
    save_progress(progress)

    # ── Load data ─────────────────────────────────────────────────────────────
    ppl_ids    = load_pg19_tokens(args.ppl_tokens, tokenizer)
    speed_prompt = tokenizer.decode(ppl_ids[:args.speed_ctx], skip_special_tokens=True)
    print(f"  PPL corpus: {ppl_ids.shape[0]} tokens")
    print(f"  Speed prompt: {tokenizer(speed_prompt, return_tensors='pt').input_ids.shape[1]} tokens\n")

    # ── Method definitions ────────────────────────────────────────────────────
    # (label, ppl_transform, speed_fn, speed_kwargs)
    def _stream_t(n_sink, window):
        def t(kv): return _trim_to_streaming_window(kv, n_sink, window)
        return t

    def _cle_t(budget_ratio):
        state = {"done": False}
        def t(kv):
            if state["done"]: return kv
            imp    = compute_cross_layer_importance(kv)   # shape [n_tokens]
            n      = imp.shape[0]
            budget = max(1, int(n * budget_ratio))
            keep   = imp.topk(budget).indices.sort().values
            state["done"] = True
            return evict_kv_cache(kv, keep)
        return t

    def _yoco_t(split_idx):
        from kv_utils import apply_yoco
        def t(kv): return apply_yoco(kv, split_idx)
        return t

    # PPL transforms (applied per-token inside sliding window)
    ppl_methods = [
        ("Baseline",               None),
        ("StreamingLLM(W=128)",    _stream_t(4, 128)),
        ("StreamingLLM(W=256)",    _stream_t(4, 256)),
        ("StreamingLLM(W=512)",    _stream_t(4, 512)),
        ("CLE-Medium(-50%KV)",     _cle_t(0.50)),
        ("CLE-Heavy(-70%KV)",      _cle_t(0.30)),
        ("YOCO(split=4,-33%KV)",   _yoco_t(4)),
        ("YOCO(split=5,-17%KV)",   _yoco_t(5)),
    ]

    # Speed functions
    speed_methods = [
        ("Baseline",             greedy_generate,    {}),
        ("StreamingLLM(W=128)",  streaming_generate, {"window": 128, "n_sink": 4}),
        ("StreamingLLM(W=256)",  streaming_generate, {"window": 256, "n_sink": 4}),
        ("StreamingLLM(W=512)",  streaming_generate, {"window": 512, "n_sink": 4}),
        ("CLE-Medium(-50%KV)",   cle_generate,       {"budget_ratio": 0.50}),
        ("CLE-Heavy(-70%KV)",    cle_generate,       {"budget_ratio": 0.30}),
        ("PLD(K=5)",             pld_generate,       {"K": 5}),
        ("PLD(K=10)",            pld_generate,       {"K": 10}),
        ("YOCO(split=4,-33%KV)", yoco_generate,      {"split_idx": 4}),
        ("YOCO(split=5,-17%KV)", yoco_generate,      {"split_idx": 5}),
    ]

    # ── PPL evaluation ────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  PPL Evaluation  ({ppl_ids.shape[0]} tokens)")
    print(f"{'═'*60}")

    for label, transform in ppl_methods:
        if label in progress["ppl"]:
            print(f"\n── {label} ── (already done: PPL={progress['ppl'][label]})")
            continue
        print(f"\n── {label} ──")
        # reset stateful transforms
        if callable(transform):
            # rebuild to reset internal state
            if "CLE" in label:
                ratio = 0.50 if "50%" in label else 0.30
                transform = _cle_t(ratio)
            elif "YOCO" in label:
                si = 4 if "split=4" in label else 5
                transform = _yoco_t(si)
        ppl = compute_ppl(model, ppl_ids,
                          window=args.ppl_window, stride=args.ppl_stride,
                          device=args.device, kv_transform=transform, label=label)
        progress["ppl"][label] = round(ppl, 3)
        save_progress(progress)
        print(f"  → PPL = {ppl:.3f}")

    # ── Speed benchmark ───────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Speed Benchmark  (ctx={args.speed_ctx}, gen={args.max_new_tokens})")
    print(f"{'═'*60}")

    for label, fn, kw in speed_methods:
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
            acc = f"  AccR={s['acc_rate']*100:.1f}%" if s["acc_rate"] else ""
            print(f"  TTFT={s['ttft_ms']:.1f}ms  TPOT={s['tpot_ms']:.2f}ms  "
                  f"Tput={s['throughput']:.1f}tok/s  "
                  f"KV={s['kv_mb']:.2f}MB  GFLOPs={s['total_gflops']:.3f}{acc}")
        except Exception as e:
            print(f"  ERROR: {e}")
            progress["speed"][label] = {"error": str(e)}
            save_progress(progress)

    # ── Final summary ─────────────────────────────────────────────────────────
    progress["status"] = "done"
    save_progress(progress)

    base_ppl   = progress["ppl"].get("Baseline", 1)
    base_tpot  = progress["speed"].get("Baseline", {}).get("tpot_ms", 1)
    base_kv    = progress["speed"].get("Baseline", {}).get("kv_mb", 1)
    base_fl    = progress["speed"].get("Baseline", {}).get("total_gflops", 1)

    print(f"\n{'═'*95}")
    print(f"  Full Summary")
    print(f"{'═'*95}")
    print(f"  {'Method':<26} {'PPL':>8} {'ΔPPL':>7} | {'TTFT':>7} {'TPOT':>7} {'Tok/s':>7} "
          f"{'KV↓%':>6} {'GFLOPs':>8} {'Spd':>6} {'AccR':>6}")
    print(f"  {'─'*93}")

    all_labels = list(dict.fromkeys(
        [l for l,_ in ppl_methods] + [l for l,_,_ in speed_methods]))
    for lbl in all_labels:
        ppl   = progress["ppl"].get(lbl, float("nan"))
        dppl  = ppl - base_ppl
        sp    = progress["speed"].get(lbl, {})
        tpot  = sp.get("tpot_ms", float("nan"))
        tput  = sp.get("throughput", float("nan"))
        ttft  = sp.get("ttft_ms", float("nan"))
        kv    = sp.get("kv_mb", float("nan"))
        fl    = sp.get("total_gflops", float("nan"))
        acc   = sp.get("acc_rate")
        spd   = base_tpot / tpot if tpot and tpot > 0 else float("nan")
        kvrd  = (1 - kv/base_kv)*100 if kv and base_kv else float("nan")
        acc_s = f"{acc*100:.1f}%" if acc else "  ---"
        print(f"  {lbl:<26} {ppl:>8.2f} {dppl:>+7.2f} | "
              f"{ttft:>7.1f} {tpot:>7.2f} {tput:>7.1f} "
              f"{kvrd:>5.1f}% {fl:>8.3f} {spd:>5.2f}x {acc_s:>6}")

    out = Path(args.out_dir) / "eval_full_results.json"
    with open(out, "w") as f:
        json.dump(progress, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()

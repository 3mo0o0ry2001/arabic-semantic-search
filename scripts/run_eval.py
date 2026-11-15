"""Benchmark lexical, dense and hybrid Arabic retrieval on ARCD.

Example:
  python scripts/run_eval.py --arcd data/arcd.json --distractors data/arabic-squad.json \
      --sizes 0 2000 --models-dir models --models e5-small e5-base bge-m3

Dense embeddings are read from --cache when present (see scripts/encode_cached.py);
otherwise they are computed. Query latency is always measured on uncached,
single-query, CPU encoding.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arabic_search.bm25 import BM25  # noqa: E402
from arabic_search.data import load_benchmark  # noqa: E402
from arabic_search.dense import SPECS, DenseRetriever, OnnxEncoder  # noqa: E402
from arabic_search.metrics import bootstrap_ci, per_query, rrf, top_ids  # noqa: E402

DEPTH = 100
BASELINE = "BM25 (normalized + light stem)"
EXTRA_PAIRS = [  # (A, B): is A better than B?
    ("Hybrid RRF (BM25 + bge-m3)", "Hybrid RRF (BM25 + multilingual-e5-small)"),
    ("Hybrid RRF (BM25 + bge-m3)", "bge-m3"),
    ("multilingual-e5-small", "multilingual-e5-base"),
]


def lexical_rankings(retriever: BM25, queries: list[dict]) -> dict[str, list[str]]:
    return {q["id"]: top_ids(retriever.scores(q["text"]), retriever.ids, DEPTH) for q in queries}


def dense_rankings(retriever: DenseRetriever, queries: list[dict]) -> dict[str, list[str]]:
    scores = retriever.scores_batch([q["text"] for q in queries])
    return {q["id"]: top_ids(scores[i], retriever.ids, DEPTH) for i, q in enumerate(queries)}


def query_latency_ms(fn, queries: list[dict], n: int = 50, warmup: int = 3) -> float:
    """Median wall-clock time of one query, after a few warm-up calls."""
    for q in queries[:warmup]:
        fn(q["text"])
    times = []
    for q in queries[warmup:warmup + n]:
        t0 = time.perf_counter(); fn(q["text"]); times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def hardware_info(with_ort: bool) -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "logical_cpus": os.cpu_count()}
    if with_ort:
        import onnxruntime
        info["onnxruntime"] = onnxruntime.__version__
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arcd", required=True)
    ap.add_argument("--distractors")
    ap.add_argument("--sizes", type=int, nargs="+", default=[0], help="number of distractor passages per run")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--models", nargs="*", default=[], choices=sorted(SPECS))
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--out", default="results")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    encoders = {m: OnnxEncoder(Path(args.models_dir) / m, SPECS[m]) for m in args.models}
    meta = hardware_info(bool(encoders))
    runs_report, cmp_report = [], []

    for size in args.sizes:
        bench = load_benchmark(args.arcd, args.distractors, size)
        qids = [q["id"] for q in bench.queries]
        groups = [sorted(bench.qrels[q])[0] for q in qids]   # cluster = gold paragraph
        n_pass = len(bench.passages)
        print(f"\n== corpus: {n_pass} passages, {len(qids)} queries, {len(set(groups))} gold paragraphs ==", flush=True)
        runs: dict[str, dict[str, list[str]]] = {}
        latency: dict[str, float] = {}

        for name, stem in (("BM25 (normalized)", False), (BASELINE, True)):
            bm = BM25(stem=stem).index(bench.passages)
            runs[name] = lexical_rankings(bm, bench.queries)
            latency[name] = query_latency_ms(bm.search, bench.queries)
        bm25_best = runs[BASELINE]

        for m, enc in encoders.items():
            dr = DenseRetriever(enc, args.cache).index(bench.passages, bench.fingerprint())
            label = SPECS[m].name
            runs[label] = dense_rankings(dr, bench.queries)
            latency[label] = query_latency_ms(dr.search, bench.queries)
            hybrid = f"Hybrid RRF (BM25 + {label})"
            runs[hybrid] = {qid: rrf(bm25_best[qid], runs[label][qid])[:DEPTH] for qid in qids}
            latency[hybrid] = latency[label] + latency[BASELINE]   # sum: both run per query

        pq = {name: per_query(r, bench.qrels, qids) for name, r in runs.items()}
        for name, m in pq.items():
            lo, hi = bootstrap_ci(m["MRR@10"], groups, args.boot)
            row = {"corpus_passages": n_pass, "distractors": size, "retriever": name,
                   **{k: float(v.mean()) for k, v in m.items()},
                   "MRR@10_ci95": [lo, hi], "query_ms_p50": latency[name]}
            runs_report.append(row)
            print(f"  {name:45s} R@1 {row['Recall@1']:.3f}  R@10 {row['Recall@10']:.3f}  "
                  f"MRR@10 {row['MRR@10']:.3f} [{lo:.3f},{hi:.3f}]  nDCG@10 {row['nDCG@10']:.3f}  {latency[name]:.1f} ms", flush=True)

        pairs = [(n, BASELINE) for n in runs if n != BASELINE] + [p for p in EXTRA_PAIRS if p[0] in runs and p[1] in runs]
        for a, b in pairs:
            diff = pq[a]["MRR@10"] - pq[b]["MRR@10"]
            lo, hi = bootstrap_ci(diff, groups, args.boot)
            cmp_report.append({"distractors": size, "corpus_passages": n_pass, "a": a, "b": b,
                               "delta_mrr10": float(diff.mean()), "ci95": [lo, hi], "excludes_zero": lo > 0 or hi < 0})

        best = max((r for r in runs_report if r["distractors"] == size), key=lambda r: r["MRR@10"])["retriever"]
        texts = {p["id"]: p for p in bench.passages}
        misses = [{"query": q["text"], "gold": [texts[g]["title"] for g in bench.qrels[q["id"]]],
                   "top3": [texts[p]["title"] for p in runs[best][q["id"]][:3]]}
                  for q in bench.queries if not bench.qrels[q["id"]] & set(runs[best][q["id"]][:10])]
        (out / f"misses_{size}.json").write_text(json.dumps({"retriever": best, "misses": misses}, ensure_ascii=False, indent=1), encoding="utf-8")

    (out / "results.json").write_text(json.dumps({"meta": meta, "runs": runs_report, "comparisons": cmp_report},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [f"_Latency: median of one query at a time (batch size 1, CPU, uncached), measured on {meta['platform']}, "
             f"{meta['logical_cpus']} logical CPUs" + (f", onnxruntime {meta['onnxruntime']}" if 'onnxruntime' in meta else "") + ". "
             f"Hybrid latency is the sum of its two parts. Confidence intervals: {args.boot}-resample bootstrap "
             f"over gold paragraphs (questions about the same paragraph are not independent)._"]
    for size in args.sizes:
        rows = [r for r in runs_report if r["distractors"] == size]
        lines += [f"\n### Corpus: {rows[0]['corpus_passages']:,} passages ({size:,} distractors)\n",
                  "| Retriever | Recall@1 | Recall@10 | MRR@10 [95% CI] | nDCG@10 | Query ms (p50) |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in sorted(rows, key=lambda r: -r["MRR@10"]):
            lo, hi = r["MRR@10_ci95"]
            lines.append(f"| {r['retriever']} | {r['Recall@1']:.3f} | {r['Recall@10']:.3f} | "
                         f"{r['MRR@10']:.3f} [{lo:.3f}, {hi:.3f}] | {r['nDCG@10']:.3f} | {r['query_ms_p50']:.1f} |")
        lines += [f"\nPaired difference in MRR@10 on the same corpus (A - B):\n",
                  "| A | B | Δ MRR@10 | 95% CI | CI excludes 0 |", "|---|---|---:|---:|:---:|"]
        for c in (c for c in cmp_report if c["distractors"] == size):
            lines.append(f"| {c['a']} | {c['b']} | {c['delta_mrr10']:+.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | "
                         f"{'yes' if c['excludes_zero'] else 'no'} |")
    (out / "results.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"\nwrote {out/'results.json'} and {out/'results.md'}")


if __name__ == "__main__":
    main()

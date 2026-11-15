"""Search the ARCD corpus from the command line.

  python scripts/search.py "متى ولد المتنبي؟"                       # BM25
  python scripts/search.py "متى ولد المتنبي؟" --model bge-m3 --hybrid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arabic_search.bm25 import BM25  # noqa: E402
from arabic_search.data import load_benchmark  # noqa: E402
from arabic_search.dense import SPECS, DenseRetriever, OnnxEncoder  # noqa: E402
from arabic_search.metrics import rrf  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--arcd", default=str(ROOT / "data" / "arcd.json"))
    ap.add_argument("--model", choices=sorted(SPECS))
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--hybrid", action="store_true", help="fuse BM25 and dense results with RRF")
    ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()

    bench = load_benchmark(args.arcd)
    by_id = {p["id"]: p for p in bench.passages}
    lexical = [pid for pid, _ in BM25().index(bench.passages).search(args.query, 100)]
    ranking = lexical
    if args.model:
        dense = DenseRetriever(OnnxEncoder(Path(args.models_dir) / args.model, SPECS[args.model]), ROOT / ".cache")
        dense.index(bench.passages, bench.fingerprint())
        semantic = [pid for pid, _ in dense.search(args.query, 100)]
        ranking = rrf(lexical, semantic) if args.hybrid else semantic

    for rank, pid in enumerate(ranking[: args.k], start=1):
        p = by_id[pid]
        print(f"{rank}. [{p['title']}] {p['text'][:220]}…\n")


if __name__ == "__main__":
    main()

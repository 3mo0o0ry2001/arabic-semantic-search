"""Resumable embedding: encode the corpus and queries in time-boxed chunks.

Useful on slow or time-limited machines. Each call encodes for at most
--budget seconds, saves its progress, and exits; rerun until it prints DONE.
When finished it writes the same cache files run_eval.py reads, so the
benchmark then only does the (cheap) scoring.

  python scripts/encode_cached.py --model bge-m3 --sizes 0 5000 --budget 150
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arabic_search.data import load_benchmark  # noqa: E402
from arabic_search.dense import SPECS, OnnxEncoder, passage_text, query_cache_key  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(SPECS))
    ap.add_argument("--arcd", default=str(ROOT / "data" / "arcd.json"))
    ap.add_argument("--distractors", default=str(ROOT / "data" / "arabic-squad.json"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[0, 5000])
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--cache", default=str(ROOT / ".cache"))
    ap.add_argument("--budget", type=float, default=150, help="seconds of encoding per call")
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()

    spec = SPECS[args.model]
    cache = Path(args.cache)
    # The largest corpus is a superset whose prefix is every smaller corpus.
    benches = {s: load_benchmark(args.arcd, args.distractors, s) for s in args.sizes}
    big = benches[max(args.sizes)]
    texts = [passage_text(p) for p in big.passages]
    queries = [q["text"] for q in big.queries]
    tag = hashlib.sha256(("\n".join(texts) + "\n#Q\n" + "\n".join(queries)).encode()).hexdigest()[:10]
    work = cache / "partial" / f"{spec.name}-{tag}"
    work.mkdir(parents=True, exist_ok=True)

    jobs = [("p", texts), ("q", queries)]
    enc = OnnxEncoder(Path(args.models_dir) / args.model, spec)
    t0 = time.perf_counter()
    for kind, items in jobs:
        for start in range(0, len(items), args.chunk):
            f = work / f"{kind}-{start:06d}.npy"
            if f.exists():
                continue
            if time.perf_counter() - t0 > args.budget:
                done = len(list(work.glob("*.npy")))
                total = sum(-(-len(x) // args.chunk) for _, x in jobs)
                print(f"PARTIAL {done}/{total} chunks")
                return
            batch = items[start:start + args.chunk]
            vecs = enc.encode_passages(batch) if kind == "p" else enc.encode_queries(batch)
            np.save(f, vecs)

    P = np.vstack([np.load(f) for f in sorted(work.glob("p-*.npy"))])
    Q = np.vstack([np.load(f) for f in sorted(work.glob("q-*.npy"))])
    assert P.shape[0] == len(texts) and Q.shape[0] == len(queries)
    for s, b in benches.items():
        assert [p["id"] for p in b.passages] == [p["id"] for p in big.passages[: len(b.passages)]]
        np.save(cache / f"{spec.name}-{b.fingerprint()}.npy", P[: len(b.passages)])
    np.save(cache / f"{spec.name}-{query_cache_key(queries)}.q.npy", Q)
    print(f"DONE passages={P.shape} queries={Q.shape}")


if __name__ == "__main__":
    main()

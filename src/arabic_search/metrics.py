"""Ranking metrics, rank fusion and bootstrap statistics."""
from __future__ import annotations

import math

import numpy as np


def rank_of_first_relevant(ranked_ids: list[str], relevant: set[str]) -> int | None:
    for r, pid in enumerate(ranked_ids, start=1):
        if pid in relevant:
            return r
    return None


def per_query(rankings: dict[str, list[str]], qrels: dict[str, set[str]],
              qids: list[str] | None = None, ks=(1, 5, 10, 20)) -> dict[str, np.ndarray]:
    """Per-query Recall@k, MRR@10 and nDCG@10 (binary relevance), in the order of `qids`."""
    qids = list(qrels) if qids is None else qids
    out = {f"Recall@{k}": np.zeros(len(qids)) for k in ks}
    out["MRR@10"] = np.zeros(len(qids))
    out["nDCG@10"] = np.zeros(len(qids))
    for i, qid in enumerate(qids):
        relevant, ranked = qrels[qid], rankings.get(qid, [])
        for k in ks:
            out[f"Recall@{k}"][i] = len(relevant & set(ranked[:k])) / len(relevant)
        r = rank_of_first_relevant(ranked[:10], relevant)
        out["MRR@10"][i] = 1 / r if r else 0.0
        dcg = sum(1 / math.log2(j + 1) for j, pid in enumerate(ranked[:10], start=1) if pid in relevant)
        idcg = sum(1 / math.log2(j + 1) for j in range(1, min(len(relevant), 10) + 1))
        out["nDCG@10"][i] = dcg / idcg
    return out


def evaluate(rankings: dict[str, list[str]], qrels: dict[str, set[str]], ks=(1, 5, 10, 20)) -> dict[str, float]:
    """Recall@k, MRR@10 and nDCG@10 averaged over queries."""
    return {name: float(v.mean()) for name, v in per_query(rankings, qrels, ks=ks).items()}


def bootstrap_ci(values, groups, n_boot: int = 2000, alpha: float = 0.05, seed: int = 13) -> tuple[float, float]:
    """Percentile CI for the mean of `values`, resampling whole `groups` (cluster bootstrap).

    ARCD has about three questions per paragraph, so questions are not independent;
    resampling paragraphs instead of questions avoids understating the uncertainty.
    Passing the per-query difference of two systems gives a paired comparison.
    """
    values = np.asarray(values, dtype=float)
    _, inverse = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse).astype(float)
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def top_ids(scores: np.ndarray, ids: list[str], k: int) -> list[str]:
    k = min(k, len(ids))
    part = np.argpartition(-scores, k - 1)[:k]
    return [ids[i] for i in part[np.argsort(-scores[part], kind="stable")]]


def rrf(*rankings: list[str], k: int = 60, depth: int = 100) -> list[str]:
    """Reciprocal Rank Fusion (Cormack et al., 2009)."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for r, pid in enumerate(ranking[:depth], start=1):
            fused[pid] = fused.get(pid, 0.0) + 1 / (k + r)
    return sorted(fused, key=lambda p: (-fused[p], p))

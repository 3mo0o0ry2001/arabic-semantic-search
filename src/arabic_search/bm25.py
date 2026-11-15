"""Okapi BM25 over Arabic-normalized, light-stemmed tokens."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from .normalize import tokenize


class BM25:
    name = "bm25"

    def __init__(self, k1: float = 1.2, b: float = 0.75, stem: bool = True, title_boost: int = 1):
        self.k1, self.b, self.stem, self.title_boost = k1, b, stem, title_boost

    def index(self, passages: list[dict]) -> "BM25":
        self.ids = [p["id"] for p in passages]
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len = np.zeros(len(passages), dtype=np.float32)
        for i, p in enumerate(passages):
            # Repeating the title gives it extra weight without a separate field model.
            toks = tokenize(p["text"], stem=self.stem) + tokenize(p.get("title", ""), stem=self.stem) * self.title_boost
            self.doc_len[i] = len(toks)
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))
        self.avgdl = float(self.doc_len.mean()) if len(passages) else 0.0
        n = len(passages)
        self.idf = {t: math.log(1 + (n - len(ps) + 0.5) / (len(ps) + 0.5)) for t, ps in self.postings.items()}
        return self

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.ids), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)
        for term in set(tokenize(query, stem=self.stem)):
            for i, tf in self.postings.get(term, ()):
                s[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm[i])
        return s

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        s = self.scores(query)
        top = np.argsort(-s, kind="stable")[:k]
        return [(self.ids[i], float(s[i])) for i in top]

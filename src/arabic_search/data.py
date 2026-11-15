"""Build a retrieval benchmark (corpus, queries, relevance labels) from ARCD.

ARCD (Mozannar et al., 2019) has 1,395 crowd-written Arabic questions over 465
Arabic Wikipedia paragraphs. Each question's source paragraph is its single
relevant passage. Optionally, paragraphs from Arabic-SQuAD (machine-translated
SQuAD) are added as distractors to make the corpus larger and harder.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from .normalize import normalize


@dataclass
class Benchmark:
    passages: list[dict]                       # {"id", "title", "text", "source"}
    queries: list[dict]                        # {"id", "text"}
    qrels: dict[str, set[str]] = field(default_factory=dict)  # query id -> relevant passage ids

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for p in self.passages:
            h.update(p["id"].encode()); h.update(p["text"].encode())
        return h.hexdigest()[:12]


def _clean_question(q: str) -> str:
    """ARCD has stray leading dashes and a dangling "ال" after some questions."""
    q = q.strip().lstrip("-").strip()
    return re.sub(r"\s+ال$", "", q).strip()


def _key(text: str) -> str:
    return hashlib.sha1(normalize(text).encode()).hexdigest()


def load_benchmark(arcd_path: str | Path, distractor_path: str | Path | None = None,
                   n_distractors: int = 0, seed: int = 13) -> Benchmark:
    arcd = json.loads(Path(arcd_path).read_text(encoding="utf-8"))["data"]
    passages: list[dict] = []
    by_key: dict[str, str] = {}
    queries: list[dict] = []
    qrels: dict[str, set[str]] = {}

    for article in arcd:
        for para in article["paragraphs"]:
            k = _key(para["context"])
            if k not in by_key:  # identical paragraphs collapse to one passage
                pid = f"arcd-{len(by_key)}"
                by_key[k] = pid
                passages.append({"id": pid, "title": article["title"], "text": para["context"].strip(), "source": "arcd"})
            pid = by_key[k]
            for qa in para["qas"]:
                q = _clean_question(qa["question"])
                if not q:
                    continue
                qid = f"q-{qa['id']}"
                if qid in qrels:          # duplicate question id: merge labels
                    qrels[qid].add(pid)
                    continue
                queries.append({"id": qid, "text": q})
                qrels[qid] = {pid}

    if distractor_path and n_distractors > 0:
        squad = json.loads(Path(distractor_path).read_text(encoding="utf-8"))["data"]
        pool = [(a["title"], p["context"]) for a in squad for p in a["paragraphs"]]
        rng = random.Random(seed)
        rng.shuffle(pool)
        added = 0
        for title, text in pool:
            k = _key(text)
            if k in by_key or len(normalize(text).split()) < 20:
                continue
            by_key[k] = pid = f"dist-{added}"
            passages.append({"id": pid, "title": title.replace("_", " "), "text": text.strip(), "source": "arabic-squad"})
            added += 1
            if added == n_distractors:
                break

    return Benchmark(passages, queries, qrels)

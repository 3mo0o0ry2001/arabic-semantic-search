import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arabic_search.bm25 import BM25
from arabic_search.data import load_benchmark
from arabic_search.dense import DenseRetriever, ModelSpec, OnnxEncoder
from arabic_search.metrics import evaluate, rrf, top_ids
from arabic_search.normalize import light_stem, normalize, tokenize


def test_normalize_unifies_letter_variants_and_strips_diacritics():
    assert normalize("أَحْمَدُ إلى مدرسةٍ") == "احمد الي مدرسه"
    assert normalize("كتـــاب!") == "كتاب"


def test_light_stem_keeps_three_letter_minimum():
    assert light_stem("والكتاب") == "كتاب"
    assert light_stem("المعلمون") == "معلم"
    assert light_stem("ولد") == "ولد"  # too short to strip


def test_tokenize_drops_question_words():
    assert "متي" not in tokenize("متى ولد الشاعر؟")


def test_bm25_ranks_matching_passage_first():
    docs = [{"id": "a", "title": "القاهرة", "text": "القاهرة عاصمة مصر وأكبر مدنها"},
            {"id": "b", "title": "الرياض", "text": "الرياض عاصمة المملكة العربية السعودية"}]
    bm = BM25().index(docs)
    assert bm.search("ما هي عاصمة مصر؟", k=1)[0][0] == "a"
    assert bm.search("الرياض", k=1)[0][0] == "b"


def test_metrics_single_relevant():
    rankings = {"q1": ["x", "a", "y"], "q2": ["b"], "q3": ["z"]}
    qrels = {"q1": {"a"}, "q2": {"b"}, "q3": {"c"}}
    m = evaluate(rankings, qrels, ks=(1, 5))
    assert m["Recall@1"] == pytest.approx(1 / 3)
    assert m["Recall@5"] == pytest.approx(2 / 3)
    assert m["MRR@10"] == pytest.approx((0.5 + 1) / 3)
    assert m["nDCG@10"] == pytest.approx((1 / np.log2(3) + 1) / 3)


def test_rrf_prefers_documents_ranked_well_by_both():
    assert rrf(["a", "b", "c"], ["b", "a", "d"])[:2] in (["a", "b"], ["b", "a"])
    assert rrf(["a", "b"], ["b", "c"])[0] == "b"


def test_top_ids_is_sorted_and_bounded():
    assert top_ids(np.array([0.1, 0.9, 0.5]), ["a", "b", "c"], 5) == ["b", "c", "a"]


class _FakeEncoding:
    def __init__(self, ids):
        self.ids, self.attention_mask = ids, [1] * len(ids)


class _FakeTokenizer:
    def enable_truncation(self, max_length): pass
    def enable_padding(self): pass
    def encode_batch(self, texts):
        n = max(len(t) for t in texts)
        return [_FakeEncoding([ord(c) % 7 for c in t.ljust(n)]) for t in texts]


class _FakeInput:
    def __init__(self, name): self.name = name


class _FakeSession:
    """Hidden state = one-hot of the first character id, so pooling is checkable."""
    def get_inputs(self): return [_FakeInput("input_ids"), _FakeInput("attention_mask")]
    def run(self, _, feed):
        ids = feed["input_ids"]
        return [np.eye(7, dtype=np.float32)[ids] + 1e-3]


@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_onnx_encoder_returns_unit_vectors_and_applies_prefixes(pooling):
    spec = ModelSpec("fake", pooling, "query: ", "passage: ")
    enc = OnnxEncoder(".", spec, session=_FakeSession(), tokenizer=_FakeTokenizer())
    q = enc.encode_queries(["abc", "de"])
    assert q.shape == (2, 7)
    assert np.allclose(np.linalg.norm(q, axis=1), 1)
    retr = DenseRetriever(enc).index([{"id": "p1", "text": "abc"}, {"id": "p2", "text": "xyz"}])
    assert len(retr.search("abc", k=2)) == 2


def test_benchmark_dedupes_paragraphs_and_adds_distractors(tmp_path):
    arcd = {"data": [{"title": "T", "paragraphs": [
        {"context": "نص أول عن موضوع", "qas": [{"id": "1", "question": " - سؤال؟ ", "answers": []}]},
        {"context": "نَص أول عن موضوع", "qas": [{"id": "2", "question": "سؤال آخر", "answers": []}]}]}]}
    words = " ".join(["كلمة"] * 25)
    squad = {"data": [{"title": "S_X", "paragraphs": [{"context": f"{words} {i}", "qas": []} for i in range(5)]}]}
    a, s = tmp_path / "a.json", tmp_path / "s.json"
    a.write_text(json.dumps(arcd, ensure_ascii=False), encoding="utf-8")
    s.write_text(json.dumps(squad, ensure_ascii=False), encoding="utf-8")
    b = load_benchmark(a, s, n_distractors=3)
    assert sum(p["source"] == "arcd" for p in b.passages) == 1   # diacritics-only difference collapses
    assert sum(p["source"] == "arabic-squad" for p in b.passages) == 3
    assert b.queries[0]["text"] == "سؤال؟"
    assert b.qrels["q-1"] == b.qrels["q-2"] == {"arcd-0"}


# ---- statistics ----
from arabic_search.metrics import bootstrap_ci, per_query


def test_per_query_means_match_evaluate():
    rankings = {"q1": ["x", "a", "y"], "q2": ["b"], "q3": ["z"]}
    qrels = {"q1": {"a"}, "q2": {"b"}, "q3": {"c"}}
    pq = per_query(rankings, qrels, ["q1", "q2", "q3"])
    assert list(pq["MRR@10"]) == [0.5, 1.0, 0.0]
    assert pq["MRR@10"].mean() == pytest.approx(evaluate(rankings, qrels)["MRR@10"])


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    rng = np.random.default_rng(0)
    vals = rng.random(300)
    groups = np.repeat(np.arange(100), 3)
    lo, hi = bootstrap_ci(vals, groups, n_boot=500)
    assert lo < vals.mean() < hi
    assert (lo, hi) == bootstrap_ci(vals, groups, n_boot=500)


def test_paired_difference_of_identical_systems_is_exactly_zero():
    vals = np.random.default_rng(1).random(60)
    lo, hi = bootstrap_ci(vals - vals, np.repeat(np.arange(20), 3), n_boot=200)
    assert lo == hi == 0.0


def test_cluster_bootstrap_is_wider_than_naive_when_clusters_are_correlated():
    # Questions inside a paragraph share one value: effective sample size = number of paragraphs.
    vals = np.repeat(np.random.default_rng(2).random(40), 5)
    cluster = bootstrap_ci(vals, np.repeat(np.arange(40), 5), n_boot=1000)
    naive = bootstrap_ci(vals, np.arange(200), n_boot=1000)
    assert cluster[1] - cluster[0] > 1.5 * (naive[1] - naive[0])


def test_dense_search_does_not_write_query_cache(tmp_path):
    spec = ModelSpec("fake", "mean")
    enc = OnnxEncoder(".", spec, session=_FakeSession(), tokenizer=_FakeTokenizer())
    retr = DenseRetriever(enc, tmp_path).index([{"id": "p1", "text": "abc"}])
    retr.search("abc")
    assert not list(tmp_path.glob("*.q.npy"))

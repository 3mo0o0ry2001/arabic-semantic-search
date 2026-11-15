# Arabic Semantic Search: a measured comparison of BM25, embeddings and hybrid retrieval

A small, reproducible benchmark and search CLI for Arabic passage retrieval. It compares
a lexical baseline (BM25 with Arabic normalization and light stemming, default parameters), three
multilingual embedding models, and hybrid fusion, and reports confidence intervals
instead of single numbers.

This is a standalone personal project. It is not part of any employer's product or site.

## Findings

Evaluated on ARCD: 1,395 questions written by crowd-workers over 465 Arabic Wikipedia
paragraphs (460 after de-duplication). One run uses only those 460 paragraphs; a second adds
2,000 distractor paragraphs from Arabic-SQuAD (machine-translated), for 2,460 in total.
Each question has one gold paragraph. Intervals are 95% bootstrap CIs over gold paragraphs.

| Retriever | MRR@10, 460 passages | MRR@10, 2,460 passages | Recall@10, 2,460 |
|---|---:|---:|---:|
| BM25, normalized only | 0.749 | 0.721 | 0.896 |
| BM25, normalized + light stem | 0.801 | 0.771 | 0.933 |
| multilingual-e5-small | 0.770 | 0.740 | 0.909 |
| multilingual-e5-base | 0.736 | 0.707 | 0.900 |
| bge-m3 | 0.771 | 0.752 | 0.920 |
| Hybrid RRF, BM25 + e5-small | 0.810 | 0.780 | 0.946 |
| Hybrid RRF, BM25 + e5-base | 0.801 | 0.780 | 0.945 |
| **Hybrid RRF, BM25 + bge-m3** | **0.819** | **0.801** | **0.954** |

Full tables (Recall@1/10, nDCG@10, per-system CIs, latency) are in [`results/results.md`](results/results.md).

What the numbers support (paired differences in MRR@10, 95% CI):

- **Light stemming is the cheapest large win.** +0.052 [+0.037, +0.068] over normalization alone on 460 passages, +0.050 on 2,460.
- **No embedding model beat BM25 with stemming on its own.** e5-small is 0.030 lower [-0.052, -0.007] and e5-base 0.064 lower on 2,460 passages. bge-m3 is 0.019 lower there, and that interval includes zero.
- **Hybrid fusion helps, modestly.** BM25 + bge-m3 beats stemmed BM25 by +0.018 [+0.003, +0.034] on 460 passages and by +0.030 [+0.013, +0.047] on 2,460. The hybrids with e5-small and e5-base do not differ from BM25 significantly.
- **bge-m3 over e5-small in a hybrid** is +0.009 [-0.003, +0.021] on 460 passages (not distinguishable) and +0.021 [+0.009, +0.033] on 2,460.
- **e5-base scored below e5-small** (+0.034 [+0.016, +0.051] in favour of small). I did not investigate why, so I offer no explanation.

## Limitations

Read these before trusting any number above.

1. **Title leak in the distractor run.** Passages are indexed with their title. All 460 gold paragraphs have Arabic titles; all 2,000 distractors have English titles, so Arabic queries can match gold titles but never distractor titles. For BM25 this is worth about 0.03 MRR@10 on the 2,460 corpus (0.771 with titles, 0.742 without). I did not re-run the dense models without titles, so the 2,460-passage results are optimistic for every system, and the 460-passage run is the cleaner comparison.
2. **Distractors are machine-translated.** They are easier to tell apart from native Wikipedia text than a real corpus would be.
3. **Lexical overlap is probably high.** Annotators wrote questions while reading the paragraph. This plausibly favours BM25, but I have not tested it.
4. **Many misses are unanswerable out of context.** With the best system on 2,460 passages, 64 of 1,395 questions have no relevant passage in the top 10. Several are context-dependent, for example "ما هي الحدود في الشمال؟" ("what are the borders in the north?") or "متى توفيَ؟" ("when did he die?"). See `results/misses_2000.json`.
5. **Quantized models.** The ONNX weights are int8 (Xenova exports), so scores can differ from the published full-precision models.
6. **One dataset, one gold paragraph per question, no tuning of BM25 parameters or RRF `k`.**
7. **Latency** is single-query CPU encoding on a 2-core cloud VM (onnxruntime 1.23.2), p50 over 50 queries after warm-up: e5-small about 5 ms, e5-base about 15 ms, bge-m3 about 40 ms; BM25 under 1 ms. It was not measured on a laptop or GPU.

## Setup

```bash
pip install -r requirements.txt            # numpy, onnxruntime, tokenizers, pytest
python scripts/download_data.py            # ARCD + Arabic-SQuAD from the SOQAL repo
```

Download the quantized ONNX models into `models/<name>/` (each needs `config.json`,
`tokenizer.json`, `tokenizer_config.json` and `onnx/model_quantized.onnx`):

| Folder | Source repo on Hugging Face |
|---|---|
| `models/e5-small` | `Xenova/multilingual-e5-small` |
| `models/e5-base` | `Xenova/multilingual-e5-base` |
| `models/bge-m3` | `Xenova/bge-m3` |

## Run

```bash
python -m pytest -q tests                                   # 15 tests

# Optional, for slow or time-limited machines: resumable embedding, one model at a time
python scripts/encode_cached.py --model bge-m3 --sizes 0 2000 --budget 100000

python scripts/run_eval.py --arcd data/arcd.json --distractors data/arabic-squad.json \
    --sizes 0 2000 --models-dir models --models e5-small e5-base bge-m3

python scripts/search.py "ما هي عاصمة مصر؟" --model bge-m3 --hybrid -k 5
```

## Method

- **Normalization** (`normalize.py`): strips diacritics and tatweel, unifies alef, ya and ta marbuta. **Light stemming** removes common prefixes (وال، بال، كال، فال، لل، ال، و) and suffixes, keeping stems of at least 3 characters. Stopwords are removed.
- **BM25** (`bm25.py`): Okapi, k1 = 1.2, b = 0.75, title added to the passage text.
- **Dense** (`dense.py`): ONNX Runtime, no PyTorch. E5 uses mean pooling with `query: ` / `passage: ` prefixes; bge-m3 uses CLS pooling with no prefix. Vectors are L2-normalized; scoring is a dot product.
- **Hybrid**: Reciprocal Rank Fusion (k = 60, depth 100) of stemmed BM25 and one dense model.
- **Metrics** (`metrics.py`): Recall@k, MRR@10, nDCG@10 with binary relevance. Confidence intervals resample whole gold paragraphs, because about three questions share each paragraph. Paired differences use the same resampling on per-query differences.
- **Data cleaning** (`data.py`): duplicate paragraphs are merged by normalized hash; stray leading dashes and dangling " ال" are stripped from questions.

## Data and licences

ARCD and Arabic-SQuAD come from the SOQAL repository (Mozannar, El Hajal, Maamary and Hajj, "Neural Arabic Question Answering", WANLP 2019), which carries an MIT licence. This repository downloads them rather than redistributing them. I did not check the licences of the underlying SQuAD or Wikipedia text, or of the model weights; check them before any commercial use.

## Layout

```
src/arabic_search/   normalize, data, bm25, dense, metrics
scripts/             download_data, encode_cached, run_eval, search
tests/               unit tests (15)
results/             results.md, results.json, misses_*.json
```

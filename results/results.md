_Latency: median of one query at a time (batch size 1, CPU, uncached), measured on Linux-6.8.0-138-generic-x86_64-with-glibc2.35, 2 logical CPUs, onnxruntime 1.23.2. Hybrid latency is the sum of its two parts. Confidence intervals: 2000-resample bootstrap over gold paragraphs (questions about the same paragraph are not independent)._

### Corpus: 460 passages (0 distractors)

| Retriever | Recall@1 | Recall@10 | MRR@10 [95% CI] | nDCG@10 | Query ms (p50) |
|---|---:|---:|---:|---:|---:|
| Hybrid RRF (BM25 + bge-m3) | 0.715 | 0.967 | 0.819 [0.800, 0.838] | 0.856 | 39.6 |
| Hybrid RRF (BM25 + multilingual-e5-small) | 0.701 | 0.961 | 0.810 [0.791, 0.830] | 0.848 | 5.4 |
| Hybrid RRF (BM25 + multilingual-e5-base) | 0.687 | 0.960 | 0.801 [0.782, 0.820] | 0.841 | 15.9 |
| BM25 (normalized + light stem) | 0.703 | 0.955 | 0.801 [0.780, 0.821] | 0.839 | 0.1 |
| bge-m3 | 0.650 | 0.940 | 0.771 [0.749, 0.793] | 0.814 | 39.6 |
| multilingual-e5-small | 0.656 | 0.946 | 0.770 [0.749, 0.791] | 0.814 | 5.3 |
| BM25 (normalized) | 0.642 | 0.923 | 0.749 [0.725, 0.773] | 0.792 | 0.1 |
| multilingual-e5-base | 0.600 | 0.939 | 0.736 [0.714, 0.759] | 0.787 | 15.8 |

Paired difference in MRR@10 on the same corpus (A - B):

| A | B | Δ MRR@10 | 95% CI | CI excludes 0 |
|---|---|---:|---:|:---:|
| BM25 (normalized) | BM25 (normalized + light stem) | -0.052 | [-0.068, -0.037] | yes |
| multilingual-e5-small | BM25 (normalized + light stem) | -0.031 | [-0.051, -0.009] | yes |
| Hybrid RRF (BM25 + multilingual-e5-small) | BM25 (normalized + light stem) | +0.009 | [-0.004, +0.023] | no |
| multilingual-e5-base | BM25 (normalized + light stem) | -0.065 | [-0.088, -0.040] | yes |
| Hybrid RRF (BM25 + multilingual-e5-base) | BM25 (normalized + light stem) | +0.001 | [-0.015, +0.015] | no |
| bge-m3 | BM25 (normalized + light stem) | -0.030 | [-0.053, -0.006] | yes |
| Hybrid RRF (BM25 + bge-m3) | BM25 (normalized + light stem) | +0.018 | [+0.003, +0.034] | yes |
| Hybrid RRF (BM25 + bge-m3) | Hybrid RRF (BM25 + multilingual-e5-small) | +0.009 | [-0.003, +0.021] | no |
| Hybrid RRF (BM25 + bge-m3) | bge-m3 | +0.048 | [+0.032, +0.065] | yes |
| multilingual-e5-small | multilingual-e5-base | +0.034 | [+0.016, +0.051] | yes |

### Corpus: 2,460 passages (2,000 distractors)

| Retriever | Recall@1 | Recall@10 | MRR@10 [95% CI] | nDCG@10 | Query ms (p50) |
|---|---:|---:|---:|---:|---:|
| Hybrid RRF (BM25 + bge-m3) | 0.698 | 0.954 | 0.801 [0.779, 0.821] | 0.839 | 40.6 |
| Hybrid RRF (BM25 + multilingual-e5-base) | 0.668 | 0.945 | 0.780 [0.760, 0.801] | 0.821 | 14.8 |
| Hybrid RRF (BM25 + multilingual-e5-small) | 0.666 | 0.946 | 0.780 [0.758, 0.801] | 0.821 | 5.5 |
| BM25 (normalized + light stem) | 0.665 | 0.933 | 0.771 [0.748, 0.793] | 0.811 | 0.3 |
| bge-m3 | 0.634 | 0.920 | 0.752 [0.730, 0.773] | 0.794 | 40.4 |
| multilingual-e5-small | 0.627 | 0.909 | 0.740 [0.717, 0.763] | 0.782 | 5.2 |
| BM25 (normalized) | 0.616 | 0.896 | 0.721 [0.696, 0.745] | 0.764 | 0.1 |
| multilingual-e5-base | 0.575 | 0.900 | 0.707 [0.684, 0.732] | 0.755 | 14.5 |

Paired difference in MRR@10 on the same corpus (A - B):

| A | B | Δ MRR@10 | 95% CI | CI excludes 0 |
|---|---|---:|---:|:---:|
| BM25 (normalized) | BM25 (normalized + light stem) | -0.050 | [-0.066, -0.034] | yes |
| multilingual-e5-small | BM25 (normalized + light stem) | -0.030 | [-0.052, -0.007] | yes |
| Hybrid RRF (BM25 + multilingual-e5-small) | BM25 (normalized + light stem) | +0.009 | [-0.007, +0.024] | no |
| multilingual-e5-base | BM25 (normalized + light stem) | -0.064 | [-0.089, -0.038] | yes |
| Hybrid RRF (BM25 + multilingual-e5-base) | BM25 (normalized + light stem) | +0.009 | [-0.008, +0.027] | no |
| bge-m3 | BM25 (normalized + light stem) | -0.019 | [-0.043, +0.006] | no |
| Hybrid RRF (BM25 + bge-m3) | BM25 (normalized + light stem) | +0.030 | [+0.013, +0.047] | yes |
| Hybrid RRF (BM25 + bge-m3) | Hybrid RRF (BM25 + multilingual-e5-small) | +0.021 | [+0.009, +0.033] | yes |
| Hybrid RRF (BM25 + bge-m3) | bge-m3 | +0.049 | [+0.032, +0.067] | yes |
| multilingual-e5-small | multilingual-e5-base | +0.034 | [+0.016, +0.051] | yes |

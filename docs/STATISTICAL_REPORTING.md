# Accuracy and questions with initial disagreement

Full-benchmark accuracy scores the final option for every question; missing or invalid final options and technical failures count as incorrect.

The initial-disagreement cohort comprises questions with shared valid initial answers and at least two distinct initial options: eight answers in MA-RAG-int and 17 in MedLA. Methods are compared on the same question IDs. Missing or invalid final answers on these questions count as incorrect.

| Evaluation | Questions | Baseline correct | EDR correct | Accuracy, baseline → EDR | Gain (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MedXpertQA / Qwen | 363 | 162 | 189 | 44.63% → 52.07% | +7.44 |
| MedXpertQA / Gemma initial | 351 | 52 | 71 | 14.81% → 20.23% | +5.41 |
| MedDDx-Expert / Gemma | 468 | 125 | 140 | 26.71% → 29.91% | +3.21 |
| MMLU-Med / Gemma | 767 | 465 | 501 | 60.63% → 65.32% | +4.69 |
| MedXpertQA / Gemma seed 2026091900 | 341 | 62 | 73 | 18.18% → 21.41% | +3.23 |
| MedXpertQA / Gemma seed 2026091909 | 356 | 64 | 73 | 17.98% → 20.51% | +2.53 |

A correction is a baseline-incorrect/EDR-correct outcome; a regression is a baseline-correct/EDR-incorrect outcome. The correction rate divides corrections by baseline errors in the same cohort. Method comparisons use the same initial-disagreement cohort, and Gemma runs are summarized separately.

```bash
python -B tools/summarize_initial_disagreement.py
python -B tools/summarize_initial_disagreement.py --check
```

[Results](../analysis/initial_disagreement_metrics.json) include question IDs, accuracy, corrections and regressions, paired bootstrap intervals, and two-sided exact McNemar tests, together with fixed-allocation, random-allocation and MAD-M²(O) comparisons. Full-benchmark results and candidate-evolution definitions are in [Unified metrics](SYSTEMATIC_METRICS.md).

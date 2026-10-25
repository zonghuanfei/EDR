# Gemma Multi-Run Evaluation

English | [中文](README.zh-CN.md)

Three complete 490-question runs evaluate EDR using independently generated
initial materials. Each pair shares its valid initial responses and inference
configuration. The initial run and two recorded seeds are listed separately.
Complete-benchmark scores use original final predictions.

| Run | MA-RAG-int | EDR | Full gain | Initial-disagreement questions |
| --- | ---: | ---: | ---: | ---: |
| Initial run | 74/490 | 93/490 | **+19; +3.88 pp** | 52/351 -> 71/351; **+5.41 pp** |
| Seed 2026091900 | 84/490 | 95/490 | **+11; +2.24 pp** | 62/341 -> 73/341; **+3.23 pp** |
| Seed 2026091909 | 84/490 | 93/490 | **+9; +1.84 pp** | 64/356 -> 73/356; **+2.53 pp** |

EDR improves accuracy in all three runs. The mean complete-benchmark gain is
**13 correct answers (+2.65 percentage points)**. The initial-disagreement cohorts
are defined separately from each run's shared valid initial answers and contain
351, 341 and 356 questions, respectively.

Full results and per-run statistics are in the corresponding `results/`
directories and [the unified analysis](../../../../docs/SYSTEMATIC_METRICS.md).
Question IDs and paired statistics are in [Initial-disagreement results](../../../../analysis/initial_disagreement_metrics.json). Recompute from the package root with `python -B tools/summarize_initial_disagreement.py`.

# Gemma Multi-Run Evaluation

English | [中文](README.zh-CN.md)

Three complete 490-question runs evaluate EDR using independently generated
initial materials. Each pair shares its valid initial responses and inference
configuration. The initial run and two recorded seeds are listed separately.
Complete-benchmark scores use original final predictions.

| Run | MA-RAG-int | EDR | Full gain | Routing-active subset |
| --- | ---: | ---: | ---: | ---: |
| Initial run | 74/490 | 93/490 | **+19; +3.88 pp** | 33/225 -> 51/225; **+8.00 pp** |
| Seed 2026091900 | 84/490 | 95/490 | **+11; +2.24 pp** | 46/234 -> 57/234; **+4.70 pp** |
| Seed 2026091909 | 84/490 | 93/490 | **+9; +1.84 pp** | 44/226 -> 54/226; **+4.42 pp** |

EDR improves accuracy in all three runs. The mean complete-benchmark gain is
**13 correct answers (+2.65 percentage points)**. Pooling active question-run
observations gives 123/685 for baseline and 162/685 for EDR, a **5.69-point** gain.

Full results and per-run statistics are in the corresponding `results/`
directories and [the unified analysis](../../../../docs/SYSTEMATIC_METRICS.md).
`results/routing_subset_analysis.json` summarizes full-benchmark and routing-active outcomes.

# Accuracy and routing-active subset

Full-benchmark accuracy uses the final option for every question. Missing or invalid final options and technical failures count as incorrect.

A question is routing-active when at least one post-initial EDR round assigns one or more responses to the independent branch (`independent > 0`). The active-subset comparison uses both conditions' recorded final outcomes on those same question IDs.

| Evaluation | Total questions | Full correct: baseline → EDR | Active questions | Active correct: baseline → EDR |
| --- | ---: | ---: | ---: | ---: |
| MedXpertQA / Qwen | 490 | 209 → 236 | 167 | 68 → 92 |
| MedXpertQA / Gemma initial | 490 | 74 → 93 | 225 | 33 → 51 |
| MedXpertQA / Gemma seed 2026091900 | 490 | 84 → 95 | 234 | 46 → 57 |
| MedXpertQA / Gemma seed 2026091909 | 490 | 84 → 93 | 226 | 44 → 54 |

```bash
python -B tools/build_routing_subset_analysis.py --check
```

The corresponding `results/routing_subset_analysis.json` files contain the full-benchmark and routing-active summaries. Cross-framework results and candidate-evolution definitions are in [Unified EDR metrics](SYSTEMATIC_METRICS.md).

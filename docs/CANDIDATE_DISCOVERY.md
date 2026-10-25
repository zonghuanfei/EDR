# Correct candidate discovery after the initial round

English | [中文](CANDIDATE_DISCOVERY.zh-CN.md)

EDR expands the availability of correct options after an initially unsuccessful
response pool. This analysis follows the observed trajectories, complementing
analyses of candidate retention and final answer accuracy.

## Definition

The primary cohort contains questions with eight valid, matched first-round
choices in baseline and EDR, initial disagreement, and no correct initial
choice. A discovery occurs when any response in a later round selects the
gold option. Each question is counted once, even if that option subsequently
disappears and reappears. Discovery is distinct from final answer correctness.

The denominator is the primary cohort size, determined from the shared initial
choices. Invalid or unmatched initial pools are excluded from this diagnostic,
not from the original full-benchmark scores. Accuracy on questions with initial disagreement is defined in [Statistical reporting](STATISTICAL_REPORTING.md).

| Evaluation | Primary cohort | Baseline discoveries | EDR discoveries |
| --- | ---: | ---: | ---: |
| MedXpertQA / Qwen | 100 | 5 (5.00%) | 16 (16.00%) |
| MedXpertQA / Gemma initial | 214 | 10 (4.67%) | 45 (21.03%) |
| MedXpertQA / Gemma 2026091900 | 202 | 9 (4.46%) | 43 (21.29%) |
| MedXpertQA / Gemma 2026091909 | 218 | 12 (5.50%) | 49 (22.48%) |

In the three Gemma runs, the first correct choice appears exclusively in the
independent branch on 34, 30, and 41 questions, respectively. The files retain
separate categories for discussion-only, both branches, and rounds without a
branch label.

On Qwen MedXpertQA, gated random allocation discovers the correct option on
17 of the same 100 questions, compared with 16 for EDR. The discovered option
becomes the final answer on seven and twelve questions, respectively. The
full-benchmark correct counts are 216 and 236. Discovery and final selection
thus describe distinct parts of collaborative refinement.

## Files and reconstruction

Each of the four MedXpertQA evaluation directories contains
`results/candidate_discovery.json`. It includes cohort sizes, discovery rates,
question IDs, first-discovery rounds and channels, cumulative discovery counts,
final candidate presence, and final correctness. Qwen also includes gated random allocation with seed 42.
No model is queried by the reconstruction command:

```bash
python tools/analyze_candidate_discovery.py --experiment medxpertqa_qwen
```

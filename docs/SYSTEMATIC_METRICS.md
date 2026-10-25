# Unified EDR Metrics Across Frameworks

This analysis connects final accuracy on the full benchmark and on questions
with initial disagreement to candidate evolution using the retained trajectories. It covers
the four primary dataset/model/framework conditions and the two additional
Gemma MedXpertQA runs, as well as the MAD-M² comparisons and gated-random
allocation ablation.

## Files and reproduction

- [`../analysis/systematic_edr_metrics.json`](../analysis/systematic_edr_metrics.json):
  aggregate metrics, per-question flags, event IDs.
- [`../tools/analyze_systematic_metrics.py`](../tools/analyze_systematic_metrics.py):
  offline analysis; requires Python and NumPy.
- [`../tools/test_systematic_metrics.py`](../tools/test_systematic_metrics.py):
  focused tests for event definitions and paired statistics.

Run from the package root:

```bash
python -B tools/analyze_systematic_metrics.py --output analysis/systematic_edr_metrics.json
python -B tools/test_systematic_metrics.py
```

The script reads only package files. It verifies question coverage, shared
initial choices, gold labels and agreement with the original score files.
The output contains no corpus passages or long generated answers.

## Evaluation units and denominators

| Metric | Evaluation set | Definition |
| --- | --- | --- |
| Full accuracy and gain | Every released question in the run | Score the original final option against gold; a missing final option is incorrect. Gain is EDR accuracy minus baseline accuracy in percentage points. |
| Relative accuracy gain | Full run | Increase in correct answers divided by baseline correct answers. |
| Accuracy and gain with initial disagreement | D: shared valid initial answers with at least two distinct options | Compare final outcomes on the same question IDs; missing final options count as incorrect. |
| Correction rate with initial disagreement | Baseline errors within D | Baseline-incorrect/EDR-correct questions divided by all baseline errors in D. |
| Correct-candidate loss | D with gold present initially and valid terminal medical-answer pools in both arms | Fraction whose terminal pool no longer contains gold, separately for baseline and EDR. |
| Subsequent discovery | D with gold absent initially | Fraction with gold in at least one valid subsequent medical-answer round. Each question counts once. |
| Early incorrect consensus | All of D | Wrong unanimous termination before the host's round cap: answer round 8 in MA-RAG, discussion round 3 in MedLA. |

D contains questions with the same valid initial choices in both conditions
and at least two distinct legal choices. It excludes initially unanimous
questions and incomplete initial materials. Cohort membership is determined
from the shared initial answers.

A terminal pool is the final recorded medical-answer pool, rather than the
most recent earlier complete round. An incomplete terminal pool or technical failure is excluded
from candidate-loss comparisons. In MedLA, a valid initial-only trajectory uses
its initial pool as the terminal pool. Discovery can still be recorded from
an earlier valid round when a later technical failure prevents completion.

[Initial-disagreement results](../analysis/initial_disagreement_metrics.json) summarize final accuracy, paired outcome changes and method comparisons on D. Recompute them with `python -B tools/summarize_initial_disagreement.py`.

## Complete-run results

| Run | N | Baseline correct | EDR correct | Gain (pp) | Paired 95% CI (pp) |
| --- | ---: | ---: | ---: | ---: | --- |
| MedXpertQA / Qwen | 490 | 209 | 236 | +5.51 | [3.06, 8.16] |
| MedXpertQA / Gemma initial | 490 | 74 | 93 | +3.88 | [1.02, 6.73] |
| MedXpertQA / Gemma seed 2026091900 | 490 | 84 | 95 | +2.24 | [-0.41, 4.90] |
| MedXpertQA / Gemma seed 2026091909 | 490 | 84 | 93 | +1.84 | [-1.43, 5.10] |
| MedDDx-Expert / Gemma | 483 | 135 | 150 | +3.11 | [0.41, 5.80] |
| MMLU-Med / Gemma | 1089 | 766 | 802 | +3.31 | [1.93, 4.68] |

Confidence intervals are percentile intervals from 20,000 paired-question
bootstrap resamples. Multinomial sampling of the per-question correctness
differences (-1, 0, +1) is equivalent to resampling paired questions for this
accuracy-difference statistic. The analysis uses statistical seed 20261003
to make interval calculation reproducible; this is not a generation seed.
The JSON also reports exact two-sided McNemar tests.

## Shared mechanism evidence

| Primary condition | Candidate loss, baseline -> EDR (n) | Discovery, baseline -> EDR (n) | Early incorrect consensus, baseline -> EDR (D) |
| --- | --- | --- | --- |
| MedXpertQA / Qwen | 103 -> 81 (262) | 5 -> 16 (100) | 191 -> 161 (363) |
| MedXpertQA / Gemma initial | 84 -> 67 (133) | 10 -> 45 (214) | 294 -> 217 (351) |
| MedDDx-Expert / Gemma | 292 -> 263 (422) | 1 -> 3 (46) | 326 -> 263 (468) |
| MMLU-Med / Gemma | 235 -> 199 (684) | 28 -> 42 (83) | 272 -> 219 (767) |

All cells above show event counts, followed by their common comparison
denominator. The JSON also gives rates and question IDs. Candidate loss,
discovery and early termination improve in the same direction across the four
primary conditions. Gemma's initial MedXpertQA discovery rate rises from
4.67% to 21.03%; MMLU-Med rises from 33.73% to 50.60%.

Early-consensus metrics above use D, giving 326/263 for MedDDx and 272/219
for MMLU. Per-benchmark analysis JSON files additionally retain totals that
include initial unanimous stops: 327/264 and 292/239, respectively.

Logical solver depth counts recorded medical-answer stages, including a
retained partial stage. Shared stages count once in each arm. MedLA's initial
stage counts once, with subsequent discussion stages counted separately.
Probability probes and final judges are not medical-answer stages. Retrieval
counts represent logical retrieval stages, not retry attempts; MedLA uses one
shared retrieval per question.

## Module comparisons and allocation ablation

The two MAD-M2 comparisons use original complete-run outcomes:

- MA-RAG / MedXpertQA: baseline 74, MAD-M2(O) 78, EDR 93 correct out of 490.
- MedLA / MedDDx-Expert: baseline 135, MAD-M2(O) 145, EDR 150 out of 483.

The Qwen allocation comparison uses all 490 question IDs. EDR scores
236/490 (48.16%), compared with 216/490 (44.08%) for gated random allocation
with allocation seed 42. EDR is uniquely correct on 26 questions and random
allocation on six, yielding a 4.08-point difference (exact two-sided McNemar
`p = 0.00053505`).

EDR generates 1,286 independent responses over 861 subsequent rounds;
random allocation generates 1,701 over 969. These correspond to 24.4% fewer
independent responses and 11.1% fewer subsequent rounds for EDR. The counts
exclude the common initial answer round and do not measure total token cost.

The policies share the initial answers, ordered first retrieval and history
order, then compute their probes and generate subsequent responses separately.
Random allocation applies the same positive-route rule on its own trajectory.
The JSON retains full paired scores, uncertainty and allocation summaries.

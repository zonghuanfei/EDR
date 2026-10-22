# MMLU-Med: EDR Transfer to MedLA

English | [中文](README.zh-CN.md)

This directory contains the complete **1,089-question MMLU-Med medical test**,
comparing MedLA with a shared MedCorp evidence interface against the same
framework plus EDR. The model is **Gemma 3 12B IT**, BF16. Both conditions
share all 17 initial materials and the ordered retrieval evidence. The run
completed on 3 October 2026. Dataset identity is recorded in [DATASET.md](DATASET.md).

## Main Results

| Condition | Correct / total | Accuracy |
| --- | ---: | ---: |
| MedLA + shared evidence | 766 / 1,089 | 70.34% |
| MedLA + shared evidence + EDR | 802 / 1,089 | 73.65% |
| EDR gain | +36 | **+3.31 percentage points** |

EDR repairs **48** baseline errors and introduces **12** harms. All 48 repairs
actually activate the independent branch. The relative accuracy increase is
4.70%. On routing-inactive questions, baseline and EDR endpoints are identical,
with request-validated common-prefix replay until the first history removal.

The paired-question bootstrap 95% interval is **[+1.93, +4.68] percentage
points**; the exact two-sided McNemar p-value is **3.18e-6**. The analysis uses
20,000 question-level resamples, with its statistical resampling seed stored
in `results/analysis_1089.json`.

Question 78 has a shared initial-stage technical failure. It has no final
prediction in either condition and counts as
incorrect in both. All 1,089 IDs are retained; 1,088 have complete paired traces.

## Intervention and Continued Reasoning

| Group | Questions | Baseline correct | EDR correct | Gain |
| --- | ---: | ---: | ---: | ---: |
| Full benchmark | 1,089 | 766 | 802 | +3.31 pp |
| EDR-active | 264 | 111 | 147 | **+13.64 pp** |
| Valid, routing-inactive | 824 | 655 | 655 | 0.00 pp |

Among valid initial materials, 767 questions have disagreement and 321 are
unanimous. EDR activates on 264 questions, or 24.26% of complete pairs, and
makes **560 history-masked generations**, with 3,189 three-view option probes.
Per-call replay counts are in [results/execution_audit.json](results/execution_audit.json).

| Trajectory measure | Baseline | EDR |
| --- | ---: | ---: |
| Subsequent discussion rounds, total | 953 | 1,063 |
| Discussion rounds / benchmark question | 0.88 | 0.98 |
| Initial + discussion stages / question | 1.88 | 1.98 |
| Early incorrect consensus within initial disagreement | 272 | **219** |
| Questions reaching the final judge | 24 | 55 |
| Retrieval stages / question | 1.00 | 1.00 |

Early incorrect consensus means a wrong unanimous endpoint before the third
discussion round among the 767 shared initial-disagreement questions.
Its count decreases by 53 questions, or 19.49%. With retrieval held constant, EDR preserves an
independent reasoning channel that keeps incorrect consensus from prematurely
ending correction. Twelve repaired questions initially contain no correct
choice among the 17 answers; the correct choice appears later. Another
36 repairs recover a choice already present initially, repairing 14.75%
of the 244 baseline errors with initial gold-option coverage.

Discussion counts exclude option-probe and final-judge calls. The denominator
for mean stages is all 1,089 questions, including the attempted initial stage
of the technical failure. The full-record analysis stores the ID lists.

## Token Consumption

Gemma input, output and total tokens are extracted from the model-service usage receipts for this complete 1089-question MedLA comparison. Initial medical reasoning and logic checks, representative responses, probability probes and final adjudication are included. Shared or replayed responses are charged to both conditions. All token-bearing recorded calls are counted.

| Condition | Input (million) | Output (million) | Total (million) |
| --- | ---: | ---: | ---: |
| MedLA | 245.880 | 35.165 | 281.045 |
| MedLA + EDR | 292.180 | 35.061 | 327.241 |

EDR increases accuracy by **3.31 percentage points** with **16.44% more total tokens**. Output tokens decrease by **0.30%**. History-masked representative contexts partly offset the input-token overhead of the three-context probes.

[Aggregate, phase-level and per-question token counts](results/token_usage_1089.json) are saved separately from the reasoning traces. Table values are rounded to three decimal places.

## Execution Environment

| Setting | Value |
| --- | --- |
| Solver and probability model | Gemma 3 12B IT, `gemma-3-12b-it` |
| Precision | BF16, no quantization |
| Medical agents | 17 initial agents; three eliminations and a native logic check per agent |
| Discussion | Up to three rounds; native representative per distinct answer; native final judge |
| Generation | Temperature 0.7; maximum 8,000 output tokens |
| Examples | Native one-shot examples from a 100-item bank |
| Retrieval | One shared question-only stage; BM25 32, MedCPT rerank, six documents |
| Evidence block | Document title and up to 1,200 content characters per document |

The model, generation and retrieval settings above describe the experiment.
Serving deployment details are not part of this release.

The EDR formula, N-representative allocation, shared-prefix replay and native
MedLA workflow are the same as the MedDDx-Expert transfer. See
[EDR integration](../../docs/INTEGRATION.md) and [Methods](../../docs/METHODS.md).

## Included Files

| File | Contents |
| --- | --- |
| `results/final_scores_1089.json` | Both scores and all paired predictions/correctness flags |
| `results/analysis_1089.json` | Repair/harm IDs, active-subset metrics, candidate coverage, depth and paired statistics |
| `results/execution_audit.json` | Raw-request verification of masks, common-prefix replay and four-option probes |
| `results/technical_failures.json` | Shared initial failure and its scoring status |
| `traces/shared_initial_1089.jsonl` | Initial medical reasoning and logic reports once per question; document IDs/titles/scores |
| `traces/medla_shared_evidence_1089.jsonl` | Full generated baseline representative and judge answers |
| `traces/medla_shared_evidence_plus_edr_1089.jsonl` | Full generated answers plus E/H/EH probabilities and actual branch allocation |

Every JSONL has 1,089 records. Shared initial references avoid duplicating the
17 starting histories.

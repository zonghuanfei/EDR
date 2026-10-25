# MedDDx-Expert: EDR Transfer to MedLA

English | [中文](README.zh-CN.md)

This directory contains the complete 483-question evaluation of **MedLA with
a shared MedCorp evidence interface** and **the same framework plus EDR**,
using Gemma 3 12B IT. The experiment was completed on 2 October 2026.
The released Expert/hard split was evaluated in full, with no question selection.

## Main Results

| Condition | Correct / total | Accuracy |
| --- | ---: | ---: |
| MedLA + shared evidence | 135 / 483 | 27.95% |
| MedLA + shared evidence + EDR | 150 / 483 | 31.06% |
| EDR gain | +15 | **+3.11 percentage points** |

EDR increases accuracy by **11.11% relative to the baseline**. Paired outcome
changes comprise **30 corrections and 15 answer regressions**.

The full-benchmark paired-bootstrap 95% confidence interval for the gain is
**[+0.41, +5.80] percentage points**; the exact two-sided McNemar test gives
**p = 0.0357**. The statistical unit is one paired question. The bootstrap uses
20,000 question-level resamples; its resampling seed is recorded separately
from generation seeds in `results/analysis_483.json`.

Four questions (IDs 63, 261, 307 and 314) have shared initial-stage technical
failures. They have
no final prediction in either condition and count as incorrect in both.
Thus the headline denominator is 483, including all four failures; 479
questions have complete paired reasoning trajectories.

## Final accuracy on questions with initial disagreement

Of the 479 questions with valid shared initial answers, 468 contain at least two distinct
options and 11 are unanimous. Final outcomes are compared on the same initial-disagreement cohort.

| Evaluation group | Questions | Baseline correct | EDR correct | Accuracy gain |
| --- | ---: | ---: | ---: | ---: |
| Full benchmark | 483 | 135 | 150 | +3.11 pp |
| Initial disagreement | 468 | 125 | 140 | **+3.21 pp** |
| Initial unanimity | 11 | 10 | 10 | 0.00 pp |

On the initial-disagreement questions, accuracy increases from 26.71% to 29.91%.
EDR corrects 30 of 343 baseline errors (8.75%), with 15 answer regressions.

Question IDs and paired statistics are in [Initial-disagreement results](../../analysis/initial_disagreement_metrics.json). Recompute from the package root with `python -B tools/summarize_initial_disagreement.py`.

EDR generates 804 history-masked responses and makes 2,346 three-view probability probes.
Per-call counts are in [results/execution_audit.json](results/execution_audit.json).

## Mechanism Evidence

| Trajectory measure | Baseline | EDR |
| --- | ---: | ---: |
| Discussion rounds, total | 590 | 782 |
| Discussion rounds per benchmark question | 1.22 | 1.62 |
| Initial + discussion stages per question | 2.22 | 2.62 |
| Early incorrect consensus within initial disagreement | 326 | **263** |
| Questions reaching the final judge | 9 | 48 |
| Retrieval stages per question | 1.00 | 1.00 |

Early incorrect consensus is a wrong unanimous answer reached before the
third discussion round among the 468 shared initial-disagreement questions.
It decreases by **63 questions (19.33%)**. Together with increased discussion depth,
this supports EDR's mechanism: preserving evidence-based independence keeps
incorrect consensus from prematurely closing the reasoning process.
Both conditions use the same single retrieval stage and six documents;
the difference in reasoning depth occurs within the existing discussion limit.
The initial + discussion measure excludes probability probes and final-judge
calls. Its denominator is all 483 questions, including attempted initial
stages on technical failures.

Among 297 baseline errors whose initial 17 answers already contained the
correct option, EDR repairs 29 (**9.76%**). The remaining repair, question 21,
introduces the correct option after it was absent from all initial answers.
These two patterns demonstrate both recovery of initially available correct
candidates and subsequent discovery of a correct choice.

| Case ID | Gold | Baseline trajectory | EDR trajectory |
| --- | --- | --- | --- |
| 2 | C | Three discussion rounds, then judge selects B | Second round uses one independent representative; C/C consensus |
| 21 | D | Initial choices contain only A/B/C; first discussion ends A/A/A | Two independent representatives introduce D in round 1; D/D consensus in round 2 |
| 85 | D | C/A/C/A, then A/A | D/A/C/A, then D/D/A, then D/D; independence is used in all three rounds |

All case IDs follow the upstream zero-based question index. Full generated
reasoning is retained in the shared-initial file and the two condition traces.

## Token Consumption

Gemma input, output and total tokens are extracted from the model-service usage receipts for this complete 483-question MedLA comparison. Initial medical reasoning and logic checks, representative responses, probability probes and final adjudication are included. Shared or replayed responses are charged to both conditions. All token-bearing recorded calls are counted.

| Condition | Input (million) | Output (million) | Total (million) |
| --- | ---: | ---: | ---: |
| MedLA | 140.114 | 18.387 | 158.501 |
| MedLA + EDR | 163.372 | 18.274 | 181.646 |

EDR increases accuracy by **3.11 percentage points** with **14.60% more total tokens**. Output tokens decrease by **0.61%**. History-masked representative contexts partly offset the input-token overhead of the three-context probes.

[Aggregate, phase-level and per-question token counts](results/token_usage_483.json) are saved separately from the reasoning traces. Table values are rounded to three decimal places.

## Execution Environment

| Setting | Value |
| --- | --- |
| Solver and probability model | Gemma 3 12B IT, served as `gemma-3-12b-it` |
| Precision | BF16, no quantization |
| Generation | Temperature 0.7; maximum 8,000 output tokens |
| Initial reasoning | 17 agents; three sequential eliminations each; native initial logic checks |
| Discussion | At most three rounds, one native representative per distinct answer |
| Logic examples | Native one-shot examples from a 100-item example bank |
| Retrieval | MedCorp; BM25 recall 32; MedCPT rerank; keep six documents |
| Evidence input | Document title and at most 1,200 content characters per document |

The model, generation and retrieval settings above describe the experiment.
Serving deployment details are not part of this release.

## Framework and EDR Integration

The released MedLA elimination, logic-checking, representative selection,
rebuttal and final-judge functions are reused directly from the pinned Python
source. Both arms receive the same per-question evidence block in all native
model roles. Retrieval uses the original question only, before initial
reasoning; the resulting ordered evidence is shared throughout that question.
Initial materials are generated once and shared. Later identical requests are
replayed from the baseline until EDR first removes history.

At each representative discussion round, EDR probes evidence-only (E),
history-only (H) and combined (EH) distributions over the original A-D options.
The original top-choice-change gate and total-variation score are retained:

```text
TV_E = 0.5 * sum_a |p_EH(a) - p_E(a)|
TV_H = 0.5 * sum_a |p_EH(a) - p_H(a)|
s = max(0, TV_E - TV_H)
m8 = 0 if the E/EH top choices agree or have tied peaks;
     otherwise min(6, floor(6*s + 0.5))
m = min(N - 1, floor(N*m8/8 + 0.5))
```

N is the actual number of selected representatives, normally two to four.
The first m representatives in native order receive the question, original
options, current candidate labels and evidence, without their own previous
answer/reasoning or peer reasoning/logic reports. The other representatives
retain native discussion context. The current candidate list is a prompt
input; the native answer parser retains the original legal A-D answer space.
Representative pruning, unanimity stopping and final arbitration remain
under MedLA's native control. Initial logic reports are the original reports,
not additional checks regenerated after every round.

The common HTTP/format adapters are identical in both arms. The public adapter
reuses `src/edr.py`; the complete upstream framework is fetched separately.
See [integration instructions](../../docs/INTEGRATION.md).

## Included Files

| File | Contents |
| --- | --- |
| `results/final_scores_483.json` | Aggregate scores and all 483 paired final predictions/correctness flags |
| [Initial-disagreement results](../../analysis/initial_disagreement_metrics.json) | Paired accuracy, corrections and method comparisons on initial-disagreement questions |
| `results/analysis_483.json` | Repair/harm IDs, reasoning depth, candidate coverage and paired statistics |
| `results/execution_audit.json` | Raw-request verification of actual masks, common-prefix replay and option probes |
| `results/technical_failures.json` | The four shared initial failures and their scoring/retry status |
| `traces/shared_initial_483.jsonl` | Initial 17 accepted elimination reasonings and logic reports, plus shared retrieval IDs/titles/scores; partial outputs for failures |
| `traces/medla_shared_evidence_483.jsonl` | Complete generated representative answers, option changes and terminal decisions |
| `traces/medla_shared_evidence_plus_edr_483.jsonl` | The same, with actual branch allocation, probability distributions and four-label probe log probabilities |

Each JSONL has 483 records. Condition trajectories refer to the shared initial
record by ID, avoiding duplicate storage of the 17 initial reasoning histories.
Final-judge text and earlier discussion-format outputs are retained when present.
Dataset question files and model weights remain upstream resources.

## Objective History-Masking Comparison

The complete [MAD-M2 module-transfer comparison](madmm_objective/README.md)
adds **145/483 (30.02%)** for MedLA + objective history masking, against
**135/483 (27.95%)** for MedLA and **150/483 (31.06%)** for MedLA + EDR.
All three conditions share the same initial materials and retrieval evidence.
Relative to MAD-M2, EDR gains **1.04 percentage points**; MAD-M2 itself gains
**2.07 points** over the baseline. On the 468 initial-disagreement questions,
objective masking answers 135 correctly (28.85%) and EDR answers 140 (29.91%),
a 1.07-point difference. Its 483 generated trajectories, screening
scores and masks are stored in the child directory. The original baseline/EDR
results and traces above are unchanged.

## Sources

- [MedLA repository](https://github.com/alexander2618/MedLA), commit `5c12cfc8d67170b1f4b131b9c120a54a573c634d`.
- [MedLA paper](https://doi.org/10.1609/aaai.v40i2.37052), AAAI 2026; [open manuscript](https://arxiv.org/abs/2509.23725).
- [Dataset identity and provenance](DATASET.md).

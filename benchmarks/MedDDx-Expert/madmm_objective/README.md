# MedLA + MAD-M2 Objective History Masking

English | [中文](README.zh-CN.md)

This directory contains a complete 483-question **module-transfer comparator**
on MedDDx-Expert with Gemma 3 12B IT. It transfers the official objective
memory-score and selection rules of *Multi-Agent Debate with Memory Masking*
to MedLA's representative discussion rounds. The host is the same MedLA plus
shared MedCorp evidence interface used by the existing baseline and EDR runs.

## Full-Benchmark Results

| Condition | Correct / total | Accuracy | Gain over baseline |
| --- | ---: | ---: | ---: |
| MedLA + shared evidence | 135 / 483 | 27.95% | - |
| MedLA + shared evidence + MAD-M2(O) | 145 / 483 | 30.02% | +2.07 pp |
| MedLA + shared evidence + EDR | 150 / 483 | 31.06% | +3.11 pp |

EDR answers five more questions correctly than objective masking, an observed
**+1.04 percentage-point** difference. Objective masking itself improves on
the baseline: 26 repairs and 16 harms, net +10. Comparing MAD-M2 against EDR
gives 33 repairs and 38 harms, net -5. Every ID is retained in the result file.

There are 479 complete cases and four inherited shared-initial failures
(63, 261, 307, 314). All four count as incorrect in every condition. They
are not regenerated for MAD-M2. There are no new technical failures.

## Final accuracy on questions with initial disagreement

The three methods are compared on the same 468 questions whose shared valid initial answers contain at least two distinct options.

| Condition | Correct / questions | Accuracy |
| --- | ---: | ---: |
| MedLA | 125 / 468 | 26.71% |
| MedLA + MAD-M²(O) | 135 / 468 | 28.85% |
| MedLA + EDR | 140 / 468 | 29.91% |

EDR exceeds MAD-M²(O) by 1.07 percentage points.

Question IDs and paired statistics are in [Initial-disagreement results](../../../analysis/initial_disagreement_metrics.json). Recompute from the package root with `python -B tools/summarize_initial_disagreement.py`.

## Transferred Mechanism

The score follows the official `src/models.py` implementation:

```text
score = exp(mean(finite rank-1 token log probabilities))
threshold = sorted(scores, reverse=True)[n // 2]
keep_i = score_i > threshold
```

The official variable is named `perplexity`, but the implemented quantity is
rank-1 geometric confidence; higher scores are retained. The released paper
describes keeping the single lowest-perplexity response. Its default
three-agent setup normally also retains one response under the released
strict-middle rule. This transfer follows **the released code**, giving
17 -> 8, 4 -> 2, 3 -> 1 and 2 -> 1 without ties; ties are not filled.

For MedLA's initial packet, the three accepted medical-elimination responses
are scored together using a token-weighted log average. Initial logic reports
remain attached to retained peer packets but are not included in the score.
Initial medical responses are scored by teacher forcing their decoded texts
under the original messages, including one end-of-turn token. Subsequent medical
generations request `logprobs` with `top_logprobs=1`.

Before each discussion round, low-score **own and peer histories** are masked
from the medical representatives' input. All current candidate labels and the
shared evidence remain. Native representative selection, elimination rules,
stopping and final judging are retained. The final judge is unmasked.
The same generation seeds and temperature apply to corresponding calls.

## Actual Execution

| Measure | Count |
| --- | ---: |
| Questions with initial disagreement / masking | 468 |
| Masking rounds | 583 |
| Actual history-filtered medical generation calls | 1,766 |
| Initial-history teacher-forced scoring calls | 23,868 |
| First mask, 17 -> 8 | 468 |
| Subsequent masks, 4 -> 2 / 3 -> 1 / 2 -> 1 | 46 / 45 / 24 |

## Environment and Files

The model is Gemma 3 12B IT in BF16, with temperature 0.7 and an
8,000-token output limit. Initial responses and the six-document MedCorp
evidence block are shared with the baseline.

- `results/final_scores_483.json`: all three scores, 483 paired endpoints and repair/harm IDs.
- `results/execution_audit.json`: full raw-request audit.
- `results/technical_failures.json`: inherited failures and zero new failures.
- `traces/medla_shared_evidence_plus_madmm_objective_483.jsonl`: 483 records with full generated representative/judge answers, per-round scores, masks and actual visible-history metadata.

Shared initial text is in [`../traces/shared_initial_483.jsonl`](../traces/shared_initial_483.jsonl).
Baseline and EDR traces are in the parent directory. The shared reference
inside each comparator trace is relative to its `traces/` directory.

See [integration instructions](../../../docs/INTEGRATION.md).

## Sources

- [MAD-M2 paper](https://arxiv.org/abs/2603.20215), ICLR 2026.
- [Official implementation](https://github.com/tmlr-group/MAD-MM), commit `f02069add08280b764d059a2f06ca0043aa093e2`.
- [Parent dataset identity](../DATASET.md) and [MedLA transfer](../README.md).

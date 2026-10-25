# MedXpertQA: Evidence-Discussion Routing with Qwen

English | [中文](README.zh-CN.md)

This evaluation uses all 490 MedXpertQA text questions, each with ten options
(A–J). EDR coordinates evidence-only reassessment and history-informed
discussion within eight solver responses per round. Model and inference
settings are in [DATASET.md](DATASET.md) and [configuration.json](configuration.json).

## Final-answer results

| Condition | Correct / total | Accuracy | Gain over MA-RAG-int |
| --- | ---: | ---: | ---: |
| Direct model | 147 / 490 | 30.00% | — |
| MA-RAG-int | 209 / 490 | 42.65% | — |
| MA-RAG-int + EDR | **236 / 490** | **48.16%** | **+5.51 pp** |
| Gated random allocation, seed 42 | 216 / 490 | 44.08% | +1.43 pp |

EDR corrects 34 baseline errors and changes seven correct answers to incorrect
ones, yielding 27 additional correct answers. Compared with random allocation,
EDR is uniquely correct on 26 questions and random allocation on six, giving
a 4.08-point advantage. Exact two-sided McNemar tests give `p = 0.00002532`
against baseline and `p = 0.00053505` against random allocation. Full paired
results and confidence intervals are in [comparisons.json](results/comparisons.json).

The direct model answers once without retrieval at temperature 0. Multi-agent
conditions use temperature 0.7. Every missing or tied final answer and every
technical failure is counted incorrect in the 490-question denominator.

## Shared initial state

All 490 questions share the baseline's eight initial answers and entropy
values. The 127 initially unanimous questions stop after round 1. Among the
363 initially disagreeing questions, the EDR and random strategies preserve
the baseline's first-query order, retrieved-document order and descending-
entropy history order. Question 29 retains a failed initial query and is
scored incorrect in both strategy runs. The 362 successful first retrievals
match the baseline's ordered query and document metadata.

After the common initial state, each strategy computes its own E/H/EH probes
and generates its subsequent responses. Gold labels enter only final scoring
and the diagnostic analyses.

## State-dependent allocation

EDR compares option distributions under evidence-only (`E`), history-only
(`H`), and evidence-plus-history (`EH`) contexts. A changed top option and the
relative total-variation shift determine the independent count, from zero to
six; the remaining responses retain history. The total is eight per round.

The gated random ablation retains the same positive-route rule. When its
adaptive reference count is positive, it uniformly selects one to six
independent responses using allocation seed 42; otherwise all eight responses
use history. The local draw is keyed by seed, dataset, question ID and round,
independently of worker scheduling. The allocation seed is not a model-generation
seed. Both policies run the complete collaboration workflow.

| Allocation measure | EDR | Random allocation s42 |
| --- | ---: | ---: |
| Independent responses | 1,286 | 1,701 |
| Subsequent solver rounds | 861 | 969 |
| All solver rounds, including initial | 1,351 | 1,459 |
| Correct final answers | **236** | 216 |

EDR uses 24.4% fewer independent responses and 11.1% fewer subsequent rounds
while answering 20 more questions correctly. These are response and round
counts; probability probes are separate calls.

## Gated fixed-4 allocation

This policy uses the EDR positive-allocation gate. An activated round assigns
four independent and four discussion responses; otherwise, all eight use history.
It answers **214/490 questions (43.67%)**, with 1,816 independent responses over
984 subsequent rounds, or 23.07% of subsequent responses. The run shares initial
responses, ordered first retrieval and history order with the other Qwen conditions.

Its [score](results/edr_gated_fixed4_490.json) and
[trajectory](traces/edr_gated_fixed4_490.jsonl) retain all 490 question IDs.
The [allocation summary](../../../docs/ALLOCATION_ABLATION.md) reports it
alongside random allocation and EDR.

## Final accuracy on questions with initial disagreement

The four multi-agent policies share their initial answers. The following results use
the same 363 questions with initial disagreement, with missing or invalid final answers scored incorrect.

| Condition | Correct / questions | Accuracy | Gain over MA-RAG-int |
| --- | ---: | ---: | ---: |
| MA-RAG-int | 162 / 363 | 44.63% | — |
| Gated fixed 4+4 | 167 / 363 | 46.01% | +1.38 pp |
| Gated random allocation, seed 42 | 169 / 363 | 46.56% | +1.93 pp |
| MA-RAG-int + EDR | 189 / 363 | 52.07% | +7.44 pp |

EDR corrects 34 of 201 baseline errors (16.92%), with seven answer regressions.
Its accuracy exceeds fixed 4+4 and random allocation by 6.06 and 5.51 percentage points, respectively.

Question IDs and paired statistics are in [Initial-disagreement results](../../../analysis/initial_disagreement_metrics.json). Recompute from the package root with `python -B tools/summarize_initial_disagreement.py`.

## Candidate evolution

| Process outcome | Baseline | EDR | Random s42 | Comparison set |
| --- | ---: | ---: | ---: | --- |
| Correct candidate lost from terminal pool | 103 | 81 | 89 | 262 initially covered disputed questions with valid terminal pools |
| Correct option discovered after round 1 | 5 | 16 | 17 | 100 initially uncovered disputed questions |
| Discovered correct option selected finally | 5 | 12 | 7 | Same 100 questions |
| Early incorrect consensus | 191 | 161 | 161 | 363 initially disputed questions |

Discovery records whether a correct option appears; final selection records
whether it becomes the terminal answer. The two quantities describe different
parts of candidate refinement. Candidate-loss comparisons require completed
terminal pools; full accuracy still includes all 490 questions.

## Files and results

| File | Contents |
| --- | --- |
| [results/direct_model_490.json](results/direct_model_490.json) | Direct-model predictions and score |
| [results/ma_rag_int_490.json](results/ma_rag_int_490.json) | Baseline predictions and score |
| [results/ma_rag_int_plus_edr_490.json](results/ma_rag_int_plus_edr_490.json) | EDR predictions and score |
| [results/edr_gated_random_s42_490.json](results/edr_gated_random_s42_490.json) | Random-allocation predictions and score |
| [results/comparisons.json](results/comparisons.json) | Complete-run paired changes, uncertainty and allocation counts |
| [Initial-disagreement results](../../../analysis/initial_disagreement_metrics.json) | Paired accuracy, corrections and allocation-policy comparisons on initial-disagreement questions |
| [results/candidate_discovery.json](results/candidate_discovery.json) | Discovery cohorts and per-question events |
| `traces/` | Full generated answers, decisions, routing and retrieval metadata |

## Running the allocation ablation

The implementation is in [edr_random_allocation.py](../../../src/edr_random_allocation.py).
From the package root, generate a baseline and use its raw receipts for the
random condition:

```bash
python -B src/run_marag.py --root ../MA-RAG --dataset MedXpertQA --condition baseline --output ../research_runs/qwen_baseline --model qwen3.8-27b-nvfp4 --base-url http://127.0.0.1:18080/v1 --retriever-url http://127.0.0.1:18990
python -B src/run_marag.py --root ../MA-RAG --dataset MedXpertQA --condition gated_random --source-baseline ../research_runs/qwen_baseline --allocation-seed 42 --output ../research_runs/qwen_random_s42 --model qwen3.8-27b-nvfp4 --tokenizer /path/to/qwen-tokenizer --base-url http://127.0.0.1:18080/v1 --retriever-url http://127.0.0.1:18990
```

Use the same baseline for the paired `--condition edr` run. Setup and local
validation are described in [integration](../../../docs/INTEGRATION.md).
New outputs remain outside this package.

To check the published results, run `python -B tools/build_medxpertqa_results.py --check`,
`python -B src/test_random_allocation.py` and `python -B tools/verify_package.py`
from the package root. Corpus passages,
full request prompts and raw per-token arrays are obtained or retained
separately under the relevant resource terms.

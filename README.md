# MA-RAG + Evidence-Discussion Routing (EDR)

English | [中文](README.zh-CN.md)

The repository contains code, documentation and result summaries. Download the
[complete code and trajectory package](https://github.com/zonghuanfei/EDR/releases/download/v1.0/EDR_Code_and_Trajectories.zip)
for all experiment trajectories. Run the reproduction commands below from the extracted package directory.

This package provides the training-free EDR module, framework adapters,
question-level predictions and reasoning trajectories for MedXpertQA,
MedDDx-Expert and MMLU-Med. Evaluations cover Qwen and Gemma,
MA-RAG-int and MedLA, repeated Gemma runs, gated fixed-4 and random-allocation ablations,
and the MAD-M² objective history-masking comparator. Dataset sources and
inference settings are recorded in each `DATASET.md`. Upstream frameworks,
question files, weights and MedCorp are obtained separately.

## Complete-benchmark results

Final predictions are scored on every question. Missing or invalid final
predictions and technical failures count as incorrect.

| Benchmark / model / run | N | Baseline correct | EDR correct | Accuracy, baseline → EDR | Gain (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MedXpertQA / Qwen | 490 | 209 | 236 | 42.65% → **48.16%** | **+5.51** |
| MedXpertQA / Gemma initial | 490 | 74 | 93 | 15.10% → **18.98%** | **+3.88** |
| MedXpertQA / Gemma seed 2026091900 | 490 | 84 | 95 | 17.14% → **19.39%** | **+2.24** |
| MedXpertQA / Gemma seed 2026091909 | 490 | 84 | 93 | 17.14% → **18.98%** | **+1.84** |
| MedDDx-Expert / Gemma | 483 | 135 | 150 | 27.95% → **31.06%** | **+3.11** |
| MMLU-Med / Gemma | 1089 | 766 | 802 | 70.34% → **73.65%** | **+3.31** |

Baselines are MA-RAG-int for MedXpertQA, and MedLA with the
shared MedCorp evidence interface for MedDDx-Expert and MMLU-Med.
[Unified metrics](docs/SYSTEMATIC_METRICS.md) define all evaluation sets and
provide the paired statistics and candidate-evolution results.

## Routing-active correction

A question is active when a subsequent round assigns at least one independent
answer. Both conditions are compared on those same IDs.

| Evaluation | Active questions | Correct, baseline → EDR | Gain (pp) |
| --- | ---: | ---: | ---: |
| MedXpertQA / Qwen | 167 | 68 → 92 | **+14.37** |
| MedXpertQA / Gemma initial | 225 | 33 → 51 | **+8.00** |
| MedDDx-Expert / Gemma | 274 | 60 → 75 | **+5.47** |
| MMLU-Med / Gemma | 264 | 111 → 147 | **+13.64** |

Complete-benchmark and routing-active outcomes are reported separately; see
[Statistical reporting](docs/STATISTICAL_REPORTING.md) for their definitions.

## Candidate evolution

Across the four primary conditions, EDR reduces correct-candidate loss and
early incorrect consensus, and increases subsequent discovery of correct
options absent from the initial pool. Each process metric uses its stated
comparison set. MedLA shares one ordered evidence block throughout each
question, so its improvements occur with the same single retrieval stage.
See [mechanism statistics](docs/SYSTEMATIC_METRICS.md) and
[candidate discovery](docs/CANDIDATE_DISCOVERY.md).

## Module comparisons and allocation ablation

| Framework / benchmark / model | N | Baseline correct | MAD-M²(O) correct | EDR correct |
| --- | ---: | ---: | ---: | ---: |
| MA-RAG-int / MedXpertQA / Gemma | 490 | 74 | 78 | **93** |
| MedLA / MedDDx-Expert / Gemma | 483 | 135 | 145 | **150** |

The comparator applies the official objective score and strict-median masking
rule with shared initial materials. Each `madmm_objective/` directory retains
the screening scores, masks and generated reasoning.

On MedXpertQA/Qwen, EDR scores **236/490 (48.16%)**, compared with
**216/490 (44.08%)** for gated random allocation with allocation seed 42.
The paired difference is **4.08 percentage points** (exact McNemar
`p = 0.000535`). EDR generates **1,286 versus 1,701** independent responses
and executes **861 versus 969** subsequent solver rounds. The reductions
are **24.4%** and **11.1%**, respectively.

The random strategy uses the same positive-route rule and uniformly selects
one to six independent responses on activated rounds. Its local draw is keyed
by allocation seed, dataset, question ID and round. The strategies share initial
answers, ordered first retrieval and history order, then each computes its
own probes and subsequent responses. See the
[Qwen evaluation](benchmarks/MedXpertQA/qwen3.8%2027b/README.md).

The gated fixed-4 allocation scores **214/490 (43.67%)**, with 1,816 independent
responses over 984 subsequent rounds (23.07% of subsequent responses).
It shares initial responses, ordered first retrieval and history order with
the other Qwen conditions.
See [allocation results](docs/ALLOCATION_ABLATION.md).

## Contents and reproduction

- `src/edr.py`, `src/plugin_install.py`: EDR routing hook.
- `src/run_marag.py`: MA-RAG baseline, EDR, gated-random and objective-masking entry point.
- `src/edr_random_allocation.py`: gated random allocation and per-round draw validation.
- `src/technical_retry.py`: same-request whole-call technical retry.
- `src/madmm_objective.py`, `src/madmm_plugin.py`: objective history masking.
- `src/medla_transfer/`: native MedLA adapters and offline tests.
- `benchmarks/`: predictions, complete question coverage, reasoning traces and scores.
- `analysis/systematic_edr_metrics.json`: unified results and per-question events.
- `docs/`: methods, statistical definitions, trace schema and integration notes.
- `tools/`: offline statistics and result checks.

MA-RAG traces retain full answers, round decisions, routing probabilities and
allocations, queries and document ID/title/score metadata. MedLA stores 17
initial materials once per question and references them from condition
traces. Trajectories are uncompressed UTF-8 JSONL, one question per line.
Corpus passages, repeated prompts and large per-token arrays are not
redistributed.

```bash
python -B tools/build_medxpertqa_results.py --check
python -B src/test_random_allocation.py
python -B tools/analyze_systematic_metrics.py --output analysis/systematic_edr_metrics.json
python -B tools/test_systematic_metrics.py
python -B tools/verify_package.py
```

See [integration](docs/INTEGRATION.md), [methods](docs/METHODS.md),
[trace format](docs/TRACE_FORMAT.md) and [model/API requirements](docs/MODEL_AND_API.md).
The integration guide includes baseline and `--condition gated_random --allocation-seed 42`
commands for new Qwen evaluations.
`python tools/summarize_rounds_retrieval.py` reconstructs logical solver and
retrieval counts; probability probes are separate inference calls.

## License and attribution

Original software uses [MIT](LICENSE-CODE.txt). Data, reasoning trajectories
and documentation use [CC BY 4.0](LICENSE-DATA.txt), with attribution to Huanfei
Zong. Third-party material retains its terms. See [license scope](LICENSE.md)
and [third-party notices](NOTICE.md).

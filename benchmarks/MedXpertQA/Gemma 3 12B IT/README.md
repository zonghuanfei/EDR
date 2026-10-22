# MedXpertQA: Cross-Model Evaluation with Gemma 3 12B IT

English | [中文](README.zh-CN.md)

This directory contains the initial full-benchmark EDR evaluation with
Gemma 3 12B IT, extending the main MedXpertQA experiment to a second model.
Both Gemma conditions use MA-RAG-int, MedCorp retrieval, eight solver
responses per round, and the same inference settings. EDR adaptively
allocates those responses between evidence-only reasoning and
history-conditioned discussion after the shared initial round.

## Complete-Benchmark Outcome

| Condition | Correct / 490 | Accuracy | Difference from MA-RAG-int |
| --- | ---: | ---: | ---: |
| MA-RAG-int | 74 / 490 | 15.10% | -- |
| MA-RAG-int + EDR | **93 / 490** | **18.98%** | **+19 questions; +3.88 percentage points** |

On the 225 routing-active questions, EDR repairs **33** baseline errors and
changes **15** previously correct answers to incorrect ones, yielding a net
gain of 18 questions. Accuracy within this activated subset increases from
33/225 (14.67%) to 51/225 (22.67%), a gain of **8.00 percentage points**.

The [Qwen experiment](../qwen3.8%2027b/README.md) on the same 490 questions
records a complete-benchmark gain from 209/490 to 236/490. Both models show
positive full-benchmark and routing-active gains.

## Where the Difference Appears

The 489 questions with a valid eight-option initial decision include 351 with
first-round disagreement and 138 with unanimous initial choices. Initial
answers match between conditions on these 489 questions. Question 438 retains
eight generated responses in each condition, but invalid option labels
(`None`) prevent a valid round decision; its two response pools differ.
Both conditions count this question as a technical failure and score it
incorrect. The endpoint comparison is:

| Initial state | Questions | MA-RAG-int correct | EDR correct | EDR difference |
| --- | ---: | ---: | ---: | ---: |
| Disagreement | 351 | 52 | **71** | **+19** |
| Unanimous | 138 | 22 | 22 | 0 |
| Invalid initial option parsing | 1 | 0 | 0 | 0 |

EDR activates its independent evidence-only branch on 225 questions and 706
subsequent rounds. On this activated subset, MA-RAG-int gets 33 correct and EDR
gets 51 correct: 33 repairs, 15 harms, and a net gain of 18 questions. The
observed improvement is concentrated in questions where EDR changes the
independent/discussion allocation.

Three Gemma runs are summarized in [multiseed/README.md](multiseed/README.md).
All three show positive full-benchmark gains.

## Objective History-Masking Comparator

The [MAD-M2 objective comparison](madmm_objective/README.md) transfers the
published objective history selector to this same MA-RAG-int host and shared
initial material. The 490-question observed endpoint scores are baseline
74/490, objective masking 78/490 and EDR 93/490. EDR exceeds this comparator
by 15 correct answers (+3.06 percentage points). Its separate directory
contains full generated answers, per-round masks and an execution audit;
the original baseline and EDR trajectories remain unchanged.

## Independent Reasoning and Continued Correction

EDR preserves alternative interpretations when shared discussion begins to
converge on an incorrect answer. Under MA-RAG's unanimity stopping rule,
maintaining these alternatives keeps the retrieval-and-reasoning process
active and creates further opportunities to revise the answer.

Among the 351 initially disagreeing questions, early incorrect-consensus
terminations fall from **294 to 217**. Here, early incorrect consensus means
that all eight responses select the same wrong option and terminate in rounds
2-7, before the eight-round limit. On 155 questions, EDR remains non-unanimous
at the round where the baseline has already stopped incorrectly. It repairs
**28 of these 155 questions (18.06%)** through continued reasoning. The
independent branch is active by that comparison round on 140 of these
questions, including 27 subsequent repairs.

Question 27 provides a concrete example. The gold answer is J, and both
conditions start from the same eight responses:

| Stage | MA-RAG-int | EDR independent branch | EDR discussion branch | EDR progression |
| --- | --- | --- | --- | --- |
| Round 1 | `D D D D D D B H` | Shared initial response pool | Shared initial response pool | Initial disagreement |
| Round 2 | `D D D D D D D D`; stops with D | `H D J J J D` | `D D` | D leads, but disagreement keeps the process active |
| Round 3 | Already terminated | `J J J J J J` | `J J` | Unanimous correct answer J |

The independent branch introduces J while the discussion branch remains
unanimously committed to D. Preserving this correct alternative allows another
retrieval-and-discussion cycle, after which all eight responses select J.
This trajectory shows how maintaining independence can turn an incorrect
consensus trajectory into a successful correction.

## Reasoning and Retrieval Counts

Counts use all 490 questions, including technical failures. Each solver round
is one eight-response stage; a retained stage without a valid final decision
counts once. A retrieval round is one retained query-and-retrieval stage.
Totals are followed by means per question in parentheses.

| Condition | Solver rounds, total (mean) | Retrieval rounds, total (mean) | Retrieval queries, total (mean) |
| --- | ---: | ---: | ---: |
| MA-RAG-int | 1,159 (2.37) | 669 (1.37) | 2,676 (5.46) |
| MA-RAG-int + EDR | 1,629 (3.32) | 1,139 (2.32) | 4,556 (9.30) |

The totals include four partial solver rounds under MA-RAG-int and six under
EDR. Every retained retrieval stage contains four queries. Shared prefixes
count once in each condition; technical retries are not extra logical stages.
EDR's E/H/EH probes are separate inference calls outside these counts.
Recompute the statistics with `python tools/summarize_rounds_retrieval.py`
from the package root.

## Scoring and Trace Scope

Exact final-option accuracy is computed over all 490 questions. Technical
failures count as incorrect: four under MA-RAG-int and six under EDR. EDR also
has eight **completed** trajectories without one final option; these count
incorrect. Failure IDs and all
question-level predictions are in [final_scores_490.json](results/final_scores_490.json).
Model, generation and retrieval settings are in [DATASET.md](DATASET.md).

The EDR condition reuses baseline first-round solver, query, and retrieval
receipts when available. The EDR core source in this package is identical to
the one used in the Gemma run.

See [DATASET.md](DATASET.md) for dataset identity and model settings, and
the two files under `traces/` for the 490-question trajectories. Each trace
retains generated answers, round decisions, EDR routing distributions,
queries, and retrieved-document ID/title/score metadata.
Run `python tools/verify_gemma_medxpertqa.py` from
the package root to check question coverage and score/trace consistency.

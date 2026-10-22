# Benchmark trajectory format

For the MA-RAG evaluations, each uncompressed UTF-8 JSONL line represents one question. The `id`
is the MedXpertQA question index (0-489);
`gold`, `final_option`, `correct`, and
`first_round_option` accompany an ordered `rounds` array. Every solver round
has eight `answers` with candidate index, full model-generated text and finish
reason. The `decision` field gives the eight extracted options, entropy values,
round prediction and usage. `routing` gives EDR branch counts, trigger score,
top options and E/H/EH option probability distributions where applicable.
`retrieval` gives generated queries and deduplicated document ID/title/score
metadata. A missing score is represented as JSON null, not an inferred value.

The Qwen MedXpertQA directory contains the `ma_rag_int`,
`ma_rag_int_plus_edr` and `edr_gated_random` conditions. The random trace is
named `edr_gated_random_s42_490.jsonl`. Both strategy traces contain 490
records, including question 29's initial-query technical failure, and retain
`status`, `run`, `generation_seed` and `history_order`. The random condition
also retains `allocation_seed`; each route includes its reference
`adaptive_independent` and `trigger` together with the realized independent
and discussion counts. The strategies compute probes separately after their
shared initial state.

Qwen MedXpertQA answer objects retain index, full text and finish reason. Their allocation
totals are recorded in `routing`; per-answer `channel` is not present in this
projection. Analyses therefore record those answer sources as `unspecified`.
`direct_model_490.jsonl` contains one direct answer and final option per
question without retrieval or routing rounds.

Generated answer text is retained as produced and can paraphrase a source.

The MA-RAG conditions use `ma_rag_int` and `ma_rag_int_plus_edr` as condition names,
with filenames `<condition>_<count>.jsonl` in `traces/`.

The Gemma MedXpertQA model directory uses the same condition names and trace
projection. Both files contain 490 records. Each has a `status` of `complete`
or `technical_failure`; the latter retains available partial rounds and a
null final option. A completed run may also have a null final option when its
eight-round endpoint does not yield one selected choice. Its full-benchmark
result file lists technical failures separately from completed null outcomes.

`results/final_scores_<count>.json` uses `conditions` for aggregate scores,
`edr_vs_ma_rag_int` for paired changes, and `rows` for per-question results.
Each row has `id`, `gold`, `<condition>_prediction`, and `<condition>_correct`.

## Gated fixed-4 allocation

`edr_gated_fixed4_490.jsonl` retains all 490 question records. The independent
count is zero or four in each subsequent round. This condition shares initial
responses, ordered first retrieval and history order with the other Qwen conditions.

## MedLA / MedDDx-Expert

The MedDDx-Expert directory has three UTF-8 JSONL files, each containing
483 question records with source IDs 0-482:

- `shared_initial_483.jsonl`: 17 accepted initial elimination reasonings and
  native logic reports per valid question, the initial answer list, and six
  shared document IDs/titles/scores. The four technical failures instead retain
  available partial initial outputs and failure metadata.
- `medla_shared_evidence_483.jsonl`: baseline discussion and terminal decisions.
- `medla_shared_evidence_plus_edr_483.jsonl`: EDR discussion and terminal decisions.

Condition records use `shared_initial_ref` to identify the common initial
material by file and question ID. `rounds` denotes subsequent discussion only,
up to three rounds. Each round keeps all selected representative answers
(usually two to four), not only the representatives surviving to the end.
Answers include the full generated `text`, previous/current option, actor,
channel, finish reason, common-prefix replay flag and request hash. Earlier
format outputs are retained when present. `final_decision` identifies consensus
or includes the final judge's full generated answer.

EDR `routing` stores actual `independent_count` and `discussion_count` for N
representatives. `reference_independent_8` is the original eight-response
allocation before its explicit N-representative mapping; it is not an actual
agent count. The record also contains E/H/EH distributions, top choices, TV
distances, shift score, independent actor names and four-label probe log
probabilities. Baseline rounds have zero independent representatives and no
probability probes. Original full A-D options remain the legal answer space;
`candidate_labels_before` describes the current native representative pool.

`results/final_scores_483.json` stores the two condition summaries and 483
paired prediction rows. `analysis_483.json` reconstructs all reported metrics.
Each missing technical prediction counts as incorrect in the full denominator.
The benchmark's `DATASET.md` and bilingual README define the model, retrieval
setup, scoring and metric denominators.

中文：MedLA 轨迹将首轮 17 份材料共享保存一次，后续每轮保留实际代表回答，而非固定八份。
`independent_count` 是实际独立人数，`reference_independent_8` 是映射前的八回答参考人数。
四道技术失败也保留索引、可用部分首轮输出及失败状态，不从全量分母删除。

## MedLA / MMLU-Med

The MMLU-Med directory uses the same three-file MedLA projection with suffix
`1089`, internal IDs 0-1088, and an additional `source_id` retaining the MIRAGE
subject-specific identifier. Its shared initial file contains 17 accepted
initial materials per valid question. The sole shared technical failure,
ID 78, retains partial initial outputs and is counted incorrect in both arms.
The four-option E/H/EH probability distributions, actual representative
allocations and full generated subsequent answers follow the definitions above.
`analysis_1089.json` reconstructs the benchmark, activated-subset, candidate
coverage and discussion-depth metrics from these records.

## MedLA / Objective History Masking

`MedDDx-Expert/madmm_objective/traces/medla_shared_evidence_plus_madmm_objective_483.jsonl`
contains 483 records. Its `shared_initial_ref` points to the existing parent
shared-initial file, so the same 17 initial materials are stored only once.
The four inherited initial failures retain null predictions and count wrong
for all three conditions.

Each round's `history_mask` record retains the scored agent pool, scalar confidence scores,
strict-median threshold, retained actor names and mask. The score segments
preserve their scalar scores and valid-token counts, allowing the token-weighted
aggregation to be checked without redistributing token arrays. Answer records
retain the full generated text and actual visible own/peer histories. Logic
reports remain attached to retained peers but are not included in the confidence
score; the native final judge is not history-masked. Shared-initial references
and reference score files are relative paths within the public package.

中文：MMLU-Med 采用相同 MedLA 轨迹格式，保留上游学科题目 ID；第 78 题为共同首轮技术失败。
MAD-M²对照引用父目录首轮材料，每轮保留标量评分、有效 token 数、阈值、保留名单及实际可见历史，
并保留完整生成论证；不重复保存首轮长文，不分发大型逐 token 数组。

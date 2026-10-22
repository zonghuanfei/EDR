# MedXpertQA benchmark identity

All scores and trajectories in this directory use the same 490-question file
distributed in the official MA-RAG repository:

```text
Repository: https://github.com/NJU-RL/MA-RAG
Repository revision used: 423f031cc0ffca679a047c885be999b9604100a9
Repository-relative file: datasets/MedXpertQA.json
Records: 490, IDs 0 through 489
```

Every question has ten non-empty options, A through J, and one gold option. The
file contains 190 diagnosis, 168 treatment, and 132 basic-science questions.
The primary metric is exact final-option accuracy. Gold answers are used only
after inference for scoring.

MedXpertQA is the question benchmark. MedCorp is a separate retrieval corpus
used by MA-RAG; it is not part of this benchmark directory and is not copied
into this package. Obtain both upstream resources under their original terms.

The multi-agent conditions use Qwen3.8-27B-NVFP4: eight solver candidates per
round at temperature 0.7, thinking disabled and an 8,192-token completion cap,
for at most eight rounds; a query agent at temperature 0 with thinking enabled
and at most four numbered queries; and EDR option probes at temperature 1.0
constrained to one legal A-J option token. Retrieval uses MedCorp (BM25, then
MedCPT reranking); 32 documents are retrieved per query, two are retained and
duplicates removed. The `direct_model` condition answers each question once at
temperature 0 with thinking disabled and no retrieval. Solver decoding is
stochastic and no generation seed was explicitly fixed. The primary metric is
exact final-option accuracy; gold answers are used only after inference.

Weights are from `Inferact/Qwen3.8-27B-NVFP4`, Hugging Face revision
`6128240ebaf4eaa7bad2b3d1c72c37d677c5f462`. The served ID is
`qwen3.8-27b-nvfp4`; use the tokenizer from the same snapshot.
See [model identity and API requirements](../../../docs/MODEL_AND_API.md).

The EDR and gated-random strategies share initial answers, entropy values,
ordered first queries and retrieval, and descending-entropy history order.
Each strategy then executes its own probes and subsequent responses. The
random condition records allocation seed 42 and the realized independent
count in each route. The full settings are in [configuration.json](configuration.json).

- `results/`: per-condition scores, per-question predictions and paired
  comparisons for all 490 questions.
- `traces/`: UTF-8 JSONL trajectories with full generated answers, decisions,
  routing probabilities, allocations and document ID/title/score metadata.

EDR and gated random each retain question 29's initial query technical failure
with a null final prediction, counted incorrect in the full denominator.
Retrieved passages and complete request prompts are not redistributed.

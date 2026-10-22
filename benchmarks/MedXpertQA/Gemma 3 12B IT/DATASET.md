# MedXpertQA dataset and Gemma run

This directory contains the Gemma 3 12B IT evaluation on the same public
490-question MedXpertQA file used in the Qwen experiment. The questions and
retrieval corpus are not redistributed here.

```text
Repository: https://github.com/NJU-RL/MA-RAG
Repository revision: 423f031cc0ffca679a047c885be999b9604100a9
Question file: datasets/MedXpertQA.json
Question IDs: 0-489 (all 490 questions)
Retrieval corpus: MedCorp, obtained separately from the upstream project
```

Each item has ten options (A-J) and one gold option. The benchmark contains
190 diagnosis, 168 treatment, and 132 basic-science questions. All 490
questions are scored in one full-benchmark result; no question was selected
or excluded according to its answer or model outcome.

`results/final_scores_490.json` contains the score, paired comparisons, and
one row per question. `traces/` contains one UTF-8 JSONL file per condition,
with 490 records each. A technical failure has a null final option and is
counted incorrect; its available partial rounds are retained. A completed
run without a single final option is also counted incorrect but is not called
a technical failure. The traces
retain generated answers, round decisions, EDR routing, and retrieved-document
ID/title/score metadata.

The two conditions are MA-RAG-int and MA-RAG-int + EDR, both using the same
Gemma model and MedCorp retrieval setup. Both use `gemma-3-12b-it` in
bfloat16 without weight quantization, eight solver candidates per round at
temperature 0.7 with an 8,192-token completion cap and at most eight rounds,
a query agent at temperature 0 with at most four queries, and EDR option
probes at temperature 1.0. Retrieval uses MedCorp (BM25, then MedCPT
reranking); each query returns 32 documents, two are retained and duplicates
removed. Scoring is exact final-option accuracy over all 490 questions.
See `README.md` for the outcome analysis. A new run also requires
the upstream MA-RAG code, MedXpertQA questions, and MedCorp index under their
respective terms.

The weight repository is [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it).
Use its matching local tokenizer. Model sources and the probability-interface
check are in [model/API requirements](../../../docs/MODEL_AND_API.md).

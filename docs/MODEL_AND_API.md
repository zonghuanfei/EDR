# Model Identity and API Requirements

## Weights and Tokenizers

| Evaluation model | Weight repository | Recorded precision | Served ID |
| --- | --- | --- | --- |
| Qwen3.8-27B-NVFP4 | [Inferact/Qwen3.8-27B-NVFP4](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4) | NVFP4 | `qwen3.8-27b-nvfp4` |
| Gemma 3 12B IT | [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) | BF16, no weight quantization | `gemma-3-12b-it` |

The Qwen download metadata identifies Hugging Face revision
`6128240ebaf4eaa7bad2b3d1c72c37d677c5f462` and ModelScope mirror revision
`6900f2d1557157a1144cf7480d0e3ab3e112c012`.
Download the tokenizer from the same weight snapshot, not a different Qwen size.
The Gemma repository is the instruction-tuned 12B release; use its matching tokenizer.
Model weights and their original licenses are obtained separately.
The model API ID must match the name configured on the serving endpoint.

## Probability Endpoint

The retained services used vLLM 0.28-family builds. The client requires these
capabilities, checked independently of a service's version string:

- `/v1/chat/completions` with indexed `n` responses and generation token log
  probabilities. MA-RAG requires `top_logprobs=20` for native entropy ordering.
- EDR probes: `n=1`, one output token, temperature 1.0, and exact capital-label
  probabilities for every legal option. The requests send `allowed_token_ids`
  and `logprob_token_ids` at the JSON body's top level, along with `logprobs`
  and `top_logprobs`. These extension fields must be supported by the endpoint.
- MedLA MAD rescoring additionally requires `/v1/completions` accepting token-ID
  prompts and returning `prompt_logprobs`; new medical answers use
  `logprobs=true, top_logprobs=1`.

EDR does not impute missing option probabilities. New command-line runs check
all three probe views before evaluating questions, so an incompatible endpoint
fails before a benchmark run. The standalone hook records a fallback when an
individual later probe fails.

With an already running service, this explicit preflight sends three small
test requests, not benchmark questions:

```bash
python -B src/endpoint_check.py --base-url http://127.0.0.1:18080/v1 --model gemma-3-12b-it --tokenizer /path/to/gemma-tokenizer --labels ABCDEFGHIJ
```

Use `--labels ABCD` for MedLA. Every label
must map to one distinct tokenizer token. The client requirements do not install
or launch a GPU model service.

## Retrieval Endpoint

The existing MedCorp service exposes `/retriever`, accepting `query`,
`total_k`, `top_k`, `combine_docs=false` and `use_reranker=true`. It returns
`documents` with ID/title/content and a corresponding `scores` array.
MA-RAG recalls 32 and keeps two documents per query; MedLA recalls 32 and
keeps six for its one question-only retrieval. Published traces retain
document ID/title/score metadata rather than passage text.

Original model documentation: [Qwen quantized model card](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4),
[Google Gemma model card](https://huggingface.co/google/gemma-3-12b-it).

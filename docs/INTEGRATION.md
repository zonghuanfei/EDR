# Framework Integration

Use Python 3.11 or later. Install the client dependencies and obtain the separately licensed upstream
MA-RAG checkout:

```bash
python -m pip install -r requirements.txt
git clone https://github.com/NJU-RL/MA-RAG.git ../MA-RAG
git -C ../MA-RAG checkout 423f031cc0ffca679a047c885be999b9604100a9
```

`src/run_marag_queryguard_w8.py` supplies the concrete HTTP/cache, solver,
intrinsic-entropy ordering, parsing and retrieval interfaces. It loads the
five original prompts from the pinned `ma_rag_entropy.py`; its local parser
dependencies are included. `src/run_marag.py` is the shared runnable entry point.
The model and MedCorp services must already be available; this package does not
launch services. See [model and API requirements](MODEL_AND_API.md).

## MA-RAG Baseline and EDR

Examples use Gemma with a matching local tokenizer. Use the Qwen served ID and
tokenizer for Qwen conditions:

```bash
python -B src/run_marag.py --root ../MA-RAG --dataset MedXpertQA --condition baseline --output ../research_runs/marag_baseline --model gemma-3-12b-it --base-url http://127.0.0.1:18080/v1 --retriever-url http://127.0.0.1:18990
python -B src/run_marag.py --root ../MA-RAG --dataset MedXpertQA --condition edr --output ../research_runs/marag_edr --source-baseline ../research_runs/marag_baseline --model gemma-3-12b-it --tokenizer /path/to/gemma-tokenizer --base-url http://127.0.0.1:18080/v1 --retriever-url http://127.0.0.1:18990
```

These commands evaluate all 490 MedXpertQA questions. `--validate-only` checks
local data, prompts and tokenizer without sending API requests. Outputs are required
to be outside the release directory; new runs retain receipts and write
`scores.json`.

Each paired EDR question reuses valid first-round solver, query and retrieval
receipts from the baseline. Invalid frozen initial materials remain failures
rather than being regenerated. Request-validated reused query outputs retain
their source query order. Gold labels enter only the scoring step.

## Gated random allocation

`src/edr_random_allocation.py` retains the EDR positive-route rule and replaces
the adaptive independent count with a discrete-uniform draw from 1 to 6.
Untriggered rounds assign all eight responses to discussion. The local draw
is keyed by allocation seed, dataset, question ID and round; it is unaffected
by worker scheduling and does not set a model-generation seed. The released
comparison uses allocation seed 42.

Generate a Qwen baseline, then run the random condition with its raw receipts:

```bash
python -B src/run_marag.py --root ../MA-RAG --dataset MedXpertQA --condition baseline --output ../research_runs/qwen_baseline --model qwen3.8-27b-nvfp4 --base-url http://127.0.0.1:18080/v1 --retriever-url http://127.0.0.1:18990
python -B src/run_marag.py --root ../MA-RAG --dataset MedXpertQA --condition gated_random --source-baseline ../research_runs/qwen_baseline --allocation-seed 42 --output ../research_runs/qwen_random_s42 --model qwen3.8-27b-nvfp4 --tokenizer /path/to/qwen-tokenizer --base-url http://127.0.0.1:18080/v1 --retriever-url http://127.0.0.1:18990
```

Use the same baseline directory for the corresponding `--condition edr` run.
The EDR and random policies share frozen initial answers and ordered first
retrieval, then execute their own probes and subsequent generations.
`--source-baseline` requires raw request and retrieval receipts; the compact
published JSONL files are for trajectory inspection and offline analysis.

The published random trace retains actual allocations, reference adaptive
counts and trigger states. Recompute the published scores and metrics with:

```bash
python -B tools/build_medxpertqa_results.py --check
python -B tools/analyze_systematic_metrics.py --output analysis/systematic_edr_metrics.json
```

## Gated fixed-4 allocation

`src/edr_gated_fixed4.py` provides the fixed-allocation hook. On positive
routes it generates four independent and four discussion responses; other
rounds use eight discussion responses. For a new evaluation, reuse a baseline
run with the same model and inference settings:

```bash
python -B src/run_marag.py --root ../MA-RAG --dataset MedXpertQA --condition gated_fixed4 --source-baseline ../research_runs/qwen_baseline --output ../research_runs/qwen_fixed4 --model qwen3.8-27b-nvfp4 --tokenizer /path/to/qwen-tokenizer --base-url http://127.0.0.1:18080/v1 --retriever-url http://127.0.0.1:18990
```

Published outcomes and retrieval pairing are described in
[allocation results](ALLOCATION_ABLATION.md).

## MAD-M²

The objective history-masking adapter is run with a baseline source:

```bash
python -B src/run_marag.py --root ../MA-RAG --dataset MedXpertQA --condition madmm_objective --output ../research_runs/marag_madmm --source-baseline ../research_runs/marag_baseline --model gemma-3-12b-it --base-url http://127.0.0.1:18080/v1 --retriever-url http://127.0.0.1:18990
```

`src/madmm_plugin.py` applies the objective score and strict-median selector
to actual solver history. It retains evidence, native history order, query
context, response count, voting and stopping. Native entropy and MAD scores
require generation-token probabilities during inference.

## MedLA and Offline Verification

The native MedLA EDR and MAD adapters are in `src/medla_transfer/`.
[MedLA integration](MEDLA_INTEGRATION.md) gives complete MedDDx-Expert and
MMLU-Med commands, the representative-allocation mapping and mocked tests.

```bash
python -B src/test_edr_core.py
python -B src/test_reproduction.py
python -B src/test_random_allocation.py
python -B tools/verify_package.py
python -B tools/verify_gemma_medxpertqa.py
python -B tools/test_systematic_metrics.py
```

These checks do not call a model or retriever. For recomputing cross-framework
statistics, run `tools/analyze_systematic_metrics.py` with a separate
`--output` path. Raw receipts for a new run are distinct from the compact
published JSONL files, which do not redistribute corpus passages or full prompts.

中文：本包提供MA-RAG与MedLA接入器、门控随机分配、客观历史屏蔽和离线结果检查。
随机策略通过`--condition gated_random --allocation-seed 42`运行，沿用EDR正路由规则，
触发后在一至六个独立回答之间均匀抽样；分配种子不传给生成模型。逐轮分配保存在JSONL中。
完整运行需要另行获取上游题目、模型与MedCorp；新输出保存到包外目录。

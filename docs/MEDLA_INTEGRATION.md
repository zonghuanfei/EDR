# MedLA Adapter

The adapter loads native functions from the fixed MedLA checkout and reuses
`src/edr.py`. It keeps 17 initial agents, three sequential eliminations per
agent, initial logic checks, native representative selection, at most three
discussion rounds, unanimity stopping and the final judge. The EDR context
change applies only to selected representative discussion turns.

## Setup and Offline Tests

From the release root, obtain the upstream framework separately:

```bash
git clone https://github.com/alexander2618/MedLA.git ../MedLA
git -C ../MedLA checkout 5c12cfc8d67170b1f4b131b9c120a54a573c634d
python -m pip install -r src/medla_transfer/requirements.txt
export MEDLA_UPSTREAM="/path/to/MedLA"
export MEDLA_RETRIEVER_URL="http://127.0.0.1:18990/retriever"
python -B src/medla_transfer/test_medla.py
python -B src/medla_transfer/test_madmm_medla.py
```

In PowerShell, set the two environment variables with
`$env:MEDLA_UPSTREAM='E:\path\to\MedLA'` and
`$env:MEDLA_RETRIEVER_URL='http://127.0.0.1:18990/retriever'` instead of `export`.
The mocked tests need the upstream Python files but do not call an API.
See [model/API requirements](MODEL_AND_API.md) for weights and probability
output requirements.

## Complete Paired Evaluations

```bash
python -B src/medla_transfer/run_medla.py --root ../research_runs/medddx --base http://127.0.0.1:18080/v1 --model gemma-3-12b-it --tokenizer /path/to/gemma-tokenizer --concurrency 8
python -B src/medla_transfer/prepare_mmlu_med.py --output ../research_data/mmlu
python -B src/medla_transfer/run_medla.py --root ../research_runs/mmlu --dataset-jsonl ../research_data/mmlu/mmlu.jsonl --dataset-name MMLU-Med --expected-count 1089 --base http://127.0.0.1:18080/v1 --model gemma-3-12b-it --tokenizer /path/to/gemma-tokenizer --concurrency 8
```

The first command reads the complete 483-question MedDDx-Expert/hard file
from MedLA. The MMLU preparation script obtains the pinned MIRAGE source and
normalizes its full 1,089-question medical test without changing order or
option labels. Both use the same four-option adapter.

Each paired question generates its initial materials once and shares one
ordered six-document MedCorp block across both conditions. Each document
contributes its title and up to 1,200 content characters. Later identical
baseline requests are replayed until the first actual history removal.
For N native representatives, the independent count is
`min(N-1, floor(N*m8/8 + 0.5))`, where `m8` is the original EDR allocation.
The first selected representatives in native order receive no own/peer history;
the others retain native discussion context.

New output contains `paired_results.json`, per-case raw receipts and
`summary.json`. Full-denominator scoring retains technical failures. The
public run entry point is for complete dataset evaluation; operational queue
controllers and serving lifecycle scripts are not distributed.

## MAD-M2 on the Same MedDDx Initial Materials

After producing a new MedDDx paired run:

```bash
python -B src/medla_transfer/run_madmm_medla.py --source ../research_runs/medddx --root ../research_runs/medddx_madmm --preflight
python -B src/medla_transfer/run_madmm_medla.py --source ../research_runs/medddx --root ../research_runs/medddx_madmm --base http://127.0.0.1:18080/v1 --model gemma-3-12b-it --tokenizer /path/to/gemma-tokenizer --concurrency 8
```

The preflight makes zero model calls. The comparator freezes the initial
medical reasonings and uses the official strict-middle selector to mask
own/peer medical history, retaining evidence, candidate labels, native
representative selection and final adjudication. It uses generation top-1
log probabilities for new answers; saved initial text without token arrays
is teacher-forced under its original context, including the end-of-turn
token. Segment scores are combined by valid-token count.

`--source` is the newly generated raw run, not the compact published JSONL
directory: inference requires retrieved passage text and complete messages.
The published trajectories and scalar routing/masking records support
offline result verification without redistributing those inputs.

中文：MedLA 接入器复用固定版本原生推理函数，保留首轮17个智能体、代表选择、
至多三轮讨论和终轮裁决。EDR只改变指定代表的历史可见性；两组共享首轮材料与
六篇检索证据。上面的命令分别覆盖 MedDDx-Expert 全483题、MMLU-Med 全1089题
及同首轮材料的 MAD 对照。输出必须写到发布包之外，测试不消耗模型调用。

# MMLU-Med Dataset Identity

- Evaluation: the complete 1,089-question, four-option medical test from the pinned MIRAGE benchmark.
- Source: [MIRAGE](https://github.com/gzxiong/MIRAGE), commit `392943af99cd94cafd50a0de2e7fca24bbf65494`.
- Source file: [`benchmark.json`](https://raw.githubusercontent.com/gzxiong/MIRAGE/392943af99cd94cafd50a0de2e7fca24bbf65494/benchmark.json), key `mmlu`.
- Answer space: A-D, exactly one answer.
- Ordering: original source order, without sampling, filtering or option permutation.
- Local IDs: 0-1088. `source_id` preserves the upstream subject/item key, such as `anatomy-000`.

| Subject | Questions |
| --- | ---: |
| Anatomy | 135 |
| Clinical knowledge | 265 |
| College biology | 144 |
| College medicine | 173 |
| Medical genetics | 100 |
| Professional medicine | 272 |
| Total | 1,089 |

The original MMLU benchmark is described in [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300).
The host framework is [MedLA](https://github.com/alexander2618/MedLA), commit
`5c12cfc8d67170b1f4b131b9c120a54a573c634d`. Both conditions use the same
MedCorp evidence interface: BM25 recall 32, MedCPT reranking, six retained
documents, one question-only retrieval stage. Retrieval text is not included.

Question files, upstream framework, model weights and retrieval corpus are
obtained separately under their original terms. This package contains
experimental predictions and generated reasoning, not the question bank.

中文：完整评估 MIRAGE 固定版本中六个医学相关学科的 MMLU-Med 测试，共1089题，
四选一。保留原始顺序及来源题号，没有抽样或筛题。原题、MedLA、模型和检索库另行获取；
本目录保存实验结果和生成轨迹。

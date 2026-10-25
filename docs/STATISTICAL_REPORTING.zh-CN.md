# 准确率与首轮分歧题目

完整题集准确率按全部题目的最终选项计算，缺失或无效终答及技术失败计错。

首轮分歧集合包含两组共享有效首轮回答、且至少出现两个不同选项的题目。MA-RAG-int使用八份首轮回答，MedLA使用17份。各方法在同一题目集合上比较最终预测；入选题目后续缺失或无效的终答计错。

| 实验 | 题数 | 基线正确数 | EDR正确数 | 准确率：基线→EDR | 提升 (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MedXpertQA / Qwen | 363 | 162 | 189 | 44.63% → 52.07% | +7.44 |
| MedXpertQA / Gemma initial | 351 | 52 | 71 | 14.81% → 20.23% | +5.41 |
| MedDDx-Expert / Gemma | 468 | 125 | 140 | 26.71% → 29.91% | +3.21 |
| MMLU-Med / Gemma | 767 | 465 | 501 | 60.63% → 65.32% | +4.69 |
| MedXpertQA / Gemma seed 2026091900 | 341 | 62 | 73 | 18.18% → 21.41% | +3.23 |
| MedXpertQA / Gemma seed 2026091909 | 356 | 64 | 73 | 17.98% → 20.51% | +2.53 |

纠错指基线答错而EDR答对，答案回退指基线答对而EDR答错。纠错率为纠错数除以同一集合中的基线错误数。方法对照使用相同的首轮分歧集合，各次Gemma运行分别统计。

```bash
python -B tools/summarize_initial_disagreement.py
python -B tools/summarize_initial_disagreement.py --check
```

[统计结果](../analysis/initial_disagreement_metrics.json)保存题目编号、准确率、纠错与回退、配对bootstrap区间及双侧精确McNemar检验结果，并包含固定分配、随机分配和MAD-M²(O)的对照。全量成绩与候选演化定义见[统一指标](SYSTEMATIC_METRICS.md)。

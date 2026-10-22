# 准确率与路由激活子集

完整题集准确率按全部题目的最终选项计算。缺失或无效最终选项及技术失败均计错。

若一道题至少有一个EDR后续轮次向独立分支分配一个或更多回答（`independent > 0`），则记为路由激活题。激活子集在同一批题目上比较两组记录的最终答案。

| 实验 | 总题数 | 全量正确数：基线 → EDR | 激活题数 | 激活题正确数：基线 → EDR |
| --- | ---: | ---: | ---: | ---: |
| MedXpertQA / Qwen | 490 | 209 → 236 | 167 | 68 → 92 |
| MedXpertQA / Gemma initial | 490 | 74 → 93 | 225 | 33 → 51 |
| MedXpertQA / Gemma seed 2026091900 | 490 | 84 → 95 | 234 | 46 → 57 |
| MedXpertQA / Gemma seed 2026091909 | 490 | 84 → 93 | 226 | 44 → 54 |

```bash
python -B tools/build_routing_subset_analysis.py --check
```

对应的`results/routing_subset_analysis.json`保存完整题集和激活子集汇总。跨架构结果与候选演化定义见[统一指标](SYSTEMATIC_METRICS.md)。

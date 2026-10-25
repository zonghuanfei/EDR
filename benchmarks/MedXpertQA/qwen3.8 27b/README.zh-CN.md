# MedXpertQA：Qwen上的证据—讨论路由

[English](README.md) | 中文

本实验使用全部490道MedXpertQA文本题，每题十个选项（A–J）。EDR在每轮八个回答中协调
仅依据证据的独立重审与结合历史的讨论。模型及推理设置见[DATASET.md](DATASET.md)和
[configuration.json](configuration.json)。

## 最终答案成绩

| 条件 | 正确数 / 总数 | 准确率 | 相对MA-RAG-int增益 |
| --- | ---: | ---: | ---: |
| 直接回答 | 147 / 490 | 30.00% | — |
| MA-RAG-int | 209 / 490 | 42.65% | — |
| MA-RAG-int + EDR | **236 / 490** | **48.16%** | **+5.51个百分点** |
| 门控随机分配，种子42 | 216 / 490 | 44.08% | +1.43个百分点 |

EDR纠正34个基线错误，另有7题由对变错，净增加27个正确答案。与随机分配相比，EDR独有
正确26题，随机分配独有正确6题，准确率相差4.08个百分点。相对基线与随机分配的精确双侧
McNemar检验分别为`p = 0.00002532`和`p = 0.00053505`。完整配对结果及置信区间见
[comparisons.json](results/comparisons.json)。

直接回答使用温度0，每题回答一次且不检索；多智能体回答温度为0.7。缺失或平票终答及技术
失败均在490题分母中计错。

## 共同首轮状态

490题均共享基线的八个初始回答及熵。127题首轮一致后直接停止；363题首轮存在分歧。
EDR与随机策略保留基线的首次查询顺序、检索文档顺序及熵降序历史排列。第29题保留首次查询
失败，两策略均计错；其余362次成功首检索的查询与文档有序元数据均与基线相同。

共同首轮之后，两策略分别计算E/H/EH探针并生成后续回答。标准答案仅用于最终计分及诊断分析。

## 状态相关分配

EDR比较仅证据（E）、仅历史（H）及证据加历史（EH）三种上下文下的选项分布，依据首选项
变化与相对总变差偏移分配零至六个独立回答，其余回答保留历史；每轮总数为八。

门控随机消融沿用相同的正路由规则。当其自适应参考人数大于零时，以分配种子42均匀抽取
一至六个独立回答；否则八个回答均使用历史。局部抽样由种子、数据集、题号和轮次共同确定，
不受工作线程调度影响。分配种子不作为模型生成种子。两策略均执行完整的协作流程。

| 分配指标 | EDR | 随机分配s42 |
| --- | ---: | ---: |
| 独立回答数 | 1,286 | 1,701 |
| 后续回答轮数 | 861 | 969 |
| 包含首轮的全部回答轮数 | 1,351 | 1,459 |
| 最终答对数 | **236** | 216 |

EDR少生成24.4%的独立回答，少执行11.1%的后续轮次，同时多答对20题。这里统计回答数量和
轮次，概率探针另计为推理调用。

## 门控固定4+4分配

该策略沿用EDR的正路由门控：触发时分配四份独立回答和四份讨论回答，
未触发时八份回答均使用历史。全量成绩为214/490（43.67%），后续984轮
共分配1,816份独立回答，占后续回答的23.07%。该组与其他Qwen条件共享
首轮回答、首次有序检索和历史排序。

[成绩](results/edr_gated_fixed4_490.json)和
[轨迹](traces/edr_gated_fixed4_490.jsonl)保留全部490个题号。
固定分配、随机分配及EDR的汇总见[分配消融结果](../../../docs/ALLOCATION_ABLATION.md)。

## 首轮分歧题目的最终准确率

四种多智能体策略共享首轮回答；以下结果均在363道首轮存在分歧的题目上计算。后续未得到有效终答的题目计错。

| 条件 | 正确数 / 题数 | 准确率 | 相对MA-RAG-int增益 |
| --- | ---: | ---: | ---: |
| MA-RAG-int | 162 / 363 | 44.63% | — |
| 门控固定4+4 | 167 / 363 | 46.01% | +1.38 pp |
| 门控随机分配，种子42 | 169 / 363 | 46.56% | +1.93 pp |
| MA-RAG-int + EDR | 189 / 363 | 52.07% | +7.44 pp |

EDR纠正了201个基线错误中的34个（16.92%），另有7题由对变错。相对固定4+4和随机分配，EDR的准确率分别提高6.06和5.51个百分点。

题目编号与配对统计见[首轮分歧统计](../../../analysis/initial_disagreement_metrics.json)。在源码包根目录运行 `python -B tools/summarize_initial_disagreement.py` 可复算。

## 候选演化

| 过程指标 | 基线 | EDR | 随机s42 | 统计集合 |
| --- | ---: | ---: | ---: | --- |
| 终端候选池丢失正确选项 | 103 | 81 | 89 | 首轮覆盖正确选项且终端池有效的262道分歧题 |
| 首轮后发现正确选项 | 5 | 16 | 17 | 首轮未覆盖正确选项的100道分歧题 |
| 发现后最终答对 | 5 | 12 | 7 | 同上100题 |
| 提前错误一致 | 191 | 161 | 161 | 首轮分歧的363题 |

候选发现记录正确选项是否出现，最终选择记录其是否成为终答，分别描述候选修正的不同环节。
候选丢失统计要求有效终端回答池，完整准确率仍保留全部490题。

## 文件与结果

| 文件 | 内容 |
| --- | --- |
| [results/direct_model_490.json](results/direct_model_490.json) | 直接回答逐题预测与成绩 |
| [results/ma_rag_int_490.json](results/ma_rag_int_490.json) | 基线逐题预测与成绩 |
| [results/ma_rag_int_plus_edr_490.json](results/ma_rag_int_plus_edr_490.json) | EDR逐题预测与成绩 |
| [results/edr_gated_random_s42_490.json](results/edr_gated_random_s42_490.json) | 随机分配逐题预测与成绩 |
| [results/comparisons.json](results/comparisons.json) | 完整运行的配对变化、区间及分配数量 |
| [首轮分歧统计](../../../analysis/initial_disagreement_metrics.json) | 首轮分歧题目的配对准确率、纠错及分配策略对照 |
| [results/candidate_discovery.json](results/candidate_discovery.json) | 候选发现集合及逐题事件 |
| `traces/` | 完整生成回答、决策、路由与检索元数据 |

## 运行分配消融

实现见[edr_random_allocation.py](../../../src/edr_random_allocation.py)。在数据包根目录先生成
基线，再将其原始请求及检索记录用于随机条件：

```bash
python -B src/run_marag.py --root ../MA-RAG --dataset MedXpertQA --condition baseline --output ../research_runs/qwen_baseline --model qwen3.8-27b-nvfp4 --base-url http://127.0.0.1:18080/v1 --retriever-url http://127.0.0.1:18990
python -B src/run_marag.py --root ../MA-RAG --dataset MedXpertQA --condition gated_random --source-baseline ../research_runs/qwen_baseline --allocation-seed 42 --output ../research_runs/qwen_random_s42 --model qwen3.8-27b-nvfp4 --tokenizer /path/to/qwen-tokenizer --base-url http://127.0.0.1:18080/v1 --retriever-url http://127.0.0.1:18990
```

配对的`--condition edr`运行使用同一基线。依赖安装及本地验证见[接入说明](../../../docs/INTEGRATION.md)。
新运行结果保存到包外目录。

核对已发布结果时，在数据包根目录运行`python -B tools/build_medxpertqa_results.py --check`、
`python -B src/test_random_allocation.py`及`python -B tools/verify_package.py`。
检索正文、完整请求提示及原始逐token数组按资源条款另行获取或保存。

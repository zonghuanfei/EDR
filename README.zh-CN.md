# MA-RAG + 证据—讨论路由（EDR）

[English](README.md) | 中文

本仓库展示源码、说明与结果汇总。包含全部实验轨迹的
[完整代码与轨迹包](https://github.com/zonghuanfei/EDR/releases/download/v1.0/EDR_Code_and_Trajectories.zip)
可直接下载；下文的复算命令在解压后的数据包目录中运行。

本数据包提供无需训练的EDR模块、框架适配器、逐题预测及推理轨迹，覆盖MedXpertQA、
MedDDx-Expert、MMLU-Med三个医学基准，以及Qwen、Gemma两种模型和
MA-RAG-int、MedLA两种协作架构。实验包括Gemma重复运行、门控固定4+4和随机分配消融及
MAD-M²客观历史屏蔽对照。数据来源和推理设置见各目录的`DATASET.md`；上游框架、
题目文件、模型权重与MedCorp检索库另行获取。

## 完整题集结果

全部题目按记录的最终选项计分，缺失或无效终答及技术失败均计错。

| 数据集 / 模型 / 运行 | N | 基线正确数 | EDR正确数 | 准确率：基线→EDR | 增益 (pp) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MedXpertQA / Qwen | 490 | 209 | 236 | 42.65%→**48.16%** | **+5.51** |
| MedXpertQA / Gemma 初始运行 | 490 | 74 | 93 | 15.10%→**18.98%** | **+3.88** |
| MedXpertQA / Gemma seed 2026091900 | 490 | 84 | 95 | 17.14%→**19.39%** | **+2.24** |
| MedXpertQA / Gemma seed 2026091909 | 490 | 84 | 93 | 17.14%→**18.98%** | **+1.84** |
| MedDDx-Expert / Gemma | 483 | 135 | 150 | 27.95%→**31.06%** | **+3.11** |
| MMLU-Med / Gemma | 1089 | 766 | 802 | 70.34%→**73.65%** | **+3.31** |

MedXpertQA的基线为MA-RAG-int；MedDDx-Expert、MMLU-Med的基线为采用共享
MedCorp证据接口的MedLA。统计集合、配对检验与候选演化结果见[统一统计说明](docs/SYSTEMATIC_METRICS.md)。

## 路由激活题的纠错

至少一个后续轮次实际分配独立回答的题目构成激活集合，两组比较同一批题目。

| 实验 | 激活题数 | 正确数：基线→EDR | 增益 (pp) |
| --- | ---: | ---: | ---: |
| MedXpertQA / Qwen | 167 | 68→92 | **+14.37** |
| MedXpertQA / Gemma 初始运行 | 225 | 33→51 | **+8.00** |
| MedDDx-Expert / Gemma | 274 | 60→75 | **+5.47** |
| MMLU-Med / Gemma | 264 | 111→147 | **+13.64** |

完整题集与激活子集分别报告，定义见[统计报告口径](docs/STATISTICAL_REPORTING.zh-CN.md)。

## 候选演化

四个主要条件中，EDR均减少正确候选丢失与提前错误一致，并增加首轮缺失正确选项时的后续
发现。各指标使用对应统计集合。MedLA每题全过程共享同一有序证据块，其提升发生在相同的
一次检索条件下。事件数见[统一机制统计](docs/SYSTEMATIC_METRICS.md)与
[候选发现分析](docs/CANDIDATE_DISCOVERY.zh-CN.md)。

## 模块对照与分配消融

| 框架 / 数据集 / 模型 | N | 基线正确数 | MAD-M²(O)正确数 | EDR正确数 |
| --- | ---: | ---: | ---: | ---: |
| MA-RAG-int / MedXpertQA / Gemma | 490 | 74 | 78 | **93** |
| MedLA / MedDDx-Expert / Gemma | 483 | 135 | 145 | **150** |

客观模块采用官方评分和严格中位阈值，与对应宿主共享首轮材料；各`madmm_objective/`目录
保留筛选分数、掩码与完整论证。

MedXpertQA/Qwen中，EDR答对**236/490（48.16%）**，分配种子为42的门控随机策略答对
**216/490（44.08%）**，配对差值为**4.08个百分点**（精确McNemar `p = 0.000535`）。
EDR生成**1,286对1,701**个独立回答，执行**861对969**个后续轮次，分别减少**24.4%**和**11.1%**。

随机策略沿用相同的正路由规则，在触发轮次均匀抽取一至六个独立回答，抽样由分配种子、
数据集、题号和轮次共同确定。两策略共享首轮回答、
首次有序检索和历史排序，随后分别计算探针并生成后续回答。详见
[Qwen实验说明](benchmarks/MedXpertQA/qwen3.8%2027b/README.zh-CN.md)。

门控固定4+4答对214/490题（43.67%），后续984轮共分配1,816份独立回答，
占后续回答的23.07%。该组与其他Qwen条件共享首轮回答、首次有序检索和历史排序。
汇总见[分配消融结果](docs/ALLOCATION_ABLATION.md)。

## 内容与复算

- `src/edr.py`、`src/plugin_install.py`：EDR路由插件。
- `src/run_marag.py`：MA-RAG基线、EDR、门控随机分配及客观屏蔽入口。
- `src/edr_random_allocation.py`：门控随机分配与逐轮抽样核验。
- `src/technical_retry.py`：同请求整体技术重试。
- `src/madmm_objective.py`、`src/madmm_plugin.py`：客观历史屏蔽。
- `src/medla_transfer/`：MedLA接入器及离线测试。
- `benchmarks/`：完整题目覆盖、推理轨迹、预测与成绩。
- `analysis/systematic_edr_metrics.json`：统一结果与逐题事件。
- `docs/`：方法、统计定义、轨迹格式与接入说明。
- `tools/`：离线分析与结果复算。

MA-RAG保留完整回答、轮次决策、路由概率与分配、查询及文档ID/标题/分数。MedLA每题共享
保存17份首轮材料，条件轨迹按题号引用。轨迹为未压缩UTF-8 JSONL，每行一道题；不随包
分发检索正文、重复提示词及大型逐token数组。

```bash
python -B tools/build_medxpertqa_results.py --check
python -B src/test_random_allocation.py
python -B tools/analyze_systematic_metrics.py --output analysis/systematic_edr_metrics.json
python -B tools/test_systematic_metrics.py
python -B tools/verify_package.py
```

接入见[接入说明](docs/INTEGRATION.md)、[方法](docs/METHODS.md)、[轨迹格式](docs/TRACE_FORMAT.md)
及[模型与接口要求](docs/MODEL_AND_API.md)。接入说明提供Qwen基线及
`--condition gated_random --allocation-seed 42`的新运行命令。
`python tools/summarize_rounds_retrieval.py`可复算逻辑推理
轮次和检索次数；探针单独计为推理调用。

## 许可与署名

原创软件采用[MIT](LICENSE-CODE.txt)；数据、轨迹和文档采用[CC BY 4.0](LICENSE-DATA.txt)，
署名为Huanfei Zong（宗焕斐）。第三方材料沿用原有条款。适用范围见[许可证说明](LICENSE.md)，
来源见[第三方声明](NOTICE.md)。

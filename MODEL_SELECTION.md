# 任务 7.1 中文文本情绪模型选型记录

## 约束

- 服务器：4 核 8 GB，不能访问 Hugging Face。
- 不新建容器、不修改既有内存上限、不占用 OpenFace 所在 Vision 容器。
- CPU 短文本推理，目标为数百毫秒级。
- 输出必须兼容前端 14 类标签。

## 候选对比

| 候选 | 参数/权重 | 标签 | 优点 | 风险 | 结论 |
|---|---:|---|---|---|---|
| `LXDaugh/chinese-6-emotion-model` | 约 0.1B / 实测 390.6 MB | anger、fear、happy、neutral、sad、surprise | 中文、多情绪、体积符合限制、可直接使用 Transformers | 不含焦虑/抑郁专门标签 | **主选** |
| `Johnson8187/Chinese-Emotion-Small` | 约 0.3B / F32 约 1 GB | 平淡、关切、开心、愤怒、悲伤、疑问、惊奇、厌恶 | 中文对话八分类，标签较丰富，公开 4000 条标注数据说明 | 权重和运行内存偏大，会与 Knowledge 向量模型争抢 1.5 GB 限额 | 不部署，仅作对照 |
| `uer/roberta-base-finetuned-jd-binary-chinese` | 约 0.1B / 约 409 MB | 正面、负面 | 项目任务 1 已跑通，成熟稳定 | 电商评论二分类，无法支撑情绪轨迹 | 淘汰 |

模型卡：

- https://huggingface.co/LXDaugh/chinese-6-emotion-model
- https://huggingface.co/Johnson8187/Chinese-Emotion-Small
- https://huggingface.co/uer/roberta-base-finetuned-jd-binary-chinese

## 标签兼容策略

模型标签被标准化到前端已有标签：

| 模型标签 | 系统标签 |
|---|---|
| angry | anger |
| fear | fear |
| happy | happiness |
| neutral | neutral |
| sad | sadness |
| surprise | surprise |

系统映射器同时兼容 positive、negative、mixed、joy、calm、anxiety、
depression、disgust 等前端标签，便于未来无缝替换更丰富的模型。接口返回
主标签、最多三个候选情绪、置信度、强度、效价和完整分数，不把输出解释
成临床诊断。

## 架构决定

模型部署在 Knowledge 容器：该容器已经安装 CPU PyTorch 与 Transformers
依赖，避免 Agent 重复引入运行时，也不会挤占 OpenFace 的 Vision 内存。
Agent 通过 `TEXT_SENTIMENT_BASE_URL=http://knowledge:8002` 调用文本端点。
模型失败时记录 `error_modalities.text` 并继续纯文字对话，保持原有降级能力。

## 本地实测

在 CPU、2 推理线程下输入“我最近压力很大，晚上也很害怕，常常睡不着”：

- 主标签：`fear`
- 候选标签：`fear`、`sadness`
- 置信度：`0.999633`
- 单次模型前向耗时：`111.35 ms`（不含首次权重加载）

模型随附的 `top_level_summary.json` 记录最佳开发集阈值为 `0.4`、宏平均
F1 为 `0.8790`，因此服务默认阈值采用 `0.4`，而不是自行猜测阈值。

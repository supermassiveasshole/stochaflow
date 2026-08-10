# Evaluation 已完成：当前用法和历史想法

> 文档类型：已完成说明，不是待办计划
>
> 工作状态：已完成（Done）
>
> 当前可用性：可以评估 checkpoint，也可以只读取已经保存的预测重新计算指标。

Evaluation 的基础功能已经完成，不再占用开发排期，也没有一个名为“Evaluation 后续”的
待办项目。下面先说明用户现在怎样完成正式评估，再保存几项过去讨论过、但尚未被选择的
独立改进想法。

## 用户现在可以做什么

### 直接评估一个 checkpoint

用户准备一份完整的 Evaluation 配置。配置写清楚：要读取哪个 checkpoint、使用刚训练出的
原始权重（raw）还是指数滑动平均权重（EMA）、评估哪一部分数据、计算哪些指标、预期有多少
个样本。然后运行：

```bash
stochaflow evaluate \
  --config path/to/evaluation.yaml \
  --device cuda \
  --output-dir outputs/evaluations/candidate-a
```

框架会加载冻结的 checkpoint，按配置运行推理和指标，检查样本是否缺失或重复，最后
发布不可变的 `result.json`。这个结果可以用于比较 checkpoint，但 Evaluation 本身不会
替用户决定模型是否发布。

### 不重新运行模型，重新计算已有预测的指标

如果上一次 Evaluation 已经保存了完整预测文件包（prediction artifact），用户可以把新配置的
`subject.kind` 和 `data.source` 都设为 `prediction_artifact`，再次运行同一个
`stochaflow evaluate` 命令。框架会读取并认证已保存的预测，再计算指标；它不会加载
checkpoint，也不会重新生成预测。

这适合修正指标实现、增加任务自己的指标，或者核对同一批预测。完整配置和 Python
调用示例见[配置与工作流文档](../configuration/workflows.md#独立-checkpoint-evaluation)。

## 当前功能保证什么

- checkpoint 路径、raw/EMA 选择、数据 split、指标和样本数量都必须显式写入配置；
- 样本 ID 缺失、重复、超出计划或数量不足时会失败，不会发布看似完整的结果；
- 离线重算只读取预测文件，不会悄悄重新运行模型；
- 训练期 checkpoint 选择只使用 validation 结果，final test 不会反过来改变选择；
- 不同任务可以解释自己的数据、预测和指标；框架不要求所有任务都是图像生成。

稳定行为以 [`SPEC.md`](../../SPEC.md)、[`ARCHITECTURE.md`](../../ARCHITECTURE.md) 和
[公开工作流文档](../configuration/workflows.md)为准。

## 过去讨论过、但没有排期的想法

下列内容不是当前 Evaluation 的缺口，也不是一个待执行计划。只有出现真实使用者和
明确需求时，维护者才会把其中一项单独加入 [`ROADMAP.md`](../../ROADMAP.md)。

| 保留的想法 | 对用户可能有什么用 | 何时才值得重新讨论 |
| --- | --- | --- |
| 缓存参考数据的特征 | 多次计算 FID/KID 时少做重复工作 | 实际评估中，参考特征计算已成为主要耗时 |
| 正式记录速度和显存 | 把质量、耗时、吞吐量和显存放进可复现的测试报告 | 有具体模型选择需要同时比较质量和速度 |
| 比较结果并生成报告 | 把多个不可变结果整理成表格或发布报告 | 至少一个真实发布流程反复编写相同的比较代码 |
| 新任务的评估方案 | 为超分辨率、latent 或 consistency 定义各自的样本和指标 | 对应任务被选中实施；它随任务一起交付，不属于通用 Evaluation 待办 |
| 发送结果到 W&B 或 MLflow | 在外部系统中展示已经发布的结果 | 有维护者愿意负责一个可选 extension，且网络失败不影响本地结果 |

这些想法的旧设计、失败边界和历史讨论保存在
[Evaluation 设计备忘](notes/post-training-evaluation-support-plan/design-notes.md)中。保存它们
只是为了不丢失思路，不表示已经承诺实现。

## 本页何时需要更新

- 当前 `evaluate` 命令、subject 类型或结果格式发生变化时，先更新规范和公开文档，再
  同步本页的使用说明；
- 某个保留想法被明确加入路线图时，为它建立一个能说明实际输入、操作和输出的功能
  计划；
- 在此之前，Evaluation 保持“已完成”，不出现在候选或暂停的产品排期中。

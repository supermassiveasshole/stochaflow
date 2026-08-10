# 训练指标、诊断与模型选择

训练只记录 loss 时，我们能看出优化是否在推进，却很难回答更具体的问题：预测误差
是否真的下降、重建质量在哪一段变好、哪一个 checkpoint 应该保留，以及最终报告能否
在同一份数据上重放。Stochaflow 没有把这些问题都塞进一个“指标”接口，因为它们读取的
数据、运行的时机和能否影响模型选择并不相同。

## 看曲线、做额外检查和正式评估是三件事

普通 phase Metric 跟随现有的 train、validation 或 test 数据循环工作。它只累计
`TrainingStrategy` 已经计算出的值，不会为了算指标再次调用模型、额外采样或读取另一份
数据。例如，validation MAE 可以随每个 validation batch 更新，最后得到一条
`valid/metrics/...` 曲线。

Training Diagnostic 用于训练期间的额外观察，例如固定 seed 生成样本网格、测量一次
采样耗时或保存重建图。它可以按自己的间隔运行并写日志或运行产物（artifact），但不会
进入 epoch history、`best.pt` 或 early stopping。Diagnostic 即使读取了 validation batch，
也不会因此变成正式的验证结果。

需要完整采样、固定样本数量或数据完整性检查时，应使用 Evaluation。训练内的
epoch-end validation Evaluation 可以每隔若干 epoch 对当前 raw 或 EMA 权重运行完整协议，
再把 FID/KID 等 `valid/metrics/...` 结果交给现有模型选择逻辑。独立的
`stochaflow evaluate` 则冻结 checkpoint、数据和协议，发布可审计、可离线重放的最终结果。

在扩展代码里还会看到两个更靠近训练步骤的概念：`loss` 是参与反向传播的标量，
`TrainStepOutput.metrics` 是低成本、无状态的 batch 报告。只有
`TrainStepOutput.metric_updates` 会进入跨 batch 累计的 phase Metric。名字相似不代表
它们拥有相同的生命周期。

## 从一个普通 validation 指标开始

下面的配置使用 Gaussian 训练已经产生的 prediction/target，把 MAE 分别累计到
validation 和 test。它不增加模型调用：

```yaml
metrics:
  - id: prediction_mae
    name: mae
    channel: gaussian.prediction_target
    phases: [validation, test]
    params: {}

trainer:
  early_stopping:
    enabled: false
    monitor: valid/metrics/prediction_mae
    mode: min
    patience: 8
    min_delta: 0.001
```

`channel` 可以理解为 Strategy 交给 Metric 的一包参数。核心框架不知道 batch 中哪个
Tensor 是 prediction 或 target；具体 Strategy 解释 batch，再按一个明确名称发出参数。
当前内置训练 Strategy 提供这些 channel：

| 训练任务 | Channel | 交给 Metric 的值 |
| --- | --- | --- |
| `supervised` | `supervised.prediction_target` | prediction、target |
| Gaussian denoising | `gaussian.prediction_target` | 模型预测、当前 prediction type 对应的训练 target |
| Gaussian denoising | `gaussian.clean_reconstruction` | 由预测恢复的干净样本、原始干净样本 |

内置 `mse` 和 `mae` 都消费 prediction/target 形式的 channel。`mean` 用于显式的 scalar
及可选权重。`fid` 和 `kid` 是图像分布指标，通常由具体任务的 EvaluationBuilder 提供
real/fake image updates；它们不会自动接到 Gaussian batch 上，并且需要安装 `quality` 依赖。

Stochaflow 不把任意 `torchmetrics.*` 类名当成配置接口。任务需要分类指标、逐类别结果
或其他参数形状时，extension 应注册一个稳定的 Metric 名称，并让自己的 Strategy 或
EvaluationBuilder 提供匹配的 channel。完整示例见
[按类别验证与自定义 Metric](../tutorials/class-metrics.md)。

## 一轮数据怎样变成曲线和 checkpoint

每条 Metric 声明在每个 phase 都有独立实例，因此 train、validation 和 test 不会共享
累计状态。训练 phase 先暂存一次 optimizer window 的更新；只有 backward 和 optimizer
step 成功后才提交，混合精度 overflow 等被跳过的 window 不会污染统计。validation 和
test 没有 optimizer commit 点，在一次 evaluation step 成功后立即更新。

phase 结束时，结果会变成普通标量，并使用稳定名称：

```text
train/loss
valid/loss
test/loss
train/metrics/<metric-id>[/<subkey>]
valid/metrics/<metric-id>[/<subkey>]
test/metrics/<metric-id>[/<subkey>]
```

如果一个 Metric 返回逐类别等平面 mapping，subkey 会接在 metric id 后面。例如
`valid/metrics/class_recall/macro`。Logger、训练 history 和 checkpoint 使用同一套名称，
TensorBoard 不会再发明另一套 tag。

模型选择只接受 `valid/loss` 或 `valid/metrics/<id>[/<subkey>]`。`enabled: false` 只关闭
提前终止；只要存在 validation，配置的 monitor 仍用于更新 `best.pt`。把它改为
`enabled: true` 后，同一个 monitor 还会按 `patience` 和 `min_delta` 决定何时停止。
train、test、system 和 `diagnostics/...` 数值都不能控制模型选择。普通 validation monitor
每个 epoch 都必须存在且为有限值；没有 validation DataLoader 时不会伪造 best checkpoint。

v12 checkpoint 保存完成 epoch 的 canonical scalar mapping、monitor policy 和恢复
best/early-stopping 所需的状态，但不保存 Metric 实例、尚未完成的 phase state 或
`MetricUpdate` payload。Strict resume 会从下一个完整 epoch 重新构造 Metric，并核对保存的
配置和选择政策。

## 生成质量需要完整协议

FID、KID 或超分辨率质量通常不能只看训练 batch：它们可能需要固定的 real/fake 配对、
精确样本数、稳定 sample ID 和完整生成过程。把这些工作伪装成普通 phase Metric，会让
Metric 同时接管采样、数据和 checkpoint，最终无法说明结果究竟比较了什么。

训练期间确实需要用生成质量选择 checkpoint 时，配置
`trainer.validation_evaluation`。它只在声明的 epoch 到期时产生新 observation；未到期的
epoch 不复用旧值，也不推进 patience。最终 benchmark 仍应在唯一 checkpoint 选定后运行
独立 Evaluation，不能用 test 结果反过来改选模型。完整配置见
[训练内 epoch-end validation Evaluation](workflows.md#训练内-epoch-end-validation-evaluation)
和[独立 checkpoint Evaluation](workflows.md#独立-checkpoint-evaluation)。

需要查 YAML 字段时使用[配置字段参考](reference.md#metrics)；需要查看曲线和
排查 event 文件时使用 [TensorBoard 指南](../tutorials/tensorboard.md)；需要实现新的 Metric、
channel 或 EvaluationBuilder 时再进入 [Extension 公共 API](../api/extensions.md#metrics)。

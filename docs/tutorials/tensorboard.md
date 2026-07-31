# 使用 TensorBoard 观察训练

Stochaflow 的基础安装已经包含 TensorBoard。内置 MNIST 配置默认同时启用本地日志和
TensorBoard，因此不需要修改 Python 代码，也不需要额外安装可选依赖。

## 启用日志后端

配置中的 `logging.backends` 决定训练指标写往哪里。下面的配置会保留本地
`metrics.jsonl` 和 `train.log`，同时生成 TensorBoard event 文件：

```yaml
logging:
  log_every: 100
  backends:
    - name: local
      params:
        console: false
        append: false
    - name: tensorboard
      params: {}
```

`log_every` 是训练 batch 指标的记录间隔。每轮汇总、验证指标和到期的 diagnostic
指标不依赖这个间隔。需要自定义目录名时，可以给 TensorBoard 后端传入
`params: {subdir: tensorboard}`；默认值就是 `tensorboard`。

## 恢复时启用固定质量监控

如果 checkpoint 来自未启用 TensorBoard 或使用了另一套 diagnostics 的训练，可以在
strict resume 启动时加载 observation-only 配置：

```powershell
uv run stochaflow train `
  --resume outputs/mnist/<run>/checkpoints/latest.pt `
  --observability-config `
  examples/built-in/image-generation/configs/overlays/mnist-observability.yaml
```

该示例完整替换 diagnostic 列表，使用 EMA、固定 `seed: 123`、32 个样本和确定性
DDIM-50，并记录固定 timestep 的 `x0` 重建指标与面板。它把 logging backends 替换为
local 与 TensorBoard，但没有声明 `log_every`，所以继续使用 checkpoint 中的间隔。
observability config 顶层只允许 `diagnostics` 和 `logging`，不能改变模型、数据、
optimizer、scheduler、EMA、训练进度或 extension selection。
这仍是训练侧监控配置；`configs/sample/` 下的 DDPM/DDIM profile 只影响独立的
checkpoint-backed inference，不拥有或覆盖训练 diagnostics。

diagnostic 与 logger 不恢复运行时状态：resume 会创建新的兄弟 run、新的 logger 和新的
TensorBoard event 文件。旧 run/event 不会被热加载、重开或续写。把 TensorBoard
`--logdir` 指向共同的 experiment output root，即可把恢复前后的两个时间戳 run 放在
同一页面比较。生效配置、overlay 文件 SHA-256 和显式 logging 字段都会固化到新 run
manifest 与 checkpoint metadata，便于确认曲线使用的监控协议。

## 启动训练和 TensorBoard

先启动训练：

```powershell
uv run stochaflow train `
  --config examples/built-in/image-generation/configs/train/mnist.yaml
```

终端表格中的 `Output` 是本次运行目录，例如
`outputs/mnist/20260725_203859`。在另一个 PowerShell 窗口中查看这一组实验：

```powershell
uv run tensorboard --logdir outputs/mnist --port 6006
```

浏览器打开 <http://localhost:6006>。TensorBoard 可以在训练仍在进行时读取新数据。
若只想查看一次运行，把 `--logdir` 收窄到该运行的 `tensorboard` 目录：

```powershell
uv run tensorboard `
  --logdir outputs/mnist/20260725_203859/tensorboard `
  --port 6006
```

一次运行的 event 文件实际位于
`<output>/<timestamp>/tensorboard/<experiment-name>/`。将 `--logdir` 指向更高一层
的 `outputs/mnist`，可以在同一页面勾选和比较多个时间戳运行。

## 重点查看哪些面板

在 **Scalars** 中优先观察：

| 指标 | 含义 | 使用建议 |
| --- | --- | --- |
| `train/loss` | 每个训练轮次的平均损失 | 看长期下降趋势，不要只看单个 batch |
| `valid/loss` | 验证集平均损失 | 用于发现过拟合和选择 checkpoint |
| `best/valid/loss` | 当前最佳验证损失 | 应随有效改进阶梯式下降 |
| `train/lr/group_0` | 第一个参数组的学习率 | 核对 warmup 和 cosine 曲线是否符合配置 |
| `train/step/loss` | 按 `log_every` 记录的 batch 损失 | 波动较大，适合发现突发异常或 NaN |
| `system/train/duration_seconds` | 单轮训练耗时 | 对比吞吐变化和数据加载瓶颈 |
| `valid/metrics/prediction_mae` | 验证集上的预测目标 MAE | 补充观察去噪预测误差，不代替生成质量评价 |
| `valid/metrics/clean_reconstruction_mse` | 验证集上的干净图像重建 MSE | 观察由当前噪声状态恢复 `x0` 的误差 |

建议对 loss 使用适度平滑，但比较最低点和尾段趋势时同时查看原始曲线。学习率曲线
不应开启过强平滑，否则 warmup 拐点会被掩盖。

启用了 `diffusion_quality` diagnostic 后：

- **Images** 会显示 `diagnostics/denoiser/reconstruction` 以及各 sampler profile 的
  `samples`；仅在配置中启用轨迹时才会显示 `trajectory`；
- **Scalars** 会增加 timestep 分桶损失、噪声统计、重建 MSE/PSNR、采样延迟、
  样本多样性等 `diagnostics/...` 指标；
- **Text** 中的 `config` 是本次运行解析后的配置，适合核对实验差异。

MNIST 默认监控协议使用 EMA 权重、固定 seed 和确定性 DDIM-50，每 10 轮生成同一组
32 个潜变量对应的样本网格。这样相邻 checkpoint 的变化来自模型本身，而不是随机初始
噪声。训练期间每 100 step 还会记录：

- 10 个 timestep 区间的去噪损失，用来定位模型在哪段噪声强度上学习不足；
- 预测噪声与目标噪声的均值、标准差和误差；
- `t=50/250/500/750/900` 的 `x0` 重建 MSE/PSNR，用来区分低噪声细节恢复和高噪声
  结构恢复能力。

诊断失败策略设为 `warn`，避免可视化故障终止长时间训练；采样轨迹默认关闭，因为它
写入量大且不利于跨 epoch 比较。MNIST 配置也不默认启用基于 ImageNet 特征的 KID/FID，
这些指标不能可靠判断生成图是否像一个合法数字。生成质量仍需结合固定样本网格判断，
不能只看训练 loss 或样本均值/方差。

训练配置中的普通 phase metrics 会在完整 validation epoch 上聚合，并在恢复 best
checkpoint 后对 test split 再计算一次；对应 test tag 使用 `test/metrics/...`。它们与
按 cadence 运行、可能产出图像 artifact 的 `diagnostics/...` 是两条独立通道。

## 常见问题

### 页面显示 “No dashboards are active”

先确认 event 文件存在：

```powershell
Get-ChildItem outputs/mnist -Recurse -Filter "events.out.tfevents.*"
```

如果没有文件，检查运行所用 YAML 的 `logging.backends` 是否包含
`name: tensorboard`，并确认终端显示的 `Output` 与 `--logdir` 指向同一实验根目录。

### 端口 6006 已被占用

换一个端口即可：

```powershell
uv run tensorboard --logdir outputs/mnist --port 6007
```

### 曲线更新不及时

训练进程会持续写入 event 文件，正常情况下稍等自动刷新即可。训练被强制终止时，
最后一小段缓冲数据可能尚未 flush；本地 `metrics.jsonl` 和 `train.log` 可用于交叉
检查。正常训练结束或受控中断会关闭 logger 并刷新 writer。

### 多次实验名称看起来相同

TensorBoard 的 run 名来自 event 文件所在目录。把 `--logdir` 指向
`outputs/mnist` 时，时间戳目录仍能区分各次运行；可在左侧 run 选择器中只勾选
需要比较的实验。为关键实验保留配置、checkpoint 和对应时间戳，避免只按曲线颜色
识别运行。

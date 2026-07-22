# 纵向扩展参考项目

仓库在 `examples/extension-projects/` 提供两个彼此独立、可安装的 Python distribution。
它们不是新的核心 recipe，也不限制用户必须采用相同 repo、包管理器或目录结构；它们用于
证明现有 Registry、DataBuilder、TrainingBuilder/Strategy、checkpoint 与
SamplingBuilder/Writer 边界能够在不修改 runner 的前提下承载完整任务。

两个项目都必须先安装到 `stochaflow` CLI 所在 Python environment，再由 YAML 中的
`extensions.plugins` 显式选择。配置中的 `data/`、`outputs/` 等相对路径按命令启动目录
解析，因此示例命令应从对应项目根目录执行。

## Physics reconstruction

`physics-reconstruction/` 展示 Kolmogorov flow reconstruction：

- DataBuilder 通过 memory-mapped `[trajectory, time, height, width]` `.npy` 数据按
  trajectory 划分，并生成连续三帧样本；
- 项目模型以 persistent buffers 保存 normalization 与 PDE 常数；
- 自定义 TrainingStrategy 解释原始物理场 batch，复用离散 Gaussian Process、
  `gaussian_training_target()` 和通用 Objective；
- baseline SDEdit 从 sparse observation 构造 partial-noised initial state，并直接复用
  内置 DDPM/DDIM；
- exact guided 路径使用项目自己的 narrow Dynamics 与 `guided-ddim` Sampler，在共享的
  DDIM transition 后、observer 事件前施加 physics correction；
- 项目 writer 逐 batch 写入预分配 `.npy`，不在 writer 内再次拼接全量 field。

生产配置遵循真实 schema `[40, 320, 256, 256]`：前 36 条 trajectory 形成 11448 个
训练窗口，后 4 条形成 1272 个 reconstruction 窗口。该数据没有独立 validation split；
Builder 允许实验显式提供 validation range，tiny 测试才使用它。训练数据和预训练权重
不随仓库分发，因此该项目验证扩展、数学和生命周期，不声明复现论文精度。

参考实现有两种不同的物理梯度：模型 condition 使用 residual mean-square，accepted
transition 后的 correction 使用按中心场能量归一化的 residual loss。项目保留两者，且
主配置关闭 `clip_denoised`，因为标准化涡量不服从图像的 `[-1, 1]` 范围。

partial-noise 配置使用 Stochaflow 的 public state time。比如 `t=240, r=30` 的 DDIM
schedule 是 `[240, 232, ..., 8, 0]`，initial marginal 与第一个 reverse source 完全对齐；
这有意修正参考脚本中 initial alpha 与首个 solver coordinate 不一致的问题。首版不暴露
非零 CFG weight：参考 conditional wrapper 会自行计算 condition，第三 positional 参数在
无 label 模式下并不构成可靠的 conditional/unconditional CFG 语义。

Stage 7 还在真实 `[40, 320, 256, 256] float32` mmap 数据上完成了一个
`[1, 3, 256, 256]` MPS 容量 smoke：生产 `T=1000, t=240` 下的训练反传、两步 baseline
DDIM 和两步 guided DDIM 均成功。进程累计 RSS 高水位为 392,019,968 bytes；MPS current
allocation 在训练后为 16,060,416 bytes、两条采样后为 6,086,400 bytes，driver
allocation 约 1.15 GB。MPS 字段是同步后的 current/driver allocation，不是可重置的峰值；
该证据只证明真实单 batch shape/device/capacity，不代表完整 solver 延迟或科学精度。

## Frozen-teacher distillation

`knowledge-distillation/` 展示 frozen-teacher online distillation：

- DataBuilder 构建带标签的 deterministic classification batch；
- primary model 是 student，teacher 与 temperature-KL Objective 由 TrainingBuilder 构建；
- Builder 加载普通 PyTorch teacher `state_dict`、冻结 teacher，并以
  `ManagedTrainingModule(mode="eval")` 声明训练资产；
- Strategy 只执行 student/teacher forward、task loss 与 distillation loss 组合；
- checkpoint 将 teacher 和额外 Objective 保存到稳定命名的
  `training_assets_state_dict`；
- sampling-only Builder 只构建 student，既不构建 teacher，也不读取其 bootstrap 文件。

训练 resume 时，TrainingBuilder 仍需先用 bootstrap 文件构造结构兼容的 teacher；随后
checkpoint state 覆盖它并成为 runtime state 权威。bootstrap 因而是训练构造资源，而非
resume state 权威。sampling checkpoint view 不构造 TrainingBuilder，所以 student-only
sampling 可以完全脱离该文件。

## 安装与运行

两个目录都是普通 distribution，可以使用任意符合 Python packaging 标准的工具。例如：

```bash
cd examples/extension-projects/physics-reconstruction
python -m pip install -e ".[test]"
python -m stochaflow_physics_reconstruction.tools.prepare_tiny_data \
  --output-dir data/tiny
stochaflow train --config experiments/tiny/train.yaml
```

```bash
cd examples/extension-projects/knowledge-distillation
python -m pip install -e ".[test]"
python tools/create_teacher_bootstrap.py --output data/teacher.pt
stochaflow train --config experiments/tiny/train.yaml
```

仓库验收会从临时副本构建 Stochaflow 和两个扩展 wheel，在不向 `PYTHONPATH` 注入 checkout
源码的环境中运行 entry-point discovery、train、resume 和 checkpoint-only sample。reference
project 不会被打入 Stochaflow 自身 wheel。

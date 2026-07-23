# 纵向扩展参考项目

仓库在 `examples/extension-projects/` 提供两个彼此独立、可安装的 Python
distribution。它们不是核心内置 recipe，也不规定用户必须采用相同的仓库布局或包管理器。
它们用于展示一个任务如何同时使用 entry point、DataBuilder、TrainingBuilder/Strategy、
checkpoint、SamplingBuilder 和领域 writer，而不修改 Stochaflow runner。

| 项目 | 主要展示 | 不代表 |
| --- | --- | --- |
| `physics-reconstruction/` | 条件 Gaussian 训练、partial-noise reconstruction、复用内置 DDPM/DDIM、组合 DDIM primitive 的 physics-guided sampler、领域 artifact | 预训练模型、完整数据分发或论文精度复现 |
| `knowledge-distillation/` | frozen teacher、managed auxiliary assets、多个 Objective、student-only sampling | 通用蒸馏模式或效果 benchmark |

两个项目都必须先安装到 `stochaflow` CLI 所在的 Python environment，再由 YAML
`extensions.plugins` 精确选择。配置中的 `data/`、`outputs/` 等相对路径按进程启动目录
解析，因此下面的命令都从对应项目根目录运行。

## Physics reconstruction

这个项目展示三帧 Kolmogorov-vorticity reconstruction。项目代码拥有：

- mmap 数据读取、trajectory 范围划分和连续三帧 batch；
- conditional denoiser、normalization 与 PDE 常数；
- 将原始物理场 batch 转换为 Gaussian marginal/target 的 TrainingStrategy；
- 从 sparse observation 构造 partial-noised initial state 的 SamplingBuilder；
- 在普通 Gaussian prediction 上复用内置 DDPM/DDIM 的 baseline；
- 复用公开 DDIM schedule/transition primitive，并在 accepted transition 后施加 PDE
  correction 的项目 Sampler；
- 输出 `reconstructions.npy` 与领域指标的 artifact writer。

condition gradient 与 post-transition correction 是两种不同策略。前者可以封装在
Gaussian Dynamics 的模型 callable 中，继续使用内置 DDPM/DDIM；后者改变数值 transition，
因此由项目自己的 Sampler 实现。核心 Process、内置 Sampler 和顶层 YAML 都没有
physics-specific 分支。

### Tiny end-to-end

```bash
cd examples/extension-projects/physics-reconstruction
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m stochaflow_physics_reconstruction.tools.prepare_tiny_data \
  --output-dir data/tiny
stochaflow train \
  --config experiments/tiny/train.yaml \
  --skip-final-sample
```

使用生成的 run directory 运行三种独立采样策略：

```bash
stochaflow sample \
  --checkpoint outputs/tiny/<run> \
  --config experiments/tiny/sample-baseline-ddim.yaml \
  --output-dir outputs/sample-baseline-ddim

stochaflow sample \
  --checkpoint outputs/tiny/<run> \
  --config experiments/tiny/sample-baseline-ddpm.yaml \
  --output-dir outputs/sample-baseline-ddpm

stochaflow sample \
  --checkpoint outputs/tiny/<run> \
  --config experiments/tiny/sample-guided-ddim.yaml \
  --output-dir outputs/sample-guided-ddim
```

partial-noise time 与 reverse schedule 使用同一套 public state-time 语义：initial
marginal 的时间就是第一个 reverse source，最终到达 clean state `0`。

### 真实数据入口与证据边界

production 配置期望 `[40, 320, 256, 256]` 的 high-resolution `.npy` 数据和一个 sparse
`.npz` observation。项目 prepare tool 生成 mmap-ready observation、normalization
统计与严格 positional-alignment sidecar：

```bash
python tools/prepare_kolmogorov.py \
  --reference data/kolmogorov.npy \
  --sparse data/kolmogorov_sparse.npz \
  --sparse-key u3232 \
  --output-dir data \
  --held-out-trajectories 4 \
  --smoothing-kernel 7
```

仓库不分发数据、checkpoint 或训练输出。维护者参考环境曾使用真实 mmap 输入完成一次
optimizer update，以及 baseline/guided 的完整 30/40-step 单样本 runtime smoke；机器可读
记录位于 `benchmarks/results/stage7-physics-macos-arm64.json`。它只证明该环境中的
production path 可以执行，不代表收敛训练、1272-sample 全量运行、科学精度或跨平台容量。
大规模 output 与 trajectory 的限制见
[Sampling artifact 容量](sampling-capacity.md)。

## Frozen-teacher distillation

这个项目展示一个 deterministic classification student 如何使用冻结 teacher：

- TrainingBuilder 构建 teacher 与 temperature-KL Objective，加载普通 PyTorch
  bootstrap `state_dict`，冻结 teacher，并声明稳定命名的 managed assets；
- TrainingStrategy 只执行 student/teacher forward，并组合 task 与 distillation loss；
- checkpoint 把 teacher 和额外 Objective 写入 `training_assets_state_dict`；
- sampling-only Builder 只构建 primary student，不构建 teacher 或训练 Objective。

安装和首次训练：

```bash
cd examples/extension-projects/knowledge-distillation
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python tools/create_teacher_bootstrap.py --output data/teacher.pt
stochaflow train --config experiments/tiny/train.yaml
```

strict resume 使用 checkpoint 保存的完整配置。bootstrap 文件仍用于构造结构兼容的
teacher，随后 checkpoint state 覆盖它并成为 runtime 权威：

```bash
stochaflow train --resume outputs/tiny/<run-id> --epochs 3
```

checkpoint-only sampling 不构建 TrainingBuilder，因此可以在删除 bootstrap 文件后只加载
student：

```bash
stochaflow sample --checkpoint outputs/tiny/<run-id>
```

这条路径展示的是 frozen-teacher 组合边界。独立 teacher optimizer、交替更新、offline
teacher cache 或多 teacher policy 仍由具体 extension 或新的训练循环 family 负责。

## Packaging 与隔离验证

两个项目都通过标准 `[project.entry-points."stochaflow.extensions"]` 声明聚合注册模块。
安装后不需要源码扫描或 `PYTHONPATH` 注入。仓库测试会从临时副本分别构建 Stochaflow 和
extension wheel，在隔离环境中验证 entry-point discovery、train、resume 和
checkpoint-only sample；reference project 不会被打入 Stochaflow wheel。

要创建自己的最小项目，优先从：

```bash
stochaflow init my-research-project
```

开始，再按任务需要替换生成的 DataBuilder、model、TrainingBuilder/Strategy 和
SamplingBuilder。完整角色说明见[框架特性与架构](../framework.md)与
[扩展与 Registry](extensions.md)。

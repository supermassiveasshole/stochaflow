# 常用工作流

## 训练

通用入口是：

```bash
uv run stochaflow train --config configs/ddpm_mnist.yaml
```

runner 加载配置、应用 CLI 覆盖、创建一个时间戳 run 目录，调用一次 DataBuilder，
然后构建一套训练组件。所有训练参数见[CLI 参数索引](reference.md#cli-参数索引)。

配置解析本身不会导入 extension。runner 先根据 `extensions.plugins` 发现并预检已安装的
`stochaflow.extensions` entry points，再激活聚合注册模块、执行跨组件校验并开始构建。
省略 `extensions` 或写 `plugins: []` 不加载第三方代码；`plugins: null` 才表示选择当前
环境的全部插件。推荐在可复现配置中写确定的插件名列表。

### 创建扩展项目

```bash
stochaflow init my-research-project
cd my-research-project
python -m pip install -e ".[test]"
stochaflow train --config experiments/example/train.yaml
```

`init` 只生成普通、可安装的单 distribution repository；它不创建环境、运行安装命令或
覆盖非空目录，也不要求使用 `uv`。生成 repo 可以容纳多份 Stochaflow 实验和任意其他
研究代码，CLI 只运行显式配置选择的组件。扩展必须安装到 `stochaflow` CLI 所在 Python
environment。默认配置含相对 `data/` 和 `outputs/` 路径，应从项目根目录启动命令。
在提供安全 descriptor-relative 文件系统原语的平台，`init` 可写入已存在的空真实目录；
其他平台需删除该空目录，让 `init` 自行创建目标目录。

### Smoke run

快速验证 schema、数据下载、模型 forward、反向传播与 checkpoint 路径：

```bash
uv run stochaflow train \
  --config configs/ddpm_flowers102.yaml \
  --epochs 1 \
  --limit-batches 2 \
  --limit-validation-batches 1 \
  --skip-final-sample
```

`--limit-*` 只截断本次运行，不修改 YAML。若想测试最终采样，移除
`--skip-final-sample`；若 `sampling.builder` 为 null，训练后不执行默认采样。
`shape` 是否必需由具体 Builder 决定。

这些 smoke 覆盖也不会重写 LR scheduler 的 `T_max`、`total_steps` 或其他构造参数。
它们是具体 PyTorch scheduler 的显式配置，而不是框架可推断的通用 run length。若要运行
一份具有不同调度周期的完整实验，应同时修改 YAML，使 scheduler 参数与训练计划一致。

### CLI 覆盖优先级

新训练有效值按下列顺序决定，后者优先：

1. dataclass 默认值；
2. YAML；
3. `--device`、`--output-dir`、`--epochs` 等 CLI 覆盖；
4. runner 为本次 run 生成的 `experiment.exp_id` 和时间戳输出目录。

`--no-progress` 会关闭本次运行的进度条；`--deterministic` 启用 Torch 支持的确定性
行为。配置随机 seed 仍应固定，但跨设备、PyTorch 版本和第三方算子不保证逐位一致。
同理，`--epochs` 和 `--limit-batches` 不会按参数名猜测并改写 optimizer/scheduler
constructor kwargs。

strict resume 不再把 checkpoint config 与外部 YAML 当作两份可合并配置；它以 checkpoint
保存的 config/state 为唯一 base，只应用 device、output root、目标 epoch、progress 和
batch limits 等文档化安全运行时覆盖。

### Config 与 checkpoint 权威

Stochaflow 先为每个命令选择唯一 base config，再应用该 workflow 明确允许的覆盖；不会把
config、checkpoint 和 CLI 当作三份对等配置做通用 merge。

| Workflow | 权威 base config | Checkpoint 角色 | 后续覆盖 |
| --- | --- | --- | --- |
| `train --config ...` | 外部完整 config | 无 | train CLI flags |
| `train --resume ...` | checkpoint config | 完整训练 state | 安全 train runtime flags |
| `sample --checkpoint ...` | checkpoint config | 推理 state | sample CLI flags |
| 完整 config sampling | 外部完整 config | 推理 state | sample CLI flags |
| sampling-only overlay | checkpoint config | 推理 state | 完整 `sampling`/显式 `extensions.plugins`，再加 sample CLI flags |

config 字段覆盖进入 resolved config；`limit-batches`、deterministic、启动 cwd、lineage、
skip-final-sample 和插件 version acceptance 等 invocation 事实进入独立 manifest，而不是
扩张组件 schema。

## 输出目录

新训练在 `experiment.output_dir` 下创建唯一时间戳目录：

```text
outputs/<experiment>/<YYYYMMDD_HHMMSS>/
  checkpoints/
    best.pt
    latest.pt
    epoch_XXXX.pt
  metrics.jsonl
  train.log
  resolved_config.yaml
  run_manifest.yaml
  diagnostics/
    diffusion_quality/
      epoch_NNNN/
        manifest.yaml
        denoiser/
        <sampler-profile>/
  samples/
    final/
      samples.png
      samples.pt
      resolved_sampling.yaml
```

`resolved_config.yaml` 只保存最终可重建组件配置；`run_manifest.yaml` 另外记录 config
来源、实际插件 provenance、version acceptance、启动 cwd、runtime-only CLI options 和
checkpoint lineage。启用 TensorBoard、W&B、diagnostic 或 trajectory 后会增加对应
子目录/artifact。
`artifacts.checkpoint_every` 控制编号 checkpoint 的频率；`latest.pt` 和 `best.pt`
按恢复与模型选择规则更新。

## 恢复训练

严格恢复必须显式指定 checkpoint 文件或 run directory：

```bash
uv run stochaflow train \
  --resume outputs/ddpm_mnist/<run>/checkpoints/latest.pt
```

`--resume` 与 `--config` 互斥。checkpoint v8 保存 resolved config、primary inference
model、可选 Process/Objective、可选 EMA model、optimizer、scheduler、EMA、具名
training assets 和训练进度。它只保存
`data: {name, params}`，不保存 Dataset、PyTorch Sampler、DataLoader、partition 或数值
solver 的运行时状态。
恢复时 checkpoint config 用于重建完全相同的训练资产，随后加载完整
optimizer/scheduler/EMA/progress state；checkpoint 会校验 concrete class identity、具名资产
拓扑和 scheduler state 有无。更换模型、TrainingBuilder、optimizer、scheduler 或其构造
参数属于以已有权重开始的新训练，而不是 resume；weights-only warm start 是未来独立
workflow，不属于 `--resume`。

每次 resume invocation 都创建新的 `exp_id` 和兄弟 run directory，而不是重开并覆盖旧
输出。默认 output root 取原 resolved run directory 的父目录，`--output-dir` 可以覆盖；
epoch/global step 与训练 state 连续，lineage 记录在新 run manifest 和 checkpoint metadata。
严格恢复还延续 best metric 与 early-stopping wait；因此 `latest.pt` 或 epoch checkpoint 需要
同一 `checkpoints/` 目录中的 `best.pt`。推荐直接传 run directory。单独移动一个
`metadata.checkpoint_kind: best` 的 best checkpoint 仍可恢复；单独移动 latest/epoch 会因
缺少全局 best 权重而拒绝，而不是把 latest 错当成 best。候选 best 的 epoch、metric、
monitor、mode、resolved config 和 extension provenance 必须与所选 checkpoint 一致；因此
已被未来 epoch 覆盖的 mutable `best.pt` 或来自另一 run 的同形状权重都不能用于恢复。通过
校验且能载入当前资产拓扑的 inherited best 会在训练开始前原子物化到新 run 的
`checkpoints/best.pt`，并记录当前 resolved config/provenance；后续恢复和 sampling 不依赖
父 run。
Strict resume 还要求合法的 `epoch`、`global_step` 和 v8 RNG snapshot，并在 selected state
与 inherited-best 全部验证后恢复 Python、NumPy、Torch CPU 及适用的 CUDA RNG。普通
checkpoint load 不修改全局 RNG。sampling 不恢复 checkpoint RNG snapshot，而是按
`sampling.seed`（为 `null` 时使用 `experiment.seed`）重新初始化 Python、NumPy 与 Torch
全局 RNG。device override 仍受支持，但跨设备或 CUDA topology 不保证逐位一致。

checkpoint 不保存 DataBuilder、Dataset、DataLoader iterator/worker、Sampler 或用户私有
generator 的 runtime state。自定义随机 loader 应由 experiment seed 和 epoch 确定，并在需要
时响应 duck-typed `set_epoch(epoch)`；核心只承诺 epoch-boundary 的全局 RNG 与训练资产恢复。
若 checkpoint config 保存的是相对 output path，它仍以本次进程启动 cwd 解释；从其他目录
恢复时应显式传 `--output-dir`。DataBuilder 私有 params 中的相对路径遵循同一 cwd 规则，
核心不会猜测并重写不透明字段。

v8 payload 只允许 Tensor、primitive 与普通 container，并始终由
`torch.load(..., weights_only=True)` 读取。扩展代码/class 不会 freeze 在 checkpoint 中；
恢复环境需要安装记录的 entry-point distribution。实现变化造成的不兼容由 state/资产契约
报错，Stochaflow 不保存或迁移第三方源码。

## K-fold

K-fold 是支持它的图像 recipe 的私有能力。一次配置只运行一个 fold，例如第三个 fold
（索引从 0 开始）：

```yaml
data:
  name: image
  params:
    source:
      kind: image_folder
      path: ./data/images
    partition:
      mode: kfold
      num_folds: 5
      fold_index: 2
    image:
      size: [64, 64]
```

`fold_index` 不可省略。运行全部五折需要五次独立运行，通常由项目脚本或外部 sweep
分别覆盖这个字段。每次运行独立构建 DataBuilder、模型、optimizer、日志和 checkpoint；
核心 runner 不展开 fold，也不跨 fold 共享训练状态。

## 训练期多 sampler diagnostic

`diffusion_quality` 在相同 EMA 权重、初始噪声和固定 seed 下并排运行多个 sampler。
轻量 denoiser 指标、sampler 指标、图片和参考质量算法都是独立 provider；配置可以
替换或禁用任一类别，而无需修改 diagnostic orchestrator。每个 sampler profile 必须
提供稳定且唯一的 `id`，`name` 对应 samplers Registry。

```yaml
diagnostics:
  - name: diffusion_quality
    params:
      modules: []
      cadence:
        step_every: 100
        artifact_every_epochs: 5
      sampling:
        sample_num: 16
        batch_size: 16
        seed: 123
      use_ema: true
      failure_policy: raise
      samplers:
        - id: ddpm_full
          name: ddpm
          params: {}
          trajectory:
            enabled: true
            every_steps: 100
            gif_fps: 8
        - id: ddim_50
          name: ddim
          params: {num_inference_steps: 50, eta: 0.0}
          trajectory:
            enabled: true
            every_steps: 5
            gif_fps: 8
      providers:
        step_metrics:
          - name: timestep_bucket_loss
            params: {buckets: 10}
          - name: noise_alignment
            params: {}
          - name: x0_reconstruction
            params: {timesteps: [50, 250, 500, 900]}
        sampler_metrics:
          - name: sample_statistics
            params: {}
          - name: sampling_performance
            params: {}
        denoiser_artifacts:
          - name: reconstruction_panel
            params:
              timesteps: [50, 250, 500, 900]
              max_samples: 16
        sampler_artifacts:
          - name: sample_grid
            params: {nrow: 4}
          - name: trajectory
            params: {}
      reference:
        enabled: false
        every_epochs: 20
        num_real: 2048
        num_fake: 2048
        batch_size: 64
        metrics:
          - name: kid
            params: {subsets: 100, subset_size: 1000}
          - name: fid
            params: {}
```

`modules` 会在解析 provider 名称前按顺序导入第三方模块。某个 `providers` 分类省略时
使用内置组合，显式写成 `[]` 时禁用该分类。provider 名称、构造参数、重复项和输出
冲突都会严格校验。

Local logger 记录 artifact 路径，TensorBoard 和 W&B 同时显示 PNG。启用 KID/FID
前需要 `uv sync --extra quality`，并且本次训练必须有 validation DataLoader。参考指标
只用于监控，不参与 best checkpoint 或 early stopping。

`failure_policy: raise` 会让采样或 provider 异常终止训练；`warn` 按 provider/profile
隔离失败、继续执行其余组件，并把错误同时写入 `diagnostics/system/error_count`、日志
和 epoch manifest。未知 sampler/provider、重复名称、缺少 trajectory 接口等配置错误
始终在训练开始前失败。

## checkpoint 采样

checkpoint 内含模型和训练配置，所以可以只给 checkpoint：

```bash
uv run stochaflow sample \
  --checkpoint outputs/ddpm_mnist/<run>/checkpoints/best.pt
```

切换 Sampler 或修改 solver 参数时，提供一份外部 YAML，修改其中完整的 sampling 段：

```yaml
sampling:
  shape: [1, 32, 32]
  num_samples: 64
  batch_size: 16
  builder:
    name: standard_denoising
    params:
      weights: auto
      prediction_type: epsilon
      clip_denoised: true
      sampler:
        name: ddim
        params: {num_inference_steps: 100, eta: 0.0}
      trajectory: {enabled: true, every_steps: 5}
  writers:
    - {name: tensor, params: {}}
    - {name: image, params: {grid_nrow: 8, gif_fps: 8}}
```

```bash
uv run stochaflow sample \
  --checkpoint outputs/ddpm_mnist/<run>/checkpoints/best.pt \
  --config path/to/sampling.yaml
```

CLI 不提供 sampler-specific flags；Sampler 参数完全属于 Builder。这样自定义 Builder
可以组合 condition、多个 Sampler 或非固定 shape initial state，而无需扩充核心 CLI。

也可以只给外部配置，让 runner 在 `experiment.output_dir` 下寻找最新 `best.pt`：

```bash
uv run stochaflow sample --config configs/ddpm_mnist.yaml
```

同时给出两者时，有两种明确输入形态：

- 只含 `sampling` 与可选 `extensions` 的 lightweight overlay：checkpoint config 是 base，
  overlay 的整个 `sampling` 段替换它；
- 完整 Stochaflow config：外部 config 整体权威，checkpoint 只提供 state。

sampling 不恢复 optimizer/scheduler state，因此外部配置可以改变 `num_samples`、
`batch_size`、`shape`、SamplingBuilder、Sampler/solver 参数、trajectory、writers 以及
raw/EMA 选择。核心不比较两份完整 config 或根据字段名推断兼容性；最终配置构建的
model/Process 必须能严格加载 checkpoint state，否则由正常 state contract 报错。

lightweight overlay 中 `extensions: {}` 保留 checkpoint 的插件 selection；只有 raw YAML
明确含 `extensions.plugins` 时才完整替换，不执行追加/去重 merge。完整外部 config 也按
自己的 selection 激活插件。若复用 checkpoint provenance 中同名插件，name/distribution/
target 必须保持 identity；仅 version mismatch 可以在导入前由交互式确认或
`--force-extension-version-mismatch` 接受。

## 采样形状与 EMA

`standard_denoising` 与 image diagnostic 使用 `sampling.shape`，它不含 batch 维且与
DataBuilder 独立；自定义 Builder 可以在 shape 为 null 时运行。`ema.enabled` 与
`ema.use_for_sampling` 同时为 true 且 checkpoint 含 EMA 时，
采样优先使用 EMA 权重。

每次采样都写 `resolved_sampling.yaml`，其中记录完整最终 config、config source、实际插件
provenance/version acceptance、checkpoint lineage、启动 cwd、runtime options，以及
Builder metadata/artifacts。实际 Process 声明或 `process: null` 也保留在最终 config 中。
`sampling.writers` 决定其他输出：`tensor`
写 PT，`image` 写 PNG/GIF；开启 trajectory 后，两者会写各自支持的 trajectory
artifact。

所有注册 Sampler 通过相同的完整 `sample(dynamics, initial_state, ...)` 生命周期执行，但
不共享万能数学接口。内置 DDPM/DDIM 要求 Gaussian Dynamics；其他算法 family 可定义
自己的 Dynamics capability，并由所属 Builder 与 Sampler 在调用边界验证。
trajectory 是 observer 对 initial、accepted step 和唯一 final observation 的抽样，不会
改变 solver 循环。保留的 state 在 observation 到达时复制，内置 Tensor 路径会立即转存到
CPU，避免后续原地更新污染历史或让显存随 trajectory 长度增长。`trajectory.pt` 按声明
顺序保存 step index、coordinate 和 state。

## 本地文档

安装并构建文档站点：

```bash
uv sync --extra docs
uv run python tools/generate_config_reference.py --check
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

开发时自动重建：

```bash
uv run sphinx-autobuild docs docs/_build/html
```

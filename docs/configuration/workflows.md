# 常用工作流

## 训练

通用入口是：

```bash
stochaflow train --config path/to/train.yaml
```

runner 加载配置、应用 CLI 覆盖、创建一个时间戳 run 目录，调用一次 DataBuilder，
然后构建一套训练组件。所有训练参数见[CLI 参数索引](reference.md#cli-参数索引)。
源码 checkout 中可以把命令写成 `uv run stochaflow ...`，并直接使用仓库内
`configs/` 示例；发布 wheel 不包含这些 repo-local 配置。

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

仓库中的 [纵向扩展参考项目](reference-projects.md) 进一步展示一个领域任务如何在自己的
distribution 内组合 DataBuilder、TrainingBuilder/Strategy、SamplingBuilder、Sampler 和
Writer，以及 frozen-teacher 资产如何严格 resume。它们是普通示例包，不是新的核心模板。

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

`--no-progress` 会关闭本次运行的进度条；`--deterministic` 启用 PyTorch 的严格
deterministic-algorithm 模式，遇到没有确定性实现的算子会报错，而不是静默使用非确定性
kernel。配置随机 seed 仍应固定，但跨设备、PyTorch 版本和第三方算子不保证逐位一致。
同理，`--epochs` 和 `--limit-batches` 不会按参数名猜测并改写 optimizer/scheduler
constructor kwargs。

strict resume 不再把 checkpoint config 与外部完整 YAML 当作两份可合并配置；它以
checkpoint 保存的 config/state 为唯一 base，只应用 device、output root、目标 epoch、
progress 和 batch limits 等文档化安全运行时覆盖。可选的
`--observability-config` 只有一个更窄的职责：在不改变任何训练资产或恢复状态的前提下，
替换本次兄弟 run 的 diagnostics 和显式 logging 字段。

### Config 与 checkpoint 权威

Stochaflow 先为每个命令选择唯一 base config，再应用该 workflow 明确允许的覆盖；不会把
config、checkpoint 和 CLI 当作三份对等配置做通用 merge。

| Workflow | 权威 base config | Checkpoint 角色 | 后续覆盖 |
| --- | --- | --- | --- |
| `train --config ...` | 外部完整 config | 无 | train CLI flags |
| `train --resume ...` | checkpoint config | 完整训练 state | 安全 train runtime flags；可选 observability config |
| `sample --checkpoint ...` | checkpoint config | 推理 state | sample CLI flags |
| 完整 config sampling | 外部完整 config | 推理 state | sample CLI flags |
| sampling-only overlay | checkpoint config | 推理 state | 完整 `sampling`/显式 `extensions.plugins`，再加 sample CLI flags |

config 字段覆盖进入 resolved config；`limit-batches`、deterministic、启动 cwd、lineage、
skip-final-sample 和插件 version acceptance 等 invocation 事实进入独立 manifest，而不是
扩张组件 schema。

目录输入和默认输出遵循下表；显式 checkpoint 文件始终按原路径使用：

| 调用 | 目录中选择的 checkpoint | 默认输出 |
| --- | --- | --- |
| `train --resume <run-or-root>` | 递归查找最近修改的 `checkpoints/latest.pt` | 在原 run 的 output root 下创建新的兄弟 run |
| `sample --checkpoint <run-or-root>` | 递归查找最近修改的 `checkpoints/best.pt` | `<checkpoint-run>/samples/<timestamp>/` |
| `sample --config <complete-config>` | 在 `experiment.output_dir` 下查找最近修改的 `best.pt` | `<checkpoint-run>/samples/<timestamp>/` |

`--output-dir` 会覆盖对应命令的输出位置。sampling 目录总会写
`resolved_sampling.yaml`；训练 run 总会写 `resolved_config.yaml` 和
`run_manifest.yaml`。

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
来源、实际插件 provenance、version acceptance、启动 cwd、runtime-only CLI options、
checkpoint lineage 和 `selected_components`。同一 component identity 摘要也写入 checkpoint
metadata；sampling 的 `resolved_sampling.yaml` 根据最终 sampling config 写入同样的字段。
它只列出 typed 顶层配置所选择的 framework component names（可选项显式为 `null`，列表
保持声明顺序），不会递归解释 Builder/Process 的私有 `params`，也不表示 sampling
invocation 实际构建了训练或数据组件。完整 config 仍是重建权威，摘要只用于审计。
启用 TensorBoard、W&B、diagnostic 或 trajectory 后会增加对应子目录/artifact。
`artifacts.checkpoint_every` 控制编号 checkpoint 的频率；`latest.pt` 和 `best.pt`
按恢复与模型选择规则更新。

## 恢复训练

严格恢复必须显式指定 checkpoint 文件或 run directory：

```bash
stochaflow train \
  --resume outputs/ddpm_mnist/<run>/checkpoints/latest.pt
```

### Strict Resume Observability Overlay

长训练恢复时，可以保留全部 checkpoint 训练语义，只为新 invocation 配置监控：

```bash
stochaflow train \
  --resume outputs/ddpm_mnist/<run>/checkpoints/latest.pt \
  --observability-config configs/overlays/mnist_observability.yaml
```

`--observability-config` 只限 strict resume，与 fresh `--config` training 同用会失败。
YAML 顶层严格只允许 `diagnostics` 和 `logging`；至少应声明其中一个，其他任何顶层字段
都会被拒绝。这不是通用 config merge：

- 显式 `diagnostics` 完整替换 checkpoint 保存的有序 diagnostic 列表；
- `logging` 只覆盖其中显式声明的字段；显式 `backends` 完整替换 backend 列表；
- 省略的 `logging.log_every`、`logging.backends` 或 `logging.torch_logs` 分别继承
  checkpoint 值；
- 模型、Process、TrainingBuilder、Objective、数据、optimizer、scheduler、EMA、
  trainer、sampling、artifacts 和 extension selection 均不可通过该文件改变；
- diagnostic 的 `params.modules` 不能引入 checkpoint 未选择的新 provider module。

空的顶层 mapping 和空的 `logging: {}` 都会被拒绝；若只需关闭诊断，应显式使用
`diagnostics: []`。

仓库示例固定使用 EMA、确定性 DDIM-50、`seed: 123` 和 32 个样本，并记录固定 timestep
的 `x0` 重建 MSE/PSNR、重建面板和样本网格。它显式启用 local 与 TensorBoard backends，
但省略 `log_every`，因此沿用 checkpoint 的记录间隔。`diagnostics: []` 可显式关闭全部
diagnostics；诊断组件和 logger 遵守只观测契约，不拥有 checkpoint-restored state。
provider cache、错误计数、打开的文件和 TensorBoard writer 都会为本次 invocation
重新创建。

observability config 只在进程启动时读取和校验，不会监视文件变化或在训练中热加载。
resume 仍创建新的兄弟 run，因此 local 日志和 TensorBoard event 文件写入新目录；它既不
续写旧 event 文件，也不重开旧 logger。若要在 TensorBoard 比较恢复前后曲线，应把
`--logdir` 指向包含两个时间戳目录的共同 output root。

最终生效的 `diagnostics`/`logging` 会写入新 run 的 `resolved_config.yaml`，并成为该 run
后续 checkpoint 的权威配置。`run_manifest.yaml` 和 checkpoint metadata 的
`config_overlays` 审计记录会固化 `kind: observability`、`source_path`、
`source_sha256`、生效的 `sections`，以及 `logging_fields` 中本次显式提供的字段；同时
`runtime_options.observability_config` 记录本次 CLI 输入。源 checkpoint 与旧 run
不会被改写。再次应用 observability config 时会继承历史记录并追加本次 entry；从新
checkpoint 再次 resume 且不传该选项时，使用的就是先前已固化的 effective config 和
审计链。

`--resume` 与 `--config` 互斥。当前 checkpoint v9 保存 resolved config、primary
inference model、可选 Process/Objective、可选 EMA model、optimizer、scheduler、EMA、具名
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
Strict resume 还要求合法的 `epoch`、`global_step` 和 v9 RNG snapshot，并在 selected state
与 inherited-best 全部验证后恢复 Python、NumPy、Torch CPU 及适用的 CUDA/MPS RNG。
早期 v8 checkpoint 若缺少 MPS RNG 字段，在 MPS resume 时会警告并继续，但不能保证该
随机流精确延续。普通 checkpoint load 不修改全局 RNG。sampling 不恢复 checkpoint RNG snapshot，而是按
`sampling.seed`（为 `null` 时使用 `experiment.seed`）重新初始化 Python、NumPy 与 Torch
全局 RNG。device override 仍受支持，但跨设备、CUDA topology 或 backend 版本不保证逐位
一致。

checkpoint 不保存 DataBuilder、Dataset、DataLoader iterator/worker、Sampler 或用户私有
generator 的 runtime state。内置图像 recipe 会由 experiment seed 与 epoch 重建索引顺序，
但 worker 侧随机 crop/flip 不保证与不中断运行逐位一致。自定义随机 loader 应同样由 seed
和 epoch 确定，并在需要时响应 duck-typed `set_epoch(epoch)`；核心只承诺 epoch-boundary
的全局 RNG 与训练资产恢复。
若 checkpoint config 保存的是相对 output path，它仍以本次进程启动 cwd 解释；从其他目录
恢复时应显式传 `--output-dir`。DataBuilder 私有 params 中的相对路径遵循同一 cwd 规则，
核心不会猜测并重写不透明字段。

当前 v9 payload 只允许 Tensor、primitive 与普通 container，并始终由
`torch.load(..., weights_only=True)` 读取；legacy v8 只按
[受限规则](compatibility-and-migration.md#v8-到-v9-的受限迁移)迁移。扩展代码/class
不会 freeze 在 checkpoint 中；恢复环境需要安装记录的 entry-point distribution。
实现变化造成的不兼容由 state/资产契约报错，Stochaflow 不保存或迁移第三方源码。

## K-fold

K-fold 是支持它的图像 recipe 的私有能力。一次配置只运行一个 fold，例如第三个 fold
（索引从 0 开始）：

```yaml
data:
  name: image
  params:
    source:
      name: image_folder
      params:
        root: ./data/images
        layout: flat
      materialization:
        cache_root: ./data
        policy: ensure
        verification: full
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
只用于监控，不参与 best checkpoint 或 early stopping。MPS 运行中，FID 会把 feature
accumulation 与距离计算放到 CPU，以避开 MPS 不支持的 double-precision linear algebra；
KID 仍使用配置的 runtime device。

`failure_policy: raise` 会让采样或 provider 异常终止训练；`warn` 按 provider/profile
隔离失败、继续执行其余组件，并把错误同时写入 `diagnostics/system/error_count`、日志
和 epoch manifest。未知 sampler/provider、重复名称、缺少 trajectory 接口等配置错误
始终在训练开始前失败。

## checkpoint 采样

checkpoint 内含模型和训练配置，所以可以只给 checkpoint：

```bash
stochaflow sample \
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
stochaflow sample \
  --checkpoint outputs/ddpm_mnist/<run>/checkpoints/best.pt \
  --config path/to/sampling.yaml
```

CLI 不提供 sampler-specific flags；Sampler 参数完全属于 Builder。这样自定义 Builder
可以组合 condition、多个 Sampler 或非固定 shape initial state，而无需扩充核心 CLI。

也可以只给外部配置，让 runner 在 `experiment.output_dir` 下寻找最新 `best.pt`：

```bash
stochaflow sample --config path/to/complete-config.yaml
```

同时给出两者时，有两种明确输入形态：

- 只含 `sampling` 与可选 `extensions` 的 lightweight overlay：checkpoint config 是 base，
  overlay 的整个 `sampling` 段替换它；
- 完整 Stochaflow config：外部 config 整体权威，checkpoint 只提供 state。

sampling 不恢复 optimizer/scheduler state，因此外部配置可以改变 `num_samples`、
`batch_size`、`shape`、SamplingBuilder、Sampler/solver 参数、trajectory、writers 以及
raw/EMA 选择。核心不比较两份完整 config 或根据字段名推断兼容性；最终配置构建的
primary model 必须能严格加载 checkpoint state。若最终配置仍选择 Process，该 Process
也必须严格加载对应 state；但完整外部 config 可以明确写 `process: null`，让兼容的
direct-transform Builder 不构建 Process，并丢弃 checkpoint 中未使用的 Process state。
lightweight overlay 与 checkpoint-only sampling 不具备这个“删除算法资产”的权限。

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
离散 Gaussian family 同时公开 DDPM/DDIM transition 与 DDIM schedule primitive，供项目
Sampler 组合 post-transition correction 或其他 family 内算法；这些 primitive 不进入
通用 `Sampler`/`GenerativeDynamics` 根接口。
trajectory 是 observer 对 initial、accepted step 和唯一 final observation 的抽样，不会
改变 solver 循环。保留的 state 在 observation 到达时复制，内置 Tensor 路径会立即转存到
CPU，避免后续原地更新污染历史或让显存随 trajectory 长度增长。`trajectory.pt` 按声明
顺序保存 step index、coordinate 和 state。

## 大规模 sampling 容量

当前 runtime 在 writer 开始前物化完整 `SamplingOutput`：所有 sample batch
和已保留 trajectory 都已整体存在。内置 Standard Builder 会将 writer-ready
Tensor 转存到 CPU；公共 contract 不强制自定义 Builder 的设备。对该内置路径，
调小 `batch_size` 可以减少单次 accelerator 工作集，但在 `num_samples` 不变时
不会减少 writer 前的全量 CPU output。当前 writer contract 也不是 streaming
contract。

规划大规模 sampling 时应分别估算：

1. 单个 device batch 的 model/solver 工作集；
2. 所有 final sample 在 Builder 返回前的整体驻留；
3. 每个保留 observation 带来的 trajectory 倍数；
4. writer 拼接、stack、编码和序列化的临时副本；
5. 最终文件与临时目录空间。

主运行若只需要 final state，应关闭 trajectory。可视化 trajectory 应使用独立的小样本
调用，并增大 `every_steps`。领域 Builder 可以先逐 batch 计算指标，再只把 writer
真正需要的场或通道放入 `SamplingOutput`；这会减少 payload，但仍不会把 lifecycle 变成
streaming。

Physics reconstruction 参考项目提供了一个具体例子：1272 个
`[3, 256, 256] float32` state 的 dense 30/40-step trajectory 会达到数十 GiB，因此正式
配置采用 final-only，preview 与主输出分离。不要把这些任务数值当作其他数据的通用上限。
完整公式、benchmark profile、安全预检和证据分级见
[Sampling artifact 容量](sampling-capacity.md)。其中 RSS、accelerator peak、编码开销
和耗时只属于指定参考主机，不是跨平台容量保证。

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

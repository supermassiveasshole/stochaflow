# 常用工作流

## 训练

通用入口是：

```bash
stochaflow train --config path/to/train.yaml
```

runner 加载配置、应用 CLI 覆盖、创建一个时间戳 run 目录，调用一次 DataBuilder，
然后构建一套训练组件。所有训练参数见[CLI 参数索引](reference.md#cli-参数索引)。
源码 checkout 中可以把命令写成 `uv run stochaflow ...`，并直接使用仓库内
`examples/` 示例；发布 wheel 不包含这些 repo-local 配置。

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
  --config examples/built-in/image-generation/configs/train/mnist.yaml \
  --epochs 1 \
  --limit-batches 2 \
  --limit-validation-batches 1
```

这是仓库当前唯一维护的 built-in 完整训练配置。DDPM 与 DDIM-50 位于
`configs/sample/`，是复用训练 checkpoint 的请求 profile，不是另外两份训练 recipe。
仓库不再维护 CIFAR-10、Flowers102 或 multi-source runnable YAML；这一收敛不改变
DataSource/DataBuilder 的通用扩展能力。

`--limit-*` 只截断本次运行，不修改 YAML。该 train 文件没有顶层 `sampling`，
当前默认也不会在训练结束后隐式启动 inference；DDPM/DDIM 结果通过后续显式
`stochaflow sample` 产生。`shape` 是否必需由 checkpoint recipe 决定。

这些 smoke 覆盖也不会重写 LR scheduler 的 `T_max`、`total_steps` 或其他构造参数。
它们是具体 PyTorch scheduler 的显式配置，而不是框架可推断的通用 run length。若要运行
一份具有不同调度周期的完整实验，应同时修改 YAML，使 scheduler 参数与训练计划一致。

### CLI 覆盖优先级

新训练有效值按下列顺序决定，后者优先：

1. dataclass 默认值；
2. YAML；
3. `--device`、`--output-dir`、`--epochs` 等 CLI 覆盖；
4. runner 为本次 run 生成的 `experiment.exp_id` 和时间戳输出目录。

`--progress` 与 `--no-progress` 互斥，并显式开启或关闭本次运行的进度条；都不指定时
继承 YAML 或 checkpoint 中的值。最终生效值写入新 run 及其 checkpoint，因此后续
resume 默认继承最近一次选择。`--deterministic` 启用 PyTorch 的严格
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

Stochaflow 为训练/恢复选择唯一训练 config authority；`sample` 则显式保留两条平行权威：
checkpoint 提供 state 与 fixed recipe，完整 sample config 提供本次 mutable invocation。
二者不会合并为一份配置，也不会与 CLI 做通用 merge。

| Workflow | 配置权威 | Checkpoint 角色 | 后续覆盖 |
| --- | --- | --- | --- |
| `train --config ...` | 外部完整 config | 无 | train CLI flags |
| `train --resume ...` | checkpoint config | 完整训练 state | 安全 train runtime flags；可选 observability config |
| `sample --checkpoint ... --config ...` | 必填、完整且独立的 `sample:` config | v12 推理 state + fixed `inference_recipe` | sample CLI runtime flags |
| `evaluate --config ...` | 必填、完整且独立的 evaluation config | config 内 subject 引用的 v12 inference state、training config 与 data identity | device/output 与 extension-version acceptance |

config 字段覆盖进入 resolved config；`limit-batches`、deterministic、启动 cwd、lineage、
插件 version acceptance 等 invocation 事实进入独立 manifest，而不是
扩张组件 schema。

目录输入和默认输出遵循下表；显式 checkpoint 文件始终按原路径使用：

| 调用 | 目录中选择的 checkpoint | 默认输出 |
| --- | --- | --- |
| `train --resume <run-or-root>` | 递归查找最近修改的 `checkpoints/latest.pt` | 在原 run 的 output root 下创建新的兄弟 run |
| `sample --checkpoint <run-or-root>` | 递归查找最近修改的 `checkpoints/best.pt` | `<checkpoint-run>/samples/<timestamp>/` |
| `evaluate --config <file>` | config 中的 checkpoint path；相对路径以 config 目录为基准 | `<checkpoint-run>/evaluations/<timestamp>/`，或显式的新 output directory |

`sample` 始终要求显式 `--checkpoint` 与 `--config`；checkpoint 目录只是一种便利输入，会递归选择最近修改的
`best.pt`。没有 validation 的 run 不创建 best；对这类 run 采样时应显式传
`checkpoints/latest.pt`，它表示 final checkpoint，而不是经过验证集选择的 best。需要冻结
精确 lineage 时也应传 checkpoint 文件。`train --output-dir` 是新建 timestamped run 的
父目录，而 `sample --output-dir` 是本次 artifact 的最终目录。
sampling 目录总会写
`resolved_sampling.yaml`；训练 run 总会写 `resolved_config.yaml` 和
`run_manifest.yaml`。

## 输出目录

新训练在 `experiment.output_dir` 下创建唯一时间戳目录：

```text
outputs/<experiment>/<YYYYMMDD_HHMMSS>/
  checkpoints/
    best.pt        # 仅在启用 validation-based best tracking 时存在
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
```

训练 run 不自动创建 sample artifact。独立 `sample` 调用默认写入 checkpoint run 下唯一的
`samples/<timestamp>/`，其中包含 writers 产物与 `resolved_sampling.yaml`。

`resolved_config.yaml` 只保存最终可重建组件配置；`run_manifest.yaml` 另外记录 config
来源、实际插件 provenance、version acceptance、启动 cwd、runtime-only CLI options、
checkpoint lineage 和 `selected_components`。同一 training-owned 摘要也写入 checkpoint
metadata；sampling 的 `resolved_sampling.yaml` 使用另一投影，只组合 checkpoint-owned
model/Process/recipe 与 sample config 选择的 Sampler/writers。两侧都不递归解释
Builder/Process 的私有 `params`，也不伪造另一 authority 的组件。完整 checkpoint config
与完整 sample config 分别保持重建和 invocation 权威，摘要只用于审计。

训练 manifest 创建时先写入 `status: running`。只有训练、可选 phase test、终端最终报告
和 logger 关闭都成功后，才原子更新为 `status: completed` 并写入 `outcome`。该 outcome
记录最终 epoch 与完整 canonical `final_metrics`、latest/best/selected checkpoint 及选择
类型、`stopped_early`、完整 `phase_test_metrics`、manifest 路径，以及可选 local
metrics/log 路径。没有 test split 时 `phase_test_metrics` 是空 mapping；没有 local logger
时两个日志路径为 `null`。失败或收尾未完成的 manifest 不发布 `outcome`，因此消费者只应
把 `status: completed` 视为可消费的成功结果。

启用 TensorBoard、W&B、diagnostic 或 trajectory 后会增加对应子目录/artifact。
`artifacts.checkpoint_every` 控制编号 checkpoint 的频率；`latest.pt` 在每个完成 epoch 后
更新。`best.pt` 只由 `valid/loss` 或 `valid/metrics/<id>[/<subkey>]` 更新；没有 validation
时默认关闭 best tracking。此时显式采样应传 final `latest.pt`。

## 恢复训练

严格恢复必须显式指定 checkpoint 文件或 run directory：

```bash
stochaflow train \
  --resume outputs/mnist/<run>/checkpoints/latest.pt
```

### Strict Resume Observability Overlay

长训练恢复时，可以保留全部 checkpoint 训练语义，只为新 invocation 配置监控：

```bash
stochaflow train \
  --resume outputs/mnist/<run>/checkpoints/latest.pt \
  --observability-config \
    examples/built-in/image-generation/configs/overlays/mnist-observability.yaml
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
重新创建。diagnostic 只能写日志和 artifact；其 scalar 不进入 epoch history/checkpoint，
也不能成为 best 或 early-stopping monitor。

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

`--resume` 与 `--config` 互斥。当前 checkpoint v12 保存 resolved config、primary
inference model、可选 Process/Objective、可选 EMA model、optimizer、scheduler、EMA、具名
training assets、训练进度，以及始终存在的 fixed `inference_recipe` 字段。它只保存
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
启用 validation-based best tracking 时，严格恢复还延续 best metric 与 early-stopping
wait；因此这类 run 的 `latest.pt` 或 epoch checkpoint 需要同一 `checkpoints/` 目录中的
`best.pt`。推荐直接传 run directory。候选 best 的 epoch、metric、monitor、mode、resolved
config 和 extension provenance 必须与所选 checkpoint 一致；通过校验且能载入当前资产
拓扑的 inherited best 会在训练开始前原子物化到新 run 的 `checkpoints/best.pt`。

没有 validation 的 run 不保存 best selection state，也不依赖 `best.pt`；它的
`latest.pt` 是 final checkpoint。显式请求 best tracking 或 early stopping 却没有
validation DataLoader 会在训练循环前失败。
Strict resume 还要求合法的 `epoch`、`global_step` 和 v12 RNG snapshot，并在 selected state
与 inherited-best 全部验证后恢复 Python、NumPy、Torch CPU 及适用的 CUDA/MPS RNG。
普通 checkpoint load 不修改全局 RNG。checkpoint-backed inference 不恢复 checkpoint
RNG snapshot，而是按完整 sample config 中显式的 `sample.seed` 重新初始化 Python、
NumPy 与 Torch 全局 RNG，不回退到训练配置的 `experiment.seed`。device override 仍受
支持，但跨设备、CUDA topology 或 backend 版本不保证逐位一致。

checkpoint 不保存 DataBuilder、Dataset、DataLoader iterator/worker、Sampler 或用户私有
generator 的 runtime state。内置图像 recipe 会由 experiment seed 与 epoch 重建索引顺序，
但 worker 侧随机 crop/flip 不保证与不中断运行逐位一致。自定义随机 loader 应同样由 seed
和 epoch 确定，并在需要时响应 duck-typed `set_epoch(epoch)`；核心只承诺 epoch-boundary
的全局 RNG 与训练资产恢复。
若 checkpoint config 保存的是相对 output path，它仍以本次进程启动 cwd 解释；从其他目录
恢复时应显式传 `--output-dir`。DataBuilder 私有 params 中的相对路径遵循同一 cwd 规则，
核心不会猜测并重写不透明字段。

当前 v12 payload 只允许 Tensor、primitive 与普通 container，并始终由
`torch.load(..., weights_only=True)` 读取；旧 checkpoint 格式不迁移。扩展代码/class
不会 freeze 在 checkpoint 中；恢复与 inference 环境需要安装记录的 entry-point
distribution。实现变化造成的不兼容由 state/资产/recipe 契约报错，Stochaflow 不保存或
迁移第三方源码。

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
        cache_root: ./.stochaflow-cache
        policy: ensure
        verification: full
    partition:
      mode: kfold
      num_folds: 5
      fold_index: 2
    image:
      size: [64, 64]
```

`cache_root` 中的数据 artifact 使用统一 schema-v2 lifecycle。managed source 在 cache
中拥有实际内容；referenced source 只缓存索引且要求 external root 与 cache 分离。
`policy: require` 完全只读，适合已预热的生产 cache；`ensure` 才允许构建或修复。
strict resume 使用 checkpoint 保存的 exact identity、绕过当前 locator 并强制 full
验证。旧 artifact cache/binding 不会迁移，升级后应重新 materialize 并创建新 run。

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
        shape: [1, 32, 32]
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

这里的 `diagnostics[].params.sampling.shape` 由训练期 diagnostic 自己拥有，用于构造
它的固定监控样本；它不会读取或借用独立 sample workflow 的 `sample.shape`。同理，
checkpoint-backed sample profile 不会改变训练 diagnostics。

Local logger 记录 artifact 路径，TensorBoard 和 W&B 同时显示 PNG。启用 KID/FID
前需要 `uv sync --extra quality`，并且本次训练必须有 validation DataLoader。参考指标
与其他 diagnostic scalar 一样只写入 logger/manifest，不进入 epoch history、checkpoint
metrics、best checkpoint 或 early stopping。MPS 运行中，FID 会把 feature
accumulation 与距离计算放到 CPU，以避开 MPS 不支持的 double-precision linear algebra；
KID 仍使用配置的 runtime device。

`failure_policy: raise` 会让采样或 provider 异常终止训练；`warn` 按 provider/profile
隔离失败、继续执行其余组件，并把错误同时写入 `diagnostics/system/error_count`、日志
和 epoch manifest。未知 sampler/provider、重复名称、缺少 trajectory 接口等配置错误
始终在训练开始前失败。

每个 epoch manifest 的 `profiles[].metrics` 保存对应 sampler profile 的独立结果；
顶层 `combined_metrics` 保存所有 profile 指标的扁平合并结果，便于一次读取完整的
epoch diagnostic 汇总。

## 训练内 epoch-end validation Evaluation

昂贵的生成质量指标应由完整 Evaluation 产生，而不是塞进普通 batch metric 或
Diagnostic。`trainer.validation_evaluation` 声明一个 absolute-epoch cadence、raw/EMA
variant、task-owned EvaluationBuilder、Metrics、exact output keys 和 completeness
protocol。例如：

```yaml
trainer:
  early_stopping:
    enabled: false
    monitor: valid/metrics/distribution/aggregate.fid
    mode: min
    patience: 20
    min_delta: 0.0
  validation_evaluation:
    enabled: true
    start_epoch: 20
    every_epochs: 20
    include_final: true
    weights: ema
    evaluation:
      name: my-project.class-conditional-generation
      params:
        sampling: {profile: validation-ddpm-v1}
    metrics:
      - id: distribution
        name: my-project.class-aware-distribution
        channel: my-project.image-pairs
        params: {}
    metric_keys:
      - valid/metrics/distribution/aggregate.fid
      - valid/metrics/distribution/aggregate.kid_mean
    protocol:
      id: my-project-validation-v1
      expected_examples: 900
      strict_complete: true
```

EvaluationBuilder 拥有完整验证语义：它选择 validation data，调用当前 checkpoint-bound
sampling recipe 产生 fake，绑定 real/fake 与 sample IDs，并向 Metric 发出 task-owned
updates。FID/KID provider 只是 Metric 内部的 stateful computation；它不拥有采样、split、
checkpoint 或 completeness。

到期 epoch 的结果必须精确包含 `metric_keys` 声明的 canonical
`valid/metrics/*` surface。Trainer 把它们合并进该 epoch 的 validation observations，并
复用现有 monitor、`best.pt` 和 early-stopping 逻辑。非到期 epoch 不复用上一次 FID/KID、
不更新 `best.pt`、也不推进 patience；到期 Evaluation 失败、缺 key、非 finite、重复 ID 或
数量不完整都会 fail closed。`include_final` 可保证目标训练的最终 epoch 额外评估一次。

profile digest、metric keys、cadence、last evaluated epoch 和最后一组 metrics 都进入
strict-resume state。训练内运行不配置 prediction sink，也不发布 standalone immutable
result bundle，所以它是选模用 validation evidence，不是 formal benchmark。要从已经存在
的多个 checkpoints 选一个，只需对每个 subject 运行同一 standalone validation
Evaluation，再按同一个 primary metric 比较；无需另一套 selection runtime。

## Checkpoint-backed inference

`sample` 是统一的 checkpoint-backed inference 命令：MNIST、AFHQ 等生成任务产生图像，
Physics 任务执行重建，direct-transform 任务也可以产生 prediction。数值 `Sampler`
只是某些 recipe 的内部协作，不是运行该命令的前提。

v12 checkpoint 除模型、可选 EMA/Process state 和训练配置外，还保存
`inference_recipe`。它固定内部 `SamplingBuilder` identity 与不可覆盖的 contract；
`null` 表示该 checkpoint 不支持 `sample`。CLI 始终要求显式 checkpoint 和完整 sample
config，checkpoint 不提供 sampler、options、shape、数量、batch、seed 或 writers 的
mutable defaults：

```bash
stochaflow sample \
  --checkpoint outputs/mnist/<run>/checkpoints/best.pt \
  --config examples/built-in/image-generation/configs/sample/mnist-ddpm.yaml
```

完整 sample config 例如：

```yaml
sample:
  shape: [1, 32, 32]
  num_samples: 64
  batch_size: 16
  seed: 123
  sampler:
    name: ddim
    params: {num_inference_steps: 100, eta: 0.0}
  options:
    weights: auto
    clip_denoised: true
    trajectory: {enabled: true, every_steps: 5}
  writers:
    - {name: tensor, params: {}}
    - {name: image, params: {grid_nrow: 8, gif_fps: 8}}
```

```bash
stochaflow sample \
  --checkpoint outputs/mnist/<run>/checkpoints/best.pt \
  --config path/to/sample.yaml
```

仓库提供两份可直接使用的 MNIST sample profile：

```bash
stochaflow sample \
  --checkpoint outputs/mnist/<run>/checkpoints/best.pt \
  --config examples/built-in/image-generation/configs/sample/mnist-ddpm.yaml

stochaflow sample \
  --checkpoint outputs/mnist/<run>/checkpoints/best.pt \
  --config examples/built-in/image-generation/configs/sample/mnist-ddim-50.yaml
```

config 顶层只允许 `sample` 与可选 `extensions`。`sample` 内的 sampler、options、shape、
num_samples、batch_size、seed 和 writers 是完整调用权威，不能省略后指望继承 checkpoint
训练配置。`options` 不能含 `sampler`，也不能覆盖 recipe fixed contract 中的字段；
`extensions.plugins` 只能追加插件，不能删除 checkpoint-required plugins，也不能写
`null` 来选择整个环境。训练配置不接受顶层 `sampling`，训练结束也不会自动采样。

CLI 不提供 sampler-specific flags；solver 参数属于 `sample.sampler.params`。这样
checkpoint recipe 可以组合 condition、guidance、多个内部组件或非固定 shape initial
state，而无需扩充核心 CLI。

### 固定 contract 与完整 sample config

训练侧 `TrainingBuilder` 根据实际训练语义返回 `TrainingPlan.inference_recipe`。例如
Gaussian denoising 把 `prediction_type` 固化进 contract，避免 sample config 把 epsilon
模型当作 v-prediction 使用。运行时按以下顺序构造内部 Builder 参数：

```text
complete sample.options
        + complete sample.sampler
        + immutable inference_recipe.contract
```

sample config 与 fixed contract 发生 key 冲突时直接失败，而不是覆盖训练语义。内部 recipe
name、contract 和完整 sample settings 都写入 `resolved_sampling.yaml`。

### 权重、shape 与输出

`standard_denoising` 使用 `sample.shape`，它不含 batch 维且与 DataBuilder 独立；
自定义 recipe 可以在 shape 为 null 时运行。`weights: auto` 通常位于
`sample.options`：checkpoint 含 EMA model state 时选择 EMA，否则选择 raw；训练配置不再
提供 `ema.use_for_sampling`。
正式评估应显式请求 `raw` 或 `ema`。

每次 inference 都写 `resolved_sampling.yaml`，其中记录 checkpoint path、稳定 bytes 的
SHA-256、format version、epoch/global step lineage、v12 recipe、完整 sample config 及其
source、实际插件 provenance/version acceptance、启动 cwd、runtime options，以及 recipe
metadata/artifacts。manifest 中的 writer artifact 使用 bundle-relative portable path。
`sample.writers` 决定其他输出：
`tensor` 写 PT，`image` 写 PNG/GIF；开启 trajectory 后，两者会写各自支持的
trajectory artifact。

默认输出是 checkpoint run 下唯一的 `samples/<timestamp>/`。显式 `--output-dir` 指向
最终目录，而不是自动创建 timestamp 子目录。该最终目录必须尚不存在：runtime 先在同级
私有 staging 内完成全部 writers 和 manifest，再用 no-replace rename 原子发布；任一步失败
都会清理 staging 且不留下最终目录。已有目录会 fail closed，内容不会被覆盖或合并。

所有注册 Sampler 通过相同的完整 `sample(dynamics, initial_state, ...)` 生命周期执行，但
不共享万能数学接口。内置 DDPM/DDIM 要求 Gaussian Dynamics；其他算法 family 可定义
自己的 Dynamics capability，并由所属 recipe 与 Sampler 在调用边界验证。
离散 Gaussian family 同时公开 DDPM/DDIM transition 与 DDIM schedule primitive，供项目
Sampler 组合 post-transition correction 或其他 family 内算法；这些 primitive 不进入
通用 `Sampler`/`GenerativeDynamics` 根接口。

trajectory 是 observer 对 initial、accepted step 和唯一 final observation 的抽样，不会
改变 solver 循环。保留的 state 在 observation 到达时复制，内置 Tensor 路径会立即转存到
CPU，避免后续原地更新污染历史或让显存随 trajectory 长度增长。`trajectory.pt` 按声明
顺序保存 step index、coordinate 和 state。

## 独立 checkpoint Evaluation

`stochaflow evaluate` 对一个配置显式引用的冻结 authority 运行独立 protocol。当前支持：

- `checkpoint`：安全读取 v12 checkpoint，显式解析 raw/EMA，构造 checkpoint DataBuilder
  的 validation/test split，并执行 live inference；
- `prediction_artifact`：认证一个已经 complete 的 versioned prediction manifest 与 shards，
  按 exact sample plan 提供 records，并在不构造 checkpoint model 或原 DataBuilder 的情况
  下重新计算 metrics。

两条路径都不恢复 optimizer、scheduler、GradScaler 或 training RNG，不继续训练，也不调用
`stochaflow sample` operation。需要完整生成的 task 可以消费 runtime 注入的窄
`EvaluationSamplingCapability`，通过同一个 SamplingBuilder execution seam 取得
writer-free in-memory output。具体 task batch/record、模型签名和 metric channel 仍由已安装
extension 注册的 `EvaluationBuilder` 与 Metric 解释。

### Live checkpoint evaluation

一份完整 evaluation config 例如：

```yaml
version: 1
name: candidate-a-validation
purpose: selection_candidate

extensions:
  plugins: [my-project]

subject:
  kind: checkpoint
  path: ../outputs/run/checkpoints/best.pt
  weights: ema

data:
  source: checkpoint
  split: validation

evaluation:
  name: my-project.supervised-evaluation
  params: {profile: paired-v1}

metrics:
  - id: prediction_mae
    name: my-project.mae
    channel: predictions.targets
    params: {}

protocol:
  id: paired-validation-v1
  expected_examples: 100
  strict_complete: true
```

evaluation schema 与 training/sample schema 平行而不合并。顶层必填字段是 `version`、
`name`、`purpose`、`subject`、`data`、`evaluation`、`metrics` 和 `protocol`；
`extensions` 可省略。unknown 或 duplicate field 失败。当前约束是：

- `subject.kind` 与 `data.source` 必须选择同一个 authority：都为 `checkpoint` 或都为
  `prediction_artifact`；
- checkpoint subject 的 `weights` 必须显式写 `raw` 或 `ema`，不能写 sampling 的
  `auto`；prediction-artifact subject 不接受 `weights`；
- subject 相对路径以 evaluation YAML 所在目录为基准，而不是进程 cwd；
- split 只能为 `validation` 或 `test`；offline config 的 split 必须与 producer manifest
  冻结的 split 相同；
- `selection_candidate` 必须使用 validation，`final_test` 必须使用 test，`benchmark`
  可以显式使用其中任一 split；
- `protocol.expected_examples` 必须是正整数；`strict_complete` 默认 `true`，duplicate
  sample ID、超额样本或最终数量不足都会 fail closed；
- `evaluation.name` 选择注册的 `EvaluationBuilder`；`metrics[].channel` 必须由其
  Evaluator 声明，core 不定义通用 prediction/target/image batch schema。

运行命令：

```bash
stochaflow evaluate \
  --config path/to/evaluation.yaml \
  --device cuda \
  --output-dir outputs/evaluations/candidate-a
```

只有 `--config` 必填。`--device`、`--output-dir` 与
`--force-extension-version-mismatch` 是 runtime options；CLI 没有独立 `--checkpoint`
或 arbitrary config patch。省略 output 时，runtime 根据 subject path 在相邻
`evaluations/<timestamp>/` 选择新目录。显式 output directory 必须不存在，当前没有
overwrite/resume；目标冲突在评估前 fail closed。

Python 的 path-first API 返回 immutable `EvaluationRunOutcome`：

```python
from stochaflow.evaluation import run_evaluation

outcome = run_evaluation(
    "path/to/evaluation.yaml",
    output_dir="outputs/evaluations/candidate-a",
    device_name="cuda",
)
print(outcome.metrics, outcome.result_path)
```

需要自行控制插件 activation 的宿主可以先调用 `resolve_evaluation_inputs()`，再把明确
激活得到的 `ResolvedExtensions` 交给 `run_resolved_evaluation()`。当前
`EvaluationRunRequest` 是数据 contract，不是 `run_evaluation()` 的参数形式。

没有 artifact sink 的成功目录包含：

```text
candidate-a/
├── resolved_evaluation.yaml
├── result.json
└── evaluation_manifest.yaml
```

`result.json` 保存 protocol digest、checkpoint SHA-256/format/epoch/global step、
requested/resolved weights、lineage、DataBuilder/split/artifact identity、
`eval/metrics/*`、`eval/measurements/*`、sample completeness 与 extension/builder/metric
provenance。manifest 最后发布并记录 result SHA-256；任何加载、评估、完整性或写入失败
都会清理未完成目录。`strict_complete: false` 时数量不足可产生显式 `incomplete` result，
而不是伪装为 complete。

### 从 live run 发布 predictions

prediction artifact 是 opt-in task contract，不是每次 checkpoint evaluation 的隐式副作用。
Builder 若从 `EvaluationBuilderContext.artifact_root` 创建
`JsonlPredictionArtifactSink`（或兼容的 `EvaluationArtifactSink`），并把它放入
`EvaluationPlan.artifact_sink`，Evaluator 必须在每个 `EvaluationStepOutput.records` 中
返回与 `sample_ids` 同序的 typed `PredictionRecord`。runtime 逐 batch 流式写 unpublished
shards；只有 sink finalize、protocol completeness policy 和 ordered sample plan 全部通过时，
成功目录才会额外包含：

```text
candidate-a/
├── predictions/
│   ├── prediction_manifest.json
│   └── predictions.jsonl
├── resolved_evaluation.yaml
├── result.json
└── evaluation_manifest.yaml
```

`predictions.jsonl` 是内置 sink 的默认 shard 名；custom sink 可以发布一个或多个 manifest
声明的 portable relative shard paths。`result.json` 和 completion manifest 的 `artifacts`
会以相对路径引用 `predictions/prediction_manifest.json`，并记录 manifest SHA-256 与整个
artifact digest。

prediction manifest schema v1 冻结：

- producer evaluation identity/authority/protocol、normalized training config 与 extension
  provenance；
- 原 source subject identity/content digest、resolved weights、inference profile/digest；
- producer data identity、governed split、preprocess/postprocess；
- ordered `sample_id`/`input_id`/`replicate_index` plan 与 digest；
- canonical JSONL shard format、path、size、record count 与 SHA-256；
- expected/observed/missing/unexpected/duplicate/failed/skipped completeness；
- deterministic gallery sample IDs。

gallery 是可审计的 ID selection，不是自动生成的图片目录。默认选择由 protocol ID 与
sample ID 的稳定 hash 决定，不受 input/shard 枚举顺序影响；task 也可以在发布 API 中显式
声明 IDs。manifest 发布后，loader 会重算并验证这项选择。

safe loader 只接受 strict canonical JSON/JSONL 和 normalized portable relative paths，不
反序列化 pickle。manifest、artifact 或 shard digest/size/count 不匹配，missing、duplicate、
unexpected record，`sample_id` 与 input/replicate identity 不匹配，incomplete status，或
sink/sample-plan 与实际 evaluation IDs 不一致，都会失败且不发布 result。启用 prediction
sink 时 artifact 自身只允许对其声明的 exact sample plan 为 complete；若
`strict_complete: false` 让 evaluation 相对更大的 protocol count 产生 `incomplete` result，
这与 artifact 对自身 sample plan 的 completeness 是两个必须分别读取的状态。

### Offline scoring

offline config 继续使用同一个 `stochaflow evaluate` 命令，只更换 subject/data authority；
`subject.path` 可以指向 artifact 根目录或 `prediction_manifest.json`：

```yaml
version: 1
name: candidate-a-offline-rescore
purpose: selection_candidate

extensions:
  plugins: [my-project]

subject:
  kind: prediction_artifact
  path: ../candidate-a/predictions/prediction_manifest.json

data:
  source: prediction_artifact
  split: validation

evaluation:
  name: my-project.supervised-evaluation
  params: {profile: paired-v1}

metrics:
  - id: prediction_mae
    name: my-project.mae
    channel: predictions.targets
    params: {}

protocol:
  id: paired-validation-v1
  expected_examples: 100
  strict_complete: true
```

```bash
stochaflow evaluate \
  --config path/to/offline-evaluation.yaml \
  --device cpu \
  --output-dir outputs/evaluations/candidate-a-offline
```

offline loader 先认证 producer manifest 和所有 shards，再按 sample ID 精确 join 并恢复
manifest 的 ordered sample plan；目录枚举、文件名排序或 shard 内 record 顺序都不是 join
authority。Builder 收到 `ResolvedPredictionArtifactSubject`、`inference=None` 和已排序的
`PredictionRecord`；它负责解释 task-private payload 并产生 metric updates，不得重新运行
模型或扫描 producer 目录。runtime 最后还要求 Evaluator 返回的完整 ordered IDs 与
artifact sample plan 完全相同，因此“数量相同但 ID 错误”也会失败。

新的 offline `result.json` 以 prediction artifact 为 subject，记录 artifact/manifest 与
sample-plan digest、producer identity、原 source subject 与 resolved weights、data/split、
inference/pre/postprocess/gallery 和 extension lineage。producer artifact 按字节保持不变；
offline run 发布自己的 immutable result bundle，不覆盖 live result 或 producer manifest。
常规 offline Builder 不声明新 sink，因此 `artifacts` 为空。`gate_result_path` 仍为 `None`。

### AFHQ-v2 formal full-test profile

maintained AFHQ-v2 source-checkout showcase 已提供 pixel-image generation Evaluation profile：
`experiments/evaluation/formal-ddim50-cfg2-official-test.yaml`。在该文件中只把
`subject.path` 替换为与其 v-prediction/fixed-variance recipe contract 匹配的唯一冻结 v12
checkpoint，然后运行：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow evaluate \
  --config examples/showcases/afhq-v2/experiments/evaluation/formal-ddim50-cfg2-official-test.yaml \
  --device cuda \
  --output-dir outputs/afhq-v2/evaluations/adm-ddim50-cfg2-official-test
```

profile 使用完整 authenticated official test split：cat/dog/wild 分别为
493/491/483，共 1,467 个 reference 与相同 allocation 的 generated samples。它固定 EMA、
sampling seed `20260726`、deterministic DDIM-50、CFG 2.0、KID 100 subsets /
subset size 300 / metric seed `20260726` 与
FID feature 2048，并通过 `REGISTRIES.metrics` 的 `kid`/`fid` providers 同时报告 aggregate
和 per-class 结果。

checkpoint runtime 先解析并固定 subject 的 raw/EMA variant，再向 AFHQ Builder 注入绑定
该 primary model 的 sampling capability；Builder 不能再次选择权重。capability 从 checkpoint
恢复 Process/assets，并通过 shared SamplingBuilder execution seam 生成，不运行普通 writers。
AFHQ sink 把同序 real/fake/class records 发布到 `predictions/`，所以复制这份 profile、把
`subject.kind`/`data.source` 改为 `prediction_artifact` 并指向该 manifest 后，可用同一个
`stochaflow evaluate` 完成 offline replay，而不加载 checkpoint、model 或原 DataBuilder。
subject 必须匹配 profile 固定的 prediction/variance recipe。learned-range-v checkpoint
需要相应的 `2C` model contract；DDPM 会消费 variance half，而 DDIM 明确只消费 prediction
half。正式 test 始终只在 validation 选出唯一 checkpoint 后运行一次，不能反向改变
训练 monitor 或 best checkpoint。

旧 `stochaflow-afhq-v2-evaluate` 与旧
`experiments/evaluation/ddim50-cfg2-kid-fid.yaml` 只作为历史结果对照；它们不属于当前
maintained evidence surface，也不提供 compatibility guarantee。

当前 runtime 提供通用 prediction persistence/replay substrate；core FID/KID providers 与上述 AFHQ
profile 已闭合当前普通像素图像生成的 formal Evaluation vertical slice。SR、consistency、
latent、distillation 等任务不属于本次缺口清单；若未来实现相应任务，其 monitoring 与
Evaluation protocol 必须在同一任务变更中一起交付，而不是提前扩张当前 runtime。reference
cache、performance curve、通用 comparison/gate 是可选后续增强。

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

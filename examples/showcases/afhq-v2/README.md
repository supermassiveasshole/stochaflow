# AFHQ-v2 class-conditional generation showcase

这个可安装 example 展示一条完整、可恢复的 AFHQ-v2 生成数据流：

```text
官方 AFHQ-v2 archive
  -> AFHQV2ImageDataSource：读取、处理、完整性与安全审计
  -> 带类别语义和 official train/test 的确定性 128×128 managed artifact
  -> 内置 class_labeled_image Builder：分层划分、Dataset、Sampler、DataLoader
  -> 带 class_label 的 train/validation/test loaders
  -> ADM-UNet 或 DiT-S/8 类条件 Gaussian 训练
  -> validation、best checkpoint 与 official test
  -> EMA + classifier-free guidance DDIM sampling
  -> tensor、PNG、trajectory 与可追溯 manifest
```

这个 extension 只注册 `AFHQV2ImageDataSource`（`afhq-v2.official`）。它负责读取和
处理官方来源，并发布带 identity、类别映射和标签 inventory 的标准
`ClassLabeledImageFolderArtifactPayload`。配置直接选择 core
`class_labeled_image` Builder，由框架完成 train/validation 分层划分以及 Dataset、
Sampler、collate 和 DataLoader 组装。ADM-UNet、DiT、Gaussian Process、训练
Strategy、CFG SamplingBuilder、DDIM、Trainer、EMA、checkpoint 和 artifact writers
也都使用 Stochaflow 的内置注册路径；runner 中没有 AFHQ 名称分支。

## 数据与许可证

权威入口是 [ClovaAI StarGAN v2](https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq)
维护的完整 AFHQ-v2 archive。数据采用
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)；请先确认用途符合
非商业许可证要求。

packaged source lock 固定：

- 15,803 张 512×512 RGB PNG；
- `cat: 0`、`dog: 1`、`wild: 2`；
- official train/test 为 14,336/1,467；
- archive 大小为 6,955,288,636 bytes；
- 项目对完整官方 archive 审计得到的 SHA-256 为
  `6f2540f22c6d8ebb8879a2bc0227666dd4fc765cc355cb073b63a835d679e4e3`。

上游没有发布 checksum；这里的摘要是本项目对固定官方下载内容的审计值。source 不会在
官方入口失败时静默切换到第三方镜像。

DataSource 对外提供 official train/test 及经过 source lock 认证的类别标签，数量为
14,336/1,467，不决定某次训练的 validation。内置 Builder 按
`partition.validation_per_class: 300` 从 official train 进行确定性、逐类分层划分；
loader 最终使用的 train/validation/test 数量为 13,436/900/1,467，其中 train 的
cat/dog/wild 数量为 4,765/4,378/4,293。

## 安装

这个 showcase 是独立的 uv project，并通过本地 editable source 使用当前 checkout 的
Stochaflow。在仓库根目录执行：

```bash
uv sync --project examples/showcases/afhq-v2 --locked
```

这会创建/同步 showcase 自己的环境、安装三个 example 命令，并注册 extension entry
point `stochaflow-afhq-v2`。entry point 只注册
`AFHQV2ImageDataSource`（`afhq-v2.official`）；训练配置直接调用内置
`class_labeled_image` Builder。example 不引入平行的 builder、source envelope、
image recipe 或 loader config 类型；模型、训练、采样和评估所需的通用配置与生命周期
也都来自 Stochaflow。

- `stochaflow-afhq-v2-prepare`；
- `stochaflow-afhq-v2-capacity`；
- `stochaflow-afhq-v2-evaluate`。

正式 KID/FID 评估需要 showcase 声明的 optional quality extra：

```bash
uv sync --project examples/showcases/afhq-v2 --locked --extra quality
```

## 1. 准备并验证数据

第一次运行使用 `ensure`。它可以下载官方 archive，也可以使用已经下载的本地文件：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow-afhq-v2-prepare \
  --cache-root ./data \
  --resolution 128 \
  --downloader auto
```

```bash
uv run --project examples/showcases/afhq-v2 stochaflow-afhq-v2-prepare \
  --cache-root ./data \
  --archive /path/to/afhq_v2.zip \
  --resolution 128
```

本地 archive 仍必须匹配 source lock 的字节数、SHA-256、ZIP inventory 和完整数据
contract。下载遵循标准 `HTTPS_PROXY`/`HTTP_PROXY` 环境变量；不要把含凭据的代理地址
写入 YAML。

准备成功后，用只读模式进行一次完整验证：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow-afhq-v2-prepare \
  --cache-root ./data \
  --resolution 128 \
  --policy require \
  --verification full
```

`require` 不访问网络、不重建也不修复。缺失或被修改的文件会直接失败。production 和
smoke 配置都固定为 `require/full`，因此长训练不会在启动时悄悄下载或改变数据。

prepare 命令通过注册的 `AFHQV2ImageDataSource` 执行，与内置 Builder 使用相同的
source 参数解析、materialization policy 和 identity 校验。命令只准备并汇报 official
train/test artifact；validation 参数属于 Builder，不是 prepare CLI 的输入。

默认缓存根为：

```text
data/
├── raw/afhq-v2/<source-sha256>/afhq_v2.zip
└── prepared/afhq-v2/128/<preparation-key>/
    ├── train/{cat,dog,wild}/
    ├── test/{cat,dog,wild}/
    ├── files.sha256
    └── dataset_manifest.yaml
```

消费者必须使用 payload inventory，不应绕过 DataSource 直接扫描缓存目录。
DataSource 对外只暴露 official train/test。
准备阶段执行一次 Lanczos resize，不 crop，并使用固定 PNG 编码。训练阶段要求
authenticated size 精确为 128×128，不再在线 resize。DataBuilder 在内存中从 official
train 生成分层 validation，并输出 `(images, {"class_label": labels})`；shuffle 和
horizontal flip 都由 `(seed, epoch, sample identity)` 确定，因此 `num_workers: 2` 与
persistent workers 下的 epoch-boundary resume 仍可重建同一顺序和增强。

## 2. 真实训练容量报告

准备好 AFHQ-v2 artifact 后，在目标 CUDA 主机运行真实训练 profiler：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow-afhq-v2-capacity \
  --config examples/showcases/afhq-v2/experiments/production/train-adm-128.yaml \
  --micro-batches 4 6 8 \
  --precisions fp32 bf16-mixed \
  --warmup-updates 5 \
  --measured-updates 25 \
  --output outputs/benchmarks/afhq-v2/adm-128-capacity.json
```

每个 trial 都通过已注册 DataBuilder、TrainingBuilder、Trainer 和 precision runtime 执行。
DataBuilder 通过 `ImageSourceFactory` 解析注册的 DataSource、验证同一 official
train/test artifact，再执行运行时分层划分并组装 loader；capacity 工具不会使用独立
source 或训练循环。micro batch 4/6/8 分别使用 accumulation 8/5/4，对应 effective batch
32/30/32。默认每个 trial warmup 5 次，并测量至少 25 次成功 optimizer updates；
报告 images/s、updates/s、allocated/reserved peak VRAM、data-wait/compute、细分
forward/backward/optimizer 时间、非有限 loss/gradient，以及同一 batch 下 BF16 相对
FP32 的吞吐和显存差值。报告还记录 Python、PyTorch、CUDA、cuDNN、GPU 型号和
capability，便于比较结果。CUDA phase 使用异步 Events，整段测量仅在开始和结束同步，
因此 phase 观测不会在每次 forward/backward/optimizer 边界串行化训练。

报告固定同一 DataBuilder 返回的 source artifact bindings；运行时 partition policy 则由
每个 trial 的完整 resolved config、canonical SHA-256 和 seed 冻结。output directory、
core/extension 版本及 Python source tree digest 也会进入 code identity。device 和
precision support 在 meta model、DataBuilder 及 trial output 创建之前完成 preflight；
全部 precision 不受支持时不会读取数据或构建模型。

命令默认要求 CUDA，避免 `device: auto` 在无 CUDA 时静默运行巨型 CPU trial。仅做有界
测试或调试时显式传入 `--device cpu`；CPU 报告的 VRAM 字段为 `null`，不应用于
production batch 选择。

## 3. 真实数据 smoke

smoke 配置使用已经准备好的真实 AFHQ-v2 artifact，而不是 synthetic data。它缩小
ADM-UNet、batch、diffusion steps 和采样步数，同时仍经过 class-aware loader、训练、
validation、official test、checkpoint、diagnostic 和最终 CFG sample：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/smoke/train-adm-128.yaml \
  --limit-batches 4 \
  --limit-validation-batches 2 \
  --limit-test-batches 2
```

smoke 使用 CPU/FP32、micro batch 2、累积 2 次和 4 个训练 micro-batches，产生 2 次
optimizer update。它证明 wiring 与 artifact contract 可运行，不代表模型质量或
production 性能。

## 4. Production 训练

主展示模型是 ADM-UNet：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/production/train-adm-128.yaml
```

DiT-S/8 使用相同 data、Process、Objective、effective batch、scheduler、diagnostic 和
sampling protocol：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/production/train-dit-128.yaml
```

两个 production 配置均固定：

- 200 epochs、micro batch 8、gradient accumulation 4，即 effective batch 32；
- `bf16-mixed`、AdamW、EMA 和 step-level warmup cosine；
- class-condition dropout 0.1 和 `v` prediction；
- 每 5 epochs 保存周期 checkpoint；
- 每个 epoch 运行 validation，恢复 best checkpoint 后运行一次 official test；
- 每 5 epochs 产出固定 seed 的 balanced class diagnostic 和 DDIM-50 CFG 2.0 artifact。

在 `drop_last: true` 下：

```text
micro-batches/epoch = floor(13,436 / 8) = 1,679
optimizer updates/epoch = ceil(1,679 / 4) = 420
total updates = 200 × 420 = 84,000
warmup updates = round(0.02 × 84,000) = 1,680
```

最后不足 4 个 micro-batches 的 accumulation window 会按实际长度 flush。scheduler、
EMA、global step 和 diagnostics 只在 optimizer update 成功后推进。production 默认
使用 `device: auto`；当它选中 CUDA 时要求 CUDA BF16 capability，框架不会静默降级。
CPU BF16 autocast 可用但不代表适合 production 吞吐。如果目标 CUDA 不支持 BF16，
应明确评估并修改为 `fp16-mixed`，而不是把一次运行中的 precision 改写为另一种语义。

TensorBoard：

```bash
uv run --project examples/showcases/afhq-v2 tensorboard --logdir outputs
```

## 5. Strict resume

训练输出根目录下会创建时间戳 run。使用 run directory 或 `latest.pt` 继续未完成的
200-epoch 作业：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --resume outputs/afhq-v2/adm-128/<run-id> \
  --epochs 200
```

checkpoint v9 严格恢复 model、Process、Objective、optimizer、scheduler、EMA、
precision/scaler topology、global step、epoch-boundary RNG、训练循环状态和 data
artifact identity。恢复前会用 `require/full` 重新验证同一个 prepared artifact；source、
recipe、manifest 或内容 identity 不一致时，在恢复训练资产前失败。

resume 创建新的 sibling run，不续写旧日志。它不能通过 config 替换 model、optimizer、
precision 或 accumulation。`--observability-config` 只用于允许的 diagnostics/logging
覆盖。

## 6. CFG 采样与结果

从训练 run 的 best checkpoint 生成每类 12 张、共 36 张 DDIM-50 sample：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow sample \
  --checkpoint outputs/afhq-v2/adm-128/<run-id> \
  --config examples/showcases/afhq-v2/experiments/sampling/ddim50-cfg2.yaml \
  --output-dir outputs/afhq-v2/samples/adm-ddim50-cfg2
```

sampling overlay 固定 seed `20260726`、CFG 2.0、EMA `auto` selection 和每 5 个求解步
保留一次 trajectory。class 顺序为 cat、dog、wild。非平凡 CFG scale 将 conditional
和 null labels 拼成双 batch，只进行一次模型 forward；DDIM 本身不解释 class 或
guidance。

一次 production run 的主要结果如下：

```text
outputs/afhq-v2/adm-128/<run-id>/
├── resolved_config.yaml
├── run_manifest.yaml
├── train.log
├── metrics.jsonl
├── tensorboard/
├── checkpoints/
│   ├── best.pt
│   ├── latest.pt
│   └── epoch_*.pt
├── diagnostics/class_conditional_diffusion_quality/epoch_*/
│   ├── manifest.yaml
│   ├── denoiser/
│   └── ddim_50/
└── samples/final/
```

显式 sampling 目录包含 `samples.pt`、`samples.png`、`trajectory.pt`、
`trajectory.png`、`trajectory.gif` 和 `resolved_sampling.yaml`。训练 manifest 记录
data artifact binding；checkpoint 记录同一 identity；sampling manifest 记录 checkpoint
lineage、seed、conditions、guidance、weights 和 artifact 路径，因此结果可以追溯到同一
训练输入。

## 7. 正式 KID/FID 评估

训练期 validation 和 diagnostic 不冒充正式生成质量结果。对冻结的 best checkpoint
执行 class-aware post-training evaluation：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow-afhq-v2-evaluate \
  --checkpoint outputs/afhq-v2/adm-128/<run-id>/checkpoints/best.pt \
  --config examples/showcases/afhq-v2/experiments/evaluation/ddim50-cfg2-kid-fid.yaml \
  --output-dir outputs/afhq-v2/evaluations/adm-ddim50-cfg2
```

协议从 authenticated official test split 按 manifest 顺序选择 cat/dog/wild 各 300 张
real images，并以相同 class allocation、EMA、seed `20260726`、DDIM-50 和 CFG 2.0
生成各 300 张 fake images。它报告 aggregate 与 per-class KID/FID；aggregate 是主要
分布指标，per-class 数值只作为细分诊断，不能忽略有限样本偏差。

命令先验证 checkpoint、execution device、quality provider 和严格
`DataArtifactBindings`，再调用现有 DataBuilder 与 SamplingBuilder。输出包含
`evaluation-result.json`、其 SHA-256 sidecar、immutable result manifest、规范化
sampling overlay，以及完整 sampling artifacts。结果冻结 checkpoint SHA-256、
epoch/global step、raw/EMA、data identity、extension provenance、metric protocol、
依赖版本和所有 artifact hashes。

评估开始时会把 checkpoint 的同一份已认证字节复制到 final output 的同文件系统私有
staging；checkpoint resolve、progress、内存权重和采样都只使用这份 snapshot。采样、
指标、result、digest 与 manifest 全部完成后，staging 才以 atomic no-replace rename
发布。失败或并发目标冲突不会留下半成品正式目录，也不会覆盖已有结果。

所有 data、checkpoint、benchmark 和普通 run artifacts 都位于被忽略的 `data/` 或
`outputs/`，不得提交到 Git。

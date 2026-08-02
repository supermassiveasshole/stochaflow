# AFHQ-v2 class-conditional generation showcase

这个可安装 example 展示一条完整、可恢复的 AFHQ-v2 生成数据流：

```text
官方 AFHQ-v2 archive
  -> AFHQV2ImageDataSource：读取、处理、完整性与安全审计
  -> 带类别语义和 official train/test 的确定性 128×128 managed artifact
  -> 内置 class_labeled_image Builder：分层划分、Dataset、Sampler、DataLoader
  -> 带 class_label 的 train/validation/test loaders
  -> ADM-UNet 或 DiT-B/8 类条件 Gaussian 训练
  -> training phase validation 与 epoch checkpoints
  -> validation-only Evaluation 选择唯一 subject
  -> frozen subject 的 one-shot official-test Evaluation
  -> EMA + classifier-free guidance DDIM sampling
  -> tensor、PNG、trajectory 与可追溯 manifest
  -> public full-official-test Evaluation、KID/FID 与可离线重放 predictions
```

这个 extension 只注册 `AFHQV2ImageDataSource`（`afhq-v2.official`）。它发布带
identity、类别映射和标签 inventory 的标准
`ClassLabeledImageFolderArtifactPayload`。配置使用 core
`class_labeled_image` Builder；Dataset、Sampler、collate 和 DataLoader 都不由 source
构造。ADM-UNet、DiT、Gaussian Process、训练 Strategy、CFG SamplingBuilder、
DDPM/DDIM、Trainer、EMA、checkpoint 和 artifact writers 也都使用 Stochaflow 的
内置注册路径；runner 中没有 AFHQ 名称分支。

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

`AFHQV2ImageDataSource` 对外提供 official train/test 及经过 source lock 认证的类别标签，数量为
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

这会创建/同步 showcase 自己的环境、安装 example 命令，并注册 extension entry point
`stochaflow-afhq-v2`。该 entry point 注册 `afhq-v2.official`、正式 AFHQ
`EvaluationBuilder` 与 class-aware distribution Metric；训练配置调用内置
`class_labeled_image` Builder。example 不引入平行的 source envelope、image recipe 或
loader config 类型；模型、训练、采样和评估所需的通用配置与生命周期都来自
Stochaflow。

- `stochaflow-afhq-v2-prepare`；
- `stochaflow-afhq-v2-capacity`；
- public `stochaflow evaluate`；
- legacy `stochaflow-afhq-v2-evaluate` 仅作为历史结果对照，不属于当前 maintained P2
  evidence surface，也不提供 compatibility guarantee。

### 当前 checkout 的正式 Evaluation 安装契约

正式 KID/FID 评估需要 showcase 声明的 optional `quality` extra。正式 P2 Evaluation
当前必须从本仓库 checkout 同步，并使用以下完整命令：

```bash
uv sync --project examples/showcases/afhq-v2 --locked --extra quality
```

这条 source-checkout 路径通过 showcase 的 `[tool.uv.sources]` 把 Stochaflow 绑定到
同一 checkout，不会要求 AFHQ wheel 从已发布 core 解析当前尚未发布的 Evaluation API。
不要把独立安装 AFHQ wheel 当作当前正式 P2 Evaluation 的入口。

仓库的 installed-wheel gate 会从当前 checkout 同时构建 core 与 AFHQ wheels，再以
`--no-deps --offline` 安装。它只验证 wheel 内容、隔离后的 extension entry point 与当前
core wheel 的 Evaluation 激活；不验证 AFHQ wheel 的 released-core resolver，也不证明
现有 GitHub Release core 包含这些尚未发布的 API。

协调 core/AFHQ 0.2 release，以及发布后在全新环境中只安装 AFHQ wheel、实际解析已发布
core 的 resolver smoke，是 post-release、non-blocking 的 follow-up。
它们不是当前 source-checkout P2 readiness 的 merge blocker；真实 core release 存在前
不得写入未来 release wheel URL。

## 1. 准备并验证数据

第一次运行使用 `ensure`。它可以下载官方 archive，也可以使用已经下载的本地文件。
如果 v2 object 已存在，`ensure` 与 `require` 都直接从最终 artifact root 加载 payload，
不会进入下载或图片转换逻辑，也不要求原始 ZIP 仍然存在：

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

`require` 不访问网络、不重建也不修复，因此所有 checked-in 训练配置都不会在启动时
悄悄下载或改变数据。ADM production 为降低每次 fresh training 启动时的完整 PNG
rehash 成本，固定使用 `require/manifest`；DiT production 与 smoke 固定使用
`require/full`。strict resume 注入 expected identity，因此无论原训练配置选择哪种
verification，恢复时都会强制执行 full verification。

prepare 命令通过注册的 `AFHQV2ImageDataSource` 执行，与内置 Builder 使用相同的
source 参数解析、materialization policy 和 identity 校验。命令只准备并汇报 official
train/test artifact；validation 参数属于 Builder，不是 prepare CLI 的输入。

未显式传入时，prepare CLI 的默认缓存根是 `./.stochaflow-cache`；checked-in 训练配置
显式使用 `./data`。统一 v2 cache 的相关部分如下：

```text
data/
├── source-acquisition/afhq-v2/raw/afhq-v2/<source-sha256>/afhq_v2.zip
└── data-artifacts/v2/managed/<artifact-type-digest>/
    ├── objects/<artifact-digest>/
    │   ├── manifest.json
    │   ├── inventory/
    │   └── data/
    │       ├── _index/images.json
    │       ├── train/{cat,dog,wild}/
    │       └── test/{cat,dog,wild}/
    ├── locators/
    ├── locks/
    ├── quarantine/
    └── staging/
```

消费者必须使用 payload inventory，不应绕过 DataSource 直接扫描缓存目录。
`AFHQV2ImageDataSource` 对外只暴露 official train/test。
准备阶段执行一次 Lanczos resize，不 crop，并使用固定 PNG 编码。训练阶段要求
authenticated size 精确为 128×128，不再在线 resize。DataBuilder 在内存中从 official
train 生成分层 validation，并输出 `(images, {"class_label": labels})`；shuffle 和
horizontal flip 都由 `(seed, epoch, sample identity)` 确定，因此 `num_workers: 2` 与
persistent workers 下的 epoch-boundary resume 仍可重建同一顺序和增强。

`manifest.json` 是 framework schema-v2 envelope；AFHQ 的 domain 只声明 resolution、
class mapping、official partition roots/counts 和 `_index/images.json` descriptor。
`verification: manifest` 验证 envelope 与这个 producer-defined cheap contract；
`verification: full` 还会认证 framework inventory 中的全部转换后 PNG。expected
identity 总是强制 full。旧 `prepared/afhq-v2` cache、`dataset_manifest.yaml`、
`files.sha256` 和 schema-v1 checkpoint binding 没有 reader 或迁移路径；升级后必须重新
materialize，并从新 run 开始。

## 2. 真实训练容量报告

准备好 AFHQ-v2 artifact 后，在目标 CUDA 主机运行真实训练 profiler：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow-afhq-v2-capacity \
  --config examples/showcases/afhq-v2/experiments/profiling/train-adm-128-p2.yaml \
  --micro-batches 1 4 6 8 \
  --precisions bf16-mixed \
  --warmup-updates 5 \
  --measured-updates 25 \
  --output outputs/benchmarks/afhq-v2/adm-128-capacity.json
```

每个 trial 都通过已注册 DataBuilder、TrainingBuilder、Trainer 和 precision runtime 执行。
DataBuilder 通过 `ImageSourceFactory` 解析注册的 DataSource、验证同一 official
train/test artifact，再执行运行时分层划分并组装 loader；capacity 工具不会使用独立
source 或训练循环。本次 micro batch 1/4/6/8 分别使用 accumulation 32/8/5/4，effective
batch 分别为 32/32/30/32。每个 trial warmup 5 次，并测量 25 次成功 optimizer updates；
报告 images/s、updates/s、allocated/reserved peak VRAM、data-wait/compute、细分
forward/backward/optimizer 时间和非有限 loss/gradient。若同一 batch 同时请求 FP32 与
BF16，报告还会提供 BF16 相对 FP32 的吞吐和显存差值。报告记录 Python、PyTorch、CUDA、
cuDNN、GPU 型号和 capability，便于比较结果。CUDA phase 使用异步 Events，整段测量仅在
开始和结束同步，因此 phase 观测不会在每次 forward/backward/optimizer 边界串行化训练。

报告固定同一 DataBuilder 返回的 source artifact bindings；运行时 partition policy 则由
每个 trial 的完整 resolved config、canonical SHA-256 和 seed 冻结。output directory、
core/extension 版本及 Python source tree digest 也会进入 code identity。device 和
precision support 在 meta model、DataBuilder 及 trial output 创建之前完成 preflight；
全部 precision 不受支持时不会读取数据或构建模型。

命令默认要求 CUDA，避免 `device: auto` 在无 CUDA 时静默运行巨型 CPU trial。仅做有界
测试或调试时显式传入 `--device cpu`；CPU 报告的 VRAM 字段为 `null`，不应用于
production batch 选择。

当前有效 schema-v3 report 来自 RTX 4090 24,564 MiB、PyTorch 2.11 / cu128，使用 corrected
105,197,187-parameter topology、P2 training 与 BF16 mixed precision。四档都完成 5 次
warmup 和 25 次 measured optimizer updates：

| Micro batch | Accumulation | Effective batch | Images/s | Peak allocated (GiB) | Peak reserved (GiB) | Non-finite |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 32 | 12.945 | 2.980 | 3.043 | 0 |
| 4 | 8 | 32 | 47.057 | 5.251 | 5.436 | 0 |
| 6 | 5 | 30 | 54.543 | 6.761 | 6.980 | 0 |
| 8 | 4 | 32 | 60.068 | 8.260 | 8.506 | 0 |

这是 corrected topology 的 operational capacity 与 25-update sustained evidence。micro
batch 8 是本次已测候选中吞吐最高的一档，并不等于硬件可支持的绝对上限。两份 maintained
ADM production YAML 因此都使用 micro batch 8 / accumulation 4，保持 effective batch 32、每
epoch 420 optimizer updates 和总计 84,000 updates。该报告不证明长训练稳定性、收敛、
生成质量或 standard/P2 A/B 收益；未来 DGX Spark 运行属于独立的跨设备证据。

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

### P2 真实数据 tiny wiring lane

P2 的最小端到端入口是独立的 maintained profile：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/smoke/train-adm-128-p2.yaml \
  --device cuda \
  --limit-validation-batches 2 \
  --limit-test-batches 2
```

它使用真实 prepared AFHQ-v2、tiny class-conditional ADM、epsilon/fixed-variance P2
Builder、`k: 1`、`gamma: 1`、4 个 training micro-batches 和 accumulation 2，因此只产生
2 次 optimizer update。当前工作站的 CUDA 实跑已覆盖 optimizer、EMA update、step/epoch
diagnostics、limited validation、limited official test 和 best/latest/epoch checkpoint。
这证明 P2 与真实 AFHQ task lifecycle 的 wiring；它不使用 production topology，也不提供
capacity、收敛、质量或 standard/P2 A/B 证据。

### P2 corrected full-topology bounded sanity

105M corrected ADM 的短硬件 sanity 使用另一份明确标为 profiling 的配置：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/profiling/train-adm-128-p2.yaml \
  --device cuda \
  --limit-validation-batches 2 \
  --limit-test-batches 2
```

该 profile 保留 production ADM 的 105,197,187 参数 canonical topology、128×128 data、
cosine 1,000-step Process 和 BF16 mixed precision，但只用 micro batch 1、accumulation 1、
8 个 optimizer updates、短 scheduler，并关闭 diagnostics。当前工作站实跑完成 8 次 update、
validation、official-test lifecycle 与 checkpoint publication；训练 compute 观测值为
4.34 optimizer steps/s。

这个数字只描述该有界运行的 compute phase，不是 effective-batch-32 production throughput。
该 profile 本身也不是 `stochaflow-afhq-v2-capacity` 的替代品，不能把它的 4.34 steps/s
解释成 micro batch 上限、peak VRAM、长训练稳定性、收敛、质量或 A/B 收益。上面的独立
schema-v3 sweep 才是 production batch 与 peak-memory 口径的 authority。

## 4. Production 训练

现有 v-prediction 主展示模型是 ADM-UNet：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/production/train-adm-128.yaml
```

当前分支合并后的 production-quality 实验入口是独立的 P2 candidate：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/production/train-adm-128-p2.yaml \
  --device cuda \
  --deterministic
```

它固定 corrected 105,197,187-parameter ADM、真实 AFHQ、epsilon/fixed P2 Builder、
`k: 1`、`gamma: 1`、BF16、micro batch 8 / accumulation 4、200 epochs / 84,000 updates
和 EMA decay 0.9999。这是单臂 absolute-quality candidate，不是 standard/P2 因果对照。
若未来要声称“P2 优于 standard”，必须另跑一份只把同一 production profile 的 `gamma`
改为 0 的 matched control；不能拿 v-prediction baseline 代替。

DiT-B/8 候选配置不属于当前 P2 closeout；它使用与 v-prediction ADM 相同的 data、Process、
Objective、effective batch、scheduler、diagnostic 和 sampling protocol：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/production/train-dit-128.yaml
```

`adm_unet` 现在直接表示 canonical ADM graph：initial convolution、每个 encoder
ResBlock 和每个 residual downsample 都保存 skip；decoder 每个 resolution 使用
`num_res_blocks + 1` 个 ResBlock，并逐 block 消费 skip。production 的
`num_res_blocks: 2` 因而表示 encoder 每级 2 blocks、decoder 每级 3 blocks。
`attention_resolutions: [32, 16, 8]` 是实际空间尺寸，使用
GroupNorm/QKV/output-projection residual attention；middle 固定为
`ResBlock → Attention → ResBlock`。checked-in production config 的 exact parameter
count 是 105,197,187。

这是 breaking cutover。旧 topology fields 与旧 ADM raw/EMA/optimizer checkpoint 都会
fail closed；没有 legacy mode、partial load、state adapter 或自动 config conversion，
必须 fresh train。

v-prediction ADM 与 DiT 候选配置均固定 effective batch 32，其中：

- ADM 使用实测 micro batch 8、gradient accumulation 4；
- DiT 使用 micro batch 32、gradient accumulation 1；
- 两者均训练 200 epochs；
- `bf16-mixed`、AdamW、EMA 和 step-level warmup cosine；
- class-condition dropout 0.1 和 `v` prediction；
- 每 5 epochs 保存周期 checkpoint；
- 每个 epoch 的 phase validation、`best.pt` 与 phase test 只服务训练 lifecycle/监控，不作
  P2 formal-quality subject selection；
- 每 5 epochs 产出固定 seed 的 balanced class diagnostic 和 DDIM-50 CFG 2.0 artifact。

上面的 P2 smoke/profiling profiles 是 readiness lanes，不会替换 maintained
`train-adm-128-p2.yaml`，也不会把其 2-update/8-update scheduler、micro batch 或无
diagnostics 设置提升为 production 默认值。

在 `drop_last: true` 下，ADM 的更新计划为：

```text
micro-batches/epoch = floor(13,436 / 8) = 1,679
optimizer updates/epoch = ceil(1,679 / 4) = 420
total updates = 200 × 420 = 84,000
warmup updates = round(0.02 × 84,000) = 1,680
```

这组 8 / 4 设置来自上面的 RTX 4090 schema-v3 P2 capacity report，在保持 effective batch
32 与 84,000-update schedule 的同时取本次已测候选中的最高吞吐档。它不是长期训练质量
结论，也不声称 micro batch 8 是硬件上限；迁移到其他设备时应重跑同一 capacity protocol，
并记录 hardware adaptation。

DiT 的更新计划为：

```text
micro-batches/epoch = floor(13,436 / 32) = 419
optimizer updates/epoch = ceil(419 / 1) = 419
total updates = 200 × 419 = 83,800
warmup updates = round(0.02 × 83,800) = 1,676
```

ADM 最后不足 4 个 micro-batches 的 accumulation window 会按实际长度 flush。scheduler、
EMA、global step 和 diagnostics 只在 optimizer update 成功后推进。P2 production 固定
`device: cuda`，命令也显式传入 `--device cuda`；缺少 CUDA 时会在创建 run 前失败，不会
静默启动巨型 CPU 作业。CUDA 还必须支持 BF16。如果目标 CUDA 不支持 BF16，应先形成新的
容量验证与 hardware adaptation，而不是在一次正式运行中改写 precision。

TensorBoard：

```bash
uv run --project examples/showcases/afhq-v2 tensorboard --logdir outputs
```

## 5. Strict resume

训练输出根目录下会创建时间戳 run。使用 run directory 或 `latest.pt` 继续未完成的
200-epoch 作业：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --resume outputs/afhq-v2/adm-128-p2/<run-id> \
  --epochs 200 \
  --artifact-verification-workers 8 \
  --device cuda \
  --deterministic
```

checkpoint v12 严格恢复 model、Process、Objective、optimizer、scheduler、EMA、
precision/scaler topology、global step、epoch-boundary RNG、训练循环状态和 data
artifact identity。恢复前会用 `require/full` 重新验证同一个 prepared artifact；source、
materialization、manifest 或内容 identity 不一致时，在恢复训练资产前失败。运行时
validation policy 不进入 artifact identity，而由 checkpoint 的 resolved config 与 seed
固定。artifact 哈希默认使用 `min(8, logical CPUs)` 个线程；可以在 YAML 的
`source.materialization.verification_workers` 配置 `1..8` 范围内的整数，或用
`--artifact-verification-workers` 仅覆盖本次启动。

当前 Gaussian inference recipe 还显式冻结 `variance.mode`（production 为 `fixed`）。
v10 及更早 checkpoint 会被 strict resume 拒绝，框架不会补写或迁移；旧 ADM
checkpoint 还另外存在完整 topology/state 不兼容，因此 sampling 也会失败。

resume 创建新的 sibling run，不续写旧日志。它不能通过 config 替换 model、optimizer、
precision 或 accumulation。`--observability-config` 只用于允许的 diagnostics/logging
覆盖。

## 6. CFG 采样与结果

可以从训练 run 的 convenience `best.pt` 生成每类 12 张、共 36 张 DDIM-50 展示 sample；
该命令不决定 formal Evaluation subject：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow sample \
  --checkpoint outputs/afhq-v2/adm-128/<run-id> \
  --config examples/showcases/afhq-v2/experiments/sampling/ddim50-cfg2.yaml \
  --output-dir outputs/afhq-v2/samples/adm-ddim50-cfg2
```

这份 profile 是完整、独立的 sample invocation：`sample:` 显式冻结 sampler、options、
shape、sample 数量、batch、seed 和 writers，不从 checkpoint config 继承采样默认值。
checkpoint 只提供训练时冻结的 inference recipe 与模型、Process 等推理资产。训练命令
不会自动执行这次采样；必须另行运行 `stochaflow sample`，并显式提供必填的 `--config`
来选择完整 profile。class 顺序为 cat、dog、wild。
非平凡 CFG scale 将 conditional 和 null labels 拼成双 batch，只进行一次模型
forward；DDIM 本身不解释 class 或 guidance。

production 当前使用 fixed variance。其他 class-conditional learned-range recipe 的
`2C` CFG 只 guide prediction half；scale 0/1 返回完整 unconditional/conditional
branch，其他 scale 保留 conditional variance half。DDPM 消费 learned variance，DDIM
明确忽略 variance half。

### Corrected ADM 结果状态

canonical topology 切换前的 900-real/900-generated DDIM-50 AFHQ 指标、checkpoint 与
sample panel 来自旧的、不兼容 ADM graph，已从 current result surface 移除。它们不能
证明 corrected ADM、learned variance 或 P2。在 corrected production 200-epoch long run
完成并冻结新 checkpoint、resolved config 与 evaluation artifacts 前，本 README 不发布
corrected ADM 的 production long-run quality baseline。下文已经发布的单 epoch 受控 A/B
数值只属于 pipeline/protocol readiness evidence，不能替代该 baseline。

新的三类 production run 使用 public class-aware formal profile，统一报告 aggregate 与
cat/dog/wild per-class KID/FID。

### Production 输出布局

一次 production run 的主要 artifact 如下：

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
└── diagnostics/class_conditional_diffusion_quality/epoch_*/
│   ├── manifest.yaml
│   ├── denoiser/
│   └── ddim_50/
```

独立 `stochaflow sample` 命令指定的 sampling 目录包含 `samples.pt`、`samples.png`、
`trajectory.pt`、`trajectory.png`、`trajectory.gif` 和 `resolved_sampling.yaml`。
训练 manifest 记录 data artifact binding；checkpoint 记录同一 identity；sampling
manifest 记录 checkpoint lineage、seed、conditions、guidance、weights 和 artifact 路径，
因此结果可以追溯到同一训练输入。

## 7. 正式 KID/FID 评估

训练期 `valid/loss`、phase-test metric 和 diagnostic 不冒充正式生成质量结果，也不用于
P2 production subject selection。机器可读的
[`p2-production-closeout-policy.yaml`](experiments/evaluation/p2-production-closeout-policy.yaml)
已经冻结 eligible epochs 20、40、60、80、100、120、140、160、180、200，primary 为
validation aggregate FID lower，tie-break 为 aggregate KID mean lower、再 earliest epoch。
这 900 张 validation real/fake 只用于候选排序，没有 pass/fail 或 acceptance 权限。
训练完成后，对每个候选 EMA 分别运行 validation-only profile；每次替换
`REPLACE_WITH_RUN_ID`、zero-padded `epoch_XXXX.pt` 与新的 output directory：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow evaluate \
  --config examples/showcases/afhq-v2/experiments/evaluation/selection-ddim50-cfg2-validation-epsilon.yaml \
  --device cuda \
  --output-dir outputs/afhq-v2/evaluations/p2-selection/<epoch-id>
```

该 `purpose: selection_candidate` profile 固定 validation split、cat/dog/wild 各 300 个
real/fake、EMA、epsilon/fixed、seed `20260726`、DDIM-50、eta 0 与 CFG 2.0。只按预声明规则
从这些 validation Evaluation results 冻结唯一 subject 并保存选择记录；不得用
`valid/loss`、diagnostic、phase test 或 official test 改选。

对与 checked-in v-prediction/fixed-variance contract 匹配的另一个已冻结 subject，使用
下面的 public class-aware post-training profile。先只替换 `subject.path`；
`subject.weights: ema`、完整 protocol、采样方法、类别 allocation 和 metrics 保持不变：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow evaluate \
  --config examples/showcases/afhq-v2/experiments/evaluation/formal-ddim50-cfg2-official-test.yaml \
  --device cuda \
  --output-dir outputs/afhq-v2/evaluations/adm-ddim50-cfg2-official-test
```

当前 P2 production candidate 必须把 validation 选出的唯一 epoch EMA 填入
`formal-ddim50-cfg2-official-test-epsilon.yaml` 的 `REPLACE_WITH_RUN_ID` 和
`REPLACE_WITH_SELECTED_EPOCH_CHECKPOINT.pt`，然后只运行一次 full official test：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow evaluate \
  --config examples/showcases/afhq-v2/experiments/evaluation/formal-ddim50-cfg2-official-test-epsilon.yaml \
  --device cuda \
  --output-dir outputs/afhq-v2/evaluations/p2-production-official-test
```

这次 one-shot result 只验收 P2 candidate 的 absolute quality。它必须 exact/complete
1,467 examples，并同时满足 aggregate FID ≤ 35、aggregate KID mean ≤ 0.01 和每类
FID ≤ 65；aggregate FID ≤ 30 是 aspirational target。它不能反向改变 selection，也不能
单独证明 P2 相对 standard 的 superiority。这些 `internal_project_acceptance`
thresholds 只适用于 `train-adm-128-p2.yaml` 的 epsilon/fixed P2 subject；standard-v ADM
和 DiT 配置不继承它们。

profile 消费完整 authenticated official test split：cat/dog/wild 分别为
493/491/483 张，共 1,467 张 reference，并以相同 class allocation 生成 1,467 张 fake。
它固定 EMA、seed `20260726`、deterministic DDIM-50、CFG 2.0、128×128 RGB、KID
100 subsets / subset size 300 / seed `20260726` 和 FID feature 2048。结果同时包含 aggregate 与 per-class
KID/FID；aggregate 是主要分布指标，per-class 数值只作细分诊断，不能忽略有限样本偏差。

这是“AFHQ-v2 官方 test split + 本项目自定义 class-conditional 128×128 / DDIM-50 /
CFG 2.0 diffusion protocol”，不是论文复现协议。P2 论文及其
[作者仓库](https://github.com/jychoi118/P2-weighting)使用 unconditional AFHQ-Dog
256×256 设置；[CVPR 2022 论文](https://openaccess.thecvf.com/content/CVPR2022/html/Choi_Perception_Prioritized_Training_of_Diffusion_Models_CVPR_2022_paper.html)
报告的 AFHQ-Dog P2 FID 11.55 因数据子集、分辨率、条件方式、real/fake sample plan 和
采样协议均不同，不能与这里的 FID 横向比较。

runtime 只把已经解析为 EMA（或另一个 profile 显式选择的 raw）的 primary model 注入
AFHQ Builder，Builder 不能再次选择权重。生成经由 checkpoint-bound
`EvaluationSamplingCapability` 调用与 `stochaflow sample` 共用的 SamplingBuilder execution
seam；该调用不发布普通 sampling writers。FID/KID provider 通过
`REGISTRIES.metrics` 构造，AFHQ Metric 负责 aggregate/per-class scopes 和每类
reference/generated exact completeness。

成功目录为：

```text
outputs/afhq-v2/evaluations/adm-ddim50-cfg2-official-test/
├── predictions/
│   ├── prediction_manifest.json
│   └── predictions.jsonl
├── resolved_evaluation.yaml
├── result.json
└── evaluation_manifest.yaml
```

`predictions/` 冻结 exact sample plan、EMA/source checkpoint、data/split、采样 profile、
pre/postprocess、extension lineage、shard digests 和 deterministic gallery IDs。若要增加或
复核 metrics，复制 formal profile，并仅把 authority 改为：

```yaml
subject:
  kind: prediction_artifact
  path: ../../../../../outputs/afhq-v2/evaluations/adm-ddim50-cfg2-official-test/predictions/prediction_manifest.json
data:
  source: prediction_artifact
  split: test
```

然后继续运行 `stochaflow evaluate`。offline replay 认证 manifest/shards 并按 exact
sample plan 重放同一 AFHQ Metric，不加载 checkpoint、不构造模型、不再次执行
SamplingBuilder 或原 DataBuilder，也不修改 producer artifact。

旧 `stochaflow-afhq-v2-evaluate` 与
`experiments/evaluation/ddim50-cfg2-kid-fid.yaml` 仅作为历史结果对照；它们不属于当前
maintained P2 evidence surface，也不提供 compatibility guarantee。新的 benchmark、报告
和 P2 证据都必须来自 public `stochaflow evaluate`。checked-in
`formal-ddim50-cfg2-official-test.yaml` 继续固定当前 v-prediction production baseline。

### Epsilon/fixed control–P2 A/B protocol and result

历史 control 与 P2 treatment 使用与当前 production official profile 相同的 epsilon/fixed
协议字段。该 profile 现在以 production run/selected-epoch placeholder fail closed；历史
两臂分别把 subject path 替换为预算终点的 `latest.pt`。协议固定 epsilon prediction、fixed
variance、完整 official test 493/491/483 allocation、EMA、seed
`20260726`、deterministic DDIM-50、eta 0、CFG 2.0、evaluation batch 30 与同一 KID/FID
参数。对每个 arm 只替换 `subject.path` 和新的 output directory，然后分别运行：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow evaluate \
  --config examples/showcases/afhq-v2/experiments/evaluation/formal-ddim50-cfg2-official-test-epsilon.yaml \
  --device cuda \
  --output-dir outputs/afhq-v2/evaluations/<control-or-p2>-epsilon-latest
```

已完成的受控训练使用同一 corrected 105,197,187-parameter ADM、真实 AFHQ、seed
`20260726`、deterministic runtime、BF16、micro batch 8 / accumulation 4、完整 1 epoch
（1,679 micro-batches / 420 optimizer updates）以及从 step 0 开始、decay 0.999 的 EMA。
control 仍走 P2 Builder，但设置 `gamma: 0` 形成 strict-standard epsilon control；treatment
只把这一算法字段改为 `gamma: 1`。control 训练 4m28s，treatment 训练 4m23s。

两个 arm 冻结各自在相同 budget 结束时的 `latest.pt` EMA，而不是按不可比的
`valid/loss` 选择 `best.pt`：control loss 与 P2-weighted loss 不是
共同 selection objective，这样会同时改变训练方法和 checkpoint-selection policy。

正式 protocol ID 是
`afhq-v2-adm-epsilon-ddim50-cfg2-official-test-v1`。两臂都完整发布 1,467 个 unique IDs，
cat/dog/wild 分别为 493/491/483；`sample_ids_sha256` 相同（`b66fc...d6c1`）。control
checkpoint SHA 为 `6dd0...2196`，P2 为 `b02b...fa4a`。lower-is-better 结果为：

| Scope | Control FID | P2 FID | FID delta | Control KID mean | P2 KID mean | KID delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aggregate | 369.621427 | 371.250343 | +1.628916 | 0.476357937 | 0.479742199 | +0.003384262 |
| Cat | 381.980901 | 383.273453 | +1.292552 | 0.551076353 | 0.553966224 | +0.002889871 |
| Dog | 382.850132 | 385.106413 | +2.256281 | 0.484312266 | 0.488923877 | +0.004611611 |
| Wild | 370.417725 | 371.661225 | +1.243500 | 0.502315342 | 0.504731715 | +0.002416373 |

P2 在 aggregate 和每个 class 上都一致略差，因此这一轮没有显示收益。KID delta 与其
reported standard deviation 同量级；单 seed、单 epoch 不能支持统计显著声明，也不能作为
200-epoch promotion evidence 或推广为一般质量结论。工程 wiring、受控协议和 formal
Evaluation readiness 已完成，但 production long-run gate 仍然开放。

public runtime 在评估开始前认证稳定 checkpoint snapshot、execution device、quality
provider 和严格 `DataArtifactBindings`。只有采样、metrics、prediction manifest、result
与 completion manifest 全部成功后才原子发布新目录；失败或并发目标冲突不会留下
半成品正式目录，也不会覆盖已有结果。

所有 data、checkpoint、benchmark 和普通 run artifacts 都位于被忽略的 `data/` 或
`outputs/`，不得提交到 Git。

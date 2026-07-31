# AFHQ-v2 generation showcase

这个可安装 example 展示一条完整、可恢复的 AFHQ-v2 生成数据流：

```text
官方 AFHQ-v2 archive
  -> AFHQV2ImageDataSource：读取、处理、完整性与安全审计
  -> 带类别语义和 official train/test 的确定性 128×128 managed artifact
  -> 内置 class_labeled_image Builder：分层划分、Dataset、Sampler、DataLoader
  -> 带 class_label 的 train/validation/test loaders
  -> ADM-UNet 或 DiT-B/8 类条件 Gaussian 训练
  -> validation、best checkpoint 与 official test
  -> EMA + classifier-free guidance DDIM sampling
  -> tensor、PNG、trajectory 与可追溯 manifest
```

同一 extension 另提供独立 research lane：

```text
authenticated AFHQ-v2 train/dog
  -> guided-diffusion BOX/bicubic/center-crop 256px materialization
  -> generic unlabeled image artifact + core image Builder
  -> unconditional canonical ADM + learned-range variance
  -> constant/P2 controlled A/B + 1000/250-step ancestral DDPM
```

这个 extension 注册两个 DataSource：`afhq-v2.official` 发布带 identity、类别映射和
标签 inventory 的标准 `ClassLabeledImageFolderArtifactPayload`；
`afhq-v2.dog` 只物化 authenticated train/dog subset，并发布不含 labels 的通用
`ImageFolderArtifactPayload`。前者使用 core `class_labeled_image` Builder，后者使用
core `image` Builder；Dataset、Sampler、collate 和 DataLoader 都不由 source 构造。
ADM-UNet、DiT、Gaussian Process、训练 Strategy、CFG SamplingBuilder、DDPM/DDIM、
Trainer、EMA、checkpoint 和 artifact writers 也都使用 Stochaflow 的内置注册路径；
runner 中没有 AFHQ 名称分支。

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

这会创建/同步 showcase 自己的环境、安装三个 example 命令，并注册 extension entry
point `stochaflow-afhq-v2`。entry point 注册 `afhq-v2.official` 与
`afhq-v2.dog`；训练配置分别调用内置 `class_labeled_image` 与 `image` Builder。
example 不引入平行的 builder、source envelope、image recipe 或 loader config 类型；
模型、训练、采样和评估所需的通用配置与生命周期也都来自 Stochaflow。

- `stochaflow-afhq-v2-prepare`；
- `stochaflow-afhq-v2-capacity`；
- `stochaflow-afhq-v2-evaluate`。

正式 KID/FID 评估需要 showcase 声明的 optional quality extra：

```bash
uv sync --project examples/showcases/afhq-v2 --locked --extra quality
```

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

### P2-compatible Dog artifact

准备/验证 unlabeled 256px train/dog artifact：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow-afhq-v2-prepare \
  --profile dog \
  --cache-root ./data \
  --policy ensure \
  --verification full

uv run --project examples/showcases/afhq-v2 stochaflow-afhq-v2-prepare \
  --profile dog \
  --cache-root ./data \
  --policy require \
  --verification full
```

Dog profile 默认且只接受 resolution 256，使用同一 authenticated archive 的 4,678 张
official train/dog images。materializer pin
OpenAI guided-diffusion commit
`8fb3ad9197f16bbc40620447b2742e13458d2831`，按逐级 BOX downsample、bicubic resize、
integer center crop 的顺序对齐 `center_crop_arr`，再用固定 RGB PNG policy 编码。
transform、source subset、file inventory 与 Pillow/NumPy/zlib 版本进入 artifact
identity。horizontal flip 留给 generic `image` Builder 的 seeded runtime policy。

`afhq-v2.dog` 返回只有 train 的普通 `ImageFolderArtifactPayload`；路径中的 `dog/`
不会作为 class label 进入 batch。这条 source 准确命名为
**P2-compatible AFHQ-v2 Dog**。P2 公开材料没有冻结历史 AFHQ version、file list、seed
和 metric/checkpoint protocol，因此不能称为 exact P2 AFHQ-D reproduction。

## 2. 真实训练容量报告

准备好 AFHQ-v2 artifact 后，在目标 CUDA 主机运行真实训练 profiler：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow-afhq-v2-capacity \
  --config examples/showcases/afhq-v2/experiments/production/train-adm-128.yaml \
  --micro-batches 1 2 4 8 \
  --precisions fp32 bf16-mixed \
  --warmup-updates 5 \
  --measured-updates 25 \
  --output outputs/benchmarks/afhq-v2/adm-128-capacity.json
```

每个 trial 都通过已注册 DataBuilder、TrainingBuilder、Trainer 和 precision runtime 执行。
DataBuilder 通过 `ImageSourceFactory` 解析注册的 DataSource、验证同一 official
train/test artifact，再执行运行时分层划分并组装 loader；capacity 工具不会使用独立
source 或训练循环。micro batch 1/2/4/8 分别使用 accumulation 32/16/8/4，均保持
effective batch 32。默认每个 trial warmup 5 次，并测量至少 25 次成功 optimizer updates；
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

仓库尚无 corrected 105M ADM 的 4090/DGX measured capacity report。promotion gate
要求在 4090 上完成 BF16 micro batch 1/2/4/8 trials，并在 DGX Spark 上用同一 resolved
config 执行 smoke/resume/sample。checked-in micro batch 1 是 provisional memory-safe
default，不是实测最大容量或吞吐最优结论。

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

DiT-B/8 使用相同 data、Process、Objective、effective batch、scheduler、diagnostic 和
sampling protocol：

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

两个 production 配置均固定 effective batch 32，其中：

- ADM 暂用 micro batch 1、gradient accumulation 32；
- DiT 使用 micro batch 32、gradient accumulation 1；
- 两者均训练 200 epochs；
- `bf16-mixed`、AdamW、EMA 和 step-level warmup cosine；
- class-condition dropout 0.1 和 `v` prediction；
- 每 5 epochs 保存周期 checkpoint；
- 每个 epoch 运行 validation，恢复 best checkpoint 后运行一次 official test；
- 每 5 epochs 产出固定 seed 的 balanced class diagnostic 和 DDIM-50 CFG 2.0 artifact。

在 `drop_last: true` 下，ADM 的更新计划为：

```text
micro-batches/epoch = floor(13,436 / 1) = 13,436
optimizer updates/epoch = ceil(13,436 / 32) = 420
total updates = 200 × 420 = 84,000
warmup updates = round(0.02 × 84,000) = 1,680
```

这组 micro batch/accumulation 是 corrected topology 完成目标设备 profile 前的保守默认，
不证明 micro batch 8 仍可用。若实测允许提高 micro batch，必须相应降低 accumulation，
保持 effective batch 32 与 84,000-update schedule，并记录 hardware adaptation。

DiT 的更新计划为：

```text
micro-batches/epoch = floor(13,436 / 32) = 419
optimizer updates/epoch = ceil(419 / 1) = 419
total updates = 200 × 419 = 83,800
warmup updates = round(0.02 × 83,800) = 1,676
```

ADM 最后不足 32 个 micro-batches 的 accumulation window 会按实际长度 flush。scheduler、
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
  --epochs 200 \
  --artifact-verification-workers 8
```

checkpoint v10 严格恢复 model、Process、Objective、optimizer、scheduler、EMA、
precision/scaler topology、global step、epoch-boundary RNG、训练循环状态和 data
artifact identity。恢复前会用 `require/full` 重新验证同一个 prepared artifact；source、
materialization、manifest 或内容 identity 不一致时，在恢复训练资产前失败。运行时
validation policy 不进入 artifact identity，而由 checkpoint 的 resolved config 与 seed
固定。artifact 哈希默认使用 `min(8, logical CPUs)` 个线程；可以在 YAML 的
`source.materialization.verification_workers` 配置 `1..8` 范围内的整数，或用
`--artifact-verification-workers` 仅覆盖本次启动。

当前 Gaussian inference recipe 还显式冻结 `variance.mode`（production 为 `fixed`）。
变更前缺少该字段的 v10 Gaussian checkpoint 会被 strict resume 拒绝，框架不会补写；
non-ADM sampling 仍采用 fixed-compatible default。旧 ADM checkpoint 还另外存在完整
topology/state 不兼容，因此 sampling 也会失败。

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

这份 partial request 只显式选择 DDIM-50 和 CFG 2.0。sample 数量、batch、seed、
class allocation、weights、trajectory 和 writers 都从 checkpoint defaults 继承；
checkpoint-required AFHQ extension 也不需要重复声明。class 顺序为 cat、dog、wild。
非平凡 CFG scale 将 conditional 和 null labels 拼成双 batch，只进行一次模型
forward；DDIM 本身不解释 class 或 guidance。

production 当前使用 fixed variance。其他 class-conditional learned-range recipe 的
`2C` CFG 只 guide prediction half；scale 0/1 返回完整 unconditional/conditional
branch，其他 scale 保留 conditional variance half。DDPM 消费 learned variance，DDIM
明确忽略 variance half。

### Corrected ADM 结果状态

canonical topology 切换前的 900-real/900-generated DDIM-50 AFHQ 指标、checkpoint 与
sample panel 来自旧的、不兼容 ADM graph，已从 current result surface 移除。它们不能
证明 corrected ADM、learned variance 或 P2。corrected production 完成长训练并冻结新
checkpoint、resolved config 与 evaluation artifacts 前，本 README 不发布 AFHQ quality
数值。

即使新的三类 production run 使用现有 class-aware 900-sample evaluator，其结果也不能与
P2 的 50,000-fake ancestral protocol 直接比较：data domain、sample count、reference
set 与 solver 都不同。

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

## 7. P2-compatible AFHQ-v2 Dog research

research authority 与 production config 分离：

```text
experiments/research/p2-afhq-v2-dog-256/
├── train-base.yaml
├── p2-loss-weighting.yaml
├── sample-ddpm-250.yaml
└── sample-ddpm-1000.yaml
```

唯一完整 `train-base.yaml` 固定 unlabeled Dog-256、canonical unconditional
93,563,910-parameter ADM、epsilon prediction、learned-range/rescaled VB、linear
`T=1000`、constant weighting、MSE mean、AdamW `2e-5`/weight decay 0、EMA 0.9999、
FP32、batch 8 和 300,000 updates/2.4M images。training/generation seeds 分别为
`20260730`/`20260731`。

effective batch 8 是 frozen protocol fact；如果 4090 只能使用更小 micro batch，必须用
accumulation 保持 effective batch，并在 resolved config/provenance 中记录 adaptation，
不能静默改变 FP32 reference recipe。

restricted resolver 只允许 P2 variant 把
`training.params.loss_weighting` 改为 `{name: p2, k: 1.0, gamma: 1.0}`，并记录
base/override SHA-256、唯一 changed path 和 resolved-config SHA-256：

```bash
uv run --project examples/showcases/afhq-v2 \
  python -m stochaflow_afhq_v2.tools.benchmark_config \
  --variant constant \
  --output outputs/afhq-v2/research/configs/constant.yaml \
  --provenance outputs/afhq-v2/research/configs/constant.provenance.json

uv run --project examples/showcases/afhq-v2 \
  python -m stochaflow_afhq_v2.tools.benchmark_config \
  --variant p2 \
  --output outputs/afhq-v2/research/configs/p2.yaml \
  --provenance outputs/afhq-v2/research/configs/p2.provenance.json
```

P2 使用 cumulative-SNR 权重 `(1 + snr(t))^-1`，只乘 epsilon simple MSE，不做 batch
renormalization，也不乘 detached-mean learned-variance VB term。sample profiles只选择
EMA 1,000-step full ancestral 或 250-step uniform-section selected-pair ancestral
DDPM；250-step 不是 DDIM，P2 training fields 不出现在 sample request。

这里冻结的是 algorithm-compatible protocol，不是已完成 benchmark。正式声明仍需要完成
两组 run、50,000 fake completeness、固定 reference/FID/KID implementation、capacity
evidence 和 checkpoint policy。历史 AFHQ version/file list 无法从 P2 公开材料恢复，
因此只能称 P2-compatible AFHQ-v2 Dog，不能称 exact P2 AFHQ-D reproduction。

两份 sample request 继承 50,000 samples，但它们目前只是 frozen future protocol
inputs。当前 Tensor writer 会缓冲并拼接完整 `50000 × 3 × 256 × 256` FP32 output，
samples payload 约 39 GB，尚未计额外运行时内存。正式执行必须等待 sharded prediction
artifact 与 completeness authority；不要直接把当前一次性 `stochaflow sample` 当作
operational 50k benchmark。

## 8. 正式 KID/FID 评估

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
sample request，以及完整 sampling artifacts。结果冻结 checkpoint SHA-256、
epoch/global step、raw/EMA、data identity、extension provenance、metric protocol、
依赖版本和所有 artifact hashes。

评估开始时会把 checkpoint 的同一份已认证字节复制到 final output 的同文件系统私有
staging；checkpoint resolve、progress、内存权重和采样都只使用这份 snapshot。采样、
指标、result、digest 与 manifest 全部完成后，staging 才以 atomic no-replace rename
发布。失败或并发目标冲突不会留下半成品正式目录，也不会覆盖已有结果。

所有 data、checkpoint、benchmark 和普通 run artifacts 都位于被忽略的 `data/` 或
`outputs/`，不得提交到 Git。

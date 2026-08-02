# AFHQ-v2 类条件生成

`examples/showcases/afhq-v2` 是一条完整的 128×128 pixel-space generation
纵向切片。它把经过固定来源身份验证的 AFHQ-v2 数据准备、class-aware loading、
ADM-UNet/DiT、混合精度训练、validation/test、checkpoint resume、
classifier-free guidance、正式 KID/FID evaluation 和可离线重放 prediction artifact 串在
同一组公开生命周期中。

example 本身是一个可安装 extension。它注册
`AFHQV2ImageDataSource`（`afhq-v2.official`）、正式 AFHQ `EvaluationBuilder` 与
class-aware distribution Metric。DataSource 发布带 identity、类别映射和标签 inventory
的 128px class-labeled official train/test artifact。production 配置直接使用 core
`class_labeled_image` Builder；partition、Dataset、Sampler、collate 和 DataLoader 仍由
该 Builder 负责。example 复用 core 的 source envelope、image recipe、loader config、
strict-resume artifact binding 和 public Evaluation runtime，不定义平行框架。

模型、Gaussian Process、TrainingBuilder、SamplingBuilder、DDIM、Trainer、EMA、
checkpoint 和 writers 都是内置能力。核心 runner 不按 AFHQ 名称分支。

## 前提与许可证

AFHQ-v2 的权威入口由
[StarGAN v2](https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq)
维护，许可证为
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)。
使用前请确认用途符合非商业条款。

本 example 的 source lock 固定 6,955,288,636-byte 官方 archive 及项目本地审计的
SHA-256：

```text
6f2540f22c6d8ebb8879a2bc0227666dd4fc765cc355cb073b63a835d679e4e3
```

上游没有发布 checksum；该摘要不是上游声明。source 失败时不会静默回退到其他镜像。
完整 archive 包含 15,803 张 512×512 RGB PNG，class mapping 为
`cat: 0`、`dog: 1`、`wild: 2`。

## 安装 example

这个 showcase 是独立的 uv project，并通过本地 editable source 使用当前 checkout 的
Stochaflow。在仓库根目录运行：

```bash
uv sync --project examples/showcases/afhq-v2 --locked
```

同步后，Stochaflow extension entry point 是 `stochaflow-afhq-v2`，同时可使用：

```text
stochaflow-afhq-v2-prepare
stochaflow-afhq-v2-capacity
stochaflow evaluate
stochaflow-afhq-v2-evaluate  # historical comparison only
```

### 当前 checkout 的正式 Evaluation 安装契约

prepare 命令和训练配置使用同一个已注册 `AFHQV2ImageDataSource`、公开
`DataArtifactStore`、packaged source lock 与 schema-v2 artifact identity contract。
正式 KID/FID profile 需要 showcase 声明的 optional
`quality` extra。正式 P2 Evaluation 当前必须从本仓库 checkout 同步，并使用以下完整
命令：

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

## 准备 artifact

第一次准备：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow-afhq-v2-prepare \
  --cache-root ./data \
  --resolution 128 \
  --downloader auto
```

在交互式终端中，prepare CLI 会把 `full` artifact 验证进度写入 stderr；最终 JSON
摘要仍单独写入 stdout。`--progress` 可以强制显示，`--no-progress` 可以关闭，因此
重定向或自动化脚本不会混入进度文本。

已经持有官方 ZIP 时，可以避免再次下载：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow-afhq-v2-prepare \
  --cache-root ./data \
  --archive /path/to/afhq_v2.zip \
  --resolution 128
```

`--archive` 只改变字节获取方式，不绕过 lock、inventory、图片和 official split
校验。网络下载遵循 `HTTPS_PROXY`/`HTTP_PROXY`；也可用 `--downloader` 显式选择
`auto`、`curl` 或 `python`。不要把代理凭据写入 YAML。

准备流程依次执行：

1. 获取并按 byte count、SHA-256 验证官方 archive；
2. 拒绝 traversal、链接、重复或大小写冲突路径、异常压缩比和非预期文件；
3. 验证完整 split/class count，以及每张 512×512 RGB PNG；
4. 一次 Lanczos resize 到 128×128，不 crop，并使用固定 PNG 参数；
5. 把 official train/test record 写入 managed `data/_index/images.json` sidecar；
6. 由 framework 为 sidecar 与全部 PNG 生成分片 stored-file inventory、写入统一
   `manifest.json`，完整复核后原子发布。

DataSource artifact 保留上游 official split：

| split | cat | dog | wild | total |
| --- | ---: | ---: | ---: | ---: |
| train | 5,065 | 4,678 | 4,593 | 14,336 |
| test | 493 | 491 | 483 | 1,467 |

DataBuilder 随后根据训练配置，从 official train 的每类按固定 sample identity 排名保留
300 张作为运行时 validation。loader 使用的 train/validation/test 数量才是
13,436/900/1,467。partition policy 不进入 source artifact identity，而是由 resolved
training config 和 seed 固定。

未显式传入时 prepare CLI 默认使用 `./.stochaflow-cache`；本教程命令和 checked-in
训练配置显式使用 `./data`。统一 v2 cache 的相关结构为：

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

raw archive cache 只服务于可恢复 acquisition，不是第二套 DataArtifact lifecycle，也不进入
artifact identity。archive override、downloader、proxy 和 credentials 同样不进入
identity 或 manifest。

## 切换到只读离线模式

准备完成后，显式验证 production 将使用的 artifact：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow-afhq-v2-prepare \
  --cache-root ./data \
  --resolution 128 \
  --policy require \
  --verification full \
  --artifact-verification-workers 8
```

`require` 不创建目录、lock 或 locator，不下载、不重建、不隔离损坏项，因此所有
checked-in 训练 YAML 都不会在启动时改变 artifact。ADM production 为避免每次 fresh
training 启动都重新 hash 全部 PNG，固定使用 `policy: require` 和
`verification: manifest`；DiT production 与 smoke 使用 `require/full`。manifest
contract 不匹配仍会在 run directory、model 和 optimizer 创建前失败；strict resume
与正式 evaluation 注入 expected identity，并无条件强制 full verification。cache hit
不进入 acquisition callback，因此即使 raw ZIP 被删除或不可访问，仍可离线加载已发布
artifact。

统一 `manifest.json` 是 framework envelope 加 AFHQ producer-defined cheap contract：
domain 只保存 resolution、class mapping、official partition roots/counts 与 canonical
sidecar descriptor。`manifest` 验证该 envelope 和 sidecar；`full` 还认证 represented
content 的全部 stored-file inventory。strict resume 注入 expected identity，并无条件
强制 `full`。

prepared artifact 已经是精确 128×128，只公开 official train/test，并认证每个样本的
class label。内置 `class_labeled_image` Builder 通过 core `ImageSourceConfig`、
`ImageSourceFactory`、`ImageRecipeConfig`、`LoaderRecipeConfig`、verified
`ClassLabeledImageDataset` 和 loader helpers 完成通用解析与组装。它在 Dataset 构建前执行
逐类 validation 划分，并提供确定性 shuffle/horizontal flip；split、shuffle 和增强都由
显式 seed 与 sample identity 决定。persistent workers 不拥有推进中的增强随机状态，
因此 epoch-boundary resume 可以重建同一 sample order 和 batch Tensor。

artifact 的 source、materialization recipe、prepared content 与 manifest identity 会写入
`run_manifest.yaml` 和 checkpoint metadata。strict resume 在构建 Dataset 前注入并比对
expected binding。

这是 breaking cache/checkpoint contract。旧 `prepared/afhq-v2` layout、
`dataset_manifest.yaml`、`files.sha256` 和 schema-v1 artifact binding 不会被发现或升级；
需要重新 materialize，并开始新的 run。capacity report 仍是 showcase 私有工具；正式
evaluation 已使用 public operation/result lifecycle，AFHQ Builder/Metric/profile 的任务语义
由 extension 拥有。framework 尚未提供通用 provenance 或 capacity model。

## 在目标主机检查容量

先准备并 full-verify AFHQ-v2 artifact，再在目标 CUDA 主机运行真实训练 profiler：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow-afhq-v2-capacity \
  --config examples/showcases/afhq-v2/experiments/profiling/train-adm-128-p2.yaml \
  --micro-batches 1 4 6 8 \
  --precisions bf16-mixed \
  --warmup-updates 5 \
  --measured-updates 25 \
  --output outputs/benchmarks/afhq-v2/adm-128-capacity.json
```

该命令按正常注册和构建路径使用 DataBuilder、TrainingBuilder、Trainer、optimizer、
scheduler、EMA 和 precision runtime。DataBuilder 通过 core factory 解析注册的
DataSource，按所选 P2 profiling config 执行 manifest verification，再进行运行时分层
划分并组装 loaders；前置的 prepare 命令已独立完成 full verification。capacity 工具
不维护第二套 source、partition 或训练循环。本次 micro batch 1/4/6/8 的 accumulation
分别为 32/8/5/4，对应 effective batch 32/32/30/32。每个 BF16 trial warmup 5 次，并
测量 25 次成功 optimizer updates。

JSON 报告包含 images/s、updates/s、allocated/reserved peak VRAM、data-wait/compute
时间及比值、forward/backward/optimizer 时间、非有限 loss/gradient 与运行环境身份；同时
请求同一 micro batch 的 FP32 和 BF16 时还会计算吞吐比和显存差值。默认要求 CUDA；为了防止
`device: auto` 在无 CUDA 时静默运行 production 模型，仅有界测试或调试可显式使用
`--device cpu`。CPU profile 的 VRAM 字段为 `null`。

CUDA phase timing 使用异步 Events，只有整段 measurement 开始和结束执行同步，不会在
每个 forward/backward/optimizer 边界阻塞。报告保存同一 DataBuilder 返回的 source
artifact bindings；partition policy 则由每个 trial 的完整 resolved config、canonical
SHA-256 和 seed 冻结。报告还包含 output directory、core/extension 版本和 Python source
tree digest。device index 与全部 precision support 会先于 meta model、DataBuilder 和
trial output preflight；若全部 precision 都不受支持，命令只返回 unsupported trial
records，不访问数据。

当前有效 schema-v3 report 来自 RTX 4090 24,564 MiB、PyTorch 2.11 / cu128，使用 corrected
105,197,187-parameter topology、P2 training 与 BF16 mixed precision。结果如下：

| Micro batch | Accumulation | Effective batch | Images/s | Peak allocated (GiB) | Peak reserved (GiB) | Non-finite |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 32 | 12.945 | 2.980 | 3.043 | 0 |
| 4 | 8 | 32 | 47.057 | 5.251 | 5.436 | 0 |
| 6 | 5 | 30 | 54.543 | 6.761 | 6.980 | 0 |
| 8 | 4 | 32 | 60.068 | 8.260 | 8.506 | 0 |

四档都完成 5 次 warmup 与 25 次 measured optimizer updates，且没有 non-finite loss 或
gradient observation。这个 schema-v3 report 提供 operational capacity 和 sustained
evidence；micro batch 8 只是已测候选中吞吐最高的一档，不是显存上限。两份 maintained ADM
production YAML 均据此使用 micro batch 8 / accumulation 4。该报告不证明长训练稳定性、
收敛、质量或 standard/P2 A/B 收益；DGX Spark 复跑会是单独的跨设备证据。

## 运行真实 smoke

smoke 使用真实 prepared AFHQ-v2，而不是 synthetic fixture。它只缩小模型和计算预算：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/smoke/train-adm-128.yaml \
  --limit-batches 4 \
  --limit-validation-batches 2 \
  --limit-test-batches 2
```

该配置使用 CPU/FP32、tiny ADM、8-step diffusion、micro batch 2、accumulation 2 和
2-step DDIM。一次运行仍会：

1. full-verify AFHQ artifact；
2. 读取带真实 class labels 的 training batches；
3. 执行两次 optimizer update；
4. 执行 limited validation；
5. 保存 best/latest/epoch checkpoint；
6. 恢复 best checkpoint 并执行 limited official test；
7. 从 selected best checkpoint 运行每类一张的 CFG final inference，并写 conditional
   diagnostic artifacts。

smoke 只证明 end-to-end wiring、state 和 artifact contract，不证明收敛质量或
production 吞吐。

### P2 tiny wiring smoke

P2 使用单独的真实 AFHQ maintained profile：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/smoke/train-adm-128-p2.yaml \
  --device cuda \
  --limit-validation-batches 2 \
  --limit-test-batches 2
```

该配置选择 `class_conditional_p2_gaussian_denoising`，固定 epsilon prediction、fixed
variance、`k: 1`、`gamma: 1`，并用 tiny ADM、4 个 training micro-batches、accumulation
2 产生 2 次 optimizer update。当前工作站的 CUDA 实跑已覆盖真实 class labels、optimizer、
EMA、diagnostics、limited validation/test 和 checkpoint publication。它是 task wiring lane，
不使用 production topology，也不证明 capacity、收敛、质量或 A/B 收益。

### P2 full-topology profiling sanity

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/profiling/train-adm-128-p2.yaml \
  --device cuda \
  --limit-validation-batches 2 \
  --limit-test-batches 2
```

这份 profile 保留 corrected 105,197,187-parameter ADM topology、128×128 real data、
cosine 1,000-step Process 与 BF16 mixed precision，只运行 micro batch 1 / accumulation 1
的 8 个 optimizer updates 和短 scheduler，并关闭 diagnostics。当前工作站实跑完成 8 次
update、validation/test lifecycle 与 checkpoint publication，compute phase 为 4.34
optimizer steps/s。

它不是 production effective-batch-32 recipe，也不是 capacity tool。这个运行本身不证明
peak VRAM、micro batch 上限、end-to-end throughput、长训练稳定性、质量或 standard/P2
A/B。production batch 与 peak-memory 数值必须引用上面的独立 schema-v3 sweep，不能从
4.34 compute steps/s 推算。

## Production 配置与更新公式

ADM-UNet v-prediction baseline 配置：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/production/train-adm-128.yaml
```

当前分支合并后的 P2 production candidate 使用独立 maintained profile：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/production/train-adm-128-p2.yaml \
  --device cuda \
  --deterministic
```

它保留相同 corrected ADM、真实 AFHQ、BF16、micro batch 8 / accumulation 4 与
84,000-update budget，但固定 epsilon/fixed P2 Builder、`k: 1`、`gamma: 1` 和 EMA decay
0.9999。它是本轮单臂 absolute-quality candidate，不是 matched standard control，也不能与
v-prediction baseline 直接比较后声称“P2 优于 standard”。该因果声明若未来需要，必须从
同一 production profile 只把 `gamma` 改为 0 后另跑 control。

DiT-B/8 候选配置（不属于当前 P2 closeout）：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/production/train-dit-128.yaml
```

ADM 使用 canonical input/output block graph：initial convolution、每个 encoder
ResBlock 和每个 residual downsample 都保存 skip；每个 decoder resolution 使用
`num_res_blocks + 1` 个 ResBlock，并逐 block 消费 skip。production 的
`num_res_blocks: 2` 因而表示 encoder 每级 2 blocks、decoder 每级 3 blocks。
`attention_resolutions: [32, 16, 8]` 是实际空间尺寸，在每个对应 ResBlock 后使用
GroupNorm/QKV/residual attention；middle 始终是
`ResBlock → Attention → ResBlock`。模型从 128×128 到达 8×8 后再还原。
该 checked-in class-conditional configuration 的 exact parameter count 是
105,197,187。

DiT-B/8 使用 8×8 patches、768 hidden size、12 blocks 和 12 heads。v-prediction ADM 与
DiT 候选实现同一
`ClassConditionalDenoiser` capability，并共享：

- 128×128 data 与固定 class mapping；
- cosine 1,000-step discrete Gaussian Process；
- `v` prediction、MSE mean 与 condition dropout 0.1；
- effective batch 32；ADM 使用实测 micro batch 8、accumulation 4，DiT 使用
  micro batch 32、accumulation 1；
- BF16 mixed precision、AdamW、EMA 和 step scheduler；
- validation/test 与 class-balanced diagnostic/sampling protocol。

这是 breaking ADM cutover。旧 `transformer_depths`、`middle_transformer_depth` 等
配置字段已删除；旧 stage-level skip/Spatial Transformer checkpoint 的 raw、EMA 与
optimizer state 均不能 resume、sample、partial load 或转换。必须 fresh train。
corrected ADM 尚无已发布的长训练质量结果。

生产配置不启用 early stopping。ADM 的 micro-batches 和 optimizer updates 为：

```text
floor(13,436 / 8) = 1,679 micro-batches
ceil(1,679 / 4) = 420 optimizer updates
200 × 420 = 84,000 total updates
round(0.02 × 84,000) = 1,680 warmup updates
```

micro batch 8 / accumulation 4 来自上面的 RTX 4090 schema-v3 P2 capacity report，在
保持 effective batch 与 optimizer-update schedule 不变时采用本次已测候选中的最高吞吐档。
它不声称 micro batch 8 是绝对容量上限，也不提供长训练质量结论。迁移到其他硬件时应重跑
同一 capacity protocol，并记录完整 hardware adaptation。

DiT 的对应计划为：

```text
floor(13,436 / 32) = 419 micro-batches
ceil(419 / 1) = 419 optimizer updates
200 × 419 = 83,800 total updates
round(0.02 × 83,800) = 1,676 warmup updates
```

Trainer 对每个 accumulation window 的 scalar micro-batch losses 等权平均。最后的 partial
window 按实际 micro-batch 数归一化并 flush。只有 optimizer step 成功时才推进 global
step、step scheduler、EMA、diagnostic 和 update-level logging；FP16 overflow 会跳过整个
window 的这些生命周期。

P2 production profile 固定 `device: cuda`，启动命令也显式传入 `--device cuda`，因此缺少
CUDA 时会在创建 run 前失败，不会静默落入巨型 CPU 作业。`bf16-mixed` 还要求
`torch.cuda.is_bf16_supported()`，不支持时不会自动退回 FP32。需要 FP16 的 CUDA 主机必须先
形成一份新的、经过容量验证的 hardware adaptation，而不能在当前正式运行中改写 precision。

训练流程每 epoch 执行 phase validation，并聚合 `valid/loss`、
`valid/metrics/prediction_mae` 与 `valid/metrics/clean_reconstruction_mse`。训练器维护的
`best.pt` 和 fit 后 phase test 属于训练 lifecycle/监控；这些指标衡量 Gaussian 预测目标或
局部重建误差，不是 FID/KID 生成分布质量，不能用于 P2 production subject selection。

P2 长训的选择与验收规则已经冻结在机器可读的
[`p2-production-closeout-policy.yaml`](../../examples/showcases/afhq-v2/experiments/evaluation/p2-production-closeout-policy.yaml)。
eligible epochs 固定为 20、40、60、80、100、120、140、160、180、200；primary 是
validation `eval/metrics/distribution/aggregate.fid`（lower），tie-break 依次是
`eval/metrics/distribution/aggregate.kid_mean`（lower）和最早 epoch。长训完成后，对每个候选 EMA
分别运行 validation-only profile；把 `REPLACE_WITH_RUN_ID` 与 zero-padded
`epoch_XXXX.pt` 改成真实值，并为每次运行使用新的 output directory：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow evaluate \
  --config examples/showcases/afhq-v2/experiments/evaluation/selection-ddim50-cfg2-validation-epsilon.yaml \
  --device cuda \
  --output-dir outputs/afhq-v2/evaluations/p2-selection/<epoch-id>
```

该 profile 以 `purpose: selection_candidate` 固定 validation split、cat/dog/wild 各 300 个
real/fake、EMA、epsilon/fixed、DDIM-50、eta 0、CFG 2.0 和 seed `20260726`。只按预声明规则
从这些 validation Evaluation results 冻结一个唯一 subject，并保存选择记录。不得使用
`valid/loss`、diagnostic、phase-test 或 official-test 结果挑 checkpoint；official test 在
subject 冻结后只运行一次。这里的 900 张 validation real/fake 只有排序权，没有 pass/fail
或 acceptance 权限。

## 日志、diagnostic 与 checkpoint

在另一终端查看 TensorBoard：

```bash
uv run --project examples/showcases/afhq-v2 tensorboard --logdir outputs
```

production diagnostic `class_conditional_diffusion_quality`：

- reconstruction 使用当前真实 batch 的原始 labels；
- sampler 以 cat/dog/wild 各 4 张的固定顺序运行；
- 使用固定 seed、EMA、CFG 2.0 和 DDIM-50；
- 每 5 epochs 写 sample grid、reconstruction panel、trajectory 和 manifest；
- 记录 timestep bucket loss、noise alignment、sample statistics 与 sampling timing。

这些结果是训练监控，不是正式 post-training dataset metric。`reference.enabled: false`
明确禁用不具备 class-aware protocol 的 reference metric。

checkpoint v12 保存完整 managed training state、precision/scaler topology、resolved
config、data binding、epoch-boundary RNG，以及 TrainingBuilder 固化的
`class_conditional_denoising` inference recipe。它把 `v` prediction 固定在 contract
中，并显式冻结 `variance: {mode: fixed}`；独立 sample config 不能覆盖。v11 及更早
checkpoint 不会自动补写或迁移；旧 ADM checkpoint 还另有 state/topology
不兼容。production 每 5 epochs 写
`epoch_*.pt`，每 epoch 更新 `latest.pt`，并维护 `best.pt`。

strict resume：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --resume outputs/afhq-v2/adm-128-p2/<run-id> \
  --epochs 200 \
  --artifact-verification-workers 8 \
  --device cuda \
  --deterministic
```

恢复以 checkpoint config 为权威；model、optimizer、precision、accumulation 和 data
identity 不能由新 config 替换。`--artifact-verification-workers` 只覆盖本次完整验证的
线程数，不改变 checkpoint config 或 artifact identity。新运行写入 sibling timestamp
directory，不续写旧日志。

## Classifier-free guidance sampling

训练后可从显式指定的 checkpoint 生成 class-balanced DDIM-50 展示结果；该展示路径不决定
formal Evaluation 的 subject：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow sample \
  --checkpoint outputs/afhq-v2/adm-128/<run-id> \
  --config examples/showcases/afhq-v2/experiments/sampling/ddim50-cfg2.yaml \
  --output-dir outputs/afhq-v2/samples/adm-ddim50-cfg2
```

这份 profile 是完整、独立的 `sample:` invocation，显式冻结 sampler、options、shape、
count、batch、seed、class allocation、weights、trajectory 和 writers；它不从 checkpoint
config 继承采样默认值。checkpoint 只提供训练时冻结的 inference recipe 与模型、Process
等推理资产。训练命令不会自动执行这次采样；必须另行运行 `stochaflow sample`，并显式
提供必填的 `--config` 来选择完整 profile。recipe name 与 `v` prediction contract
保持不变。
CFG 公式为：

```text
prediction = unconditional
           + scale × (conditional - unconditional)
```

scale 0 只运行 null branch，scale 1 只运行 conditional branch；其他非负 scale 将
conditional/null batch 拼接后单次 forward。DDIM 只消费组装好的 Gaussian dynamics，
不认识 class label 或 guidance。

production 目前是 fixed variance，模型输出 `C` channels。若另一个 class-conditional
recipe 使用 learned-range `2C` output，则 CFG 只外推 prediction half：

```text
guided_prediction = unconditional_prediction
                  + scale * (conditional_prediction - unconditional_prediction)
guided_output = [guided_prediction, conditional_variance]
```

scale 0/1 仍返回完整 unconditional/conditional branch；只有其他 scale 使用
conditional variance half。DDPM 消费 learned variance，DDIM 明确忽略 variance half。

## Corrected ADM 结果状态

canonical topology 切换前的 900-real/900-generated DDIM-50 AFHQ 数值与样本来自旧的、
checkpoint-incompatible ADM graph。它们已经从 current result surface 移除，不能用于
证明 corrected ADM、learned variance 或 P2。在 corrected production 200-epoch long run
完成并冻结新的 checkpoint、resolved config 与 evaluation artifacts 前，本页不发布
corrected ADM 的 production long-run quality baseline。下文已经发布的单 epoch 受控 A/B
数值只属于 pipeline/protocol readiness evidence，不能替代该 baseline。

AFHQ maintained pixel-image Evaluation 已迁移到 public `EvaluationBuilder`/Metric/profile
路径。该完成状态
表示正式执行与 artifact contract 已具备，不表示 corrected ADM/P2 的长训练质量数值已经
产生。

## Public full-official-test KID/FID Evaluation

训练期 loss、sample statistics 和 diagnostic artifacts 只用于监控。任何 production-quality
正式运行都必须先
按与其 recipe 匹配的预声明 validation policy 冻结唯一 checkpoint subject，再在 profile 中
只替换 `subject.path`；不得用 official test 参与选择。v-prediction/fixed-variance subject
使用下面的 public profile：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow evaluate \
  --config examples/showcases/afhq-v2/experiments/evaluation/formal-ddim50-cfg2-official-test.yaml \
  --device cuda \
  --output-dir outputs/afhq-v2/evaluations/adm-ddim50-cfg2-official-test
```

当前 P2 production candidate 先完成上文 300/class validation selection，再把
`REPLACE_WITH_RUN_ID` 与 `REPLACE_WITH_SELECTED_EPOCH_CHECKPOINT.pt` 替换为选中的唯一
zero-padded epoch EMA（例如 `epoch_0120.pt`），然后只执行一次：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow evaluate \
  --config examples/showcases/afhq-v2/experiments/evaluation/formal-ddim50-cfg2-official-test-epsilon.yaml \
  --device cuda \
  --output-dir outputs/afhq-v2/evaluations/p2-production-official-test
```

该 one-shot official-test result 用于 P2 candidate 的 absolute-quality acceptance，并要求
exact/complete 1,467 examples、aggregate FID ≤ 35、aggregate KID mean ≤ 0.01，且
cat/dog/wild 各自 FID ≤ 65；aggregate FID ≤ 30 是 aspirational target，不是 pass 所必需。
所有 required thresholds 必须同时通过。结果不能反向改变候选列表、selection metric 或
tie-break，也不能单独支持 P2 相对 standard 的 superiority 声明。这组阈值是
`internal_project_acceptance`，只适用于 `train-adm-128-p2.yaml` 产生的 epsilon/fixed P2
subject；standard-v ADM 与 DiT 配置不继承它们。

formal profile 固定：

- authenticated official test split 的全部 1,467 张 reference：cat/dog/wild 分别为
  493/491/483；
- EMA、同样的 493/491/483 generated allocation、seed `20260726`、deterministic
  DDIM-50、CFG 2.0；
- aggregate 与 per-class KID/FID；aggregate 是主要分布指标，per-class 结果仅作细分
  诊断；
- KID 100 subsets、subset size 300、seed `20260726`，FID feature 2048。

因此这里评估的是“AFHQ-v2 官方 test split + 本项目自定义 class-conditional 128×128 /
DDIM-50 / CFG 2.0 diffusion protocol”，不是论文复现。P2
[作者仓库](https://github.com/jychoi118/P2-weighting)给出的训练/预训练设置与
[CVPR 2022 论文](https://openaccess.thecvf.com/content/CVPR2022/html/Choi_Perception_Prioritized_Training_of_Diffusion_Models_CVPR_2022_paper.html)
中的 AFHQ-Dog-256 benchmark 均为 unconditional 256×256；论文的 P2 FID 11.55 不能与
这里的数值横比，因为数据子集、分辨率、条件方式、real/fake sample plan 和采样协议不同。

core 先把 checkpoint 中明确选择的 EMA（或另一个 profile 的 raw）解析为一个 pinned
primary model；AFHQ Builder 没有第二次权重选择权。它消费 checkpoint-bound
`EvaluationSamplingCapability`，通过与 `stochaflow sample` 共用的 SamplingBuilder
execution seam 完成 writer-free 生成。AFHQ evaluator 解释 real/fake/class payload，
`REGISTRIES.metrics` 中的 FID/KID providers 负责统计；core 不按 AFHQ、sampler 或 metric
名称分支。

live run 原子发布 `resolved_evaluation.yaml`、`result.json`、
`evaluation_manifest.yaml` 与 `predictions/`。prediction manifest 冻结 exact sample plan、
checkpoint/EMA、data/split、sampling profile、pre/postprocess、extension lineage、shard
digests 与 deterministic gallery IDs。

增加或复核 metrics 时，复制 formal profile 并只把 authority 改为：

```yaml
subject:
  kind: prediction_artifact
  path: ../../../../../outputs/afhq-v2/evaluations/adm-ddim50-cfg2-official-test/predictions/prediction_manifest.json
data:
  source: prediction_artifact
  split: test
```

再次执行 `stochaflow evaluate` 会认证并重放 paired records，不加载 checkpoint、不构造
model 或原 DataBuilder、不再次采样，也不修改 live producer。旧
`stochaflow-afhq-v2-evaluate` 与旧
`experiments/evaluation/ddim50-cfg2-kid-fid.yaml` 只作为历史结果对照；它们不属于当前
maintained P2 evidence surface，也不提供 compatibility guarantee。

### Epsilon/fixed control–P2 A/B

历史 P2 对比不使用上面的 v-prediction profile，而使用与当前 production official profile
相同的 epsilon/fixed 协议字段。该 profile 现在以 production run/selected-epoch 占位符
fail closed；历史 control/P2 运行分别把 subject path 替换为各自预算终点的 `latest.pt`。
协议为两个 arm 固定 epsilon/fixed recipe、full official test 493/491/483、EMA、seed
`20260726`、deterministic DDIM-50、eta 0、CFG 2.0、evaluation batch 30 与相同 KID/FID
provider 参数。每个 arm 只替换 `subject.path` 与新的 output directory：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow evaluate \
  --config examples/showcases/afhq-v2/experiments/evaluation/formal-ddim50-cfg2-official-test-epsilon.yaml \
  --device cuda \
  --output-dir outputs/afhq-v2/evaluations/<control-or-p2>-epsilon-latest
```

已完成的受控训练在两个 arm 都使用 corrected 105,197,187-parameter ADM、真实 AFHQ、同一
seed、deterministic runtime、BF16、micro batch 8 / accumulation 4、完整 1 epoch
（1,679 micro-batches / 420 optimizer updates）和从 step 0 开始、decay 0.999 的 EMA。
control 使用 P2 Builder `gamma: 0` 作为 strict-standard epsilon control；treatment 唯一
算法变化是 `gamma: 1`。训练耗时分别为 4m28s 与 4m23s。

两个 arm 在同一预算终点冻结各自 `latest.pt` 的 EMA，不使用 `best.pt`：control loss 与
P2-weighted validation loss 不是共同 selection objective，分别选 best 会把
checkpoint-selection policy 与训练目标一起改变。

正式 protocol ID 为 `afhq-v2-adm-epsilon-ddim50-cfg2-official-test-v1`。两臂均完整发布
1,467 个 unique IDs，使用相同 exact sample plan（`sample_ids_sha256` 为
`b66fc...d6c1`）；control/P2 checkpoint SHA 分别为 `6dd0...2196` 与 `b02b...fa4a`。
lower-is-better 结果如下：

| Scope | Control FID | P2 FID | FID delta | Control KID mean | P2 KID mean | KID delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aggregate | 369.621427 | 371.250343 | +1.628916 | 0.476357937 | 0.479742199 | +0.003384262 |
| Cat | 381.980901 | 383.273453 | +1.292552 | 0.551076353 | 0.553966224 | +0.002889871 |
| Dog | 382.850132 | 385.106413 | +2.256281 | 0.484312266 | 0.488923877 | +0.004611611 |
| Wild | 370.417725 | 371.661225 | +1.243500 | 0.502315342 | 0.504731715 | +0.002416373 |

P2 在 aggregate 和每类上都略差，所以单 epoch 对比没有显示收益。KID delta 与 reported
standard deviation 同量级；单 seed、单 epoch 不能称为统计显著，也不能作为 200-epoch
promotion evidence 或一般质量结论。受控 pipeline 与 protocol readiness 已关闭，production
long-run gate 仍然开放。

## 结果与追溯

训练 run 的稳定布局包括：

```text
outputs/afhq-v2/adm-128/<run-id>/
├── resolved_config.yaml
├── run_manifest.yaml
├── train.log
├── metrics.jsonl
├── tensorboard/
├── checkpoints/{best.pt,latest.pt,epoch_*.pt}
└── diagnostics/class_conditional_diffusion_quality/epoch_*/
```

独立 `stochaflow sample` 命令指定的输出目录包含：

```text
samples.pt
samples.png
trajectory.pt
trajectory.png
trajectory.gif
resolved_sampling.yaml
```

`run_manifest.yaml` 冻结 resolved training config、extension provenance、selected
components 和 data identity；checkpoint 保存相同 lineage；`resolved_sampling.yaml`
再记录 checkpoint、seed、conditions、guidance、weights、model evaluation counts 和
artifact 路径。不要只复制 PNG 而丢弃其 manifest。

正式 evaluation 目录的 `result.json` 与 `evaluation_manifest.yaml` 固定同一 checkpoint、
weights、data、protocol、metrics 和 prediction artifact digests；离线重放另发一个新的
immutable result bundle，不覆盖 producer。

`data/`、`outputs/`、checkpoint、capacity/evaluation JSON 和普通 run artifacts 都是
本地结果，不进入版本控制。

## 扩展边界

这条 showcase 特意不增加 DataBuilder、通用 condition schema、Dataset/Sampler
registry 或 AFHQ-specific config hierarchy：

- `AFHQV2ImageDataSource` 负责获取、处理、验证并发布 source-locked official
  train/test `ClassLabeledImageFolderArtifactPayload`，并由 prepare CLI 与内置
  Builder 通过同一 source contract 调用；
- 内置 `class_labeled_image` Builder 消费标准 payload，负责逐类 validation 划分、
  Dataset、Sampler、collate、DataLoader、deterministic augmentation 与
  `class_label` batch 语义；
- class-conditional TrainingStrategy 解释 `class_label`、执行 dropout 和计算 loss；
- ADM/DiT 只实现 class-conditioned denoiser forward；
- SamplingBuilder 分配 labels 并组装 CFG；
- Gaussian Process 和 DDPM/DDIM 保持 model-free、condition-free；
- public Evaluation runtime 只解析 subject/data、管理 pinned raw/EMA、完整性、MetricEngine
  与原子发布，并向任务 Builder 注入窄 sampling capability；
- AFHQ `EvaluationBuilder` 固定 official-test allocation、解释 class-aware records、声明
  replayable sink，并通过共享 SamplingBuilder execution seam 生成；
- FID/KID 通过 `REGISTRIES.metrics` 构造，AFHQ Metric 只拥有 aggregate/per-class binding
  与每类 reference/generated completeness；
- Trainer 只管理自动优化、precision、accumulation、EMA、checkpoint 和 diagnostic
  cadence。

因此，只有来源能够发布不含 native validation 的完整 class-labeled artifact，并且
实验也需要相同的逐类 derived holdout、augmentation、sampler、loader、resume 与
`class_label` batch 语义时，才只需实现一个窄的 ImageDataSource。official validation、
sharded storage 或其他 runtime recipe 需要独立 DataBuilder，不能仅凭 payload 类型
判断兼容。兼容 `ClassConditionalDenoiser` 的新模型仍可通过注册与配置进入同一训练和
采样流程，而不需要修改 runner。

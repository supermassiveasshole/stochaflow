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
`quality` extra。正式 Evaluation 当前必须从本仓库 checkout 同步，并使用以下完整
命令：

```bash
uv sync --project examples/showcases/afhq-v2 --locked --extra quality
```

这条 source-checkout 路径通过 showcase 的 `[tool.uv.sources]` 把 Stochaflow 绑定到
同一 checkout，不会要求 AFHQ wheel 从已发布 core 解析当前尚未发布的 Evaluation API。
不要把独立安装 AFHQ wheel 当作当前正式 Evaluation 的入口。

仓库的 installed-wheel gate 会从当前 checkout 同时构建 core 与 AFHQ wheels，再以
`--no-deps --offline` 安装。它只验证 wheel 内容、隔离后的 extension entry point 与当前
core wheel 的 Evaluation 激活；不验证 AFHQ wheel 的 released-core resolver，也不证明
现有 GitHub Release core 包含这些尚未发布的 API。

协调 core/AFHQ 0.2 release，以及发布后在全新环境中只安装 AFHQ wheel、实际解析已发布
core 的 resolver smoke，是 post-release、non-blocking 的 follow-up。
它们不是当前 source-checkout Evaluation 的 merge blocker；真实 core release 存在前
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
  --config examples/showcases/afhq-v2/experiments/production/train-adm-128-learned-range-v.yaml \
  --micro-batches 8 \
  --precisions bf16-mixed \
  --warmup-updates 5 \
  --measured-updates 25 \
  --output outputs/benchmarks/afhq-v2/adm-128-capacity.json
```

该命令按正常注册和构建路径使用 DataBuilder、TrainingBuilder、Trainer、optimizer、
scheduler、EMA 和 precision runtime。DataBuilder 通过 core factory 解析注册的
DataSource，按所选 training config 执行 manifest verification，再进行运行时分层
划分并组装 loaders；前置的 prepare 命令已独立完成 full verification。capacity 工具
不维护第二套 source、partition 或训练循环。工具按目标 effective batch 解析
accumulation；上面的 production candidate 使用 micro batch 8 / accumulation 4。每个
BF16 trial warmup 5 次，并测量 25 次成功 optimizer updates。

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

默认 fixed-variance ADM 的有效 schema-v3 sweep 来自 RTX 4090 24,564 MiB、PyTorch 2.11 /
cu128。它使用 canonical graph、五层 `[1,1,2,3,4]` / 8x8 scale layout、
105,197,187 parameters 与 BF16 mixed precision：

| Micro batch | Accumulation | Effective batch | Images/s | Peak allocated (GiB) | Peak reserved (GiB) | Non-finite |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 32 | 12.945 | 2.980 | 3.043 | 0 |
| 4 | 8 | 32 | 47.057 | 5.251 | 5.436 | 0 |
| 6 | 5 | 30 | 54.543 | 6.761 | 6.980 | 0 |
| 8 | 4 | 32 | 60.068 | 8.260 | 8.506 | 0 |

四档都完成 5 次 warmup 与 25 次 measured optimizer updates，且没有 non-finite loss 或
gradient observation。这个 schema-v3 report 提供默认五层模型的 operational capacity
和 sustained evidence；micro batch 8 只是该 sweep 已测候选中吞吐最高的一档，不是显存
上限。

fresh learned-range-v quality candidate 有自己的精确 evidence。它保留 canonical graph，
但联合使用四层 `[1,2,3,4]` / 16x16 scale layout 和 `2C` learned-range output；exact
parameter count 是 100,351,366。在同一 RTX 4090 / PyTorch 2.11 / CUDA 12.8 / BF16
环境，micro batch 8 / accumulation 4 完成 25 次 measured optimizer updates，吞吐为
45.17 images/s，peak reserved memory 为 10.455 GiB，non-finite loss/gradient
observation 为 0。

两份 production YAML 都使用 8 / 4 并保持 effective batch 32，但各自由自己的容量测量
支持；不能把默认模型的吞吐或显存数字外推给 candidate。两组报告都不证明长训练稳定性、
收敛或质量；DGX Spark 复跑会是单独的跨设备证据。

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

## Production 配置与更新公式

ADM-UNet v-prediction baseline 配置：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/production/train-adm-128.yaml
```

当前 production-quality candidate 是 fresh learned-range-v ADM：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/production/train-adm-128-learned-range-v.yaml \
  --device cuda
```

它保留 canonical ADM input/output-block graph、真实 AFHQ、cosine Process、v prediction、
BF16、micro batch 8 / accumulation 4、optimizer、step scheduler、seed、EMA 与 200 epochs /
84,000 updates；同时把默认五层 `[1,1,2,3,4]` / 8x8 scale layout 改为四层
`[1,2,3,4]` / 16x16，并把 fixed variance 改为 learned range。模型输出为 `2C`，hybrid
objective 同时记录 simple loss 与 variational bound。该实验必须随机初始化；fixed-variance
或旧 topology checkpoint 都不能 resume。

这是 topology + variance 的联合 quality candidate，不是 isolated learned-variance 或
isolated topology ablation，也不是 exact epsilon-prediction IDDPM reproduction；质量变化
不能只归因于 learned variance。

DiT-B/8 候选配置：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/production/train-dit-128.yaml
```

ADM 使用 canonical input/output block graph：initial convolution、每个 encoder
ResBlock 和每个 residual downsample 都保存 skip；每个 decoder resolution 使用
`num_res_blocks + 1` 个 ResBlock，并逐 block 消费 skip。production 的
`num_res_blocks: 2` 因而表示 encoder 每级 2 blocks、decoder 每级 3 blocks。
默认 fixed-variance config 使用 `[1,1,2,3,4]`、从 128x128 到达 8x8、在
`attention_resolutions: [32, 16, 8]` 放置 attention，exact parameter count 是
105,197,187。learned-range candidate 使用同一 canonical graph，但采用 `[1,2,3,4]`、
到达 16x16、在 `[32, 16]` 放置 attention，并输出 6 channels；它的 exact parameter count
是 100,351,366。两者都在对应 ResBlock 后使用 GroupNorm/QKV/residual attention，middle
始终是 `ResBlock → Attention → ResBlock`。

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

learned-range candidate 的 micro batch 8 / accumulation 4 来自自己的 RTX 4090
schema-v3 capacity evidence：100,351,366 parameters、45.17 images/s、10.455 GiB peak
reserved memory，并在 25 次 measured optimizer updates 中记录 0 个 non-finite
observation。它不声称 micro batch 8 是绝对容量上限，也不提供长训练质量结论。迁移到其他
硬件时应重跑同一 capacity protocol，并记录完整 hardware adaptation。

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

learned-range production profile 固定 `device: cuda`，启动命令也显式传入 `--device cuda`，因此缺少
CUDA 时会在创建 run 前失败，不会静默落入巨型 CPU 作业。`bf16-mixed` 还要求
`torch.cuda.is_bf16_supported()`，不支持时不会自动退回 FP32。需要 FP16 的 CUDA 主机必须先
形成一份新的、经过容量验证的 hardware adaptation，而不能在当前正式运行中改写 precision。

训练流程每 epoch 执行 ordinary phase validation；从 epoch 100 到 epoch 200 每 10 epochs
额外运行完整生成 Evaluation。到期运行固定 EMA、DDPM-100 / CFG 2、300/class
validation real/fake、900 generated samples、sampling batch 15、exact IDs 和
aggregate/per-class FID/KID。Trainer 直接以
`valid/metrics/distribution/aggregate.fid`（lower）维护 `best.pt`；KID 与 per-class 结果
一同记录。FID/KID Metric 只消费 image-pair updates，采样与 completeness 由 Evaluation
拥有。

非到期 epoch 不复用旧 FID/KID，也不推进 patience。`valid/loss`、Diagnostic、phase test
与 official test 都不能覆盖这一 selection authority。对已有 checkpoints 做补充选择时，
逐个运行同一 standalone validation Evaluation 并比较相同 metric 即可，不需要额外的
selection runtime。

## 日志、diagnostic 与 checkpoint

在另一终端查看 TensorBoard：

```bash
uv run --project examples/showcases/afhq-v2 tensorboard --logdir outputs
```

production diagnostic `class_conditional_diffusion_quality`：

- reconstruction 使用当前真实 batch 的原始 labels；
- sampler 以 cat/dog/wild 各 4 张的固定顺序运行；
- 使用固定 seed 和 EMA，以 DDPM-100 观察 learned variance，并用 DDIM-50 观察 prediction
  continuity；
- 每 10 epochs 为两个 sampler 写 sample grid、reconstruction panel 和 manifest；
- 记录 timestep bucket loss、noise alignment、sample statistics 与 sampling timing。

这些结果是训练监控，不是 dataset metric。Diagnostic 无论读取哪个 split 都不参与
checkpoint selection；完整 validation Evaluation 才产生上面的 FID/KID observations。

checkpoint v12 保存完整 managed training state、precision/scaler topology、resolved
config、data binding、epoch-boundary RNG，以及 TrainingBuilder 固化的
`class_conditional_denoising` inference recipe。每个 checkpoint 固定自己的 prediction
contract：当前 recipe 为 `v` + `learned_range`，独立 sample config 不能覆盖。v11 及更早
checkpoint 不会自动补写或迁移；旧 ADM checkpoint 还另有 state/topology
不兼容。production 每 50 epochs 写 `epoch_*.pt`，每 epoch 更新 `latest.pt`，并由完整
validation aggregate FID 维护 `best.pt`。

strict resume：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --resume outputs/afhq-v2/adm-128-learned-range-v/<run-id> \
  --epochs 200 \
  --artifact-verification-workers 8 \
  --device cuda
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
  --checkpoint outputs/afhq-v2/adm-128-learned-range-v/<run-id> \
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

learned-range production 模型输出 `2C` channels，CFG 只外推 prediction half：

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
证明 corrected ADM 或当前 topology + variance candidate。在 learned-range-v run 完成、
validation 选中 checkpoint 并冻结新的 resolved config 与 official-test artifacts 前，本页
不发布该 recipe 的 production-quality baseline。即使完成，联合改动也不能支持 isolated
learned-variance、isolated topology 或 exact epsilon-prediction IDDPM 结论。

AFHQ maintained pixel-image Evaluation 已迁移到 public `EvaluationBuilder`/Metric/profile
路径。该完成状态表示正式执行与 artifact contract 已具备，不表示 learned-range-v 的
长训练质量数值已经产生。

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

learned-range-v 的 final-test profile 必须匹配 checkpoint 的 `v` + `learned_range`
contract，并只引用训练中 aggregate validation FID 选出的 `best.pt`。它固定：

- authenticated official test split 的全部 1,467 张 reference：cat/dog/wild 分别为
  493/491/483；
- EMA、同样的 493/491/483 generated allocation、seed `20260726`，以及 validation
  Evaluation 已冻结的 sampler 与 CFG；
- aggregate 与 per-class KID/FID；aggregate 是主要分布指标，per-class 结果仅作细分
  诊断；
- KID 100 subsets、subset size 200、seed `20260726`，FID feature 2048。

E200 完成并冻结 validation-selected checkpoint 后，只把 checked-in profile 的
`subject.path` 替换为该 `best.pt` 的精确绝对文件路径，然后执行：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow evaluate \
  --config examples/showcases/afhq-v2/experiments/evaluation/formal-ddpm100-cfg2-official-test-learned-range-v.yaml \
  --device cuda \
  --output-dir outputs/afhq-v2/evaluations/adm-learned-range-v-ddpm100-cfg2-official-test
```

这次 full official-test Evaluation 只运行一次，且不能反向改变 selection。它发布 aggregate
与 per-class FID/KID、exact sample completeness、prediction artifact 和 immutable result
bundle。standard-v/fixed-variance subject 继续使用自己冻结的 DDIM-50 / CFG 2.0 profile；
两种 recipe contract 不可互换。

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
  path: ../../../../../outputs/afhq-v2/evaluations/adm-learned-range-v-ddpm100-cfg2-official-test/predictions/prediction_manifest.json
data:
  source: prediction_artifact
  split: test
```

再次执行 `stochaflow evaluate` 会认证并重放 paired records，不加载 checkpoint、不构造
model 或原 DataBuilder、不再次采样，也不修改 live producer。旧
`stochaflow-afhq-v2-evaluate` 与旧
`experiments/evaluation/ddim50-cfg2-kid-fid.yaml` 只作为历史结果对照；它们不属于当前
maintained evidence surface，也不提供 compatibility guarantee。

## 结果与追溯

训练 run 的稳定布局包括：

```text
outputs/afhq-v2/adm-128-learned-range-v/<run-id>/
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

# AFHQ-v2 生成与 P2-compatible Dog research

`examples/showcases/afhq-v2` 是一条完整的 128×128 pixel-space generation
纵向切片。它把经过固定来源身份验证的 AFHQ-v2 数据准备、class-aware loading、
ADM-UNet/DiT、混合精度训练、validation/test、checkpoint resume、
classifier-free guidance 和结果 artifact 串在同一组公开生命周期中。

example 本身是一个可安装 extension，注册两个窄 DataSource：

- `AFHQV2ImageDataSource`（`afhq-v2.official`）发布带 identity、类别映射和标签
  inventory 的 128px class-labeled official train/test artifact；
- `AFHQV2DogImageDataSource`（`afhq-v2.dog`）只发布 authenticated train/dog subset，
  使用 pinned guided-diffusion 256px transform，并返回不含 class labels 的通用
  `ImageFolderArtifactPayload`。

类条件 production 配置直接使用 core `class_labeled_image` Builder；Dog research
配置使用 core `image` Builder。partition、Dataset、Sampler、collate 和 DataLoader
仍由 Builder 负责。example 复用 core 的 source envelope、image recipe、loader config
及 strict-resume artifact binding，不定义平行框架或 dataset-name-specific Builder。

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
stochaflow-afhq-v2-evaluate
```

prepare 命令通过 `--profile official|dog` 选择与训练配置相同的已注册 DataSource；
两条路径共用公开 `DataArtifactStore`、packaged source lock 与 schema-v2 artifact
identity contract。
正式 KID/FID 评估需要同步 showcase 声明的 optional `quality` extra：

```bash
uv sync --project examples/showcases/afhq-v2 --locked --extra quality
```

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

## 准备 P2-compatible AFHQ-v2 Dog artifact

Dog research lane 复用同一份 authenticated AFHQ-v2 archive，但只物化 official
`train/dog` 的 4,678 张图，并默认生成独立 256×256 artifact：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow-afhq-v2-prepare \
  --profile dog \
  --cache-root ./data \
  --policy ensure \
  --verification full
```

`--profile dog` 默认 resolution 256，其他 resolution 会 fail closed。已有 archive 时可
继续使用 `--archive /path/to/afhq_v2.zip`；source byte count、SHA-256、完整 ZIP
inventory 和 train/dog count 仍由同一 packaged source lock 认证。

materialization recipe 固定
OpenAI guided-diffusion commit
`8fb3ad9197f16bbc40620447b2742e13458d2831` 中的 `center_crop_arr` 语义：

1. 在最短边仍至少是目标尺寸两倍时，反复使用 Pillow BOX 把宽高各减半；
2. 以 `output_resolution / min(width, height)` 缩放，使用 bicubic；
3. 使用整数中心 offset 裁成 256×256；
4. 丢弃 metadata，以固定 PNG 编码参数写出 RGB/8-bit 文件。

artifact identity 还记录 transform、source subset、file inventory，以及 materialization
时使用的 Pillow、NumPy 和 zlib 版本。水平翻转不在 DataSource 中执行；它是 generic
`image` DataBuilder 的 seeded runtime policy，所以不会把训练 augmentation 混入 prepared
content identity。

`afhq-v2.dog` 返回普通、无标签的 `ImageFolderArtifactPayload`：只有 `train`，
`validation/test` 均为 null。目录中的 `dog/` 只是 source-relative path，不进入 batch
作为 class label。unconditional recipe 应使用：

```yaml
data:
  name: image
  params:
    source:
      name: afhq-v2.dog
      materialization:
        cache_root: ./data
        policy: require
        verification: full
      params:
        resolution: 256
    partition: {mode: none}
    image:
      size: [256, 256]
      channels: 3
      normalize: true
      random_horizontal_flip: true
```

这条路径准确表示 **P2-compatible AFHQ-v2 Dog**。P2 公开材料没有冻结历史 AFHQ
version、archive checksum、train file list、seeds、KID implementation 或 checkpoint
selection policy，因此它不能被命名为 exact P2 AFHQ-D reproduction。

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

Dog artifact 使用相同 read-only 验证路径：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow-afhq-v2-prepare \
  --profile dog \
  --cache-root ./data \
  --policy require \
  --verification full
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
需要重新 materialize，并开始新的 run。capacity report 与 evaluation result lifecycle
仍是 showcase 私有能力；framework 尚未提供通用 provenance 或 capacity model。

## 在目标主机检查容量

先准备并 full-verify AFHQ-v2 artifact，再在目标 CUDA 主机运行真实训练 profiler：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow-afhq-v2-capacity \
  --config examples/showcases/afhq-v2/experiments/production/train-adm-128.yaml \
  --micro-batches 1 2 4 8 \
  --precisions fp32 bf16-mixed \
  --warmup-updates 5 \
  --measured-updates 25 \
  --output outputs/benchmarks/afhq-v2/adm-128-capacity.json
```

该命令按正常注册和构建路径使用 DataBuilder、TrainingBuilder、Trainer、optimizer、
scheduler、EMA 和 precision runtime。DataBuilder 通过 core factory 解析注册的
DataSource，按 ADM production config 执行 manifest verification，再进行运行时分层
划分并组装 loaders；前置的 prepare 命令已独立完成 full verification。capacity 工具
不维护第二套 source、partition 或训练循环。micro batch 1/2/4/8 的 accumulation
分别为 32/16/8/4，保持 effective batch 32。每个 FP32 和 BF16 trial 默认 warmup 5 次，
并测量至少 25 次成功 optimizer updates。

JSON 报告包含 images/s、updates/s、allocated/reserved peak VRAM、data-wait/compute
时间及比值、forward/backward/optimizer 时间、非有限 loss/gradient、运行环境身份，以及
同一 micro batch 下 BF16 相对 FP32 的吞吐比和显存差值。默认要求 CUDA；为了防止
`device: auto` 在无 CUDA 时静默运行 production 模型，仅有界测试或调试可显式使用
`--device cpu`。CPU profile 的 VRAM 字段为 `null`。

CUDA phase timing 使用异步 Events，只有整段 measurement 开始和结束执行同步，不会在
每个 forward/backward/optimizer 边界阻塞。报告保存同一 DataBuilder 返回的 source
artifact bindings；partition policy 则由每个 trial 的完整 resolved config、canonical
SHA-256 和 seed 冻结。报告还包含 output directory、core/extension 版本和 Python source
tree digest。device index 与全部 precision support 会先于 meta model、DataBuilder 和
trial output preflight；若全部 precision 都不受支持，命令只返回 unsupported trial
records，不访问数据。

当前仓库没有 corrected 105M ADM 的 4090 或 DGX measured capacity report。promotion
前必须在 4090 上完成 BF16 micro batch 1/2/4/8 forward/backward trials，并在 DGX
Spark 上用同一 resolved config 重复 smoke/resume/sample。checked-in micro batch 1 只是
provisional memory-safe default，不是吞吐最优值或实测显存承诺；profile 结果才能决定是否
提高 micro batch 并相应降低 accumulation。

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

ADM-UNet 主配置：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --config examples/showcases/afhq-v2/experiments/production/train-adm-128.yaml
```

DiT-B/8 对照：

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

DiT-B/8 使用 8×8 patches、768 hidden size、12 blocks 和 12 heads。二者实现同一
`ClassConditionalDenoiser` capability，并共享：

- 128×128 data 与固定 class mapping；
- cosine 1,000-step discrete Gaussian Process；
- `v` prediction、MSE mean 与 condition dropout 0.1；
- effective batch 32；ADM 暂用 micro batch 1、accumulation 32，DiT 使用
  micro batch 32、accumulation 1；
- BF16 mixed precision、AdamW、EMA 和 step scheduler；
- validation/test 与 class-balanced diagnostic/sampling protocol。

这是 breaking ADM cutover。旧 `transformer_depths`、`middle_transformer_depth` 等
配置字段已删除；旧 stage-level skip/Spatial Transformer checkpoint 的 raw、EMA 与
optimizer state 均不能 resume、sample、partial load 或转换。必须 fresh train。
corrected ADM 尚无已发布的长训练质量结果。

生产配置不启用 early stopping。ADM 的 micro-batches 和 optimizer updates 为：

```text
floor(13,436 / 1) = 13,436 micro-batches
ceil(13,436 / 32) = 420 optimizer updates
200 × 420 = 84,000 total updates
round(0.02 × 84,000) = 1,680 warmup updates
```

micro batch 1 / accumulation 32 是 corrected topology 尚未完成 target-device profile
时的 provisional memory-safe default；它不证明 micro batch 8 仍可用，也不代表最终
production throughput。若 4090/DGX profile 允许更大 micro batch，必须保持 effective
batch 与 optimizer-update schedule，并记录完整 hardware adaptation。

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

production 使用 `device: auto`。当它选中 CUDA 时，`bf16-mixed` 要求
`torch.cuda.is_bf16_supported()`；不支持时会在创建 run 前失败，不会自动退回 FP32。
CPU BF16 autocast 可用但不代表适合 production 吞吐。需要 FP16 的 CUDA 主机必须显式
选择 `fp16-mixed`；该模式只支持 CUDA 并使用 GradScaler。

训练流程每 epoch 执行 validation，以 `valid_loss` 选择 best checkpoint。fit 结束后先恢复
best，再执行 official test。Stochaflow 当前没有独立的 `validate` CLI；不能把 sampling
或 sample statistics 当作模型 validation，也不能把当前 diagnostic 冒充正式 FID/KID。

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

checkpoint v10 保存完整 managed training state、precision/scaler topology、resolved
config、data binding、epoch-boundary RNG，以及 TrainingBuilder 固化的
`class_conditional_denoising` inference recipe。它把 `v` prediction 固定在 contract
中，并显式冻结 `variance: {mode: fixed}`；独立 request 不能覆盖。变更前缺少
`variance` contract 的 v10 Gaussian checkpoint 会因 recipe equality 被 strict resume
拒绝，但 non-ADM sampling 仍保留 fixed-compatible default；不会自动补写 checkpoint。
旧 ADM sampling 另因 state/topology 不兼容而失败。production 每 5 epochs 写
`epoch_*.pt`，每 epoch 更新 `latest.pt`，并维护 `best.pt`。

strict resume：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow train \
  --resume outputs/afhq-v2/adm-128/<run-id> \
  --epochs 200 \
  --artifact-verification-workers 8
```

恢复以 checkpoint config 为权威；model、optimizer、precision、accumulation 和 data
identity 不能由新 config 替换。`--artifact-verification-workers` 只覆盖本次完整验证的
线程数，不改变 checkpoint config 或 artifact identity。新运行写入 sibling timestamp
directory，不续写旧日志。

## Classifier-free guidance sampling

训练后，从 best checkpoint 生成 class-balanced DDIM-50 结果：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow sample \
  --checkpoint outputs/afhq-v2/adm-128/<run-id> \
  --config examples/showcases/afhq-v2/experiments/sampling/ddim50-cfg2.yaml \
  --output-dir outputs/afhq-v2/samples/adm-ddim50-cfg2
```

这份 partial request 只把 checkpoint 默认 solver 原子替换为 DDIM-50，并把
`guidance_scale` 调整为 2.0。count、batch、seed、class allocation、weights、
trajectory 和 writers 都继续继承 checkpoint；request 不重复声明这些值。`options`
只做 top-level shallow merge，recipe name 与 `v` prediction contract 保持不变。
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
证明 corrected ADM、learned variance 或 P2。corrected production 运行完成并冻结新的
checkpoint、resolved config 与 evaluation artifacts 前，本页不发布 AFHQ quality 数值。

即使未来使用当前 900-sample class-aware evaluator得到新结果，它也不能和 P2
论文的 50,000-fake ancestral protocol 直接比较：sample count、data domain、
sampling algorithm 和 reference set 均不同。

## P2-compatible AFHQ-v2 Dog research protocol

research authority 与三类 production config 分开，位于：

```text
examples/showcases/afhq-v2/experiments/research/p2-afhq-v2-dog-256/
├── train-base.yaml
├── p2-loss-weighting.yaml
├── sample-ddpm-250.yaml
└── sample-ddpm-1000.yaml
```

`train-base.yaml` 是唯一完整训练 base，并冻结：

- authenticated `afhq-v2.dog` unlabeled 256px artifact；
- canonical unconditional ADM：base 128、multipliers `[1, 1, 2, 2, 4, 4]`、
  encoder 每级 1 ResBlock、16×16 attention、6-channel learned-range output、
  93,563,910 parameters；
- linear beta schedule、`T=1000`、uniform timestep sampling 与 epsilon prediction；
- `variance: {mode: learned_range, loss: rescaled_variational_bound}`；
- constant simple-loss baseline、MSE mean、AdamW `lr=2e-5`/weight decay 0；
- FP32、effective batch 8、EMA 0.9999 per update；
- 600 epochs × 500 updates = 300,000 updates = 2.4M images seen；
- training seed `20260730` 与 sampling seed `20260731`。

这里的 effective batch 8 是 formal protocol fact，不是对任一 GPU 的 microbatch
capacity 承诺。若 4090 trial 需要更小 micro batch，必须用 accumulation 保持 effective
batch 8，并在 resolved config/provenance 中记录 hardware adaptation；FP32 reference
semantics 不因硬件适配而改变。

`p2-loss-weighting.yaml` 不是第二份完整配置。restricted resolver 只允许它替换
`training.params.loss_weighting` 为 `{name: p2, k: 1.0, gamma: 1.0}`；其他路径、
缺字段或额外字段都会被拒绝。分别解析 baseline/P2：

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

resolver 写出可直接训练的完整 typed config，并在 provenance sidecar 记录 base/override
source SHA-256、唯一 changed path 与 canonical resolved-config SHA-256；已有 target
不会被覆盖。A/B 必须从相同初始化和 data order开始，唯一训练变化是 loss weighting。

P2 权重只乘 epsilon simple MSE：

```text
snr(t) = alpha_bar_t / (1 - alpha_bar_t)
weight(t) = (1 + snr(t))^-1
loss = mean(weight * per_sample_simple_loss + per_sample_variational_bound)
```

权重来自 cumulative marginal，不做 batch renormalization。learned-range hybrid 定义为
simple loss 加 `0.001 ×` 完整 VLB；uniform single-timestep estimator 将它实现为
`T / 1000 ×` sampled VB term。该 term 使用 detached mean branch、timestep 1 decoder
NLL/其余 posterior KL，P2 不对它加权。training diagnostic 的
`timestep_loss_weight` 是这里的优化系数，与只用于 metric batch 聚合的
`loss_aggregation_weight` 不同。

两个 sample profile 只改变 solver request：

- `sample-ddpm-1000.yaml`：完整 1,000-step ancestral DDPM；
- `sample-ddpm-250.yaml`：uniform-section selected-pair 250-step ancestral DDPM。

250-step profile 不是 DDIM，也不是在 adjacent posterior 中跳过 750 次 model
evaluation。两份 profile 都继承 checkpoint 固化的 epsilon/learned-range inference
contract，并使用 EMA 与 seed `20260731`；P2 `k/gamma` 不出现在 sampling config。

这些文件冻结 algorithm-compatible protocol，但没有发布 benchmark 结果。正式质量声明
还需要完整 data/reference manifest、50,000 fake completeness、固定 FID/KID
implementation、checkpoint selection policy、4090/DGX capacity evidence 和完成的两组
run。由于 P2 公开材料没有关闭历史 AFHQ version/file-list 等 gaps，最终名称仍应是
“P2-compatible AFHQ-v2 Dog”，不能声称 exact P2 AFHQ-D reproduction。

两份 checked-in sample request 继承 base 中的 50,000 samples，但目前只是冻结未来
protocol input，不能直接当作可运行的 formal benchmark。当前 Tensor writer 会在内存中
保留 batch 并拼接完整 `50000 × 3 × 256 × 256` FP32 tensor，单 samples payload 约
39 GB（还未计运行时额外开销）。正式 50k 执行必须等待 sharded prediction artifact、
resume/completeness manifest 与对应 evaluation authority；当前不要直接对这些 profile
运行一次性 `stochaflow sample` 并声称 benchmark complete。

## 正式 class-aware KID/FID 评估

训练期 loss、sample statistics 和 diagnostic artifacts 只用于监控。冻结 best
checkpoint 后运行独立评估：

```bash
uv run --project examples/showcases/afhq-v2 stochaflow-afhq-v2-evaluate \
  --checkpoint outputs/afhq-v2/adm-128/<run-id>/checkpoints/best.pt \
  --config examples/showcases/afhq-v2/experiments/evaluation/ddim50-cfg2-kid-fid.yaml \
  --output-dir outputs/afhq-v2/evaluations/adm-ddim50-cfg2
```

checked-in protocol 固定：

- official test split 中 cat/dog/wild 各 300 张真实图，按 authenticated manifest
  顺序选择；
- EMA、每类 300 张生成图、seed `20260726`、DDIM-50、CFG 2.0；
- aggregate 与 per-class KID/FID；aggregate 是主要分布指标，per-class 结果仅作细分
  诊断；
- KID 100 subsets、subset size 300，FID feature 2048。

评估命令不重新实现数据或采样循环。它以 checkpoint 中的 config 和
`DataArtifactBindings` 严格重建内置 `class_labeled_image` Builder 的 official test
loader，再通过现有 `class_conditional_denoising` SamplingBuilder 生成有序 class
blocks。quality provider、execution device 和每个 metric scope 在数据读取、输出目录
和采样前预检。

`evaluation-result.json` 冻结 checkpoint SHA-256、format/epoch/global step、raw/EMA、
data identity、extension provenance、class allocation、seed、guidance、sampler、metric
参数/实现/依赖版本、aggregate/per-class 数值和 sampling artifact hashes。旁边的
`evaluation-result.sha256` 与 `evaluation-manifest.json` 用于完整性复核；不要只复制
metric JSON 而丢弃 sampling manifest 和 artifacts。

命令先把 checkpoint 的同一份已认证字节复制到 final output 的同文件系统私有
staging；resolve、progress、内存权重与采样都使用该 snapshot。采样、metric、result、
digest 和 manifest 全部成功后才以 atomic no-replace rename 发布 final directory。
任何失败或并发目标冲突都会清理 staging，不会留下半提交结果或覆盖已有目录。

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
├── diagnostics/class_conditional_diffusion_quality/epoch_*/
└── samples/final/
```

显式 sample 输出包含：

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

`data/`、`outputs/`、checkpoint、capacity/evaluation JSON 和普通 run artifacts 都是
本地结果，不进入版本控制。

## 扩展边界

这条 showcase 特意不增加 DataBuilder、通用 condition schema、Dataset/Sampler
registry 或 AFHQ-specific config hierarchy：

- `AFHQV2ImageDataSource` 负责获取、处理、验证并发布 source-locked official
  train/test `ClassLabeledImageFolderArtifactPayload`；
- `AFHQV2DogImageDataSource` 复用 acquisition/authentication，只物化 train/dog，
  发布普通无标签 `ImageFolderArtifactPayload`；两者都由 prepare CLI 与内置 Builder
  通过同一 source contract 调用；
- 内置 `class_labeled_image` Builder 消费标准 payload，负责逐类 validation 划分、
  Dataset、Sampler、collate、DataLoader、deterministic augmentation 与
  `class_label` batch 语义；
- 内置 `image` Builder 消费 Dog payload，构造 unconditional Tensor batches；Dog source
  不构造 Dataset、partition、sampler、collate 或 DataLoader；
- class-conditional TrainingStrategy 解释 `class_label`、执行 dropout 和计算 loss；
- ADM/DiT 只实现 class-conditioned denoiser forward；
- SamplingBuilder 分配 labels 并组装 CFG；
- Gaussian Process 和 DDPM/DDIM 保持 model-free、condition-free；
- Trainer 只管理自动优化、precision、accumulation、EMA、checkpoint 和 diagnostic
  cadence。

因此，只有来源能够发布不含 native validation 的完整 class-labeled artifact，并且
实验也需要相同的逐类 derived holdout、augmentation、sampler、loader、resume 与
`class_label` batch 语义时，才只需实现一个窄的 ImageDataSource。official validation、
sharded storage 或其他 runtime recipe 需要独立 DataBuilder，不能仅凭 payload 类型
判断兼容。兼容 `ClassConditionalDenoiser` 的新模型仍可通过注册与配置进入同一训练和
采样流程，而不需要修改 runner。

# AFHQ-v2 类条件生成

`examples/showcases/afhq-v2` 是一条完整的 128×128 pixel-space generation
纵向切片。它把 source-locked AFHQ-v2 数据准备、class-aware loading、ADM-UNet/DiT、
混合精度训练、validation/test、checkpoint resume、classifier-free guidance 和结果
artifact 串在同一组公开生命周期中。

example 本身是一个可安装 extension，但只拥有 AFHQ-specific 的两部分：

- `afhq-v2.official` 负责官方 archive 到 managed artifact 的安全、确定性转换；
- `afhq-v2.class-images` 负责把 artifact 组装成
  `(images, {"class_label": labels})` loaders。

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

在仓库根目录运行：

```powershell
uv pip install -e examples/showcases/afhq-v2
```

安装后，Stochaflow extension entry point 是 `stochaflow-afhq-v2`，同时可使用：

```text
stochaflow-afhq-v2-prepare
stochaflow-afhq-v2-capacity
stochaflow-afhq-v2-evaluate
```

这些命令和训练配置使用同一份 packaged source lock 与 preparation implementation。
正式 KID/FID 评估需要先安装仓库的 optional `quality` extra：

```powershell
uv sync --extra quality
```

## 准备 artifact

第一次准备：

```powershell
uv run stochaflow-afhq-v2-prepare `
  --cache-root .\data `
  --resolution 128 `
  --validation-per-class 300
```

已经持有官方 ZIP 时，可以避免再次下载：

```powershell
uv run stochaflow-afhq-v2-prepare `
  --cache-root .\data `
  --archive D:\downloads\afhq_v2.zip `
  --resolution 128 `
  --validation-per-class 300
```

`--archive` 只改变字节获取方式，不绕过 lock、inventory、图片和 split 校验。网络下载
遵循 `HTTPS_PROXY`/`HTTP_PROXY`；不要把代理凭据写入 YAML。

准备流程依次执行：

1. 获取并按 byte count、SHA-256 验证官方 archive；
2. 拒绝 traversal、链接、重复或大小写冲突路径、异常压缩比和非预期文件；
3. 验证完整 split/class count，以及每张 512×512 RGB PNG；
4. 从 official train 的每类按固定 sample identity 排名保留 300 张 validation；
5. 一次 Lanczos resize 到 128×128，不 crop，并使用固定 PNG 参数；
6. 写入 `files.sha256` 和 `dataset_manifest.yaml`，完整复核后原子发布。

最终 split 为：

| split | cat | dog | wild | total |
| --- | ---: | ---: | ---: | ---: |
| train | 4,765 | 4,378 | 4,293 | 13,436 |
| validation | 300 | 300 | 300 | 900 |
| test | 493 | 491 | 483 | 1,467 |

缓存结构：

```text
data/
├── raw/afhq-v2/<source-sha256>/afhq_v2.zip
└── prepared/afhq-v2/128/<preparation-key>/
    ├── train/{cat,dog,wild}/
    ├── validation/{cat,dog,wild}/
    ├── test/{cat,dog,wild}/
    ├── files.sha256
    └── dataset_manifest.yaml
```

## 切换到只读离线模式

准备完成后，显式验证 production 将使用的 artifact：

```powershell
uv run stochaflow-afhq-v2-prepare `
  --cache-root .\data `
  --resolution 128 `
  --validation-per-class 300 `
  --policy require `
  --verification full
```

`require` 不下载、不重建、不隔离损坏项。production 与 smoke YAML 已固定
`policy: require` 和 `verification: full`；缺失、替换或内容漂移会在 run directory、
model 和 optimizer 创建前失败。

prepared artifact 已经是精确 128×128。AFHQ DataBuilder 不做在线 resize/crop，只做
认证读取、Tensor 转换、可选 normalization 和训练 horizontal flip。训练 shuffle 与 flip
由 `(run seed, epoch, sample identity)` 决定；persistent workers 不拥有推进中的增强
随机状态。因此 epoch-boundary resume 可以重建同一 sample order 和 batch Tensor。

artifact 的 source、materialization recipe、prepared content 与 manifest identity 会写入
`run_manifest.yaml` 和 checkpoint metadata。strict resume 在构建 Dataset 前注入并比对
expected binding。

## 在目标主机检查容量

先准备并 full-verify AFHQ-v2 artifact，再在目标 CUDA 主机运行真实训练 profiler：

```powershell
uv run stochaflow-afhq-v2-capacity `
  --config examples/showcases/afhq-v2/experiments/production/train-adm-128.yaml `
  --micro-batches 4 6 8 `
  --precisions fp32 bf16-mixed `
  --warmup-updates 5 `
  --measured-updates 25 `
  --output outputs/benchmarks/afhq-v2/adm-128-capacity.json
```

该命令按正常注册和构建路径使用 DataBuilder、TrainingBuilder、Trainer、optimizer、
scheduler、EMA 和 precision runtime。它不维护第二套训练循环。micro batch 4/6/8 的
accumulation 分别为 8/5/4，因此 effective batch 为 32/30/32。每个 FP32 和 BF16 trial
默认 warmup 5 次，并测量至少 25 次成功 optimizer updates。

JSON 报告包含 images/s、updates/s、allocated/reserved peak VRAM、data-wait/compute
时间及比值、forward/backward/optimizer 时间、非有限 loss/gradient、运行环境身份，以及
同一 micro batch 下 BF16 相对 FP32 的吞吐比和显存差值。默认要求 CUDA；为了防止
`device: auto` 在无 CUDA 时静默运行 production 模型，仅有界测试或调试可显式使用
`--device cpu`。CPU profile 的 VRAM 字段为 `null`。

CUDA phase timing 使用异步 Events，只有整段 measurement 开始和结束执行同步，不会在
每个 forward/backward/optimizer 边界阻塞。报告同时保存同一 DataBuilder artifact
bindings、每个 trial 的完整 resolved config 及 canonical SHA-256、seed、output directory，
以及 core/extension 版本和 Python source tree digest。device index 与全部 precision
support 会先于 meta model、DataBuilder 和 trial output preflight；若全部 precision 都不
受支持，命令只返回 unsupported trial records，不访问数据。

## 运行真实 smoke

smoke 使用真实 prepared AFHQ-v2，而不是 synthetic fixture。它只缩小模型和计算预算：

```powershell
uv run stochaflow train `
  --config examples/showcases/afhq-v2/experiments/smoke/train-adm-128.yaml `
  --limit-batches 4 `
  --limit-validation-batches 2 `
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
7. 生成每类一张的 CFG acceptance sample 和 conditional diagnostic artifacts。

smoke 只证明 end-to-end wiring、state 和 artifact contract，不证明收敛质量或
production 吞吐。

## Production 配置与更新公式

ADM-UNet 主配置：

```powershell
uv run stochaflow train `
  --config examples/showcases/afhq-v2/experiments/production/train-adm-128.yaml
```

DiT-S/8 对照：

```powershell
uv run stochaflow train `
  --config examples/showcases/afhq-v2/experiments/production/train-dit-128.yaml
```

ADM 保留高分辨率卷积路径，并在 32×32、16×16 和 middle 使用 Transformer blocks。
DiT-S/8 使用 8×8 patches、384 hidden size、12 blocks 和 6 heads。二者实现同一
`ClassConditionalDenoiser` capability，并共享：

- 128×128 data 与固定 class mapping；
- cosine 1,000-step discrete Gaussian Process；
- `v` prediction、MSE mean 与 condition dropout 0.1；
- micro batch 8、accumulation 4、effective batch 32；
- BF16 mixed precision、AdamW、EMA 和 step scheduler；
- validation/test 与 class-balanced diagnostic/sampling protocol。

生产配置不启用 early stopping。每 epoch 的 micro-batches 和 optimizer updates 为：

```text
floor(13,436 / 8) = 1,679 micro-batches
ceil(1,679 / 4) = 420 optimizer updates
200 × 420 = 84,000 total updates
round(0.02 × 84,000) = 1,680 warmup updates
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

```powershell
uv run tensorboard --logdir outputs
```

production diagnostic `class_conditional_diffusion_quality`：

- reconstruction 使用当前真实 batch 的原始 labels；
- sampler 以 cat/dog/wild 各 4 张的固定顺序运行；
- 使用固定 seed、EMA、CFG 2.0 和 DDIM-50；
- 每 5 epochs 写 sample grid、reconstruction panel、trajectory 和 manifest；
- 记录 timestep bucket loss、noise alignment、sample statistics 与 sampling timing。

这些结果是训练监控，不是正式 post-training dataset metric。`reference.enabled: false`
明确禁用不具备 class-aware protocol 的 reference metric。

checkpoint v9 保存完整 managed training state、precision/scaler topology、resolved
config、data binding 和 epoch-boundary RNG。production 每 5 epochs 写
`epoch_*.pt`，每 epoch 更新 `latest.pt`，并维护 `best.pt`。

strict resume：

```powershell
uv run stochaflow train `
  --resume outputs/afhq-v2/adm-128/<run-id> `
  --epochs 200
```

恢复以 checkpoint config 为权威；model、optimizer、precision、accumulation 和 data
identity 不能由新 config 替换。新运行写入 sibling timestamp directory，不续写旧日志。

## Classifier-free guidance sampling

训练后，从 best checkpoint 生成 class-balanced DDIM-50 结果：

```powershell
uv run stochaflow sample `
  --checkpoint outputs/afhq-v2/adm-128/<run-id> `
  --config examples/showcases/afhq-v2/experiments/sampling/ddim50-cfg2.yaml `
  --output-dir outputs/afhq-v2/samples/adm-ddim50-cfg2
```

overlay 固定每类 12 张、batch 12、seed `20260726`、CFG 2.0 和每 5 solver steps 的
trajectory。CFG 公式为：

```text
prediction = unconditional
           + scale × (conditional - unconditional)
```

scale 0 只运行 null branch，scale 1 只运行 conditional branch；其他非负 scale 将
conditional/null batch 拼接后单次 forward。DDIM 只消费组装好的 Gaussian dynamics，
不认识 class label 或 guidance。

## 正式 class-aware KID/FID 评估

训练期 loss、sample statistics 和 diagnostic artifacts 只用于监控。冻结 best
checkpoint 后运行独立评估：

```powershell
uv run stochaflow-afhq-v2-evaluate `
  --checkpoint outputs/afhq-v2/adm-128/<run-id>/checkpoints/best.pt `
  --config examples/showcases/afhq-v2/experiments/evaluation/ddim50-cfg2-kid-fid.yaml `
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
`DataArtifactBindings` 严格重建 AFHQ DataBuilder 的 official test loader，再通过现有
`class_conditional_denoising` SamplingBuilder 生成有序 class blocks。quality provider、
execution device 和每个 metric scope 在数据读取、输出目录和采样前预检。

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

这条 showcase 特意不增加通用 condition schema 或 Dataset/Sampler registry：

- AFHQ DataSource 只获取并发布 source-locked artifact；
- AFHQ DataBuilder 解释 class directory、构造 deterministic sampler/augmentation 和
  labeled loaders；
- class-conditional TrainingStrategy 解释 `class_label`、执行 dropout 和计算 loss；
- ADM/DiT 只实现 class-conditioned denoiser forward；
- SamplingBuilder 分配 labels 并组装 CFG；
- Gaussian Process 和 DDPM/DDIM 保持 model-free、condition-free；
- Trainer 只管理自动优化、precision、accumulation、EMA、checkpoint 和 diagnostic
  cadence。

因此新增另一个 labeled dataset 只需提供自己的 DataBuilder/source composition；兼容
`ClassConditionalDenoiser` 的新模型可以通过注册与配置进入同一训练和采样流程，而不需要
修改 runner。

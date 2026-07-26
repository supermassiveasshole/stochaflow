# AFHQ-v2 DataSource showcase

这个安装型 extension 展示一条完整的数据生命周期：

```text
官方 AFHQ-v2 archive
  -> 下载与 source-lock 校验
  -> 安全、确定性 materialization
  -> ManagedDataArtifact
  -> 内置 image DataBuilder
  -> Dataset/DataLoader
  -> DDPM training
```

训练不依赖独立的 preparation script，也没有 Trainer pre/post hook。source 插件只负责
产生 artifact；Dataset、在线随机翻转、normalize、sampler 和 DataLoader 仍由
DataBuilder 负责。

## 数据与许可

权威入口是 [ClovaAI StarGAN v2](https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq)
维护的完整 `afhq-v2-dataset`。AFHQ-v2 包含 15,803 张 512×512 RGB PNG，类别为
cat、dog 和 wild，官方 train/test 数量为 14,336/1,467。

数据采用 [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)；
请勿在未经许可的商业场景中使用。

packaged source lock 固定官方 Dropbox URL、6,955,288,636 bytes，以及 Stochaflow
对完整官方 archive 本地审计得到的 SHA-256：

```text
6f2540f22c6d8ebb8879a2bc0227666dd4fc765cc355cb073b63a835d679e4e3
```

上游没有发布 checksum；该值是本项目对固定官方下载内容的审计值，而不是上游声明。
source 不会在官方入口失败时静默切换第三方镜像。

## 安装与首次训练

在仓库根目录运行：

```powershell
uv pip install -e examples/showcases/afhq-v2

uv run stochaflow train `
  --config examples/showcases/afhq-v2/experiments/ddpm_128.yaml
```

示例配置使用 `materialization.policy: ensure`。DataBuilder 会在创建训练 run 之前：

1. 验证 exact prepared cache hit；
2. cache miss 时下载并校验官方 archive；
3. 审计 ZIP 与完整数据 contract；
4. 确定性产生 128×128 artifact；
5. 再由 artifact 构造 Dataset/DataLoader 并开始训练。

下载遵循标准 `HTTPS_PROXY`/`HTTP_PROXY` 环境变量，例如：

```powershell
$env:HTTPS_PROXY = "http://127.0.0.1:6268"
uv run stochaflow train `
  --config examples/showcases/afhq-v2/experiments/ddpm_128.yaml
```

不要把带凭据的 proxy URL 写入 YAML；环境变量不会进入 manifest 或 checkpoint。

如果已经取得官方 ZIP，可在 `source.params` 临时增加：

```yaml
archive: D:/downloads/afhq_v2.zip
```

archive 仍必须通过 lock 中的 byte count、SHA-256、ZIP 和数据 contract 校验。

## 离线严格训练

首次 materialization 成功后，把配置改为：

```yaml
materialization:
  cache_root: ./data
  policy: require
  verification: full
```

`require` 不访问网络、不重建，也不要求 raw ZIP 继续保留；它根据 packaged source
lock 和 recipe 推导 exact preparation key，并完整验证 prepared artifact。缺失或
损坏会在创建 run、模型或 optimizer 前失败。

## Artifact

默认输出：

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

固定 validation 算法从官方 train 的每类保留 300 张，因此 prepared
train/validation/test 为 13,436/900/1,467，总数仍为 15,803。处理采用一次
Lanczos resize、不 crop、固定 PNG 参数。source、Pillow recipe、inventory、
manifest 和最终 artifact 都有独立 digest。

实际 artifact bindings 会写入 `run_manifest.yaml` 和 checkpoint metadata。
strict resume 在构建数据前注入 checkpoint 的 expected identity；若缺失历史 identity，
或 source、recipe、prepared content identity 不同，会在创建新 run 以及恢复模型、
optimizer、scheduler 或 EMA 前拒绝恢复。

## 示例边界

`experiments/ddpm_128.yaml` 使用当前内置 UNet 和无条件 Gaussian denoising
TrainingBuilder，目标是展示可复现数据准备与一个有视觉冲击力的 128×128 训练入口。
类别目录及 class mapping 保存在 artifact manifest 中，但当前 recipe 不把类别作为
模型条件。class-conditional backbone、AMP 与 gradient accumulation 属于后续独立的
训练能力，不隐藏在 DataSource 中。

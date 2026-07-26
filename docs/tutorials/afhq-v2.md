# AFHQ-v2 数据准备与训练

仓库中的 `examples/showcases/afhq-v2` 是一个可安装的 DataSource extension。它把
AFHQ-v2 官方完整 archive 转换为经过审计、可复现的 128×128 managed artifact，再复用
内置 `image` DataBuilder、UNet、Gaussian training 与 DDIM sampling。

这个 showcase 主要验证数据生命周期，而不是隐藏新的训练语义：

- DataSource 负责下载、校验、安全解包、确定性预处理和原子发布；
- Dataset、随机翻转、sampler、collate 与 DataLoader 仍由内置 image recipe 组装；
- 当前模型是无条件 pixel-space diffusion，目录中的 cat、dog、wild mapping 会进入
  manifest，但不会作为模型 condition；
- 当前 Trainer 尚未在这个示例中启用 AMP 或 gradient accumulation。

[StarGAN v2 官方入口](https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq)
列出的 AFHQ-v2 使用 CC BY-NC 4.0，请先确认用途满足非商业许可要求。完整官方 archive
6,955,288,636 bytes（约 6.48 GiB），包含 15,803 张 512×512 RGB PNG；第一次运行还需要为原始 archive、
staging 和 128×128 产物预留额外磁盘空间。

## 安装 extension

在仓库根目录执行：

```powershell
uv pip install -e examples/showcases/afhq-v2
```

安装后，entry point 会注册 `afhq-v2.official`。核心 runner 不包含对这个名称的特殊
分支。

## 第一次准备并训练

```powershell
uv run stochaflow train `
  --config examples/showcases/afhq-v2/experiments/ddpm_128.yaml
```

配置使用 `materialization.policy: ensure`。在创建训练 run、模型和 optimizer 之前，
DataSource 会依次完成：

1. 检查 exact prepared artifact；
2. 缓存未命中时下载并按固定字节数和 SHA-256 校验官方 archive；
3. 审计 ZIP 路径、链接、重复项、压缩比例、图片模式、尺寸和完整数据集计数；
4. 以固定 validation 划分、Lanczos resize 和 PNG 编码生成 staging artifact；
5. 完整验证 manifest、inventory 与全部图片后原子发布。

下载遵循标准 `HTTPS_PROXY`/`HTTP_PROXY` 环境变量。不要把含凭据的代理地址写入 YAML。
如果已经取得官方 ZIP，可在 `source.params` 中临时加入：

```yaml
archive: D:/downloads/afhq_v2.zip
```

本地文件仍须通过随 extension 发布的 source lock，不会因为是手动下载而跳过校验。

## 缓存结构与离线模式

默认缓存结构为：

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

第一次成功后，可将配置改为只读离线验证：

```yaml
materialization:
  cache_root: ./data
  policy: require
  verification: full
```

`require` 不下载、不修复，也不会隔离损坏文件；缺失或篡改会直接失败。`ensure` 可以在
持有 preparation lock 时隔离损坏的 prepared artifact，并从已校验的官方 archive
确定性重建。锁采用有界等待并记录 owner 元数据；超时不会自动删除疑似 stale lock。
在 Windows 上应让训练账户独占缓存目录写权限：实现会拒绝 junction/reparse point 并
复核路径 identity，但并发的同权限写入者仍可能让 pathname-based 隔离/删除只能
fail-stop，而不能提供 POSIX descriptor-relative 操作的零错误目标副作用保证。

## Resume 与观测

run manifest 和 checkpoint metadata 会保存完整
`ManagedDataArtifactIdentity`。strict resume 会在创建兄弟 run 和恢复模型、optimizer、
scheduler、EMA 之前重新 materialize 并完整验证同一 identity；source、recipe 或内容
任一变化都会拒绝恢复。

示例已启用 local 与 TensorBoard logging，并配置固定 seed 的 DDIM 质量诊断。训练时可在
另一个终端查看：

```powershell
uv run tensorboard --logdir outputs
```

生产配置为 200 epochs、`num_workers: 2`、batch size 16。它是面向单卡一至两天预算的
起点，不是所有 GPU 的通用最优值；显存不足时应先降低 batch size，吞吐不足时再基于实际
DataLoader 和 GPU profiler 调整 worker、prefetch 与模型宽度。

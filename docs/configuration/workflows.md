# 常用工作流

## 训练

通用入口是：

```bash
uv run stochaflow train --config configs/ddpm_mnist.yaml
```

runner 加载配置、应用 CLI 覆盖、创建时间戳 run 目录，然后为每个 DataBundle（或
每个 K-fold）构建独立训练组件。所有训练参数见[CLI 参数索引](reference.md#cli-参数索引)。

### Smoke run

快速验证 schema、数据下载、模型 forward、反向传播与 checkpoint 路径：

```bash
uv run stochaflow train \
  --config configs/ddpm_flowers102.yaml \
  --epochs 1 \
  --limit-batches 2 \
  --limit-validation-batches 1 \
  --skip-final-sample
```

`--limit-*` 只截断本次运行，不修改 YAML。若想测试最终采样，移除
`--skip-final-sample`。

### CLI 覆盖优先级

训练有效值按下列顺序决定，后者优先：

1. dataclass 默认值；
2. YAML；
3. `--device`、`--output-dir`、`--epochs` 等 CLI 覆盖；
4. runner 为本次 run 生成的 `experiment.exp_id` 和时间戳输出目录。

`--no-progress` 会关闭本次运行的进度条；`--deterministic` 启用 Torch 支持的确定性
行为。配置随机 seed 仍应固定，但跨设备、PyTorch 版本和第三方算子不保证逐位一致。

## 输出目录

新训练在 `experiment.output_dir` 下创建唯一时间戳目录：

```text
outputs/<experiment>/<YYYYMMDD_HHMMSS>/
  checkpoints/
    best.pt
    latest.pt
    epoch_XXXX.pt
  metrics.jsonl
  train.log
  resolved_config.yaml
  samples/
    final/
      samples.png
      samples.pt
      resolved_sampling.yaml
```

启用 TensorBoard、W&B、diagnostic 或 trajectory 后会增加对应子目录/artifact。
`artifacts.checkpoint_every` 控制编号 checkpoint 的频率；`latest.pt` 和 `best.pt`
按恢复与模型选择规则更新。

## 恢复训练

从配置输出根目录下最新 run 的 `checkpoints/latest.pt` 恢复：

```bash
uv run stochaflow train \
  --config configs/ddpm_mnist.yaml \
  --resume
```

显式指定 checkpoint：

```bash
uv run stochaflow train \
  --config configs/ddpm_mnist.yaml \
  --resume outputs/ddpm_mnist/<run>/checkpoints/latest.pt
```

checkpoint 保存模型、optimizer、scheduler、EMA、训练进度和新 schema 配置。恢复时
配置与 checkpoint 必须兼容；旧 `data.dataset`/`data.source` checkpoint 会收到明确
迁移错误，不自动转换。

## K-fold

运行所有 fold：

```yaml
data:
  splits:
    mode: kfold
    num_folds: 5
    fold_index: null
```

只运行第三个 fold（索引从 0 开始）：

```yaml
data:
  splits:
    mode: kfold
    num_folds: 5
    fold_index: 2
```

划分先按 YAML source 顺序合并，再使用 `experiment.seed` 生成全局索引。每个 fold
独立构建 DataBundle、模型、optimizer、日志与 checkpoint；不要跨 fold 共享训练状态。

## checkpoint 采样

checkpoint 内含模型和训练配置，所以可以只给 checkpoint：

```bash
uv run stochaflow sample \
  --checkpoint outputs/ddpm_mnist/<run>/checkpoints/best.pt
```

切换为 DDIM 并覆盖参数：

```bash
uv run stochaflow sample \
  --checkpoint outputs/ddpm_mnist/<run>/checkpoints/best.pt \
  --sampler ddim \
  --sampler-param num_inference_steps=100 \
  --sampler-param eta=0.0
```

`--sampler-param KEY=VALUE` 可重复，value 按 YAML 标量/列表规则解析。若
`--sampler` 改变名称，原采样器构造参数会先清空，再应用 CLI 参数。

也可以只给外部配置，让 runner 在 `experiment.output_dir` 下寻找最新 `best.pt`：

```bash
uv run stochaflow sample --config configs/ddpm_mnist.yaml
```

同时给出两者时：checkpoint 提供权重与基础训练结构；外部配置提供 sampling 段，
但模型、训练 diffusion、noise schedule 和通道数必须兼容；最后应用 sample CLI
覆盖。配置和 checkpoint 中的 `data.modules` 都会在解析自定义组件前加载。

## 采样形状与 EMA

独立采样、训练后验收、trajectory 和 DDPM diagnostic 都使用
`data.image.channels × sample_bucket.height × sample_bucket.width`，不依赖最后一个训练
batch。`ema.enabled` 与 `ema.use_for_sampling` 同时为 true 且 checkpoint 含 EMA 时，
采样优先使用 EMA 权重。

每次采样写出 `samples.pt`、PNG 网格和 `resolved_sampling.yaml`；开启
`sampling.debug.trajectory.enabled` 后还会写 trajectory Tensor、静态网格与 GIF。

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

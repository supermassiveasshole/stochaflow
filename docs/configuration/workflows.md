# 常用工作流

## 训练

通用入口是：

```bash
uv run stochaflow train --config configs/ddpm_mnist.yaml
```

runner 加载配置、应用 CLI 覆盖、创建一个时间戳 run 目录，调用一次 DataBuilder，
然后构建一套训练组件。所有训练参数见[CLI 参数索引](reference.md#cli-参数索引)。

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
`--skip-final-sample`；若 `sampling.shape` 为 null，训练后默认不执行固定 shape
采样。

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
  diagnostics/
    diffusion_quality/
      epoch_NNNN/
        manifest.yaml
        denoiser/
        <sampler-profile>/
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

checkpoint 保存模型、optimizer、scheduler、EMA、训练进度和 resolved 配置。v4 只保存
`data: {name, params}`，不保存 Dataset、Sampler、DataLoader 或 partition 运行时状态。
恢复时配置与 checkpoint 必须兼容；旧格式会收到明确错误，不自动转换。

## K-fold

K-fold 是支持它的图像 recipe 的私有能力。一次配置只运行一个 fold，例如第三个 fold
（索引从 0 开始）：

```yaml
data:
  name: image
  params:
    source:
      kind: image_folder
      path: ./data/images
    partition:
      mode: kfold
      num_folds: 5
      fold_index: 2
    image:
      size: [64, 64]
```

`fold_index` 不可省略。运行全部五折需要五次独立运行，通常由项目脚本或外部 sweep
分别覆盖这个字段。每次运行独立构建 DataBuilder、模型、optimizer、日志和 checkpoint；
核心 runner 不展开 fold，也不跨 fold 共享训练状态。

## 训练期多 sampler diagnostic

`diffusion_quality` 在相同 EMA 权重、初始噪声和固定 seed 下并排运行多个 sampler。
轻量 denoiser 指标、sampler 指标、图片和参考质量算法都是独立 provider；配置可以
替换或禁用任一类别，而无需修改 diagnostic orchestrator。每个 sampler profile 必须
提供稳定且唯一的 `id`，`name` 对应 diffusions Registry。

```yaml
diagnostics:
  - name: diffusion_quality
    params:
      modules: []
      cadence:
        step_every: 100
        artifact_every_epochs: 5
      sampling:
        sample_num: 16
        batch_size: 16
        seed: 123
      use_ema: true
      failure_policy: raise
      samplers:
        - id: ddpm_full
          name: ddpm
          params: {}
          trajectory:
            enabled: true
            params: {state_interval: 100}
            gif_fps: 8
        - id: ddim_50
          name: ddim
          params: {num_inference_steps: 50, eta: 0.0}
          trajectory:
            enabled: true
            params: {step_interval: 5}
            gif_fps: 8
      providers:
        step_metrics:
          - name: timestep_bucket_loss
            params: {buckets: 10}
          - name: noise_alignment
            params: {}
          - name: x0_reconstruction
            params: {timesteps: [50, 250, 500, 900]}
        sampler_metrics:
          - name: sample_statistics
            params: {}
          - name: sampling_performance
            params: {}
        denoiser_artifacts:
          - name: reconstruction_panel
            params:
              timesteps: [50, 250, 500, 900]
              max_samples: 16
        sampler_artifacts:
          - name: sample_grid
            params: {nrow: 4}
          - name: trajectory
            params: {}
      reference:
        enabled: false
        every_epochs: 20
        num_real: 2048
        num_fake: 2048
        batch_size: 64
        metrics:
          - name: kid
            params: {subsets: 100, subset_size: 1000}
          - name: fid
            params: {}
```

`modules` 会在解析 provider 名称前按顺序导入第三方模块。某个 `providers` 分类省略时
使用内置组合，显式写成 `[]` 时禁用该分类。provider 名称、构造参数、重复项和输出
冲突都会严格校验。

Local logger 记录 artifact 路径，TensorBoard 和 W&B 同时显示 PNG。启用 KID/FID
前需要 `uv sync --extra quality`，并且本次训练必须有 validation DataLoader。参考指标
只用于监控，不参与 best checkpoint 或 early stopping。

`failure_policy: raise` 会让采样或 provider 异常终止训练；`warn` 按 provider/profile
隔离失败、继续执行其余组件，并把错误同时写入 `diagnostics/system/error_count`、日志
和 epoch manifest。未知 sampler/provider、重复名称、缺少 trajectory 接口等配置错误
始终在训练开始前失败。

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
但模型、训练 diffusion 和 noise schedule 必须兼容；最后应用 sample CLI
覆盖。配置和 checkpoint 中的 `extensions.modules` 都会在解析自定义组件前加载。

## 采样形状与 EMA

独立采样、训练后验收、trajectory 和 diffusion quality diagnostic 使用
`sampling.shape`，它不含 batch 维且与 DataBuilder 独立。`ema.enabled` 与
`ema.use_for_sampling` 同时为 true 且 checkpoint 含 EMA 时，
采样优先使用 EMA 权重。

每次采样都写 `resolved_sampling.yaml`。`sampling.writers` 决定其他输出：`tensor`
写 PT，`image` 写 PNG/GIF；开启 trajectory 后，两者会写各自支持的 trajectory
artifact。

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

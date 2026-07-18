# 自定义代码扩展支持实施计划

- 状态：Stage 2 完成，Stage 3 待实施
- 制定日期：2026-07-17
- 目标分支：`feature/custom-code-extension-support`

## 目标

将 Stochaflow 从包含内置实验能力的 Python 包扩展为“核心框架 + 用户项目 +
可插拔扩展”的训练框架。用户安装 Stochaflow 后，可以创建独立项目，编写并注册
自定义模型、数据管线、算法、Loss 和训练策略，再通过统一的 CLI 和
YAML 配置完成训练、恢复与采样。核心数据管线不假设样本是图像，图像、多分辨率
图像、空间场、向量、序列和结构化条件数据通过相同的扩展边界接入。

目标用户流程：

```text
安装 stochaflow
→ stochaflow project create my_project
→ 编写并注册 extension
→ 在项目清单中声明 extension 模块
→ 在 YAML 中选择数据源、batching、模型、Loss 和训练策略
→ stochaflow train --config configs/train.yaml
```

首版默认边界：

- 使用本地 `src` 布局项目；
- 通过项目清单显式加载扩展，不扫描目录；
- Dataset 单样本与 batching 由 DataPipeline 按需定义；送入 Trainer 的 batch 可以是
  Tensor，或由 mapping、tuple、list 和 Tensor 组成的嵌套结构。核心数据管线不定义
  `input`、`target`、`condition` 等业务字段；
- 多分辨率图像 bucket 作为内置图像管线能力保留，不作为所有 Dataset 必须实现的
  核心契约；
- 支持多模型、多 Loss，但只有一个优化器和一次反向传播；
- 核心 Trainer 继续管理训练循环、反传、调度、日志和 checkpoint；
- 扩展相关 schema 直接使用新格式，不提供 legacy 迁移或兼容别名。

## Stage 1：统一扩展注册与配置入口

### 目标

为所有可配置组件建立统一、语义明确的全局扩展入口。

### 实施内容

- 新增顶层 `extensions.modules` 配置，扩展模块在配置校验和组件构建前按声明顺序
  导入。
- 在 resolved config 和 checkpoint 中保存最终生效的扩展模块列表，确保仅使用
  checkpoint 采样时也能加载扩展。
- 新增稳定的 `stochaflow.extensions` 公开入口，首批导出 `REGISTRIES`、
  `Registry`、`RegistryError`、组件基类和 `ComponentConfig`。
- 保持现有 Registry 和内置组件行为不变。

### 验收条件

- 内置 YAML 全部改用顶层 `extensions.modules`。
- 自定义模型、DatasetFactory 和 diffusion 可以通过 `extensions.modules` 注册。
- 重复名称、错误基类和缺失模块在训练开始前明确报错。
- Pytest、Ruff 和 Pyright 全部通过。

## Stage 2：注册化、模态无关的数据管线

### 目标

将扩展点上移到整个 `DataPipeline`。核心只通过 Registry 构建管线并消费
`DataBundle`，不规定 Dataset 类型、split、混合、sampler、collate 或 batch 语义。
复杂数据组织由用户自定义 DataPipeline 完整拥有，不新增全局 BatchingStrategy 或
SplitStrategy Registry。

### 配置接口

通用 map-style 数据使用：

```yaml
data:
  name: map
  params:
    dataset:
      id: physics
      factory: physics_fields
      params: {}
      splits: {train: train, validation: validation, test: test}
    splits: {mode: official}
    dataloader: {batch_size: 64, steps_per_epoch: auto}

sampling:
  shape: [32]
```

多分辨率图像使用独立内置管线：

```yaml
data:
  name: multi_resolution_image
  params:
    datasets:
      - id: mnist
        factory: mnist
        params: {}
    image: {channels: 1, normalize: true}
    batching:
      buckets:
        - {name: square_32, height: 32, width: 32}
      base_bucket: square_32
      dynamic_batch_size: true
    dataloader: {batch_size: 128, steps_per_epoch: auto}

sampling:
  shape: [1, 32, 32]
```

- `StochaflowConfig.data` 是 `ComponentConfig`，由 `REGISTRIES.data_pipelines` 构建；
- `DataPipelineContext` 保存复制后的 `params` 与实验 seed；
- `DataPipeline.build()` 返回非空 `list[DataBundle]`；`SplitData` 可以不暴露 Dataset，
  但训练 split 必须通过 loader 长度或 `num_batches` 提供有限 epoch；
- 内置 `map` 支持单一 DatasetFactory、固定 batch、默认 collation 以及 none、official、
  random_holdout、kfold；
- 内置 `multi_resolution_image` 承接多源权重、bucket、动态 batch 与确定性 set_epoch；
- `sampling.shape` 描述单个生成状态的默认形状，不包含 batch 维，可以是任意 rank；
- `sampling.batch_size` 与 data pipeline 独立；配置不读取旧 data schema。

### 数据契约

- `DatasetFactoryContext` 只携带 `source_id` 与 factory `params`；
- `DatasetView` 保留 `dataset` 和稳定的 `sample_keys`，将强制 `bucket_ids` 替换为可选
  的通用 batch metadata。metadata 可以描述 image size、sequence length、point count
  或用户策略需要的其他逐样本信息；
- `DatasetSelection`、split 与图像 bucket 类型不再属于顶层稳定扩展 API；
- 核心将 `Batch` 视为 `Any`，默认 collation 保留 Tensor、mapping、tuple 和 list 的嵌套
  结构，不提取第一个元素，也不丢弃 label、condition、mask 或 metadata；
- 内置 torchvision Factory 返回原始样本与图像尺寸 metadata；图像管线负责预处理并
  使用默认 collation，因此 label 和 condition 不再丢失。

### Sampling artifact 与 Diagnostic 解耦

- sampling runtime 从 `sampling.shape` 或具体 sampler 的推导结果获得状态形状，不再读取
  data pipeline；
- 新增 `REGISTRIES.sampling_artifact_writers` 与 `SamplingArtifactWriter`。内置 `tensor`
  保存 PT，内置 `image` 自行校验 NCHW、通道数并保存 PNG/GIF；
- writer 返回的 key 必须非空且唯一，路径必须存在，任一 writer 失败则采样失败；
- `DiagnosticBuildContext.sample_shape` 改为可选任意 rank，图像 diagnostic 自行验证；
- checkpoint format 升级为 v3，训练恢复和 checkpoint-only sampling 拒绝旧格式。

### 验收条件

- 当前内置图像配置迁移到 `multi_resolution_image` 后，bucket 选择、动态 batch size、
  多数据源混合和训练结果语义保持不变；
- 固定形状的非图像 Tensor Dataset 可以通过 YAML/CLI 完成最小训练和采样；
- paired super-resolution 风格的 mapping batch 能完整保留 `condition`、`target` 和
  `metadata` 并到达训练 step；
- 自定义 DataPipeline 和 sampling writer 可以通过 `extensions.modules` 注册并由配置构建；
- 同一个 structured batch 在 train、validation 和 device transfer 中保持结构；
- split、sample key、sampling weight 和 checkpoint resume 回归测试通过；
- 配置、公开 API、README、数据管线、扩展手册和配置参考同步更新；
- Pytest、Ruff 和 Pyright 全部通过。

### 逻辑提交

提交主题为 `Refactor modality-neutral data pipeline`。

## Stage 3：用户项目系统与 CLI 脚手架

### 目标

让用户无需操作 `PYTHONPATH`，即可创建和使用独立 Stochaflow 项目。

### 实施内容

- 新增命令：

  ```bash
  stochaflow project create my_project
  stochaflow train --project /path/to/project --config configs/train.yaml
  stochaflow sample --project /path/to/project --checkpoint checkpoints/best.pt
  ```

- 创建以下项目结构：

  ```text
  my_project/
  ├── stochaflow.project.yaml
  ├── pyproject.toml
  ├── .gitignore
  ├── configs/
  │   └── train.yaml
  ├── src/
  │   └── my_project/
  │       ├── __init__.py
  │       └── extensions/
  │           └── __init__.py
  └── tests/
      └── test_extensions.py
  ```

- 项目清单采用以下首版结构：

  ```yaml
  schema_version: 1

  project:
    name: my_project
    source_roots:
      - src

  extensions:
    modules:
      - my_project.extensions
  ```

- 项目发现顺序固定为：显式 `--project`、配置文件目录向上查找、当前工作目录向上
  查找；找不到项目时继续支持独立配置。
- CLI 加载项目的 `source_roots`，再按清单顺序导入扩展模块。
- 项目清单模块与配置中的 `extensions.modules` 按顺序合并并稳定去重。
- 项目名称中的连字符转换为 Python 包名中的下划线；非空目标目录拒绝覆盖。

### 验收条件

- CLI 可以生成合法的 `src` 项目。
- 新项目中的自定义组件可以直接被 `train` 和 `sample` 使用。
- checkpoint-only sampling 使用同一套项目发现逻辑。
- 使用临时合成数据完成端到端 CLI 训练测试，不依赖网络下载。

## Stage 4：Loss 与训练策略扩展 API

### 目标

移除当前 `diffusion + objective → train_step_fn` 的硬编码选择逻辑，将训练行为提升为
正式扩展点。

### 配置接口

```yaml
training:
  strategy:
    name: diffusion_denoising
    params: {}
  losses:
    primary:
      name: ddpm_epsilon
      params: {}

trainer:
  num_epochs: 30
  device: auto
```

- `training` 描述训练算法和 Loss，`trainer` 继续描述循环与运行参数。
- 移除顶层 `objective`，内置配置直接改用 `training.losses.primary`；旧字段作为未知
  配置字段报错。
- 新增 `REGISTRIES.losses` 和 `REGISTRIES.training_strategies`，移除
  `REGISTRIES.objectives`，不提供兼容别名。

### 训练策略契约

公开的训练策略需要提供：

- `training_step(batch)`：返回可反向传播的总 Loss；
- `evaluation_step(batch)`：验证阶段计算，默认可复用训练计算；
- `trainable_parameters()`：声明单个优化器应更新的参数；
- `to(device)`、`train_mode()` 和 `eval_mode()`：管理学生、教师和 Loss 模块；
- `state_dict()` 和 `load_state_dict()`：保存与恢复策略自身状态。

`TrainStepOutput` 包含：

- `loss`：核心 Trainer 用于反向传播；
- `metrics`：总 Loss 和各分项 Loss；
- `diagnostics`：供诊断插件消费的 Tensor 或中间结果。

核心 Trainer 继续负责清空梯度、一次反向传播、梯度裁剪、单 optimizer step、
scheduler、EMA、日志、验证、early stopping 和 checkpoint。

TrainingStrategy 接收 Stage 2 保留下来的完整 structured batch，并自行解释
`state`、`condition`、`target`、`mask` 等字段；核心 Trainer 和数据管线不执行固定位置
解包或字段重命名。

### Checkpoint 格式

- checkpoint 格式升级，加入可选的训练策略状态。
- 不在策略状态中重复保存主模型权重。
- 新格式不承诺读取旧 checkpoint。
- sampling 只构建推理所需组件，不强制实例化训练策略。

### 验收条件

- DDPM/DDIM 内置训练行为保持不变。
- 用户可以从 YAML 构建自定义 Loss 和自定义训练策略。
- 分项 Loss 正确写入日志与进度信息。
- 新 schema 下的 checkpoint、恢复训练和采样测试全部通过。

## Stage 5：蒸馏纵向案例

### 目标

使用一个真实的用户扩展项目验证训练策略接口，而不只验证孤立的单元测试。

### 实施内容

- 提供自定义蒸馏 Loss 和 `KnowledgeDistillationStrategy`。
- 从配置构建教师模型并加载教师 checkpoint。
- 冻结教师参数并始终保持 eval 模式。
- 训练学生模型，将基础 Loss 与蒸馏 Loss 按配置权重组合。
- 分别记录总 Loss、基础 Loss 和蒸馏 Loss。
- 提供完整的项目清单、扩展代码和 YAML 示例。

### 验收条件

- 教师参数不产生梯度，且训练前后不变化。
- 学生参数在训练后发生更新。
- 两个 Loss 都参与总 Loss，分项指标均被记录。
- checkpoint 可以恢复学生、optimizer、scheduler 和策略状态。
- 恢复后 global step 与训练过程连续。
- 最终 checkpoint 可以由标准 `stochaflow sample` 使用。

## Stage 6：可复现性与文档收尾

### 实施内容

- 在 resolved config 和 checkpoint metadata 中记录项目名称、清单版本、扩展模块、
  batching 策略、训练策略及 Loss 注册名。
- 缺失扩展时报告具体模块、项目根目录，并提示通过 `--project` 修复。
- 补充项目创建、自定义 DatasetFactory、自定义 batching、structured batch、
  自定义模型、自定义 Loss、自定义训练策略、蒸馏示例、破坏性变更和 checkpoint
  可移植性文档。
- 更新 README、配置参考和故障排查手册。
- 完成最终构建与质量检查。

### 最终验证

```bash
uv run pytest
uv run ruff check .
uv run pyright
uv build
```

## 首版明确不做

- 自动扫描 `extensions/` 目录；
- Python entry point 插件市场；
- 多 optimizer 或交替更新；
- extension 接管完整 epoch 循环；
- 将用户扩展源码打包进 checkpoint；
- 自动上传或分发用户项目。

## 施工规则

- 严格按 Stage 1 到 Stage 6 的顺序实施。
- 每个 Stage 独立开发、测试和验收，未通过验收不进入下一 Stage。
- 每个 Stage 对应一个逻辑提交，避免跨 Stage 混合修改。
- 如实施中发现必须改变公开接口或首版边界，先更新本计划并确认，再继续施工。

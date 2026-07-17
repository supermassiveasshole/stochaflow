# 自定义代码扩展支持实施计划

- 状态：Stage 1 完成，Stage 2 待实施
- 制定日期：2026-07-17
- 目标分支：`feature/custom-code-extension-support`

## 目标

将 Stochaflow 从包含内置实验能力的 Python 包扩展为“核心框架 + 用户项目 +
可插拔扩展”的训练框架。用户安装 Stochaflow 后，可以创建独立项目，编写并注册
自定义模型、数据组件、算法、Loss 和训练策略，再通过统一的 CLI 和 YAML 配置完成
训练、恢复与采样。

目标用户流程：

```text
安装 stochaflow
→ stochaflow project create my_project
→ 编写并注册 extension
→ 在项目清单中声明 extension 模块
→ 在 YAML 中选择模型、Loss 和训练策略
→ stochaflow train --config configs/train.yaml
```

首版默认边界：

- 使用本地 `src` 布局项目；
- 通过项目清单显式加载扩展，不扫描目录；
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

## Stage 2：用户项目系统与 CLI 脚手架

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

## Stage 3：Loss 与训练策略扩展 API

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

## Stage 4：蒸馏纵向案例

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

## Stage 5：可复现性与文档收尾

### 实施内容

- 在 resolved config 和 checkpoint metadata 中记录项目名称、清单版本、扩展模块、
  训练策略及 Loss 注册名。
- 缺失扩展时报告具体模块、项目根目录，并提示通过 `--project` 修复。
- 补充项目创建、自定义模型、自定义 Loss、自定义训练策略、蒸馏示例、破坏性变更和
  checkpoint 可移植性文档。
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

- 严格按 Stage 1 到 Stage 5 的顺序实施。
- 每个 Stage 独立开发、测试和验收，未通过验收不进入下一 Stage。
- 每个 Stage 对应一个逻辑提交，避免跨 Stage 混合修改。
- 如实施中发现必须改变公开接口或首版边界，先更新本计划并确认，再继续施工。

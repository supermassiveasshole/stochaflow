# Extension 重构决策记录

本文记录 extension 重构中已经确定的边界、明确删除的方案和仍需评审的技术债。它不重复
[实施计划](../custom-code-extension-support-plan.md)，而是作为每个 Stage 提交检查点的架构
审阅入口。后续批注应优先修改本文的决策或待定项，再更新实施计划和代码。

## Stage 3 检查点

状态：实现、针对性测试、代码审查和维护者审阅均已完成；检查点为 `1bddd63`。

### 最终职责边界

```text
Process   -> 可选的、模型无关的 probability path 与持久状态
Dynamics  -> family 内的生成方向、模型适配和 prediction semantics
Sampler   -> 完整数值求解循环及 accepted-step lifecycle
Builder   -> 任务组合、兼容性验证、初始化和结果规范化
Observer  -> 消费 SamplingObservation
Writer    -> 物化已经形成的 SamplingOutput
```

框架只统一 Registry、配置、checkpoint 和完整 `Sampler.sample()` 生命周期。不同算法
family 不需要共享 `predict()`、`drift()`、`score()`、`denoise()` 或 `step()` 数学接口。

### 取舍与去留

| 问题 | 决策 | 原因 |
| --- | --- | --- |
| Process 是否必需 | 保留为可选资产 | direct transform 或其他方法不应伪造 probability path |
| Process 根接口 | 只保留无数学方法的 `Process` | 数学能力由真实 family 定义，避免 universal API |
| Gaussian Process 契约 | 保留 `DiscreteGaussianDenoisingProcess` | 明确整数时间和 adjacent posterior，不暗示连续 Gaussian 兼容 |
| Dynamics Registry/YAML | 删除 | Dynamics 是 Builder 组装出的运行时对象，没有独立配置身份 |
| Process 工厂化 Dynamics | 删除 | Process 必须保持 model-free；组合责任属于 Builder/diagnostic |
| Sampler 公共接口 | 保留完整 `sample()` | 能表达 multistep、内部子步、自适应和 rejection；不强制单步 `step()` |
| condition/guidance 参数 | 不进入 Process 或 Sampler | 由 model callable、Dynamics wrapper 和 Builder 拥有 |
| trajectory API | 保留 Observer 事件流 | 避免复制 sample/trajectory 两套求解循环 |
| Gaussian schedule 状态 | 构造期生成 Process-owned snapshot | 消除可变 schedule 与缓存 posterior 之间的双重权威状态 |
| learnable schedule | 本 Stage 拒绝 | 需要动态 coefficient capability，不能静默套用固定快照语义 |
| checkpoint 缺失 Process | 省略 `process_state_dict` | 空 mapping 是合法 Process state，不能同时表示“不存在” |
| 旧 diffusion API/v4 checkpoint | 删除且不迁移 | Stage 3 尚未发布，保留兼容层会固化错误边界 |

### 已验证的 OCP 场景

- Gaussian family：同一 `DiscreteGaussianDenoisingProcess` 可复用 DDPM、DDIM 和自定义
  `GaussianDenoisingDynamics` wrapper。
- 新算法 family：测试私有 Flow Process、VectorField Dynamics、Sampler 和 Builder 可经过
  既有 Registry、checkpoint 与 sampling runtime，不修改核心 dispatch。
- direct transform：测试私有 Builder 在 `process: null` 下不创建 Dynamics 或 Sampler，
  仍可经过 checkpoint-only sampling、writer 和 manifest。
- task variation：condition、CFG、physics guidance 和 partial initialization 留在自定义
  Builder/Dynamics；核心 Gaussian Process 和 Sampler 不增加任务字段。

### 当前刻意保留的临时限制

以下内容不是 Stage 3 的最终架构承诺，应由后续 Stage 处理：

- 训练入口仍是 Gaussian epsilon bridge，并按当前 `objective` 选择 train step；正式
  `TrainingStrategy`、Objective 注入和 structured batch 解释留给 Stage 4。
- 顶层 `model`、`data` 和 `objective` 仍是必填项，即使某个 sampling-only Builder 理论上
  不需要全部资产；只有 `process` 在本 Stage 变为可选。
- EMA 只跟踪 primary inference model；多模型、教师模型和额外训练资产尚无
  checkpoint 契约。
- diffusion-quality diagnostic 仍是 Gaussian/image 专用能力，不代表通用 diagnostic 必须
  理解 Process 或 Sampler。
- `SamplingOutput` 仍整体驻留内存；streaming/chunked artifact 边界留给容量 Stage。

## Stage 4 检查点

状态：实现、针对性验证、独立审查、问题修复和最终复审均已完成；检查点提交主题为
`Stage 4: Add extensible training builders`。

维护者批注和参考实现调研形成以下约束：

1. 顶层 `training: {name, params}` 选择注册的 `TrainingBuilder`，而不直接选择
   Strategy。Builder 是训练侧的 composition root，组装并返回 `TrainingPlan`。
2. `TrainingStrategy` 只定义训练逻辑：解释 batch、调用已注入的模型/Objective/
   Process，返回单一标量 loss、metrics 和可选 diagnostics。它没有 Registry 或
   YAML 身份，也没有 `to/train/eval`、parameter selection、optimizer、factory、state
   或 checkpoint API。
3. `TrainingPlan` 明确区分 primary model、可选 Process、可选 Objective 和具名
   auxiliary modules。Builder 可以使用注入的 model/objective factory 构建辅助资产；
   Strategy 只接收 Builder 显式注入的 Python 对象。
4. Plan 被验证后，core/Trainer 统一管理全部资产的 device、mode、optimizer、EMA、
   gradient clipping 和 checkpoint。`requires_grad` 决定参数是否进入 optimizer；
   auxiliary `mode="eval"` 表示像 frozen teacher 这样的模块始终保持 eval。
5. `supervised` 和 `gaussian_denoising` 是内置 TrainingBuilder，分别组装通用
   supervised Strategy 和 Gaussian family Strategy。Objective 保留为唯一损失抽象，
   不增加重复的 Loss 基类或 Registry。
6. 蒸馏通过自定义 Builder 组装 student、teacher、task/distillation Objective 和
   `KnowledgeDistillationStrategy`。teacher 作为固定 eval、无梯度 auxiliary module 由核心
   托管；Strategy 只定义 forward、loss 组合和 metrics。
7. Trainer 只保留循环与自动优化责任。Stage 4 只支持“一个标量总 loss +
   一个 optimizer”；多 optimizer/manual optimization 不塞入 Strategy 或 Plan 的可选字段。
8. checkpoint v6 保留 primary Model、可选 Process/Objective、EMA 和 optimizer/scheduler
   固定字段；辅助模块按稳定名称进入 `training_assets_state_dict`。不保存
   `strategy_state_dict`；checkpoint-only sampling 不构建任何训练侧 Builder/Strategy/assets。
9. Gaussian diagnostic 不得根据 primary model 类型或签名重建调用方式。可复用该
   diagnostic 的 Strategy 通过一个可选窄 capability 同时暴露 `prediction_type` 和
   diagnostic prediction callable；这仍属于 Strategy 已拥有的模型调用语义，不包含
   sampler、artifact 或 diagnostic lifecycle。无法在 diagnostic 上下文中提供 condition
   的 Strategy 不实现该 capability，并在组合边界得到明确错误。

### 蒸馏适用边界

| 场景 | 组合方式 | 当前自动循环是否支持 |
| --- | --- | --- |
| Frozen-teacher online distillation | Builder 加载并冻结 teacher；Plan 将其声明为固定 eval auxiliary；Strategy 合成 task/distillation loss | 支持 |
| Offline distillation | DataBuilder 直接提供预计算 teacher target；Strategy 计算并合成 loss | 支持，不需要 teacher asset |
| Feature/logit/score distillation | 由具体 Builder/Strategy 私有解释中间特征、temperature 或权重 | 支持，只要返回单一标量总 loss |
| 联合训练 student 与 teacher | 两者都作为可训练资产并共享一个 optimizer | 契约可表达，但需具体 Builder 明确选择参数和一致的单步 loss |
| 独立 teacher optimizer、交替更新或 manual backward | 新的训练 loop family | 不在 Stage 4 自动循环内 |

蒸馏的变化点分为两类：资产组合变化归 `TrainingBuilder`，单步数学变化归
`TrainingStrategy`。Trainer 不识别“蒸馏”这个任务名称，也不增加 teacher 专用分支。
这保持 Strategy 的单一职责，同时避免为了一个多模型任务把顶层 YAML 扩展成通用计算图。

### 参考实现取舍

- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) 将
  `training_step` 和 `configure_optimizers` 放进 `LightningModule`，由 Trainer 自动化循环。
  Stochaflow 只采用“任务计算与循环分离”：Strategy 可以持有 Builder 注入的 model
  引用，但不成为 model 容器，不配置 optimizer，也不成为第二个 lifecycle owner；资产组合
  由独立 Builder/Plan 表达。
- [Hugging Face Accelerate checkpointing](https://huggingface.co/docs/accelerate/main/usage_guides/checkpoint)
  由调用方先构造 model/optimizer/dataloader，再由 runtime prepare。Stochaflow 同样在
  TrainingBuilder 构造资产并通过 TrainingPlan 交给 runtime 托管，再将已构造的依赖
  注入 Strategy，而不反向让 Strategy 创建或登记资产。
- [PyTorch `Module.state_dict`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict)
  会递归保存子模块参数和 buffer。Strategy 因此不继承 `nn.Module`；Plan 中的
  主资产和具名 auxiliary modules 由核心分别保存，避免同一权重重复。

这些决策保证简单训练的 Builder/Strategy 都很短，同时让蒸馏等多模块组合
不污染 Strategy 职责。Strategy 始终只有一个变化理由：训练计算语义变化。

### 独立审查与修复

首轮独立审查不修改代码，按影响分类如下：

| 分类 | Finding | 处理结果 |
| --- | --- | --- |
| Critical | 无 | — |
| High | auxiliary mapping 声明顺序会改变 optimizer 参数位置，resume 时可能静默错绑 Adam state | auxiliary 按稳定名称排序，并增加反转声明顺序的 resume 回归测试 |
| Medium | 空/冻结 primary model 配合可训练 auxiliary 时，EMA 无法区分“未 store”和“合法空快照” | 使用显式 stored-state 标记，空 EMA 快照可安全 checkpoint/restore |
| Medium | checkpoint clone 假定所有 module state 都是 Tensor | Tensor 执行 detach/clone，任意 extra state 执行 deepcopy |
| Medium | diagnostic 仅凭 prediction type 猜测 `model(state, time)` 签名 | 新增 `GaussianDiagnosticSemantics` prediction callable，并用独立非标准签名 Strategy 验证 |
| Suggestion | Trainer 重新暴露可变 managed-assets dict | 暴露只读 mapping，Plan 仍是唯一权威资产清单 |
| Suggestion | Strategy metric 可覆盖核心 `train/loss` 或 `train/epoch` | 隔离到 `train/strategy/*` 命名空间 |

架构修正后进行第二轮独立复审，发现两个 checkpoint/public-API Medium 和一个代码收缩
Suggestion；由于它们直接影响本 Stage 已承诺的契约，仍在本 Stage 解决：

- 保留 PyTorch `state_dict()` 的 `OrderedDict` 与 `_metadata`，versioned module 可以恢复；
- 从稳定 `stochaflow.extensions` 导出并文档化 `GaussianDiagnosticSemantics`；
- 删除 `GaussianTrainingRuntime` 未使用的 primary model 字段，只保留 Process、prediction
  type 和 task-adapted callable。

最终独立复审结果为 Critical、High、Medium 和 Suggestion 均无遗留，未发现新的 SOLID、
OCP、API 或技术债问题。

### 保留限制

- 自动训练 lifecycle 仍只有一个 optimizer、一个标量总 loss 和一次 backward；多 optimizer、
  交替更新或 manual backward 属于未来真实需求驱动的新 loop family。
- EMA 只跟踪 primary inference model；Process、Objective 和 auxiliary module 保存 raw state。
- 内置 Gaussian Strategy 只支持 bare Tensor 或空 condition mapping；conditional/SR 使用
  自定义 TrainingBuilder/Strategy，并仅在能提供 diagnostic condition 时声明
  `GaussianDiagnosticSemantics`。
- Strategy 不拥有持久状态；需要持久化的 teacher、额外 Objective 或其他 `nn.Module`
  必须以稳定名称进入 TrainingPlan auxiliary assets。

### 验证与逻辑提交

Stage 4 针对性验证覆盖训练组合、蒸馏、checkpoint v6、sampling-only、diagnostics、配置和
公开 API；常规门禁为 Ruff、Pyright 与生成配置引用检查。完整 pytest、build、Sphinx 和
额外静态检查留到整版 feature 分支合并验收。

逻辑提交主题：`Stage 4: Add extensible training builders`。

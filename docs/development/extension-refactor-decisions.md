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
- `SamplingOutput` 仍整体驻留内存；Stage 6 已验证 final-only DFSR 容量并将
  dense trajectory 限定为小样本 preview，真实总输出超过预算时再设计增量
  artifact lifecycle。

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
| 多 teacher 或多蒸馏 Objective | Builder 组装具名辅助资产；Strategy 在一次 step 内组合预测和 loss | 支持，只要返回单一标量总 loss |
| 联合训练 student 与 teacher | Plan 将两者声明为可训练资产并共享一个 optimizer；Strategy 返回一致的单步总 loss | 支持单 optimizer、单 backward 形态 |
| 独立 teacher optimizer、交替更新或 manual backward | 新的训练 loop family | 不在 Stage 4 自动循环内 |

蒸馏的变化点分为两类：资产组合变化归 `TrainingBuilder`，单步数学变化归
`TrainingStrategy`。Trainer 不识别“蒸馏”这个任务名称，也不增加 teacher 专用分支。
这保持 Strategy 的单一职责，同时避免为了一个多模型任务把顶层 YAML 扩展成通用计算图。
Strategy 可以通过普通构造函数接收任意已经组装好的依赖，但这种依赖注入不赋予它
资产所有权：device、mode、可训练参数、optimizer 和 checkpoint 仍由 Plan 与核心管理。

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
| Medium | checkpoint clone 假定所有 module state 都是 Tensor | Tensor 执行 detach/clone，其他 checkpoint-safe extra state 执行 deepcopy；Stage 5 再收窄可持久化值域 |
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

## Stage 4.1 实施检查点：成熟依赖原生 Provider 与 params 直传

状态：实现、针对性验证、独立代码审查与修复均已完成；checkpoint 格式为 v7，逻辑提交
主题为 `Stage 4.1: Reuse native PyTorch optimizers and schedulers`。

Stage 4 完成后重新检查 Registry 边界，发现当前实现把 Adam、AdamW 和多种标准
PyTorch LR scheduler 逐项注册，同时在本地配置 reference 复制它们的完整构造参数。
这种做法没有增加 Stochaflow 语义，却使每个 PyTorch 新实现、签名变化和默认值变化都
变成框架维护工作。

成熟框架调研进一步表明，上一版为保留 `auto` run length 而提出公共
`LRSchedulerBuilder` 仍然过重。Hydra 使用 target 加构造 kwargs，并允许调用方注入运行时
依赖；LightningCLI 使用 `class_path + init_args`，把模型 parameters 和 optimizer 在实例化
边界注入；Lightning 只把 `interval`、`frequency`、`monitor` 等训练生命周期放在 scheduler
外层；Transformers 的 `get_scheduler()` 则是具有明确 `num_training_steps` 契约的领域工厂，
不是从任意 constructor 字段名推断语义。

据此修正确认规则如下：

1. 配置仍需记录实现选择与显式参数，以便 resolved config/checkpoint 可复现；“需要选择”
   不等于“需要在 Registry 重新声明”。
2. `torch.optim.<Class>` 和 `torch.optim.lr_scheduler.<Class>` 由 allowlisted native-provider
   resolver 直接解析，分别验证 PyTorch `Optimizer` 与当前 Trainer scheduler contract；
   不允许借此任意 import 其他 Python module。两个前缀也是 Registry 保留 namespace，扩展
   注册时直接拒绝占用。
3. Core 将 trainable parameters 或 optimizer 作为位置参数注入，并将配置 `params` 原样作为
   constructor kwargs 传递；不复制上游签名、默认值或本地参数 dataclass。
4. `REGISTRIES.optimizers` 与 `REGISTRIES.lr_schedulers` 分别只保留给第三方
   `Optimizer`/`LRScheduler` 子类，并要求与 native provider 使用相同构造协议；删除公共
   `LRSchedulerBuilder`、context 和混合 callable factory。
5. scheduler 的 `interval=step|epoch` 仍属于核心训练 lifecycle，因为 PyTorch scheduler
   不知道调用方希望以 batch 还是 epoch 为单位推进。当前自动循环对 optimizer 和 scheduler
   都只调用无参数 `step()`；需要 closure 的 optimizer 或需要 metric 的 scheduler 暂不伪装
   成兼容组件，真实需求出现时再定义明确 lifecycle contract。
6. 本地文档不复制上游 class 的完整参数表，只记录 allowlisted namespace、core 注入参数、
   Stochaflow lifecycle 字段和上游链接。
7. `OptimizerConfig` 与 `ComponentConfig` 具有完全相同的 name/params 结构且没有独立行为，
   因而删除并直接复用 `ComponentConfig`；`LRSchedulerConfig` 因拥有 interval lifecycle
   字段而保留。factory 深复制 params 后传入 constructor，不修改调用方 mapping。
8. `T_max`、`total_steps` 和 `num_training_steps` 等只属于具体 constructor；首版必须显式
   配置，不支持 `auto` sentinel，也不根据参数名猜测 run context。内置 warmup-cosine 若
   保留，应是显式接收 `warmup_steps`/`total_steps` 的真实 `LRScheduler` 子类。
9. Registry 注册期只证明基类关系；constructor compatibility 在构建期通过统一调用验证并
   保留原始异常链。optimizer 与 scheduler 构造后分别使用其 bound
   `inspect.signature(component.step).bind()` 验证当前 Trainer 所需的零参数调用：默认参数与
   `*args` 合法，必需参数或无法可靠取得 signature 的实现拒绝。scheduler 还必须保留 core
   注入的同一个 optimizer。

配置中的 `params` 只是 Stochaflow 对 constructor kwargs 的统一命名。PyTorch optimizer
构造器的首个 `params` 是待优化 parameter iterable，运行时 `.defaults`/`.param_groups` 是
对象状态；scheduler 也没有跨实现统一的参数描述字段。因此“直接读 PyTorch params”在这里
应理解为原样调用 `Class(runtime_dependency, **configured_kwargs)`，而不是从对象状态或反射
结果生成一套新的 Stochaflow schema。resolved config 保存显式 kwargs；未显式写出的上游
默认值由锁定的 PyTorch 版本决定。

这里拒绝四个替代方案：继续逐项 alias 会复制 PyTorch namespace；任意 class-path import
扩大可信代码边界；通过 `inspect.signature()` 生成稳定 constructor schema 会把 best-effort
反射误当成上游契约；公共 run-aware Builder 则只为 `auto` 糖建立第二套 scheduler 构造体系。
对已构造 bound `step()` 做零参数 `bind()` 是消费方窄调用契约验证，不承担 constructor
schema 或字段推断职责。

[PyTorch Optimizer 文档](https://docs.pytorch.org/docs/stable/optim.html)将 optimizer 定义为
接收待优化 parameters 并通过 `step()` 更新它们的有状态对象；
[LRScheduler](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LRScheduler.html)
接收 optimizer，提供 `step()`、`state_dict()` 和 `load_state_dict()`，其 state 不重复保存
optimizer。这与 Stochaflow 当前由 core 注入 parameters/optimizer、分别 checkpoint 两份 state
的 lifecycle 一致，因此无需为每个标准实现增加包装层。

[Hydra instantiate](https://hydra.cc/docs/advanced/instantiate_objects/overview/)证明 target、
constructor kwargs 与调用点依赖注入可以保持分离；
[LightningCLI](https://lightning.ai/docs/pytorch/latest/cli/lightning_cli_advanced.html)采用
`class_path + init_args`，同时明确 signature parsing 只能 best effort；
[Lightning optimization](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)
将 scheduler instance 与 `interval`/`monitor` 生命周期元数据分开；
[Transformers optimization](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
对确实需要总步数的内置 factory 使用显式 `num_training_steps`，而不是通用字段名推断。

该原则不能机械扩展为任意 `torch.nn` class 都自动满足 Stochaflow Model、Objective 或其他
语义契约；只有依赖的原生行为与消费方所需 contract 完全一致时，才使用 native provider。

Stage 4.1 不实现 closure-required optimizer、metric-driven scheduler、多个 scheduler、
scheduler chaining 配置图、任意 class-path import、配置插值或通用 run-aware factory。
绑定后的 `step()` 必须可无参数调用；需要 closure 或 validation metric 的实现等待真实
lifecycle 设计。

Resume 状态边界已确认：当前没有需要兼容的用户 checkpoint，因此不增加旧格式迁移、
config/state 合并、reset flag 或兼容层。
[Optimizer state_dict](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.state_dict.html)
的 parameter groups 包含 learning rate、weight decay 等超参数；
[LRScheduler state_dict](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LRScheduler.html)
保存除 optimizer 外的实例属性。因此 `resume` 明确定义为恢复同一训练的完整
optimizer/scheduler state；更换 optimizer、scheduler 或超参数属于基于权重的新训练，不是
resume。当前未发布 checkpoint 直接记录并校验 optimizer/scheduler concrete class identity，
同时严格校验 scheduler state 的存在性；加载时先恢复 scheduler state、再恢复 optimizer
state，以符合 PyTorch 契约。格式直接升级到 v7 并拒绝 v6 草案；Stage 4.1 不提前实现
weights-only workflow。

逻辑提交主题：`Stage 4.1: Reuse native PyTorch optimizers and schedulers`。

## Stage 5 实施检查点

状态：实现、聚焦验证、独立审查与问题修复均已完成；checkpoint 格式为 v8，逻辑提交
主题为 `Stage 5: Add entry-point extension projects`。

Stage 5 初版计划曾经提议 `stochaflow.project.yaml`、
Stochaflow `--project`、source-root `sys.path` 激活与 config/checkpoint 祖先发现。
这是一个过度设计：它复制了 Python packaging 和包管理器已有的项目、环境与 import
职责，还将脚手架便利错误地扩张成一套核心运行时概念。

修正后的用户流程只有：

```text
stochaflow init my-project
→ cd my-project
→ 用用户选择的包管理器安装当前 distribution
→ 修改已生成的 extension 与 experiments/example/train.yaml
→ stochaflow train --config experiments/example/train.yaml
```

最终实现遵守以下边界：

1. scaffold 生成普通、可构建、可发布的 `src` 布局单 distribution 多实验 Python repo；
   它不定义 Stochaflow workspace，也不约束 scope 外实验的技术或目录。
2. `pyproject.toml` 用 `[project.entry-points."stochaflow.extensions"]` 暴露一个纯 module
   聚合入口；模块导入后由 decorator 直接注册组件，不增加 registration callback。模板
   生成一个入口，但协议允许同一 distribution 发布多个唯一 entry-point name。
3. `ExtensionsConfig.plugins` 选择 entry-point name：省略 `extensions` 默认不激活第三方
   插件，显式 `plugins: null` 才发现当前环境全部插件，空列表同样禁用第三方插件，非空
   列表精确选择。全量发现是显式 opt-in；`init` 直接把自身插件名写入完整示例配置，避免
   环境变化隐式改变生成项目的运行。
4. extension distribution、Stochaflow CLI 与任务依赖必须安装在同一 Python environment；
   pip、uv、conda、Poetry 或 PDM 均可管理该环境，Stochaflow 不替用户安装或切换环境。
5. Stochaflow 不提供 `--project`、项目发现、source-root 激活、`sys.path` 注入或
   `ProjectManifest` API。相对路径继续按进程 cwd 解析；模板明确要求从 repo root 运行。
6. `init` 只生成文件，不运行包管理器、不创建环境且不覆盖非空目录。
7. 默认 scaffold 只生成一个可运行的 DataBuilder + Model + TrainingBuilder/Strategy +
   direct SamplingBuilder 纵向示例。它不生成 Process/Sampler 空壳，也不生成或命名任何
   scope 外实验。
8. CLI reference generator 需要支持 `init` positional 参数；模板必须作为 package data
   同时进入 wheel/sdist。

插件 provenance 与恢复策略也在构建前固定：

- resolved config 保存确定的插件名列表；checkpoint 另外保存 name、distribution、version
  和 target；checkpoint-only resume/sampling 只恢复这份列表，不自动加载后来安装的插件；
- name、distribution 或 target 改变是插件身份错误，始终失败；
- 只有 version 改变时允许知情继续：交互 CLI 汇总 warning 后一次询问，默认 No；非交互
  默认失败；`--force-extension-version-mismatch` 可显式跳过询问；
- library API 永不 prompt，默认抛出类型化错误，调用方必须传入显式 allow policy；
- 接受的 expected/current version 和接受方式写入 manifest/checkpoint，专用 force 不绕过
  Registry、state 或其他兼容性检查。

配置解析和插件代码执行必须是两个阶段：先无副作用地解析配置、读取 entry-point metadata、
完成 selection/provenance 预检，再由 CLI 或显式 library policy 决定是否导入。Prompt 只属于
CLI；library API 永不读取 stdin。插件激活集合在一个 Python 进程内固定：相同 selection
重复激活幂等，不同 selection 明确失败，不尝试从全局 decorator Registry 卸载组件。CLI 的
一次 invocation 是标准隔离边界；需要运行不同插件集合的嵌入式调用方应使用不同进程。

config/checkpoint 权威规则是独立的 CLI input-resolution 问题，不与“项目发现”绑定。
sampling 已收口为：checkpoint-only 使用保存配置，显式完整 config 整体成为 base，
sampling-only 文件只替换 checkpoint base 的 `sampling` 与显式 `extensions`；插件列表不做
隐式追加。若外部 config/overlay 复用 checkpoint 中同一 entry-point name，则仍校验该名字的
distribution/target identity，并将 version 差异纳入一次性确认；新增或删除的插件不进行
集合相等比较。本次实际加载的 provenance 成为新 artifact/checkpoint 的基准，历史差异只作为
acceptance audit 保存。

训练 resume 的 config 规则已最终确认。Stage 4.1 已确认 full resume 恢复完整
optimizer/scheduler state，而 PyTorch state 会覆盖 `lr`、`weight_decay`、`T_max` 等构造值；
因此 `train --resume CHECKPOINT` 只表示 checkpoint config/state 权威的 strict full resume，
并与外部 `--config` 互斥。用新 config 加载旧模型权重属于独立 weights-only warm-start，
不恢复 optimizer/scheduler/epoch，本 Stage 不实现该入口。

该严格边界不适用于 sampling。Sampling 不恢复 optimizer/scheduler/epoch，可由本次外部
sampling config 自由改变样本数量、shape、Builder/Sampler、solver 参数、trajectory、writer
和 raw/EMA 选择；checkpoint 在该路径只提供所需资产 state，实际可加载性由 state contract
验证。

### 成熟设计调研

- [PyPA plugin discovery](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-package-metadata)
  将 entry point 作为已安装 distribution 发布插件元数据的标准机制；
  [Python `importlib.metadata`](https://docs.python.org/3/library/importlib.metadata.html#entry-points)
  提供按 group 查询和加载 entry point 的标准库 API。
- [pytest plugin discovery](https://docs.pytest.org/en/stable/how-to/writing_plugins.html#plugin-discovery-order-at-tool-startup)
  证明 entry point 适合环境级插件发现，也说明全量 autoload 会使运行依赖当前环境；因此
  Stochaflow 的生成配置显式列出自身插件，而不是依赖全量发现。
- [Django applications](https://docs.djangoproject.com/en/stable/ref/applications/) 使用显式
  installed-app list 激活已安装组件，支持“packaging 负责可发现、配置负责本次启用”的边界。
- [Hydra terminology](https://hydra.cc/docs/1.3/advanced/terminology/) 区分 Primary Config、
  Input Config、Overrides 和最终 Output Config；CLI override 修改单一组合结果，而不是让
  多份完整 config 互相做 equality comparison。Hydra 还把本次最终 config 与
  [overrides 保存到输出目录](https://hydra.cc/docs/1.2/tutorials/basic/running_your_app/working_directory/)，
  对应 Stochaflow 的 resolved config/checkpoint snapshot。
- [PyTorch Lightning checkpointing](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html)
  将 full resume 与 model load 区分：`Trainer.fit(..., ckpt_path=...)` 恢复完整训练状态；
  `load_from_checkpoint()` 默认使用 checkpoint hyperparameters，但显式 kwargs 可以覆盖。
- [Diffusers ConfigMixin](https://huggingface.co/docs/diffusers/api/configuration) 从保存的 config
  构造对象，并让显式 kwargs 覆盖同名配置；`save_pretrained()/from_pretrained()` 同时保存/
  恢复组件和配置，不要求另一份 config 与保存配置完全相等。
- [Accelerate checkpointing](https://huggingface.co/docs/accelerate/main/en/basic_tutorials/migration#state)
  明确 save/load state 用于同一训练程序和环境；其 ProjectConfiguration 管理 checkpoint
  存储位置，不成为另一份模型配置。
- [uv workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/) 与其他包管理器的
  workspace 都属于特定环境工具能力，不应成为 Stochaflow scaffold 或 runtime 的前提。

据此采用：**standard entry points + explicit activation + one authoritative base config +
runtime state validation**。sampling 的显式完整 config 选择组件图，checkpoint 只提供 state；
训练 full resume 不套用这句规则，因为 optimizer/scheduler state 本身包含超参数。
兼容性由插件身份、严格 state dict、具名资产、
optimizer/scheduler/EMA 和 format contract 验证，而不是完整 config equality。

训练 resume 继续采用“一次 invocation 一个 timestamped run”：checkpoint config 在旧 run
的同级目录创建新 run，`--output-dir` 可以改变新的 output root。训练 state 连续，日志/
artifact 不原地覆盖，并在 checkpoint metadata 记录 `resumed_from`。这避免为了原地追加而
把 logger-specific resume 行为塞进 Trainer 或通用 Logger contract。外部完整 config 与
`--resume` 明确互斥。

### 独立审查与修复

Stage 5 实现完成后进行只读独立审查，并将影响安全发布、恢复正确性或配置权威的 finding
全部在本 Stage 收口：

| Finding | 修复结果 |
| --- | --- |
| scaffold 的 staging pathname 在 rename/并发替换后仍可能被清理，存在 TOCTOU 删除外部对象风险 | 成功发布后不再触碰旧 staging pathname；失败清理绑定本次对象身份或 descriptor。现有空目录只在具备安全 descriptor-relative 原语的平台支持，否则写入前拒绝 |
| strict resume 可能把 sibling `best.pt` 中另一 run 或未来 epoch 的权重作为历史 best | 同时校验 resolved config、extension provenance、epoch、metric、monitor 和 mode；被未来训练覆盖的 mutable best 明确拒绝 |
| 新 run 继续依赖父 run 的 best checkpoint | 先用当前资产 topology 验证，再以当前 config/provenance 原子物化 inherited best 到新 run |
| checkpoint CPU 预加载后恢复的 EMA state 可能仍停留在 CPU | 完成 restore 后显式迁移 EMA state 到 Trainer device |
| checkpoint 与 inherited-best 写入可能暴露半写文件 | 集中使用同目录临时文件与 `os.replace()` 原子发布，并在失败时清理本次临时项 |
| strict resume 重新从 experiment seed 开始随机流，并宽松接受缺失 progress | v8 保存 data-only Python/NumPy/Torch RNG snapshot；恢复前要求合法 `epoch`/`global_step`，在全部 state 校验后恢复适用随机流，并用随机训练的中断/续训等价测试验证 |
| sampling 的外部 config、overlay 与 checkpoint provenance 权威可能漂移 | 完整外部 config 整体权威；overlay 只替换显式 sampling/plugin selection；checkpoint 只提供 state，同名插件仍校验 identity/version，本次实际 provenance 成为新基准 |

scaffold 的安全保证有意按平台能力分层：不存在目标可通过同级 staging rename 发布；现有
空目录只在能安全执行 descriptor-relative exclusive create/rollback 时支持。框架不以一个
不可兑现的“所有平台对已有目录原子替换”承诺掩盖文件系统差异。

### Rejected alternatives

- **Stochaflow project manifest/source-root 激活：** 重复 `pyproject.toml`、标准 packaging
  与 Python import，并引入 manifest/config 双重插件权威。
- **`extensions.modules` 作为长期公共协议：** import path 缺少 distribution identity 与
  version provenance，checkpoint 也无法在导入扩展代码前完成版本预检。
- **只要安装就始终自动激活全部插件：** 新安装一个无关 distribution 就可能改变旧运行；
  保留 `plugins: null` 作为显式语义，但 init 配置写入确定列表。
- **entry point 指向 callback 或插件对象：** decorator 已经是注册入口，第二套 callback
  生命周期只会增加重复抽象；首版 target 必须是纯 module。
- **train/sample 自己接受 `--project`：** 迫使 Stochaflow 解释环境和本地源码路径，也不能
  让全局 CLI 跨 Python environment 导入另一环境的 extension。
- **init 时自动 editable install：** 会未经请求修改环境，并将包管理失败与文件生成混成
  一个事务。
- **强制使用 uv 或生成多 package monorepo workspace：** 把工具偏好与一个可能场景固化为
  框架约束；普通单 distribution repo 已允许用户组织多个实验并自行演进。
- **生成某一科学领域或其他模型 family 的目录：** 示例场景不应成为项目模板符号；模板
  只生成最小 Stochaflow vertical slice。
- **版本有差异即永远拒绝：** 对 patch/rebuild/editable 工作流过严；warning + opt-in 保留
  用户决定权。反之通用 `--force` 权限过宽，因此只提供 version-scoped flag。
- **完整 config equality/compatibility comparison：** 把显式 override 误当成冲突，并且
  无法比实际 state contract 更准确地判断可恢复性。
- **为每个 Registry 生成模板：** 代码量大、学习路径噪声高，并重新制造“每种方法都需要
  Process/Dynamics/Sampler”的错误抽象。
- **原地续写旧 run：** 需要所有 logger、artifact writer 和第三方 tracking backend 共享
  一套 append/resume run-id 契约；当前没有这一真实公共能力，强行实现会污染 lifecycle。

### Trade-offs 与已知限制

标准 packaging 使 init→install→run 不绑定包管理器，也使 extension 可测试、可发布、可由
用户自己的 lockfile 锁定。代价是训练前必须把项目和 Stochaflow CLI 安装到同一环境；
全局 pipx/uvx CLI 不能直接看到另一个隔离环境中的 extension。checkpoint 不携带扩展源码，
distribution version 也无法检测 editable code 在版本号不变时的修改。核心继续将任意
component `params` 当作不透明 mapping，因此不猜测或重写其中的路径；生成项目以 repo root
为 cwd 作为明确运行契约。

Stage 5 最终计划复审发现：若 checkpoint 继续允许任意 pickle `extra_state`，runtime 为读取
内嵌 config/provenance 而执行 `torch.load(weights_only=False)` 时就可能动态导入 extension
class，使“版本确认后才导入插件”成为不可实现的承诺。由于没有任何用户 checkpoint 需要
兼容，v8 直接将 payload 收窄为 PyTorch `weights_only=True` 默认支持的 Tensor、primitive
与普通 container；保存边界递归拒绝 custom class、custom tensor subclass 和任意 pickle
global，extension 不得在预检前扩展 safe globals。自定义额外状态必须编码为该数据值域。

该决定取代 Stage 4 草案中“任意可 pickle extra state”的宽松表述。拒绝采用 sidecar/双文件
envelope，因为它降低 checkpoint 单文件可移植性；也拒绝仅降低 preflight 文档承诺，因为那会
让版本提示发生在扩展代码已经执行之后。`weights_only=True` 不是 DoS 或资源隔离沙箱，这一
限制继续在公开文档说明。

strict resume 的 best continuity 还有一个刻意保留的限制：每个 run 只有一个 mutable
`best.pt`。恢复 latest/current checkpoint 时可以校验并本地化匹配的 inherited best；若用户
选择更早的历史 epoch，而 sibling `best.pt` 已被后续 epoch 覆盖，框架会明确拒绝，不能从现有
文件可靠重建“当时的 best”。支持任意历史点需要 versioned immutable best 或把 best snapshot
嵌入每个 checkpoint，会增加存储和保留策略，等待真实需求后再设计。匹配 best 一旦物化到新
run，后续 resume 和 sampling 不再依赖父目录。

strict resume 的随机状态边界同样收口为 epoch-boundary process-global streams：v8 以
weights-only-safe 的 list/primitive/Tensor 编码保存 Python `random`、NumPy global
`RandomState`、Torch CPU 与当时可用的 CUDA RNG。恢复前完整解析全部字段，selected state 与
inherited best 恢复完成后才一次性恢复 RNG。普通 checkpoint load 不修改全局 RNG；
sampling 不恢复 checkpoint RNG snapshot，而是按 `sampling.seed`（为 `null` 时使用
`experiment.seed`）重新初始化 Python、NumPy 与 Torch 全局 RNG。CPU 目标
忽略 CUDA snapshot，CUDA 目标仅在保存了 CUDA state 时要求 device count 兼容，因此 device
override 可运行，但跨设备/拓扑不保证 bitwise continuity。DataBuilder、loader iterator、
worker、sampler 和用户 generator 属于扩展 runtime，不进入核心 checkpoint；自定义随机 loader
必须通过 seed + epoch/`set_epoch` 自行满足可重建性。

TrainingDiagnostic 是观察性扩展。Trainer 在每个 diagnostic lifecycle callback 外隔离并恢复
Python、NumPy 和 Torch 全局 RNG；这同时避免 strict resume 重新触发 `on_fit_start` 时改变
后续训练随机流。需要跨 callback 延续随机状态的 diagnostic 应持有自己的 generator。

checkpoint 只冻结 state、resolved config 和插件 provenance，不冻结 extension 源码或 wheel。
version 未变化的 editable 源码修改无法检测；若修改最终导致 state 或行为不兼容，由严格 state
contract 或扩展运行错误暴露，框架不承诺为任意用户代码演化提供兼容层。

### 验证与逻辑提交

Stage 5 聚焦验证覆盖配置与插件预检/激活、CLI 版本策略、v8 checkpoint-safe state、严格
resume 与 inherited-best 连续性、scaffold 安全边界、sampling config/provenance 权威、
builder/Trainer runtime，以及隔离环境中安装 Stochaflow wheel、执行 `init`、安装生成项目
wheel、短训练、连续恢复和 checkpoint-only sampling 的端到端路径。常规门禁为 Ruff、Pyright
与配置 reference 生成/检查。

完整 pytest、build、Sphinx 和其他 CI 静态检查留到整个 feature 分支合并验收，不计入本
Stage 的日常收口声明。

逻辑提交主题：`Stage 5: Add entry-point extension projects`。

## Stage 6 检查点

状态：实现、参考主机基准、针对性验证和独立审查均已完成。

### 容量决策

- 保留现有 `SamplingBuilder.run() -> SamplingOutput` 和 Writer `write()` 公共
  lifecycle，不为理论上的大输出预建 streaming/event bus；
- 内置 Standard Builder 会将 writer-ready Tensor 转存到 owning CPU state；
  `SamplingBatch.samples: Any` 公共 contract 不强制自定义 Builder 的设备；
- DFSR 真实主 profile 是 1272 个 `float32 [3,256,256]` final state、trajectory
  关闭。在 16 GiB macOS arm64 参考主机上，5 个 fresh measured repeat 的最大
  peak RSS 为 2.160 GiB（13.50%），不触发 70% 容量门；
- 全量 1272-sample/31-state dense trajectory 的 raw `SamplingOutput` 为 29.81 GiB，
  当前 Tensor writer 结构性峰值下界为 87.57 GiB，明确不在当前支持边界；
- trajectory 只作为独立 preview：`num_samples <= 8`、`every_steps >= 10`、
  accepted steps 不超过 40。Tensor preview 与 high-entropy PNG/grid/GIF preview 均已
  使用 5 个 fresh repeat 实测；
- 只有单 batch 可容纳而真实 final artifact 仍超过主机预算的证据，才触发
  同步、有背压且具备 abort/cleanup/最终发布语义的最小 batch lifecycle 设计。

### 工具与生产收缩

`tools/benchmark_sampling_capacity.py` 使用受版本控制的 profile，每个 repeat 在
fresh subprocess 中调用真实 `SamplingOutput` 验证和内置 Writer。工具记录
lifetime RSS high-water、CUDA allocated/reserved（可用时）、artifact bytes、wall
time、host/disk 环境和 tool/profile SHA-256；默认容量/磁盘 preflight 与 worker
timeout 防止误执行超大 profile。CI 只跑 tiny contract，已提交的主机结果由
hash/statistics 测试防止静默过期。

sampling input 在验证完整 v8 checkpoint 后只保留私有 inference view：raw/EMA
model、可选 Process、config/metadata 和 format。optimizer、scheduler、Objective、
training assets、resume EMA shadow、RNG 与训练进度不再与全量 sampling output
重叠常驻。这个窄 view 不是可传给通用 checkpoint restore 的完整 payload。

### 验证与逻辑提交

聚焦验证覆盖 profile/schema/formula、fresh worker/cleanup、preflight、结果哈希与统计、
Tensor/image Writer、checkpoint inference view、raw/EMA/Process 恢复和跨 batch owning
state。常规门禁为 Ruff、Pyright 和聚焦 Pytest；文档以 Sphinx `-W` 构建。
完整分支验收留到 feature merge 前的最终循环。

逻辑提交主题：`Stage 6: Validate sampling artifact capacity`。

## Stage 7 实施检查点（已完成）

状态：两个独立 reference distribution、聚焦验证、wheel/entry-point CLI 验收、真实
Physics batch 容量验证、独立审查与问题修复均已完成；`src/stochaflow/**` no-change gate
通过，Stage 8 待实施。

### 发现的架构冲突

原 Stage 7 同时要求 Physics 案例复用内置 DDPM/DDIM，又要求覆盖论文中的
physics-guided DFSR。两者不能在不限定语义的情况下同时成立：论文的 guided update 在
每个 accepted DDIM transition 后显式执行 `x_next = x_next - dx`。`dx` 虽由当前 state 的
PDE residual gradient 产生，但它被施加的位置属于 solver transition，而不是
`GaussianDenoisingDynamics.predict()` 返回的 prediction parameterization。

把该 correction 隐藏进通用 Gaussian prediction 会改变算法含义，也会让 DDPM/DDIM 在未知
前提下看似可互换；把 physics callback 加入内置 Sampler 则直接违反 OCP。因此 Stage 7 在
确认前暂停编码。

### 推荐边界

1. baseline DFSR/SDEdit 使用项目 SamplingBuilder 构造 partial-noised initial state，复用
   内置 Discrete Gaussian Process 和 DDPM/DDIM；两者的切换只改变项目 sampler 配置；
2. exact physics-guided DFSR 使用项目私有窄 Dynamics capability 与注册的
   `GuidedDDIMSampler`，复用公开的 DDIM schedule/transition primitive，并由该 Sampler
   拥有 post-transition correction；Stage 3.1 完成后核心和内置 DDIM 无需再改变；
3. normalization/PDE constants 由项目 model 的构造配置和 buffers 持有。DataBuilder 返回
   raw field，Strategy 与 SamplingBuilder 通过 model 的项目私有接口复用同一状态；不向
   DataLoaders 增加 metadata；
4. Physics 与 distillation 作为两个独立可安装 distribution 验收 entry point isolation，
   但这不构成对用户 repo、monorepo 或包管理器的约束；
5. Stage 7 公共 API 变化为零，且 `src/stochaflow/**` 是明确的 no-change gate；
6. tiny deterministic E2E 证明算法和 lifecycle，真实 256² batch 证明 shape/device/capacity。
   没有训练权重时不得把随机模型结果描述为论文精度复现。

### 其他已确认缺口

- DFSR 训练只使用高分辨率三连帧，LR/sparse observation 只在 reconstruction 阶段出现；
  conventional paired image SR 是另一项内置 recipe，不能作为 DFSR 的实现；
- sampling runtime 在 `torch.no_grad()` 下调用 Builder。项目的 PDE gradient helper 可以在
  最窄范围使用 `torch.enable_grad()`，随后 detach；该行为需要回归测试，不需要核心变更；
- Writer 只序列化 reconstruction 与 Builder/evaluator 已计算的 metric summary，不拥有
  PDE residual 数学；
- distillation teacher 使用项目定义的普通 PyTorch state 初始化；resume 后同名
  `training_assets_state_dict` 是 runtime state 权威。扩展不读取 Stochaflow 私有 checkpoint
  payload；
- Stage 6 只证明 final artifact 容量，Stage 7 必须单独记录真实 model、condition、residual
  autograd 与 input 共存时的单 batch host/accelerator peak。

完整实施范围、测试矩阵和 rejected alternatives 已写入
[实施计划的 Stage 7](../custom-code-extension-support-plan.md)。实施结果证明 baseline
DDPM/DDIM、项目级 guided DDIM、frozen-teacher resume 和 student-only sampling 均可只用
公开扩展契约完成；任何新的核心契约需求仍必须退回 Design。

真实容量验收在 2026-07-22 使用本地 PhysicsNeMo mmap 数据
`[40, 320, 256, 256] float32`、PyTorch 2.11.0、Apple-silicon MPS、batch
`[1, 3, 256, 256]` 和生产时间域 `T=1000, t=240`。一次 train
forward/backward、两步 baseline DDIM 和两步 guided DDIM 均成功；进程累计 RSS 高水位为
392,019,968 bytes，MPS current allocation 在训练后为 16,060,416 bytes、两条采样后均为
6,086,400 bytes，MPS driver allocation 约 1.15 GB。MPS 没有该工具可重置的 peak API，
因此这些 MPS 数字是同步后的 current/driver allocation，不冒充峰值；两步 smoke 也不
代表完整 30/40-step latency 或论文精度。

## Stage 3.1 离散 Gaussian primitive 修正（已完成）

维护者复审 `codex/refactor-diffusion-schedules` 与当前 Stage 3 后确认：完整
`Sampler.sample()` 与具体算法的单 transition API 并不冲突。禁止的是所有 family 都必须
实现的 universal `step()`，而不是离散 Gaussian family 自己的 DDPM adjacent transition、
DDIM selected-pair transition 或 DDIM schedule resolver。

当前 DDIM 将 prediction、selected-pair 数学、RNG、loop 与 observer 全部内联在
`DDIMSampler.sample()`，并把 schedule resolver 私有化；DDPM 虽保留 Process posterior
数学，也丢失了一个内聚的公开 adjacent transition。exact DFSR 因而只能复制算法。该结果
否定了 Stage 7 的原始 no-core-change 前提，并构成发布前必须修复的 OCP 回归。

已批准的修正边界：

1. framework root 仍只有完整 `Sampler.sample()`；不增加 universal step、Dynamics 方法、
   hook、callback 或 Registry；
2. 离散 Gaussian family 公开 transition mean/standard-deviation result、DDPM adjacent
   transition、DDIM selected-pair transition 和 DDIM schedule resolver；
3. transition 不调用模型、不发送 observation、不应用任务 correction；完整 Sampler 委托
   primitive 并继续拥有 RNG、loop 与 lifecycle；
4. Dynamics 负责 source-state model prediction。项目可以定义更窄的 guided Dynamics 同时
   产生 `GaussianPrediction` 与 correction；guided Sampler 在共享 DDIM transition 后应用
   correction；
5. 公开 Gaussian prediction normalization 与 training-target 小型 helper，不建立把 batch、
   Objective、diagnostics 再次捆绑起来的“大训练 helper”；
6. 稳定 extension 入口补充 family primitive、`PerSampleObjective` 和
   `compute_objective()`；Sampler 统计改称 `num_dynamics_evaluations`；
7. Stage 3.1 完成并形成独立 checkpoint 后，Stage 7 才重新启用 `src/stochaflow/**`
   no-change gate。

明确拒绝恢复 model-owning diffusion、旧多套 sample/trajectory API、Observer state mutation、
target-aware 通用 Dynamics，以及把 physics 参数加入内置 DDIM。

实现、聚焦回归、Ruff、Pyright 与两轮独立审查均已完成。Stage 7 重新启用
`src/stochaflow/**` no-change gate，项目扩展必须只依赖本检查点公开的 family primitive。

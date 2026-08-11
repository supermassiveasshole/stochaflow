# 让运行依赖显式，让训练诊断只拿到真正需要的能力

> 文档类型：架构维护记录
>
> 排期状态：不参与排期
>
> 当前可用性：运行组装与 Training Diagnostic 窄能力都已收口；用户命令和配置保持不变

2026-07-31 的一份 Clean Architecture 草案提出了很多方向。它后来只提交在未合入主线的
远端审查分支 `codex/metrics-system`（提交 `af13619`），没有进入 `main` 或当前排期。按当前
代码重新核对后，没有理由恢复那份“大一统重构”：独立 Evaluation 已经完成，当前按领域
划分的 package 也比重新建立 `foundation/contracts/application/builtins` 等通用目录更容易理解。

不过，草案里有一个判断仍然重要：**一段代码应当明确说明自己依赖什么，而不是因为拿到了
一个大对象或导入了一个工具模块，就能碰到整个运行时。** 2026-08-10 开始这轮维护时，仓库有
两处具体事实说明这件事当时还没有完全做到。

第一处曾是运行组件的组装。`stochaflow.utils.factory` 直接导入 Process、TrainingBuilder、
Trainer、Diagnostics、checkpoint 和 logging，并在模块导入时调用
`load_builtin_components()`。它实际承担的是应用启动和依赖组装，而不只是一个普通工具。
训练、采样、推理和 Evaluation 又从不同位置调用这里的函数，因此很难从 package 名称判断
一次操作真正会加载和构造什么。下面“把对象组装放回具体操作的入口”一节保留了当时的推导，
随后的小节记录实际采用的结果。

第二处是训练诊断。`FitStartEvent`、`TrainBatchEndEvent` 和 `TrainEpochEndEvent` 曾把完整的
`trainer: Any` 交给 Diagnostic；Gaussian 诊断还会从 `Mapping[str, Any]` 中读取约定好的
字符串键。当时的诊断能够工作，但一个 Diagnostic 的真实依赖只能靠阅读实现猜出来。Trainer
以后只要改名或移动内部属性，Diagnostic 就可能在运行中才失败。它也与当前规范中“诊断应
声明自己需要的能力”这条规则不一致。

这两件事有同一个原因：组装边界没有把依赖收窄。解决它们不需要重写整个仓库，而是需要把
“谁负责创建对象”和“谁可以读取哪些运行状态”说清楚，并由测试守住。

## 先把已经完成和已经放弃的部分分开

本提案不会重新打开 Evaluation。用户现在已经可以独立评估 checkpoint 或预测文件包，框架
也会检查样本完整性并发布不可变结果。任务自己的 `EvaluationBuilder` 解释任务数据，核心
runtime 负责模块模式、指标状态、样本 ID 和结果发布；这是当前稳定设计。

旧草案设想过统一的 `PhaseEvaluationBuilder`、`PairedEvaluationBuilder`、
`GenerationEvaluationBuilder` 和 `GroupedGenerationEvaluationBuilder` catalog。当前没有两种
独立任务证明这些固定 profile 需要成为框架抽象，因此本提案不创建它们。AFHQ 继续通过自己
的 Builder 声明任务协议，同时复用通用 Evaluation runtime。

本提案也不把 Trainer 按旧草案列出的类名机械拆成六个对象。文件长或方法多不是独立重构的
理由。只有当某项职责必须通过一个窄接口交给 Diagnostic、Evaluation 或其他调用者时，才提取
对应协作者。

Extension 的启动耗时另有条件性性能复查负责。本提案不承诺降低导入时间，不设置毫秒预算，
也不把 lazy import 当作成功标准。如果依赖整理顺便减少了导入模块，只记录结果；不能为了
追求数字而改变 Extension 的对象身份、验证时机或错误保证。

## 当时怎样推导出具体操作入口

当时没有直接移动文件，而是先记录真实发生的事情。维护者用全新 Python 进程核对了：

- 只导入 `stochaflow.data`、`stochaflow.sampling` 或 `stochaflow.evaluation`；
- 显示根命令和各子命令帮助；
- 构造一次最小训练、采样和 Evaluation；
- 准备 Extension 激活，以及真正激活一个选中的 Extension。

这里记录的是“加载了哪些 Stochaflow package、Registry 是否发生变化”，不是性能排名。结果
成为改造前的行为基线，也暴露了哪些调用依赖偶然的 import 顺序。

随后把 `utils.factory` 中的组装代码交还给实际使用它的操作。训练入口负责创建完整的
训练协作，采样和 Evaluation 只构造各自需要的只读推理对象。当时用两条简单规则选择最终
实现位置：

1. `utils` 不再需要导入 Trainer、TrainingDiagnostic 或任务运行时来完成普通工具职责；
2. CLI 只解析参数并调用对应操作，不成为所有领域对象都能互相看见的第二个 factory。

内置组件仍然和外部 Extension 走同一个 Registry 检查与构造规则。差别只在于内置组件来自
框架发行包，外部组件来自用户明确选择的 entry point。内置注册必须由一次可定位、幂等的操作
初始化触发，不能继续要求调用者碰巧先导入某个聚合模块。初始化失败后应终止本次进程中的
初始化，不能留下部分 Registry 状态后继续运行。

最终决定保留 `stochaflow.utils.factory` 的现有调用方式作为薄转发，不发弃用警告，也没有删除
计划。框架内部的新代码不再依赖它；这个入口只负责兼容既有调用，不重新承担完整运行组装。

## 运行组装最后落在什么位置

2026-08-10 完成的实现没有增加新的用户命令，也没有用 lazy import 追求启动耗时数字。它只把
已有责任放回可定位的位置。`stochaflow._builtin_activation` 现在是唯一负责内置组件注册的
位置；导入普通公开入口时只定义类，不再顺手改写 Registry。训练、采样和 Evaluation 分别声明自己的
固定加载范围，并且都在外部 Extension 激活前完成内置注册。采样和 Evaluation 因而不会为了
构造只读推理对象而加载 Trainer 或 Training Diagnostics，也不会加载 Logger。

初始化由进程级锁保护。重复调用、两个范围重叠以及并发首次调用都只注册一次。如果模块导入、
基类检查或名称冲突失败，当前进程会保留第一次错误并拒绝继续初始化，提示调用者重启；已经发生
的 Registry 写入不假装能够安全回滚。进入已经核对输入的执行阶段后，只检查对应范围已经完成，不再
临时补导入。这使“内置组件先注册，随后才允许用户选中的 Extension 注册”成为可测试的顺序，
而不是依赖某个文件碰巧先被 import。

对象构造也分开了。`stochaflow._component_factory` 只负责 Model、Process 和 Objective 三种共享
声明；完整训练运行时只由 `stochaflow.training.composition` 组装。旧的
`stochaflow.utils.factory` 仍保留原有调用方式作为薄转发，但框架内部不再依赖它。CLI 解析
`sample` 或 `evaluate` 时也不会先加载训练 runner；只有真正选择 `train` 才导入训练运行时。
这些变化不改变配置、checkpoint、命令参数或 Extension 使用的 Registry。

## Diagnostic 最后只拿到明确需要的事实和能力

2026-08-11 完成的实现没有机械拆分 Trainer，也没有建立通用 Diagnostic schema。三个事件现在
只携带已经发生的事实：fit 开始时的 iterable 和恢复后的 step、成功 optimizer step 的 batch、
loss、step、epoch 与观察对象，以及 epoch 结束时不可修改的 metrics。事件不再携带 Trainer 或
完整 `TrainStepOutput`。Trainer 用同一个内部调用入口执行三个 callback，统一隔离 RNG、传播
异常，并要求 callback 返回 `None`。

Strategy 若要把任务特有的中间结果交给 Diagnostic，只需把一个普通 Python 对象放入
`TrainStepOutput.diagnostic_observation`。框架核心（core）会在一次真正执行 optimizer 更新的
累积窗口（optimizer window）成功后，原样转交最后一个小批次（microbatch）的对象，不读取
字段，也不建立跨算法家族（family）的 batch 或观察值格式。Gaussian 算法家族
使用自己的 `GaussianStepObservation` 和 `ClassConditionalGaussianStepObservation`，显式保存
已从反向传播图分离（detached）的时间、prediction、训练目标、clean sample、可选逐样本 loss
与类条件事实；其他算法家族可以定义完全不同的类型。

Diagnostic 在构造时通过 `DiagnosticBuildContext` 得到组件名、logger、输出目录和
`DiagnosticModelAccess`，并按需要求明确的 Strategy 或 Process 窄能力接口（Protocol）。缺失
能力会在训练开始前失败，YAML 不能伪造或覆盖 runtime 注入值。普通 Diagnostic 不必声明空的
cadence、artifact 或 failure-policy 配置；只有确实拥有这些行为的具体类型才声明它们。

需要临时调用当前模型的 Diagnostic 只使用 `DiagnosticModelAccess.evaluation()`。这个窄能力
不暴露 model、Trainer、optimizer、scheduler、checkpoint manager 或 EMA 对象。这个保护范围
（context）会隔离
全局 RNG、固定本次随机种子、启用 inference mode、选择 raw/EMA 权重、管理所有 module 的 eval
mode，并在退出时按模块的父子依赖顺序完整恢复。主体、EMA copy、mode 切换或某个恢复动作失败时，
其余恢复仍会继续；
状态恢复失败是不可降级错误，不能被 provider 的 `warn` 策略吞掉。

普通 `MetricUpdate` 继续走现有指标通道；训练期额外采样、重建和 artifact 仍属于 Diagnostic；
冻结 subject、data 和 protocol 后发布正式证据仍属于 Evaluation。这次收口没有改变模型选择、
checkpoint、配置 schema 或 CLI 行为。

## 用失败案例守住已经完成的边界

这次整理不是只移动 import。当前回归测试守住以下证据：

- AST 依赖检查能指出从工具/契约层反向导入训练 runtime 的具体文件和 import；
- package 依赖图没有因为迁移新增循环，重要 operation 的导入闭包由全新进程测试固定；
- 未运行显式初始化时，不会因为导入 facade 偶然注册一部分 built-in；
- 初始化重复调用是幂等的，wrong-base、重复名称和中途失败仍会立即失败；
- 一个独立自定义 Diagnostic 只使用声明的事件事实和能力，不导入或接收 Trainer；
- 现有 Gaussian Diagnostics 的指标、artifact、cadence、failure policy 和 RNG/mode 保护保持；
- 训练选择仍只消费 validation，Diagnostic 不能修改 optimizer、EMA、checkpoint 或 best/latest；
- 源码 checkout 与安装后的 wheel 运行相同的 train、sample、evaluate 和 Extension 测试。

测试不应规定整个仓库只能采用一份僵硬的 package 分层表。它只检查已经写进
`ARCHITECTURE.md` 的责任边界，以及本次迁移明确禁止的依赖方向。

## 最终结果

普通用户继续运行相同的 `train`、`sample` 和 `evaluate` 命令，配置、checkpoint 和
运行产物含义不变。Extension 作者不会获得第二套注册方式；Diagnostic 作者则能从类型和构造
参数看出自己可以使用哪些事实与能力。

代码层面可以直接确认：

- operation 组装不再以 `utils.factory` 作为所有领域共享的隐藏入口；
- built-in 初始化有唯一、显式、幂等且可测试的调用位置；
- Diagnostic event 和 provider contract 中不再出现完整 `trainer: Any`；
- 算法家族自己的（family-specific）观察值不再依赖未声明的跨模块字符串键；
- 当前 Evaluation、Sampling、checkpoint、Extension 激活和训练选择语义没有改变；
- 规范、架构文档和公开 Extension 文档只描述最终实现，不把本提案中的候选名字写成当前 API。

## 排期与后续边界

这两项是现有架构维护，不是等待排期的新产品，因此不占用 [`ROADMAP.md`](../../../ROADMAP.md)
的 `Next`。如果下一项工作会新增训练 Diagnostic、改变 Trainer 生命周期、增加 operation，
或扩展 built-in 注册范围，应沿用这里已经建立的窄边界，避免重新复制隐藏依赖。与这些边界
无关的小型维护和算法实现不需要额外排期许可。

当前规范与实现证据见：

- [`SPEC.md`](../../../SPEC.md) 的 Training、Evaluation 与 Extension contracts；
- [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) 的 package 责任归属与各操作组装入口；
- [`_builtin_activation`](../../../src/stochaflow/_builtin_activation.py) 的 built-in 注册生命周期；
- [Training composition](../../../src/stochaflow/training/composition.py) 的完整训练运行时组装；
- [`utils.factory`](../../../src/stochaflow/utils/factory.py) 的旧调用兼容转发；
- [Diagnostic contracts](../../../src/stochaflow/training/diagnostics/contracts.py)；
- [Training Strategy output](../../../src/stochaflow/training/strategy.py)；
- [Evaluation 已完成说明](../post-training-evaluation-support-plan.md)；
- [Extension 条件性性能复查](../extension-import-boundary-and-activation-latency-plan.md)。

本提案不改变 Physics/KD，不增加 Evaluation 功能，不实施 Extension 性能优化，也不改变当前
`ROADMAP.md` 的 `In progress: None` 与 `Next: None`。

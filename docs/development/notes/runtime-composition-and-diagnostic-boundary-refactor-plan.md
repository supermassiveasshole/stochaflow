# 让运行依赖显式，让训练诊断只拿到真正需要的能力

> 文档类型：架构维护记录
>
> 排期状态：不参与排期
>
> 当前可用性：`train`、`sample`、`evaluate` 和 Extension 现在都能使用；本提案不补产品功能

2026-07-31 的一份 Clean Architecture 草案提出了很多方向。它后来只提交在未合入主线的
远端审查分支 `codex/metrics-system`（提交 `af13619`），没有进入 `main` 或当前排期。按当前
代码重新核对后，没有理由恢复那份“大一统重构”：独立 Evaluation 已经完成，当前按领域
划分的 package 也比重新建立 `foundation/contracts/application/builtins` 等通用目录更容易理解。

不过，草案里有一个判断仍然重要：**一段代码应当明确说明自己依赖什么，而不是因为拿到了
一个大对象或导入了一个工具模块，就能碰到整个运行时。** 当前仓库有两处具体事实说明这件事
还没有完全做到。

第一处是运行组件的组装。`stochaflow.utils.factory` 直接导入 Process、TrainingBuilder、
Trainer、Diagnostics、checkpoint 和 logging，并在模块导入时调用
`load_builtin_components()`。它实际承担的是应用启动和依赖组装，而不只是一个普通工具。
训练、采样、推理和 Evaluation 又从不同位置调用这里的函数，因此很难从 package 名称判断
一次操作真正会加载和构造什么。

第二处是训练诊断。`FitStartEvent`、`TrainBatchEndEvent` 和 `TrainEpochEndEvent` 仍把完整的
`trainer: Any` 交给 Diagnostic；Gaussian 诊断还会从 `Mapping[str, Any]` 中读取约定好的
字符串键。现有诊断能够工作，但一个 Diagnostic 的真实依赖只能靠阅读实现猜出来。Trainer
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

## 把对象组装放回具体操作的入口

第一步不是移动文件，而是记录今天真实发生的事情。维护者应生成当前 package 导入图，并用
全新 Python 进程分别执行以下操作：

- 只导入 `stochaflow.data`、`stochaflow.sampling` 或 `stochaflow.evaluation`；
- 显示根命令和各子命令帮助；
- 构造一次最小训练、采样和 Evaluation；
- 准备 Extension 激活，以及真正激活一个选中的 Extension。

这里记录的是“加载了哪些 Stochaflow package、Registry 是否发生变化”，不是性能排名。结果
会成为迁移前的行为基线，也会暴露哪些调用仍依赖偶然的 import 顺序。

随后把 `utils.factory` 中的组装代码按使用它的操作归还给明确 owner。训练入口负责创建完整的
训练协作，采样和 Evaluation 只构造各自需要的只读推理对象。具体模块名可以在实施时选择，
但必须满足两条简单规则：

1. `utils` 不再需要导入 Trainer、TrainingDiagnostic 或任务运行时来完成普通工具职责；
2. CLI 只解析参数并调用对应操作，不成为所有领域对象都能互相看见的第二个 factory。

内置组件仍然和外部 Extension 走同一个 Registry 检查与构造规则。差别只在于内置组件来自
框架发行包，外部组件来自用户明确选择的 entry point。内置注册必须由一次可定位、幂等的操作
初始化触发，不能继续要求调用者碰巧先导入某个聚合模块。初始化失败后应终止本次进程中的
初始化，不能留下部分 Registry 状态后继续运行。

迁移期间可以保留 `stochaflow.utils.factory` 的薄转发函数，避免一次改动所有内部调用。新的
内部代码不得再增加对它的依赖；所有调用迁走并通过安装包测试后，再决定是在 pre-1.0 阶段
删除转发，还是把它保留为明确记录的兼容入口。

## Diagnostic 不再得到整台 Trainer

Diagnostic 在构造时应声明自己需要什么。例如，一个只记录 loss 的诊断只需要批次结束事实；
一个生成预览图的诊断还需要只读模型视图、临时切换 eval/EMA 的保护以及 writer；一个计算
Gaussian 重建结果的诊断需要该 family 明确提供的重建能力。它们都不需要得到 optimizer、
scheduler、checkpoint manager 和 Trainer 的其他可变状态。

实施时先为仓库现有 Diagnostic 列一张依赖清单。每项依赖必须归入以下一种来源：

- 事件事实，例如 epoch、global step、loss 和已经完成的 Strategy 输出；
- TrainingBuilder 在组装时注入的只读能力；
- Diagnostic 自己拥有的临时缓存或输出 sink；
- 明确禁止读取的训练状态。

然后用窄能力替换 `event.trainer`。事件只携带不可变事实；需要调用模型的 Diagnostic 在创建时
收到只读能力，而不是运行到 callback 才从 Trainer 上找属性。能力不存在或与任务不兼容时，
组装阶段就给出可操作错误，不能等到若干 epoch 后才失败。

Gaussian Strategy 与 Diagnostic 之间约定的字符串键也要收窄，但不能因此建立通用 batch 或
通用诊断 schema。Gaussian family 可以定义自己的 typed observation，其他 family 只实现自己
确实需要的契约。普通 `MetricUpdate` 继续走现有指标通道；训练期额外采样、重建和 artifact
仍属于 Diagnostic；冻结 subject、data 和 protocol 后发布正式证据仍属于 Evaluation。

当这些调用边界建立后，再判断 Trainer 内部是否需要提取小协作者。如果一个对象只为减少文件
行数、却仍然可以访问整个 Trainer，它没有改善架构，不应创建。

## 用失败案例证明边界真的存在

这次整理不能只靠移动 import。至少需要以下回归证据：

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

## 什么结果才算完成

完成后，普通用户继续运行相同的 `train`、`sample` 和 `evaluate` 命令，配置、checkpoint 和
运行产物含义不变。Extension 作者不会获得第二套注册方式；Diagnostic 作者则能从类型和构造
参数看出自己可以使用哪些事实与能力。

代码层面应能直接确认：

- operation 组装不再以 `utils.factory` 作为所有领域共享的隐藏入口；
- built-in 初始化有唯一、显式、幂等且可测试的调用位置；
- Diagnostic event 和 provider contract 中不再出现完整 `trainer: Any`；
- family-specific observation 不再依赖未声明的跨模块字符串键；
- 当前 Evaluation、Sampling、checkpoint、Extension 激活和训练选择语义没有改变；
- 规范、架构文档和公开 Extension 文档只描述最终实现，不把本提案中的候选名字写成当前 API。

如果实施调查发现 `utils.factory` 的当前位置并未造成错误依赖，或者收窄 Diagnostic 所需代价
明显高于现有两个调用者的收益，可以缩小方案并记录原因。允许得出“只修 Diagnostic，不移动
其他 composition”的结论；不允许为了宣称完成而进行没有行为或边界收益的大规模改名。

## 什么时候值得先处理

这两项是现有架构维护，不是等待排期的新产品，因此不占用 [`ROADMAP.md`](../../../ROADMAP.md)
的 `Next`。如果下一项工作会新增训练 Diagnostic、改变 Trainer 生命周期、增加 operation，
或扩展 built-in 注册范围，应先完成与那项工作直接相关的边界修正，避免继续复制隐藏依赖。
与这些边界无关的小型维护和算法实现不必等待本文全部完成。

真正动手前仍需重新核对当前 import 图、Diagnostic 调用者和 Extension 激活实现，因为这些
细节可能已经变化。维护可以缩小到证据支持的部分；它不会因为被记录在这里就自动获得大范围
重构授权。

当前规范与实现证据见：

- [`SPEC.md`](../../../SPEC.md) 的 Training、Evaluation 与 Extension contracts；
- [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) 的 package ownership 与 composition roots；
- [`utils.factory`](../../../src/stochaflow/utils/factory.py) 的当前组装和 built-in 注册；
- [Diagnostic contracts](../../../src/stochaflow/training/diagnostics/contracts.py)；
- [Training Strategy output](../../../src/stochaflow/training/strategy.py)；
- [Evaluation 已完成说明](../post-training-evaluation-support-plan.md)；
- [Extension 条件性性能复查](../extension-import-boundary-and-activation-latency-plan.md)。

本提案不改变 Physics/KD，不增加 Evaluation 功能，不实施 Extension 性能优化，也不改变当前
`ROADMAP.md` 的 `In progress: None` 与 `Next: None`。

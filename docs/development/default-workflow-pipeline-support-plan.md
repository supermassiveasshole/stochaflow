# 内置操作与工作流组合计划

> 工作状态：候选
>
> 当前结论：Stochaflow 已有彼此独立的训练、采样和正式 Evaluation
> 生命周期。本文候选只让调用方显式串联操作并传递结构化结果；需要恢复、
> 重试或分支的[通用工作流编排器](general-workflow-orchestrator-plan.md)由独立暂停计划负责。
>
> 规范来源：[`ROADMAP.md`](../../ROADMAP.md)、[`SPEC.md`](../../SPEC.md)、
> [`ARCHITECTURE.md`](../../ARCHITECTURE.md)
>
> 最后核对：2026-08-09

## 完成后用户能做什么

用户可以用普通 Python 明确调用一个操作，把它的结构化结果或 artifact 交给
下一个操作。本文永久保留两条真实路径；首个交付从中选择一条完整实现，另一条继续作为
同等明确的后续用户结果，而不是被删除或假装已经完成：

```text
训练教师模型
    -> 具体 distillation/Consistency owner 导出 versioned teacher bundle
    -> workflow 用 typed binding 把 bundle 交给蒸馏训练
    -> 对学生 checkpoint 运行正式 Evaluation
```

```text
生成图像
    -> 取得 SamplingRunResult.artifacts
    -> 显式启动超分辨率操作
    -> 发布恢复后的图像和正式 Evaluation 结果
```

每个操作继续拥有独立配置、失败边界和输出目录。调用方能够看到每一步用了
什么输入、产生了什么输出，以及失败发生在哪一步。

## 当前仓库已经支持什么

- 训练可以发布结构化 outcome 和 checkpoint；checkpoint 保存恢复训练和推理所需
  的受治理状态。
- sampling runtime 负责解析采样配置和 checkpoint、投影只读推理资产、验证完整
  output、调用 writers，并原子发布 manifest。
- `SamplingBuilder.run()` 解释具体任务输入，使用注入的模型、Process、inference
  assets 和 invocation context 完成初始化及按 `num_samples`/`batch_size` 分批，
  最后返回 writer-ready `SamplingOutput`。
- 正式 Evaluation 是独立操作，可评估 checkpoint 或完整 prediction artifact，
  并发布不可变结果。
- `super_resolution` DataBuilder 和教程证明了低分辨率/高分辨率数据组合，但没有
  内置超分辨率训练、推理和发布工作流。
- frozen-teacher 训练边界允许 Builder 声明教师资产、Strategy 组合教师和学生的
  前向与 loss，但没有 versioned teacher-bundle exporter 或内置“训练教师后立即蒸馏”
  的产品工作流。

这些能力是可复用基础，不代表跨操作工作流已经实现。

## 还没有支持什么

- 训练和 sampling 尚未同时提供经过承诺的、对称的 public library operation API。
- 当前 `run_sampling()` 位于 `stochaflow.sampling.runtime`，不是 sampling facade
  承诺的稳定入口。
- 没有统一的 typed binding 来说明“上一步的哪个输出成为下一步的哪个输入”。
- 没有可列出、查看和复制完整 operation 配置的严格操作配方目录。本文使用候选名
  `OperationRecipe`；它不同于当前的 `SamplingRecipe` 和 data recipe。
- 没有由具体 distillation/Consistency 实现拥有的 versioned teacher-bundle exporter。
- 没有内置的训练后蒸馏工作流或生图后超分辨率工作流。
- 没有负责重试、分支、并行或远程调度的通用工作流编排器；该方向由独立暂停计划负责。
- 计划中的 `OperationRecipe`、operation request/result 和 workflow descriptor 都不是当前
  公共 API。

## 什么时候可以开始或重新审查

显式顺序组合可以作为独立候选被路线图选为 `Next`。开始前必须确认：

- 至少一条任务工作流有明确维护者和 bounded smoke test；
- 每个被串联操作已经有独立配置和结构化结果；
- artifact binding 不要求 core 理解图像、condition 或教师模型的私有字段；
- 具体 distillation/Consistency owner 已接受 teacher-bundle schema、exporter、loader 和
  compatibility 的完整责任；
- sampling public library entry 是否纳入本次交付已经由维护者决定。

## 要完成哪些工作

### 提供对称的可编程操作入口

- 动作：为被组合的训练、采样和 Evaluation 操作确认稳定的 Python 调用入口；
  CLI 只解析参数并调用同一实现。
- 原因：工作流不能复制 runner 逻辑，也不能通过进程内调用 CLI 模拟组合。
- 影响范围：operation runtime、公开 facade、CLI 薄封装和 API 文档。
- 交付物：每个纳入范围的操作都有明确 request、result、异常和输出目录规则。
- 验证方法：同一配置通过 CLI 和 Python 入口得到等价的 manifest 与核心结果。
- 完成条件：组合层无需导入私有 CLI helper，也没有按任务名称分支。

### 定义结果和输入之间的显式绑定

- 动作：用窄类型描述 checkpoint、prediction artifact、sampling artifact 和正式
  Evaluation result 之间允许的传递关系。
- 原因：路径字符串不能说明 artifact 类型、identity、完整性或使用哪一个
  checkpoint variant。
- 影响范围：operation result、artifact manifest 和组合辅助函数。
- 交付物：可验证的 binding；错误类型、缺失 artifact 或 identity 不匹配时立即失败。
- 验证方法：成功、类型错误、缺失文件、digest 不匹配和重复执行均有测试。
- 完成条件：下游操作只消费显式声明的 artifact，不猜目录内容。

### 建立只物化完整配置的操作配方目录

- 动作：定义严格、版本化的 `OperationRecipe` 描述和目录，使用户可以列出、查看并复制
  经过维护的 training、sampling 或 Evaluation 配置；它只物化对应 operation 的完整
  配置，不执行操作或串联 workflow。
- 原因：默认配置需要可发现的 identity、输入、输出、成熟度和限制，但不能因此建立按
  recipe name 分支的第二套 runtime。
- 影响范围：Recipe descriptor/parser、first-party config source、external provider 发现、
  installed-wheel scaffold、catalog CLI/API 和 provenance；现有 operation loader 保持权威。
- 交付物：built-in 与 extension 共用的 catalog path，`list/show/init` 或等价窄接口，以及包含
  recipe identity/version、entry kind、inputs/outputs、maturity 和 template digest 的 manifest。
- 验证方法：物化配置必须通过对应 strict parser，并与手写配置得到相同 selected component
  identities；独立 extension、installed wheel、复制后脱离 catalog 运行和未知 recipe tests。
- 完成条件：用户能发现并复制默认配置，run evidence 可追溯 recipe/template provenance；
  runner 不检查 recipe id，目录不持有 live asset，也不执行多步工作流。

### 支持普通 Python 的顺序组合

- 动作：提供小型组合 helper 或示例函数，按调用方写出的顺序执行操作，并把每步
  result 传给下一步。
- 原因：首版只需要可理解、可调试的顺序执行，不需要新的 DAG runtime。
- 影响范围：workflow 文档、示例和少量 library helper；不改变 Builder 职责。
- 交付物：首个被选择路径的维护示例；另一条路径继续保留在下面的独立任务卡中。
- 验证方法：首个路径的 bounded end-to-end test 检查顺序、artifact identity、失败停止和重跑。
- 完成条件：首个路径的每一步都可以单独运行，组合只是显式复用公共操作；未选择的路径不被
  错写为已实现。

### 交付训练后蒸馏路径

- 动作：如果首个任务选择训练后蒸馏，由具体 distillation/Consistency owner 提供 teacher checkpoint 到 versioned teacher
  bundle 的 exporter、strict loader、学生训练配置和正式 Evaluation；本计划只提供 teacher
  bundle result 到 distillation request 的 typed binding。
- 原因：蒸馏的 teacher/target/loss 语义属于具体训练任务，不属于通用工作流层。
- 影响范围：具体 distillation/Consistency extension 的 exporter/loader、训练 operation、typed
  artifact binding 和示例工作流；workflow core 不定义 teacher payload。
- 交付物：`train teacher -> export teacher bundle -> distill student -> evaluate student` 的
  维护示例，以及可验证的 teacher-bundle binding。
- 验证方法：bundle 绑定 teacher checkpoint、Process/prediction semantics 和 exporter identity；
  学生 manifest 引用该 bundle，损坏/不兼容 bundle fail closed，resume 不重解释来源。
- 完成条件：该路径被选择时，exporter 与 bundle compatibility 由具体 owner 测试和发布；
  工作流层只有 typed binding，没有 teacher 模型类型、target 或 loss 分支。未被选择时，本任务
  保留为后续，不影响另一条首个路径完成。

### 交付生图后超分路径

- 动作：如果首个任务选择生图后超分，让超分辨率操作消费明确选择的 sampling artifact，并发布
  新的图像 artifact。
- 原因：sampling 与恢复是两个配置权威不同的操作，不能隐式共享参数。
- 影响范围：sampling result、超分辨率输入 adapter、writer 和示例工作流。
- 交付物：`sample -> super resolve -> evaluate` 的维护示例。
- 验证方法：输入顺序、sample identity、输出尺寸、writer manifest 和恢复指标均可核对。
- 完成条件：该路径被选择时，超分辨率可以独立运行，也可以消费上游 sampling artifact；未被
  选择时，本任务保留为后续，不影响另一条首个路径完成。

### 保持单次操作的职责边界

- 动作：让每个 operation 保留自己的配置解释和失败规则；组合层只传递带类型的结果或
  受治理 artifact，不传递任意可变上下文。
- 原因：上游配置不能静默控制下游，Builder 也不能获得跨 run 状态。
- 影响范围：operation request/result、artifact binding、组合 helper 和文档。
- 交付物：普通 sampling output 与正式 Evaluation evidence 的清晰区分；
  `OperationRecipe` 只展开普通配置，不执行操作或多步工作流。
- 验证方法：错误配置、错误 artifact 类型和跨步骤身份不匹配都在下游执行前失败；
  independent extension 不需要修改 core dispatch。
- 完成条件：teacher-bundle 语义仍由具体 distillation/Consistency owner 负责；候选接口在
  进入 `SPEC.md`、公开 API 文档和测试前都不是公共契约。

## 如何证明已经完成

- CLI 与 Python operation parity tests。
- typed binding 的成功和 fail-closed tests。
- 操作配方目录的 built-in/external parity、strict materialization、installed-wheel 和复制后独立
  运行 tests。
- 首个被选择路径的 bounded end-to-end test：训练教师、蒸馏学生和正式 Evaluation，或
  sampling artifact、超分辨率恢复和正式 Evaluation。
- 被选择路径能拆开独立执行，组合层不复制 Training/Sampling/Evaluation runtime。
- 另一条路径仍在本文和开发索引中可达；只有它也完成后，才可作为通用编排器的第二条证据。
- independent extension test 证明新增任务不需要修改 core dispatch。
- public docs 明确区分“独立操作”“显式顺序组合”和“通用工作流编排器”。

## 明确不包含什么

- 不在本计划中实现 latent、Stable Diffusion 或 consistency 的算法细节。
- 不提供万能 `run(kind=...)`、任意 YAML object graph 或通用 Pipeline 基类。
- 不让操作配方目录执行 workflow，也不把 exporter utilities 伪装成现有 operation entry。
- 不让训练配置自动触发 final sampling 或正式 Evaluation。
- 不负责集群调度、队列、权限、远程重试或可视化 DAG 编辑器。
- 不因候选 API 出现在本文而承诺兼容性。

## 详细设计和研究资料在哪里

- [原始完整设计、候选接口和测试矩阵](notes/default-workflow-pipeline-support-plan/design-archive.md)
- [超分辨率工作流支持计划](super-resolution-workflow-support-plan.md)
- [Consistency 与蒸馏计划](consistency-distillation-support-plan.md)
- [Latent Diffusion 计划](latent-diffusion-support-plan.md)
- [Stable Diffusion 计划](stable-diffusion-component-native-support-plan.md)
- [正式 Evaluation 后续计划](post-training-evaluation-support-plan.md)
- [通用工作流编排器计划](general-workflow-orchestrator-plan.md)

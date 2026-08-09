# Consistency Distillation 支持计划

> 工作状态：候选
>
> 规范来源：[`ROADMAP.md`](../../ROADMAP.md)、[`SPEC.md`](../../SPEC.md)、
> [`ARCHITECTURE.md`](../../ARCHITECTURE.md)

- 文档性质：候选开发计划；不是当前公开 API、能力声明或实施排期
- 最近复核：2026-08-09
- 受众：算法、Training、Sampling、Evaluation 与 extension contract 维护者
- 关联记录：
  [默认工作流与可组合任务](default-workflow-pipeline-support-plan.md)、
  [Evaluation 后续决策](post-training-evaluation-support-plan.md)

## 完成后用户能做什么

如果根路线图选择本计划，用户可以用一个冻结的 diffusion teacher 训练更快的 student。首版
student 直接从带噪图像预测最终干净图像，并能只依赖自己的 checkpoint 完成一步或少量几步
采样；运行时不需要 teacher。

首版只覆盖无条件图像任务、一个固定 teacher 和一套正式 Evaluation。Evaluation 会同时报告
student 的质量、速度、资源使用和与 teacher 的对照。这不表示仓库当前已有内置 consistency
工作流，也不允许提前扩张 core。

## 当前仓库已经支持什么

当前仓库已经提供的是通用 frozen-teacher 组合基础，不是 consistency 算法支持：

| 已有能力 | 当前边界 |
| --- | --- |
| `TrainingBuilder -> TrainingPlan` | Builder 可构造、加载并冻结辅助 teacher，声明 `mode="eval"`；core 负责 device、mode、checkpoint 与严格状态恢复 |
| `TrainingStrategy` | 可消费注入的 teacher/objectives，组合 forward、单个 scalar loss 与 metric updates；仍使用一个自动 optimizer 生命周期 |
| Gaussian family | 已有 discrete VP Process、epsilon/x0/v/score normalization、selected-pair coefficients 与 DDIM behavior oracle |
| Checkpoint inference | 可投影显式 inference assets，并冻结任务专用 sampling recipe；未投影 teacher 不进入 inference view |
| Sampling | `SamplingBuilder` 可直接完成非数值 transform；完整 few-step 数值循环由 task/family Sampler 拥有 |
| Evaluation | 已有 checkpoint subject、显式 raw/EMA、exact SamplePlan、provider identity 与 immutable result bundle 基础能力 |

仓库中的 frozen-teacher architecture fixture 只证明 Builder/Strategy/managed-asset 边界可用；
它不实现 endpoint consistency、teacher trajectory pair、target student EMA、student-only
consistency sampling 或相应质量协议，也不是本计划的实施入口。

## 还没有支持什么

1. 没有内置能力或持续维护的 extension 提供 consistency Objective、`TrainingBuilder`、
   `TrainingStrategy`、endpoint operator、pair sampler、`Sampler`、`SamplingBuilder` 或 Evaluation
   profile。
2. automatic Trainer 的 EMA 只跟踪 primary model 的 inference shadow。它没有独立、可在
   lower loss branch 调用、只在成功 optimizer step 后更新并 strict-resume 的 target-student
   EMA lifecycle；standard consistency distillation 因此不能由当前 automatic loop 正确表达。
3. `DDIMSampler.transition` 属于 sampling policy。Training 不得反向依赖它；当前没有经审查的
   training-facing、family-specific deterministic selected-pair primitive。
4. 没有 versioned teacher bundle、Process/prediction fingerprint、strict bootstrap 与来源
   lineage contract。
5. 没有 endpoint identity parameterization、grid/pair policy、Pseudo-Huber distance、clean
   anchor schedule 或 terminal-time stability policy。
6. 没有冻结 teacher/student baseline、seed bank、NFE/forward-call 口径和正式
   quality-speed Evaluation protocol。

## 什么时候可以开始或重新审查

只有以下条件全部满足，根路线图才能把本记录选为进行中：

1. `ROADMAP.md` 明确选择该首个完整功能、决策负责人、预算和需要回答的产品问题。
2. 冻结首个 teacher、dataset、prediction type、student compatibility、seed bank、资源上限与
   任务专用 Evaluation 协议；不预先承诺绝对 FID。
3. 重新核验附录中的论文、参考实现、provider/API、许可证与版本；历史草案不作兼容性证据。
4. 对 target-student update 做架构决策：若窄的 core-managed relation 通过公共抽象准入，
   同批更新规范与架构；否则定义独立、受支持的 training-loop family。Strategy 内更新 target、
   Diagnostic 副作用和复用 inference EMA 均不可接受。
5. 对 teacher transition 做 ownership 决策：若 built-in DDIM 与 distillation 成为两个真实
   消费者，则提取窄的 Gaussian family primitive 并让双方复用；不得复制数学或让 training
   import sampling policy。
6. 证明 teacher + online student + target student + 可选 inference EMA 在目标设备上的容量
   与数值稳定性；未测量的硬件推断不构成启动证据。

首次实现必须放在临时、可安装的独立 extension prototype 中，通过公共 Registry、Builder、
`TrainingStrategy`、`Sampler`、`SamplingBuilder` 和 Evaluation 约定验证。只有证据证明值得长期
维护后，才决定是否形成持续维护的任务或更窄的共享约定。完整公式和消融设计保存在文末链接的
研究附录中。

## 要完成哪些工作

### 确定算法范围和验收口径

- **动作：** 确认 endpoint CM 而非任意 \(t\rightarrow s\) operator；固定 teacher、相邻
  pair grid、identity boundary、clean-anchor 消融、baseline 与 SamplePlan，并记录 student 从
  \(x_t\) 预测最终状态以及 teacher selected-pair target 的精确定义。
- **原因：** trajectory target、clean reconstruction 与 final quality 需要可独立归因。
- **影响范围：** 算法契约、实验配置、Evaluation protocol 与支持声明。
- **交付物：** 一页算法约定、实验表、seed/protocol identity 和失败判据。
- **验证方法：** 审查时间参数、pair、baseline、seed 与 benchmark 选择规则。
- **完成条件：** 维护者能区分 trajectory target、clean reconstruction 与 final quality；
  没有暗含第二时间参数或未声明的 benchmark 选择权。

### 解决训练状态的更新和恢复

- **动作：** 由该任务的 extension 实现 versioned teacher bundle exporter 和严格 loader；exporter
  从训练结果中选择明确的 checkpoint，并只导出声明过的 teacher 资产。工作流只负责绑定已导出
  bundle，不推断或重组 teacher。Builder 冻结 teacher 并声明 eval auxiliary；为 target student
  选择获批 lifecycle，固定 parameter/buffer、dtype、mode、update cadence 与 resume semantics。
- **原因：** teacher、online student、target 与 inference EMA 具有不同状态所有权。
- **影响范围：** TrainingBuilder/Plan、Trainer lifecycle、checkpoint 与 inference projection。
- **交付物：** 独立 extension 约定测试样例、teacher bundle exporter/loader、bundle schema、错误
  矩阵及 checkpoint state。
- **验证方法：** 从训练结果选择 checkpoint、导出 bundle、独立加载再由工作流绑定的往返测试；
  同时覆盖 fresh/resume 对照、无梯度检查、成功/跳过 optimizer-step 与错误 bundle tests。
- **完成条件：** 只有 online student 进入 optimizer；teacher/target 无梯度；target 只在成功
  optimizer step 后更新；中断恢复与不中断运行的 target state/counter 一致。

### 建立 pair 与 endpoint 数学

- **动作：** 复用或提取 family-specific deterministic selected-pair primitive；实现
  endpoint operator、grid validation、Pseudo-Huber/MSE 和可关闭的 SNR-weighted anchor。
- **原因：** training 不得依赖 sampling policy，也不能复制 selected-pair 数学。
- **影响范围：** Gaussian family primitive、extension math、Objective 与 contract tests。
- **交付物：** extension-local narrow capability 与纯 tensor/contract tests。
- **验证方法：** identity、非法 pair、fixed-seed replay、finite gradient 与 anchor 消融。
- **完成条件：** \(t=0\) bitwise identity；非法 pair 会被明确拒绝；fixed seed pair 可重放；
  terminal state、loss 和 gradients 有限；`anchor=0` 等价于纯 consistency 路径。

### 接入单优化器训练

- **动作：** Builder 组合 teacher、target、operator 与 Objectives；Strategy 只解释 batch、
  执行 forward 并产生一个 scalar loss 和任务专用 diagnostics。
- **原因：** 算法变化必须服从现有单 optimizer 组合与职责边界。
- **影响范围：** extension Builder/Strategy、配置、metrics 与 automatic loop integration。
- **交付物：** tiny overfit config、fresh/resume workflow 与 focused failure tests。
- **验证方法：** bounded train/resume、loss trend、资产 mode/device 与无 core dispatch 检查。
- **完成条件：** consistency/anchor loss 可观测且按预期下降；core runner 无注册名分支；
  Process 保持 model-free，root Dynamics 不新增 universal math。

### 交付 student-only inference

- **动作：** 固定 inference recipe；one-step 直接调用 endpoint operator；few-step Sampler 拥有
  denoise-renoise schedule、RNG、observer events 和 ephemeral state。
- **原因：** inference 必须仅恢复 student assets，并由完整 numerical owner 管理循环。
- **影响范围：** checkpoint recipe、SamplingBuilder、Sampler、sample config 与 manifest。
- **交付物：** 独立 sample config、raw/EMA 选择、NFE/forward-call metadata 与 replay tests。
- **验证方法：** forward-call 计数、teacher/target 删除、raw/EMA、seed 与 observer replay。
- **完成条件：** one-step 恰好一次 student forward；移除 teacher bundle 与 target state 后
  checkpoint sampling 仍成立；target EMA 与 inference EMA 永不混用。

### 建立正式评估和是否长期维护的证据

- **动作：** 用相同 subject/data/SamplePlan/provider identity 对比 teacher 1/2/4/8/50 NFE
  与 student 1/2/4/8 NFE，并记录质量、重建、latency、memory 和实际 model evaluations。
- **原因：** 是否长期维护必须同时依据质量、速度、资源和失败原因，不能只挑一个最好结果。
- **影响范围：** EvaluationBuilder、task profile、immutable results 与公开报告。
- **交付物：** immutable Evaluation results、clean-anchor/target-policy 消融、复现说明和限制。
- **验证方法：** protocol digest、completeness、NFE 计数、baseline 一致性与重复 seed 检查。
- **完成条件：** 正式结果不由 `TrainingDiagnostic` 代替；同 NFE 与高 NFE teacher
  baseline 都存在；失败可归因到 pair error、consistency error、anchor conflict 或
  preconditioning instability。

## 如何证明已经完成

- **功能：** finite consistency loss、精确 identity boundary、anchor 可开关/调度、teacher
  始终 frozen/eval、strict resume、student-only one/few-step sampling 全部有测试。
- **架构：** Builder 负责资产组合；Strategy 不移动、冻结或保存资产；Sampler 拥有
  完整数值循环；新能力通过 extension/Registry/config 进入；无 task-name core dispatch。
- **状态：** teacher、target、primary、inference EMA 名称和生命周期分离；checkpoint 与
  inference projection 遇到不匹配会明确拒绝；sampling 不恢复仅训练使用的资产。
- **研究：** 固定 seed 完成 `anchor=0` 与至少两个非零权重、MSE/Pseudo-Huber、grid 和 target
  policy 对比；报告完整 NFE-quality curve，而非挑选单点。
- **工程：** focused tests、`uv run ruff check .`、`uv run pyright`、严格文档构建及相关
  config/reference checks 通过；稳定行为同步 SPEC/ARCH/public docs/CHANGELOG。

## 明确不包含什么

首个完整功能不包含、但明确保留以下未来支持构想。每一类都必须由对应负责人根据触发证据另行
提出，不能通过给首版增加可空字段提前实现。

| 未来构想 | 触发证据 | 负责人 |
| --- | --- | --- |
| \(G(x_t,t,s)\) Consistency Trajectory Model；continuous-time VP/VE、Heun PF-ODE teacher stepper、analytic preconditioning | 被选中的算法结果确实需要任意 \(t\rightarrow s\) 映射或连续时间 teacher，且固定复现实验表明 endpoint 离散方案不足 | consistency 算法 extension 与对应 Gaussian family 维护者 |
| non-constant target EMA、可训练 buffer、non-zero dropout、同步 dropout RNG | 消融证明固定 EMA 或 dropout-free 路径限制目标质量，并且状态更新、恢复和 RNG 规则已有可测试设计 | 任务 `TrainingBuilder` 与 training runtime 维护者 |
| conditional generation、classifier-free guidance、guided teacher、consistency SR | 根路线图选择具体 conditional 或 SR 用户结果，并准备了数据、conditioning 约定和正式 Evaluation | 对应任务 extension 的 Builder/Evaluation 负责人 |
| latent consistency、text-to-image、多模态 batch | Latent Diffusion、Stable Diffusion 或多模态前置能力已经验收，且新的完整任务无法由现有 endpoint 方案直接表达 | 对应 latent、text-to-image 或多模态任务负责人 |
| adversarial/perceptual loss、offline cached teacher pairs | 质量消融或 teacher 计算成本给出可测缺口；缓存还需两个真实读取方和完整数据身份 | 任务 Objective 负责人；缓存另由 DataSource/DataBuilder 负责人审查 |
| distributed teacher/student execution、progressive-distillation 对照 | 单设备实测先证明资源瓶颈，或基准明确要求分阶段蒸馏；不能只凭模型规模启动 | 任务负责人和 training/distributed 计划负责人 |
| independent optimizers、alternating updates、manual backward、通用 distillation schema | 获选算法无法服从单 optimizer 自动循环；通用 schema 还需至少两个独立任务证明同一约定 | 独立 training-loop family 负责人；公共 schema 由架构维护者审查 |
| 将 prototype 提升为持续维护的 example、built-in 或公共 core 抽象 | 至少两个真实使用方复用、验收证据完整，并由根路线图明确选择维护级别 | 根路线图负责人和相应 core/extension 维护者 |

## 详细设计和研究资料在哪里

完整数学、论文/provider 调研、候选 API、teacher bundle/config 草案、消融矩阵、风险和历史
实施清单保存在
[`notes/consistency-distillation-support-plan/design-and-research-notes.md`](notes/consistency-distillation-support-plan/design-and-research-notes.md)。
附录是重启时的研究输入，不是当前 API、排期或兼容性承诺。

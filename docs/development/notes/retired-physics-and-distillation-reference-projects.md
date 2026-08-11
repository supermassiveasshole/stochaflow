# 退役的 Physics/KD reference project 留下了什么

> 文档类型：历史与研究备忘
>
> 工作状态：不参与排期
>
> 当前可用性：两个旧项目已于 2026-08-11 从仓库移除，不能再安装或运行；它们验证过的
> 通用边界仍由框架代码、测试和公开教程维护。

仓库曾经保留 `physics-reconstruction` 和 `knowledge-distillation` 两个可安装 extension，
用它们反复验证训练、checkpoint 和采样能否由第三方项目组合。它们后来都没有继续成为真实
用户任务：没有受维护的数据发布、模型质量基线和正式 Evaluation，却必须在每次公共训练接口
调整时一起修改。高频迭代期间，这种维护成本已经大于它们提供的回归价值，所以维护者在
2026-08-11 决定删除项目，而不是继续把“能编译、能跑 tiny smoke”误写成领域支持。

删除项目不等于删除其中的设计判断。下面保留的是以后重做真实任务时仍值得使用的部分；旧包的
目录、命令和输出数值不是兼容承诺，也不会仅为了历史完整而恢复。

## KD 真正验证的是训练资产与推理资产可以不同

旧 KD fixture 使用一个确定性的 synthetic classification 任务，让学生同时接受普通分类损失
和冻结教师给出的 temperature-KL 损失。`TrainingBuilder` 负责从普通 PyTorch
`state_dict` 构造教师与 logit calibrator，冻结它们，并把教师、额外 Objective 和
calibrator 声明为有稳定名字的 managed training assets；`TrainingStrategy` 只解释 batch、
执行学生/教师 forward 并组合损失。这样，设备、mode、optimizer、checkpoint 和继续训练仍由
core 管理，任务代码不需要接管训练循环。

它最有价值的验证发生在训练结束以后。Checkpoint 保存教师、额外 Objective 和 calibrator
的训练状态，但只把 calibrator 声明成学生生成时需要的 embedded inference asset。独立采样
只重建学生，并按 role 请求 calibrator；它不构造教师、训练 Objective 或 TrainingBuilder。
旧验收甚至会先更换本地 bootstrap，再继续训练，随后删除 bootstrap 并从另一个工作目录采样，
用确定的 calibrated logits 证明推理读取的是 checkpoint 中嵌入的状态，而不是最初的资产路径。

这个例子也特意没有外部数据：seed 和配置生成全部 in-memory splits，
`artifact_bindings=None`，因此不会伪造一个 DataArtifact 或创建无意义的 cache。以后若使用真实
分类或生成数据，acquisition 必须重新回到 `DataSource → DataArtifactStore → DataBuilder`
边界，不能照搬这个 synthetic 特例。

这些判断现在由 [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) 的 Training、inference 和
extension 边界以及对应 core tests 承接。它们只证明框架可以组合冻结资产，并不证明当前存在
一个可用的蒸馏任务。

KD 的重建节点已经有明确归属：只有
[Consistency Distillation 候选](../consistency-distillation-support-plan.md)被 `ROADMAP.md`
选为实际工作，并且首个教师、学生、数据与质量—速度 Evaluation 都确定以后，才按届时的公共
契约新建一份可安装扩展，验证 teacher bundle、学生训练和不依赖教师的推理。那次工作应服务
真实的 consistency 任务，而不是恢复旧的 synthetic classification fixture。

## Physics 真正留下的是领域数学如何停在扩展里

旧 Physics fixture 以三帧 Kolmogorov-vorticity reconstruction 为例。外部
`[trajectory, time, H, W]` NumPy 数组由领域 `DataSource` 验证，再通过 framework store
发布 referenced DataArtifact；扩展自己的 DataBuilder 才解释 trajectory range、mmap
Dataset view 和连续三帧 batch。`.npy` header、shape/dtype、外部文件 inventory、分段规则和
positional alignment 属于任务，identity、locator、lock、staging、verification、quarantine
和 atomic publication 属于通用 Store。这条分界仍可从
[数据构建文档](../../configuration/data-pipeline.md)直接理解，不需要保留旧项目才能成立。

训练侧把原始物理场归一化，采样 Gaussian marginal，并将 PDE condition 交给条件 denoiser；
普通 baseline 仍复用内置 DDPM/DDIM。旧实现刻意区分了两种看起来相近、实际责任不同的
guidance：如果 condition 只改变模型给出的 `clean/epsilon` prediction，它可以封装在
Gaussian Dynamics 的 callable 中；如果 correction 发生在一次 accepted transition 之后，
它已经改变数值算法，必须由项目自己的 Sampler 管理。后者可以复用公开的 DDIM schedule 和
`transition()` primitive，但不复制 DDIM 方程，也不把 PDE 参数塞进通用 Sampler。这个判断
现在保存在[复用 Gaussian family 教程](../../tutorials/reuse-gaussian-components.md)。

旧 writer 还说明领域输出不是普通图片：它按 batch 写 `reconstructions.npy` 和指标，使用
同目录替换完成发布，失败时清理不完整文件，并拒绝覆盖既有结果。partial noising 的 marginal
time 必须与 reverse schedule 的第一个 source time 相同，最终到达 clean state `0`，避免初始
加噪与第一步反推坐标错开。

40 条 trajectory、后四条 held-out、1272 个三帧结果以及 30/40-step dense trajectory 的
数字只服务旧 fixture 的 mmap、对齐检查和输出估算。它们从未证明收敛质量或科学复现，也随
旧任务一起退休，不能继续充当公共 capacity profile。原来的证据分成 tiny deterministic
E2E、真实 batch 容量测量、production-path bounded smoke 和尚未完成的全量科学验证四层；
只有前三层中的集成行为曾被检查。[Sampling artifact 容量](../../configuration/sampling-capacity.md)
只保留任何新任务都能重新代入的通用公式、测量工具和证据分级，不再保存旧 Physics 数值。

Physics 当前没有产品计划 owner，也不因为旧 fixture 被删除就自动成为某个计划的任务。只有
一个真实的物理重建需求已经确定数据许可与发布方式、可维护模型、科学指标和 formal
Evaluation，并被根 `ROADMAP.md` 明确选中后，才从当前接口重新实现领域 extension。届时
[大规模数据处理计划](../hierarchical-data-pipeline-support-plan.md)可能提供 mmap 之外的数据
分层、预算和回压能力，但它只可能承接数据路径，不能替 Physics 决定模型、数值算法或验收标准；
如果该计划选择了别的真实 workload，就不应重建 Physics。

## 这次决定覆盖了此前的临时保护

早期 Hydra 配置整理曾提到清理这两个项目，维护者随后要求当时不要顺带改动 Physics/KD。
那是一条防止配置重构扩大范围的临时决定，不是永久维护承诺。2026-08-11 的退役决定明确覆盖
它：当前公开文档只描述仍能使用的框架和示例，独有思路保存在本备忘，未来恢复则必须分别满足
上面的真实任务与排期条件。

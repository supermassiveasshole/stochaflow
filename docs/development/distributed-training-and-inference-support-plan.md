# 固定单机分布式训练与后续采样计划

> 文档类型：功能计划
>
> 工作状态：暂停（Parked）
>
> 当前可用性：训练、采样和 Evaluation 现在都只支持一个进程；若本计划以后被选中，首轮只会交付 Linux 单机、固定进程数的 DDP 训练，当前尚不可用。

## 先让一份大数据任务安全地用满一台服务器

多设备支持只在一个已经正确的单设备任务遇到明确限制时有价值：单张 GPU 仍能放下模型和一个
最小训练批次（microbatch），但单卡无法在预算内达到需要的吞吐，或一次同步更新需要更多样本
（effective global batch）。若模型或单个样本本身
放不进一张卡，DDP 复制完整模型并不能解决问题，那属于后续模型分片方向。面向几 TB 或几十 TB
数据时，首个真实目标不是同时发明
所有并行方式，而是让同一份已经准备并验证的数据在一台 Linux 服务器的 8 张 H200 上完成一场
可核对、中断后可继续训练的运行。

若本计划被选中，首轮因此只采用 Linux 单机、固定进程总数（`world size`）的 DDP（分布式数据并行）训练：`torchrun`
为每张 GPU 启动一个进程，每个进程保存一份完整模型、读取不同数据，再同步梯度。它不使用弹性
扩缩容，某个进程失败后也不原地更换成员；整场运行失败，之后从最后一个完整训练状态继续训练。
下面的命令只说明预期体验，**Stochaflow 的入口和参数当前不存在，不是公共 API，命令不能执行**：

```text
torchrun --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  <未来的 Stochaflow DDP 训练入口> \
  --config configs/train/afhq.yaml
```

这个范围刻意比“多设备训练与采样”窄。CPU/Gloo 多进程测试可以在 CI 中检查协调逻辑，真正的
产品承诺则来自指定 Linux、CUDA、PyTorch、NCCL 和 8×H200 组合上的验收；CPU 测试不能替代
CUDA 数值正确性、NCCL 通信或吞吐量证据。

## 八个进程必须先成为同一场运行

如果八个进程各自沿用今天的单进程入口，它们会分别选择 run ID、创建输出目录、构造 logger、
访问数据并尝试发布 checkpoint，最终得到八场互相覆盖或互不相干的训练。因此进程组建立以后，
必须先形成一份不可变的运行会话，再允许任何数据或输出副作用发生。

主进程负责选择 run ID、私有工作目录和最终输出位置，并把解析后的配置摘要、数据请求摘要、
进程布局和随机种子广播给其他进程。其他进程核对自己看到的配置与环境，任一摘要不一致就让
整场运行在读数据前失败。每个进程可以按自己的进程编号（`rank`）写独立诊断日志和临时状态；只有主进程可以
发布公共结果清单（manifest）、结构化训练结果和便携 checkpoint。成功之前的文件始终位于
私有位置，不能被读者误认为一场已完成运行。

这个会话只协调一次运行，不成为通用工作流编排器，也不替代负责取得外部数据的 `DataSource`、
负责组装训练数据视图的 `DataBuilder`、负责执行训练生命周期的 Trainer 或 checkpoint 各自已有的责任。

## 固定 DDP 为什么需要独立训练执行者

`Trainer` 是负责执行一场训练生命周期的组件。分布式支持不会变成现有 `Trainer` 中四处散落的
`if distributed`、`if fsdp` 或并行模式分支：现有 `Trainer` 继续只负责稳定的单进程训练；固定
拓扑 DDP 则由新的 `DDPTrainer` 建立全体进程共识、包裹执行模型、协调更新和共同失败。用户选择
的是 DDP 运行方式，不直接配置 Trainer 类；未来训练入口在接触数据前选择内部执行者。
`DDPTrainer` 这个名称只是设计标记，当前不存在，也不构成公共 API。

`DDPTrainer` 先打开分布式会话，再把 rank、设备和 collective 能力交给数据与训练计划的组装过程；
组装完成后，它核对该计划是否满足 DDP 限制，随后执行训练，并在所有退出路径关闭会话。负责
进程组机械操作的私有 session helper 只是它的协作者，不是第二个生命周期 owner。这样
`DataBuilder` 能在创建 reader 前取得 rank，单进程 `Trainer` 又不必知道分布式会话存在。

如果以后证明确实需要 FSDP（模型分片训练），届时要重新设计分片参数、optimizer、EMA 和 checkpoint 的完整
生命周期，而不是给单进程 `Trainer` 或 `DDPTrainer` 继续加分支。独立的 `FSDPTrainer` 是一个
候选形状，不是现在固定的类；张量、流水线或混合并行也要按自己的协作关系重新判断。不同训练
执行者可以复用 Strategy、指标和原子写入等稳定组件，但只通过窄接口共享。若继承现有
`Trainer` 必须覆盖私有钩子或跳过父类顺序，就不为了代码复用而继承，而是实现同一个最小训练
执行契约。这样扩展现有能力不会形成一个所有并行方式都必须理解的臃肿基类。

## 同一个模型为什么有保存对象和执行对象

DDP 并不是把任意模型包一层以后就自然生效。当前任务的 `TrainingStrategy` 负责调用模型并计算
loss；如果它仍持有包装前的原始模型，Trainer 即使创建了 DDP wrapper，实际 forward 也会绕过
梯度同步。

未来的训练组合必须明确区分两种身份。canonical model 是 Builder 选定的原始模型，继续负责
组件身份、EMA、checkpoint 和便携导出；execution model 是 Trainer 完成设备放置后交给 Strategy
实际调用的对象，在 DDP 模式中就是包装后的模型。Builder 产生的 Strategy 必须通过一个窄而
明确的执行绑定接收后者，不能永久捕获原始模型，也不能让任务按模型名称或具体类寻找 wrapper。
内置任务和外部 Extension 要通过同一个契约，并用一个独立的第三方 Strategy 证明它可以替换。

首轮只支持 primary model 这一棵可训练参数树完整放进单张 GPU，不把其他可训练资产悄悄纳入
同一 optimizer。其余会被保存的状态必须不可变、能显式同步并证明各进程一致，否则组合阶段就
拒绝这项任务。原因很直接：最终只保存一份公共训练状态，不能让主进程的偶然副本掩盖其他进程
已经漂移。具体的状态、buffer 和模型兼容矩阵留在技术附录。

## 数据分工决定训练的数学含义

一个进程的编号叫 `rank`，参加运行的进程总数叫 `world size`。这些事实必须在 `DataBuilder`
创建 Dataset view、sampler、worker 和 DataLoader 之前提供。Builder 仍然拥有任务数据的含义；
core 不能在事后猜测并替换任意 DataLoader 的 sampler。

Builder 返回的不应只有三个 iterable，还要给出这次运行的数据分工证据：每个 rank 负责哪些
已准备分片或样本范围，一个 epoch 预计产生多少个 optimizer 更新窗口，尾部是丢弃、补齐还是
拒绝，以及 validation 怎样做到无遗漏、无重复。大数据 reader 应直接把已经准备好的 shard 或
有界索引范围分给 rank 和 worker，不能让每个进程为了随机顺序建立覆盖全体样本的巨大排列，
也不能把数亿编号集中到主进程内存。具体的有界覆盖算法由 reader 和 prepared manifest 共同证明。
所有 rank 必须看到相同的数据身份和分工摘要；训练中提前耗尽、额外产生 batch 或与声明不符时，
整场运行立即失败，而不是用不等长输入的容错机制掩盖错误。

预计分工还不够，因为过滤、解码失败或自定义 collate 可能在运行时改变实际 batch。`DataBuilder`
还要随 iterable 提供一个只报告运行事实、而不暴露任务 batch 内容的窄接口，在每个训练窗口报告单调窗口编号、实际样本数
或 loss 权重、分工标记以及有界的已处理覆盖证据。`DDPTrainer` 只核对这些控制事实与事先声明
是否一致，不查看或解释不透明 batch，也不给通用 batch 强加 `sample_id` 字段。无法提供这类事实
的 reader，只能在已准备数据中互不重叠的样本范围、确定性顺序和精确处理记录足以共同证明覆盖
时进入首轮；只给一个总数或摘要值（digest）不能证明无重无漏。

普通训练 YAML 中的 `batch_size` 保持 per-rank batch（每个 rank 的 batch）这一含义。一次更新
实际看到的样本量是 `world size × per-rank batch × gradient accumulation`，运行清单必须记录
这个 effective global batch。框架不会自动缩放学习率。首轮要求每个实际提交的同步窗口都达到
声明的完整 per-rank batch 与 accumulation，各 rank 的有效样本数、microbatch 数和 loss 聚合权重
相同；仅为日志提供的加权平均不能悄悄修正不等权梯度。任何 partial tail 都只能按 Builder 已声明
的方式一致丢弃，或在开始训练前拒绝。以后若允许可变权重，必须另行定义梯度缩放并逐 step 记录
实际 effective global batch。

validation 不得沿用会填充重复样本的训练分工。各 rank 可以拥有不同数量的 validation 记录，
因此本地推理必须使用各 rank 已同步的 canonical model 快照，而不是会在每次 forward 参加通信的
DDP execution model。本地读取期间不做逐 batch 的跨进程同步，结束后所有 rank 按同一顺序参加一次全局完整性和
指标合并；即使某个 rank 没有记录也必须参加。这样 best checkpoint 和 early stopping 看到的
仍是一份无重复、无遗漏的全局结果。

若 validation 含随机推理，改变 rank、进程数或分片不能改变同一个样本的预测；做不到时只允许
主进程用完整数据视图执行。没有记录的 rank 仍要参加最终指标合并，但本地推理本身不得要求每个
batch 都通信，否则不等长 validation 会发生死锁。具体随机数和 Metric 单位元规则留在附录。

## 几十 TB 数据不能因为八张卡而完整验证八次

`DataArtifact` 的身份和严格恢复验证仍然是训练准入条件，DDP 不能让 rank 自己编造 artifact
handle，也不能把一个运行时 receipt 当成可复制凭证。但如果八个进程分别对几十 TB 数据做同一
次全量读取和哈希，训练尚未开始就会把验证成本放大八倍。

TB 级验收前，大数据管线中负责验证数据的组件必须先明确选择一种准入方式，DDP 只消费结论。
最简单但昂贵的方式是每个 rank 各自完整验证，并把八次
读取作为已测成本；更实用的方式是先修改 Data 规范，让 `DataArtifactStore` 根据一次受信的
当前机器/当前运行验证，为每个进程的这一次请求分别签发短期验证凭证（receipt）。后者必须绑定
artifact 身份、解析后的实际位置（locator）、验证策略、运行会话、时效和失效规则，而不是广播
一个 receipt 或布尔值。它不是写入 checkpoint 的永久信任，也不能跨机器、跨位置或跨运行复用。若这种证明尚不能保持现有
信任边界，可以先用小数据验收 DDP 正确性，却不能宣称 TB 级准入已经完成。验收会记录实际读取与
哈希的字节数，并验证内容替换、过期 locator 和错误数据身份都能在训练前被拒绝。

## 一次更新只有在所有进程同意后才算发生

gradient accumulation 期间，非最后一个 microbatch 的 forward 和 backward 都要处于 DDP 的
不同步范围，最后一个 microbatch 才进行梯度同步。反向完成以后，各 rank 先共同确认数据窗口、
loss、梯度和混合精度检查有效；任何 rank 发现非有限值或决定跳过，所有 rank 都跳过同一次更新。

混合精度控制也必须拆成“先检查、全体决定、再提交”三步，不能让某个进程已经更新 optimizer，
另一个进程才发现数值溢出。现有把检查与更新合在一起的单进程辅助逻辑不能原样复用。精确的
GradScaler 状态机和单 rank overflow 测试留在技术附录。

optimizer 尝试成功以后还要再取得全体成功结论，随后才共同推进 EMA、学习率调整器、
`global_step` 和本次训练指标。一个 rank 的 `optimizer.step()` 抛错而其他 rank 已经修改了内存
不能被伪装成可回滚事务：整场运行会失败，任何进程都不得保存或发布这个半完成状态，恢复只能
回到上一个完整 epoch。这条全体确认规则防止一次局部 AMP skip 或 Python 异常把八份 optimizer、
EMA 和 scheduler 状态永久分开。

loss、训练统计和 validation 指标也不能采用“八个局部平均数再平均”。可加统计量要合并总和与
计数；其他 Metric 必须通过分布式训练绑定声明并实现精确的状态合并规则。只有最终 scalar、没有
可合并状态的 Metric 在 DDP 组合阶段会被拒绝。这个能力不让 `MetricEngine` 猜测 batch、phase
或模型语义，具体统计仍由 Metric provider 拥有。checkpoint 选择、early stopping 和任何依赖
validation 的 scheduler 只能读取同一份全局结果；主进程形成选择决定并广播，所有 rank 再执行
相同状态变化。当前 Diagnostic 或训练中的 live Evaluation（只为选模提供观察，不是正式 Evaluation
证据）若没有声明 all-rank 合并或 rank-zero 完整数据
视图，就在组合阶段被拒绝，不能让每个 rank 把局部分片当作全局观察或写入同一产物路径。

## 继续训练的状态和便携模型是两种结果

固定 DDP 中每个 rank 都有完整的模型和 optimizer，因此第一版不需要为了保存而引入 FSDP2 或
分片式模型 checkpoint。一个可继续训练的 bundle 由两类状态组成：公共部分只保存一次 canonical
model 与共同的 optimizer 等训练状态，各进程部分只保存自己的随机数和数据执行状态。两部分必须
绑定同一个数据身份、数据分工、有效全局 batch 和固定拓扑；运行时验证凭证永不写入 checkpoint，
恢复时重新取得。首轮只在完整 epoch 边界保存和恢复，并要求相同进程数、rank 布局、数据分工和
已验证 artifact；不承诺 batch 中途、epoch 中途或改变进程数后继续训练。

即使只从完整 epoch 继续，reader 与运行时随机变换也必须能从 run seed、epoch 和经过认证的
sample ID 重新构造，或由 `DataBuilder` 明确提供版本化的 epoch-boundary resume capability。
依赖 persistent worker 已消耗 RNG、却只保存一个初始 seed 的数据路径在首轮被拒绝；不能把这种
路径称为 exact resume，也不为此偷偷加入 mid-epoch cursor。

各 rank 先把自己的必需状态写入私有目录并报告摘要，公共状态与所有 rank-local 状态都写完且
核对成功后，主进程才原子发布 bundle 的结果清单。缺失、重复、摘要不符或发布中断的目录都不是
可恢复状态。另一方面，canonical model 还要按现有 checkpoint-v12 语义导出一份普通便携文件，
让单进程 sampling 和 Evaluation 无需知道训练曾经使用八个进程。它仍要明确 raw/EMA 选择，携带
Process、已声明 inference assets、固定 recipe 与来源，并在未初始化进程组的新进程中通过现有
sample/evaluate 路径验证。分布式训练 bundle 不能伪装
成 `best.pt`，便携 v12 也不能冒充包含完整 optimizer 与 rank-local RNG 的恢复状态。

便携文件采用独立、可重试的原子发布，而不是成为每一个继续训练 bundle 的必需成员。它必须绑定
已经提交的 common-state digest、epoch、`global_step` 和 raw/EMA 选择，重试导出不得推进训练状态。
若一次运行要求发布 `best` 或 `latest` 便携文件而导出或单进程验证失败，已经提交的 bundle 仍是
有效恢复证据，但这场运行不能发布完整成功结果，也不得让旧 `best.pt` 冒充当前同步点。

## 失败时宁可没有结果，也不能有伪成功结果

首版把故障处理建立在固定成员的整场失败上。进程组有明确超时，顶层 worker 保留原始 traceback，
日志按 rank 分开并由主进程给出一次失败摘要。某个 worker 崩溃、collective 超时、收到中断或
写状态失败时，由 launcher 结束其余 worker；异常清理不会再要求所有 rank 进入一个可能永远等
不到的 barrier。私有输出尽力清理，无法清理的目录保留为明确的失败证据，但不会发布成功清单。

进程组超时只能约束已经进入 collective 的通信，不能自动终止所有 rank 一起卡住的 DataLoader、
共享存储读取或 checkpoint 写入。首轮必须为这些阶段设置可观察的进度期限和 launcher 硬期限，
或明确把强制停止交给公司调度系统；没有这层监控时不宣称任意 I/O 卡死都能有界退出。故障矩阵
还要覆盖 reader 卡死、checkpoint 写入卡死、DataLoader 子进程泄漏，以及下一场运行能否重新取得
工作目录或数据锁。

成功退出时每个进程按当前 PyTorch 约定释放进程组；某个 rank 已死亡的异常路径不为清理再建立
一次 barrier。8×H200 的验收既要证明训练结果，也要记录拓扑、通信、存储、DataLoader worker、
主机内存、每卡显存、数据等待和 checkpoint 停顿，防止用硬件环境问题解释框架错误。调试选项
不能未经测量成为默认值；source checkout 和 wheel 必须一致。Windows PC 仍可准备数据和运行
单卡路径，但不因此成为首轮 DDP 的承诺平台。多 worker 的启动和资源矩阵留在附录。

## 多进程采样在训练闭合以后单独证明

多进程采样保留为本方向的后续功能，但不再与第一项 DDP 训练一起交付。它需要另一套独立证据：
稳定的全局样本编号把请求分给各 rank，每个 rank 只写私有临时结果，主进程在不把大 tensor 集中
复制到内存的前提下检查缺失和重复，最后只发布一份普通采样结果。只能写单文件的 writer 还要
拥有自己的合并契约。改变进程数不能改变样本身份，失败不能留下局部成功目录。

这些规则与训练的数据覆盖、梯度同步和恢复状态不同，提前实现只会扩大大数据训练的首轮风险。
分布式 Evaluation、FSDP2、多节点、弹性成员、epoch 中途恢复、张量并行、流水线并行、三维
并行、DeepSpeed、Megatron、ZeRO 以及按全局 batch 自动调整学习率也都保留为各自需要测量和
完成标准的后续方向；完成固定 DDP 不表示它们已经可用。

## 什么证据会让计划开始，又怎样证明它已经完成

本计划继续保持 Parked。开始前要先选定一项已经通过正式 Evaluation 的大数据任务，用可重复的
单卡测量证明在要求的 effective global batch 和训练质量下，gradient accumulation、混合精度、
编译和 activation checkpointing 等办法仍不能满足吞吐或总训练时间预算，同时确认模型和一个
microbatch 仍能放入单卡，
并固定当时的 Linux、8×H200、CUDA、PyTorch、NCCL、数据格式和存储环境。大数据的 prepare、
verify、copy、adopt 和单卡读取可以在此之前独立完成；只有服务器多卡训练依赖本计划。

完成证据分成四类。正确性要证明现有单进程行为不变，DDP 确实同步模型，并在相同 effective
global batch 下得到预先约定的数值等价结果；数据与恢复要证明没有漏样、重复扫描或伪造继续状态；
性能要分别报告固定总工作量和固定每卡 batch 的 1/2/4/8 卡结果；故障要证明任何进程失败都不会
留下成功清单。完整 CPU/Gloo、CUDA/NCCL、checkpoint 和故障注入矩阵保存在附录。

性能对照要分别定义冷启动和稳定热缓存，固定存储位置与 OS page-cache 条件，记录 warmup、重复
次数、方差、GPU clocks、ECC/Xid 和 thermal throttling；不能让后跑的 4/8 卡试验因为 2 TB RAM
已经缓存数据而获得不可比较的优势。

只有这些证据在 source checkout 和 wheel 中重复成立，用户文档才能列出那一组经过验收的平台、
设备和 backend；未验证组合继续被拒绝。

平台调研、候选接口、数据分工算法、全局指标状态、checkpoint 事务、故障矩阵和后续并行方式的
详细草案保存在
[多设备支持研究附录](notes/distributed-training-and-inference-support-plan/research-and-api-draft.md)。
附录不是当前接口；真正开始实施时，所有外部库和硬件结论都必须按当时依赖版本重新核对。

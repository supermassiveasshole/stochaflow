# 固定单机 DDP 的实现取舍与待复核问题

> 文档类型：实施决策记录
>
> 排期状态：不参与排期
>
> 当前可用性：本页记录固定单机 DDP 软件首版已经采用的边界；CLI、核心循环和 CPU/Gloo 正确性测试已经存在，但真实 CUDA/NCCL 与 8×H200 验收尚未完成。它不是用户文档，也不表示里程碑已经通过验收。
>
> 最后复核：2026-08-13

这次实现不是把 `torchrun` 套在现有命令外面。八个普通 `Trainer` 会各自读取完整数据、创建输出目录、
写日志和保存 checkpoint；即使模型参数碰巧同步，也不是一场可恢复、可核对的训练。因此本轮先决定
谁拥有运行会话、哪个模型参加同步、数据怎样证明分工，以及一次更新什么时候才算共同发生。下面
同时保留已经采用的决定和仍需维护者事后复核的限制，避免实现细节悄悄变成长期产品承诺。

## 训练循环不在旧 Trainer 里增加并行模式

现有 `Trainer` 继续只执行单进程训练。固定拓扑 DDP 使用独立的 `DDPTrainer`，以后若选择 FSDP，
也要按它自己的参数分片、optimizer、EMA 和 checkpoint 生命周期设计独立执行者。首轮不通过
`if distributed`、`if fsdp` 或覆写一组私有方法来复用旧循环；两个执行者只复用已经稳定的数据
对象、纯构造函数和窄能力接口。重复的循环代码要等两套语义都稳定后再判断是否值得提取。

`TrainingPlan.primary_model` 始终是负责身份、EMA、checkpoint 和便携导出的原始模型。DDP 包装对象
只负责训练 forward。`TrainingBuilder` 可以选择实现公开的
`TrainingExecutionBindingBuilder` 能力，把包装后的执行模型重新交给 Strategy；缺少这项能力的
第三方 Builder 在任何 forward 发生前被拒绝。绑定只返回一份执行 Strategy，不能趁机替换 Process、
Objective、辅助模块或 inference asset。这个能力会成为 Extension 契约，需要独立第三方实现测试。

事后需要复核：公开这一能力意味着现有自定义 Builder 若想使用 DDP 必须显式适配，但单进程用法
完全不变。另一种“core 猜测 Strategy 内模型字段并替换”的方案已拒绝，因为它既不可靠，也违反
任务组合边界。

## 启动方式必须显式选择 DDP

首轮采用专用的 `--ddp` 入口选择，不建立一个提前容纳 DDP、FSDP、张量并行和未知后端的通用
`parallel_mode` 配置。检测到 `WORLD_SIZE > 1` 却没有 `--ddp` 时会直接失败，避免八个进程误跑
单进程入口；带 `--ddp` 却没有合法 `torchrun` 环境也会失败。设备由 `LOCAL_RANK` 决定，DDP 下
不接受用户指定某个固定 CUDA 编号。

训练 operation 拥有会话的进入与退出规则；私有 session helper 只解析 launcher 环境、绑定本地
设备、初始化和销毁 process group。operation 在接触 DataSource、DataBuilder、logger 或输出目录
以前选择 DDP 执行家族，并把整个 composition 与 `DDPTrainer` 调用放在自己的 `try/finally`
会话范围内。`DDPTrainer` 拥有训练循环和 all-rank 提交语义，不负责重开或接管进程组。不会只凭
环境变量静默改变普通 `train` 的语义。

事后需要复核：命令行选择是否应在未来成为完整配置的一部分。目前它属于运行拓扑，与固定
topology resume bundle 一起保存和核对，但不会写进任务 YAML，也不会据此自动调整学习率。

## 首轮只接纳能证明分工的数据

`DataBuilderContext` 只取得 `rank` 和 `world_size` 的不可修改投影，不取得 CUDA device、backend 或
collective。普通 batch 仍是任务自有的不透明对象；core 不读取 tensor 形状或约定键。要进入 DDP，
DataBuilder 返回的训练 reader 必须实现窄的 ranked-data 能力，逐窗口给出 Builder 自己认证的
ordinal、实际样本数、loss 权重和分工 token。动态过滤、weighted mixture、padding、重复样本和不完整
gradient-accumulation 窗口在首轮被拒绝，而不是由 core 猜测或偷偷修正。

第一条内置路径只接 `class_labeled_image` 这种已有稳定样本身份、按样本和 epoch 派生随机变换的
map-style 数据。每个 rank 的 YAML `batch_size` 仍是 per-rank batch；有效全局 batch 是
`world_size × per-rank batch × gradient accumulation`，不自动缩放学习率。训练分工丢弃不足一个
完整全局更新窗口的尾部，并在计划中明确报告丢弃数量。

训练期 validation 首轮采用 rank 0 的完整数据视图，其他 rank 不执行 prediction，但计划会向
所有 rank 声明 rank 0 的精确 batch 数；每个 ordinal 都有一次失败心跳，最终再共同核对 coverage
receipt、结果和选择决定。这避开了当前 Gaussian validation 随机数随分片变化的问题，也不需要用
padding 制造重复样本，同时让 rank 0 中途读取失败能传播给空闲 rank。分片 validation 留到
evaluator 能按 sample/protocol identity 派生随机数并且 DataBuilder 能提供有界覆盖证明之后。

事后需要复核：ranked reader 与 coverage 能力是否作为公开 Extension contract 保留；本轮选择“是”。
它增加了第三方 DDP DataBuilder 的实现成本，却让普通 iterable 不会被误判为可安全分布式运行。

## 第一个正确路径先支持 fp32 与 bf16

非最后一个 microbatch 的 forward 和 backward 都必须处于 DDP `no_sync()` 范围，最后一个才同步
梯度。所有 rank 在 optimizer mutation 以前共同确认窗口完整、loss 权重一致且梯度有限；任何一个
rank 拒绝，本次 optimizer、EMA、scheduler、`global_step` 和 Metric 都不推进。optimizer 调用本身
若在某个 rank 抛错，整场运行失败并从上一完整 epoch 恢复，不能假装内存修改可以回滚。

首轮产品路径只接纳 fp32 和 bf16。当前 `GradScaler.step()` 与 `update()` 合在一起，无法只用公开
PyTorch 契约完成“一张卡 overflow、所有卡用完全相同 scale/growth tracker 跳过”的提交协议。
fp16 不会通过访问 GradScaler 私有字段勉强实现；它会在 staged precision contract 和故障注入测试
闭合后再加入。这不影响目标 H200 上的 bf16 验收，但必须在使用文档中明确拒绝，而不是静默退化。

## 当前 Diagnostic、live Evaluation 与 train-phase Metric 先拒绝

普通 validation 由 rank 0 使用 canonical model 和完整 validation view 执行，结果随后广播。当前
Training Diagnostic 和训练期 live Evaluation 没有 rank-zero/full-view 或 all-rank merge 声明，
每个进程直接运行会把局部数据误当全局数据并争抢产物路径，因此 DDP composition 会拒绝它们。
正式独立 Evaluation 仍在训练结束后用便携 checkpoint 单独运行，不因这次实现重新打开。

首轮也拒绝 train phase Metric。validation Metric 只在 rank 0 完整视图中运行，因此继续使用现有
`MetricEngine`，不扩大它的分布式契约。以后若要分片 validation，每个 provider 必须声明并证明
可合并状态，不能平均各 rank 的局部均值或从 TorchMetrics 基类推断能力。

事后需要复核：哪些 Diagnostic 或 live Evaluation 最先值得增加明确运行策略。目前没有被选中的
用户结果，因此不为了“支持矩阵完整”提前发明通用 mode 字段。

## 继续训练 bundle 与便携 v12 是两种结果

固定 DDP 的 exact resume 使用独立 bundle：一个 common state、每个 rank 一份 RNG/reader state，
以及最后原子发布的 inventory。它只在完整 epoch 边界保存，并绑定相同 world size、rank 布局、
数据分工、per-rank batch 和 accumulation。runtime receipt 不写入；每次恢复仍要重新取得当前数据
验证凭证。首轮只接纳能按 `(run seed, epoch, authenticated sample identity)` 重建的 reader，或者
显式提供版本化 epoch-boundary state 的 reader。

Sampling 与 Evaluation 继续消费普通 checkpoint-v12。为了不同时发明 v13，便携文件仍保留 v12
校验器要求的 precision、RNG 和 scaler 字段，但 metadata 标记它来自哪个已提交 common state；这些
字段只是格式兼容信息，不能让 `.pt` 文件冒充 DDP exact-resume bundle。单进程 train resume 和 DDP
resume 都会拒绝错误的结果类型。

便携导出是 bundle 提交后的独立、可重试发布。导出失败不撤销可恢复 bundle，但这场 run 不能发布
完整成功结果，也不能留下可发现的陈旧 `best.pt` 或 `latest.pt`。

事后需要复核：portable 导出的 cadence。首轮按已提交的 latest/best 同步点导出；若大模型导出暂停
成为真实瓶颈，再决定异步或降低频率，不在这次实现里预设。

## 失败工作区和外部副作用

主 rank 在最终目录旁创建隐藏工作区；其他 rank 只写自己的私有日志和 rank-local state。公共
logger、W&B、resolved config、manifest、portable alias 和 structured outcome 只有主 rank 可以发布。
所有进程成功、logger 完成且便携结果验证后，主 rank 才把工作区原子改名为最终 run 目录。

若失败前没有完整 bundle，工作区尽力清理；若已有完整 bundle，则保留隐藏工作区和失败说明，
但不让普通 checkpoint 搜索发现其中的 alias。清理逐项尝试，不能覆盖最先发生的训练异常。

事后需要复核：W&B 这类远端副作用无法随本地目录回滚。本轮只保证它由 rank 0 创建一次，并在
失败时标记失败；若用户要求“最终目录未发布就完全不可见”，必须选择支持 staging/commit 语义的
外部 logger，或禁用该后端。

## 暂不宣称 TB 级数据准入已经解决

当前 `DataArtifactStore` 的 full-verification receipt 只能由每个进程在当次请求中取得，不能把 rank 0
的 handle 广播给其他 rank。第一条小规模正确性路径允许每个 rank 独立验证并比较 artifact identity；
验收会记录实际重复读取和哈希的字节数。它不能成为几十 TB 数据已经高效准入的证据。

大数据计划需要另行定义由 Store 签发、带时效和位置约束的 node/run attestation，或者接受每个节点
重新 full verify。DDP runtime 只比较有界身份事实，不伪造或序列化 DataArtifact receipt。

## 仍由验收环境决定的事项

以下事项在代码正确性闭合后仍不能由开发机替用户决定：

- 第一项真实大数据 workload、正式 Evaluation 基线和允许的数值误差；
- Linux、8×H200、CUDA、PyTorch、NCCL、driver、拓扑和共享存储的精确版本；
- 公司 scheduler 对 DataLoader、共享存储和 checkpoint I/O 卡死提供的 hard deadline。process-group
  timeout 只能约束已经进入 collective 的进程，不能承诺终止所有外部 I/O 卡死；
- fixed-topology equality 是否还要绑定 GPU 型号。当前 manifest 不记录 GPU UUID、型号或 driver；
  首轮 8×H200 验收应在外部验收记录中保存这些环境证据，再决定是否把其中任何字段提升为稳定
  manifest 契约。当前 exact resume 只强制单机、world/local world size、rank 映射、
  backend/device type、数据分工与 batch 语义相同，以便在另一台等价八卡服务器继续训练。

当前代码处于“软件首版已实现、等待平台验收”。没有这些实机证据时，ROADMAP 不能把 DDP 标为
Done；公开用户文档可以说明 Linux 单机 CUDA/NCCL 的软件准入边界，但不能把某组 8×H200、driver
或存储组合列为已验证环境，也不能给出未经测量的吞吐和容量结论。

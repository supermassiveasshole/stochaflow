# Distributed 支持：PyTorch 调研、运行契约与历史阶段

> 本附录服务于
> [`Distributed Training 与 Inference 支持计划`](../../distributed-training-and-inference-support-plan.md)。
> 内容是未来重开时的研究记录与设计草案，不是当前 API、支持矩阵或版本承诺。
> 所有上游语义必须在启动前根据当前 lockfile、目标硬件和官方文档重新验证。
> 最后核对：2026-08-13

## 首轮只解决固定单机 DDP 训练

首轮实现范围是 Linux、单机、固定进程数、非弹性的 DDP 训练。每个进程绑定一张 CUDA
设备并保存完整模型副本；完整模型必须能放入一张目标设备。`torchrun` 只负责启动同一个
训练入口，Stochaflow 不实现 launcher、调度器、弹性成员管理或集群控制面。

历史讨论使用 D0–D4 定位首轮工作。它们只是本附录中的实施顺序，不是路线图状态，也不表示
工作已经获准开始：

| 阶段 | 要闭合的结果 | 明确不包含 |
| --- | --- | --- |
| D0 | 用真实 workload 证明 gradient accumulation 等单卡办法仍不能满足时间或吞吐预算，并冻结单机平台、数据、batch、指标、恢复与故障契约 | 写实现、扩大平台承诺 |
| D1 | 在任何运行副作用前建立固定分布式 session，所有 rank 共享一份运行身份和输出目标 | 自动拉起进程、弹性 restart、多节点 rendezvous |
| D2 | 独立 `DDPTrainer` 用 execution binding、rank-aware data、梯度累积和全局 step/Metric/selection 形成一条正确训练路径 | FSDP2、多个可训练根、通用并行 mode |
| D3 | 发布一份 DDP 继续训练状态和一份普通 portable checkpoint，并验证 epoch-boundary exact resume | DCP、reshard、mid-epoch cursor、改变 world size 恢复 |
| D4 | 在目标 Linux CUDA/NCCL 与 8×H200 机器上完成正确性、故障和扩展效率验收，再公开实际支持范围 | 用 CPU/Gloo 或 mock 推导生产支持 |

Replicated sampling 不再与 D0–D4 捆绑。它有不同的全局 sample ID、writer merge、随机数和
发布事务，应在 DDP 训练闭合后作为独立候选实施。FSDP2、Distributed Checkpoint（DCP）、
多节点、弹性运行和正式分布式 Evaluation 同样只保留为研究候选。

## 启动前重新核对 PyTorch 与 NCCL

2026-08-13 的核对只用于修正计划边界，并没有形成平台支持承诺。真正启动 D0 时至少重新确认：

1. `torchrun` 提供的 rank、local rank、world size、failure propagation、restart 和 rendezvous
   语义，以及顶层异常记录方式。
2. `torch.distributed` 在目标 PyTorch、CUDA、NCCL 与 Linux 组合中的初始化、timeout、
   async error、teardown 和 multiprocessing 约束。
3. DDP 对参数注册顺序、unused parameters、static graph、mixed precision 与 gradient
   accumulation 的当前约束，特别是 `no_sync()` 必须同时包住 forward 和 backward。
4. DataLoader worker 的 start method、共享内存、pinned memory、文件句柄与 NUMA 行为。
5. NCCL watchdog、flight recorder 和 debug 环境变量的当前名称、默认值与失败行为。

不应仅凭 import 成功认定支持。CPU/Gloo 可以检查 session、collective 顺序、失败传播和小型
DDP 等价性，但不能证明 CUDA/NCCL 正确、稳定或高效。目标组合还需要一次可执行 spike：初始化、
累积 forward/backward、一次 optimizer step、Metric merge、checkpoint save/load、单 rank
异常退出、SIGINT 和资源释放。

## 技术选择不能伪装成一个 mode enum

| 需求 | 当前选择 | 采用条件 | 当前归属 |
| --- | --- | --- | --- |
| 能放入单卡的多设备训练 | DDP | 在所需 effective global batch 与质量下，gradient accumulation 等单卡办法仍不能满足时间或吞吐预算 | D0–D4 |
| 固定 DDP 训练恢复 | common state + rank-local state | 相同单机拓扑、完整 epoch 边界 | D3 |
| 普通采样与 Evaluation | portable full-state checkpoint | 与现有 inference projection 对齐 | D3 导出，仍单进程消费 |
| 多设备采样吞吐 | replicated SPMD | DDP 训练已闭合，真实采样预算需要 | 独立后续候选 |
| 参数与 optimizer state 分片 | composable FSDP2 | 模型或训练状态不能放入单卡 | 独立研究候选 |
| 分片状态保存与 reshard | DCP | FSDP2 状态确实需要 | 与 FSDP2 一起评审 |
| 多节点或弹性运行 | `torchrun`/外部 scheduler 的相应能力 | 单机仍不满足且有真实网络环境 | 独立研究候选 |
| 正式分布式 Evaluation | exact distributed execution | 单进程 Evaluation 超出明确预算 | 独立研究候选 |

DDP、FSDP2、replicated sampling 和 distributed Evaluation 拥有不同 composition、state、
failure、performance 与运行产物（artifact）语义。它们可以复用窄的 session 或 collective
能力，但不能成为一个带大量可选字段的公共基类。

### 首轮之后仍需单独评审的方向

| 候选能力 | 开始评审所需证据 | 必须保持的边界 |
| --- | --- | --- |
| replicated sampling | 一个已验收的 portable checkpoint 采样请求超过时间预算，且单卡 batching、编译和 writer 优化仍不足 | 稳定 global sample ID、world-size-independent RNG、rank shard staging、无重无漏合并；不复用 training config |
| FSDP2 training | 完整模型、optimizer 或必要 EMA 状态不能放入一张目标设备；有可执行 PyTorch spike | 明确 sharding plan、DTensor optimizer/EMA、portable export；不把 DDP wrapper 抽象扩成通用 mode |
| DCP | FSDP2 已证明需要 sharded state 或 load-time reshard | DCP 只提供保存原语；Stochaflow 仍拥有 schema、完整 inventory 与原子发布 |
| multi-node DDP/FSDP2 | 单机 8 卡仍不能满足容量或时间预算，且有受维护网络与共享存储环境 | 明确 rendezvous、网络 timeout、node failure、存储可见性和运维 owner |
| topology-changing elastic | 固定任务确实需要自动恢复，并能为数据顺序、global batch、RNG 和 checkpoint 重映射给出等价规则 | rank 在 restart 后不可作为稳定身份；不把 launcher restart 当作 exact resume |
| formal distributed Evaluation | 单进程正式 Evaluation 超过预算，训练期分布式 validation 已证明 exact coverage 与 Metric merge | 保持 subject/protocol/全局 sample ID、结果 bundle 和原子发布契约；不把训练 Metric 伪装成正式证据 |
| distributed mid-epoch iterator resume | epoch 重算违反明确时间预算，且每个受支持 reader 都有版本化 per-rank cursor | 必须处理 prefetch、worker state、RNG 和完整 accumulation 边界；不支持 cursor 的 reader 仍只允许 epoch-boundary resume |
| HSDP、tensor/pipeline/3D parallel 或第三方 provider adapter | FSDP2 仍不足，第二个稳定 workload 也需要相同能力 | 每种并行维度或 provider 分别设计、验收；core 不猜模型切分点，不暴露 provider object |
| 自动 learning-rate scaling | 至少两个受维护 recipe 在多种 effective global batch 上得到稳定规则 | 只能显式 opt-in；始终记录实际 batch 与 learning rate，不把线性缩放当通用事实 |

`async_save` 也只在同步保存正确、immutable snapshot 与额外 CPU memory budget 均有证据后
评审。

## Launcher、session 与运行身份的所有权

首轮候选启动方式只用于说明固定拓扑，当前不能执行，也不是公共命令：

```text
torchrun --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  <未来的 Stochaflow 分布式训练入口> <普通训练参数>
```

`--max-restarts=0` 刻意关闭 elastic restart。首轮失败后由用户从已发布的 DDP resume bundle
重新运行；不能让 launcher 以可能变化的 rank 身份偷偷继续同一次 Python 生命周期。

候选顺序必须在任何 DataSource、DataBuilder、logger、output directory 或 checkpoint 副作用
之前建立 session：

```text
validated invocation + prepared extension activation
    -> parse and validate launcher environment on every rank
    -> initialize one fixed process group and bind local device
    -> compare config / extension / code identity across ranks
    -> rank zero creates run ID and intended output target
    -> broadcast one immutable run descriptor
    -> each rank enters its private log and staging namespace
    -> begin artifact admission and runtime composition
```

launcher environment 拥有 rank、local rank、world size、local world size 与 rendezvous facts；
配置不能复制这些值。Stochaflow 的私有分布式 session helper 只执行 environment 解析、device
resolution、process-group init/destroy、collective construction 和 topology/config consistency
preflight；`DDPTrainer` 拥有何时进入、何时退出以及失败时必须关闭的生命周期。helper 不启动
child process、不选择 DDP/FSDP、不构造 DataLoader，也不保存 checkpoint。

rank zero 拥有外部可见的 run ID、resolved config、普通 console/logger、最终结果清单
（manifest）和原子发布动作。其他 rank 只能写到由共同 run descriptor 派生的私有位置。每个 rank
可以保留独立故障日志，但不能自行创建第二个正式 run，也不能在其他 rank 未确认成功时发布
success outcome。

以下名称只表达候选职责，不是公共 API 或兼容承诺：

```python
@dataclass(frozen=True, slots=True)
class DistributedRunContext:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    device: torch.device
    backend: str

    @property
    def is_primary(self) -> bool: ...


class CollectiveOperations(Protocol):
    def sum_tensor(self, value: torch.Tensor) -> torch.Tensor: ...
    def max_tensor(self, value: torch.Tensor) -> torch.Tensor: ...
    def all_true(self, value: bool) -> bool: ...
    def broadcast_object(self, value: object | None, *, source: int) -> object: ...
```

通用 `barrier()` 不应成为业务组件用来掩盖顺序问题的默认工具。固定 collective 顺序由 runtime
协议决定；debug barrier 和 teardown 细节留在 session 内部。

## 不把所有并行生命周期塞进一个基础 Trainer

现有 `Trainer` 保持单进程责任，不新增 `distributed`、`fsdp` 或通用 parallel mode 分支。
D2 候选实现使用独立的 `DDPTrainer`：它拥有 DDP session、execution model、`no_sync()`、
all-rank update consensus、rank-local state 和共同失败。以后只有在 FSDP2 的容量证据和状态设计
成立后，才由另一个 `FSDPTrainer` 拥有分片参数、optimizer、EMA、DCP/portable export 等生命周期；
不能把这些条件继续加回 `DDPTrainer`。这些名称是维护者草案，不是当前导出或公共 API。

训练 operation 在任何 Data 访问前，根据已经验证的 invocation 选择内部 Trainer family；用户不
配置 Python 类。`DDPTrainer` 随后打开并拥有 distributed session，在该 context 内调用 composition
创建 rank-aware data 与 canonical `TrainingPlan`，核对 DDP admission，再完成 execution binding
与运行。session helper 只封装 init/destroy 与 collective mechanics，其生命周期始终由
`DDPTrainer` 的 `try/finally` 拥有。共享代码只能是与并行方式无关的窄协作者，例如 Strategy
调用、Metric binding、checkpoint 文件原子写入和结果对象；不能让基础 Trainer 反向认识
DDP/FSDP 状态。若继承现有 `Trainer` 需要覆盖私有 hook、绕过父类顺序或
违反 Liskov substitution，就不继承，而让这些实现满足同一个最小训练 operation contract。只有在
两个实现出现稳定、语义相同的重复以后才抽取 helper，不能预先制造一个带大量 optional hook 的
抽象父类。

## canonical model 与 execution model 必须显式连接

当前 `TrainingBuilder` 先把 primary model 注入 `TrainingStrategy`。如果 core 随后只创建 DDP
wrapper，而 Strategy 仍调用原始模型，forward/backward 就绕过了梯度同步。因此 D2 不能靠
“给 Trainer.model 包一层 DDP”完成。

候选 composition 顺序是：

```text
validated distributed invocation
    -> operation selects the internal DDPTrainer family
    -> DDPTrainer opens one fixed distributed session
    -> resolve verified artifacts + canonical model/process/objective/assets
    -> TrainingBuilder builds a rank-aware canonical TrainingPlan inside the session
    -> validate first-round DDP admission
    -> move canonical primary model to local device
    -> wrap primary model as the DDP execution model
    -> the Builder's narrow execution-binding capability returns a Strategy that uses that model
    -> construct optimizer from canonical primary parameters
    -> construct scheduler / precision / EMA controllers
    -> DDPTrainer consumes the executable plan and distributed capabilities
    -> DDPTrainer closes the session in every exit path
```

canonical model 负责 identity、declaration、EMA、checkpoint 与 portable export；execution
model 负责 forward/backward。两者的关系必须由创建 Strategy 的 Builder 通过窄能力重新绑定，
core 不检查具体 Strategy 类型、字段名或 task 名称。独立 extension Builder 必须通过同一契约
测试，不能只让内置 Gaussian 任务工作。

首轮只允许 primary model 是唯一 trainable root。trainable Process、Objective、auxiliary
module、manual backward、alternating optimizer、closure-required optimizer 和 rank-dependent
forward control flow 都在 composition 时 fail closed。它们需要多个 execution binding 或新的
training-loop family，不能由 DDP 实现猜测。

以下草案仍不是公共 API：

```python
class TrainingExecutionBindingBuilder(Protocol):
    def bind_primary_execution_model(
        self,
        plan: TrainingPlan,
        execution_model: torch.nn.Module,
    ) -> TrainingPlan: ...


class GradientSynchronization(Protocol):
    def synchronized(self, enabled: bool) -> ContextManager[None]: ...
```

FSDP2 partition、auto-wrap 和 DCP planner 不进入上述首轮契约。

## batch、accumulation 与 global step 是同一个协议

首轮把训练 YAML 的 `batch_size` 解释为每个 rank 的 batch，不自动修改 learning rate：

```text
effective global batch
    = per-rank batch × world size × gradient accumulation steps
```

单进程/DDP 等价测试必须匹配 effective global batch 和 optimizer update 数，不能只比较字面相同
的 YAML。运行 manifest 同时记录 per-rank batch、world size、accumulation、effective global
batch 与实际 learning rate。

对非最后一个 micro-batch，DDP `no_sync()` 必须同时包住 Strategy forward 和 backward；只包
backward 不会阻止 forward 准备同步。所有 rank 必须进入相同数量、相同顺序的 trainable
forward/backward。首轮要求每次同步更新具有相同的 per-rank sample count 和 loss weighting；
若某 rank 的最后 batch 不完整、提前 exhaustion、跳过 batch 或返回不同 accumulation 长度，
整次运行 fail closed。不能把 reporting 用的 `loss_aggregation_weight` 当作已经修正 DDP 梯度
权重。

一次更新的候选顺序是：

1. 在进入 trainable forward 前，共同确认所有 rank 都有下一批输入和相同 accumulation 形状。
2. 非最后 micro-batch 在 `no_sync()` 中完成 forward 与 backward；最后一个启用同步。
3. precision controller 完成 unscale 并暴露本地 non-finite/skip 状态，但尚不推进 optimizer。
4. all-rank consensus 决定是否允许本次 optimizer step；任一 rank 报告 non-finite 时，所有
   rank 一致跳过 optimizer、以相同规则更新 scaler，且不推进 EMA、scheduler、`global_step`
   或 Metric。loader/shape/collective 等结构性错误则使整次任务失败，不能降级成一次普通 skip。
5. 成功的 optimizer step 后再次确认 runtime 没有局部失败，再共同推进 EMA、scheduler、
   `global_step`、训练 Metric update 和 checkpoint cadence。

第 3–4 步要求一个明确的 staged precision contract：`prepare` 只做 unscale 和公开的本地
finite/overflow 判定，collective 得出 global decision 后，`commit` 或 `skip` 才调用 optimizer 并让
每个 scaler 做一次完全相同的 transition。不能复用当前把 `GradScaler.step()`/`update()` 合成一个
不可插入 consensus 的单进程动作，也不能读取私有 scaler 字段。测试要在单一 rank 注入 overflow，
并比较所有 rank 的 scale、growth tracker、optimizer、EMA、scheduler 与 `global_step`。

如果一个 rank 在 optimizer mutation 期间崩溃，内存中的更新无法事务回滚；正确行为是让整次
任务失败，并且不发布该 step 的 checkpoint、Metric 或 success outcome，而不是让剩余 rank
继续。`global_step` 表示已经由所有 rank 完成并确认的 optimizer update，不是局部 batch 数。

## DataBuilder 必须给出有界且可核对的数据分工

`DataBuilderContext` 可以候选性地携带 immutable `DistributedRunContext`。Builder 仍拥有
Dataset、sampler、batch sampler、collation 和 worker pipeline；core 不能拆解任意 batch，
也不能事后替换一个已经组装好的 DataLoader sampler。

Builder 返回的执行证据至少要说明：

- assignment policy、world size、rank、sampler/reader identity 与 epoch seed；
- 每 rank 预计样本数、micro-batch 数、optimizer window 数和 exhaustion 边界；
- train 的 drop/pad/duplicate policy 及其 loss-weight 影响；
- training-time validation 的全局 expected coverage 和 sample identity 规则；
- `set_epoch()` 怎样传播到 sampler、batch sampler、reader 和 worker seed；
- 索引、shuffle buffer、prefetch 和 worker memory 的上界，而不是只给平均值；
- 是否只支持完整 epoch 恢复，首轮不声明 iterator cursor。

这些是 plan-time expected facts，还需要一个 DataBuilder-owned 的运行事实通道。它与 iterable 同步，
但不改变 batch 结构；每个 optimizer window 至少产生单调 ordinal、实际 sample weight/count、
assignment token、exhaustion 标记和有界 observed-coverage update。`DDPTrainer` 只验证这组控制事实
与声明一致，再进入 trainable forward；它不解析 opaque batch，也不从 tensor shape 猜 batch size。
这份通道是 Data 执行契约的一部分，不是通用 Dataset/DataLoader registry 或 sample schema。

training-time validation 同样需要 Data owner 产生 observed coverage evidence。首轮可以采用两类实现：

- prepared manifest 已给出互斥 ranges，reader 按范围报告精确 processed count/terminal token；或
- task reader 把稳定 ID 流式写入有界的外部核验过程，最后证明 union/intersection。

只有 expected partition、总 count 或可交换 digest 不能证明无重无漏。若 reader 无法提供上述证据，
就不能进入 exact distributed validation；不得通过给 core 的通用 batch 强加 `sample_id` 绕过边界。

对大数据 exact coverage，优先让 prepared manifest 提供互斥 range/partition 及其摘要；任意
sample ID 采用外部排序归并或其他有界算法核验。不能把数亿 ID `all_gather` 到 rank zero，也不能
只用 count 加一个可交换 digest 证明无重复。weighted mixture 的声明分布本来就可能重采样，因此
train 的目标是证明行为符合显式 sampling policy；validation/test 才要求 exact union 且无交集。

普通 `DistributedSampler` 只适合长度固定、顺序稳定的 map-style dataset。它在不丢尾部时可能
用额外索引补齐各 rank，因此不能直接证明精确 validation。Iterable、预分片大文件、bucket、
mixture、多分辨率和自定义 batch sampler 必须由各自 owner 实现 rank/worker assignment。
几十 TB 数据的 shuffle 不能在每个 rank 物化全量 sample ID 或完整 `randperm`；reader 必须采用
有界索引、确定性 shard permutation 或可证明等价的流式策略。

训练可以采用明确的 equal-step drop policy。padding 只有在重复样本和 loss weighting 语义已
声明并测试时才可进入训练；首轮更安全的默认是拒绝不能形成等长更新窗口的 DataBuilder。
training-time validation 必须无遗漏、无重复；使用 padding 的副本不能成为新的 Metric 证据。
正式 standalone distributed Evaluation 不属于 D0–D4。

## TB 级 artifact verification 不能靠广播 receipt 绕过

现有 Data 合同要求每个正式 Builder 在当前请求中取得 Store 签发的验证 receipt；receipt 是
进程内、短期、不可序列化的证明。rank zero 不能把 receipt、`DataArtifact` handle 或一个布尔
“已经验证”广播给其他 rank，也不能让 Builder 直接调用 Store 伪造 admission。

这意味着未经新设计时，八个 rank 都做 full verification 是正确但可能把几十 TB 内容重复哈希
八次。D0 必须为目标数据选择并测量以下一条路径：

1. 每个 rank 独立 full verification，记录实际读取/哈希字节和启动时间；或
2. 先修改规范与 Store 边界，由 Store 根据一次可信的 node/run-scoped verification 签发每个
   进程自己的当前 receipt，并明确 locator、文件系统快照、mtime/size 不足以证明内容、时效、
   trust domain、失效与并发首次访问规则。

第二条是 Data 能力，不应藏进 distributed runtime。它必须拒绝内容替换、stale locator、不同
mount 指向不同字节、node-local cache 不一致和不完整 prepare。若这项 admission 尚未闭合，
D0–D4 可以在小型数据上验证 DDP 正确性，但不能宣称 TB 级 once-prepared admission 已完成。

## Metric merge 决定全局训练是否可信

不能平均各 rank 的 local mean。候选 Metric admission 必须在 composition 时确认下列路径之一：

- Metric/provider 已拥有经过测试的 distributed state reduction；或
- training metric binding 提供该 Metric 所需的可合并充分统计量，例如 sum/weight、count、
  confusion matrix 或其他明确的 typed state。

Stochaflow 不建立一个跨任务、跨 Metric 的通用 batch 或统计 schema。没有分布式 merge 能力的
extension Metric 在 DDP composition 时 fail closed；不能先本地 compute 再猜怎样合并标量。
所有 rank 按相同顺序进入 Metric compute/reset collective。训练 step 产生的 `MetricUpdate`
只能在 optimizer update 达成全局成功后提交，避免一个 rank 已累计而另一个 rank 已跳过。

rank zero 可以执行 best/early-stop decision 与普通日志，但它使用的是全局 validation 结果。
决定随后广播给所有 rank，使 scheduler、停止点、checkpoint selection 和下一次 collective 顺序
保持一致。Diagnostic 必须明确为 all-rank、rank-zero-with-portable-state 或 unsupported；
不允许依赖偶然调用顺序避免重复副作用。

training-time validation 如果各 rank 读取不同数量的记录，不应继续调用每次 forward 都含 collective
的 DDP execution model。候选路径是在同步训练边界后，用各 rank 已一致的 canonical model 快照执行
本地推理，再按固定顺序合并 Metric state；或者让 rank zero 使用完整数据视图单独执行并广播结果。
哪条路径被维护必须在 Builder/Metric composition 时决定，不能让空 rank 提前离开 collective。

分片本地推理若含随机性，seed/counter 必须由 evaluation protocol identity 与稳定 sample identity
派生，改变 rank、world size 或 assignment 不能改变预测；否则只能采用 rank-zero 完整视图。零记录
rank 必须能创建同一 Metric merge identity 并贡献单位元，所有 rank 先合并 sufficient state 再
compute。本地 evaluator/model path 不能隐含任何 process collective；用 `N < world size`、极不均匀
shard 和随机 evaluator 做故障测试，避免某个空 rank 提前离开后让其他 rank 死锁。

## DDP checkpoint 是 common state 加 rank-local state

固定 DDP 在每个 rank 保存模型和 optimizer 的完整副本，所以首轮不需要把 common training
state 人为分片，也不需要 DCP。DDP resume bundle 应清楚区分：

- common state：canonical primary model、Process/Objective/auxiliary/monitor 的已声明状态、optimizer、scheduler、
  precision scaler、EMA、completed epoch、global step、selection/early-stop state、配置与 artifact
  identities、`DataArtifactBindings`、loader recipe 与验证准入摘要；
- rank-local state：每 rank RNG、下一 epoch 的 sampler/reader seed 和必要的 worker seed base；
- topology：固定单机 world size、rank mapping、backend/device facts 与 bundle inventory；
- execution facts：per-rank/effective-global batch、accumulation、partition/drop policy 与 expected
  coverage；
- portable checkpoint：按现有 v12/inference projection 规则独立导出的完整推理资产，不含
  optimizer、scheduler、scaler、训练 RNG 或 rank-local 状态。

common state 可以只由 rank zero 写，但保存前所有 rank 必须确认处于同一 completed-epoch、
`global_step`、selection 与训练状态边界，并比较所有 checkpointed common state 的 canonical
摘要。非 primary 可变状态若没有同步或一致性证明就在 DDP composition 时拒绝。候选发布事务是：rank zero 创建一次 save identity 和
private staging root；所有 rank 接收它；rank zero 写 common state，每个 rank 写自己的 local
state；各文件产生 size/digest；所有 rank 共同确认 inventory；最后只有 rank zero 原子发布
manifest。任一缺失、损坏、超时或写入失败都使 staging 非正式，不能被目录 resume 选中。

恢复先在临时对象或未提交状态中验证 schema、完整 inventory、digest、config、artifact identity、
固定 topology 和每 rank local state，再把 common state 装入所有 rank，把 local state 装入对应
rank。只有 all-rank load/validation 成功后才能进入下一 epoch。恢复失败不能让部分 rank 继续。
checkpoint 只保存验证 policy 与事实，不保存或复用进程内 receipt；恢复时每个当前 Data 请求仍须
取得新的 Store 签发证明。

首轮只在完整 epoch 和完整 optimizer/accumulation 边界保存，不序列化运行中的 iterator、data
cursor、prefetch queue 或部分 gradient。改变 world size、load-time reshard 与 mid-epoch
resume 都需要独立契约。DCP 能加载某种 state layout，也不会自动定义 global batch、RNG、
data order 或整个 bundle 的原子发布。

“epoch-boundary exact”只适用于能由 `(run seed, epoch, authenticated sample ID)` 重建下一 epoch
顺序与变换的 stateless reader，或 Builder 明确提供并验证了版本化 epoch-boundary resume
capability 的路径。只保存 worker seed base 不能恢复 persistent worker 已经消耗的 RNG；这类 reader
在 D3 fail closed，除非以后按独立 mid-epoch/cursor 工作补齐状态。

portable export 是独立的正式文件。它必须能被未初始化 `torch.distributed` 的普通单进程
sample/evaluate 读取，明确 raw/EMA 选择，并携带现有 v12 所需的 Process、declared inference
assets、fixed recipe 与 provenance；其模型结果与同一同步点的 canonical DDP state 对齐。DDP resume bundle 不能伪装
成普通 `.pt` checkpoint，portable 文件也不能被用来声称可以 exact resume 训练。

portable export 采用独立、可重试的原子发布事务，并绑定已提交 bundle 的 common-state digest、
epoch、global step 与 raw/EMA selection。它失败时不撤销有效 resume bundle，也不推进任何训练
状态；但要求发布 `best`/`latest` 的运行不得发布完整 success outcome，且旧 portable 文件不能保留
当前别名。重试只能从同一 common-state digest 重新投影并在 clean process 中验证。

## Failure、torchrun 与 NCCL 运维边界

首轮固定使用单机非弹性进程组。顶层 worker 应使用当前 PyTorch 推荐的
`torch.distributed.elastic.multiprocessing.errors.record`（通常写作 `@record`）或其当时的
正式替代方式，使 launcher 保存最早可观察的 worker failure、rank、phase 和可得 traceback；不能
承诺它总能识别真实因果根源。任一 rank 异常、collective timeout、
SIGINT 或被杀后，整个任务失败；其他 rank 不得继续发布 success outcome。没有未发布
checkpoint 可以安全提交时，由用户从上一个已完成 bundle 重新运行。

不能在 exception path 使用无限 barrier 等待已经死亡的 rank。`monitored_barrier` 只适合受支持
backend 的诊断，不是生产 liveness 协议。正常执行依赖固定 collective 顺序和明确 process-group
timeout；失败后由 process-group/NCCL 监控、launcher 与进程退出共同终止整次运行。

以下变量只作为按当前 PyTorch/NCCL 版本核对的 runbook 候选，不是永久默认值：

- `TORCH_NCCL_ASYNC_ERROR_HANDLING` 与 `TORCH_NCCL_ENABLE_MONITORING`；
- `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`；
- `TORCH_NCCL_TRACE_BUFFER_SIZE` 与 `TORCH_NCCL_DUMP_ON_TIMEOUT`；
- `NCCL_DEBUG` 与每 rank 独立的 `NCCL_DEBUG_FILE`。

打开 flight recorder、dump 或高等级 NCCL debug 会消耗资源并改变时序，只能用于可复现排障。
项目文档应记录默认 timeout、如何定位 rank 日志、如何收集 topology 与 NCCL trace，以及何时
撤回 debug override。不能把 debug 环境变量当作正确性测试。

DataLoader 多进程还要避免 fork-unsafe 的 NCCL/CUDA 状态。目标 worker start method 必须按当前
PyTorch 文档和实际 reader 验证，并优先采用 `spawn` 或 `forkserver`；8 rank × N workers 的共享内存、文件句柄、pinned memory、
NUMA placement 和退出回收都属于 D4，而不是用户自己猜测的调参细节。

正常成功路径由每个 trainer process 在通信结束后释放 process group。若已有 rank 死亡，异常路径
不为 teardown 新增一个 barrier；它依赖 timeout、NCCL 监控、launcher 和进程退出完成有界终止。

process-group timeout 不覆盖所有 rank 同时停在 DataLoader、共享文件系统或 checkpoint I/O 的情况。
D4 必须为 reader progress、checkpoint progress 和整个 worker 设置可观察 deadline，或明确依赖外部
scheduler 的 hard deadline；后一种情况下，框架不声称能有界终止任意 I/O hang。测试还要注入
reader hang、checkpoint write hang、DataLoader child leak，并确认下一次 run 不被旧锁或孤儿进程阻塞。

## 8×H200 验收必须同时证明正确和更快

D4 先记录服务器 GPU/NVLink/PCIe/NUMA 拓扑、NCCL 版本、驱动、CUDA、PyTorch、容器限制、
`/dev/shm`、memlock 与 open-file limit。使用当前官方 `nccl-tests` 建立 2/4/8 卡 collective
正确性和带宽基线；这些结果是环境证据，不替代 Stochaflow 训练测试。

至少选择一个已通过正式 Evaluation、单卡行为稳定且输入管线可测的真实 workload：

- strong scaling：固定 effective global batch 和工作量，比较 1/2/4/8 卡 wall time、samples/s、
  scaling efficiency、optimizer updates/s 与结果等价性；
- weak scaling：固定 per-rank batch，随 1/2/4/8 卡扩大 effective global batch，明确 learning
  rate 不自动变化，并只比较适合该实验的问题；
- 逐次记录 data wait、host-to-device、forward/backward、gradient communication、optimizer、
  validation、checkpoint stall、GPU utilization、RAM、pinned RAM 与 HBM peak；
- 分开测 cold-start 与 steady-state，固定 storage/OS page-cache 条件、warmup、重复次数与方差，并
  记录 GPU clocks、ECC/Xid 和 thermal throttling；
- 同时变化 rank 数和 worker 数时，记录吞吐、内存与文件句柄预算，证明 reader 的索引和
  prefetch 仍有界；
- 注入单 rank Python exception、进程 kill、collective timeout、SIGINT 和 checkpoint 写入
  失败，确认有界退出、最早可观察的 worker failure 与相关 rank/phase 可见、无正式 success
  manifest、上一个 bundle 可恢复。

不预先写一个脱离 workload 的通用扩展效率百分比。D0 根据单卡基线和训练时间预算制定目标，
D4 用同一 workload 判定是否达标。CPU/Gloo、单机双卡 smoke 或模拟 collective 只能补充控制
路径证据，不能替代 8×H200 acceptance。

## 首轮验证矩阵与后续研究矩阵分开

D0–D4 必须包含：

- world-size-one：当前单进程配置、artifact、checkpoint 和结果不变；
- CPU/Gloo multi-process：session、identity broadcast、collective order、DDP step、failure 与
  teardown；
- execution binding：内置 Builder 与独立 extension Builder 都实际调用 DDP execution model；
- Data：map-style 与至少一个大数据 reader 的有界 assignment、equal-step、epoch propagation、
  drop policy、训练期 exact validation 和 early exhaustion failure；
- Training：effective global batch、`no_sync()` forward/backward、non-finite/skip consensus、
  clipping、EMA、scheduler、global step、Metric merge 与 best/early-stop consensus；
- Artifact admission：每 rank full verification 的实测成本，或经规范批准的 Store admission；
- Checkpoint：common/local inventory、损坏/缺失、发布中断、epoch-boundary fixed-world resume 与
  portable export；
- CUDA/NCCL：单机 2/4/8 卡 numerical correctness、故障注入、strong/weak scaling 与资源预算；
- installed wheel：与 source checkout 的单进程/DDP 行为一致。

以下内容不能混入首轮通过数量：replicated sampling、FSDP2/DTensor、DCP、multi-node、elastic
restart、topology-changing resume、mid-epoch cursor、formal distributed Evaluation、HSDP、
tensor/pipeline/3D parallel 和第三方 distributed provider adapter。CI 中没有目标硬件的能力
只能标为未验收，不能用 mock 推导 production support。

## 历史内容怎样归入当前阶段

旧讨论中的数据分片、全局指标、共同失败和 portable export 仍由 D0–D4 保留。2026-08-13 的
审查修正了两个容易误导实现的归类：旧附录把“fixed world-size DDP/FSDP2”写成首轮，又把
replicated sampling 放在 D4；当前阶段表取代这两个表达。D4 现在只负责首轮 DDP 的目标硬件
验收，replicated sampling 是独立后续候选。

| 旧主题 | 当前归宿 |
| --- | --- |
| characterization 与 architecture contracts | D0 |
| SPMD session、rank-zero I/O 与一份 run identity | D1 |
| DDP execution binding、rank-aware data、全局 step 与 Metric | D2 |
| distributed checkpoint 与 portable export | D3 |
| CUDA/NCCL correctness、failure 与 H200 scaling | D4 |
| replicated sampling | 独立后续候选 |
| FSDP2、DCP 与大模型 inference | 独立研究候选 |
| multi-node、elastic 与 exact formal distributed Evaluation | 各自独立研究候选 |

这些 ID 不代表依赖已满足、平台已支持或工作已经进入路线图。

## 上游核对入口

- [PyTorch Distributed overview](https://docs.pytorch.org/docs/stable/distributed.html)
- [DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [DistributedSampler](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)
- [Elastic Run / torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html)
- [PyTorch Elastic error propagation](https://docs.pytorch.org/docs/stable/elastic/errors.html)
- [ProcessGroupNCCL environment variables](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html)
- [NVIDIA NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
- [NVIDIA nccl-tests](https://github.com/NVIDIA/nccl-tests)
- [Composable FSDP `fully_shard`](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)

这些链接是重新调研的入口。短期实施计划必须引用启动时实际审核的版本化文档、源码行为与
目标硬件 spike 结果；本附录中的名称和候选值不能替代那次核对。

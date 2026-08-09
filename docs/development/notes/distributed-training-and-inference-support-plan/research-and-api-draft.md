# Distributed 支持：PyTorch 调研、API 草案与历史映射

> 本附录服务于
> [`Distributed Training 与 Inference 支持计划`](../../distributed-training-and-inference-support-plan.md)。
> 内容是未来重开时的研究清单与设计草案，不是当前 API、支持矩阵或版本承诺。
> 所有上游语义必须在启动前根据当前 lockfile、目标硬件和官方文档重新验证。
> 最后核对：2026-08-09

## 启动前的 provider/version 调研

至少重新确认：

1. `torchrun`/elastic launcher 的 environment、failure propagation、restart 和 rendezvous 语义。
2. `torch.distributed` 在目标平台的 backend 可用性、timeout、debug 和 teardown 行为。
3. DDP 对参数注册顺序、unused parameters、static graph、gradient accumulation 和 mixed
   precision 的当前约束。
4. composable FSDP2、DTensor、`fully_shard()`、gradient sync、state-dict 与 optimizer
   construction 的公开稳定性。
5. Distributed Checkpoint 的 state-dict、planner、async/save/load、reshard、storage backend
   和 failure atomicity 保证。
6. PyTorch 对 optimizer/scheduler/scaler、EMA-like state 和 RNG 的分布式保存建议。
7. NCCL/Gloo 在单机、多机、Windows/Linux、CPU/CUDA 和 CI 环境的受支持组合。
8. upstream API 是否仍可由窄 protocol 包装，而不把 experimental object 暴露为公共
   Stochaflow contract。

不应仅凭 import 成功认定支持。每个目标组合需要最小 executable spike：初始化、一次
forward/backward/step、collective reduction、save/load、异常退出和资源释放。

## 候选技术选择

| 需求 | 首选候选 | 理由 | 独立 gate |
| --- | --- | --- | --- |
| 数据并行训练 | DDP | 模型复制、成熟语义、适合能装入单卡的 workload | global data/metric/checkpoint correctness |
| 参数与 optimizer state 分片 | composable FSDP2 | 面向超单卡容量；可显式 partition | DTensor、EMA、optimizer、DCP 与 portable export |
| 推理吞吐 | replicated SPMD | 每 rank 完整模型，按 sample ID 分片 | deterministic assignment 与 artifact completeness |
| 大模型推理 | FSDP2 inference 候选 | 只有完整模型不能装入单设备时才合理 | collective forward、latency 与 failure cost |
| 训练恢复 | Distributed Checkpoint 候选 | sharded state 与 load-time reshard | atomic publication、schema、corruption handling |
| 用户可携推理 | portable full-state checkpoint | 与现有 inference projection 对齐 | 从 distributed state 显式导出 |

DDP、FSDP2 和 replicated inference 不是一个 mode enum 的等价值。它们拥有不同
composition、state、failure、performance 和 artifact 语义；应共享窄 lifecycle capability，
而不是共享一套臃肿基类。

### 首轮之后的独立候选能力

以下内容没有进入首轮固定 world-size DDP/FSDP2 范围。每一行都需要单独评审；满足一行的
开始证据不会同时授权其他行，也不会自动扩大公共支持矩阵。

| 候选能力 | 开始评审所需证据 | 必须保持的边界 |
| --- | --- | --- |
| HSDP | FSDP2 multi-node 已通过正确性和故障验收；基准证明只做 fully sharded data parallel 仍受跨节点复制或通信限制；至少第二个稳定 workload 需要相同能力 | 明确 device mesh、replicate/shard 维度、checkpoint reshard 与 portable export，不把 HSDP 当作 FSDP2 配置别名 |
| tensor/pipeline/3D parallel | 已验收的 FSDP2 仍不能满足单模型容量或吞吐预算；模型或 Builder 能显式声明切分语义；基准和第二个稳定 workload 证明不是单任务特例 | 每种 parallel dimension 先有独立 contract；3D 只能组合已经分别验收的维度，不由 core 猜模型切分点 |
| ZeRO、DeepSpeed、Megatron 或 FairScale adapter | 真实 extension consumer 需要 provider 特有 lifecycle，且已证明 public DDP/FSDP2 capability 无法表达；目标 provider 版本有可执行 spike 和维护 owner | 每个 provider 分别通过窄 adapter 接入；不得把 provider object 暴露为通用 core API，并保持 global metrics、failure ordering、resume schema 与 portable export |
| 自动 learning-rate scaling | 至少两个受维护 recipe 在多个 global batch size 和目标 optimizer 上的受控实验得到稳定规则，并明确 scaling 对 scheduler、warmup 与 resume 的影响 | 只能是显式 opt-in 的 recipe policy；runtime 仍记录 per-rank/effective global batch 与实际 learning rate，不把线性缩放当通用事实 |
| distributed mid-epoch iterator resume | epoch-boundary 重算已违反明确恢复时间或计算预算；每个受支持 DataBuilder、sampler 和 worker pipeline 都提供版本化、可序列化的 per-rank cursor；恢复后 sample sequence、RNG 与 step consensus 可证明等价 | checkpoint 只能在完整 optimizer/accumulation 边界提交；必须处理 prefetch 与 worker state；不支持 cursor 的 Builder 继续只允许 epoch-boundary resume |

`async_save` 同样只在同步保存正确、immutable snapshot 与 CPU memory budget 均有证据后评审。

## 候选平台矩阵

此表仅是验证顺序草案，启动时必须重写：

| 能力 | Linux CUDA/NCCL | CPU/Gloo | Windows | MPS |
| --- | --- | --- | --- | --- |
| single parity | acceptance | CI | CI | current public behavior |
| DDP 单机 | production candidate | correctness | probe before promise | unsupported candidate |
| DDP 多机 | acceptance candidate | debug only | parked | unsupported candidate |
| replicated sampling | production candidate | correctness | backend-gated | unsupported candidate |
| FSDP2 training | evidence-gated | non-production oracle | parked | unsupported candidate |
| FSDP2 inference | evidence-gated | non-production oracle | parked | unsupported candidate |
| topology-changing elastic | separate research | separate research | parked | unsupported candidate |

Runtime 最终应检查实际 capability，而不是从文档表格推断：distributed availability、backend、
device/local rank、world size、版本化 FSDP/DCP feature 和 requested strategy compatibility。

## Config 草案

字段和命令都未获批准：

```yaml
distributed:
  enabled: true
  training:
    strategy: ddp
    gradient_sync: accumulation_window
  inference:
    strategy: replicated
  process_group:
    backend: nccl
    timeout_seconds: 600
  checkpoint:
    backend: distributed-checkpoint
    portable_export: best
```

候选所有权：

- launcher environment 拥有 rank、local rank、world size、restart count 和 rendezvous。
- config 只选择获支持 strategy 与稳定 policy；不复制 launcher-derived topology。
- device resolver 把 local rank 映射到唯一设备并验证冲突。
- TrainingBuilder 声明 task compatibility 与 execution binding，不声明 backend。
- DataBuilder 接收 context，不读取环境变量。
- checkpoint backend config 不进入 portable inference recipe。
- sampling assignment 和 writer aggregation 属于 sampling operation，不属于 training config。

不允许任意 process-group object、Python import path、wrapper class、auto-wrap callable 或
collective function 进入 YAML。

## API 草案

以下名称只表达职责，不是兼容承诺：

```python
@dataclass(frozen=True, slots=True)
class DistributedRunContext:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    device: torch.device
    backend: str | None
    initialized: bool

    @property
    def is_primary(self) -> bool: ...


class CollectiveOperations(Protocol):
    def sum_tensor(self, value: torch.Tensor) -> torch.Tensor: ...
    def max_tensor(self, value: torch.Tensor) -> torch.Tensor: ...
    def broadcast_object(self, value: object | None, *, source: int) -> object: ...
    def barrier(self) -> None: ...


class GradientSynchronization(Protocol):
    def enabled(self, enabled: bool) -> ContextManager[None]: ...


class ParallelExecutionBindingBuilder(Protocol):
    def bind_execution_modules(
        self,
        plan: TrainingPlan,
        modules: Mapping[str, torch.nn.Module],
    ) -> TrainingPlan: ...


class FSDP2ShardingPlanBuilder(Protocol):
    def build_sharding_plan(self, plan: TrainingPlan) -> ShardingPlan: ...


class CheckpointBackend(Protocol):
    def save(self, request: CheckpointSaveRequest) -> CheckpointReceipt: ...
    def load(self, request: CheckpointLoadRequest) -> CheckpointState: ...
```

`DistributedSession` 候选职责仅限 launcher env 解析、device resolution、process-group
init/destroy、collectives construction 与 topology/config consistency preflight。它不启动
child process、不选 DDP/FSDP、不建目录、不保存 checkpoint、不替 DataBuilder 构建 sampler。

## Training composition 草案

候选顺序：

```text
validated config + verified artifacts
    -> canonical model/process/objective/assets
    -> TrainingBuilder builds canonical TrainingPlan
    -> distributed compatibility and sharding plan validation
    -> move / DDP wrap or FSDP2 partition execution modules
    -> Builder rebinds Strategy through narrow capability
    -> construct optimizer from final trainable parameters
    -> construct scheduler/scaler/EMA controller
    -> Trainer consumes executable plan and parallel capabilities
```

关键点：

- canonical module 用于 identity、declaration 和 portable export；execution module 用于
  forward/backward。两者关系必须显式。
- primary model 是首个 trainable root。trainable Process、Objective 或 auxiliary module 只有
  在多个 execution binding 被正式支持后才开放。
- FSDP2 必须在 optimizer construction 前完成 sharding。
- partition 由 model capability 或 Builder sharding plan 声明，按 bottom-up 顺序执行。
- core 不按 class name、parameter count、module path pattern 或 registry name 推断 partition。
- rank-dependent control flow、manual backward、alternating optimizer、closure-required
  optimizer 在当前自动 loop family 中 fail closed。

## Accumulation、step 与 precision

- accumulation window 只有最后一个 micro-batch 启用 gradient synchronization。
- all ranks 必须进入相同次数、相同顺序的 trainable forward/backward。
- local loader exhaustion、skipped batch、non-finite loss 和 GradScaler skipped step 需要全局
  consensus；不能让部分 rank 先推进 scheduler/global step。
- gradient clipping 通过 parallel capability 操作正确参数表示，不读取 wrapper internals。
- autocast policy 保持 Trainer-owned；FSDP mixed-precision policy 需要与参数/reduction dtype
  明确协调。
- DDP EMA 可基于 canonical full tensor；FSDP2 EMA 需要 sharded controller 或显式 full-state
  materialization policy，不能复制 DTensor 当普通 Tensor。

## Rank-aware data 草案

DataBuilderContext 可候选性地携带 immutable distributed context。Builder 返回 loaders 时还
应返回 sharding evidence：assignment policy、sampler identity、drop/pad policy、epoch seed 和
expected coverage。

普通 map-style dataset 可用 distributed sampler 类策略；multi-resolution、bucket、mixture
或自定义 batch sampler 必须由其 owner 实现 rank-aware composition。core 不拆解 batch 或
替换现成 loader 的 sampler。

Training 可接受明确的 equal-step drop/pad policy，但正式 validation/test 必须证明 exact
coverage。若 provider 通过 padding 重复样本，Metric/Evaluation 不能把重复样本当作新证据。
sampler 与 batch sampler 的 `set_epoch()` 传播必须覆盖所有 phase。

## Global Metrics、selection 与 side effects

候选 reduction 需要 task/metric 明确提供 sufficient statistics：sum/weight、count、confusion
matrix 或 provider-owned distributed state。不得默认平均各 rank 的 local mean。

rank zero 可以拥有：console、普通 logger、best/early-stop decision、manifest commit、writer
merge 和最终 outcome。所有 rank 仍需要接收 decision，并以一致顺序进入后续 collective。
Diagnostic 必须声明 all-rank、rank-zero-with-portable-state 或 unsupported；不能靠偶然调用
顺序避免重复副作用。

## Distributed checkpoint 草案

需要区分：

1. 首轮 distributed resume bundle：可能含 sharded model、optimizer、scheduler、scaler、EMA、
   rank-local RNG、completed-epoch progress 与 topology；
2. portable inference checkpoint：完整 primary/Process/declared inference assets、固定 recipe
   和 provenance，不含 training-loop state。

候选 publish protocol：rank zero 创建 invocation identity 与 private staging root；所有 rank
写自己的 state；共同验证 inventory/digest；全局成功后由 rank zero 原子发布 manifest；任一
失败都使 staging 非正式。恢复时先验证 schema、topology、artifact inventory 和 config，再
把 state 提交到 live objects，保持事务式 restore。

首轮 fixed-world exact resume 只接受完整 epoch 边界，恢复每 rank RNG、下一 epoch 的 sampler
seed、optimizer/scheduler/scaler/EMA 和 global progress；它不序列化运行中的 iterator、data
cursor、prefetch queue 或部分 accumulation window。load-time reshard 只解决 state layout，
不会自动定义 global batch、RNG 或 data-order 语义。因此 mid-epoch iterator resume 与
topology-changing resume 都需要独立 contract；不能因 DCP 能加载就宣称 exact。

Async checkpoint 只有在同步 save 正确、snapshot consistency 和内存预算清楚后评估。

## Distributed sampling/inference 草案

Replicated 模式在每 rank 装载完整 portable projection，用稳定 global sample IDs 划分请求。
random semantics 候选为 `seed + global_sample_id + task-owned stream identity`，避免结果依赖
world size 或 batch partition；具体派生算法必须版本化并测试。

```python
@dataclass(frozen=True, slots=True)
class SamplingAssignment:
    global_count: int
    sample_ids: tuple[str, ...]
    rank: int
    world_size: int
```

各 rank 向私有 shard staging 写 artifact 和 local manifest。rank zero 只聚合经过验证的
metadata；writer 若需要单文件输出，应实现窄 merge capability 并按 global sample ID 排序。
不使用 object gather 搬运图片、tensor 或大量 records。

FSDP2 inference 只有模型无法放入一张目标设备且 collective latency 可接受时考虑。所有 rank
必须参与每个 forward；不能让 rank zero 独自运行 post-training sampling。训练后 sampling
应作为显式 operation transition，消费 portable export 或保持协调的 distributed state。

## Failure 与资源治理

- launcher 报告首个 root cause；任一 rank 失败后不允许其他 rank 继续发布成功 outcome。
- process-group timeout、collective ordering、NCCL async error/debug flags 应成为运维文档，
  但不能替代 deterministic tests。
- output root、ports、CUDA device、local worker 和 staging path 必须无冲突。
- teardown 在正常、exception、SIGINT 和 launcher restart 路径都执行；不能用无限 barrier
  隐藏失败。
- materialized data artifact 可复用已有 lock/publish，但 rank-aware runtime loader 不应被
  缓存成 artifact。
- rank logs 可保留为调试 shard，正式 manifest 只在全局完成后发布。

## 模块草案

未来可能需要独立 `distributed` package，包含 context/session、collectives、training
parallelism、sampling assignment 和 checkpoint coordination。是否采用这些模块名必须在
启动时复核。现有 training/data/sampling/evaluation 包只消费窄 contracts，不能各自直接
散布 `torch.distributed` 调用。

CLI 仍应保持 thin：由 `torchrun` 或 scheduler 启动同一 command，库 entry point 接受已经
验证的 invocation 和 context。Stochaflow 不需要先实现自己的 process launcher。

## 最小验证矩阵草案

- world-size-one：所有普通 tests 与 config/artifact parity。
- CPU/Gloo multi-process：session、broadcast、reduction、DDP step、failure 和 teardown。
- Data：map-style、自定义 batch sampler、epoch propagation、drop/pad 和 exact evaluation。
- Training：accumulation、non-finite、GradScaler skip、clip、EMA、best/early-stop consensus。
- Checkpoint：all-rank save/load、损坏/缺失 shard、epoch-boundary fixed-world resume、portable
  export；mid-epoch iterator resume 不进入首轮矩阵。
- Sampling：global IDs、seed invariance、rank shards、writer merge、partial failure。
- Extension：独立 external Builder 的 execution binding 与 rank-aware DataBuilder。
- CUDA DDP：单机 numerical/performance acceptance。
- CUDA FSDP2：sharding、DTensor optimizer/EMA、DCP、portable export 和 capacity evidence。
- Multi-node：network timeout、node failure、restart、shared/object storage 和 observable root cause。
- Formal Evaluation：无重无漏 coverage、global metric equality、artifact completeness。

CI 中没有目标硬件的能力只能标为未验收，不能用 mock 推导 production support。

## 历史阶段映射

以下 ID 只用于定位旧 Git 讨论，不驱动当前排期：

| 历史 ID | 当时表达的主题 | 当前归宿 |
| --- | --- | --- |
| D0 | characterization 与 architecture contracts | 正文“需求证据”任务卡 |
| D1 | SPMD session 与 rank-zero I/O | 正文“SPMD session”任务卡 |
| D2 | DDP training | 正文“Rank-aware data”和“DDP training”任务卡 |
| D3 | distributed checkpoint | 正文“Distributed checkpoint”任务卡 |
| D4 | replicated distributed sampling | 正文“Replicated distributed sampling”任务卡 |
| D5 | FSDP2 training | 正文“FSDP2 training 与 inference”任务卡 |
| D6 | FSDP2 inference 与 multi-node hardening | 正文对应 future task card |
| D7 | exact distributed Evaluation 与 elastic | 正文最后一个 future task card |

这些 ID 不代表依赖已满足、平台已支持或 task 已进入路线图。

## 上游入口

- [PyTorch Distributed overview](https://docs.pytorch.org/docs/stable/distributed.html)
- [DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- [Composable FSDP `fully_shard`](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html)
- [Elastic Run / torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html)

这些链接是重新调研的入口。短期实施计划应引用启动时实际审核的版本化文档、源码行为与
本地 spike 结果。

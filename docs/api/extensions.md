# Extension 公共 API

第三方 extension 的训练、采样、数据与 Registry 契约应依赖：

```python
from stochaflow.extensions import ...
```

独立 EvaluationBuilder 及其 plan/step contracts 从专用公共模块导入：

```python
from stochaflow.evaluation import ...
```

不要从 `stochaflow.data`、`stochaflow.training`、`stochaflow.processes`、
`stochaflow.sampling` 或 `stochaflow.utils` 的内部模块路径导入契约。内部文件可以在尚未
发布的重构中移动；`stochaflow.extensions.__all__` 与 `stochaflow.evaluation.__all__`
是 extension 作者应使用的公共入口。

## 配置与 Registry

| 符号 | 用途 |
| --- | --- |
| `ComponentConfig` | 一个 registry/native-provider component 的 `name` 与不透明 `params` 声明 |
| `ConfigError` | extension 私有参数的统一配置校验错误 |
| `REGISTRIES` | 进程级 Registry catalog；extension 用对应 registry 的 `register(name)` 注册组件 |
| `Registry` | 具名组件容器及 resolve/create/register 契约 |
| `RegistryError` | 注册、解析或构建失败 |

`RegistryCatalog` 不是公共导出。extension 应使用 `REGISTRIES`，不要创建或依赖 catalog
内部实现。

## Data

| 符号 | 用途 |
| --- | --- |
| `IMAGE_DATA_SOURCES` | 复用内置 image recipe 时注册 `ImageDataSource` 的受限 Registry |
| `DataSource` | 只负责物化一个带 identity 的 `DataArtifact`，不构造训练数据栈 |
| `DataSourceContext` | cache root、ensure/require、verification 与 strict-resume expected identity |
| `ArtifactVerificationEvent` | `full` 验证的 producer、phase、completed 与 total 进度值 |
| `ArtifactVerificationObserver` | 接收有序验证事件的可选窄 callback contract |
| `ArtifactVerificationPhase` | 当前固定为 `validate` 的内容验证阶段 |
| `DataSourceMaterializationConfig` | `source.materialization` 的通用 typed config，可构造 `DataSourceContext` |
| `ImageDataSource` | 内置 image recipe 可消费的 source-adapter 基类 |
| `DataArtifact` | managed/referenced 内容共用的已验证 runtime handle |
| `DataArtifactIdentity` | 严格、location-independent 的 schema-v2 identity |
| `DataArtifactStore` | managed/referenced producer 共用的 framework lifecycle |
| `ManagedDataArtifactBuild` | managed producer 在 staging 写入完成后返回的 source/materialization/domain facts |
| `ReferencedDataArtifactBuild` | referenced producer 建立索引后返回的 source/materialization/content/domain facts |
| `DataArtifactLoadContext` | framework 传给无副作用 payload loader 的 verified staging/final 路径、identity、domain 与验证强度 |
| `DataArtifactValidationError` | 已持久化候选或 represented content 违反契约 |
| `canonical_artifact_json_bytes` | 严格 JSON-safe、排序、紧凑、带末尾换行的 canonical encoding |
| `canonical_artifact_digest` | canonical artifact JSON 的 SHA-256 |
| `DataArtifactBinding` | 一个稳定 role/id 到完整 artifact identity 的绑定 |
| `DataArtifactBindings` | 严格排序、唯一且可序列化的 binding 集合 |
| `ImageDimensions` | 一条由 artifact 认证的图像宽高记录 |
| `ImageDimensionTable` | 紧凑、只读、可按索引读取的认证图像尺寸表 |
| `ImageFileRecord` | reference inventory 中的相对路径、字节数、SHA-256 与认证宽高 |
| `ClassLabeledImageFileRecord` | 一个认证图像记录及其非负整数 class label |
| `ImageFilePair` | paired image payload 中严格对齐的 HR/LR 记录 |
| `TorchvisionImageArtifactPayload` | torchvision source 交给 image recipe 的公开 payload |
| `ImageFolderArtifactPayload` | 单目录或 split image folders 的公开 payload |
| `ClassLabeledImageFolderArtifactPayload` | 带连续 class mapping 与逐样本标签的 image-folder payload |
| `PairedImageFolderArtifactPayload` | paired HR/LR image folders 的公开 payload |
| `DataBuilder` | 组装一份独立训练运行的完整数据栈 |
| `DataBuilderContext` | 深复制的私有 `params`、seed 与 strict-resume artifact expectations |
| `DataLoaders` | 可重复迭代 loader、可选 `steps_per_epoch` 与本次运行的 `artifact_bindings` |

`DataBuilder` 是运行时数据组合入口；`ImageDataSource` 是复用内置 image recipe 的来源
扩展入口。`ImageDataSource` 通过 `IMAGE_DATA_SOURCES` 注册，只负责读取、处理、
materialize artifact，并返回上述公开 payload 之一；它不构造 Dataset、split、
transform、collate、PyTorch sampler 或 DataLoader。一个新来源只有同时满足目标
Builder 的完整 accepted artifact contract，并需要相同的 partition、Dataset、
transform、sampler、collate、loader、resume 与 batch 语义时，才可以只实现
DataSource；否则应实现独立的 recipe-level `DataBuilder`。

非图像 extension 可以定义自己的窄 DataSource base 与 family-local registry，由自己的
DataBuilder 选择 source、验证 `DataArtifact` binding 并组装 runtime recipe；这不要求
也不应创建 framework-global Dataset/Sampler/DataLoader registry。若 recipe 没有任何
外部输入，例如完全由 resolved config 与 experiment seed 生成的 synthetic fixture，
Builder 可以直接构造 runtime views，并明确返回没有 artifact bindings 的
`DataLoaders`。

所有 producer 都必须通过 `DataArtifactStore` 使用同一个 schema-v2 manifest、inventory、
locator、locking、publication、quarantine 与 strict-resume lifecycle。不要在 extension
中自行实现 manifest、identity 或 current-pointer 状态机。`managed` 表示 artifact 的
实际内容由 cache 拥有；`referenced` 表示 cache 只拥有索引/sidecar，represented content
仍由外部目录拥有。ownership strategy 记录在 `artifact.identity.kind`，不会改变统一
runtime handle。

最小 producer 形状：

```python
class ProjectSource(DataSource[ProjectPayload]):
    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[ProjectPayload]:
        store = DataArtifactStore(context)
        return store.materialize_referenced(
            artifact_type="project.records.v1",
            source_name="my-project.records",
            materializer_name="my-project.indexer",
            locator_key={"root": str(self.root)},
            referenced_roots={"records": self.root},
            build=self._build_index,
            load=self._load_payload,
        )
```

managed `build(data_root)` 把内容写入 framework staging `data/`，返回
`ManagedDataArtifactBuild`；framework 扫描、哈希并认证所有普通文件。referenced
`build(data_root)` 只写索引/sidecar，返回 `ReferencedDataArtifactBuild`，其中
`content_digest` 由 producer 对外部内容 inventory 计算。`load(context)` 会在发布前对
verified staging root 以 `full` 调用，并在发布后对 final object root 再次调用；cache hit
则直接使用 final root。因此 callback 必须幂等、无副作用；framework-owned
文件必须从 `context.data_root` 读取，referenced producer 还可以使用
`materialize_referenced(...)` 中已声明并由 callback closure 捕获的 external roots，
但不能在 `load` 中执行 acquisition、写入或重新物化。staging 调用的 payload 会被丢弃，
最终返回的 `DataArtifact` 只保留 final-root payload。持久化内容错误应抛
`DataArtifactValidationError`，普通 `TypeError`/`ValueError` 会被视为 producer bug。

cache hit 或 strict resume 的 `full` 加载先执行一次完整内容扫描，同时验证 manifest、
inventory、object layout、identity 和 stored files，并作为 loader 前快照；loader
返回后只执行 link-safe 元数据扫描，严格比较路径、数量、size、device/inode、mode、
mtime 与 ctime。identity-only 校验只执行第一轮。fresh materialization 的 staging 与
final object 是两个独立验证边界，各自执行一次内容验证和一次加载后元数据复查。文件
SHA-256 在 artifact I/O 层有界并行，但快照和异常始终按路径确定性排序。默认线程数为
`min(8, logical CPUs)`；`DataSourceContext.verification_workers` 可以提供 `1..8`
范围内的整数覆盖。
底层以 1 MiB `hashlib` 更新执行哈希，CPython 会在哈希与文件读取期间释放 GIL。
referenced producer 仍负责在 `load(full)` 中认证 external represented content；
framework 的加载后元数据复查只覆盖其拥有的 manifest、inventory 与 sidecar。该复查
用于检测无副作用 loader 的意外写入，不是对恶意同机修改的安全隔离。

CLI 或自定义 Builder 可以向 `DataSourceContext.verification_observer` 注入
`ArtifactVerificationObserver`。唯一的内容验证从 `completed=0` 开始，仅在 `full`
模式产生 `phase="validate"` 事件；加载后元数据复查不产生进度。callback 在调用线程
执行。observer 是临时运行时能力，不应序列化到 source config、artifact identity 或
checkpoint；observer 自身的异常会直接传播，不会把有效 artifact 归类为损坏。

`DataSourceMaterializationConfig.verification_workers` 是可序列化的运行配置，
`DataBuilderContext.verification_workers` 是可选的本次运行覆盖。两者都不会进入
artifact identity、locator 或 digest。

`DataArtifactIdentity` 固定包含 `schema_version=2`、`kind`、artifact/source/materializer
名称与 digest、content/artifact digest 及 `manifest_sha256`。`DataArtifactBindings`
同样只接受 schema v2。旧 manifest、locator、cache layout、identity 或 checkpoint
binding 不会被读取或迁移；需要重新 materialize 并启动新 run。

最小实现签名：

```python
@dataclass(frozen=True, slots=True)
class DataBuilderContext:
    params: dict[str, Any]  # 深复制
    seed: int
    strict_resume: bool = False
    expected_artifacts: DataArtifactBindings | None = None
    verification_observer: ArtifactVerificationObserver | None = None
    verification_workers: int | None = None


@dataclass(frozen=True, slots=True)
class DataLoaders:
    train: Iterable[Any]
    validation: Iterable[Any] | None = None
    test: Iterable[Any] | None = None
    steps_per_epoch: int | None = None
    artifact_bindings: DataArtifactBindings | None = None


class DataBuilder(ABC):
    def __init__(self, context: DataBuilderContext) -> None: ...

    @abstractmethod
    def build(self) -> DataLoaders: ...
```

loader 必须可重复迭代，不能是一份已经创建的 iterator。train 若没有 `len()`，必须声明
正数 `steps_per_epoch`；若有 `len()`，显式步数不能超过它。每个 epoch 开始时，核心会对
train loader 的 `sampler`/`batch_sampler` 去重后调用可选 `set_epoch(epoch)`。

## Metrics

`stochaflow.metrics` 的正式公共 API 只包含下列符号：

| 符号 | 用途 |
| --- | --- |
| `MetricSpec` | 不含 phase 的 metric 构造与 channel binding |
| `MetricUpdate` | Strategy 交给一个已声明 channel 的不透明 `args`/`kwargs` payload |
| `MetricEngine` | 一个隔离统计 scope 中的 metric 构造、update、compute 与 reset 生命周期 |
| `MetricRuntimeError` | channel、update 或 compute 结果违反 runtime contract |
| `build_metric` | 从一个 `MetricSpec` 通过 metric Registry 构造实例 |
| `ErrorOnNanMeanMetric` | 内置 `mean` 注册项的具体实现 |
| `SingleOutputMeanSquaredError` | 内置单输出 `mse` 注册项的具体实现 |
| `SingleOutputMeanAbsoluteError` | 内置单输出 `mae` 注册项的具体实现 |
| `FrechetInceptionDistanceMetric` | 内置 `fid` image-distribution adapter；需要 optional `quality` dependencies |
| `KernelInceptionDistanceMetric` | 内置 `kid` image-distribution adapter；返回稳定 `mean`/`std` mapping，需要 optional `quality` dependencies |
| `ShareableImageFeatureMetric` | composite Metric 可选实现的窄能力；仅在 extractor identity 完全相同时复用一次图像特征提取 |

Strategy 使用 `stochaflow.extensions.MetricChannelProvider` 声明 channel，并可从
`stochaflow.extensions` 导入 `MetricUpdate`。构造校验、payload detach 和训练 phase
binding 是框架内部协作，不是 extension facade 的公共 helper API。

训练配置的每个 `metrics` item 保持平面
`{id, name, channel, params, phases}` 形状；配置层把其中 task-neutral 字段组合为
`MetricSpec`，而不是公开一个继承 `MetricSpec` 的 training-specific 类型。
`MetricEngine` 的公共 surface 只有构造、`required_channels`、`update()`、`compute()`、
`reset()` 与 `to()`；接受 prepared payload 的路径属于训练 runtime 内部协议。

`REGISTRIES.metrics` 只接受 `torchmetrics.Metric` 子类。Stochaflow 不解析任意
`torchmetrics.*` class path，也不镜像上游 namespace；第三方 extension 应注册自己的
稳定名称。当前内置集合只有：

| Registry name | 稳定语义 |
| --- | --- |
| `mean` | scalar mean；固定 `nan_strategy="error"` |
| `mse` | scalar mean squared error；固定 `squared=True`、`num_outputs=1` |
| `mae` | scalar mean absolute error；固定 `num_outputs=1` |
| `fid` | TorchMetrics-backed FID；输入必须是 finite floating RGB NCHW `[0, 1]` images |
| `kid` | TorchMetrics-backed KID；输入必须是 finite floating RGB NCHW `[0, 1]` images |

`ShareableImageFeatureMetric` 不改变 Evaluation 或 `MetricUpdate` 的 payload contract。
Evaluation 仍提交 image samples；一个 composite Metric 可以按 hashable
`image_feature_extractor_identity()` 分组，在组内调用一次 `extract_image_features()`，再把
结果分别交给各成员的 `update_image_features()`。identity 必须包含所有影响特征的事实；内置
FID/KID identity 包含 extractor class、feature width 和 `antialias`。不同 identity 绝不能
共享特征，也不存在由 core `MetricEngine` 自动建立的跨 Metric cache。

`mse` 和 `mae` 不提供 vector-valued `num_outputs>1` 模式。需要分类、按类别、结构化
mapping 或其他任务 metric 时，extension 应注册一个明确的 `Metric` 实现，并通过
Strategy 自己的 channel 提供所需 payload。

phase metric 不解释 batch、prediction、target、label 或 sample weight。Strategy 用
`MetricChannelProvider.metric_channels` 声明自己能产生的 channel，再把对应
`MetricUpdate` 放入 `TrainStepOutput.metric_updates`。组合阶段会确认每条配置引用的
channel 确实由该 Strategy 提供；运行时为每个配置 phase 构造独立 metric 实例，避免
train、validation 与 test state 串扰。train metric 只提交成功 optimizer window 的
updates；validation/test 则在成功 evaluation step 后提交。

channel payload 是 Strategy 与所选 Metric 之间的不透明局部协议。`MetricUpdate.args`
必须是精确 tuple，`kwargs` 必须是精确 dict 或只读 `MappingProxyType`。payload tree
只接受 Tensor、精确的 dict/list/tuple（以及只读 dict 形式）与
`None`/`bool`/`int`/`float`/`complex`/`str`/`bytes` scalar leaf；mapping key 也必须是
这些不可变 scalar。`OrderedDict`、namedtuple、`torch.Size`、dataclass 和任意自定义
对象会被拒绝，不存在让自定义容器接管 detach 的协议。

payload 中的 Tensor 在进入 metric state 前只 detach 一次，并在 `torch.no_grad()` 下
消费。train phase 会先保存 detached payload，只有完整 optimizer lifecycle 成功后才
提交；跳过的 optimizer window、backward/step 异常都不会改变 metric state。
validation/test 没有 optimizer commit 点，因此在成功 evaluation step 后立即 update。

`MetricEngine.compute()` 的 scalar 结果使用 `<id>`；flat mapping 结果使用
`<id>/<subkey>`。训练 runtime 再添加 phase prefix：
`train/metrics/...`、`valid/metrics/...` 或 `test/metrics/...`。完成 epoch 后，logger、
history 和 checkpoint 只消费普通 scalar mapping；Metric runtime state、payload 和
额外的 source/provenance snapshot 不进入 checkpoint。

best checkpoint 与 early stopping 只接受 `valid/loss` 或
`valid/metrics/<id>[/<subkey>]`。train/test phase metric 以及 `diagnostics/...` 日志都不是
模型选择输入。

`Metric` state 可以声明 TorchMetrics 的 reduction 与 synchronization policy，但当前
Stochaflow Trainer 只承诺单进程结果。DDP、FSDP 和其他多进程 Trainer 尚未实现；不要把
一个 extension metric 的 reduction 声明解读为 distributed training 支持。

## Evaluation

`stochaflow.evaluation` 当前公共 runtime 与 maintained AFHQ pixel-image profile 使用的
组合契约包括：

| 符号 | 用途 |
| --- | --- |
| `EvaluationBuilder` | 独立 task evaluation 的唯一注册组合入口 |
| `EvaluationBuilderContext` | 只读 params、resolved subject、selected data/data identity、可选 pinned inference model/sampling capability、artifact staging root、MetricSpecs 与 protocol |
| `EvaluationPlan` | Evaluator、data、metric declarations、protocol、subject/data identity、可选 artifact sink 与 core-managed modules |
| `Evaluator` | structural batch evaluation contract；声明 `metric_channels` 并返回 `EvaluationStepOutput` |
| `EvaluationStepOutput` | 显式 example count、稳定 sample IDs、metric update groups、不透明 records 与 measurements |
| `EvaluationProtocol` | protocol id、positive expected count 与 strict completeness policy |
| `EvaluationSamplingRequest` | task profile 拥有的 writer-free sampling options/sampler/shape/count/batch/seed 与可选 frozen recipe contract |
| `EvaluationSamplingCapability` | 已绑定 checkpoint assets 与 pinned raw/EMA model 的窄 `execute()` protocol；offline subject 不提供 |
| `CheckpointEvaluationSamplingCapability` | 通过 shared SamplingBuilder execution seam 实现上述 protocol 的 core checkpoint adapter |
| `EvaluationArtifactSink` | 流式消费已经评估的 step output，finalize 为 complete `PredictionArtifactDraft`，失败时 abort |
| `JsonlPredictionArtifactSink` | 内置 canonical JSONL sink；按 exact `PredictionSampleIdentity` 校验 typed `PredictionRecord` |
| `PredictionArtifactSubjectConfig` | `subject.kind: prediction_artifact` 的 strict offline authority |
| `PredictionArtifactSubjectInputs` / `ResolvedPredictionArtifactSubject` | 已认证 manifest、shards、sample plan、records 与 producer lineage 的只读 view |
| `EvaluationResult` | 可移植、immutable subject/data/metrics/measurements/completeness/provenance facts |
| `EvaluationRunOutcome` | 本地 immutable paths 和 scalar views |

Builder 通过 `REGISTRIES.evaluation_builders.register(name)` 注册。它必须保留 context 注入
的 subject、data、data identity、MetricSpecs 与 protocol。checkpoint 路径会在 extension
激活后才构造 model，把配置中的 `raw` 或 `ema` 解析为 concrete variant，并要求 Builder
把该 model 列入 `EvaluationPlan.modules`。同一路径还提供绑定这一个 model/variant 的
`EvaluationSamplingCapability`：它从 checkpoint 恢复 Process 与 inference assets，使用
`PinnedInferenceModelProvider` 并调用 shared SamplingBuilder execution seam，返回 validated
in-memory batches 而不运行 writers。Builder 不接收可再次选择权重的 provider。

`prediction_artifact` 路径不构造 model 或 DataBuilder；context 的 `inference` 与
`sampling` 都为 `None`，data 是按 manifest sample plan 排序的 `PredictionRecord`，Builder
的 modules 可以为空。两条路径都不构造 TrainingPlan、optimizer、scheduler 或 checkpoint
loader。

Evaluator 拥有 batch/model/channel 语义，core 只传递 opaque batch、执行
`torch.inference_mode()`、检查 count/全局唯一 sample IDs、推进 task-neutral
MetricEngine、聚合 measurements 和发布 immutable result。若 checkpoint Builder 在
`EvaluationPlan.artifact_sink` 声明 sink，runtime 会将每个 step output 同步送入 sink；
finalize 后的 draft 必须与实际 ordered sample IDs 完全一致，才会在 result 下发布
`predictions/`。`JsonlPredictionArtifactSink` 要求 `EvaluationStepOutput.records` 是与
`sample_ids` 同序的 typed `PredictionRecord`，并从 `context.artifact_root` 创建 unpublished
shard。异常路径调用 `abort()` 并删除未发布 staging。

offline runtime 会认证 manifest/artifact/shard digest、strict schema、portable path、exact
sample identity/completeness、split 与 deterministic gallery selection，再按 sample plan
join records；它不会依赖 shard 文件名或枚举顺序，也不会修改 producer artifact。新 result
保留 producer、原 source subject、resolved weights、data、inference profile、training
config 与 extension provenance。core `fid`/`kid` adapters 与 maintained AFHQ-v2
source-checkout full-official-test Builder/Metric/profile 已闭合 pixel-image vertical slice；它固定完整
493/491/483 class allocation，支持 live predictions 与 offline replay。SR、其他
consistency、latent/codec 与 distillation 不属于这个 milestone 的待补 profile；未来任务
必须同步交付自己的 monitoring/checkpoint inference/Evaluation。reference cache、
performance/curve 和 comparison/gate 是可选增强。

完整注册示例与 strict YAML 见
[自定义 EvaluationBuilder](../configuration/extensions.md#自定义-evaluationbuilder)和
[独立 checkpoint Evaluation](../configuration/workflows.md#独立-checkpoint-evaluation)。

## Training

| 符号 | 用途 |
| --- | --- |
| `TrainingBuilder` | 组合注入资产和项目私有资产，返回一个 `TrainingPlan` |
| `TrainingBuilderContext` | primary model、可选 Process/Objective、私有 `params` 与受控 model/objective factory |
| `TrainingPlan` | Strategy、primary model、可选 Process/Objective、具名 auxiliary modules、inference asset projections 和可选 fixed inference recipe |
| `InferenceAssetProjection` | 将一个 managed auxiliary module 投影为 checkpoint-owned inference asset |
| `SamplingRecipe` | checkpoint 内部 SamplingBuilder identity 与不可由独立 sample config 覆盖的 JSON-safe contract |
| `ManagedTrainingModule` | 辅助 `nn.Module` 及其 core-managed mode policy |
| `TrainingStrategy` | 只定义 batch interpretation、forward、loss 与 metric 计算 |
| `DenoiserChannelLayout` | model 可选暴露的静态 `in_channels`/`out_channels` capability；Gaussian Builder 据此在组合时预检 fixed `C` 或 learned-range `2C` 输出 |
| `DeviceTransferableBatch` | 自定义领域 batch 可选择实现的显式设备迁移 capability |
| `ReferenceImageBatchSemantics` | Strategy 可选实现的 reference-metric image extraction capability |
| `TrainStepOutput` | Strategy 返回的 scalar loss、低成本 step report、diagnostics、metric channel updates 与 epoch loss reporting weight |
| `MSEObjective` | 内置 task-neutral scalar MSE Objective |
| `PerSampleObjective` | 可选的逐样本 loss capability |
| `compute_objective` | 校验并执行 scalar Objective，同时读取可选逐样本 capability |
| `TrainingDiagnostic` | training diagnostic 生命周期根契约 |
| `ContextAwareDiagnostic` | diagnostic 可选实现的构建期 context 参数能力 |
| `DiagnosticBuildContext` | diagnostic 构建期 logger 和输出目录；采样 shape 由 diagnostic 私有配置拥有 |
| `FitStartEvent` | fit 开始事件 |
| `TrainBatchEndEvent` | 成功 optimizer step 后的事件 |
| `TrainEpochEndEvent` | 一个 epoch 完成后的事件 |
| `ExperimentLogger` | extension logger backend 契约 |

Strategy 不是 `nn.Module`，也不移动、冻结、选择或序列化资产；这些生命周期由
TrainingPlan 和核心 runtime 管理。

`PerSampleObjective` 只承诺可选的逐样本报告，不包含公共 batch-reducer protocol。
内置 `MSEObjective` 可以在自己的具体实现中保持 `mean`/`sum` reduction 语义；其他
Objective 不需要为了某个 Gaussian recipe 实现通用 reducer。

Trainer 会递归迁移 batch 中的 `Tensor`、`Mapping` 的 value、tuple（包括
namedtuple）和 list；mapping key 及其他 leaf 保持不变。领域 dataclass 或自定义容器若
持有 Tensor，必须实现 `DeviceTransferableBatch.to_device(device)` 并返回迁移后的 batch；
核心不会反射 dataclass 字段，也不提供通用 batch/sample registry。

启用 image reference metrics 时，Strategy 必须实现
`ReferenceImageBatchSemantics.extract_reference_images(batch)`，显式返回 validation
batch 中的 clean image Tensor。diagnostic 不会按 mapping/list 顺序猜测第一个 Tensor，
因此 label、condition 或其他 4-D Tensor 的排列不会改变 reference dataset 语义。

core 会在每次 `TrainingDiagnostic` public callback 外保存并恢复 Python、NumPy、
Torch CPU 以及相关 CUDA/MPS device 的 global RNG state。一个 callback 内使用 global
RNG 不会改变训练或其他 callback 看到的随机流；若 diagnostic 需要跨 callback 延续自己的
随机序列，它必须持有自己的 generator/state，不能依赖 global RNG 的连续推进。

关键签名：

```python
@dataclass(frozen=True, slots=True)
class TrainStepOutput:
    loss: torch.Tensor
    metrics: Mapping[str, float | int | torch.Tensor] = ...
    diagnostics: Mapping[str, Any] = ...
    metric_updates: Mapping[str, MetricUpdate] = ...
    loss_aggregation_weight: float | int | torch.Tensor = 1.0


class TrainingStrategy(ABC):
    @abstractmethod
    def training_step(self, batch: Any) -> TrainStepOutput: ...

    def evaluation_step(self, batch: Any) -> TrainStepOutput: ...


@runtime_checkable
class DeviceTransferableBatch(Protocol):
    def to_device(self, device: torch.device) -> Self: ...


@runtime_checkable
class ReferenceImageBatchSemantics(Protocol):
    def extract_reference_images(self, batch: Any) -> torch.Tensor: ...


@runtime_checkable
class MetricChannelProvider(Protocol):
    @property
    def metric_channels(self) -> frozenset[str]: ...


@dataclass(frozen=True, slots=True)
class ManagedTrainingModule:
    module: nn.Module
    mode: Literal["follow", "eval"] = "follow"


@dataclass(frozen=True, slots=True)
class InferenceAssetProjection:
    training_asset_name: str
    declaration: ComponentConfig
    capability_role: str


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    strategy: TrainingStrategy
    primary_model: nn.Module
    process: Process | None = None
    objective: nn.Module | None = None
    auxiliary_modules: Mapping[str, ManagedTrainingModule] = ...
    inference_recipe: SamplingRecipe | None = None
    inference_assets: Mapping[str, InferenceAssetProjection] = ...


@dataclass(frozen=True, slots=True)
class SamplingRecipe:
    name: str
    contract: Mapping[str, Any] = ...


class TrainingBuilder(ABC):
    def __init__(self, context: TrainingBuilderContext) -> None: ...

    @abstractmethod
    def build(self) -> TrainingPlan: ...
```

`TrainingBuilderContext.params` 是深复制 mapping；`primary_model`、`process` 与
`objective` 是 core 已构建的身份对象，返回的 Plan 必须原样保留它们。Builder 可以通过
受控 `model_factory(ComponentConfig)`/`objective_factory(ComponentConfig)` 构建额外资产。
Plan 中所有 state root 必须互不重叠且至少包含一个可训练参数。每个 inference asset
slot 必须一对一引用已有 auxiliary module。projection 的 `declaration` 是
sampling reconstruction-only 声明，只能包含从 checkpoint state 重建 module 所需的
参数；下载地址、bootstrap path 或其他 acquisition identity 不应进入该声明。
`capability_role` 是 checkpoint 与 Builder 之间的语义身份，具体行为接口仍由请求资产的
SamplingBuilder 以自己的窄 capability 检查。

`inference_recipe`
为 null 时 checkpoint 不支持 `stochaflow sample`；非 null 时 `name` 必须选择已注册
SamplingBuilder，`contract` 只允许有限数字、字符串、布尔、null 和普通 list/dict，
并应只保存由训练组合确定、不能安全覆盖的 inference 事实。step loss 必须是浮点 scalar
Tensor，`metrics` 中的低成本 report 必须是 scalar numeric value。
`loss_aggregation_weight` 必须是有限、非负的 detached scalar，只控制 epoch loss 报告：
它不会缩放 backward loss，也不会自动成为某个 Metric 的 sample weight。Metric 若需要
权重，Strategy 必须把它显式放进对应 `MetricUpdate`。
所有 managed module 都参与声明的
device/mode、优化和 checkpoint 生命周期；EMA 只跟踪 primary model。

## Training diagnostic 与 provider 扩展边界

`TrainingDiagnostic` 是 observation-only 生命周期。callback 可以通过构建期注入的
`ExperimentLogger` 记录 `diagnostics/...` scalar 日志，也可以写 artifact；callback
必须返回 `None`，其输出不会合并进训练 epoch metric mapping、best checkpoint 判定或
early stopping。diagnostic 可以在 `FitStartEvent` 中观察本次 train/validation iterable，
但不存在 source role、protocol digest、selection eligibility 或 checkpoint provenance
绑定契约。

`TrainBatchEndEvent` 只在成功 optimizer step 后发送，`TrainEpochEndEvent` 在训练 epoch
完成后发送。diagnostic 的 cache、计数器和采样状态都是本次 invocation 的临时状态，
核心不会把它们恢复为训练 checkpoint state。正式模型选择只能使用同一训练 runtime
按 validation phase 聚合得到的 `valid/loss` 或 `valid/metrics/...`；高成本、低频或
采样型 diagnostic 即使记录 scalar 日志，也始终只是观测。

Gaussian quality pipeline 的 provider-level extension surface 也从
`stochaflow.extensions` 导出：

| Provider category | 基类与 context | 局部 Registry |
| --- | --- | --- |
| step scalar | `StepMetricProvider` / `StepMetricContext` | `DIAGNOSTIC_PROVIDERS.step_metrics` |
| sampler scalar | `SamplerMetricProvider` / `SamplerMetricContext` | `DIAGNOSTIC_PROVIDERS.sampler_metrics` |
| denoiser artifact | `DenoiserArtifactProvider` / `DenoiserArtifactContext` | `DIAGNOSTIC_PROVIDERS.denoiser_artifacts` |
| sampler artifact | `SamplerArtifactProvider` / `SamplerArtifactContext` | `DIAGNOSTIC_PROVIDERS.sampler_artifacts` |
| reference quality | `ReferenceMetricProvider` | `DIAGNOSTIC_PROVIDERS.reference_metrics` |

`DiagnosticProviderCatalog` 是这五个局部 Registry 的 typed catalog；
`ProviderSpec`、`ProviderPipelineConfig`、`ReferencePipelineConfig`、
`DiagnosticCadenceConfig`、`DiagnosticSamplingConfig`、`SamplerProfileConfig` 和
`TrajectoryProviderConfig` 是 Gaussian diagnostic 的公开配置值。

这些 Registry 只扩展 Gaussian diagnostic 内部 pipeline，不是 framework-global Metric
Registry，也不让 provider 直接选择 checkpoint。reference provider 的 `compute()` 结果
由 diagnostic 记录为观测日志；sampler statistics、延迟与 artifact 同样不进入 validation
selection。provider 模块必须来自已安装并通过
`extensions.plugins` 选择的可信 distribution；完整插件 provenance 随 resolved config 和
checkpoint 保存。

## Process 根契约

| 符号 | 用途 |
| --- | --- |
| `Process` | 可注册、可迁移、可 checkpoint 的 model-free probability-path 根类型 |
| `GenerativeDynamics` | 已组装生成方向的无行为语义根 |

`Process` 是可选资产。`GenerativeDynamics` 没有 Registry/YAML identity，也没有 universal
`predict`、`step`、`drift`、`score` 或 `denoise` 方法；算法 family 定义自己的窄行为契约。

## Discrete Gaussian family

Gaussian 实现按 layer-first convention 组织：process path/schedule 位于 Process layer，
training target/loss 位于 Training layer，Dynamics/clipping/solver 位于 Sampling
layer。跨层 kernel 只包含 prediction type、`GaussianPrediction` 与纯 Tensor prediction
normalization。这个物理目录约定不是 extension import surface；下表符号仍统一从
`stochaflow.extensions` 导入。

| 符号 | 用途 |
| --- | --- |
| `GaussianNoiseSchedule` | Gaussian schedule 的 family-specific 根契约 |
| `GaussianScales` | 一个 state time 的 signal/noise scales |
| `DiscreteVPSchedule` | immutable discrete VP coefficient capability |
| `DiscreteVPCoefficients` | 一组离散 VP transition coefficients |
| `TabulatedDiscreteVPSchedule` | 基于固定 coefficient table 的可复用 schedule 实现 |
| `DiscreteGaussianDenoisingProcess` | 整数时间、terminal prior、marginal 与 adjacent-posterior 基础 Process capability |
| `DiscreteGaussianProcess` | 内置 coefficient-snapshot Process；另外满足 selected-pair coefficients 与 learned-range variance bounds |
| `PredictionType` | Gaussian model prediction parameterization |
| `GaussianPrediction` | 归一化后的 epsilon 与 predicted-clean 结果 |
| `GaussianTransition` | 一步离散 Gaussian transition distribution |
| `GaussianDenoisingDynamics` | DDPM/DDIM 消费的窄 Gaussian Dynamics capability |
| `GaussianModelDynamics` | 将 Process、model callable、prediction/variance semantics 与 clipping 组合成 Gaussian Dynamics |
| `normalize_gaussian_prediction` | 把 epsilon/x0/v/score model output 归一化为 `GaussianPrediction` |
| `DDPMAncestralSampler` | 内置 full/respaced selected-pair ancestral sampler |
| `DDIMSampler` | 内置 discrete Gaussian DDIM sampler |
| `GaussianDiagnosticSemantics` | Strategy 可选暴露给 Gaussian diagnostics 的 prediction type、variance mode 与 model-invocation capability |
| `gaussian_training_target` | 按 Gaussian prediction type 生成对应训练 target |

这些符号只承诺 discrete Gaussian family 内的行为兼容，不是 Flow Matching、SDE 或
sigma-space solver 的 universal 接口。

### Gaussian training recipe extension

内置标准 recipe 由具体 TrainingBuilder/TrainingStrategy 表达。无条件 learned-range
recipe 例如：

```yaml
training:
  name: gaussian_denoising
  params:
    prediction_type: v
    variance:
      mode: learned_range
```

类条件版本选择 `class_conditional_gaussian_denoising`，并可另外声明
`condition_dropout`。Builder 验证 model 的 fixed `C` 或 learned-range `2C` output
contract，并把 prediction/variance 事实写入 checkpoint inference recipe。

第三方若要实现另一种 SNR weighting、prediction restriction、model signature 或 batch
语义，应提供自己的 Strategy，并以带 namespace 的 TrainingBuilder 注册完整 recipe：

```python
from typing import Any

from stochaflow.extensions import (
    REGISTRIES,
    TrainingBuilder,
    TrainingPlan,
    TrainingStrategy,
    TrainStepOutput,
)


class InverseSnrGaussianStrategy(TrainingStrategy):
    def __init__(self, model, process, objective, *, scale: float) -> None:
        self.model = model
        self.process = process
        self.objective = objective
        self.scale = scale

    def training_step(self, batch: Any) -> TrainStepOutput:
        ...  # interpret batch, call model, and own this recipe's loss semantics


@REGISTRIES.training_builders.register("my_lab.inverse_snr_gaussian")
class InverseSnrGaussianBuilder(TrainingBuilder):
    def build(self) -> TrainingPlan:
        strategy = InverseSnrGaussianStrategy(
            self.context.primary_model,
            self.context.process,
            self.context.objective,
            scale=float(self.context.params.get("scale", 1.0)),
        )
        return TrainingPlan(
            strategy=strategy,
            primary_model=self.context.primary_model,
            process=self.context.process,
            objective=self.context.objective,
        )
```

实际 Builder 应在构造 Strategy 前验证 Process、Objective、model capability、私有参数和
未知字段，并原样保留 core 注入的资产。若 checkpoint 需要支持 sampling，还必须声明与
该 model/strategy 组合匹配的 `SamplingRecipe`。`gaussian_training_target()` 可供自定义
Gaussian Strategy 复用；它不会替 Strategy 决定 batch、model call、weighting 或
reduction。

## Sampling 生命周期与 artifact

| 符号 | 用途 |
| --- | --- |
| `Sampler` | 完整数值求解生命周期；统一 `sample()`，不要求 universal `step()` |
| `SamplerResult` | final state、accepted outer-step 数与 solver diagnostics |
| `SamplingObservation` | initial/accepted/final sampling lifecycle observation |
| `SamplingObserver` | observation consumer protocol |
| `TrajectoryObserver` | 按间隔保留 initial、accepted 与 final observations |
| `SamplingBuilder` | checkpoint recipe 内部的任务级 inference 组合与执行入口；sample config 不直接选择 |
| `SamplingBuilderContext` | resolved recipe params、可选 Process、model/asset providers、device、seed、shape/count/batch size |
| `InferenceModelProvider` | 在 Builder 中选择 raw/EMA inference model 的受控入口 |
| `InferenceAssetProvider` | 按 checkpoint slot 和 role 延迟重建声明的 embedded `nn.Module` |
| `SamplingOutput` | Builder 返回的 ordered `SamplingBatch` 与 metadata |
| `SamplingBatch` | writer-ready samples 与可选 observation trajectory |
| `SamplingArtifactWriter` | 将完整 sampling artifact context 写入文件的契约 |
| `SamplingArtifactContext` | output directory、batches 与 Builder metadata |

当前 writer 生命周期接收已经整体形成的 `SamplingOutput`，不是 streaming sink。

关键签名：

```python
class Sampler(ABC):
    @abstractmethod
    def sample(
        self,
        dynamics: GenerativeDynamics,
        initial_state: Any,
        *,
        generator: torch.Generator | None = None,
        observer: SamplingObserver | None = None,
    ) -> SamplerResult: ...


class SamplingBuilder(ABC):
    def __init__(self, context: SamplingBuilderContext) -> None: ...

    @abstractmethod
    def run(self) -> SamplingOutput: ...


class InferenceAssetProvider:
    @classmethod
    def empty(cls) -> Self: ...

    def get(
        self,
        slot: str,
        *,
        expected_capability_role: str,
    ) -> nn.Module: ...


@dataclass(frozen=True, slots=True)
class SamplingBatch:
    samples: Any
    trajectory: tuple[SamplingObservation, ...] | None = None


@dataclass(frozen=True, slots=True)
class SamplingOutput:
    batches: tuple[SamplingBatch, ...]
    metadata: Mapping[str, Any]


class SamplingArtifactWriter(ABC):
    @abstractmethod
    def write(
        self,
        context: SamplingArtifactContext,
    ) -> Mapping[str, Path]: ...
```

`SamplingBuilderContext.params` 由 runtime 组合：独立完整 config 的 `sample.options`、
可选 `sample.sampler`，以及最后加入且不可覆盖的 `SamplingRecipe.contract`。sample
config 不从 training config 或 checkpoint 继承 invocation 字段；与 fixed contract 冲突
会在 Builder 构造前失败。Context 还提供可选 Process、InferenceModelProvider、
InferenceAssetProvider、device、seed、可选单 item shape、num_samples 和 batch_size。
asset provider 复用 model Registry，只在 `get()` 请求某个 slot 后构造和 strict-load
该 module；未请求的合法资产不会被构造。它先校验 slot 与 role，Builder 再校验
extension-owned capability。加载成功的 module 会迁移到 sampling device、切换为 eval
并按 slot 缓存；失败不会进入缓存。

Builder 的 batches 不能为空，metadata key
必须是字符串且整个 mapping 可 JSON 序列化。trajectory 的 step index 必须严格递增。
Writer 返回值必须非空，跨 writer artifact key 必须唯一，所有路径在返回时必须存在。

`stochaflow sample` 始终要求显式 v12 checkpoint 和显式 `--config`。YAML 顶层必须包含
完整 `sample` mapping，并可选包含 `extensions`；`sample` 必须声明 `sampler`（direct
transform 可为 null）、`options`、`shape`、`num_samples`、`batch_size`、整数 `seed` 和
非空 `writers`，其中 `options.weights` 必须显式为 `raw`、`ema` 或 `auto`。`auto` 在
checkpoint 含 EMA state 时选择 EMA，否则选择 raw；training config 不再拥有
sampling weight-selection policy。checkpoint 只提供 model/Process、fixed inference
recipe、embedded assets 与 required plugin provenance。`extensions.plugins` 只能添加
本次 inference 所需插件，不能删除 checkpoint-required plugins 或替换 recipe。

```bash
stochaflow sample --checkpoint outputs/run/checkpoints/best.pt \
  --config experiments/sample.yaml
```

## Plugin discovery、provenance 与 activation

| 符号 | 用途 |
| --- | --- |
| `ExtensionPluginProvenance` | entry-point name、canonical distribution、version 与 module target |
| `ExtensionActivationPlan` | 无导入副作用的 discovery/provenance 预检结果 |
| `ResolvedExtensions` | 成功激活后的 resolved config、provenance 与 version-acceptance audit |
| `ExtensionSelectionPolicy` | expected provenance 的 `EXACT`/`INTERSECTION` 比较策略 |
| `ExtensionVersionPolicy` | version mismatch 的 `REJECT`/`ALLOW` 策略 |
| `ExtensionVersionMismatch` | expected/current version 差异 |
| `ExtensionVersionAcceptance` | 一次明确接受 version mismatch 的审计记录 |
| `prepare_extension_plugins` | 发现并预检选择，但不导入 extension code |
| `activate_extension_plugins` | 按预检计划导入聚合模块并物化 resolved selection |
| `parse_extension_plugin_provenance` | 严格解析 checkpoint-safe provenance mappings |
| `extension_plugin_provenance_to_dicts` | 将 provenance records 转为 checkpoint/manifest-safe mappings |
| `ExtensionPluginError` | plugin discovery/identity/version/activation 错误根类型 |
| `ExtensionDiscoveryError` | entry-point discovery 或 metadata 无法形成选择 |
| `ExtensionIdentityError` | name/distribution/target 或 provenance 结构不匹配 |
| `ExtensionVersionMismatchError` | 当前 policy 拒绝 version mismatch |
| `ExtensionActivationError` | extension module 导入/激活失败 |
| `ExtensionActivationStateError` | 进程级 activation 状态冲突、重入或先前失败 |

插件 discovery 与 activation 有意分离。一次进程成功激活后 selection 固定；导入失败或
partial decorator registration 后应重启进程。

## 完整性约束

本页列出的名称与当前 `stochaflow.extensions.__all__` 一一对应。新增公共契约时应先
更新该 `__all__`，再同步本页；仅存在于内部 package 的名称不应被 extension 依赖。

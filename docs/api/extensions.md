# Extension 公共 API

第三方 extension 应只依赖：

```python
from stochaflow.extensions import ...
```

不要从 `stochaflow.data`、`stochaflow.training`、`stochaflow.processes`、
`stochaflow.sampling` 或 `stochaflow.utils` 的内部模块路径导入契约。内部文件可以在尚未
发布的重构中移动；`stochaflow.extensions.__all__` 是 extension 作者的公共入口和本页的
唯一符号清单。

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

## Training

| 符号 | 用途 |
| --- | --- |
| `TrainingBuilder` | 组合注入资产和项目私有资产，返回一个 `TrainingPlan` |
| `TrainingBuilderContext` | primary model、可选 Process/Objective、私有 `params` 与受控 model/objective factory |
| `TrainingPlan` | Strategy、primary model、可选 Process/Objective、具名 auxiliary modules、inference asset projections 和可选 fixed inference recipe |
| `InferenceAssetProjection` | 将一个 managed auxiliary module 投影为 checkpoint-owned inference asset |
| `SamplingRecipe` | checkpoint 内部 SamplingBuilder identity 与不可由 sample request 覆盖的 JSON-safe contract |
| `ManagedTrainingModule` | 辅助 `nn.Module` 及其 core-managed mode policy |
| `TrainingStrategy` | 只定义 batch interpretation、forward、loss 与 metric 计算 |
| `DenoiserChannelLayout` | model 可选暴露的静态 `in_channels`/`out_channels` capability；Gaussian Builder 据此在组合时预检 fixed `C` 或 learned-range `2C` 输出 |
| `DeviceTransferableBatch` | 自定义领域 batch 可选择实现的显式设备迁移 capability |
| `ReferenceImageBatchSemantics` | Strategy 可选实现的 reference-metric image extraction capability |
| `TrainStepOutput` | Strategy 返回的 scalar loss、metrics 与 diagnostics |
| `MSEObjective` | 内置 task-neutral scalar MSE Objective |
| `PerSampleObjective` | 可选的逐样本 loss capability |
| `compute_objective` | 校验并执行 scalar Objective，同时读取可选逐样本 capability |
| `TrainingDiagnostic` | training diagnostic 生命周期根契约 |
| `DiagnosticBuildContext` | diagnostic 构建期 logger 和输出目录；采样 shape 由 diagnostic 私有配置拥有 |
| `FitStartEvent` | fit 开始事件 |
| `TrainBatchEndEvent` | 成功 optimizer step 后的事件 |
| `TrainEpochEndEvent` | 一个 epoch 完成后的事件 |
| `ExperimentLogger` | extension logger backend 契约 |

Strategy 不是 `nn.Module`，也不移动、冻结、选择或序列化资产；这些生命周期由
TrainingPlan 和核心 runtime 管理。

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
Tensor，metric 必须是 scalar numeric value。所有 managed module 都参与声明的
device/mode、优化和 checkpoint 生命周期；EMA 只跟踪 primary model。

## Process 根契约

| 符号 | 用途 |
| --- | --- |
| `Process` | 可注册、可迁移、可 checkpoint 的 model-free probability-path 根类型 |
| `GenerativeDynamics` | 已组装生成方向的无行为语义根 |

`Process` 是可选资产。`GenerativeDynamics` 没有 Registry/YAML identity，也没有 universal
`predict`、`step`、`drift`、`score` 或 `denoise` 方法；算法 family 定义自己的窄行为契约。

## Discrete Gaussian family

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

## Sampling 生命周期与 artifact

| 符号 | 用途 |
| --- | --- |
| `Sampler` | 完整数值求解生命周期；统一 `sample()`，不要求 universal `step()` |
| `SamplerResult` | final state、accepted outer-step 数与 solver diagnostics |
| `SamplingObservation` | initial/accepted/final sampling lifecycle observation |
| `SamplingObserver` | observation consumer protocol |
| `TrajectoryObserver` | 按间隔保留 initial、accepted 与 final observations |
| `SamplingBuilder` | checkpoint recipe 内部的任务级 inference 组合与执行入口；sample request 不直接选择 |
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

`SamplingBuilderContext.params` 由 runtime 组合：checkpoint `sampling.options` 的
resolved shallow merge、可选顶层 `sampling.sampler`，以及最后加入且不可覆盖的
`SamplingRecipe.contract`。`options` 不得包含保留 key `sampler`；与 contract 冲突会在
Builder 构造前失败。Context 还提供可选 Process、InferenceModelProvider、
InferenceAssetProvider、device、seed、可选单 item shape、num_samples 和 batch_size。
asset provider 复用 model Registry，只在 `get()` 请求某个 slot 后构造和 strict-load
该 module；未请求的合法资产不会被构造。它先校验 slot 与 role，Builder 再校验
extension-owned capability。加载成功的 module 会迁移到 sampling device、切换为 eval
并按 slot 缓存；失败不会进入缓存。

Builder 的 batches 不能为空，metadata key
必须是字符串且整个 mapping 可 JSON 序列化。trajectory 的 step index 必须严格递增。
Writer 返回值必须非空，跨 writer artifact key 必须唯一，所有路径在返回时必须存在。

`stochaflow sample` 始终要求显式 v10 checkpoint。可选 request YAML 顶层只允许
`sampling` 与 optional `extensions`，不能声明 `run_after_training` 或 Builder。
未提供的 request 字段继承 checkpoint；`options` 浅合并，`sampler`/`writers` 原子替换，
插件 selection 只允许 additive 扩展。

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

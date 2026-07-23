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
| `REGISTRIES` | 进程级 Registry catalog；extension 用对应 registry 的 `register(name)` 注册组件 |
| `Registry` | 具名组件容器及 resolve/create/register 契约 |
| `RegistryError` | 注册、解析或构建失败 |

`RegistryCatalog` 不是公共导出。extension 应使用 `REGISTRIES`，不要创建或依赖 catalog
内部实现。

## Data

| 符号 | 用途 |
| --- | --- |
| `DataBuilder` | 组装一份独立训练运行的完整数据栈 |
| `DataBuilderContext` | 深复制的私有 `params` 与 experiment seed |
| `DataLoaders` | 可重复迭代的 train/validation/test loader 与可选 `steps_per_epoch` |

数据公共 API 不包含 Dataset、split、source、collate、bucket 或 PyTorch sampler
抽象；这些由具体 DataBuilder 私有拥有。

最小实现签名：

```python
@dataclass(frozen=True, slots=True)
class DataBuilderContext:
    params: dict[str, Any]  # 深复制
    seed: int


@dataclass(frozen=True, slots=True)
class DataLoaders:
    train: Iterable[Any]
    validation: Iterable[Any] | None = None
    test: Iterable[Any] | None = None
    steps_per_epoch: int | None = None


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
| `TrainingPlan` | Strategy、primary model、可选 Process/Objective 和具名 auxiliary modules |
| `ManagedTrainingModule` | 辅助 `nn.Module` 及其 core-managed mode policy |
| `TrainingStrategy` | 只定义 batch interpretation、forward、loss 与 metric 计算 |
| `TrainStepOutput` | Strategy 返回的 scalar loss、metrics 与 diagnostics |
| `MSEObjective` | 内置 task-neutral scalar MSE Objective |
| `PerSampleObjective` | 可选的逐样本 loss capability |
| `compute_objective` | 校验并执行 scalar Objective，同时读取可选逐样本 capability |
| `TrainingDiagnostic` | training diagnostic 生命周期根契约 |
| `DiagnosticBuildContext` | diagnostic 构建期 logger、输出目录和可选 sample shape |
| `FitStartEvent` | fit 开始事件 |
| `TrainBatchEndEvent` | 成功 optimizer step 后的事件 |
| `TrainEpochEndEvent` | 一个 epoch 完成后的事件 |
| `ExperimentLogger` | extension logger backend 契约 |

Strategy 不是 `nn.Module`，也不移动、冻结、选择或序列化资产；这些生命周期由
TrainingPlan 和核心 runtime 管理。

core 会在每次 `TrainingDiagnostic` public callback 外保存并恢复 Python、NumPy、
Torch CPU 以及相关 CUDA device 的 global RNG state。一个 callback 内使用 global RNG
不会改变训练或其他 callback 看到的随机流；若 diagnostic 需要跨 callback 延续自己的
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


@dataclass(frozen=True, slots=True)
class ManagedTrainingModule:
    module: nn.Module
    mode: Literal["follow", "eval"] = "follow"


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    strategy: TrainingStrategy
    primary_model: nn.Module
    process: Process | None = None
    objective: nn.Module | None = None
    auxiliary_modules: Mapping[str, ManagedTrainingModule] = ...


class TrainingBuilder(ABC):
    def __init__(self, context: TrainingBuilderContext) -> None: ...

    @abstractmethod
    def build(self) -> TrainingPlan: ...
```

`TrainingBuilderContext.params` 是深复制 mapping；`primary_model`、`process` 与
`objective` 是 core 已构建的身份对象，返回的 Plan 必须原样保留它们。Builder 可以通过
受控 `model_factory(ComponentConfig)`/`objective_factory(ComponentConfig)` 构建额外资产。
Plan 中所有 state root 必须互不重叠且至少包含一个可训练参数。step loss 必须是浮点 scalar
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
| `DiscreteGaussianDenoisingProcess` | 整数时间、terminal prior、marginal 与 adjacent-posterior Process capability |
| `DiscreteGaussianProcess` | 内置 coefficient-snapshot Process 实现 |
| `PredictionType` | Gaussian model prediction parameterization |
| `GaussianPrediction` | 归一化后的 epsilon 与 predicted-clean 结果 |
| `GaussianTransition` | 一步离散 Gaussian transition distribution |
| `GaussianDenoisingDynamics` | DDPM/DDIM 消费的窄 Gaussian Dynamics capability |
| `GaussianModelDynamics` | 将 Process、model callable、prediction semantics 与 clipping 组合成 Gaussian Dynamics |
| `normalize_gaussian_prediction` | 把 epsilon/x0/v/score model output 归一化为 `GaussianPrediction` |
| `DDPMAncestralSampler` | 内置 discrete Gaussian ancestral sampler |
| `DDIMSampler` | 内置 discrete Gaussian DDIM sampler |
| `GaussianDiagnosticSemantics` | Strategy 可选暴露给 Gaussian diagnostics 的 model-invocation capability |
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
| `SamplingBuilder` | 任务级采样组合与执行入口 |
| `SamplingBuilderContext` | 私有 params、可选 Process、model provider、device、seed、shape/count/batch size |
| `InferenceModelProvider` | 在 Builder 中选择 raw/EMA inference model 的受控入口 |
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

`SamplingBuilderContext` 提供深复制的 `params`、可选 Process、InferenceModelProvider、
device、seed、可选单样本 shape、num_samples 和 batch_size。Builder 的 batches 不能为空，
metadata key 必须是字符串且整个 mapping 可 JSON 序列化。trajectory 的 step index 必须
严格递增。Writer 返回值必须非空，跨 writer artifact key 必须唯一，所有路径在返回时必须
存在。

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

本页列出的 70 个名称与当前 `stochaflow.extensions.__all__` 一一对应。新增公共契约时应先
更新该 `__all__`，再同步本页；仅存在于内部 package 的名称不应被 extension 依赖。

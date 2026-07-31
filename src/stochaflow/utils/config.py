"""Centralized configuration schema and loading utilities."""

import re
from copy import deepcopy
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

import yaml

from stochaflow.metrics.config import MetricSpec, validate_metric_spec

TRAINING_METRIC_PHASES = frozenset({"train", "validation", "test"})
_METRIC_TAG_SEGMENT = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
_VALIDATION_MONITOR_PATTERN = re.compile(
    rf"^valid/(?:loss|metrics/{_METRIC_TAG_SEGMENT}"
    rf"(?:/{_METRIC_TAG_SEGMENT})?)$"
)


class ConfigError(ValueError):
    """Raised when a config file does not match the expected schema."""


@dataclass(slots=True)
class ComponentConfig:
    """Reusable component declaration with a registry name and free-form params."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingMetricConfig:
    """Flat training metric declaration composed into a task-neutral spec."""

    id: str
    name: str
    channel: str
    params: dict[str, Any] = field(default_factory=dict)
    phases: list[str] = field(default_factory=lambda: ["validation"])

    def to_metric_spec(self) -> MetricSpec:
        """Return the task-neutral construction declaration."""

        params = (
            dict(self.params)
            if type(cast(object, self.params)) is dict
            else self.params
        )
        return MetricSpec(
            id=self.id,
            name=self.name,
            channel=self.channel,
            params=params,
        )


def validate_training_metric_configs(
    configs: object,
    *,
    path: str = "metrics",
) -> list[TrainingMetricConfig]:
    """Validate flat training metric declarations."""

    if not isinstance(configs, list):
        raise TypeError(f"{path} must be a list")
    declarations = cast(list[object], configs)
    seen_ids: set[str] = set()
    validated: list[TrainingMetricConfig] = []
    for index, value in enumerate(declarations):
        item_path = f"{path}[{index}]"
        if not isinstance(value, TrainingMetricConfig):
            raise TypeError(f"{item_path} must be a TrainingMetricConfig")
        validate_metric_spec(value.to_metric_spec(), path=item_path)
        phases = cast(object, value.phases)
        if not isinstance(phases, list):
            raise TypeError(f"{item_path}.phases must be a list")
        if not phases:
            raise ValueError(f"{item_path}.phases must not be empty")
        seen_phases: set[str] = set()
        for phase_index, phase in enumerate(phases):
            if not isinstance(phase, str) or phase not in TRAINING_METRIC_PHASES:
                raise ValueError(
                    f"{item_path}.phases[{phase_index}] must be train, "
                    "validation, or test"
                )
            if phase in seen_phases:
                raise ValueError(
                    f"{item_path}.phases contains duplicate phase {phase!r}"
                )
            seen_phases.add(phase)
        if value.id in seen_ids:
            raise ValueError(f"{path} contains duplicate metric id {value.id!r}")
        seen_ids.add(value.id)
        validated.append(value)
    return validated


def validate_training_monitor_key(
    value: object,
    *,
    path: str = "training monitor",
) -> str:
    """Validate a validation-loss or validation-metric selection key."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{path} must not contain surrounding whitespace")
    if _VALIDATION_MONITOR_PATTERN.fullmatch(value) is None:
        raise ValueError(
            f"{path} must be 'valid/loss' or a canonical validation metric "
            "key such as 'valid/metrics/prediction_mae'"
        )
    return value


@dataclass(slots=True)
class ExtensionsConfig:
    """Installed extension plugins selected for this invocation.

    ``None`` explicitly selects every installed ``stochaflow.extensions``
    entry point.  An empty list, including the default, selects no third-party
    plugins.  Configuration parsing never imports plugin code.
    """

    plugins: list[str] | None = field(default_factory=list)


@dataclass(slots=True)
class ExperimentConfig:
    """Metadata for an experiment run."""

    name: str
    seed: int = 42
    output_dir: str = "outputs/default"
    exp_id: str | None = None


@dataclass(slots=True)
class SamplingConfig:
    """Checkpoint-bound sampling defaults without composition internals."""

    run_after_training: bool = False
    sampler: ComponentConfig | None = None
    options: dict[str, Any] = field(default_factory=dict)
    shape: list[int] | None = None
    num_samples: int = 16
    batch_size: int = 16
    seed: int | None = None
    writers: list[ComponentConfig] = field(
        default_factory=lambda: [ComponentConfig(name="tensor")]
    )


@dataclass(frozen=True, slots=True)
class SampleRequest:
    """Typed partial override for one checkpoint-bound sampling invocation."""

    sampler: ComponentConfig | None = None
    options: dict[str, Any] = field(default_factory=dict)
    shape: list[int] | None = None
    num_samples: int | None = None
    batch_size: int | None = None
    seed: int | None = None
    writers: list[ComponentConfig] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ParsedSampleRequest:
    """A typed request together with the fields explicitly supplied by its user."""

    request: SampleRequest
    provided_fields: frozenset[str]


@dataclass(slots=True)
class LRSchedulerConfig:
    """Optimizer learning-rate scheduler declaration."""

    name: str
    interval: str = "step"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EMAConfig:
    """Exponential moving average policy for model weights."""

    enabled: bool = False
    decay: float = 0.9999
    update_after_step: int = 0
    update_every: int = 1
    use_for_sampling: bool = True


@dataclass(slots=True)
class EarlyStoppingConfig:
    """Epoch-metric selection and early-stopping policy."""

    enabled: bool = False
    monitor: str = "valid/loss"
    mode: str = "min"
    patience: int = 10
    min_delta: float = 0.0


@dataclass(slots=True)
class TrainerConfig:
    """Generic trainer loop configuration."""

    num_epochs: int = 1
    device: str = "cpu"
    precision: str = "fp32"
    accumulate_grad_batches: int = 1
    max_grad_norm: float | None = None
    show_progress: bool = True
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


@dataclass(slots=True)
class LoggingConfig:
    """Logging cadence and display controls."""

    log_every: int = 100
    backends: list[ComponentConfig] = field(
        default_factory=lambda: [ComponentConfig(name="local")]
    )
    torch_logs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ArtifactConfig:
    """Checkpoint artifact cadence."""

    checkpoint_every: int = 1


@dataclass(slots=True)
class StochaflowConfig:
    """Top-level project configuration."""

    experiment: ExperimentConfig
    data: ComponentConfig
    model: ComponentConfig
    training: ComponentConfig
    objective: ComponentConfig | None = None
    process: ComponentConfig | None = None
    metrics: list[TrainingMetricConfig] = field(default_factory=list)
    extensions: ExtensionsConfig = field(default_factory=ExtensionsConfig)
    optimizer: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(
            name="torch.optim.Adam",
            params={"lr": 2e-4},
        )
    )
    lr_scheduler: LRSchedulerConfig | None = None
    ema: EMAConfig = field(default_factory=EMAConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    diagnostics: list[ComponentConfig] = field(default_factory=list)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)

    def validate(self) -> None:
        """Validate cross-field invariants."""

        data_name = cast(object, self.data.name)
        if not isinstance(data_name, str) or not data_name.strip():
            raise ConfigError("data.name must be a non-empty registry name")
        if not isinstance(cast(object, self.data.params), dict):
            raise ConfigError("data.params must be a mapping")
        training_name = cast(object, self.training.name)
        if not isinstance(training_name, str) or not training_name.strip():
            raise ConfigError("training.name must be a non-empty registry name")
        if not isinstance(cast(object, self.training.params), dict):
            raise ConfigError("training.params must be a mapping")
        if self.objective is not None:
            objective_name = cast(object, self.objective.name)
            if not isinstance(objective_name, str) or not objective_name.strip():
                raise ConfigError("objective.name must be a non-empty registry name")
            if not isinstance(cast(object, self.objective.params), dict):
                raise ConfigError("objective.params must be a mapping")
        try:
            validate_training_metric_configs(self.metrics)
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
        plugins_value = cast(object, self.extensions.plugins)
        if plugins_value is not None and not isinstance(plugins_value, list):
            raise ConfigError("extensions.plugins must be a list or null")
        if self.extensions.plugins is not None:
            seen_plugins: set[str] = set()
            for index, declared_plugin in enumerate(self.extensions.plugins):
                plugin = cast(object, declared_plugin)
                if not isinstance(plugin, str) or not plugin.strip():
                    raise ConfigError(
                        f"extensions.plugins[{index}] must be a non-empty string"
                    )
                if plugin != plugin.strip():
                    raise ConfigError(
                        f"extensions.plugins[{index}] must not contain "
                        "leading or trailing whitespace"
                    )
                if plugin in seen_plugins:
                    raise ConfigError(
                        f"extensions.plugins contains duplicate entry-point name "
                        f"'{plugin}'"
                    )
                seen_plugins.add(plugin)
        if self.trainer.num_epochs <= 0:
            raise ConfigError("trainer.num_epochs must be positive")
        precision = cast(object, self.trainer.precision)
        if not isinstance(precision, str) or precision not in {
            "fp32",
            "bf16-mixed",
            "fp16-mixed",
        }:
            raise ConfigError(
                "trainer.precision must be fp32, bf16-mixed, or fp16-mixed"
            )
        accumulation = cast(object, self.trainer.accumulate_grad_batches)
        if (
            not isinstance(accumulation, int)
            or isinstance(accumulation, bool)
            or accumulation <= 0
        ):
            raise ConfigError(
                "trainer.accumulate_grad_batches must be a positive integer"
            )
        if not 0.0 <= self.ema.decay < 1.0:
            raise ConfigError("ema.decay must satisfy 0 <= decay < 1")
        if self.ema.update_after_step < 0:
            raise ConfigError("ema.update_after_step must be non-negative")
        if self.ema.update_every <= 0:
            raise ConfigError("ema.update_every must be positive")
        if self.process is not None:
            process_name = cast(object, self.process.name)
            if not isinstance(process_name, str) or not process_name.strip():
                raise ConfigError("process.name must be a non-empty registry name")
            if not isinstance(cast(object, self.process.params), dict):
                raise ConfigError("process.params must be a mapping")
        _validate_sampling_config(self.sampling, path="sampling")
        optimizer_name = cast(object, self.optimizer.name)
        if not isinstance(optimizer_name, str) or not optimizer_name.strip():
            raise ConfigError("optimizer.name must be a non-empty string")
        if not isinstance(cast(object, self.optimizer.params), dict):
            raise ConfigError("optimizer.params must be a mapping")
        if "params" in self.optimizer.params:
            raise ConfigError(
                "optimizer.params cannot override runtime parameter 'params'"
            )
        if self.lr_scheduler is not None:
            scheduler_name = cast(object, self.lr_scheduler.name)
            if not isinstance(scheduler_name, str) or not scheduler_name.strip():
                raise ConfigError("lr_scheduler.name must be a non-empty string")
            if not isinstance(cast(object, self.lr_scheduler.params), dict):
                raise ConfigError("lr_scheduler.params must be a mapping")
            if "optimizer" in self.lr_scheduler.params:
                raise ConfigError(
                    "lr_scheduler.params cannot override runtime parameter "
                    "'optimizer'"
                )
            if self.lr_scheduler.interval not in {"step", "epoch"}:
                raise ConfigError("lr_scheduler.interval must be 'step' or 'epoch'")
        if self.trainer.max_grad_norm is not None and self.trainer.max_grad_norm <= 0:
            raise ConfigError("trainer.max_grad_norm must be positive when provided")
        if self.trainer.early_stopping.mode not in {"min", "max"}:
            raise ConfigError("trainer.early_stopping.mode must be 'min' or 'max'")
        try:
            validate_training_monitor_key(
                self.trainer.early_stopping.monitor,
                path="trainer.early_stopping.monitor",
            )
        except (TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
        if self.trainer.early_stopping.patience <= 0:
            raise ConfigError("trainer.early_stopping.patience must be positive")
        if self.trainer.early_stopping.min_delta < 0:
            raise ConfigError("trainer.early_stopping.min_delta must be non-negative")
        if self.logging.log_every <= 0:
            raise ConfigError("logging.log_every must be positive")
        if len(self.logging.backends) == 0:
            raise ConfigError("logging.backends must declare at least one backend")
        if self.artifacts.checkpoint_every <= 0:
            raise ConfigError("artifacts.checkpoint_every must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Convert the config object back into a plain dictionary."""

        return asdict(self)


def _is_dataclass_type(annotation: Any) -> bool:
    return isinstance(annotation, type) and is_dataclass(annotation)


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    if origin not in (Union, UnionType):
        return annotation
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1:
        return args[0]
    return annotation


def _allows_none(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is None:
        return annotation is Any
    return origin in (Union, UnionType) and type(None) in get_args(annotation)


def _coerce_value(annotation: Any, value: Any, path: str) -> Any:
    if value is None:
        if _allows_none(annotation):
            return None
        raise ConfigError(f"{path} must not be null")
    annotation = _unwrap_optional(annotation)
    origin = get_origin(annotation)
    if origin is list:
        (element_type,) = get_args(annotation) or (Any,)
        if not isinstance(value, list):
            raise ConfigError(f"{path} must be a list")
        return [
            _coerce_value(element_type, item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"{path} must be a mapping")
        return deepcopy(value)
    if _is_dataclass_type(annotation):
        return _coerce_dataclass(annotation, value, path)
    return value


def _coerce_dataclass(cls: type[Any], raw: Any, path: str) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must be a mapping")

    field_map = {field_info.name: field_info for field_info in fields(cls)}
    type_hints = get_type_hints(cls)
    unknown = sorted(set(raw) - set(field_map))
    if unknown:
        unknown_fields = ", ".join(f"{path}.{name}" for name in unknown)
        raise ConfigError(f"unknown config field(s): {unknown_fields}")
    missing = sorted(
        name
        for name, field_info in field_map.items()
        if name not in raw
        and field_info.default is MISSING
        and field_info.default_factory is MISSING
    )
    if missing:
        missing_fields = ", ".join(f"{path}.{name}" for name in missing)
        raise ConfigError(
            f"missing required config field(s): {missing_fields}"
        )

    kwargs: dict[str, Any] = {}
    for name, field_info in field_map.items():
        if name in raw:
            kwargs[name] = _coerce_value(
                type_hints.get(name, field_info.type),
                raw[name],
                f"{path}.{name}",
            )
    return cls(**kwargs)


def coerce_config_section(cls: type[Any], raw: Any, path: str) -> Any:
    """Build one typed configuration section from an untrusted mapping."""

    if not _is_dataclass_type(cls):
        raise TypeError("configuration section type must be a dataclass")
    return _coerce_dataclass(cls, deepcopy(raw), path)


def parse_sample_request(raw: object) -> ParsedSampleRequest:
    """Parse a strict partial sampling request without applying checkpoint defaults."""

    if not isinstance(raw, dict):
        raise ConfigError("config.sampling must be a mapping")
    request = cast(
        SampleRequest,
        coerce_config_section(SampleRequest, raw, "config.sampling"),
    )
    _validate_sample_request(
        request,
        provided_fields=frozenset(raw),
        path="sampling",
    )
    return ParsedSampleRequest(request, frozenset(raw))


def apply_sample_request(
    base: SamplingConfig,
    parsed: ParsedSampleRequest,
) -> SamplingConfig:
    """Apply a typed field-wise request to checkpoint sampling defaults."""

    request = parsed.request
    provided = parsed.provided_fields
    sampler = (
        deepcopy(request.sampler)
        if "sampler" in provided
        else deepcopy(base.sampler)
    )
    shape = deepcopy(request.shape) if "shape" in provided else deepcopy(base.shape)
    num_samples = (
        cast(int, request.num_samples)
        if "num_samples" in provided
        else base.num_samples
    )
    batch_size = (
        cast(int, request.batch_size)
        if "batch_size" in provided
        else base.batch_size
    )
    seed = request.seed if "seed" in provided else base.seed
    writers = (
        deepcopy(request.writers)
        if "writers" in provided
        else deepcopy(base.writers)
    )
    options = deepcopy(base.options)
    if "options" in provided:
        options.update(deepcopy(request.options))
    result = SamplingConfig(
        run_after_training=base.run_after_training,
        sampler=sampler,
        options=options,
        shape=shape,
        num_samples=num_samples,
        batch_size=batch_size,
        seed=seed,
        writers=writers,
    )
    _validate_sampling_config(result, path="sampling")
    return result


def _validate_sampling_config(config: SamplingConfig, *, path: str) -> None:
    if not isinstance(cast(object, config.run_after_training), bool):
        raise ConfigError(f"{path}.run_after_training must be boolean")
    if config.sampler is not None:
        _validate_component(config.sampler, path=f"{path}.sampler")
    _validate_options(config.options, path=f"{path}.options")
    _validate_shape(config.shape, path=f"{path}.shape")
    _positive_int(config.num_samples, path=f"{path}.num_samples")
    _positive_int(config.batch_size, path=f"{path}.batch_size")
    _optional_int(config.seed, path=f"{path}.seed")
    _validate_writers(config.writers, path=f"{path}.writers")


def _validate_sample_request(
    request: SampleRequest,
    *,
    provided_fields: frozenset[str],
    path: str,
) -> None:
    if "sampler" in provided_fields and request.sampler is not None:
        _validate_component(request.sampler, path=f"{path}.sampler")
    if "options" in provided_fields:
        _validate_options(request.options, path=f"{path}.options")
    if "shape" in provided_fields:
        _validate_shape(request.shape, path=f"{path}.shape")
    if "num_samples" in provided_fields:
        _positive_int(request.num_samples, path=f"{path}.num_samples")
    if "batch_size" in provided_fields:
        _positive_int(request.batch_size, path=f"{path}.batch_size")
    if "seed" in provided_fields:
        _optional_int(request.seed, path=f"{path}.seed")
    if "writers" in provided_fields:
        _validate_writers(request.writers, path=f"{path}.writers")


def _validate_component(value: ComponentConfig, *, path: str) -> None:
    name = cast(object, value.name)
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{path}.name must be a non-empty string")
    if name != name.strip():
        raise ConfigError(f"{path}.name must not contain surrounding whitespace")
    if not isinstance(cast(object, value.params), dict):
        raise ConfigError(f"{path}.params must be a mapping")


def _validate_options(value: object, *, path: str) -> None:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be a mapping")
    for key in value:
        if not isinstance(key, str) or not key:
            raise ConfigError(f"{path} keys must be non-empty strings")
    if "sampler" in value:
        raise ConfigError(
            f"{path}.sampler is reserved; use the top-level sampling.sampler field"
        )


def _validate_shape(value: object, *, path: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path} must be a non-empty list or null")
    for index, declared_dimension in enumerate(value):
        dimension = cast(object, declared_dimension)
        if not isinstance(dimension, int) or isinstance(dimension, bool):
            raise ConfigError(f"{path}[{index}] must be an integer")
        if dimension <= 0:
            raise ConfigError(f"{path}[{index}] must be positive")


def _positive_int(value: object, *, path: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{path} must be a positive integer")


def _optional_int(value: object, *, path: str) -> None:
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool)
    ):
        raise ConfigError(f"{path} must be an integer or null")


def _validate_writers(value: object, *, path: str) -> None:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{path} must declare at least one writer")
    for index, writer in enumerate(value):
        if not isinstance(writer, ComponentConfig):
            raise ConfigError(f"{path}[{index}] must be a component mapping")
        _validate_component(writer, path=f"{path}[{index}]")


def load_config(path: str | Path) -> StochaflowConfig:
    """Load and validate a YAML config file without importing extensions."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a top-level mapping")

    config = _coerce_dataclass(StochaflowConfig, raw, "config")
    config.validate()
    return config


def load_config_dict(raw: dict[str, Any]) -> StochaflowConfig:
    """Load a plain configuration without mutating it or importing extensions."""

    config = _coerce_dataclass(StochaflowConfig, raw, "config")
    config.validate()
    return config

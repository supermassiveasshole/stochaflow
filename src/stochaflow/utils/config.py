"""Centralized configuration schema and loading utilities."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when a config file does not match the expected schema."""


@dataclass(slots=True)
class ComponentConfig:
    """Reusable component declaration with a registry name and free-form params."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtensionsConfig:
    """Import paths for user-defined Registry components."""

    modules: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ExperimentConfig:
    """Metadata for an experiment run."""

    name: str
    seed: int = 42
    output_dir: str = "outputs/default"
    exp_id: str | None = None


@dataclass(slots=True)
class SamplingConfig:
    """Standalone and post-training sampling configuration."""

    builder: ComponentConfig | None = None
    shape: list[int] | None = None
    num_samples: int = 16
    batch_size: int = 16
    seed: int | None = None
    writers: list[ComponentConfig] = field(
        default_factory=lambda: [ComponentConfig(name="tensor")]
    )


@dataclass(slots=True)
class OptimizerConfig:
    """Optimizer declaration and hyperparameters."""

    name: str = "adam"
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 2e-4,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        }
    )


@dataclass(slots=True)
class LRSchedulerConfig:
    """Optimizer learning-rate scheduler declaration."""

    name: str | None = None
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
    """Validation-based early stopping policy."""

    enabled: bool = False
    monitor: str = "valid_loss"
    mode: str = "min"
    patience: int = 10
    min_delta: float = 0.0


@dataclass(slots=True)
class TrainerConfig:
    """Generic trainer loop configuration."""

    num_epochs: int = 1
    device: str = "cpu"
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
    objective: ComponentConfig
    process: ComponentConfig | None = None
    extensions: ExtensionsConfig = field(default_factory=ExtensionsConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
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
        for index, declared_module in enumerate(self.extensions.modules):
            module = cast(object, declared_module)
            if not isinstance(module, str) or not module.strip():
                raise ConfigError(
                    f"extensions.modules[{index}] must be a non-empty string"
                )
        if self.trainer.num_epochs <= 0:
            raise ConfigError("trainer.num_epochs must be positive")
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
        if self.sampling.builder is not None:
            builder_name = cast(object, self.sampling.builder.name)
            if not isinstance(builder_name, str) or not builder_name.strip():
                raise ConfigError("sampling.builder.name must be a non-empty string")
            if not isinstance(cast(object, self.sampling.builder.params), dict):
                raise ConfigError("sampling.builder.params must be a mapping")
        if self.sampling.num_samples <= 0:
            raise ConfigError("sampling.num_samples must be positive")
        if self.sampling.batch_size <= 0:
            raise ConfigError("sampling.batch_size must be positive")
        if self.sampling.shape is not None:
            if not self.sampling.shape:
                raise ConfigError("sampling.shape must not be empty when provided")
            for index, declared_dimension in enumerate(self.sampling.shape):
                dimension = cast(object, declared_dimension)
                if not isinstance(dimension, int) or isinstance(dimension, bool):
                    raise ConfigError(f"sampling.shape[{index}] must be an integer")
                if dimension <= 0:
                    raise ConfigError(f"sampling.shape[{index}] must be positive")
        if not self.sampling.writers:
            raise ConfigError("sampling.writers must declare at least one writer")
        for index, writer in enumerate(self.sampling.writers):
            writer_name = cast(object, writer.name)
            if not isinstance(writer_name, str) or not writer_name.strip():
                raise ConfigError(
                    f"sampling.writers[{index}].name must be a non-empty string"
                )
            if not isinstance(cast(object, writer.params), dict):
                raise ConfigError(
                    f"sampling.writers[{index}].params must be a mapping"
                )
        if self.lr_scheduler.name is not None:
            scheduler_name = cast(object, self.lr_scheduler.name)
            if not isinstance(scheduler_name, str) or not scheduler_name:
                raise ConfigError("lr_scheduler.name must be a non-empty string or null")
            if self.lr_scheduler.interval not in {"step", "epoch"}:
                raise ConfigError("lr_scheduler.interval must be 'step' or 'epoch'")
            if self.lr_scheduler.name == "warmup_cosine":
                warmup_steps = self.lr_scheduler.params.get("warmup_steps")
                if not isinstance(warmup_steps, int) or warmup_steps <= 0:
                    raise ConfigError(
                        "lr_scheduler.params.warmup_steps must be a positive integer"
                    )
                total_steps = self.lr_scheduler.params.get("total_steps")
                if total_steps != "auto":
                    if not isinstance(total_steps, int) or total_steps <= 0:
                        raise ConfigError(
                            "lr_scheduler.params.total_steps must be a positive "
                            "integer or 'auto'"
                        )
                    if total_steps <= warmup_steps:
                        raise ConfigError(
                            "lr_scheduler.params.total_steps must be greater than "
                            "warmup_steps"
                        )
                min_lr_ratio = self.lr_scheduler.params.get("min_lr_ratio", 0.0)
                if not isinstance(min_lr_ratio, (int, float)):
                    raise ConfigError(
                        "lr_scheduler.params.min_lr_ratio must be numeric"
                    )
                if not 0.0 <= float(min_lr_ratio) <= 1.0:
                    raise ConfigError(
                        "lr_scheduler.params.min_lr_ratio must be between 0 and 1"
                    )
        if self.trainer.max_grad_norm is not None and self.trainer.max_grad_norm <= 0:
            raise ConfigError("trainer.max_grad_norm must be positive when provided")
        if self.trainer.early_stopping.mode not in {"min", "max"}:
            raise ConfigError("trainer.early_stopping.mode must be 'min' or 'max'")
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
    union_types: set[Any] = {Union}
    try:
        from types import UnionType

        union_types.add(UnionType)
    except ImportError:
        pass
    if origin not in union_types:
        return annotation
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1:
        return args[0]
    return annotation


def _allows_none(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin is None:
        return annotation is Any
    union_types: set[Any] = {Union}
    try:
        from types import UnionType

        union_types.add(UnionType)
    except ImportError:
        pass
    return origin in union_types and type(None) in get_args(annotation)


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


def _validate_module_declarations(value: Any, *, path: str) -> list[str]:
    if not isinstance(value, list):
        raise ConfigError(f"{path} must be a list")
    for index, module in enumerate(value):
        if not isinstance(module, str) or not module.strip():
            raise ConfigError(f"{path}[{index}] must be a non-empty string")
    return value


def _load_configured_modules(config: StochaflowConfig) -> None:
    """Import explicitly configured registry extensions before validation.

    The import is intentionally local so the schema module does not create an
    import cycle with component modules that import configuration dataclasses.
    ``RegistryCatalog.load_modules`` owns idempotency across config loads.
    """

    from stochaflow.utils.registry import REGISTRIES

    _validate_module_declarations(
        config.extensions.modules,
        path="config.extensions.modules",
    )
    REGISTRIES.load_modules(config.extensions.modules)


def load_config(path: str | Path) -> StochaflowConfig:
    """Load and validate a YAML config file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a top-level mapping")

    config = _coerce_dataclass(StochaflowConfig, raw, "config")
    _load_configured_modules(config)
    config.validate()
    return config


def load_config_dict(raw: dict[str, Any]) -> StochaflowConfig:
    """Load and validate a configuration from a plain dictionary."""

    config = _coerce_dataclass(StochaflowConfig, raw, "config")
    _load_configured_modules(config)
    config.validate()
    return config

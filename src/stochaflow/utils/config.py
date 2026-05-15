"""Centralized configuration schema and loading utilities."""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when a config file does not match the expected schema."""


@dataclass(slots=True)
class ComponentConfig:
    """Reusable component declaration with a registry name and free-form params."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentConfig:
    """Metadata for an experiment run."""

    name: str
    seed: int = 42
    output_dir: str = "outputs/default"
    exp_id: str | None = None


@dataclass(slots=True)
class DataloaderConfig:
    """Generic dataloader configuration."""

    batch_size: int = 128
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int | None = None


@dataclass(slots=True)
class DataSplitConfig:
    """Dataset split policy for train/validation/test experiments."""

    validation_size: int | float = 10000
    test_split: str = "test"


@dataclass(slots=True)
class DataConfig:
    """Dataset declaration and dataloader policy."""

    dataset: ComponentConfig
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    splits: DataSplitConfig = field(default_factory=DataSplitConfig)


@dataclass(slots=True)
class DiffusionConfig:
    """Diffusion process selection and scheduler declaration."""

    name: str
    scheduler: ComponentConfig
    params: dict[str, Any] = field(default_factory=dict)


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
    """Sampling/checkpoint artifact cadence."""

    sample_every: int = 1
    checkpoint_every: int = 1


@dataclass(slots=True)
class StochaflowConfig:
    """Top-level project configuration."""

    experiment: ExperimentConfig
    data: DataConfig
    model: ComponentConfig
    diffusion: DiffusionConfig
    objective: ComponentConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)

    def validate(self) -> None:
        """Validate cross-field invariants."""

        if self.data.dataloader.batch_size <= 0:
            raise ConfigError("data.dataloader.batch_size must be positive")
        if self.data.dataloader.num_workers < 0:
            raise ConfigError("data.dataloader.num_workers must be non-negative")
        if (
            self.data.dataloader.persistent_workers
            and self.data.dataloader.num_workers == 0
        ):
            raise ConfigError(
                "data.dataloader.persistent_workers requires num_workers > 0"
            )
        if (
            self.data.dataloader.prefetch_factor is not None
            and self.data.dataloader.prefetch_factor <= 0
        ):
            raise ConfigError(
                "data.dataloader.prefetch_factor must be positive when provided"
            )
        if (
            self.data.dataloader.prefetch_factor is not None
            and self.data.dataloader.num_workers == 0
        ):
            raise ConfigError(
                "data.dataloader.prefetch_factor requires num_workers > 0"
            )
        validation_size = self.data.splits.validation_size
        if isinstance(validation_size, float):
            if not 0.0 < validation_size < 1.0:
                raise ConfigError(
                    "data.splits.validation_size must be between 0 and 1 when a float"
                )
        elif isinstance(validation_size, int):
            if validation_size <= 0:
                raise ConfigError("data.splits.validation_size must be positive")
        else:
            raise ConfigError("data.splits.validation_size must be an int or float")
        if self.trainer.num_epochs <= 0:
            raise ConfigError("trainer.num_epochs must be positive")
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
        if self.artifacts.sample_every <= 0:
            raise ConfigError("artifacts.sample_every must be positive")
        if self.artifacts.checkpoint_every <= 0:
            raise ConfigError("artifacts.checkpoint_every must be positive")
        if "num_timesteps" not in self.diffusion.scheduler.params:
            raise ConfigError("diffusion.scheduler.params must include num_timesteps")

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


def _coerce_value(annotation: Any, value: Any, path: str) -> Any:
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


def load_config(path: str | Path) -> StochaflowConfig:
    """Load and validate a YAML config file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a top-level mapping")

    config = _coerce_dataclass(StochaflowConfig, raw, "config")
    config.validate()
    return config

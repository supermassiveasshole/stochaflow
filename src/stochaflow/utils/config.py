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

    mode: str = "none"
    train_split: str = "train"
    validation_split: str | None = None
    validation_size: int | float | None = None
    test_split: str | None = None
    train_splits: list[str] | None = None
    num_folds: int | None = None
    fold_index: int | None = None


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
    data: DataConfig
    model: ComponentConfig
    diffusion: DiffusionConfig
    objective: ComponentConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    diagnostics: list[ComponentConfig] = field(default_factory=list)
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
        split_mode = self.data.splits.mode
        valid_split_modes = {"random_holdout", "official", "all", "none", "kfold"}
        if split_mode not in valid_split_modes:
            raise ConfigError(
                "data.splits.mode must be one of: "
                + ", ".join(sorted(valid_split_modes))
            )
        validation_size = self.data.splits.validation_size
        if validation_size is not None:
            if isinstance(validation_size, float):
                if not 0.0 < validation_size < 1.0:
                    raise ConfigError(
                        "data.splits.validation_size must be between 0 and 1 "
                        "when a float"
                    )
            elif isinstance(validation_size, int):
                if validation_size <= 0:
                    raise ConfigError("data.splits.validation_size must be positive")
            else:
                raise ConfigError(
                    "data.splits.validation_size must be an int, float, or null"
                )
        if split_mode == "random_holdout" and validation_size is None:
            raise ConfigError(
                "data.splits.validation_size is required for random_holdout"
            )
        if split_mode == "all" and not self.data.splits.train_splits:
            raise ConfigError("data.splits.train_splits is required for all mode")
        if split_mode == "kfold":
            if self.data.splits.num_folds is None or self.data.splits.num_folds < 2:
                raise ConfigError("data.splits.num_folds must be at least 2 for kfold")
            fold_index = self.data.splits.fold_index
            if fold_index is not None and not 0 <= fold_index < self.data.splits.num_folds:
                raise ConfigError(
                    "data.splits.fold_index must be in [0, num_folds) when provided"
                )
        if self.trainer.num_epochs <= 0:
            raise ConfigError("trainer.num_epochs must be positive")
        if not 0.0 <= self.ema.decay < 1.0:
            raise ConfigError("ema.decay must satisfy 0 <= decay < 1")
        if self.ema.update_after_step < 0:
            raise ConfigError("ema.update_after_step must be non-negative")
        if self.ema.update_every <= 0:
            raise ConfigError("ema.update_every must be positive")
        if self.lr_scheduler.name is not None:
            if not isinstance(self.lr_scheduler.name, str) or not self.lr_scheduler.name:
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


def load_config_dict(raw: dict[str, Any]) -> StochaflowConfig:
    """Load and validate a configuration from a plain dictionary."""

    config = _coerce_dataclass(StochaflowConfig, raw, "config")
    config.validate()
    return config

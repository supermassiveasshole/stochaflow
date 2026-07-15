"""Centralized configuration schema and loading utilities."""

from copy import deepcopy
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
    """Global split policy applied after configured datasets are combined."""

    mode: str = "none"
    validation_size: int | float | None = None
    num_folds: int | None = None
    fold_index: int | None = None


@dataclass(slots=True)
class DatasetSplitMapConfig:
    """Map logical experiment partitions to a dataset's native split names."""

    train: str = "train"
    validation: str | None = None
    test: str | None = None


@dataclass(slots=True)
class DatasetConfig:
    """One registered dataset factory participating in an experiment."""

    id: str
    factory: str
    params: dict[str, Any] = field(default_factory=dict)
    splits: DatasetSplitMapConfig = field(default_factory=DatasetSplitMapConfig)
    sampling_weight: float | None = None


@dataclass(slots=True)
class ImageDataConfig:
    """Image tensor contract shared by every configured dataset."""

    channels: int = 3
    normalize: bool = True


@dataclass(slots=True)
class ResolutionBucketConfig:
    """A named spatial resolution accepted by one training batch."""

    name: str
    height: int
    width: int


@dataclass(slots=True)
class DataBatchingConfig:
    """Resolution bucketing and epoch-length policy."""

    buckets: list[ResolutionBucketConfig]
    sample_bucket: str
    dynamic_batch_size: bool = True
    steps_per_epoch: int | str = "auto"


@dataclass(slots=True)
class DataConfig:
    """Registered dataset mixture and dataloader policy."""

    datasets: list[DatasetConfig]
    image: ImageDataConfig
    batching: DataBatchingConfig
    modules: list[str] = field(default_factory=list)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    splits: DataSplitConfig = field(default_factory=DataSplitConfig)


@dataclass(slots=True)
class DiffusionConfig:
    """Diffusion process selection and forward noise-path declaration."""

    name: str
    noise_schedule: ComponentConfig
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrajectoryDebugConfig:
    """Optional sampler-specific reverse-trajectory diagnostics."""

    enabled: bool = False
    params: dict[str, Any] = field(default_factory=dict)
    gif_fps: int = 8


@dataclass(slots=True)
class SamplingDebugConfig:
    """Debug-only sampling artifact controls."""

    trajectory: TrajectoryDebugConfig = field(default_factory=TrajectoryDebugConfig)


@dataclass(slots=True)
class SamplingConfig:
    """Standalone and post-training sampling configuration."""

    sampler: ComponentConfig | None = None
    num_samples: int = 16
    batch_size: int | None = None
    seed: int | None = None
    grid_nrow: int = 4
    debug: SamplingDebugConfig = field(default_factory=SamplingDebugConfig)


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
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    diagnostics: list[ComponentConfig] = field(default_factory=list)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)

    def validate(self) -> None:
        """Validate cross-field invariants."""

        if not self.data.datasets:
            raise ConfigError("data.datasets must declare at least one dataset")
        for index, module in enumerate(self.data.modules):
            if not isinstance(module, str) or not module:
                raise ConfigError(
                    f"data.modules[{index}] must be a non-empty string"
                )
        source_ids: set[str] = set()
        weights: list[float | None] = []
        for index, dataset in enumerate(self.data.datasets):
            path = f"data.datasets[{index}]"
            if not isinstance(dataset.id, str) or not dataset.id:
                raise ConfigError(f"{path}.id must be a non-empty string")
            if dataset.id in source_ids:
                raise ConfigError(f"duplicate data source id '{dataset.id}'")
            source_ids.add(dataset.id)
            if not isinstance(dataset.factory, str) or not dataset.factory:
                raise ConfigError(f"{path}.factory must be a non-empty string")
            if not dataset.splits.train:
                raise ConfigError(f"{path}.splits.train must be non-empty")
            for split_name in (dataset.splits.validation, dataset.splits.test):
                if split_name is not None and not split_name:
                    raise ConfigError(f"{path} split names must be non-empty or null")
            weight = dataset.sampling_weight
            if weight is not None:
                if not isinstance(weight, (int, float)) or float(weight) <= 0:
                    raise ConfigError(f"{path}.sampling_weight must be positive")
                weight = float(weight)
            weights.append(weight)
        if any(weight is None for weight in weights) and any(
            weight is not None for weight in weights
        ):
            raise ConfigError(
                "data.datasets sampling_weight must be specified for every source "
                "or omitted for every source"
            )
        has_test = [dataset.splits.test is not None for dataset in self.data.datasets]
        if any(has_test) and not all(has_test):
            raise ConfigError(
                "every dataset must declare a test split or every dataset must "
                "omit it"
            )
        if self.data.image.channels <= 0:
            raise ConfigError("data.image.channels must be positive")
        if not self.data.batching.buckets:
            raise ConfigError("data.batching.buckets must not be empty")
        bucket_names: set[str] = set()
        for index, bucket in enumerate(self.data.batching.buckets):
            path = f"data.batching.buckets[{index}]"
            if not isinstance(bucket.name, str) or not bucket.name:
                raise ConfigError(f"{path}.name must be a non-empty string")
            if bucket.name in bucket_names:
                raise ConfigError(f"duplicate resolution bucket '{bucket.name}'")
            bucket_names.add(bucket.name)
            if bucket.height <= 0 or bucket.width <= 0:
                raise ConfigError(f"{path} height and width must be positive")
        if self.data.batching.sample_bucket not in bucket_names:
            raise ConfigError(
                "data.batching.sample_bucket must name a configured bucket"
            )
        steps_per_epoch = self.data.batching.steps_per_epoch
        if steps_per_epoch != "auto" and (
            not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0
        ):
            raise ConfigError(
                "data.batching.steps_per_epoch must be a positive integer or 'auto'"
            )
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
        valid_split_modes = {"random_holdout", "official", "none", "kfold"}
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
        if split_mode == "kfold":
            if self.data.splits.num_folds is None or self.data.splits.num_folds < 2:
                raise ConfigError("data.splits.num_folds must be at least 2 for kfold")
            fold_index = self.data.splits.fold_index
            if fold_index is not None and not 0 <= fold_index < self.data.splits.num_folds:
                raise ConfigError(
                    "data.splits.fold_index must be in [0, num_folds) when provided"
                )
        if split_mode == "official":
            has_validation = [
                dataset.splits.validation is not None
                for dataset in self.data.datasets
            ]
            if any(has_validation) and not all(has_validation):
                raise ConfigError(
                    "official mode requires every dataset to declare a validation "
                    "split or every dataset to omit it"
                )
        if self.model.name == "unet":
            in_channels = self.model.params.get("in_channels")
            out_channels = self.model.params.get("out_channels")
            if in_channels != self.data.image.channels:
                raise ConfigError(
                    "model.params.in_channels must match data.image.channels"
                )
            if out_channels != self.data.image.channels:
                raise ConfigError(
                    "model.params.out_channels must match data.image.channels"
                )
            multipliers = self.model.params.get("channel_multipliers", [1])
            if not isinstance(multipliers, (list, tuple)) or not multipliers:
                raise ConfigError(
                    "model.params.channel_multipliers must be a non-empty sequence"
                )
            divisor = 2 ** (len(multipliers) - 1)
            for bucket in self.data.batching.buckets:
                if bucket.height % divisor != 0 or bucket.width % divisor != 0:
                    raise ConfigError(
                        f"resolution bucket '{bucket.name}' dimensions must be "
                        f"divisible by {divisor} for the configured UNet"
                    )
        if self.trainer.num_epochs <= 0:
            raise ConfigError("trainer.num_epochs must be positive")
        if not 0.0 <= self.ema.decay < 1.0:
            raise ConfigError("ema.decay must satisfy 0 <= decay < 1")
        if self.ema.update_after_step < 0:
            raise ConfigError("ema.update_after_step must be non-negative")
        if self.ema.update_every <= 0:
            raise ConfigError("ema.update_every must be positive")
        if self.sampling.sampler is not None and not self.sampling.sampler.name:
            raise ConfigError("sampling.sampler.name must be a non-empty string")
        if self.sampling.num_samples <= 0:
            raise ConfigError("sampling.num_samples must be positive")
        if self.sampling.batch_size is not None and self.sampling.batch_size <= 0:
            raise ConfigError("sampling.batch_size must be positive when provided")
        if self.sampling.grid_nrow <= 0:
            raise ConfigError("sampling.grid_nrow must be positive")
        if self.sampling.debug.trajectory.gif_fps <= 0:
            raise ConfigError("sampling.debug.trajectory.gif_fps must be positive")
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
        if "num_timesteps" not in self.diffusion.noise_schedule.params:
            raise ConfigError(
                "diffusion.noise_schedule.params must include num_timesteps"
            )

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


def _migrate_legacy_noise_schedule_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize pre-refactor diffusion scheduler declarations.

    Older checkpoints and YAML files used ``diffusion.scheduler`` with names
    that combined the noise-path parameterization and DDPM algorithm. New
    configurations use ``diffusion.noise_schedule`` and parameterization-native
    names. The input mapping is copied so callers do not observe mutation.
    """

    migrated = deepcopy(raw)
    diffusion = migrated.get("diffusion")
    if not isinstance(diffusion, dict) or "scheduler" not in diffusion:
        return migrated
    if "noise_schedule" in diffusion:
        raise ConfigError(
            "config.diffusion must not define both scheduler and noise_schedule"
        )

    schedule = diffusion.pop("scheduler")
    if isinstance(schedule, dict):
        legacy_names = {
            "linear_ddpm": "linear_beta",
            "cosine_ddpm": "cosine_alpha_bar",
        }
        name = schedule.get("name")
        if name in legacy_names:
            schedule["name"] = legacy_names[name]
    diffusion["noise_schedule"] = schedule
    return migrated


def _reject_legacy_data_config(raw: dict[str, Any]) -> None:
    """Fail clearly for the removed single-dataset data schema."""

    data = raw.get("data")
    if not isinstance(data, dict):
        return
    legacy_fields = sorted({"dataset", "source"}.intersection(data))
    if legacy_fields:
        rendered = ", ".join(f"data.{field}" for field in legacy_fields)
        raise ConfigError(
            f"legacy data config field(s) are no longer supported: {rendered}; "
            "declare data.datasets, data.image, and data.batching instead"
        )


def load_config(path: str | Path) -> StochaflowConfig:
    """Load and validate a YAML config file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a top-level mapping")

    _reject_legacy_data_config(raw)
    raw = _migrate_legacy_noise_schedule_config(raw)
    config = _coerce_dataclass(StochaflowConfig, raw, "config")
    config.validate()
    return config


def load_config_dict(raw: dict[str, Any]) -> StochaflowConfig:
    """Load and validate a configuration from a plain dictionary."""

    _reject_legacy_data_config(raw)
    raw = _migrate_legacy_noise_schedule_config(raw)
    config = _coerce_dataclass(StochaflowConfig, raw, "config")
    config.validate()
    return config

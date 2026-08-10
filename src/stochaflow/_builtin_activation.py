"""Deterministic process-wide activation for Stochaflow built-ins."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from threading import RLock, get_ident
from types import ModuleType

from stochaflow.utils.registry import REGISTRIES

_DATA_BUILTIN_MODULES = ("stochaflow.data",)
_METRIC_BUILTIN_MODULES = (
    "stochaflow.metrics.builtin",
    "stochaflow.metrics.reference",
)
_MODEL_BUILTIN_MODULES = ("stochaflow.models",)
_PROCESS_BUILTIN_MODULES = ("stochaflow.processes",)
_SAMPLING_BUILTIN_MODULES = ("stochaflow.sampling",)
_TRAINING_BUILTIN_MODULES = ("stochaflow.training",)
_DIAGNOSTIC_BUILTIN_MODULES = ("stochaflow.training.diagnostics",)
_LOGGER_BUILTIN_MODULES = ("stochaflow.utils.logging",)

TRAINING_BUILTIN_MODULES = (
    *_DATA_BUILTIN_MODULES,
    *_METRIC_BUILTIN_MODULES,
    *_MODEL_BUILTIN_MODULES,
    *_PROCESS_BUILTIN_MODULES,
    *_SAMPLING_BUILTIN_MODULES,
    *_TRAINING_BUILTIN_MODULES,
    *_DIAGNOSTIC_BUILTIN_MODULES,
    *_LOGGER_BUILTIN_MODULES,
)
SAMPLING_BUILTIN_MODULES = (
    *_MODEL_BUILTIN_MODULES,
    *_PROCESS_BUILTIN_MODULES,
    *_SAMPLING_BUILTIN_MODULES,
)
EVALUATION_BUILTIN_MODULES = (
    *_DATA_BUILTIN_MODULES,
    *_METRIC_BUILTIN_MODULES,
    *_MODEL_BUILTIN_MODULES,
    *_PROCESS_BUILTIN_MODULES,
    *_SAMPLING_BUILTIN_MODULES,
)
ALL_BUILTIN_MODULES = TRAINING_BUILTIN_MODULES


@dataclass(slots=True)
class BuiltinActivationRuntime:
    """Mutable bookkeeping protected by the process activation lock."""

    completed_modules: set[str] = field(default_factory=set)
    failure: BaseException | None = None
    owner_thread_id: int | None = None


_activation_lock = RLock()
_activation_runtime = BuiltinActivationRuntime()


class BuiltinActivationError(RuntimeError):
    """Report terminal failure from the built-in activation lifecycle."""


@dataclass(frozen=True, slots=True)
class BuiltinRegistration:
    """One framework-owned name registered after its module is imported."""

    registry_name: str
    component_name: str
    attribute_name: str


_REGISTRATIONS_BY_MODULE: dict[str, tuple[BuiltinRegistration, ...]] = {
    "stochaflow.data": (
        BuiltinRegistration("data_builders", "image", "ImageDataBuilder"),
        BuiltinRegistration(
            "data_builders",
            "class_labeled_image",
            "ClassLabeledImageDataBuilder",
        ),
        BuiltinRegistration(
            "data_builders",
            "super_resolution",
            "SuperResolutionDataBuilder",
        ),
        BuiltinRegistration(
            "data_builders",
            "multi_resolution_image",
            "MultiResolutionImageDataBuilder",
        ),
    ),
    "stochaflow.metrics.builtin": (
        BuiltinRegistration("metrics", "mean", "ErrorOnNanMeanMetric"),
        BuiltinRegistration("metrics", "mse", "SingleOutputMeanSquaredError"),
        BuiltinRegistration("metrics", "mae", "SingleOutputMeanAbsoluteError"),
    ),
    "stochaflow.metrics.reference": (
        BuiltinRegistration("metrics", "fid", "FrechetInceptionDistanceMetric"),
        BuiltinRegistration("metrics", "kid", "KernelInceptionDistanceMetric"),
    ),
    "stochaflow.models": (
        BuiltinRegistration("models", "adm_unet", "ADMUNet"),
        BuiltinRegistration("models", "dit", "DiT"),
        BuiltinRegistration("models", "unet", "UNet"),
    ),
    "stochaflow.processes": (
        BuiltinRegistration(
            "noise_schedules",
            "cosine_alpha_bar",
            "CosineAlphaBarSchedule",
        ),
        BuiltinRegistration(
            "noise_schedules",
            "linear_beta",
            "LinearBetaSchedule",
        ),
        BuiltinRegistration(
            "processes",
            "discrete_gaussian",
            "DiscreteGaussianProcess",
        ),
    ),
    "stochaflow.sampling": (
        BuiltinRegistration(
            "sampling_builders",
            "standard_denoising",
            "StandardDenoisingBuilder",
        ),
        BuiltinRegistration(
            "sampling_builders",
            "class_conditional_denoising",
            "ClassConditionalDenoisingBuilder",
        ),
        BuiltinRegistration("samplers", "ddim", "DDIMSampler"),
        BuiltinRegistration("samplers", "ddpm", "DDPMAncestralSampler"),
        BuiltinRegistration(
            "sampling_artifact_writers",
            "tensor",
            "TensorSamplingArtifactWriter",
        ),
        BuiltinRegistration(
            "sampling_artifact_writers",
            "image",
            "ImageSamplingArtifactWriter",
        ),
    ),
    "stochaflow.training": (
        BuiltinRegistration(
            "training_builders",
            "supervised",
            "SupervisedTrainingBuilder",
        ),
        BuiltinRegistration(
            "training_builders",
            "gaussian_denoising",
            "GaussianDenoisingTrainingBuilder",
        ),
        BuiltinRegistration(
            "training_builders",
            "class_conditional_gaussian_denoising",
            "ClassConditionalGaussianDenoisingTrainingBuilder",
        ),
        BuiltinRegistration("objectives", "mse", "MSEObjective"),
        BuiltinRegistration("lr_schedulers", "warmup_cosine", "WarmupCosineLR"),
    ),
    "stochaflow.training.diagnostics": (
        BuiltinRegistration(
            "diagnostics",
            "diffusion_quality",
            "DiffusionQualityDiagnostic",
        ),
        BuiltinRegistration(
            "diagnostics",
            "class_conditional_diffusion_quality",
            "ClassConditionalDiffusionQualityDiagnostic",
        ),
    ),
    "stochaflow.utils.logging": (
        BuiltinRegistration("loggers", "local", "LocalLogger"),
        BuiltinRegistration("loggers", "tensorboard", "TensorBoardLogger"),
        BuiltinRegistration("loggers", "wandb", "WandbLogger"),
    ),
}

_DIAGNOSTIC_PROVIDER_REGISTRATIONS = (
    ("step_metrics", "timestep_bucket_loss", "TimestepBucketLossProvider"),
    ("step_metrics", "noise_alignment", "NoiseAlignmentProvider"),
    ("step_metrics", "x0_reconstruction", "X0ReconstructionMetricProvider"),
    ("sampler_metrics", "sample_statistics", "SampleStatisticsProvider"),
    ("sampler_metrics", "sampling_performance", "SamplingPerformanceProvider"),
    (
        "denoiser_artifacts",
        "reconstruction_panel",
        "ReconstructionPanelProvider",
    ),
    ("sampler_artifacts", "sample_grid", "SampleGridProvider"),
    ("sampler_artifacts", "trajectory", "TrajectoryArtifactProvider"),
    ("reference_metrics", "fid", "FIDReferenceMetricProvider"),
    ("reference_metrics", "kid", "KIDReferenceMetricProvider"),
)


def _activation_error(*, detail: str) -> BuiltinActivationError:
    return BuiltinActivationError(
        f"built-in component activation {detail}; restart the Python process "
        "before using Stochaflow again"
    )


def _raise_poisoned() -> None:
    assert _activation_runtime.failure is not None
    raise _activation_error(detail="previously failed") from _activation_runtime.failure


def _is_activation_failure_wrapper(
    error: BaseException,
    target: BaseException,
) -> bool:
    return type(error) is BuiltinActivationError and error.__cause__ is target


def _register_module_components(module_name: str, module: ModuleType) -> None:
    for registration in _REGISTRATIONS_BY_MODULE.get(module_name, ()):
        registry = getattr(REGISTRIES, registration.registry_name)
        registry.add(
            registration.component_name,
            getattr(module, registration.attribute_name),
        )
    if module_name == "stochaflow.data":
        for component_name, source_module_name, attribute_name in (
            (
                "torchvision",
                "stochaflow.data.torchvision_source",
                "TorchvisionImageDataSource",
            ),
            (
                "image_folder",
                "stochaflow.data.folder_sources",
                "ImageFolderDataSource",
            ),
            (
                "paired_image_folders",
                "stochaflow.data.folder_sources",
                "PairedImageFolderDataSource",
            ),
        ):
            source_module = import_module(source_module_name)
            module.IMAGE_DATA_SOURCES.add(
                component_name,
                getattr(source_module, attribute_name),
            )
    if module_name != "stochaflow.training.diagnostics":
        return
    providers = module.providers
    provider_catalog = module.DIAGNOSTIC_PROVIDERS
    for registry_name, component_name, attribute_name in (
        _DIAGNOSTIC_PROVIDER_REGISTRATIONS
    ):
        provider_catalog.registry(registry_name).add(
            component_name,
            getattr(providers, attribute_name),
        )


def _activate_modules(module_names: tuple[str, ...]) -> None:
    with _activation_lock:
        if _activation_runtime.failure is not None:
            _raise_poisoned()
        current_thread_id = get_ident()
        if _activation_runtime.owner_thread_id == current_thread_id:
            reentry = RuntimeError(
                "built-in component activation re-entered while importing a "
                "built-in module"
            )
            _activation_runtime.failure = reentry
            raise _activation_error(detail="re-entered") from reentry

        _activation_runtime.owner_thread_id = current_thread_id
        try:
            for module_name in module_names:
                if module_name in _activation_runtime.completed_modules:
                    continue
                try:
                    module = import_module(module_name)
                    _register_module_components(module_name, module)
                except BaseException as exc:
                    if _activation_runtime.failure is not None:
                        if _is_activation_failure_wrapper(
                            exc,
                            _activation_runtime.failure,
                        ):
                            raise
                        _raise_poisoned()
                    _activation_runtime.failure = exc
                    raise _activation_error(
                        detail=f"failed while importing {module_name!r}"
                    ) from _activation_runtime.failure
                if _activation_runtime.failure is not None:
                    _raise_poisoned()
                _activation_runtime.completed_modules.add(module_name)
        finally:
            _activation_runtime.owner_thread_id = None


def _require_modules(module_names: tuple[str, ...], *, operation: str) -> None:
    with _activation_lock:
        if _activation_runtime.failure is not None:
            _raise_poisoned()
        missing = tuple(
            module_name
            for module_name in module_names
            if module_name not in _activation_runtime.completed_modules
        )
    if missing:
        raise RuntimeError(
            f"{operation} built-ins were not activated before resolved execution: "
            + ", ".join(missing)
        )


def activate_training_builtins() -> None:
    """Activate the deterministic built-in scope required by training."""

    _activate_modules(TRAINING_BUILTIN_MODULES)


def activate_data_builtins() -> None:
    """Activate only built-in runtime data composition."""

    _activate_modules(_DATA_BUILTIN_MODULES)


def activate_metric_builtins() -> None:
    """Activate only built-in Metric providers."""

    _activate_modules(_METRIC_BUILTIN_MODULES)


def activate_model_builtins() -> None:
    """Activate only built-in Model providers."""

    _activate_modules(_MODEL_BUILTIN_MODULES)


def activate_process_builtins() -> None:
    """Activate only built-in Process and schedule providers."""

    _activate_modules(_PROCESS_BUILTIN_MODULES)


def activate_sampling_component_builtins() -> None:
    """Activate only built-in Sampling components."""

    _activate_modules(_SAMPLING_BUILTIN_MODULES)


def activate_training_component_builtins() -> None:
    """Activate only built-in Training components."""

    _activate_modules(_TRAINING_BUILTIN_MODULES)


def activate_diagnostic_builtins() -> None:
    """Activate built-in Diagnostic types and their provider catalog."""

    _activate_modules(_DIAGNOSTIC_BUILTIN_MODULES)


def activate_sampling_builtins() -> None:
    """Activate the deterministic built-in scope required by sampling."""

    _activate_modules(SAMPLING_BUILTIN_MODULES)


def activate_evaluation_builtins() -> None:
    """Activate the deterministic built-in scope required by evaluation."""

    _activate_modules(EVALUATION_BUILTIN_MODULES)


def activate_all_builtins() -> None:
    """Activate every Stochaflow-owned built-in component."""

    _activate_modules(ALL_BUILTIN_MODULES)


def require_training_builtins() -> None:
    """Confirm that training activation completed without importing anything."""

    _require_modules(TRAINING_BUILTIN_MODULES, operation="training")


def require_sampling_builtins() -> None:
    """Confirm that sampling activation completed without importing anything."""

    _require_modules(SAMPLING_BUILTIN_MODULES, operation="sampling")


def require_evaluation_builtins() -> None:
    """Confirm that evaluation activation completed without importing anything."""

    _require_modules(EVALUATION_BUILTIN_MODULES, operation="evaluation")

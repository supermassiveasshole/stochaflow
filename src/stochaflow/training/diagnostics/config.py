"""Strict typed configuration for diffusion quality diagnostics."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

_PROFILE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PROVIDER_CATEGORIES = (
    "step_metrics",
    "sampler_metrics",
    "denoiser_artifacts",
    "sampler_artifacts",
)


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """One named diagnostic provider and its constructor parameters."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DiagnosticCadenceConfig:
    """Step and epoch cadence for the diagnostic pipeline."""

    step_every: int = 100
    artifact_every_epochs: int = 5


@dataclass(frozen=True, slots=True)
class DiagnosticSamplingConfig:
    """Shared fixed-noise sampling controls."""

    shape: tuple[int, int, int]
    sample_num: int = 16
    batch_size: int = 16
    seed: int = 123


@dataclass(frozen=True, slots=True)
class TrajectoryProviderConfig:
    """Trajectory settings for one sampler profile."""

    enabled: bool = False
    every_steps: int = 1
    gif_fps: int = 8


@dataclass(frozen=True, slots=True)
class SamplerProfileConfig:
    """One named sampler configuration evaluated by the pipeline."""

    id: str
    name: str
    params: dict[str, Any]
    trajectory: TrajectoryProviderConfig


@dataclass(frozen=True, slots=True)
class ProviderPipelineConfig:
    """Provider selections for each non-reference pipeline phase."""

    step_metrics: tuple[ProviderSpec, ...]
    sampler_metrics: tuple[ProviderSpec, ...]
    denoiser_artifacts: tuple[ProviderSpec, ...]
    sampler_artifacts: tuple[ProviderSpec, ...]


@dataclass(frozen=True, slots=True)
class ReferencePipelineConfig:
    """Reference-distribution evaluation settings."""

    enabled: bool = False
    every_epochs: int = 20
    num_real: int = 2048
    num_fake: int = 2048
    batch_size: int = 64
    metrics: tuple[ProviderSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class DiffusionQualityConfig:
    """Fully parsed configuration consumed by the orchestrator."""

    modules: tuple[str, ...]
    cadence: DiagnosticCadenceConfig
    sampling: DiagnosticSamplingConfig
    samplers: tuple[SamplerProfileConfig, ...]
    providers: ProviderPipelineConfig
    reference: ReferencePipelineConfig
    use_ema: bool
    failure_policy: str


def _positive_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _check_fields(
    value: Mapping[str, Any],
    allowed: set[str],
    *,
    path: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {path} field(s): {', '.join(unknown)}")


def _default_provider_specs() -> dict[str, tuple[ProviderSpec, ...]]:
    timesteps = [50, 250, 500, 900]
    return {
        "step_metrics": (
            ProviderSpec("timestep_bucket_loss", {"buckets": 10}),
            ProviderSpec("noise_alignment"),
            ProviderSpec("x0_reconstruction", {"timesteps": timesteps.copy()}),
        ),
        "sampler_metrics": (
            ProviderSpec("sample_statistics"),
            ProviderSpec("sampling_performance"),
        ),
        "denoiser_artifacts": (
            ProviderSpec(
                "reconstruction_panel",
                {"timesteps": timesteps.copy(), "max_samples": 16},
            ),
        ),
        "sampler_artifacts": (
            ProviderSpec("sample_grid", {"nrow": 4}),
            ProviderSpec("trajectory"),
        ),
    }


def _parse_modules(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("diffusion_quality modules must be a sequence")
    modules: list[str] = []
    for index, module in enumerate(raw):
        if not isinstance(module, str) or not module.strip():
            raise ValueError(
                f"diffusion_quality modules[{index}] must be a non-empty string"
            )
        if module in modules:
            raise ValueError(
                f"duplicate diffusion_quality provider module '{module}'"
            )
        modules.append(module)
    return tuple(modules)


def _parse_cadence(raw: Any) -> DiagnosticCadenceConfig:
    value = _mapping(
        {} if raw is None else raw,
        path="diffusion_quality cadence",
    )
    _check_fields(
        value,
        {"step_every", "artifact_every_epochs"},
        path="diffusion_quality cadence",
    )
    return DiagnosticCadenceConfig(
        step_every=_positive_int(
            value.get("step_every", 100),
            path="diffusion_quality cadence.step_every",
        ),
        artifact_every_epochs=_positive_int(
            value.get("artifact_every_epochs", 5),
            path="diffusion_quality cadence.artifact_every_epochs",
        ),
    )


def _parse_sampling(raw: Any) -> DiagnosticSamplingConfig:
    value = _mapping(
        {} if raw is None else raw,
        path="diffusion_quality sampling",
    )
    _check_fields(
        value,
        {"shape", "sample_num", "batch_size", "seed"},
        path="diffusion_quality sampling",
    )
    shape = value.get("shape")
    if shape is None:
        raise ValueError(
            "diffusion_quality sampling.shape is required with [C, H, W]"
        )
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise TypeError(
            "diffusion_quality sampling.shape must be a sequence [C, H, W]"
        )
    if len(shape) != 3 or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in shape
    ):
        raise ValueError(
            "diffusion_quality sampling.shape must contain three positive "
            "integers [C, H, W]"
        )
    seed = value.get("seed", 123)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("diffusion_quality sampling.seed must be an integer")
    return DiagnosticSamplingConfig(
        shape=(shape[0], shape[1], shape[2]),
        sample_num=_positive_int(
            value.get("sample_num", 16),
            path="diffusion_quality sampling.sample_num",
        ),
        batch_size=_positive_int(
            value.get("batch_size", 16),
            path="diffusion_quality sampling.batch_size",
        ),
        seed=seed,
    )


def _parse_trajectory(raw: Any, *, path: str) -> TrajectoryProviderConfig:
    value = _mapping({} if raw is None else raw, path=path)
    _check_fields(value, {"enabled", "every_steps", "gif_fps"}, path=path)
    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError(f"{path}.enabled must be a boolean")
    return TrajectoryProviderConfig(
        enabled=enabled,
        every_steps=_positive_int(
            value.get("every_steps", 1), path=f"{path}.every_steps"
        ),
        gif_fps=_positive_int(value.get("gif_fps", 8), path=f"{path}.gif_fps"),
    )


def _parse_samplers(raw: Any) -> tuple[SamplerProfileConfig, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("diffusion_quality samplers must be a sequence")
    if not raw:
        raise ValueError("diffusion_quality samplers must not be empty")
    profiles: list[SamplerProfileConfig] = []
    ids: set[str] = set()
    for index, item in enumerate(raw):
        path = f"diffusion_quality samplers[{index}]"
        value = _mapping(item, path=path)
        _check_fields(value, {"id", "name", "params", "trajectory"}, path=path)
        profile_id = value.get("id")
        if not isinstance(profile_id, str) or not _PROFILE_ID.fullmatch(profile_id):
            raise ValueError(f"{path}.id must match {_PROFILE_ID.pattern!r}")
        if profile_id in ids:
            raise ValueError(f"duplicate diffusion_quality sampler id '{profile_id}'")
        ids.add(profile_id)
        name = value.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}.name must be a non-empty string")
        params = _mapping(value.get("params", {}), path=f"{path}.params")
        profiles.append(
            SamplerProfileConfig(
                id=profile_id,
                name=name,
                params=dict(params),
                trajectory=_parse_trajectory(
                    value.get("trajectory", {}),
                    path=f"{path}.trajectory",
                ),
            )
        )
    return tuple(profiles)


def _parse_provider_specs(raw: Any, *, path: str) -> tuple[ProviderSpec, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError(f"{path} must be a sequence")
    specs: list[ProviderSpec] = []
    names: set[str] = set()
    for index, item in enumerate(raw):
        item_path = f"{path}[{index}]"
        value = _mapping(item, path=item_path)
        _check_fields(value, {"name", "params"}, path=item_path)
        name = value.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{item_path}.name must be a non-empty string")
        if name in names:
            raise ValueError(f"duplicate provider '{name}' in {path}")
        names.add(name)
        params = _mapping(value.get("params", {}), path=f"{item_path}.params")
        specs.append(ProviderSpec(name=name, params=dict(params)))
    return tuple(specs)


def _parse_providers(raw: Any) -> ProviderPipelineConfig:
    defaults = _default_provider_specs()
    value = _mapping(
        {} if raw is None else raw,
        path="diffusion_quality providers",
    )
    _check_fields(value, set(PROVIDER_CATEGORIES), path="diffusion_quality providers")
    parsed = {
        category: (
            _parse_provider_specs(
                value[category],
                path=f"diffusion_quality providers.{category}",
            )
            if category in value
            else defaults[category]
        )
        for category in PROVIDER_CATEGORIES
    }
    return ProviderPipelineConfig(**parsed)


def _parse_reference(raw: Any) -> ReferencePipelineConfig:
    value = _mapping(
        {} if raw is None else raw,
        path="diffusion_quality reference",
    )
    _check_fields(
        value,
        {"enabled", "every_epochs", "num_real", "num_fake", "batch_size", "metrics"},
        path="diffusion_quality reference",
    )
    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("diffusion_quality reference.enabled must be a boolean")
    num_real = _positive_int(
        value.get("num_real", 2048),
        path="diffusion_quality reference.num_real",
    )
    num_fake = _positive_int(
        value.get("num_fake", 2048),
        path="diffusion_quality reference.num_fake",
    )
    if enabled and min(num_real, num_fake) < 2:
        raise ValueError(
            "diffusion_quality reference num_real and num_fake must each be at least 2"
        )
    default_metrics = (
        ProviderSpec(
            "kid",
            {
                "subsets": 100,
                "subset_size": min(1000, num_real, num_fake),
            },
        ),
        ProviderSpec("fid"),
    )
    metrics = (
        _parse_provider_specs(
            value["metrics"],
            path="diffusion_quality reference.metrics",
        )
        if "metrics" in value
        else default_metrics
    )
    if enabled and not metrics:
        raise ValueError(
            "diffusion_quality reference.metrics must not be empty when enabled"
        )
    return ReferencePipelineConfig(
        enabled=enabled,
        every_epochs=_positive_int(
            value.get("every_epochs", 20),
            path="diffusion_quality reference.every_epochs",
        ),
        num_real=num_real,
        num_fake=num_fake,
        batch_size=_positive_int(
            value.get("batch_size", 64),
            path="diffusion_quality reference.batch_size",
        ),
        metrics=metrics,
    )


def parse_diffusion_quality_config(
    *,
    modules: Sequence[str],
    cadence: Mapping[str, Any] | None,
    sampling: Mapping[str, Any] | None,
    samplers: Sequence[Mapping[str, Any]],
    providers: Mapping[str, Any] | None,
    reference: Mapping[str, Any] | None,
    use_ema: object,
    failure_policy: str,
) -> DiffusionQualityConfig:
    """Parse the public constructor arguments into an immutable configuration."""

    if not isinstance(use_ema, bool):
        raise TypeError("diffusion_quality use_ema must be a boolean")
    if failure_policy not in {"raise", "warn"}:
        raise ValueError("diffusion_quality failure_policy must be 'raise' or 'warn'")
    return DiffusionQualityConfig(
        modules=_parse_modules(modules),
        cadence=_parse_cadence(cadence),
        sampling=_parse_sampling(sampling),
        samplers=_parse_samplers(samplers),
        providers=_parse_providers(providers),
        reference=_parse_reference(reference),
        use_ema=use_ema,
        failure_policy=failure_policy,
    )


__all__ = [
    "PROVIDER_CATEGORIES",
    "DiagnosticCadenceConfig",
    "DiagnosticSamplingConfig",
    "DiffusionQualityConfig",
    "ProviderPipelineConfig",
    "ProviderSpec",
    "ReferencePipelineConfig",
    "SamplerProfileConfig",
    "TrajectoryProviderConfig",
    "parse_diffusion_quality_config",
]

"""Optional reference-distribution metric providers and shared evaluation suite."""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import torch

from stochaflow.training.diagnostics.config import ReferencePipelineConfig
from stochaflow.training.diagnostics.contracts import ReferenceMetricProvider
from stochaflow.training.diagnostics.runtime import (
    BoundSampler,
    SeedPolicy,
    prepare_reference_images,
)


def _quality_dependency_error(metric: str) -> RuntimeError:
    return RuntimeError(
        f"{metric} requires the optional 'quality' dependencies and available "
        "Inception weights"
    )


class FIDReferenceMetricProvider(ReferenceMetricProvider):
    """TorchMetrics Fréchet inception distance adapter."""

    def __init__(
        self,
        *,
        device: torch.device,
        num_real: int,
        num_fake: int,
        feature: int = 2048,
    ) -> None:
        del num_real, num_fake
        metric_device = (
            torch.device("cpu") if device.type == "mps" else torch.device(device)
        )
        try:
            module = importlib.import_module("torchmetrics.image.fid")
            metric_cls = module.FrechetInceptionDistance
        except (ImportError, ModuleNotFoundError, AttributeError) as exc:
            raise _quality_dependency_error("FID") from exc
        try:
            metric = metric_cls(
                feature=feature,
                normalize=True,
                reset_real_features=False,
                sync_on_compute=False,
            ).to(metric_device)
            metric.set_dtype(torch.float64)
        except (ImportError, ModuleNotFoundError) as exc:
            raise _quality_dependency_error("FID") from exc
        except RuntimeError as exc:
            raise RuntimeError(
                "failed to initialize FID reference metric on "
                f"{metric_device.type}: {exc}"
            ) from exc
        self._metric_device = metric_device
        self.metric = metric

    def update(self, images: torch.Tensor, *, real: bool) -> None:
        self.metric.update(images.to(self._metric_device), real=real)

    def compute(self) -> Mapping[str, float]:
        value = self.metric.compute()
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        return {"fid": float(value)}

    def reset_fake(self) -> None:
        self.metric.reset()


class KIDReferenceMetricProvider(ReferenceMetricProvider):
    """TorchMetrics kernel inception distance adapter."""

    def __init__(
        self,
        *,
        device: torch.device,
        num_real: int,
        num_fake: int,
        subsets: object = 100,
        subset_size: object = 1000,
    ) -> None:
        if isinstance(subsets, bool) or not isinstance(subsets, int) or subsets <= 0:
            raise ValueError("kid subsets must be a positive integer")
        if (
            isinstance(subset_size, bool)
            or not isinstance(subset_size, int)
            or subset_size <= 1
        ):
            raise ValueError("kid subset_size must be an integer greater than one")
        if subset_size > min(num_real, num_fake):
            raise ValueError(
                "kid subset_size must not exceed reference num_real or num_fake"
            )
        try:
            module = importlib.import_module("torchmetrics.image.kid")
            metric_cls = module.KernelInceptionDistance
            metric = metric_cls(
                normalize=True,
                reset_real_features=False,
                subsets=subsets,
                subset_size=subset_size,
                sync_on_compute=False,
            ).to(device)
        except (
            ImportError,
            ModuleNotFoundError,
            AttributeError,
            RuntimeError,
            ValueError,
        ) as exc:
            raise _quality_dependency_error("KID") from exc
        self.metric = metric

    def update(self, images: torch.Tensor, *, real: bool) -> None:
        self.metric.update(images, real=real)

    def compute(self) -> Mapping[str, float]:
        mean, std = self.metric.compute()
        return {"kid_mean": float(mean), "kid_std": float(std)}

    def reset_fake(self) -> None:
        self.metric.reset()


ReferenceErrorHandler = Callable[[str, str, Exception], None]


@dataclass(frozen=True, slots=True)
class ReferenceMetricResult:
    """Separate selection metrics from protocol/runtime observations."""

    metrics: Mapping[str, float]
    observations: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(dict(self.metrics)),
        )
        object.__setattr__(
            self,
            "observations",
            MappingProxyType(dict(self.observations)),
        )


class ReferenceMetricSuite:
    """Cache real features once and stream shared fake batches to providers."""

    def __init__(
        self,
        providers: Sequence[tuple[str, ReferenceMetricProvider]],
        config: ReferencePipelineConfig,
        *,
        device: torch.device,
        seed_policy: SeedPolicy,
        handle_error: ReferenceErrorHandler,
        extract_images: Callable[[Any], torch.Tensor],
    ) -> None:
        self.providers = tuple(providers)
        self.config = config
        self.device = device
        self.seed_policy = seed_policy
        self.handle_error = handle_error
        self.extract_images = extract_images
        self._unavailable: set[str] = set()

    def cache_real(self, dataloader: Iterable[Any]) -> Mapping[str, float]:
        """Build deterministic real-feature state before training starts."""

        seen = 0
        started_at = time.perf_counter()
        with torch.inference_mode(), self.seed_policy.fork_rng(self.device):
            for batch in dataloader:
                image_value = cast(object, self.extract_images(batch))
                if not isinstance(image_value, torch.Tensor):
                    raise TypeError(
                        "reference image extractor must return a Tensor"
                    )
                images = image_value
                remaining = self.config.num_real - seen
                if remaining <= 0:
                    break
                prepared = prepare_reference_images(images[:remaining]).to(self.device)
                for name, provider in self.providers:
                    if name in self._unavailable:
                        continue
                    try:
                        provider.update(prepared, real=True)
                    # Reference providers are extensions with no shared exception type.
                    except Exception as exc:  # noqa: BLE001
                        self._unavailable.add(name)
                        self.handle_error("reference_real_update", name, exc)
                seen += prepared.shape[0]
                if seen >= self.config.num_real or len(self._unavailable) == len(
                    self.providers
                ):
                    break
        if seen < self.config.num_real and len(self._unavailable) < len(self.providers):
            raise ValueError(
                "validation dataloader provided only "
                f"{seen} images; reference.num_real={self.config.num_real}"
            )
        return {
            "diagnostics/reference/real_samples": float(seen),
            "diagnostics/reference/cache_seconds": time.perf_counter() - started_at,
        }

    def evaluate(
        self,
        *,
        profile_id: str,
        sampler: BoundSampler,
        sample_shape: tuple[int, int, int],
        visual_samples: torch.Tensor | None,
    ) -> ReferenceMetricResult:
        """Evaluate every active provider from one shared generated image stream."""

        started_at = time.perf_counter()
        seen = 0
        failed = set(self._unavailable)

        def update_fake(images: torch.Tensor) -> None:
            prepared = prepare_reference_images(images).to(self.device)
            for name, provider in self.providers:
                if name in failed:
                    continue
                try:
                    provider.update(prepared, real=False)
                # Reference providers are extensions with no shared exception type.
                except Exception as exc:  # noqa: BLE001
                    failed.add(name)
                    self.handle_error("reference_update", name, exc)

        try:
            if visual_samples is not None:
                initial = visual_samples[: self.config.num_fake]
                update_fake(initial)
                seen += initial.shape[0]

            generator = torch.Generator(device="cpu").manual_seed(
                self.seed_policy.base_seed + 1_000_003
            )
            while seen < self.config.num_fake and len(failed) < len(self.providers):
                count = min(self.config.batch_size, self.config.num_fake - seen)
                noise = torch.randn(
                    (count, *sample_shape),
                    generator=generator,
                    device="cpu",
                ).to(self.device)
                generated = sampler.sampler.sample(
                    sampler.dynamics,
                    noise,
                ).final_state
                if not isinstance(generated, torch.Tensor):
                    raise TypeError(
                        f"sampler '{profile_id}' must return a Tensor final_state"
                    )
                if generated.shape != noise.shape:
                    raise ValueError(
                        f"sampler '{profile_id}' returned shape "
                        f"{tuple(generated.shape)}, expected {tuple(noise.shape)}"
                    )
                update_fake(generated)
                seen += generated.shape[0]

            prefix = f"diagnostics/samplers/{profile_id}"
            observations: dict[str, float] = {
                f"{prefix}/reference_fake_samples": float(seen),
            }
            metrics: dict[str, float] = {}
            for name, provider in self.providers:
                if name in failed:
                    continue
                try:
                    values = provider.compute()
                    for suffix, value in values.items():
                        tag = f"{prefix}/{suffix}"
                        if tag in metrics:
                            raise ValueError(f"reference metric tag collision: {tag}")
                        metrics[tag] = float(value)
                # Reference providers are extensions with no shared exception type.
                except Exception as exc:  # noqa: BLE001
                    failed.add(name)
                    self.handle_error("reference_compute", name, exc)
            observations[f"{prefix}/reference_metric_seconds"] = (
                time.perf_counter() - started_at
            )
            return ReferenceMetricResult(
                metrics=metrics,
                observations=observations,
            )
        finally:
            for name, provider in self.providers:
                if name in self._unavailable:
                    continue
                try:
                    provider.reset_fake()
                # Cleanup must isolate failures from each independent provider.
                except Exception as exc:  # noqa: BLE001
                    self.handle_error("reference_reset", name, exc)


__all__ = [
    "FIDReferenceMetricProvider",
    "KIDReferenceMetricProvider",
    "ReferenceMetricResult",
    "ReferenceMetricSuite",
]

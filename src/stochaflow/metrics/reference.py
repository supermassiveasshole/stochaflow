"""Reference-distribution image metrics backed by optional TorchMetrics extras."""

from __future__ import annotations

import math
from importlib import import_module
from numbers import Real
from typing import cast

import torch
from torchmetrics import Metric

from stochaflow.utils.registry import REGISTRIES

_INCEPTION_FEATURES = frozenset({64, 192, 768, 2048})


def _quality_dependency_error(metric_name: str) -> RuntimeError:
    return RuntimeError(
        f"{metric_name} requires the optional 'quality' dependencies and "
        "available Inception weights; run `uv sync --extra quality` or "
        "install `stochaflow[quality]`"
    )


def _load_metric_class(
    *,
    module_name: str,
    class_name: str,
    metric_name: str,
) -> type[Metric]:
    try:
        module = import_module(module_name)
        metric_class = getattr(module, class_name)
    except (ImportError, ModuleNotFoundError, AttributeError) as error:
        raise _quality_dependency_error(metric_name) from error
    if not isinstance(metric_class, type) or not issubclass(metric_class, Metric):
        raise TypeError(
            f"{module_name}.{class_name} must inherit torchmetrics.Metric"
        )
    return cast(type[Metric], metric_class)


def _inception_feature(value: object, *, metric_name: str) -> int:
    if type(value) is not int or value not in _INCEPTION_FEATURES:
        supported = ", ".join(str(feature) for feature in sorted(_INCEPTION_FEATURES))
        raise ValueError(
            f"{metric_name} feature must be one of {supported}"
        )
    return value


def _positive_integer(value: object, *, path: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        if minimum == 1:
            raise ValueError(f"{path} must be a positive integer")
        raise ValueError(f"{path} must be an integer of at least {minimum}")
    return value


def _random_seed(value: object, *, path: str) -> int:
    if type(value) is not int or not 0 <= value <= (2**63 - 1):
        raise ValueError(f"{path} must be an integer in [0, 2**63 - 1]")
    return value


def _finite_real(
    value: object,
    *,
    path: str,
    strictly_positive: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{path} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{path} must be finite")
    if strictly_positive and normalized <= 0.0:
        raise ValueError(f"{path} must be positive")
    return normalized


def _reference_images(images: object, *, metric_name: str) -> torch.Tensor:
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"{metric_name} images must be a Tensor")
    if images.ndim != 4:
        raise ValueError(
            f"{metric_name} images must have rank 4 (N, 3, H, W), "
            f"got rank {images.ndim}"
        )
    if images.shape[0] <= 0 or images.shape[1] != 3:
        raise ValueError(
            f"{metric_name} images must have shape (N, 3, H, W) with N > 0"
        )
    if images.shape[2] <= 0 or images.shape[3] <= 0:
        raise ValueError(f"{metric_name} image dimensions must be positive")
    if not images.is_floating_point():
        raise TypeError(
            f"{metric_name} images must use a floating-point dtype normalized "
            "to [0, 1]"
        )
    if not bool(torch.isfinite(images).all().item()):
        raise ValueError(f"{metric_name} images must contain only finite values")
    minimum, maximum = torch.aminmax(images.detach())
    if minimum.item() < 0.0 or maximum.item() > 1.0:
        raise ValueError(f"{metric_name} images must be normalized to [0, 1]")
    return images


def _scalar_tensor(value: object, *, metric_name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise TypeError(f"{metric_name} compute result must be a scalar Tensor")
    return value.reshape(())


class ReferenceImageDistributionMetric(Metric):
    """Delegate image-distribution state to one optional TorchMetrics metric."""

    metric: Metric
    metric_name: str

    def __init__(self, metric: Metric, *, metric_name: str) -> None:
        super().__init__(sync_on_compute=False)
        self.metric = metric
        self.metric_name = metric_name

    def update(self, images: torch.Tensor, *, real: bool) -> None:
        """Accumulate one normalized real or generated image batch."""

        if type(real) is not bool:
            raise TypeError(f"{self.metric_name} real must be a bool")
        prepared = _reference_images(images, metric_name=self.metric_name)
        self.metric.update(prepared, real=real)

    def reset(self) -> None:
        """Reset both the adapter guard and all delegated real/fake state."""

        try:
            self.metric.reset()
        finally:
            super().reset()


@REGISTRIES.metrics.register("fid")
class FrechetInceptionDistanceMetric(ReferenceImageDistributionMetric):
    """Compute scalar Frechet inception distance from real/fake image updates."""

    def __init__(self, *, feature: int = 2048, antialias: bool = True) -> None:
        feature = _inception_feature(feature, metric_name="FID")
        if type(antialias) is not bool:
            raise TypeError("FID antialias must be a bool")
        metric_class = _load_metric_class(
            module_name="torchmetrics.image.fid",
            class_name="FrechetInceptionDistance",
            metric_name="FID",
        )
        try:
            metric = metric_class(
                feature=feature,
                normalize=True,
                reset_real_features=True,
                antialias=antialias,
                sync_on_compute=False,
            )
        except (ImportError, ModuleNotFoundError) as error:
            raise _quality_dependency_error("FID") from error
        metric.set_dtype(torch.float64)
        super().__init__(metric, metric_name="FID")

    def compute(self) -> torch.Tensor:
        """Return the scalar distance emitted by TorchMetrics."""

        return _scalar_tensor(self.metric.compute(), metric_name="FID")


@REGISTRIES.metrics.register("kid")
class KernelInceptionDistanceMetric(ReferenceImageDistributionMetric):
    """Compute KID mean and standard deviation from real/fake image updates."""

    def __init__(
        self,
        *,
        feature: int = 2048,
        subsets: int = 100,
        subset_size: int = 1000,
        degree: int = 3,
        gamma: float | None = None,
        coef: float = 1.0,
        seed: int = 0,
    ) -> None:
        feature = _inception_feature(feature, metric_name="KID")
        subsets = _positive_integer(subsets, path="KID subsets")
        subset_size = _positive_integer(
            subset_size,
            path="KID subset_size",
            minimum=2,
        )
        degree = _positive_integer(degree, path="KID degree")
        normalized_gamma = (
            None
            if gamma is None
            else _finite_real(
                gamma,
                path="KID gamma",
                strictly_positive=True,
            )
        )
        normalized_coef = _finite_real(
            coef,
            path="KID coef",
            strictly_positive=True,
        )
        self.seed = _random_seed(seed, path="KID seed")
        metric_class = _load_metric_class(
            module_name="torchmetrics.image.kid",
            class_name="KernelInceptionDistance",
            metric_name="KID",
        )
        try:
            metric = metric_class(
                feature=feature,
                normalize=True,
                reset_real_features=True,
                subsets=subsets,
                subset_size=subset_size,
                degree=degree,
                gamma=normalized_gamma,
                coef=normalized_coef,
                sync_on_compute=False,
            )
        except (ImportError, ModuleNotFoundError) as error:
            raise _quality_dependency_error("KID") from error
        super().__init__(metric, metric_name="KID")

    def compute(self) -> dict[str, torch.Tensor]:
        """Return the stable KID ``mean``/``std`` result mapping."""

        cuda_devices = (
            list(range(torch.cuda.device_count()))
            if torch.cuda.is_available()
            else []
        )
        mps_state = (
            torch.mps.get_rng_state()
            if self.metric.device.type == "mps"
            else None
        )
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(self.seed)
                result = self.metric.compute()
        finally:
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("KID compute result must be a (mean, std) tuple")
        mean, std = result
        return {
            "mean": _scalar_tensor(mean, metric_name="KID mean"),
            "std": _scalar_tensor(std, metric_name="KID std"),
        }


__all__ = [
    "FrechetInceptionDistanceMetric",
    "KernelInceptionDistanceMetric",
]

"""ReferenceMetricProvider execution for AFHQ-v2 class-aware evaluation."""

from __future__ import annotations

import gc
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch

from stochaflow.data import DataLoaders
from stochaflow.sampling.runtime import SamplingRunResult
from stochaflow.training.diagnostics import (
    DIAGNOSTIC_PROVIDERS,
    ReferenceMetricProvider,
)
from stochaflow.training.diagnostics.runtime import prepare_reference_images
from stochaflow_afhq_v2.tools.evaluation_config import (
    AFHQV2EvaluationProtocol,
    AFHQV2MetricSpec,
)

MetricProviderFactory = Callable[
    [AFHQV2MetricSpec, torch.device, int, int],
    ReferenceMetricProvider,
]


def default_provider_factory(
    spec: AFHQV2MetricSpec,
    device: torch.device,
    num_real: int,
    num_fake: int,
) -> ReferenceMetricProvider:
    """Construct one registered quality provider with an actionable failure."""

    try:
        provider_value = cast(
            object,
            DIAGNOSTIC_PROVIDERS.reference_metrics.create(
                spec.name,
                device=device,
                num_real=num_real,
                num_fake=num_fake,
                **spec.params,
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            f"AFHQ-v2 evaluation metric '{spec.name}' is unavailable: {exc}"
        ) from exc
    if not isinstance(provider_value, ReferenceMetricProvider):
        raise TypeError(
            f"reference metric '{spec.name}' returned an incompatible provider"
        )
    return provider_value


def release_metric_device(device: torch.device) -> None:
    """Release provider objects before the next memory-heavy lifecycle."""

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def preflight_metric_providers(
    protocol: AFHQV2EvaluationProtocol,
    *,
    device: torch.device,
    factory: MetricProviderFactory,
) -> None:
    """Construct every metric before data or sampling side effects begin."""

    scope_sizes = (
        (
            protocol.real_per_class * len(protocol.class_mapping),
            protocol.fake_per_class * len(protocol.class_mapping),
        ),
        (protocol.real_per_class, protocol.fake_per_class),
    )
    for num_real, num_fake in dict.fromkeys(scope_sizes):
        for spec in protocol.metrics:
            provider: ReferenceMetricProvider | None = None
            primary: BaseException | None = None
            try:
                provider = factory(spec, device, num_real, num_fake)
            except BaseException as exc:  # noqa: BLE001
                primary = exc
            cleanup_failures = _cleanup_providers(
                (
                    {spec.name: provider}
                    if provider is not None
                    else {}
                ),
                device=device,
                scope="preflight",
            )
            _raise_after_cleanup(primary, cleanup_failures)


def _labeled_batch(batch: object) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) != 2:
        raise TypeError(
            "AFHQ-v2 test batches must be (images, {'class_label': labels})"
        )
    images = cast(object, batch[0])
    conditions = cast(object, batch[1])
    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise TypeError("AFHQ-v2 test batch images must be a rank-4 Tensor")
    if not isinstance(conditions, Mapping) or set(conditions) != {"class_label"}:
        raise TypeError(
            "AFHQ-v2 test batch conditions must contain only class_label"
        )
    labels = cast(object, conditions["class_label"])
    if (
        not isinstance(labels, torch.Tensor)
        or labels.ndim != 1
        or labels.dtype != torch.long
        or labels.shape[0] != images.shape[0]
    ):
        raise TypeError(
            "AFHQ-v2 test class_label must be a matching 1D long Tensor"
        )
    return images, labels


def collect_real_test_images(
    loaders: DataLoaders,
    protocol: AFHQV2EvaluationProtocol,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Select balanced real images in authenticated manifest order."""

    if loaders.test is None:
        raise ValueError("AFHQ-v2 DataBuilder must expose the official test split")
    parts: dict[str, list[torch.Tensor]] = {
        name: [] for name in protocol.class_mapping
    }
    counts = dict.fromkeys(protocol.class_mapping, 0)
    label_to_name = {
        label: name for name, label in protocol.class_mapping.items()
    }
    iterator = iter(loaders.test)
    try:
        for batch in iterator:
            images, labels = _labeled_batch(batch)
            for label, class_name in label_to_name.items():
                remaining = protocol.real_per_class - counts[class_name]
                if remaining <= 0:
                    continue
                indices = torch.nonzero(labels == label, as_tuple=False).flatten()
                if indices.numel() == 0:
                    continue
                selected = images.index_select(0, indices[:remaining])
                parts[class_name].append(selected.detach().cpu().clone())
                counts[class_name] += selected.shape[0]
            if all(count == protocol.real_per_class for count in counts.values()):
                break
    finally:
        # PyTorch has no public iterator-close API. Its multiprocessing iterator
        # exposes this hook, which prevents persistent workers from surviving an
        # early break or a malformed-batch failure until garbage collection.
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            shutdown_workers()
    missing = {
        name: protocol.real_per_class - count
        for name, count in counts.items()
        if count != protocol.real_per_class
    }
    if missing:
        raise ValueError(
            "official AFHQ-v2 test split cannot satisfy real allocation: "
            f"{missing}"
        )
    return {
        name: torch.cat(class_parts, dim=0)
        for name, class_parts in parts.items()
    }, counts


def load_generated_samples(
    sampling: SamplingRunResult,
    protocol: AFHQV2EvaluationProtocol,
) -> torch.Tensor:
    """Load the tensor writer artifact emitted by the core sampling runtime."""

    samples_path = sampling.artifacts.get("samples")
    if samples_path is None:
        raise ValueError("sampling result is missing the tensor samples artifact")
    value = torch.load(samples_path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise TypeError("sampling samples artifact must contain a rank-4 Tensor")
    expected = protocol.fake_per_class * len(protocol.class_mapping)
    if value.shape[0] != expected:
        raise ValueError(
            f"sampling produced {value.shape[0]} images, expected {expected}"
        )
    if tuple(value.shape[1:]) != (3, 128, 128):
        raise ValueError(
            "sampling samples artifact must have shape (N, 3, 128, 128)"
        )
    return value


def split_fake_samples(
    samples: torch.Tensor,
    protocol: AFHQV2EvaluationProtocol,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Interpret ordered label blocks guaranteed by the SamplingBuilder."""

    offset = 0
    split: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    for class_name in protocol.class_mapping:
        class_samples = samples[offset : offset + protocol.fake_per_class]
        offset += protocol.fake_per_class
        split[class_name] = class_samples
        counts[class_name] = class_samples.shape[0]
    return split, counts


def _update(
    provider: ReferenceMetricProvider,
    images: torch.Tensor,
    *,
    real: bool,
    metric_name: str,
    scope: str,
) -> None:
    try:
        provider.update(images, real=real)
    except Exception as exc:
        kind = "real" if real else "fake"
        raise RuntimeError(
            f"reference metric '{metric_name}' failed during {scope} "
            f"{kind} update: {exc}"
        ) from exc


def _metric_values(
    providers: Mapping[str, ReferenceMetricProvider],
    *,
    scope: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, provider in providers.items():
        try:
            values_value = cast(object, provider.compute())
        except Exception as exc:
            raise RuntimeError(
                f"reference metric '{name}' failed during {scope} compute: {exc}"
            ) from exc
        if not isinstance(values_value, Mapping):
            raise TypeError(
                f"reference metric '{name}' compute result must be a mapping"
            )
        values = cast(Mapping[object, object], values_value)
        for key, value in values.items():
            if not isinstance(key, str) or not key:
                raise TypeError("reference metric result keys must be non-empty")
            try:
                metric = float(cast(Any, value))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"reference metric '{scope}/{key}' must be numeric"
                ) from exc
            if not math.isfinite(metric):
                raise ValueError(
                    f"reference metric '{scope}/{key}' must be finite"
                )
            if key in metrics:
                raise ValueError(
                    f"reference metric key collision in {scope}: {key}"
                )
            metrics[key] = metric
    return metrics


def _cleanup_providers(
    providers: Mapping[str, ReferenceMetricProvider],
    *,
    device: torch.device,
    scope: str,
) -> list[Exception]:
    failures: list[Exception] = []
    for name, provider in providers.items():
        try:
            provider.reset_fake()
        except BaseException as exc:  # noqa: BLE001
            failures.append(
                RuntimeError(
                    f"reference metric '{name}' failed during "
                    f"{scope} reset: {exc}"
                )
            )
    try:
        release_metric_device(device)
    except BaseException as exc:  # noqa: BLE001
        failures.append(
            RuntimeError(
                f"reference metric device release failed after {scope}: {exc}"
            )
        )
    return failures


def _raise_after_cleanup(
    primary: BaseException | None,
    cleanup_failures: Sequence[Exception],
) -> None:
    if primary is not None:
        for failure in cleanup_failures:
            primary.add_note(f"cleanup failure: {failure}")
        raise primary
    if cleanup_failures:
        raise ExceptionGroup(
            "reference metric cleanup failed",
            list(cleanup_failures),
        )


def _evaluate_scope(
    *,
    scope: str,
    real_groups: Sequence[torch.Tensor],
    fake_groups: Sequence[torch.Tensor],
    num_real: int,
    num_fake: int,
    protocol: AFHQV2EvaluationProtocol,
    device: torch.device,
    factory: MetricProviderFactory,
) -> tuple[dict[str, float], dict[str, str]]:
    providers: dict[str, ReferenceMetricProvider] = {}
    result: tuple[dict[str, float], dict[str, str]] | None = None
    primary: BaseException | None = None
    try:
        for spec in protocol.metrics:
            providers[spec.name] = factory(
                spec,
                device,
                num_real,
                num_fake,
            )
        identities = {
            name: f"{type(provider).__module__}.{type(provider).__qualname__}"
            for name, provider in providers.items()
        }
        with torch.inference_mode():
            for group in real_groups:
                for real in group.split(protocol.metric_batch_size):
                    prepared = prepare_reference_images(real).to(device)
                    for name, provider in providers.items():
                        _update(
                            provider,
                            prepared,
                            real=True,
                            metric_name=name,
                            scope=scope,
                        )
            for group in fake_groups:
                for fake in group.split(protocol.metric_batch_size):
                    prepared = prepare_reference_images(fake).to(device)
                    for name, provider in providers.items():
                        _update(
                            provider,
                            prepared,
                            real=False,
                            metric_name=name,
                            scope=scope,
                        )
        result = (_metric_values(providers, scope=scope), identities)
    except BaseException as exc:  # noqa: BLE001
        primary = exc
    cleanup_failures = _cleanup_providers(
        providers,
        device=device,
        scope=scope,
    )
    providers.clear()
    _raise_after_cleanup(primary, cleanup_failures)
    assert result is not None
    return result


def evaluate_reference_metrics(
    *,
    real_images: Mapping[str, torch.Tensor],
    fake_images: Mapping[str, torch.Tensor],
    protocol: AFHQV2EvaluationProtocol,
    device: torch.device,
    factory: MetricProviderFactory,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Evaluate aggregate then per-class scopes with bounded provider state."""

    aggregate_real = tuple(
        real_images[name] for name in protocol.class_mapping
    )
    aggregate_fake = tuple(
        fake_images[name] for name in protocol.class_mapping
    )
    aggregate, identities = _evaluate_scope(
        scope="aggregate",
        real_groups=aggregate_real,
        fake_groups=aggregate_fake,
        num_real=sum(images.shape[0] for images in aggregate_real),
        num_fake=sum(images.shape[0] for images in aggregate_fake),
        protocol=protocol,
        device=device,
        factory=factory,
    )
    per_class: dict[str, dict[str, float]] = {}
    for class_name in protocol.class_mapping:
        values, class_identities = _evaluate_scope(
            scope=class_name,
            real_groups=(real_images[class_name],),
            fake_groups=(fake_images[class_name],),
            num_real=real_images[class_name].shape[0],
            num_fake=fake_images[class_name].shape[0],
            protocol=protocol,
            device=device,
            factory=factory,
        )
        if class_identities != identities:
            raise ValueError(
                "reference metric provider identity changed between scopes"
            )
        per_class[class_name] = values
    return {"aggregate": aggregate, "per_class": per_class}, identities


__all__ = [
    "MetricProviderFactory",
    "collect_real_test_images",
    "default_provider_factory",
    "evaluate_reference_metrics",
    "load_generated_samples",
    "preflight_metric_providers",
    "release_metric_device",
    "split_fake_samples",
]

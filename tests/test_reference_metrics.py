"""Formal FID/KID adapters running through the task-neutral metric engine."""

from __future__ import annotations

import re
import subprocess
import sys
from types import ModuleType
from typing import Any, ClassVar

import pytest
import torch
from torchmetrics import Metric

from stochaflow.metrics import (
    MetricEngine,
    MetricRuntimeError,
    MetricSpec,
    MetricUpdate,
)
from stochaflow.metrics import reference as reference_metrics
from stochaflow.metrics.reference import (
    FrechetInceptionDistanceMetric,
    KernelInceptionDistanceMetric,
)


def test_builtin_component_load_discovers_reference_metrics() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from stochaflow.utils.factory import load_builtin_components; "
                "from stochaflow.utils.registry import REGISTRIES; "
                "load_builtin_components(); "
                "print(','.join(REGISTRIES.metrics.names()))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert {"fid", "kid"}.issubset(result.stdout.strip().split(","))


class FakeReferenceMetric(Metric):
    """Record real/fake state without loading an Inception network."""

    real_count: torch.Tensor
    fake_count: torch.Tensor
    reset_calls: int

    def __init__(self, **params: Any) -> None:
        super().__init__()
        self.params = params
        self.reset_calls = 0
        self.add_state("real_count", default=torch.tensor(0.0))
        self.add_state("fake_count", default=torch.tensor(0.0))

    def update(self, images: torch.Tensor, *, real: bool) -> None:
        count = torch.tensor(float(images.shape[0]), device=images.device)
        if real:
            self.real_count += count
        else:
            self.fake_count += count

    def reset(self) -> None:
        self.reset_calls += 1
        super().reset()


class FakeFrechetInceptionDistance(FakeReferenceMetric):
    """Return one deterministic scalar FID value."""

    instances: ClassVar[list[FakeFrechetInceptionDistance]] = []

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        type(self).instances.append(self)

    def compute(self) -> torch.Tensor:
        return self.real_count + 2.0 * self.fake_count


class FakeKernelInceptionDistance(FakeReferenceMetric):
    """Return deterministic KID mean and standard deviation values."""

    instances: ClassVar[list[FakeKernelInceptionDistance]] = []
    random_values: ClassVar[list[float]] = []

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        type(self).instances.append(self)

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        type(self).random_values.append(float(torch.rand(())))
        total = self.real_count + self.fake_count
        return total / 10.0, total / 100.0


def _install_fake_quality_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeFrechetInceptionDistance.instances.clear()
    FakeKernelInceptionDistance.instances.clear()
    FakeKernelInceptionDistance.random_values.clear()
    fid_module = ModuleType("torchmetrics.image.fid")
    kid_module = ModuleType("torchmetrics.image.kid")
    fid_module_value: Any = fid_module
    kid_module_value: Any = kid_module
    fid_module_value.FrechetInceptionDistance = FakeFrechetInceptionDistance
    kid_module_value.KernelInceptionDistance = FakeKernelInceptionDistance

    def import_quality_module(name: str) -> ModuleType:
        if name == "torchmetrics.image.fid":
            return fid_module
        if name == "torchmetrics.image.kid":
            return kid_module
        raise AssertionError(f"unexpected quality module import {name!r}")

    monkeypatch.setattr(reference_metrics, "import_module", import_quality_module)


def _quality_engine(monkeypatch: pytest.MonkeyPatch) -> MetricEngine:
    _install_fake_quality_metrics(monkeypatch)
    return MetricEngine(
        [
            MetricSpec(
                "fid_score",
                "fid",
                "images",
                {"feature": 64, "antialias": False},
            ),
            MetricSpec(
                "kid_score",
                "kid",
                "images",
                {
                    "feature": 192,
                    "subsets": 2,
                    "subset_size": 2,
                    "degree": 2,
                    "gamma": 0.25,
                    "coef": 0.5,
                    "seed": 20260802,
                },
            ),
        ]
    )


def test_fid_and_kid_run_through_metric_engine_and_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _quality_engine(monkeypatch)
    assert engine.to("cpu") is engine
    real = torch.zeros(2, 3, 4, 4)
    fake = torch.ones(3, 3, 4, 4)

    engine.update(
        {"images": MetricUpdate(args=(real,), kwargs={"real": True})}
    )
    engine.update(
        {"images": MetricUpdate(args=(fake,), kwargs={"real": False})}
    )

    global_rng_state = torch.get_rng_state()
    assert engine.compute(reset=True) == {
        "fid_score": pytest.approx(8.0),
        "kid_score/mean": pytest.approx(0.5),
        "kid_score/std": pytest.approx(0.05),
    }
    assert torch.equal(torch.get_rng_state(), global_rng_state)
    generator = torch.Generator().manual_seed(20260802)
    assert FakeKernelInceptionDistance.random_values == [
        pytest.approx(float(torch.rand((), generator=generator)))
    ]
    fid = FakeFrechetInceptionDistance.instances[-1]
    kid = FakeKernelInceptionDistance.instances[-1]
    assert fid.params == {
        "feature": 64,
        "normalize": True,
        "reset_real_features": True,
        "antialias": False,
        "sync_on_compute": False,
    }
    assert kid.params == {
        "feature": 192,
        "normalize": True,
        "reset_real_features": True,
        "subsets": 2,
        "subset_size": 2,
        "degree": 2,
        "gamma": 0.25,
        "coef": 0.5,
        "sync_on_compute": False,
    }
    assert fid.real_count.dtype == torch.float64
    assert fid.device == kid.device == torch.device("cpu")
    assert fid.real_count.item() == fid.fake_count.item() == 0.0
    assert kid.real_count.item() == kid.fake_count.item() == 0.0
    assert fid.reset_calls == kid.reset_calls == 1
    with pytest.raises(MetricRuntimeError, match="before a successful update"):
        engine.compute()


@pytest.mark.parametrize(("name", "label"), [("fid", "FID"), ("kid", "KID")])
def test_quality_dependency_failure_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    label: str,
) -> None:
    def unavailable(module_name: str) -> ModuleType:
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(reference_metrics, "import_module", unavailable)

    with pytest.raises(RuntimeError, match=rf"{label} requires.*extra quality"):
        MetricEngine([MetricSpec(label.lower(), name, "images")])


@pytest.mark.parametrize(
    ("metric_class", "params", "message"),
    [
        (FrechetInceptionDistanceMetric, {"feature": True}, "feature must be"),
        (
            FrechetInceptionDistanceMetric,
            {"antialias": 1},
            "antialias must be a bool",
        ),
        (KernelInceptionDistanceMetric, {"subsets": 0}, "positive integer"),
        (KernelInceptionDistanceMetric, {"subset_size": 1}, "at least 2"),
        (KernelInceptionDistanceMetric, {"degree": True}, "positive integer"),
        (KernelInceptionDistanceMetric, {"gamma": 0.0}, "gamma must be positive"),
        (KernelInceptionDistanceMetric, {"gamma": float("nan")}, "finite"),
        (KernelInceptionDistanceMetric, {"coef": 0.0}, "coef must be positive"),
        (KernelInceptionDistanceMetric, {"coef": float("inf")}, "finite"),
        (KernelInceptionDistanceMetric, {"seed": True}, "seed must be"),
        (KernelInceptionDistanceMetric, {"seed": -1}, "seed must be"),
        (KernelInceptionDistanceMetric, {"seed": 2**63}, "seed must be"),
    ],
)
def test_quality_metric_parameters_are_strict(
    monkeypatch: pytest.MonkeyPatch,
    metric_class: type[Metric],
    params: dict[str, Any],
    message: str,
) -> None:
    _install_fake_quality_metrics(monkeypatch)

    with pytest.raises((TypeError, ValueError), match=message):
        metric_class(**params)


@pytest.mark.parametrize(
    ("images", "real", "cause_type", "message"),
    [
        ("images", True, TypeError, "images must be a Tensor"),
        (torch.zeros(2, 3, 4), True, ValueError, "rank 4"),
        (
            torch.zeros(2, 1, 4, 4),
            True,
            ValueError,
            r"shape \(N, 3, H, W\)",
        ),
        (torch.zeros(2, 3, 4, 4, dtype=torch.uint8), True, TypeError, "dtype"),
        (torch.full((2, 3, 4, 4), 1.1), True, ValueError, r"\[0, 1\]"),
        (torch.full((2, 3, 4, 4), float("nan")), True, ValueError, "finite"),
        (torch.zeros(2, 3, 4, 4), 1, TypeError, "real must be a bool"),
    ],
)
def test_quality_metric_update_rejects_invalid_payloads_and_resets_engine(
    monkeypatch: pytest.MonkeyPatch,
    images: object,
    real: object,
    cause_type: type[Exception],
    message: str,
) -> None:
    engine = _quality_engine(monkeypatch)

    with pytest.raises(MetricRuntimeError, match=r"fid_score.*images") as error:
        engine.update(
            {
                "images": MetricUpdate(
                    args=(images,),
                    kwargs={"real": real},
                )
            }
        )

    assert isinstance(error.value.__cause__, cause_type)
    assert re.search(message, str(error.value.__cause__)) is not None
    with pytest.raises(MetricRuntimeError, match="before a successful update"):
        engine.compute()

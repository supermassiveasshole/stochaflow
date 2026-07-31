"""Tests for the extensible Gaussian simple-loss weighting boundary."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

import pytest
import torch

from stochaflow.training.gaussian_weighting import (
    ConstantGaussianSimpleLossWeighting,
    GaussianSimpleLossContext,
    GaussianSimpleLossWeighting,
    P2GaussianSimpleLossWeighting,
    build_gaussian_simple_loss_weighting,
    compute_gaussian_simple_loss_weights,
    register_gaussian_simple_loss_weighting,
)


@register_gaussian_simple_loss_weighting("tests.gaussian-scaled")
class ScaledGaussianSimpleLossWeighting(GaussianSimpleLossWeighting):
    """Small extension policy used to prove registry-only composition."""

    def __init__(self, scale: float) -> None:
        self.scale = scale

    @property
    def requires_per_sample_loss(self) -> bool:
        """Return true because this extension changes per-sample values."""

        return True

    def validate_contract(self, *, prediction_type: str) -> None:
        """Support every Gaussian prediction representation for this test."""

    def sample_weights(
        self,
        context: GaussianSimpleLossContext,
    ) -> torch.Tensor:
        """Scale one-like weights without inspecting framework internals."""

        return torch.ones_like(context.signal_to_noise_ratio) * self.scale


class FixedOutputGaussianSimpleLossWeighting(GaussianSimpleLossWeighting):
    """Test policy returning a caller-provided output object."""

    def __init__(self, output: object) -> None:
        self.output = output

    @property
    def requires_per_sample_loss(self) -> bool:
        """Return true so callers use the per-sample composition path."""

        return True

    def validate_contract(self, *, prediction_type: str) -> None:
        """Accept every prediction representation for validation tests."""

    def sample_weights(
        self,
        context: GaussianSimpleLossContext,
    ) -> torch.Tensor:
        """Return the deliberately invalid configured value."""

        return cast(torch.Tensor, self.output)


def gaussian_context(
    *,
    prediction_type: str = "epsilon",
    dtype: torch.dtype = torch.float32,
) -> GaussianSimpleLossContext:
    """Build a two-sample Gaussian weighting context."""

    return GaussianSimpleLossContext(
        prediction_type=prediction_type,  # type: ignore[arg-type]
        signal_to_noise_ratio=torch.tensor([0.25, 3.0], dtype=dtype),
    )


def test_builtin_policies_share_the_family_registry() -> None:
    constant = build_gaussian_simple_loss_weighting(
        {"name": "constant", "params": {}},
        path="loss_weighting",
    )
    p2 = build_gaussian_simple_loss_weighting(
        {"name": "p2", "params": {}},
        path="loss_weighting",
    )

    assert isinstance(constant, ConstantGaussianSimpleLossWeighting)
    assert isinstance(p2, P2GaussianSimpleLossWeighting)


def test_missing_declaration_builds_constant_policy() -> None:
    policy = build_gaussian_simple_loss_weighting(
        None,
        path="training.params.loss_weighting",
    )

    assert isinstance(policy, ConstantGaussianSimpleLossWeighting)
    assert not policy.requires_per_sample_loss
    context = gaussian_context(prediction_type="score")
    weights = compute_gaussian_simple_loss_weights(policy, context)
    assert torch.equal(weights, torch.ones_like(context.signal_to_noise_ratio))


def test_canonical_p2_declaration_builds_validated_policy() -> None:
    policy = build_gaussian_simple_loss_weighting(
        {
            "name": "p2",
            "params": {"k": 2, "gamma": 0.5},
        },
        path="training.params.loss_weighting",
    )

    assert isinstance(policy, P2GaussianSimpleLossWeighting)
    assert policy.k == 2.0
    assert policy.gamma == 0.5
    assert policy.requires_per_sample_loss
    context = gaussian_context()
    weights = compute_gaussian_simple_loss_weights(policy, context)
    expected = (2.0 + context.signal_to_noise_ratio).pow(-0.5)
    assert torch.equal(weights, expected)


def test_p2_gamma_zero_is_exact_identity() -> None:
    policy = P2GaussianSimpleLossWeighting(k=7.0, gamma=0.0)
    context = gaussian_context(dtype=torch.float64)

    weights = compute_gaussian_simple_loss_weights(policy, context)

    assert torch.equal(weights, torch.ones_like(context.signal_to_noise_ratio))


@pytest.mark.parametrize("prediction_type", ["x0", "v", "score"])
def test_p2_owns_prediction_contract_validation(prediction_type: str) -> None:
    context = gaussian_context(prediction_type=prediction_type)

    with pytest.raises(ValueError, match="requires prediction_type='epsilon'"):
        compute_gaussian_simple_loss_weights(
            P2GaussianSimpleLossWeighting(),
            context,
        )


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"k": True}, TypeError, "k must be numeric"),
        ({"k": 0.0}, ValueError, "k must be greater than zero"),
        ({"k": float("inf")}, ValueError, "k must be finite"),
        ({"gamma": False}, TypeError, "gamma must be numeric"),
        ({"gamma": -1.0}, ValueError, "gamma must be non-negative"),
        ({"gamma": float("nan")}, ValueError, "gamma must be finite"),
    ],
)
def test_p2_constructor_validates_its_parameters(
    kwargs: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        P2GaussianSimpleLossWeighting(**kwargs)  # type: ignore[arg-type]


def test_external_namespaced_policy_uses_the_same_factory() -> None:
    policy = build_gaussian_simple_loss_weighting(
        {
            "name": "tests.gaussian-scaled",
            "params": {"scale": 0.25},
        },
        path="training.params.loss_weighting",
    )
    context = gaussian_context(prediction_type="x0")

    weights = compute_gaussian_simple_loss_weights(policy, context)

    assert isinstance(policy, ScaledGaussianSimpleLossWeighting)
    assert torch.equal(
        weights,
        torch.full_like(context.signal_to_noise_ratio, 0.25),
    )


def test_public_extension_registration_requires_a_namespace() -> None:
    with pytest.raises(ValueError, match="must be namespaced"):
        register_gaussian_simple_loss_weighting("unqualified")


@pytest.mark.parametrize(
    ("value", "error_type", "message"),
    [
        ([], TypeError, "must be a mapping"),
        ({}, TypeError, r"loss_weighting\.name must be a non-empty string"),
        (
            {"name": "constant"},
            ValueError,
            r"canonical \{name, params\} declaration",
        ),
        (
            {"name": "p2"},
            ValueError,
            r"canonical \{name, params\} declaration",
        ),
        (
            {"name": " p2", "params": {}},
            ValueError,
            "must not contain leading or trailing whitespace",
        ),
        (
            {"name": "constant", "params": []},
            TypeError,
            r"loss_weighting\.params must be a mapping",
        ),
        (
            {"name": "constant", "params": {}, "extra": 1},
            ValueError,
            "unknown loss_weighting field",
        ),
    ],
)
def test_factory_rejects_noncanonical_declarations(
    value: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        build_gaussian_simple_loss_weighting(value, path="loss_weighting")


@pytest.mark.parametrize(
    "flat",
    [
        {"name": "p2", "k": 1.0},
        {"name": "p2", "gamma": 1.0},
        {"name": "p2", "k": 1.0, "gamma": 1.0},
    ],
)
def test_factory_explicitly_rejects_removed_flat_p2_config(
    flat: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError,
        match=r"parameters must be nested under loss_weighting\.params",
    ):
        build_gaussian_simple_loss_weighting(flat, path="loss_weighting")


@pytest.mark.parametrize(
    ("output", "error_type", "message"),
    [
        (None, TypeError, "weights must be a Tensor"),
        (torch.ones(2, 1), ValueError, r"shape \[B\]"),
        (torch.ones(3), ValueError, r"shape \[B\]"),
        (torch.ones(2, dtype=torch.int64), TypeError, "floating-point"),
        (torch.ones(2, dtype=torch.float64), ValueError, "share the SNR dtype"),
        (torch.tensor([1.0, float("inf")]), ValueError, "must be finite"),
        (torch.tensor([1.0, -0.1]), ValueError, "must be non-negative"),
    ],
)
def test_shared_boundary_rejects_invalid_policy_outputs(
    output: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        compute_gaussian_simple_loss_weights(
            FixedOutputGaussianSimpleLossWeighting(output),
            gaussian_context(),
        )


def test_shared_boundary_rejects_policy_output_on_another_device() -> None:
    with pytest.raises(ValueError, match="share the SNR device"):
        compute_gaussian_simple_loss_weights(
            FixedOutputGaussianSimpleLossWeighting(
                torch.ones(2, device="meta"),
            ),
            gaussian_context(),
        )


@pytest.mark.parametrize(
    ("snr", "error_type", "message"),
    [
        (torch.ones(2, 1), ValueError, r"shape \[B\]"),
        (torch.ones(2, dtype=torch.int64), TypeError, "floating-point"),
        (torch.tensor([1.0, float("nan")]), ValueError, "must not contain NaN"),
        (torch.tensor([1.0, -1.0]), ValueError, "non-negative"),
    ],
)
def test_context_validates_its_narrow_snr_contract(
    snr: torch.Tensor,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        GaussianSimpleLossContext(
            prediction_type="epsilon",
            signal_to_noise_ratio=snr,
        )


def test_weighting_module_has_no_forbidden_runtime_layer_imports() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "stochaflow"
        / "training"
        / "gaussian_weighting.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(
        module.startswith(
            (
                "stochaflow.models",
                "stochaflow.processes",
                "stochaflow.sampling",
                "stochaflow.training.objectives",
                "stochaflow.training.strategy",
                "stochaflow.training.trainer",
            )
        )
        for module in imports
    )

"""Contract tests for process-free Gaussian family mathematics."""

import pytest
import torch

from stochaflow.families import gaussian as gaussian_family
from stochaflow.families.gaussian import (
    PredictionType,
    normalize_gaussian_prediction,
)


def test_family_surface_is_limited_to_prediction_math() -> None:
    assert gaussian_family.__all__ == [
        "GaussianPrediction",
        "PredictionType",
        "normalize_gaussian_prediction",
    ]


@pytest.mark.parametrize(
    "prediction_type",
    ["epsilon", "x0", "v", "score"],
)
def test_prediction_normalization_uses_only_explicit_marginal_scales(
    prediction_type: PredictionType,
) -> None:
    clean = torch.tensor([[2.0, -1.5], [0.5, -0.25]], dtype=torch.float64)
    epsilon = torch.tensor([[0.3, -0.4], [-0.2, 0.1]], dtype=torch.float64)
    signal_scale = torch.tensor([[0.8], [0.7]], dtype=torch.float64)
    noise_scale = torch.tensor([[0.3], [0.4]], dtype=torch.float64)
    state = signal_scale * clean + noise_scale * epsilon
    outputs = {
        "epsilon": epsilon,
        "x0": clean,
        "v": signal_scale * epsilon - noise_scale * clean,
        "score": -epsilon / noise_scale,
    }

    prediction = normalize_gaussian_prediction(
        state,
        outputs[prediction_type],
        signal_scale=signal_scale,
        noise_scale=noise_scale,
        prediction_type=prediction_type,
    )

    torch.testing.assert_close(prediction.clean, clean)
    torch.testing.assert_close(prediction.epsilon, epsilon)
    assert prediction.clean.max() > 1.0


def test_family_math_rejects_scales_that_do_not_broadcast_to_state() -> None:
    with pytest.raises(ValueError, match="scales must broadcast to the state"):
        normalize_gaussian_prediction(
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            signal_scale=torch.ones(4),
            noise_scale=torch.ones(4),
        )

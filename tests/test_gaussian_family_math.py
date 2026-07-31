"""Contract tests for process-free Gaussian family mathematics."""

import pytest
import torch

from stochaflow.families import gaussian as gaussian_family
from stochaflow.families.gaussian import (
    PredictionType,
    interpolate_gaussian_log_variance,
    normalize_gaussian_prediction,
    split_gaussian_model_output,
)


def test_family_surface_is_limited_to_shared_tensor_math() -> None:
    assert gaussian_family.__all__ == [
        "GaussianPrediction",
        "PredictionType",
        "VarianceMode",
        "interpolate_gaussian_log_variance",
        "normalize_gaussian_prediction",
        "split_gaussian_model_output",
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


def test_family_math_splits_fixed_and_learned_range_model_outputs() -> None:
    state = torch.zeros(2, 3, 4, 4)
    fixed = torch.ones_like(state)
    learned = torch.cat((fixed, -fixed), dim=1)

    fixed_mean, fixed_variance = split_gaussian_model_output(
        fixed,
        state=state,
        variance_mode="fixed",
    )
    learned_mean, learned_variance = split_gaussian_model_output(
        learned,
        state=state,
        variance_mode="learned_range",
    )

    assert fixed_mean is fixed
    assert fixed_variance is None
    assert torch.equal(learned_mean, fixed)
    assert learned_variance is not None
    assert torch.equal(learned_variance, -fixed)


def test_family_math_interpolates_learned_range_endpoints() -> None:
    values = torch.tensor([[[[-1.0]]], [[[1.0]]]])
    lower = torch.tensor([[[[-3.0]]], [[[-4.0]]]], dtype=torch.float64)
    upper = torch.tensor([[[[-1.0]]], [[[-2.0]]]], dtype=torch.float64)

    result = interpolate_gaussian_log_variance(
        values,
        lower=lower,
        upper=upper,
    )

    assert result.dtype == values.dtype
    assert result[:, 0, 0, 0].tolist() == [-3.0, -2.0]

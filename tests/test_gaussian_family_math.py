"""Contract tests for process-free Gaussian family mathematics."""

import pytest
import torch

from stochaflow.families.gaussian import (
    GaussianPrediction,
    LearnedVarianceGaussianPrediction,
    PredictionType,
    gaussian_signal_to_noise_ratio,
    gaussian_training_target,
    normalize_gaussian_prediction,
    split_gaussian_model_output,
)
from stochaflow.processes import DiscreteGaussianProcess, GaussianScales
from stochaflow.sampling import (
    GaussianPrediction as SamplingGaussianPrediction,
)
from stochaflow.sampling import (
    LearnedVarianceGaussianPrediction as SamplingLearnedVarianceGaussianPrediction,
)
from stochaflow.sampling import (
    normalize_gaussian_prediction as normalize_sampling_gaussian_prediction,
)


class MixedDtypeScaleGaussianProcess(DiscreteGaussianProcess):
    """Preserve the pre-refactor Sampling contract for mixed scale dtypes."""

    def marginal_scales(
        self,
        state_times: torch.Tensor,
        broadcast_shape: torch.Size,
    ) -> GaussianScales:
        """Return shape-compatible signal/noise scales with distinct dtypes."""

        scales = super().marginal_scales(state_times, broadcast_shape)
        return GaussianScales(
            scales.signal.to(dtype=torch.float32),
            scales.noise.to(dtype=torch.float64),
        )


def test_sampling_reexports_family_prediction_types() -> None:
    assert SamplingGaussianPrediction is GaussianPrediction
    assert (
        SamplingLearnedVarianceGaussianPrediction
        is LearnedVarianceGaussianPrediction
    )


def test_sampling_wrapper_preserves_mixed_scale_dtype_compatibility() -> None:
    process = MixedDtypeScaleGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 2}}
    )
    state = torch.zeros(2, 1, dtype=torch.float32)

    prediction = normalize_sampling_gaussian_prediction(
        process,
        state,
        torch.tensor([1, 2]),
        torch.zeros_like(state),
        clip_denoised=False,
    )

    assert prediction.clean.dtype == torch.float64
    assert prediction.epsilon.dtype == torch.float32


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


@pytest.mark.parametrize(
    "prediction_type",
    ["epsilon", "x0", "v", "score"],
)
def test_training_target_uses_only_explicit_marginal_scales(
    prediction_type: PredictionType,
) -> None:
    clean = torch.tensor([[2.0, -1.5], [0.5, -0.25]])
    noise = torch.tensor([[0.3, -0.4], [-0.2, 0.1]])
    signal_scale = torch.tensor([[0.8], [0.7]])
    noise_scale = torch.tensor([[0.3], [0.4]])
    expected = {
        "epsilon": noise,
        "x0": clean,
        "v": signal_scale * noise - noise_scale * clean,
        "score": -noise / noise_scale,
    }

    target = gaussian_training_target(
        clean=clean,
        noise=noise,
        signal_scale=signal_scale,
        noise_scale=noise_scale,
        prediction_type=prediction_type,
    )

    torch.testing.assert_close(target, expected[prediction_type])


def test_signal_to_noise_ratio_preserves_scale_shape_dtype_and_device() -> None:
    signal_scale = torch.tensor([[0.8], [0.6]], dtype=torch.float64)
    noise_scale = torch.tensor([[0.4], [0.3]], dtype=torch.float64)

    snr = gaussian_signal_to_noise_ratio(
        signal_scale=signal_scale,
        noise_scale=noise_scale,
    )

    torch.testing.assert_close(
        snr,
        torch.tensor([[4.0], [4.0]], dtype=torch.float64),
    )
    assert snr.shape == signal_scale.shape
    assert snr.dtype == signal_scale.dtype
    assert snr.device == signal_scale.device


def test_model_output_split_supports_fixed_and_learned_range_heads() -> None:
    state = torch.zeros(2, 3, 4, 4)
    fixed_output = torch.randn_like(state)
    learned_output = torch.cat(
        (fixed_output, torch.full_like(fixed_output, 0.25)),
        dim=1,
    )

    fixed_mean, fixed_variance = split_gaussian_model_output(
        fixed_output,
        state=state,
        variance_mode="fixed",
    )
    learned_mean, learned_variance = split_gaussian_model_output(
        learned_output,
        state=state,
        variance_mode="learned_range",
    )

    assert fixed_mean is fixed_output
    assert fixed_variance is None
    torch.testing.assert_close(learned_mean, fixed_output)
    assert learned_variance is not None
    torch.testing.assert_close(learned_variance, torch.full_like(state, 0.25))


def test_family_math_rejects_scales_that_do_not_broadcast_to_state() -> None:
    with pytest.raises(ValueError, match="scales must broadcast to the state"):
        normalize_gaussian_prediction(
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            signal_scale=torch.ones(4),
            noise_scale=torch.ones(4),
        )

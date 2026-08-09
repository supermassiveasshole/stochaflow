"""Package-internal Gaussian prediction validation shared by samplers."""

import torch

from stochaflow.sampling.gaussian.dynamics import (
    GaussianPrediction,
    LearnedVarianceGaussianPrediction,
)


def validate_gaussian_prediction(
    value: object,
    *,
    state: torch.Tensor,
) -> GaussianPrediction:
    """Validate the prediction contract shared by Gaussian samplers."""

    if not isinstance(value, GaussianPrediction):
        raise TypeError("Gaussian dynamics must return GaussianPrediction")
    for name in ("clean", "epsilon", "model_output"):
        tensor = getattr(value, name)
        if tensor.shape != state.shape:
            raise ValueError(f"Gaussian prediction {name} must match the state shape")
        if tensor.device != state.device:
            raise ValueError(f"Gaussian prediction {name} must share the state device")
    if isinstance(value, LearnedVarianceGaussianPrediction):
        log_variance = value.log_variance
        if log_variance.device != state.device:
            raise ValueError(
                "Gaussian prediction log_variance must share the state device"
            )
        try:
            broadcast_shape = torch.broadcast_shapes(
                state.shape,
                log_variance.shape,
            )
        except RuntimeError as exc:
            raise ValueError(
                "Gaussian prediction log_variance must broadcast to the state"
            ) from exc
        if tuple(broadcast_shape) != tuple(state.shape):
            raise ValueError(
                "Gaussian prediction log_variance must broadcast to the state"
            )
    return value

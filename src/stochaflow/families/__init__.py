"""Algorithm-family mathematical contracts shared across runtime layers."""

from .gaussian import (
    GaussianPrediction,
    LearnedVarianceGaussianPrediction,
    PredictionType,
    VarianceMode,
    gaussian_signal_to_noise_ratio,
    gaussian_training_target,
    normalize_gaussian_prediction,
    split_gaussian_model_output,
)

__all__ = [
    "GaussianPrediction",
    "LearnedVarianceGaussianPrediction",
    "PredictionType",
    "VarianceMode",
    "gaussian_signal_to_noise_ratio",
    "gaussian_training_target",
    "normalize_gaussian_prediction",
    "split_gaussian_model_output",
]

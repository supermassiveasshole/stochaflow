"""Algorithm-family mathematical contracts shared across runtime layers."""

from .gaussian import (
    GaussianPrediction,
    PredictionType,
    VarianceMode,
    interpolate_gaussian_log_variance,
    normalize_gaussian_prediction,
    split_gaussian_model_output,
)

__all__ = [
    "GaussianPrediction",
    "PredictionType",
    "VarianceMode",
    "interpolate_gaussian_log_variance",
    "normalize_gaussian_prediction",
    "split_gaussian_model_output",
]

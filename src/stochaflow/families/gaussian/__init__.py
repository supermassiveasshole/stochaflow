"""Process-free Gaussian prediction mathematics shared across layers."""

from .prediction import (
    GaussianPrediction,
    PredictionType,
    normalize_gaussian_prediction,
)

__all__ = [
    "GaussianPrediction",
    "PredictionType",
    "normalize_gaussian_prediction",
]

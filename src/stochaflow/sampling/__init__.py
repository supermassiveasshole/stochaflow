"""Sampling builders, solvers, observers, and artifact writers."""

from stochaflow.inference import InferenceAssetProvider
from stochaflow.utils.sampling_recipe import SamplingRecipe

from .builder import (
    InferenceModelProvider,
    SamplingBuilder,
    SamplingBuilderContext,
    SamplingOutput,
)
from .dynamics import GenerativeDynamics
from .execution import execute_sampling_builder, validate_sampling_output
from .gaussian import (
    ClassConditionalDenoisingBuilder,
    CleanTargetVarianceReferenceGaussianDenoisingDynamics,
    DDIMSampler,
    DDPMAncestralSampler,
    GaussianDenoisingDynamics,
    GaussianModelDynamics,
    GaussianPrediction,
    GaussianTransition,
    LearnedVarianceGaussianPrediction,
    PredictionType,
    StandardDenoisingBuilder,
    TargetAwareGaussianDenoisingDynamics,
    VarianceMode,
    normalize_gaussian_prediction,
)
from .grid import (
    denormalize_samples,
    save_image_grid,
    save_trajectory_gif,
    save_trajectory_grid,
)
from .sampler import (
    Sampler,
    SamplerResult,
    SamplingObservation,
    SamplingObserver,
    TrajectoryObserver,
)
from .writers import (
    ImageSamplingArtifactWriter,
    SamplingArtifactContext,
    SamplingArtifactWriter,
    SamplingBatch,
    TensorSamplingArtifactWriter,
    write_sampling_artifacts,
)

__all__ = [
    "ClassConditionalDenoisingBuilder",
    "CleanTargetVarianceReferenceGaussianDenoisingDynamics",
    "DDIMSampler",
    "DDPMAncestralSampler",
    "GaussianDenoisingDynamics",
    "GaussianModelDynamics",
    "GaussianPrediction",
    "GaussianTransition",
    "GenerativeDynamics",
    "ImageSamplingArtifactWriter",
    "InferenceAssetProvider",
    "InferenceModelProvider",
    "LearnedVarianceGaussianPrediction",
    "PredictionType",
    "Sampler",
    "SamplerResult",
    "SamplingArtifactContext",
    "SamplingArtifactWriter",
    "SamplingBatch",
    "SamplingBuilder",
    "SamplingBuilderContext",
    "SamplingObservation",
    "SamplingObserver",
    "SamplingOutput",
    "SamplingRecipe",
    "StandardDenoisingBuilder",
    "TargetAwareGaussianDenoisingDynamics",
    "TensorSamplingArtifactWriter",
    "TrajectoryObserver",
    "VarianceMode",
    "denormalize_samples",
    "execute_sampling_builder",
    "normalize_gaussian_prediction",
    "save_image_grid",
    "save_trajectory_gif",
    "save_trajectory_grid",
    "validate_sampling_output",
    "write_sampling_artifacts",
]

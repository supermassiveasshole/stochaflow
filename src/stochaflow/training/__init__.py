"""Training package and public extension contracts."""

from .builder import (
    InferenceAssetProjection,
    ManagedTrainingModule,
    TrainingBuilder,
    TrainingBuilderContext,
    TrainingPlan,
    build_training_plan,
    trainable_parameters,
    training_module_roots,
    validate_training_plan,
)
from .builtin import SupervisedTrainingBuilder, SupervisedTrainingStrategy
from .class_conditional_gaussian import (
    ClassConditionalGaussianDenoisingTrainingBuilder,
    ClassConditionalGaussianDenoisingTrainingStrategy,
    ClassConditionalGaussianDiagnosticSemantics,
)
from .diagnostics import (
    ClassConditionalDiffusionQualityDiagnostic,
    ContextAwareDiagnostic,
    DiagnosticBuildContext,
    FitStartEvent,
    TrainBatchEndEvent,
    TrainEpochEndEvent,
    TrainingDiagnostic,
)
from .ema import ExponentialMovingAverage
from .gaussian import (
    GaussianDenoisingTrainingBuilder,
    GaussianDenoisingTrainingStrategy,
    GaussianDiagnosticSemantics,
    gaussian_training_target,
)
from .objectives import MSEObjective, PerSampleObjective, compute_objective
from .optimization import WarmupCosineLR
from .precision import (
    PRECISION_KINDS,
    PrecisionKind,
    PrecisionRuntime,
    build_precision_runtime,
    validate_precision_support,
)
from .reporting import FinalSummary, RichTrainingReporter, RunSummary
from .strategy import (
    Batch,
    DeviceTransferableBatch,
    MetricChannelProvider,
    MetricUpdate,
    ReferenceImageBatchSemantics,
    ScalarMetric,
    TrainingStrategy,
    TrainStepOutput,
    loss_aggregation_weight_to_float,
    validate_train_step_output,
)
from .trainer import Trainer

__all__ = [
    "PRECISION_KINDS",
    "Batch",
    "ClassConditionalDiffusionQualityDiagnostic",
    "ClassConditionalGaussianDenoisingTrainingBuilder",
    "ClassConditionalGaussianDenoisingTrainingStrategy",
    "ClassConditionalGaussianDiagnosticSemantics",
    "ContextAwareDiagnostic",
    "DeviceTransferableBatch",
    "DiagnosticBuildContext",
    "ExponentialMovingAverage",
    "FinalSummary",
    "FitStartEvent",
    "GaussianDenoisingTrainingBuilder",
    "GaussianDenoisingTrainingStrategy",
    "GaussianDiagnosticSemantics",
    "InferenceAssetProjection",
    "MSEObjective",
    "ManagedTrainingModule",
    "MetricChannelProvider",
    "MetricUpdate",
    "PerSampleObjective",
    "PrecisionKind",
    "PrecisionRuntime",
    "ReferenceImageBatchSemantics",
    "RichTrainingReporter",
    "RunSummary",
    "ScalarMetric",
    "SupervisedTrainingBuilder",
    "SupervisedTrainingStrategy",
    "TrainBatchEndEvent",
    "TrainEpochEndEvent",
    "TrainStepOutput",
    "Trainer",
    "TrainingBuilder",
    "TrainingBuilderContext",
    "TrainingDiagnostic",
    "TrainingPlan",
    "TrainingStrategy",
    "WarmupCosineLR",
    "build_precision_runtime",
    "build_training_plan",
    "compute_objective",
    "gaussian_training_target",
    "loss_aggregation_weight_to_float",
    "trainable_parameters",
    "training_module_roots",
    "validate_precision_support",
    "validate_train_step_output",
    "validate_training_plan",
]

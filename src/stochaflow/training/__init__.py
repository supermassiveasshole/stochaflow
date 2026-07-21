"""Training package and public extension contracts."""

from .builder import (
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
from .diagnostics import (
    ContextAwareDiagnostic,
    DiagnosticBuildContext,
    FitStartEvent,
    TrainBatchEndEvent,
    TrainEpochEndEvent,
    TrainingDiagnostic,
)
from .ema import ExponentialMovingAverage
from .gaussian import (
    GaussianDiagnosticSemantics,
    GaussianDenoisingTrainingBuilder,
    GaussianDenoisingTrainingStrategy,
)
from .objectives import MSEObjective, PerSampleObjective, compute_objective
from .reporting import FinalSummary, RichTrainingReporter, RunSummary
from .strategy import (
    Batch,
    ScalarMetric,
    TrainStepOutput,
    TrainingStrategy,
    validate_train_step_output,
)
from .trainer import Trainer

__all__ = [
    "Batch",
    "ContextAwareDiagnostic",
    "DiagnosticBuildContext",
    "ExponentialMovingAverage",
    "FinalSummary",
    "FitStartEvent",
    "GaussianDiagnosticSemantics",
    "GaussianDenoisingTrainingBuilder",
    "GaussianDenoisingTrainingStrategy",
    "MSEObjective",
    "ManagedTrainingModule",
    "PerSampleObjective",
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
    "build_training_plan",
    "compute_objective",
    "trainable_parameters",
    "training_module_roots",
    "validate_train_step_output",
    "validate_training_plan",
]

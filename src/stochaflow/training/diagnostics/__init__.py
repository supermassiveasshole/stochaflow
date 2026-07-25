"""Composable training diagnostics and provider extension interfaces."""

# Import built-ins before the orchestrator can resolve its default provider set.
from . import providers as providers
from .config import (
    DiagnosticCadenceConfig,
    DiagnosticSamplingConfig,
    DiffusionQualityConfig,
    ProviderPipelineConfig,
    ProviderSpec,
    ReferencePipelineConfig,
    SamplerProfileConfig,
    TrajectoryProviderConfig,
)
from .contracts import (
    ArtifactRecord,
    ContextAwareDiagnostic,
    DenoiserArtifactContext,
    DenoiserArtifactProvider,
    DiagnosticBuildContext,
    FitStartEvent,
    ProviderValidationContext,
    ReferenceMetricProvider,
    SamplerArtifactContext,
    SamplerArtifactProvider,
    SamplerMetricContext,
    SamplerMetricProvider,
    StepMetricContext,
    StepMetricProvider,
    TrainBatchEndEvent,
    TrainEpochEndEvent,
    TrainingDiagnostic,
)
from .diffusion_quality import DiffusionQualityDiagnostic
from .registry import DIAGNOSTIC_PROVIDERS, DiagnosticProviderCatalog

__all__ = [
    "DIAGNOSTIC_PROVIDERS",
    "ArtifactRecord",
    "ContextAwareDiagnostic",
    "DenoiserArtifactContext",
    "DenoiserArtifactProvider",
    "DiagnosticBuildContext",
    "DiagnosticCadenceConfig",
    "DiagnosticProviderCatalog",
    "DiagnosticSamplingConfig",
    "DiffusionQualityConfig",
    "DiffusionQualityDiagnostic",
    "FitStartEvent",
    "ProviderPipelineConfig",
    "ProviderSpec",
    "ProviderValidationContext",
    "ReferenceMetricProvider",
    "ReferencePipelineConfig",
    "SamplerArtifactContext",
    "SamplerArtifactProvider",
    "SamplerMetricContext",
    "SamplerMetricProvider",
    "SamplerProfileConfig",
    "StepMetricContext",
    "StepMetricProvider",
    "TrainBatchEndEvent",
    "TrainEpochEndEvent",
    "TrainingDiagnostic",
    "TrajectoryProviderConfig",
    "providers",
]

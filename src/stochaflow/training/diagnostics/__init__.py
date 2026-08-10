"""Composable training diagnostics and provider extension interfaces."""

# Expose provider implementations for the explicit built-in activation owner.
from . import providers as providers
from .class_conditional_quality import (
    ClassConditionalDiffusionQualityDiagnostic,
    DiagnosticClassAllocation,
)
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
    "ClassConditionalDiffusionQualityDiagnostic",
    "ContextAwareDiagnostic",
    "DenoiserArtifactContext",
    "DenoiserArtifactProvider",
    "DiagnosticBuildContext",
    "DiagnosticCadenceConfig",
    "DiagnosticClassAllocation",
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

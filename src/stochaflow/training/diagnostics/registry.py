"""Registries for independently extensible diagnostic providers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from stochaflow.training.diagnostics.contracts import (
    DenoiserArtifactProvider,
    ReferenceMetricProvider,
    SamplerArtifactProvider,
    SamplerMetricProvider,
    StepMetricProvider,
)
from stochaflow.utils.registry import Registry, RegistryError


class DiagnosticProviderCatalog:
    """Typed provider registries and external module discovery."""

    def __init__(self) -> None:
        self.step_metrics: Registry[type[StepMetricProvider]] = Registry(
            "diagnostic step metric provider",
            expected_type=StepMetricProvider,
        )
        self.sampler_metrics: Registry[type[SamplerMetricProvider]] = Registry(
            "diagnostic sampler metric provider",
            expected_type=SamplerMetricProvider,
        )
        self.denoiser_artifacts: Registry[type[DenoiserArtifactProvider]] = Registry(
            "diagnostic denoiser artifact provider",
            expected_type=DenoiserArtifactProvider,
        )
        self.sampler_artifacts: Registry[type[SamplerArtifactProvider]] = Registry(
            "diagnostic sampler artifact provider",
            expected_type=SamplerArtifactProvider,
        )
        self.reference_metrics: Registry[type[ReferenceMetricProvider]] = Registry(
            "diagnostic reference metric provider",
            expected_type=ReferenceMetricProvider,
        )
        self._loaded_modules: set[str] = set()

    def load_modules(self, modules: tuple[str, ...]) -> None:
        """Import configured provider modules exactly once in declaration order."""

        for module_name in modules:
            if module_name in self._loaded_modules:
                continue
            try:
                import_module(module_name)
            except Exception as exc:
                raise RegistryError(
                    f"failed to import diagnostic provider module "
                    f"'{module_name}': {exc}"
                ) from exc
            self._loaded_modules.add(module_name)

    def registry(self, category: str) -> Registry[type[Any]]:
        """Resolve a provider registry by configuration category."""

        try:
            registry = getattr(self, category)
        except AttributeError as exc:
            raise RegistryError(
                f"unknown diagnostic provider category '{category}'"
            ) from exc
        if not isinstance(registry, Registry):
            raise RegistryError(
                f"diagnostic provider category '{category}' is not a registry"
            )
        return registry


DIAGNOSTIC_PROVIDERS = DiagnosticProviderCatalog()


__all__ = ["DIAGNOSTIC_PROVIDERS", "DiagnosticProviderCatalog"]

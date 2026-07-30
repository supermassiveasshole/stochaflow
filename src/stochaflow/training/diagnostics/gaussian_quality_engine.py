"""Private orchestration engine shared by Gaussian quality diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, TypeVar

import torch

from stochaflow.processes import DiscreteGaussianDenoisingProcess
from stochaflow.training.diagnostics.config import (
    ProviderSpec,
    SamplerProfileConfig,
    parse_diffusion_quality_config,
)
from stochaflow.training.diagnostics.contracts import (
    ArtifactRecord,
    DenoiserArtifactContext,
    DenoiserArtifactProvider,
    FitStartEvent,
    ProviderValidationContext,
    ReconstructionCallable,
    ReferenceMetricProvider,
    SamplerArtifactContext,
    SamplerMetricContext,
    SamplingResult,
    StepMetricContext,
    TrainBatchEndEvent,
    TrainEpochEndEvent,
)
from stochaflow.training.diagnostics.manifest import EpochArtifactStore
from stochaflow.training.diagnostics.providers.reference import ReferenceMetricSuite
from stochaflow.training.diagnostics.registry import DIAGNOSTIC_PROVIDERS
from stochaflow.training.diagnostics.runtime import (
    BoundSampler,
    EvaluationGuard,
    SeedPolicy,
    clean_samples_from_event,
)
from stochaflow.utils.logging import ExperimentLogger


class DiagnosticTrainingRuntime(Protocol):
    """Minimum training-family runtime needed by the quality orchestrator."""

    @property
    def process(self) -> DiscreteGaussianDenoisingProcess:
        """Return the diagnostic's model-free process."""

        ...


FamilyRuntimeT = TypeVar(
    "FamilyRuntimeT",
    bound=DiagnosticTrainingRuntime,
)
FamilySamplerT = TypeVar("FamilySamplerT")
PoolSamplerT_co = TypeVar("PoolSamplerT_co", covariant=True)
RunnerSamplerT_contra = TypeVar("RunnerSamplerT_contra", contravariant=True)


class GaussianQualitySamplerPool(Protocol[PoolSamplerT_co]):
    """Typed lookup for one diagnostic family's bound samplers."""

    def get(self, profile_id: str) -> PoolSamplerT_co:
        """Return the bound sampler for one configured profile."""

        ...


class GaussianQualitySamplerRunner(Protocol[RunnerSamplerT_contra]):
    """Typed execution boundary for one diagnostic sampler family."""

    def run(
        self,
        sampler: RunnerSamplerT_contra,
        profile: SamplerProfileConfig,
        initial_noise: torch.Tensor,
    ) -> SamplingResult:
        """Generate one profile result."""

        ...


class GaussianQualityFamily(Protocol[FamilyRuntimeT, FamilySamplerT]):
    """Inject family-specific runtime, sampler, and conditioning behavior."""

    def training_runtime(self, trainer: Any) -> FamilyRuntimeT:
        """Resolve the narrow training capability needed by this family."""

        ...

    def build_sampling(
        self,
        runtime: FamilyRuntimeT,
        profiles: Sequence[SamplerProfileConfig],
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[
        GaussianQualitySamplerPool[FamilySamplerT],
        GaussianQualitySamplerRunner[FamilySamplerT],
    ]:
        """Build the family-specific sampler pool and runner."""

        ...

    def capture_batch(self, event: TrainBatchEndEvent) -> None:
        """Capture family-specific state from one successful batch."""

        ...

    def reconstruction_evaluator(
        self,
        trainer: Any,
        seed_policy: SeedPolicy,
    ) -> ReconstructionCallable:
        """Build the family-specific reconstruction service."""

        ...

    def reference_sampler(self, sampler: FamilySamplerT) -> BoundSampler:
        """Adapt a family sampler to the unconditional reference protocol."""

        ...

    def reference_image_extractor(
        self,
        trainer: Any,
    ) -> Callable[[Any], torch.Tensor]:
        """Resolve the strategy-owned validation batch image extractor."""

        ...

    def profile_manifest_metadata(
        self,
        profile: SamplerProfileConfig,
    ) -> Mapping[str, Any]:
        """Return family-specific metadata for one sampler profile."""

        ...

    def manifest_metadata(
        self,
        event: TrainEpochEndEvent,
    ) -> Mapping[str, Any]:
        """Return family-specific epoch manifest metadata."""

        ...


class GaussianQualityEngine[
    RuntimeT: DiagnosticTrainingRuntime,
    BoundSamplerT,
]:
    """Coordinate configured metrics, artifacts, samplers, and references."""

    def __init__(
        self,
        *,
        logger: ExperimentLogger,
        output_dir: str | Path,
        samplers: Sequence[Mapping[str, Any]],
        modules: Sequence[str] = (),
        cadence: Mapping[str, Any] | None = None,
        sampling: Mapping[str, Any] | None = None,
        providers: Mapping[str, Any] | None = None,
        reference: Mapping[str, Any] | None = None,
        use_ema: bool = True,
        failure_policy: str = "raise",
        diagnostic_name: str,
        family: GaussianQualityFamily[RuntimeT, BoundSamplerT],
    ) -> None:
        self.config = parse_diffusion_quality_config(
            modules=modules,
            cadence=cadence,
            sampling=sampling,
            samplers=samplers,
            providers=providers,
            reference=reference,
            use_ema=use_ema,
            failure_policy=failure_policy,
        )
        self.sample_shape = self.config.sampling.shape
        self.logger = logger
        self.output_dir = Path(output_dir) / "diagnostics" / diagnostic_name
        self.family = family
        self.seed_policy = SeedPolicy(self.config.sampling.seed)
        self._last_clean_batch: torch.Tensor | None = None
        self._sampler_pool: (
            GaussianQualitySamplerPool[BoundSamplerT] | None
        ) = None
        self._sampler_runner: (
            GaussianQualitySamplerRunner[BoundSamplerT] | None
        ) = None
        self._reference_suite: ReferenceMetricSuite | None = None
        self._error_count = 0
        self._reference_store: EpochArtifactStore | None = None
        self._reference_step = 0
        self._reference_profile = ""

        DIAGNOSTIC_PROVIDERS.load_modules(self.config.modules)
        self.step_metrics = self._build_providers(
            "step_metrics",
            self.config.providers.step_metrics,
        )
        self.sampler_metrics = self._build_providers(
            "sampler_metrics",
            self.config.providers.sampler_metrics,
        )
        self.denoiser_artifacts = self._build_providers(
            "denoiser_artifacts",
            self.config.providers.denoiser_artifacts,
        )
        self.sampler_artifacts = self._build_providers(
            "sampler_artifacts",
            self.config.providers.sampler_artifacts,
        )
        for spec in self.config.reference.metrics:
            DIAGNOSTIC_PROVIDERS.reference_metrics.resolve(spec.name)

    def _build_providers(
        self,
        category: str,
        specs: Sequence[ProviderSpec],
    ) -> tuple[Any, ...]:
        registry = DIAGNOSTIC_PROVIDERS.registry(category)
        return tuple(registry.create(spec.name, **spec.params) for spec in specs)

    def on_fit_start(self, event: FitStartEvent) -> None:
        """Validate providers, construct samplers, and cache real features."""

        system = self.family.training_runtime(event.trainer)
        with self.seed_policy.fork_rng(event.trainer.device):
            self._sampler_pool, self._sampler_runner = self.family.build_sampling(
                system,
                self.config.samplers,
                batch_size=self.config.sampling.batch_size,
                device=event.trainer.device,
            )
            validation = ProviderValidationContext(
                process=system.process,
                sample_shape=self.sample_shape,
            )
            for _, provider in self._named_non_reference_providers():
                provider.validate(validation)

            if self.config.reference.enabled:
                if event.validation_dataloader is None:
                    raise ValueError(
                        "diffusion_quality reference metrics require a validation "
                        "dataloader"
                    )
                extract_images = self.family.reference_image_extractor(
                    event.trainer
                )
                reference_providers = self._build_reference_providers(
                    event.trainer.device
                )
                for _, provider in reference_providers:
                    provider.validate(validation)
                self._reference_suite = ReferenceMetricSuite(
                    reference_providers,
                    self.config.reference,
                    device=event.trainer.device,
                    seed_policy=self.seed_policy,
                    handle_error=self._handle_reference_error,
                    extract_images=extract_images,
                )
                cache_metrics = self._reference_suite.cache_real(
                    event.validation_dataloader
                )
                self.logger.log_metrics(
                    dict(cache_metrics),
                    step=event.trainer.global_step,
                )

    def _build_reference_providers(
        self,
        device: torch.device,
    ) -> tuple[tuple[str, ReferenceMetricProvider], ...]:
        providers: list[tuple[str, ReferenceMetricProvider]] = []
        for spec in self.config.reference.metrics:
            provider = DIAGNOSTIC_PROVIDERS.reference_metrics.create(
                spec.name,
                device=device,
                num_real=self.config.reference.num_real,
                num_fake=self.config.reference.num_fake,
                **spec.params,
            )
            providers.append((spec.name, provider))
        return tuple(providers)

    def on_train_batch_end(self, event: TrainBatchEndEvent) -> None:
        """Capture clean samples and dispatch configured step metric providers."""

        try:
            self.family.capture_batch(event)
        except Exception as exc:  # noqa: BLE001
            self._handle_runtime_error(
                exc,
                step=event.global_step,
                phase="step",
                provider="orchestrator",
            )
            return
        clean = clean_samples_from_event(event.output, event.batch)
        if clean is not None:
            self._last_clean_batch = clean.detach().cpu()
        if event.global_step % self.config.cadence.step_every != 0:
            return
        diagnostics = getattr(event.output, "diagnostics", None)
        if not isinstance(diagnostics, Mapping):
            self._handle_runtime_error(
                TypeError("TrainStepOutput.diagnostics must be a mapping"),
                step=event.global_step,
                phase="step",
                provider="orchestrator",
            )
            return
        context = StepMetricContext(
            process=self.family.training_runtime(event.trainer).process,
            diagnostics=diagnostics,
            clean_samples=self._last_clean_batch,
            sample_num=self.config.sampling.sample_num,
            use_ema=self.config.use_ema,
            reconstruct=self.family.reconstruction_evaluator(
                event.trainer,
                self.seed_policy,
            ),
        )
        metrics: dict[str, float] = {}
        for spec, provider in zip(
            self.config.providers.step_metrics,
            self.step_metrics,
            strict=True,
        ):
            try:
                self._merge_metrics(
                    metrics,
                    provider.collect(context),
                    provider=spec.name,
                )
            # Providers are extension code and may raise any ordinary exception.
            except Exception as exc:  # noqa: BLE001
                self._handle_runtime_error(
                    exc,
                    step=event.global_step,
                    phase="step_metric",
                    provider=spec.name,
                )
        if metrics:
            self.logger.log_metrics(metrics, step=event.global_step)

    def on_train_epoch_end(self, event: TrainEpochEndEvent) -> None:
        """Dispatch sampler, artifact, and optional reference providers."""

        artifact_due = (
            event.epoch_index % self.config.cadence.artifact_every_epochs == 0
        )
        reference_due = (
            self.config.reference.enabled
            and event.epoch_index % self.config.reference.every_epochs == 0
        )
        if not artifact_due and not reference_due:
            return
        store = EpochArtifactStore(self.output_dir, event.epoch_index)
        combined_metrics: dict[str, float] = {}
        profile_manifest: list[dict[str, Any]] = []
        if artifact_due:
            for spec, provider in zip(
                self.config.providers.denoiser_artifacts,
                self.denoiser_artifacts,
                strict=True,
            ):
                self._run_denoiser_artifact_provider(
                    spec.name,
                    provider,
                    event,
                    store,
                )

        initial_noise = (
            self.seed_policy.initial_noise(
                self.config.sampling.sample_num,
                self.sample_shape,
                event.trainer.device,
            )
            if artifact_due
            else None
        )
        for profile in self.config.samplers:
            profile_metrics: dict[str, float] = {}
            try:
                profile_metrics = self._run_profile(
                    event,
                    store,
                    profile,
                    initial_noise=initial_noise,
                    artifact_due=artifact_due,
                    reference_due=reference_due,
                )
                self._merge_metrics(
                    combined_metrics,
                    profile_metrics,
                    provider=f"profile:{profile.id}",
                )
            # A profile invokes registered sampler and provider extensions.
            except Exception as exc:  # noqa: BLE001
                self._handle_runtime_error(
                    exc,
                    step=event.trainer.global_step,
                    phase="profile",
                    provider=profile.id,
                    store=store,
                )
            profile_manifest.append(
                {
                    **asdict(profile),
                    "metrics": profile_metrics,
                    **self.family.profile_manifest_metadata(profile),
                }
            )

        if combined_metrics:
            self.logger.log_metrics(
                combined_metrics,
                step=event.trainer.global_step,
            )
        store.write_manifest(
            {
                "epoch": event.epoch_index,
                "global_step": event.trainer.global_step,
                "sample_seed": self.config.sampling.seed,
                "sample_shape": list(self.sample_shape),
                "weights": (
                    "ema"
                    if self.config.use_ema and event.trainer.ema is not None
                    else "raw"
                ),
                "artifact_due": artifact_due,
                "reference_metrics_due": reference_due,
                "providers": asdict(self.config.providers),
                "profiles": profile_manifest,
                "combined_metrics": combined_metrics,
                **self.family.manifest_metadata(event),
            }
        )

    def _run_denoiser_artifact_provider(
        self,
        name: str,
        provider: DenoiserArtifactProvider,
        event: TrainEpochEndEvent,
        store: EpochArtifactStore,
    ) -> None:
        try:
            context = DenoiserArtifactContext(
                store=store,
                clean_samples=self._last_clean_batch,
                reconstruct=self.family.reconstruction_evaluator(
                    event.trainer,
                    self.seed_policy,
                ),
                use_ema=self.config.use_ema,
            )
            records = tuple(provider.render(context))
            self._record_artifacts(name, records, event.trainer.global_step, store)
        # Artifact providers are extension code with no shared exception type.
        except Exception as exc:  # noqa: BLE001
            self._handle_runtime_error(
                exc,
                step=event.trainer.global_step,
                phase="denoiser_artifact",
                provider=name,
                store=store,
            )

    def _run_profile(
        self,
        event: TrainEpochEndEvent,
        store: EpochArtifactStore,
        profile: SamplerProfileConfig,
        *,
        initial_noise: torch.Tensor | None,
        artifact_due: bool,
        reference_due: bool,
    ) -> dict[str, float]:
        if self._sampler_pool is None or self._sampler_runner is None:
            raise RuntimeError("diffusion_quality on_fit_start was not called")
        sampler = self._sampler_pool.get(profile.id)
        metrics: dict[str, float] = {}
        result = None
        with EvaluationGuard(
            event.trainer,
            seed=self.seed_policy.profile_seed(profile.id),
            use_ema=self.config.use_ema,
            evaluation_modules=(),
        ):
            if artifact_due:
                assert initial_noise is not None
                try:
                    result = self._sampler_runner.run(
                        sampler,
                        profile,
                        initial_noise,
                    )
                # Registered samplers may raise any ordinary exception.
                except Exception as exc:  # noqa: BLE001
                    self._handle_runtime_error(
                        exc,
                        step=event.trainer.global_step,
                        phase="sampling",
                        provider=profile.id,
                        store=store,
                    )
            if result is not None:
                metric_context = SamplerMetricContext(
                    profile_id=profile.id,
                    profile_name=profile.name,
                    result=result,
                )
                for spec, provider in zip(
                    self.config.providers.sampler_metrics,
                    self.sampler_metrics,
                    strict=True,
                ):
                    try:
                        self._merge_metrics(
                            metrics,
                            provider.collect(metric_context),
                            provider=spec.name,
                        )
                    # Metric providers are extension code with arbitrary failures.
                    except Exception as exc:  # noqa: BLE001
                        self._handle_runtime_error(
                            exc,
                            step=event.trainer.global_step,
                            phase=f"sampler_metric:{profile.id}",
                            provider=spec.name,
                            store=store,
                        )
                artifact_context = SamplerArtifactContext(
                    store=store,
                    profile_id=profile.id,
                    profile_name=profile.name,
                    trajectory_enabled=profile.trajectory.enabled,
                    trajectory_gif_fps=profile.trajectory.gif_fps,
                    result=result,
                )
                for spec, provider in zip(
                    self.config.providers.sampler_artifacts,
                    self.sampler_artifacts,
                    strict=True,
                ):
                    try:
                        records = tuple(provider.render(artifact_context))
                        self._record_artifacts(
                            spec.name,
                            records,
                            event.trainer.global_step,
                            store,
                        )
                    # Artifact providers are extension code with arbitrary failures.
                    except Exception as exc:  # noqa: BLE001
                        self._handle_runtime_error(
                            exc,
                            step=event.trainer.global_step,
                            phase=f"sampler_artifact:{profile.id}",
                            provider=spec.name,
                            store=store,
                        )
            if reference_due:
                if self._reference_suite is None:
                    raise RuntimeError("reference metric suite was not initialized")
                self._reference_store = store
                self._reference_step = event.trainer.global_step
                self._reference_profile = profile.id
                try:
                    reference_metrics = self._reference_suite.evaluate(
                        profile_id=profile.id,
                        sampler=self.family.reference_sampler(sampler),
                        sample_shape=self.sample_shape,
                        visual_samples=(result.samples if result is not None else None),
                    )
                    self._merge_metrics(
                        metrics,
                        reference_metrics,
                        provider="reference",
                    )
                # The suite invokes registered sampler and metric extensions.
                except Exception as exc:  # noqa: BLE001
                    self._handle_runtime_error(
                        exc,
                        step=event.trainer.global_step,
                        phase=f"reference:{profile.id}",
                        provider="reference",
                        store=store,
                    )
                finally:
                    self._reference_store = None
                    self._reference_profile = ""
        return metrics

    def _record_artifacts(
        self,
        provider: str,
        records: Sequence[ArtifactRecord],
        step: int,
        store: EpochArtifactStore,
    ) -> None:
        store.record(provider, records)
        for record in records:
            if record.image_tag is not None:
                self.logger.log_image(
                    record.image_tag,
                    record.path,
                    step=step,
                    caption=record.caption,
                )

    def _merge_metrics(
        self,
        target: dict[str, float],
        incoming: object,
        *,
        provider: str,
    ) -> None:
        if not isinstance(incoming, Mapping):
            raise TypeError(f"provider '{provider}' must return a metric mapping")
        collisions = sorted(set(target).intersection(incoming))
        if collisions:
            raise ValueError(
                f"provider '{provider}' metric tag collision(s): "
                + ", ".join(collisions)
            )
        for tag, value in incoming.items():
            if not isinstance(tag, str) or not tag:
                raise ValueError(f"provider '{provider}' returned an invalid metric tag")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"provider '{provider}' metric '{tag}' must be numeric"
                )
            target[tag] = float(value)

    def _named_non_reference_providers(self):
        groups = (
            (self.config.providers.step_metrics, self.step_metrics),
            (self.config.providers.sampler_metrics, self.sampler_metrics),
            (self.config.providers.denoiser_artifacts, self.denoiser_artifacts),
            (self.config.providers.sampler_artifacts, self.sampler_artifacts),
        )
        for specs, providers in groups:
            for spec, provider in zip(specs, providers, strict=True):
                yield spec.name, provider

    def _handle_reference_error(
        self,
        phase: str,
        provider: str,
        error: Exception,
    ) -> None:
        self._handle_runtime_error(
            error,
            step=self._reference_step,
            phase=f"{phase}:{self._reference_profile}",
            provider=provider,
            store=self._reference_store,
        )

    def _handle_runtime_error(
        self,
        error: Exception,
        *,
        step: int,
        phase: str,
        provider: str,
        store: EpochArtifactStore | None = None,
    ) -> None:
        if self.config.failure_policy == "raise":
            raise error
        self._error_count += 1
        if store is not None:
            store.record_error(phase=phase, provider=provider, error=error)
        self.logger.log_metrics(
            {"diagnostics/system/error_count": float(self._error_count)},
            step=step,
        )
        self.logger.log_text(
            "diagnostics/system/error",
            (
                f"phase={phase}; provider={provider}; "
                f"{type(error).__name__}: {error}"
            ),
            step=step,
        )


__all__ = [
    "DiagnosticTrainingRuntime",
    "GaussianQualityEngine",
    "GaussianQualityFamily",
    "GaussianQualitySamplerPool",
    "GaussianQualitySamplerRunner",
]

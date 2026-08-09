"""Compose configured validation Evaluation runs against live training state."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

from torch import nn

from stochaflow.evaluation import (
    CheckpointWeightVariant,
    EvaluationPlan,
    EvaluationProtocol,
    LiveEvaluationSamplingCapability,
    build_evaluation_plan,
    execute_evaluation_plan,
)
from stochaflow.evaluation.artifacts import canonical_sha256
from stochaflow.evaluation.identity import build_protocol_implementation_identity
from stochaflow.inference import InferenceAssetProvider
from stochaflow.metrics import MetricSpec
from stochaflow.training.trainer import Trainer
from stochaflow.training.validation import (
    EpochValidationCadence,
    EpochValidationEvaluator,
    EpochValidationIdentity,
    EpochValidationResult,
)
from stochaflow.utils.checkpoint import (
    inference_asset_descriptors_from_projections,
)
from stochaflow.utils.config import (
    ComponentConfig,
    ValidationEvaluationConfig,
)
from stochaflow.utils.registry import REGISTRIES, RegistryCatalog
from stochaflow.utils.sampling_recipe import sampling_recipe_to_dict
from stochaflow.utils.seed import preserve_global_rng_state

ModelFactory = Callable[[ComponentConfig], nn.Module]


@dataclass(frozen=True, slots=True)
class LiveEpochEvaluationSubject:
    """Opaque identity for one in-memory training snapshot."""

    profile_digest: str
    epoch: int
    global_step: int
    weights: CheckpointWeightVariant


def _metric_document(spec: MetricSpec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "name": spec.name,
        "channel": spec.channel,
        "params": deepcopy(spec.params),
    }


def _component_document(component: ComponentConfig) -> dict[str, Any]:
    return {
        "name": component.name,
        "params": deepcopy(component.params),
    }


class EvaluationBackedEpochValidator(EpochValidationEvaluator):
    """Run one complete configured Evaluation against a live raw/EMA model."""

    def __init__(
        self,
        *,
        trainer: Trainer,
        config: ValidationEvaluationConfig,
        validation_data: Iterable[Any],
        data_identity: Mapping[str, Any],
        model_factory: ModelFactory | None = None,
        registries: RegistryCatalog = REGISTRIES,
    ) -> None:
        if not isinstance(cast(object, trainer), Trainer):
            raise TypeError("epoch Evaluation trainer must be Trainer")
        plan = trainer.plan
        if not isinstance(cast(object, config), ValidationEvaluationConfig):
            raise TypeError(
                "epoch Evaluation config must be ValidationEvaluationConfig"
            )
        if not config.enabled:
            raise ValueError("epoch Evaluation config must be enabled")
        if config.evaluation is None or config.protocol is None:
            raise ValueError(
                "epoch Evaluation requires an evaluation builder and protocol"
            )
        if not config.protocol.strict_complete:
            raise ValueError(
                "epoch Evaluation protocol must require strict completeness"
            )
        if plan.inference_recipe is None:
            raise ValueError(
                "epoch Evaluation requires a TrainingPlan inference recipe"
            )
        recipe = plan.inference_recipe
        if config.weights not in {"raw", "ema"}:
            raise ValueError("epoch Evaluation weights must be raw or ema")
        weights = cast(CheckpointWeightVariant, config.weights)
        if config.weights == "ema" and trainer.ema is None:
            raise ValueError("EMA epoch Evaluation requires Trainer EMA state")
        if iter(validation_data) is validation_data:
            raise TypeError(
                "epoch Evaluation validation data must be re-iterable"
            )
        if not isinstance(cast(object, data_identity), Mapping):
            raise TypeError("epoch Evaluation data identity must be a mapping")
        if plan.inference_assets and model_factory is None:
            raise ValueError(
                "epoch Evaluation inference assets require a model factory"
            )
        if not isinstance(cast(object, registries), RegistryCatalog):
            raise TypeError("epoch Evaluation registries must be RegistryCatalog")

        declaration = ComponentConfig(
            name=config.evaluation.name,
            params=deepcopy(config.evaluation.params),
        )
        metric_specs = tuple(
            MetricSpec(
                id=spec.id,
                name=spec.name,
                channel=spec.channel,
                params=deepcopy(spec.params),
            )
            for spec in config.metrics
        )
        protocol = EvaluationProtocol(
            id=config.protocol.id,
            expected_examples=config.protocol.expected_examples,
            strict_complete=config.protocol.strict_complete,
        )
        cadence = EpochValidationCadence(
            first_epoch=config.start_epoch,
            every_n_epochs=config.every_epochs,
            include_final=config.include_final,
        )
        data_identity_snapshot = deepcopy(dict(data_identity))

        self._trainer = trainer
        self._plan = plan
        self._recipe = recipe
        self._declaration = declaration
        self._metric_specs = metric_specs
        self._protocol = protocol
        self._validation_data = validation_data
        self._data_identity = data_identity_snapshot
        self._weights: CheckpointWeightVariant = weights
        self._model_factory = model_factory
        self._registries = registries

        identity_subject = LiveEpochEvaluationSubject(
            profile_digest="0" * 64,
            epoch=0,
            global_step=0,
            weights=weights,
        )
        with preserve_global_rng_state(trainer.device):
            primary_was_training = bool(plan.primary_model.training)
            try:
                plan.primary_model.eval()
                identity_plan = self._build_evaluation_plan(
                    subject=identity_subject,
                    sampling=self._sampling_capability(
                        inference_assets=self._inference_assets()
                    ),
                )
            finally:
                plan.primary_model.train(primary_was_training)
        protocol_implementation = build_protocol_implementation_identity(
            evaluation_builder_name=declaration.name,
            metric_specs=metric_specs,
            declared=identity_plan.protocol_identity,
            evaluation_builder_registry=registries.evaluation_builders,
            metric_registry=registries.metrics,
            runtime_parameters={},
        )
        profile_digest = canonical_sha256(
            {
                "schema_version": 2,
                "weights": weights,
                "evaluation": _component_document(declaration),
                "metrics": [
                    _metric_document(spec) for spec in metric_specs
                ],
                "protocol": {
                    "id": protocol.id,
                    "expected_examples": protocol.expected_examples,
                    "strict_complete": protocol.strict_complete,
                },
                "data": data_identity_snapshot,
                "inference_recipe": sampling_recipe_to_dict(
                    recipe
                ),
                "inference_assets": (
                    inference_asset_descriptors_from_projections(
                        plan.inference_assets
                    )
                ),
                "protocol_implementation": protocol_implementation,
            }
        )
        self._protocol_identity = identity_plan.protocol_identity
        self._identity = EpochValidationIdentity(
            profile_digest=profile_digest,
            metric_keys=tuple(config.metric_keys),
            cadence=cadence,
        )

    @property
    def identity(self) -> EpochValidationIdentity:
        """Return the immutable live Evaluation profile identity."""

        return self._identity

    def _inference_assets(self) -> InferenceAssetProvider:
        if not self._plan.inference_assets:
            return InferenceAssetProvider.empty()
        assert self._model_factory is not None
        descriptors = inference_asset_descriptors_from_projections(
            self._plan.inference_assets
        )
        training_asset_names = {
            projection.training_asset_name
            for projection in self._plan.inference_assets.values()
        }
        states = {
            name: self._plan.auxiliary_modules[name].module.state_dict()
            for name in training_asset_names
        }
        return InferenceAssetProvider(
            descriptors=descriptors,
            state_dicts=states,
            device=self._trainer.device,
            model_factory=self._model_factory,
        )

    def _sampling_capability(
        self,
        *,
        inference_assets: InferenceAssetProvider,
    ) -> LiveEvaluationSamplingCapability:
        return LiveEvaluationSamplingCapability(
            recipe=self._recipe,
            process=self._plan.process,
            model=self._plan.primary_model,
            resolved_weights=self._weights,
            device=self._trainer.device,
            inference_assets=inference_assets,
            sampling_builder_registry=self._registries.sampling_builders,
        )

    def _build_evaluation_plan(
        self,
        *,
        subject: LiveEpochEvaluationSubject,
        sampling: LiveEvaluationSamplingCapability,
    ) -> EvaluationPlan:
        return build_evaluation_plan(
            self._declaration,
            subject=subject,
            data=self._validation_data,
            data_identity=self._data_identity,
            inference=self._plan.primary_model,
            metric_specs=self._metric_specs,
            protocol=self._protocol,
            sampling=sampling,
            registries=self._registries,
        )

    def evaluate(
        self,
        *,
        epoch: int,
        global_step: int,
    ) -> EpochValidationResult:
        """Generate validation samples, compute metrics, and restore training."""

        modules: list[nn.Module] = []
        modes: list[tuple[nn.Module, bool]] = []
        seen: set[int] = set()
        for asset in self._trainer.managed_modules.values():
            module = asset.module
            if id(module) in seen:
                continue
            seen.add(id(module))
            modules.append(module)
            modes.append((module, bool(module.training)))

        ema_stored = False
        try:
            if self._weights == "ema":
                assert self._trainer.ema is not None
                self._trainer.ema.store(self._trainer.ema_model)
                ema_stored = True
                self._trainer.ema.copy_to(self._trainer.ema_model)
            for module in modules:
                module.eval()

            sampling = self._sampling_capability(
                inference_assets=self._inference_assets(),
            )
            subject = LiveEpochEvaluationSubject(
                profile_digest=self.identity.profile_digest,
                epoch=epoch,
                global_step=global_step,
                weights=self._weights,
            )
            evaluation_plan = self._build_evaluation_plan(
                subject=subject,
                sampling=sampling,
            )
            if evaluation_plan.protocol_identity != self._protocol_identity:
                raise ValueError(
                    "epoch Evaluation protocol identity changed after preflight"
                )
            facts = execute_evaluation_plan(
                evaluation_plan,
                device=self._trainer.device,
                metric_registry=self._registries.metrics,
            )
            if facts.status != "complete":
                raise ValueError(
                    "epoch validation Evaluation must be complete before its "
                    "metrics can select a checkpoint"
                )
            metrics: dict[str, float] = {}
            prefix = "eval/metrics/"
            for name, value in facts.metrics.items():
                if not name.startswith(prefix):
                    raise ValueError(
                        "epoch validation Evaluation returned a non-canonical "
                        f"metric key {name!r}"
                    )
                metrics[f"valid/metrics/{name.removeprefix(prefix)}"] = value
            return EpochValidationResult(
                epoch=epoch,
                global_step=global_step,
                metrics=metrics,
            )
        finally:
            try:
                if ema_stored:
                    assert self._trainer.ema is not None
                    self._trainer.ema.restore(self._trainer.ema_model)
            finally:
                for module, was_training in modes:
                    module.train(was_training)


__all__ = [
    "EvaluationBackedEpochValidator",
    "LiveEpochEvaluationSubject",
]

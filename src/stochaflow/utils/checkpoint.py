"""Checkpoint save/load helpers."""

import math
import os
import random
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, cast

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.plugins import parse_extension_plugin_provenance
from stochaflow.utils.sampling_recipe import (
    SamplingRecipe,
    sampling_recipe_from_dict,
    sampling_recipe_to_dict,
    validate_sampling_recipe,
)

if TYPE_CHECKING:
    from stochaflow.training.ema import EMAStateDict, ExponentialMovingAverage
else:
    EMAStateDict = dict[str, Any]
    ExponentialMovingAverage = Any


CHECKPOINT_FORMAT_VERSION = 11
_PRECISION_KINDS = frozenset(("fp32", "bf16-mixed", "fp16-mixed"))

_CHECKPOINT_LEAF_TYPES = (type(None), bool, int, float, complex, str, bytes)


class StateDictWithMetadata(Protocol):
    _metadata: Any


class InferenceAssetDeclaration(TypedDict):
    """Reconstructable component identity for one embedded inference asset."""

    name: str
    params: dict[str, Any]


class InferenceAssetDescriptor(TypedDict):
    """Checkpoint projection of a managed asset needed during inference."""

    training_asset_name: str
    declaration: InferenceAssetDeclaration
    capability_role: str
    persistence: Literal["embedded_state"]


class InferenceAssetProjectionSource(Protocol):
    """Training-side values required to build one checkpoint descriptor."""

    @property
    def training_asset_name(self) -> str:
        """Return the managed auxiliary-module name."""

        ...

    @property
    def declaration(self) -> ComponentConfig:
        """Return the reconstruction-only component declaration."""

        ...

    @property
    def capability_role(self) -> str:
        """Return the semantic role expected by the sampling builder."""

        ...


class CheckpointState(TypedDict, total=False):
    """Serialized checkpoint payload."""

    format_version: int
    epoch: int
    global_step: int
    model_state_dict: dict[str, Any]
    process_state_dict: dict[str, Any]
    objective_state_dict: dict[str, Any]
    training_assets_state_dict: dict[str, dict[str, Any]]
    ema_model_state_dict: dict[str, Any]
    optimizer_class: str
    optimizer_state_dict: dict[str, Any]
    lr_scheduler_class: str
    lr_scheduler_state_dict: dict[str, Any]
    ema_state_dict: EMAStateDict
    precision_kind: str
    grad_scaler_class: str
    grad_scaler_state_dict: dict[str, Any]
    inference_asset_descriptors: dict[str, InferenceAssetDescriptor]
    inference_recipe: dict[str, Any] | None
    rng_state: dict[str, Any]
    config: dict[str, Any]
    metrics: dict[str, float]
    metadata: dict[str, Any]


@dataclass(slots=True)
class LoadedCheckpoint:
    """Structured result returned by ``load_checkpoint``."""

    path: Path
    epoch: int | None = None
    global_step: int | None = None
    config: dict[str, Any] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedRNGState:
    """Validated process-global RNG state ready for strict resume."""

    python: tuple[int, tuple[int, ...], float | None]
    numpy: tuple[str, np.ndarray[Any, np.dtype[np.uint32]], int, int, float]
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...]
    torch_mps: torch.Tensor | None


@dataclass(slots=True)
class CheckpointRuntimeSnapshot:
    """Rollback state for one transactional checkpoint restore."""

    model_state: OrderedDict[str, Any]
    process_state: OrderedDict[str, Any] | None
    objective_state: OrderedDict[str, Any] | None
    auxiliary_states: dict[str, OrderedDict[str, Any]]
    optimizer_state: dict[str, Any] | None
    lr_scheduler_state: dict[str, Any] | None
    ema_state: EMAStateDict | None
    grad_scaler_state: dict[str, Any] | None


def validate_inference_asset_descriptors(
    value: object,
    *,
    path: str = "inference_asset_descriptors",
) -> dict[str, InferenceAssetDescriptor]:
    """Validate and detach the embedded-inference-asset schema."""

    if type(value) is not dict:
        raise TypeError(f"{path} must be an exact dictionary")
    raw_descriptors = cast(dict[object, object], value)
    descriptors: dict[str, InferenceAssetDescriptor] = {}
    projected_training_assets: dict[str, str] = {}
    for slot_value, descriptor_value in raw_descriptors.items():
        if type(slot_value) is not str:
            raise TypeError(f"{path} slot names must be exact strings")
        slot = cast(str, slot_value)
        if not slot:
            raise ValueError(f"{path} slot names must be non-empty")
        if slot != slot.strip():
            raise ValueError(
                f"{path} slot names must not contain surrounding whitespace"
            )
        descriptor_path = f"{path}[{slot!r}]"
        descriptor = _exact_dict(
            descriptor_value,
            path=descriptor_path,
            fields={
                "training_asset_name",
                "declaration",
                "capability_role",
                "persistence",
            },
        )
        training_asset_name = _nonempty_exact_string(
            descriptor["training_asset_name"],
            path=f"{descriptor_path}.training_asset_name",
        )
        capability_role = _nonempty_exact_string(
            descriptor["capability_role"],
            path=f"{descriptor_path}.capability_role",
        )
        persistence_value = _nonempty_exact_string(
            descriptor["persistence"],
            path=f"{descriptor_path}.persistence",
        )
        if persistence_value != "embedded_state":
            raise ValueError(
                f"{descriptor_path}.persistence must be 'embedded_state'"
            )
        declaration_path = f"{descriptor_path}.declaration"
        declaration = _exact_dict(
            descriptor["declaration"],
            path=declaration_path,
            fields={"name", "params"},
        )
        declaration_name = _nonempty_exact_string(
            declaration["name"],
            path=f"{declaration_path}.name",
        )
        previous_slot = projected_training_assets.setdefault(
            training_asset_name,
            slot,
        )
        if previous_slot != slot:
            raise ValueError(
                f"{path} cannot reference training asset "
                f"{training_asset_name!r} from more than one slot "
                f"({previous_slot!r} and {slot!r})"
            )
        params_value = declaration["params"]
        if type(params_value) is not dict:
            raise TypeError(f"{declaration_path}.params must be an exact dictionary")
        _validate_checkpoint_value(
            params_value,
            path=f"{declaration_path}.params",
        )
        descriptors[slot] = {
            "training_asset_name": training_asset_name,
            "declaration": {
                "name": declaration_name,
                "params": deepcopy(cast(dict[str, Any], params_value)),
            },
            "capability_role": capability_role,
            "persistence": "embedded_state",
        }
    return descriptors


def inference_asset_descriptors_from_projections(
    projections: Mapping[str, InferenceAssetProjectionSource],
) -> dict[str, InferenceAssetDescriptor]:
    """Convert validated training projections into checkpoint descriptors."""

    descriptors: dict[str, InferenceAssetDescriptor] = {
        slot: {
            "training_asset_name": projection.training_asset_name,
            "declaration": {
                "name": projection.declaration.name,
                "params": dict(projection.declaration.params),
            },
            "capability_role": projection.capability_role,
            "persistence": "embedded_state",
        }
        for slot, projection in projections.items()
    }
    return validate_inference_asset_descriptors(
        descriptors,
        path="TrainingPlan inference asset descriptors",
    )


def inference_asset_descriptors_equal(
    left: Mapping[str, InferenceAssetDescriptor],
    right: Mapping[str, InferenceAssetDescriptor],
) -> bool:
    """Compare descriptors without ambiguous Tensor equality."""

    return _checkpoint_values_equal(left, right)


@dataclass(slots=True)
class CheckpointManager:
    """Object-oriented checkpoint IO wrapper for a training stack.

    A checkpoint manager owns references to the stateful runtime objects that
    participate in checkpointing and exposes cohesive save/load helpers around
    that state.
    """

    model: nn.Module
    process: nn.Module | None = None
    objective: nn.Module | None = None
    auxiliary_modules: dict[str, nn.Module] = field(default_factory=dict)
    optimizer: Optimizer | None = None
    lr_scheduler: LRScheduler | None = None
    ema: ExponentialMovingAverage | None = None
    precision_kind: str = "fp32"
    grad_scaler: torch.cuda.amp.GradScaler | None = None
    inference_asset_descriptors: dict[str, InferenceAssetDescriptor] = field(
        default_factory=dict
    )
    inference_recipe: SamplingRecipe | None = None

    def __post_init__(self) -> None:
        """Validate the runtime precision assets owned by this manager."""

        self.precision_kind = _validate_precision_kind(
            self.precision_kind,
            path="checkpoint manager precision_kind",
        )
        scaler_value = cast(object, self.grad_scaler)
        if scaler_value is not None and not isinstance(
            scaler_value,
            torch.cuda.amp.GradScaler,
        ):
            raise TypeError(
                "checkpoint manager grad_scaler must be "
                "torch.cuda.amp.GradScaler or None"
            )
        if self.precision_kind == "fp16-mixed":
            if scaler_value is None:
                raise ValueError(
                    "fp16-mixed checkpoint manager requires a GradScaler"
                )
            if not scaler_value.is_enabled():
                raise ValueError(
                    "fp16-mixed checkpoint manager requires an enabled GradScaler"
                )
            _validate_grad_scaler_scale(
                scaler_value.get_scale(),
                path="checkpoint manager GradScaler scale",
            )
        elif scaler_value is not None:
            raise ValueError(
                f"{self.precision_kind} checkpoint manager cannot use a GradScaler"
            )
        self.inference_asset_descriptors = validate_inference_asset_descriptors(
            self.inference_asset_descriptors,
            path="checkpoint manager inference_asset_descriptors",
        )
        self.inference_recipe = (
            validate_sampling_recipe(
                self.inference_recipe,
                path="checkpoint manager inference_recipe",
            )
            if self.inference_recipe is not None
            else None
        )
        missing_assets = sorted(
            {
                descriptor["training_asset_name"]
                for descriptor in self.inference_asset_descriptors.values()
            }
            - set(self.auxiliary_modules)
        )
        if missing_assets:
            raise ValueError(
                "checkpoint manager inference asset descriptors reference "
                "missing training assets: "
                + ", ".join(missing_assets)
            )

    @staticmethod
    def find_best(root: str | Path) -> Path:
        """Find the default inference checkpoint under a run or output root."""

        return _find_named_checkpoint(root, "best.pt")

    @staticmethod
    def find_latest(root: str | Path) -> Path:
        """Find the default resume checkpoint under a run or output root."""

        return _find_named_checkpoint(root, "latest.pt")

    def build_state(
        self,
        *,
        epoch: int | None = None,
        global_step: int | None = None,
        config: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CheckpointState:
        """Assemble a serializable checkpoint payload from managed objects."""

        state: CheckpointState = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_state_dict": _clone_module_state(self.model),
            "precision_kind": self.precision_kind,
            "inference_asset_descriptors": deepcopy(
                self.inference_asset_descriptors
            ),
            "inference_recipe": (
                sampling_recipe_to_dict(self.inference_recipe)
                if self.inference_recipe is not None
                else None
            ),
            "rng_state": capture_rng_state(),
        }
        if self.process is not None:
            state["process_state_dict"] = _clone_module_state(self.process)
        if self.objective is not None:
            state["objective_state_dict"] = _clone_module_state(self.objective)
        state["training_assets_state_dict"] = {
            name: _clone_module_state(module)
            for name, module in self.auxiliary_modules.items()
        }
        ema_state: EMAStateDict | None = None
        if self.ema is not None:
            ema_state = cast(
                EMAStateDict,
                _clone_checkpoint_data(
                    self.ema.state_dict(),
                    tensors_to_cpu=True,
                ),
            )
            state["ema_model_state_dict"] = _project_ema_module_state(
                self.model,
                state["model_state_dict"],
                ema_state,
            )
        if self.optimizer is not None:
            state["optimizer_class"] = _type_identity(self.optimizer)
            state["optimizer_state_dict"] = cast(
                dict[str, Any],
                _clone_checkpoint_data(
                    self.optimizer.state_dict(),
                    tensors_to_cpu=False,
                ),
            )
        if self.lr_scheduler is not None:
            state["lr_scheduler_class"] = _type_identity(self.lr_scheduler)
            state["lr_scheduler_state_dict"] = cast(
                dict[str, Any],
                _clone_checkpoint_data(
                    self.lr_scheduler.state_dict(),
                    tensors_to_cpu=False,
                ),
            )
        if ema_state is not None:
            state["ema_state_dict"] = ema_state
        if self.grad_scaler is not None:
            state["grad_scaler_class"] = _type_identity(self.grad_scaler)
            state["grad_scaler_state_dict"] = cast(
                dict[str, Any],
                _clone_checkpoint_data(
                    self.grad_scaler.state_dict(),
                    tensors_to_cpu=False,
                ),
            )
        if epoch is not None:
            state["epoch"] = epoch
        if global_step is not None:
            state["global_step"] = global_step
        if config is not None:
            state["config"] = cast(
                dict[str, Any],
                _clone_checkpoint_data(
                    config,
                    tensors_to_cpu=False,
                ),
            )
        if metrics is not None:
            state["metrics"] = cast(
                dict[str, float],
                _clone_checkpoint_data(
                    metrics,
                    tensors_to_cpu=False,
                ),
            )
        checkpoint_metadata = cast(
            dict[str, Any],
            _clone_checkpoint_data(
                metadata or {},
                tensors_to_cpu=False,
            ),
        )
        checkpoint_metadata.setdefault("extension_plugins", [])
        state["metadata"] = checkpoint_metadata
        _validate_checkpoint_value(state, path="checkpoint")
        _validate_v11_checkpoint_header(state)
        return state

    def save(
        self,
        path: str | Path,
        *,
        epoch: int | None = None,
        global_step: int | None = None,
        config: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save a checkpoint to disk."""

        checkpoint_path = Path(path)
        state = self.build_state(
            epoch=epoch,
            global_step=global_step,
            config=config,
            metrics=metrics,
            metadata=metadata,
        )
        return self.save_payload(state, checkpoint_path)

    @staticmethod
    def save_payload(payload: object, path: str | Path) -> Path:
        """Atomically publish one already assembled data-only checkpoint payload."""

        checkpoint_path = Path(path)
        if type(payload) is not dict:
            raise TypeError("checkpoint payload must be an exact dictionary")
        _validate_checkpoint_value(payload, path="checkpoint")
        state = cast(CheckpointState, payload)
        _validate_v11_checkpoint_header(state)
        _ensure_parent_directory(checkpoint_path)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
            dir=checkpoint_path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            torch.save(state, temporary_path)
            temporary_path.replace(checkpoint_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return checkpoint_path

    def load(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> LoadedCheckpoint:
        """Load checkpoint state into the managed runtime objects."""

        checkpoint_path = Path(path)
        state = self.load_payload(checkpoint_path, map_location=map_location)
        return self.restore_payload(state, path=checkpoint_path)

    def restore_payload(
        self,
        payload: object,
        *,
        path: str | Path,
    ) -> LoadedCheckpoint:
        """Restore managed objects from an already loaded checkpoint payload.

        Ordinary failures are transactional. If a runtime object's custom load
        hook also fails while applying the rollback snapshot, the object must be
        treated as poisoned; the raised ``RuntimeError`` retains both failures.
        """

        checkpoint_path = Path(path)
        if type(payload) is not dict:
            raise TypeError(
                f"checkpoint at '{checkpoint_path}' must contain a dictionary payload"
        )
        _validate_checkpoint_value(payload, path="checkpoint")
        state = _normalize_checkpoint_header(cast(CheckpointState, payload))
        checkpoint_descriptors = validate_inference_asset_descriptors(
            state.get("inference_asset_descriptors"),
            path="checkpoint.inference_asset_descriptors",
        )
        if not inference_asset_descriptors_equal(
            checkpoint_descriptors,
            self.inference_asset_descriptors,
        ):
            raise ValueError(
                "checkpoint inference asset descriptors do not match runtime"
            )
        checkpoint_recipe_value = state.get("inference_recipe")
        checkpoint_recipe = (
            sampling_recipe_from_dict(checkpoint_recipe_value)
            if checkpoint_recipe_value is not None
            else None
        )
        if checkpoint_recipe != self.inference_recipe:
            raise ValueError("checkpoint inference recipe does not match runtime")
        model_state_dict = cast(object, state.get("model_state_dict"))
        if not isinstance(model_state_dict, dict):
            raise TypeError("checkpoint is missing a valid model_state_dict")
        has_process_state = "process_state_dict" in state
        process_state_dict = cast(object, state.get("process_state_dict"))
        validated_process_state: dict[str, Any] | None = None
        if self.process is None:
            if has_process_state:
                raise ValueError(
                    "checkpoint contains process_state_dict but runtime has no process"
                )
        else:
            if not has_process_state:
                raise TypeError("checkpoint is missing process_state_dict")
            if not isinstance(process_state_dict, dict):
                raise TypeError("process_state_dict must be a dictionary")
            validated_process_state = process_state_dict

        has_objective_state = "objective_state_dict" in state
        objective_state_dict = cast(object, state.get("objective_state_dict"))
        validated_objective_state: dict[str, Any] | None = None
        if self.objective is None:
            if has_objective_state:
                raise ValueError(
                    "checkpoint contains objective_state_dict but runtime has no "
                    "objective"
                )
        else:
            if not has_objective_state:
                raise TypeError("checkpoint is missing objective_state_dict")
            if not isinstance(objective_state_dict, dict):
                raise TypeError("objective_state_dict must be a dictionary")
            validated_objective_state = objective_state_dict

        assets_value = cast(object, state.get("training_assets_state_dict"))
        if not isinstance(assets_value, dict):
            raise TypeError("checkpoint is missing training_assets_state_dict")
        expected_assets = set(self.auxiliary_modules)
        actual_assets = set(assets_value)
        if expected_assets != actual_assets:
            missing = sorted(expected_assets - actual_assets)
            unexpected = sorted(actual_assets - expected_assets)
            details: list[str] = []
            if missing:
                details.append("missing: " + ", ".join(missing))
            if unexpected:
                details.append("unexpected: " + ", ".join(unexpected))
            raise ValueError(
                "checkpoint training asset names do not match runtime ("
                + "; ".join(details)
                + ")"
            )
        validated_assets: dict[str, dict[str, Any]] = {}
        for name, asset_state in assets_value.items():
            if not isinstance(name, str) or not isinstance(asset_state, dict):
                raise TypeError(
                    "training_assets_state_dict must map strings to dictionaries"
                )
            validated_assets[name] = asset_state

        has_optimizer_state = "optimizer_state_dict" in state
        optimizer_state_dict = cast(object, state.get("optimizer_state_dict"))
        optimizer_class = cast(object, state.get("optimizer_class"))
        validated_optimizer_state: dict[str, Any] | None = None
        if self.optimizer is None:
            if has_optimizer_state or optimizer_class is not None:
                raise ValueError(
                    "checkpoint contains optimizer state but runtime has no optimizer"
                )
        else:
            if not has_optimizer_state:
                raise TypeError("checkpoint is missing optimizer_state_dict")
            if not isinstance(optimizer_state_dict, dict):
                raise TypeError("optimizer_state_dict must be a dictionary when provided")
            expected_optimizer_class = _type_identity(self.optimizer)
            if optimizer_class != expected_optimizer_class:
                raise ValueError(
                    "checkpoint optimizer class does not match runtime: "
                    f"{optimizer_class!r} != {expected_optimizer_class!r}"
                )
            validated_optimizer_state = optimizer_state_dict
            _validate_optimizer_state_dict_compatibility(
                self.optimizer,
                optimizer_state_dict,
            )

        has_lr_scheduler_state = "lr_scheduler_state_dict" in state
        lr_scheduler_state_dict = cast(object, state.get("lr_scheduler_state_dict"))
        lr_scheduler_class = cast(object, state.get("lr_scheduler_class"))
        validated_lr_scheduler_state: dict[str, Any] | None = None
        if self.lr_scheduler is None:
            if has_lr_scheduler_state or lr_scheduler_class is not None:
                raise ValueError(
                    "checkpoint contains lr_scheduler_state_dict but runtime has no "
                    "lr scheduler"
                )
        else:
            if not has_lr_scheduler_state:
                raise TypeError("checkpoint is missing lr_scheduler_state_dict")
            if not isinstance(lr_scheduler_state_dict, dict):
                raise TypeError(
                    "lr_scheduler_state_dict must be a dictionary when provided"
                )
            expected_lr_scheduler_class = _type_identity(self.lr_scheduler)
            if lr_scheduler_class != expected_lr_scheduler_class:
                raise ValueError(
                    "checkpoint lr scheduler class does not match runtime: "
                    f"{lr_scheduler_class!r} != {expected_lr_scheduler_class!r}"
                )
            validated_lr_scheduler_state = lr_scheduler_state_dict
            _validate_lr_scheduler_state_dict_compatibility(
                self.lr_scheduler,
                lr_scheduler_state_dict,
            )

        has_ema_state = "ema_state_dict" in state
        has_ema_model_state = "ema_model_state_dict" in state
        ema_state_dict = cast(object, state.get("ema_state_dict"))
        ema_model_state_dict = cast(object, state.get("ema_model_state_dict"))
        validated_ema_state: EMAStateDict | None = None
        if self.ema is None:
            if has_ema_state or has_ema_model_state:
                raise ValueError(
                    "checkpoint contains EMA state but runtime has no EMA"
                )
        else:
            if not has_ema_state or not has_ema_model_state:
                missing = [
                    name
                    for name, present in (
                        ("ema_state_dict", has_ema_state),
                        ("ema_model_state_dict", has_ema_model_state),
                    )
                    if not present
                ]
                raise TypeError(
                    "checkpoint is missing EMA field(s): " + ", ".join(missing)
                )
            if not isinstance(ema_state_dict, dict):
                raise TypeError("ema_state_dict must be a dictionary when provided")
            if not isinstance(ema_model_state_dict, dict):
                raise TypeError(
                    "ema_model_state_dict must be a dictionary when provided"
                )
            validated_ema_state = _validate_ema_state_dict(
                self.ema,
                ema_state_dict,
            )
            _validate_ema_model_consistency(
                self.model,
                model_state_dict,
                ema_model_state_dict,
                validated_ema_state,
            )

        epoch = cast(object, state.get("epoch"))
        if epoch is not None and type(epoch) is not int:
            raise TypeError("epoch must be an exact int when provided")
        global_step = cast(object, state.get("global_step"))
        if global_step is not None and type(global_step) is not int:
            raise TypeError("global_step must be an exact int when provided")

        config = cast(object, state.get("config"))
        if config is not None and type(config) is not dict:
            raise TypeError("config must be an exact dictionary when provided")
        metrics = cast(object, state.get("metrics"))
        if metrics is None:
            metrics = {}
        elif type(metrics) is not dict:
            raise TypeError("metrics must be an exact dictionary when provided")
        metadata = cast(object, state.get("metadata"))
        if type(metadata) is not dict:
            raise TypeError("checkpoint is missing valid metadata")

        validate_module_state_dict_compatibility(
            self.model,
            model_state_dict,
            path="checkpoint.model_state_dict",
        )
        if self.process is not None:
            assert validated_process_state is not None
            validate_module_state_dict_compatibility(
                self.process,
                validated_process_state,
                path="checkpoint.process_state_dict",
            )
        if self.objective is not None:
            assert validated_objective_state is not None
            validate_module_state_dict_compatibility(
                self.objective,
                validated_objective_state,
                path="checkpoint.objective_state_dict",
            )
        for name, module in self.auxiliary_modules.items():
            validate_module_state_dict_compatibility(
                module,
                validated_assets[name],
                path=f"checkpoint.training_assets_state_dict[{name!r}]",
            )
        if self.ema is not None:
            assert isinstance(ema_model_state_dict, dict)
            validate_module_state_dict_compatibility(
                self.model,
                ema_model_state_dict,
                path="checkpoint.ema_model_state_dict",
            )
        validated_scaler_state = self._validate_precision_topology(state)
        snapshot = self._capture_runtime_snapshot()
        prepared_model_state = cast(
            Mapping[str, Any],
            _clone_checkpoint_data(
                model_state_dict,
                tensors_to_cpu=False,
            ),
        )
        prepared_process_state = (
            None
            if validated_process_state is None
            else cast(
                Mapping[str, Any],
                _clone_checkpoint_data(
                    validated_process_state,
                    tensors_to_cpu=False,
                ),
            )
        )
        prepared_objective_state = (
            None
            if validated_objective_state is None
            else cast(
                Mapping[str, Any],
                _clone_checkpoint_data(
                    validated_objective_state,
                    tensors_to_cpu=False,
                ),
            )
        )
        prepared_assets = {
            name: cast(
                Mapping[str, Any],
                _clone_checkpoint_data(
                    asset_state,
                    tensors_to_cpu=False,
                ),
            )
            for name, asset_state in validated_assets.items()
        }
        prepared_optimizer_state = (
            None
            if validated_optimizer_state is None
            else cast(
                dict[str, Any],
                _clone_checkpoint_data(
                    validated_optimizer_state,
                    tensors_to_cpu=False,
                ),
            )
        )
        prepared_lr_scheduler_state = (
            None
            if validated_lr_scheduler_state is None
            else cast(
                dict[str, Any],
                _clone_checkpoint_data(
                    validated_lr_scheduler_state,
                    tensors_to_cpu=False,
                ),
            )
        )
        prepared_ema_state = (
            None
            if self.ema is None or validated_ema_state is None
            else _prepare_ema_state_for_runtime(
                self.ema,
                validated_ema_state,
            )
        )
        prepared_scaler_state = (
            None
            if validated_scaler_state is None
            else cast(
                dict[str, Any],
                _clone_checkpoint_data(
                    validated_scaler_state,
                    tensors_to_cpu=False,
                ),
            )
        )
        try:
            self.model.load_state_dict(prepared_model_state)
            if self.process is not None:
                assert prepared_process_state is not None
                self.process.load_state_dict(prepared_process_state)
            if self.objective is not None:
                assert prepared_objective_state is not None
                self.objective.load_state_dict(prepared_objective_state)
            for name, module in self.auxiliary_modules.items():
                module.load_state_dict(prepared_assets[name])

            if self.optimizer is not None:
                assert prepared_optimizer_state is not None
                self.optimizer.load_state_dict(prepared_optimizer_state)
            if self.lr_scheduler is not None:
                assert prepared_lr_scheduler_state is not None
                self.lr_scheduler.load_state_dict(prepared_lr_scheduler_state)
            if self.ema is not None:
                assert prepared_ema_state is not None
                self.ema.load_state_dict(prepared_ema_state)
            if self.grad_scaler is not None:
                assert prepared_scaler_state is not None
                self.grad_scaler.load_state_dict(prepared_scaler_state)
        except BaseException as restore_error:
            try:
                self._restore_runtime_snapshot(snapshot)
            except BaseException as rollback_error:
                raise RuntimeError(
                    "checkpoint restore failed and runtime rollback also failed; "
                    f"original restore error was {restore_error!r}"
                ) from rollback_error
            raise

        return LoadedCheckpoint(
            path=checkpoint_path,
            epoch=epoch,
            global_step=global_step,
            config=config,
            metrics=metrics,
            metadata=metadata,
        )

    def _validate_precision_topology(
        self,
        state: CheckpointState,
    ) -> dict[str, Any] | None:
        checkpoint_precision = _validate_precision_kind(
            state.get("precision_kind"),
            path="checkpoint.precision_kind",
        )
        if checkpoint_precision != self.precision_kind:
            raise ValueError(
                "checkpoint precision kind does not match runtime: "
                f"{checkpoint_precision!r} != {self.precision_kind!r}"
            )
        scaler_state_value = cast(object, state.get("grad_scaler_state_dict"))
        scaler_class = cast(object, state.get("grad_scaler_class"))
        if self.grad_scaler is None:
            if scaler_class is not None or scaler_state_value is not None:
                raise ValueError(
                    "checkpoint contains GradScaler state but runtime has no "
                    "GradScaler"
                )
            return None
        _validate_grad_scaler_scale(
            self.grad_scaler.get_scale(),
            path="runtime GradScaler scale",
        )
        if scaler_class is None or scaler_state_value is None:
            raise TypeError(
                "checkpoint is missing GradScaler state required by the runtime"
            )
        if scaler_class != _type_identity(self.grad_scaler):
            raise ValueError(
                "checkpoint GradScaler class does not match runtime: "
                f"{scaler_class!r} != {_type_identity(self.grad_scaler)!r}"
            )
        if type(scaler_state_value) is not dict:
            raise TypeError(
                "checkpoint grad_scaler_state_dict must be an exact dictionary"
            )
        validated_state = cast(dict[str, Any], scaler_state_value)
        temporary_scaler = deepcopy(self.grad_scaler)
        temporary_scaler.load_state_dict(deepcopy(validated_state))
        _validate_grad_scaler_scale(
            temporary_scaler.get_scale(),
            path="checkpoint GradScaler restored scale",
        )
        return validated_state

    def _capture_runtime_snapshot(self) -> CheckpointRuntimeSnapshot:
        """Capture recoverable state before applying a validated payload."""

        optimizer_state = (
            None
            if self.optimizer is None
            else cast(
                dict[str, Any],
                _clone_checkpoint_data(
                    self.optimizer.state_dict(),
                    tensors_to_cpu=False,
                ),
            )
        )
        lr_scheduler_state = (
            None
            if self.lr_scheduler is None
            else cast(
                dict[str, Any],
                _clone_checkpoint_data(
                    self.lr_scheduler.state_dict(),
                    tensors_to_cpu=False,
                ),
            )
        )
        ema_state = None if self.ema is None else self.ema.state_dict()
        grad_scaler_state = (
            None
            if self.grad_scaler is None
            else deepcopy(self.grad_scaler.state_dict())
        )
        return CheckpointRuntimeSnapshot(
            model_state=_clone_module_state(self.model),
            process_state=(
                None
                if self.process is None
                else _clone_module_state(self.process)
            ),
            objective_state=(
                None
                if self.objective is None
                else _clone_module_state(self.objective)
            ),
            auxiliary_states={
                name: _clone_module_state(module)
                for name, module in self.auxiliary_modules.items()
            },
            optimizer_state=optimizer_state,
            lr_scheduler_state=lr_scheduler_state,
            ema_state=ema_state,
            grad_scaler_state=grad_scaler_state,
        )

    def _restore_runtime_snapshot(
        self,
        snapshot: CheckpointRuntimeSnapshot,
    ) -> None:
        """Roll back every managed runtime asset after a failed restore."""

        self.model.load_state_dict(snapshot.model_state)
        if self.process is not None:
            assert snapshot.process_state is not None
            self.process.load_state_dict(snapshot.process_state)
        if self.objective is not None:
            assert snapshot.objective_state is not None
            self.objective.load_state_dict(snapshot.objective_state)
        for name, module in self.auxiliary_modules.items():
            module.load_state_dict(snapshot.auxiliary_states[name])
        if self.optimizer is not None:
            assert snapshot.optimizer_state is not None
            self.optimizer.load_state_dict(snapshot.optimizer_state)
        if self.lr_scheduler is not None:
            assert snapshot.lr_scheduler_state is not None
            self.lr_scheduler.load_state_dict(snapshot.lr_scheduler_state)
        if self.ema is not None:
            assert snapshot.ema_state is not None
            self.ema.load_state_dict(snapshot.ema_state)
        if self.grad_scaler is not None:
            assert snapshot.grad_scaler_state is not None
            self.grad_scaler.load_state_dict(snapshot.grad_scaler_state)

    @staticmethod
    def load_payload(
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
    ) -> CheckpointState:
        """Read and validate the top-level checkpoint payload."""

        checkpoint_path = Path(path)
        raw_state = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
        if type(raw_state) is not dict:
            raise TypeError(
                f"checkpoint at '{checkpoint_path}' must contain a dictionary payload"
            )
        _validate_checkpoint_value(raw_state, path="checkpoint")
        return _normalize_checkpoint_header(cast(CheckpointState, raw_state))


def capture_rng_state() -> dict[str, Any]:
    """Capture process-global RNG streams using the data-only contract."""

    return _capture_rng_state(include_cuda=True, include_mps=True)


def _capture_rng_state(
    *,
    include_cuda: bool,
    include_mps: bool,
) -> dict[str, Any]:
    python_version, python_state, python_gaussian = random.getstate()
    (
        numpy_generator,
        numpy_keys,
        numpy_position,
        numpy_has_gaussian,
        numpy_cached_gaussian,
    ) = np.random.get_state()
    cuda_states = (
        torch.cuda.get_rng_state_all()
        if include_cuda and torch.cuda.is_available()
        else []
    )
    mps_state = (
        torch.mps.get_rng_state()
        if include_mps and torch.backends.mps.is_available()
        else None
    )
    return {
        "python": {
            "version": python_version,
            "state": list(python_state),
            "gaussian_cache": python_gaussian,
        },
        "numpy": {
            "bit_generator": numpy_generator,
            "keys": np.asarray(numpy_keys, dtype=np.uint32).tolist(),
            "position": numpy_position,
            "has_gaussian": numpy_has_gaussian,
            "cached_gaussian": numpy_cached_gaussian,
        },
        "torch_cpu": torch.random.get_rng_state().detach().cpu().clone(),
        "torch_cuda": [state.detach().cpu().clone() for state in cuda_states],
        "torch_mps": (
            None if mps_state is None else mps_state.detach().cpu().clone()
        ),
    }


def parse_rng_state(
    value: object,
    *,
    require_cuda_compatibility: bool = False,
    require_mps_compatibility: bool = False,
) -> ParsedRNGState:
    """Validate and normalize one RNG snapshot without changing global RNGs."""

    root = _exact_dict(
        value,
        path="checkpoint.rng_state",
        fields={"python", "numpy", "torch_cpu", "torch_cuda"},
        optional_fields={"torch_mps"},
    )
    python_value = _exact_dict(
        root["python"],
        path="checkpoint.rng_state.python",
        fields={"version", "state", "gaussian_cache"},
    )
    python_version = _exact_int(
        python_value["version"],
        path="checkpoint.rng_state.python.version",
    )
    python_words_value = python_value["state"]
    if type(python_words_value) is not list or not python_words_value:
        raise TypeError("checkpoint.rng_state.python.state must be a non-empty list")
    python_words = tuple(
        _exact_int(
            word,
            path=f"checkpoint.rng_state.python.state[{index}]",
        )
        for index, word in enumerate(cast(list[object], python_words_value))
    )
    python_gaussian_value = python_value["gaussian_cache"]
    if python_gaussian_value is not None and type(python_gaussian_value) is not float:
        raise TypeError(
            "checkpoint.rng_state.python.gaussian_cache must be a float or null"
        )
    python_state = (
        python_version,
        python_words,
        cast(float | None, python_gaussian_value),
    )
    try:
        random.Random().setstate(python_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint.rng_state.python is invalid") from exc

    numpy_value = _exact_dict(
        root["numpy"],
        path="checkpoint.rng_state.numpy",
        fields={
            "bit_generator",
            "keys",
            "position",
            "has_gaussian",
            "cached_gaussian",
        },
    )
    numpy_generator = numpy_value["bit_generator"]
    if type(numpy_generator) is not str or not numpy_generator:
        raise TypeError(
            "checkpoint.rng_state.numpy.bit_generator must be a non-empty string"
        )
    numpy_keys_value = numpy_value["keys"]
    if type(numpy_keys_value) is not list or not numpy_keys_value:
        raise TypeError("checkpoint.rng_state.numpy.keys must be a non-empty list")
    numpy_key_values = [
        _exact_int(key, path=f"checkpoint.rng_state.numpy.keys[{index}]")
        for index, key in enumerate(cast(list[object], numpy_keys_value))
    ]
    if any(key < 0 or key > np.iinfo(np.uint32).max for key in numpy_key_values):
        raise ValueError("checkpoint.rng_state.numpy.keys values must fit uint32")
    numpy_keys = np.asarray(numpy_key_values, dtype=np.uint32)
    numpy_position = _exact_int(
        numpy_value["position"],
        path="checkpoint.rng_state.numpy.position",
    )
    numpy_has_gaussian = _exact_int(
        numpy_value["has_gaussian"],
        path="checkpoint.rng_state.numpy.has_gaussian",
    )
    numpy_cached_gaussian = numpy_value["cached_gaussian"]
    if type(numpy_cached_gaussian) is not float:
        raise TypeError(
            "checkpoint.rng_state.numpy.cached_gaussian must be a float"
        )
    numpy_state = (
        cast(str, numpy_generator),
        numpy_keys,
        numpy_position,
        numpy_has_gaussian,
        cast(float, numpy_cached_gaussian),
    )
    try:
        np.random.RandomState().set_state(numpy_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint.rng_state.numpy is invalid") from exc

    torch_cpu = _parse_rng_tensor(
        root["torch_cpu"],
        path="checkpoint.rng_state.torch_cpu",
    )
    try:
        torch.Generator(device="cpu").set_state(torch_cpu)
    except RuntimeError as exc:
        raise ValueError("checkpoint.rng_state.torch_cpu is invalid") from exc

    torch_cuda_value = root["torch_cuda"]
    if type(torch_cuda_value) is not list:
        raise TypeError("checkpoint.rng_state.torch_cuda must be a list")
    torch_cuda = tuple(
        _parse_rng_tensor(
            state,
            path=f"checkpoint.rng_state.torch_cuda[{index}]",
        )
        for index, state in enumerate(cast(list[object], torch_cuda_value))
    )
    if require_cuda_compatibility and torch_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable"
            )
        device_count = torch.cuda.device_count()
        if len(torch_cuda) != device_count:
            raise RuntimeError(
                "checkpoint CUDA RNG device count does not match runtime: "
                f"{len(torch_cuda)} != {device_count}"
            )
        for index, state in enumerate(torch_cuda):
            try:
                torch.Generator(device=f"cuda:{index}").set_state(state)
            except RuntimeError as exc:
                raise ValueError(
                    f"checkpoint.rng_state.torch_cuda[{index}] is invalid"
                ) from exc

    torch_mps_value = root.get("torch_mps")
    torch_mps = (
        None
        if torch_mps_value is None
        else _parse_rng_tensor(
            torch_mps_value,
            path="checkpoint.rng_state.torch_mps",
        )
    )
    if require_mps_compatibility and torch_mps is not None:
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "checkpoint contains MPS RNG state but MPS is unavailable"
            )
        try:
            torch.Generator(device="mps").set_state(torch_mps)
        except RuntimeError as exc:
            raise ValueError("checkpoint.rng_state.torch_mps is invalid") from exc

    return ParsedRNGState(
        python=python_state,
        numpy=numpy_state,
        torch_cpu=torch_cpu,
        torch_cuda=torch_cuda,
        torch_mps=torch_mps,
    )


def restore_rng_state(
    state: object,
    *,
    restore_cuda: bool = True,
    restore_mps: bool = True,
) -> None:
    """Restore a fully parsed RNG snapshot, rolling back on runtime failure."""

    if not isinstance(state, ParsedRNGState):
        raise TypeError("state must be a ParsedRNGState")
    parsed_state = state
    previous = parse_rng_state(
        _capture_rng_state(
            include_cuda=restore_cuda,
            include_mps=restore_mps,
        ),
        require_cuda_compatibility=restore_cuda and bool(parsed_state.torch_cuda),
        require_mps_compatibility=restore_mps
        and parsed_state.torch_mps is not None,
    )
    try:
        _apply_rng_state(
            parsed_state,
            restore_cuda=restore_cuda,
            restore_mps=restore_mps,
        )
    except Exception:
        _apply_rng_state(
            previous,
            restore_cuda=restore_cuda,
            restore_mps=restore_mps,
        )
        raise


def _apply_rng_state(
    state: ParsedRNGState,
    *,
    restore_cuda: bool,
    restore_mps: bool,
) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.random.set_rng_state(state.torch_cpu)
    if restore_cuda and state.torch_cuda:
        torch.cuda.set_rng_state_all(list(state.torch_cuda))
    if restore_mps and state.torch_mps is not None:
        torch.mps.set_rng_state(state.torch_mps)
        if not torch.equal(torch.mps.get_rng_state(), state.torch_mps):
            raise RuntimeError("MPS RNG state did not match after restoration")


def _exact_dict(
    value: object,
    *,
    path: str,
    fields: set[str],
    optional_fields: set[str] | None = None,
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{path} must be an exact dictionary")
    result = cast(dict[object, object], value)
    optional = optional_fields or set()
    actual = set(result)
    if not fields.issubset(actual) or not actual.issubset(fields | optional):
        missing = sorted(fields - set(result))
        unknown = sorted(set(result) - fields - optional, key=str)
        raise ValueError(
            f"{path} has invalid fields: missing={missing or '<none>'}, "
            f"unknown={unknown or '<none>'}"
        )
    if any(type(key) is not str for key in result):
        raise TypeError(f"{path} field names must be strings")
    return cast(dict[str, object], result)


def _exact_int(value: object, *, path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{path} must be an exact integer")
    return cast(int, value)


def _nonempty_exact_string(value: object, *, path: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{path} must be an exact string")
    result = cast(str, value)
    if not result:
        raise ValueError(f"{path} must be non-empty")
    if result != result.strip():
        raise ValueError(f"{path} must not contain surrounding whitespace")
    return result


def _parse_rng_tensor(value: object, *, path: str) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{path} must be an exact Tensor")
    tensor = cast(torch.Tensor, value)
    if tensor.dtype is not torch.uint8 or tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"{path} must be a non-empty one-dimensional uint8 Tensor")
    return tensor.detach().cpu().clone()


def _normalize_checkpoint_header(state: CheckpointState) -> CheckpointState:
    """Validate the only supported checkpoint header."""

    version = cast(object, state.get("format_version"))
    if type(version) is not int:
        raise TypeError("checkpoint format_version must be an exact integer")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format version {version!r} is unsupported; "
            f"expected version {CHECKPOINT_FORMAT_VERSION}"
        )
    _validate_v11_checkpoint_header(state)
    return state


def _validate_v11_checkpoint_header(state: CheckpointState) -> None:
    """Validate the exact inference and precision topology of a v11 checkpoint."""

    version = cast(object, state.get("format_version"))
    if type(version) is not int or version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "checkpoint writer requires format version "
            f"{CHECKPOINT_FORMAT_VERSION}, got {version!r}"
        )
    precision_kind = _validate_precision_kind(
        state.get("precision_kind"),
        path="checkpoint.precision_kind",
    )
    descriptors = validate_inference_asset_descriptors(
        state.get("inference_asset_descriptors"),
        path="checkpoint.inference_asset_descriptors",
    )
    if "inference_recipe" not in state:
        raise TypeError("v11 checkpoint is missing inference_recipe")
    inference_recipe_value = state["inference_recipe"]
    if inference_recipe_value is not None:
        sampling_recipe_from_dict(inference_recipe_value)
    if descriptors:
        assets_value = cast(object, state.get("training_assets_state_dict"))
        if type(assets_value) is not dict:
            raise TypeError(
                "checkpoint with inference asset descriptors requires an exact "
                "training_assets_state_dict"
            )
        assets = cast(dict[object, object], assets_value)
        for slot, descriptor in descriptors.items():
            training_asset_name = descriptor["training_asset_name"]
            if training_asset_name not in assets:
                raise ValueError(
                    "checkpoint inference asset descriptor "
                    f"{slot!r} references missing training asset "
                    f"{training_asset_name!r}"
                )
            if not isinstance(assets[training_asset_name], dict):
                raise TypeError(
                    "checkpoint embedded inference asset state "
                    f"{training_asset_name!r} must be a state dictionary"
                )

    has_scaler_class = "grad_scaler_class" in state
    has_scaler_state = "grad_scaler_state_dict" in state
    if precision_kind == "fp16-mixed":
        if not has_scaler_class or not has_scaler_state:
            missing = [
                name
                for name, present in (
                    ("grad_scaler_class", has_scaler_class),
                    ("grad_scaler_state_dict", has_scaler_state),
                )
                if not present
            ]
            raise TypeError(
                "fp16-mixed checkpoint is missing required field(s): "
                + ", ".join(missing)
            )
        scaler_class = cast(object, state.get("grad_scaler_class"))
        if type(scaler_class) is not str or not scaler_class:
            raise TypeError(
                "checkpoint grad_scaler_class must be a non-empty string"
            )
        scaler_state = cast(object, state.get("grad_scaler_state_dict"))
        if type(scaler_state) is not dict:
            raise TypeError(
                "checkpoint grad_scaler_state_dict must be an exact dictionary"
            )
        _validate_grad_scaler_scale(
            scaler_state.get("scale"),
            path="checkpoint grad_scaler_state_dict.scale",
        )
    elif has_scaler_class or has_scaler_state:
        unexpected = [
            name
            for name, present in (
                ("grad_scaler_class", has_scaler_class),
                ("grad_scaler_state_dict", has_scaler_state),
            )
            if present
        ]
        raise ValueError(
            f"{precision_kind} checkpoint cannot contain GradScaler field(s): "
            + ", ".join(unexpected)
        )

    _validate_v11_ema_payload(state)
    _validate_checkpoint_config(state, precision_kind=precision_kind)
    _validate_extension_plugin_metadata(state)
    parse_rng_state(state.get("rng_state"))


def _validate_v11_ema_payload(state: CheckpointState) -> None:
    """Validate the self-contained EMA projection shared by v11 readers."""

    has_ema_state = "ema_state_dict" in state
    has_ema_projection = "ema_model_state_dict" in state
    if has_ema_state != has_ema_projection:
        missing = (
            "ema_model_state_dict" if has_ema_state else "ema_state_dict"
        )
        raise TypeError(
            f"checkpoint EMA topology is missing required field: {missing}"
        )
    if not has_ema_state:
        return

    model_state_value = cast(object, state.get("model_state_dict"))
    if not isinstance(model_state_value, dict):
        raise TypeError(
            "checkpoint with EMA state requires a valid model_state_dict"
        )
    projection_value = cast(object, state.get("ema_model_state_dict"))
    if not isinstance(projection_value, dict):
        raise TypeError("ema_model_state_dict must be a dictionary")
    ema_state = _validate_serialized_ema_state_dict(
        state.get("ema_state_dict"),
        model_state=cast(Mapping[str, object], model_state_value),
    )
    _validate_serialized_ema_projection(
        cast(Mapping[str, object], model_state_value),
        cast(Mapping[str, object], projection_value),
        ema_state,
    )


def _validate_precision_kind(value: object, *, path: str) -> str:
    if type(value) is not str or value not in _PRECISION_KINDS:
        choices = ", ".join(sorted(_PRECISION_KINDS))
        raise ValueError(f"{path} must be one of: {choices}")
    return cast(str, value)


def _validate_grad_scaler_scale(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a finite positive number")
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{path} must be a finite positive number")
    return scale


def _validate_checkpoint_config(
    state: CheckpointState,
    *,
    precision_kind: str,
) -> None:
    config_value = cast(object, state.get("config"))
    if config_value is None:
        return
    if type(config_value) is not dict:
        raise TypeError("checkpoint config must be an exact dictionary")
    trainer_value = cast(object, config_value.get("trainer"))
    if trainer_value is None:
        return
    if type(trainer_value) is not dict:
        raise TypeError("checkpoint config.trainer must be an exact dictionary")
    configured_precision = cast(object, trainer_value.get("precision"))
    if configured_precision is not None and configured_precision != precision_kind:
        raise ValueError(
            "checkpoint config.trainer.precision does not match precision_kind: "
            f"{configured_precision!r} != {precision_kind!r}"
        )
    accumulation = cast(
        object,
        trainer_value.get("accumulate_grad_batches"),
    )
    if accumulation is not None and (
        type(accumulation) is not int or accumulation <= 0
    ):
        raise ValueError(
            "checkpoint config.trainer.accumulate_grad_batches must be "
            "a positive integer"
        )


def _validate_extension_plugin_metadata(state: CheckpointState) -> None:
    """Require the extension-provenance container on every checkpoint."""

    metadata = cast(object, state.get("metadata"))
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint is missing valid metadata")
    extension_plugins = cast(object, metadata.get("extension_plugins"))
    if not isinstance(extension_plugins, list):
        raise TypeError("checkpoint metadata.extension_plugins must be a list")
    parse_extension_plugin_provenance(extension_plugins)


def _validate_checkpoint_value(value: object, *, path: str) -> None:
    """Validate the recursively data-only checkpoint value contract."""

    value_type = type(value)
    if value_type in _CHECKPOINT_LEAF_TYPES:
        return
    if value_type is torch.Tensor or value_type is nn.Parameter:
        return
    if value_type is dict or value_type is OrderedDict:
        mapping = cast(dict[object, object], value)
        for key, item in mapping.items():
            _validate_checkpoint_value(key, path=f"{path}<key {key!r}>")
            _validate_checkpoint_value(item, path=f"{path}[{key!r}]")
        if value_type is OrderedDict:
            attributes = vars(value)
            unexpected_attributes = set(attributes) - {"_metadata"}
            if unexpected_attributes:
                names = ", ".join(sorted(unexpected_attributes))
                raise TypeError(
                    f"{path} contains unsupported OrderedDict attributes: {names}"
                )
            if "_metadata" in attributes:
                _validate_checkpoint_value(
                    attributes["_metadata"],
                    path=f"{path}._metadata",
                )
        return
    if value_type is list or value_type is tuple:
        sequence = cast(list[object] | tuple[object, ...], value)
        for index, item in enumerate(sequence):
            _validate_checkpoint_value(item, path=f"{path}[{index}]")
        return
    raise TypeError(
        f"{path} contains unsupported checkpoint value type "
        f"'{value_type.__module__}.{value_type.__qualname__}'; checkpoints "
        "may contain only exact Tensor or Parameter values, primitive values, "
        "and plain dict, OrderedDict, list, or tuple containers"
    )


def _clone_checkpoint_data(
    value: object,
    *,
    tensors_to_cpu: bool,
) -> object:
    """Clone one validated data-only value for transactional rollback."""

    value_type = type(value)
    if value_type in _CHECKPOINT_LEAF_TYPES:
        return value
    if value_type is torch.Tensor or value_type is nn.Parameter:
        tensor = cast(torch.Tensor, value).detach()
        return (
            tensor.to(device="cpu", copy=True)
            if tensors_to_cpu
            else tensor.clone()
        )
    if value_type is dict:
        mapping = cast(dict[object, object], value)
        return {
            _clone_checkpoint_data(key, tensors_to_cpu=tensors_to_cpu):
            _clone_checkpoint_data(item, tensors_to_cpu=tensors_to_cpu)
            for key, item in mapping.items()
        }
    if value_type is OrderedDict:
        mapping = cast(OrderedDict[object, object], value)
        cloned = OrderedDict(
            (
                _clone_checkpoint_data(key, tensors_to_cpu=tensors_to_cpu),
                _clone_checkpoint_data(item, tensors_to_cpu=tensors_to_cpu),
            )
            for key, item in mapping.items()
        )
        metadata = getattr(value, "_metadata", None)
        if metadata is not None:
            cast(StateDictWithMetadata, cloned)._metadata = deepcopy(metadata)
        return cloned
    if value_type is list:
        return [
            _clone_checkpoint_data(item, tensors_to_cpu=tensors_to_cpu)
            for item in cast(list[object], value)
        ]
    if value_type is tuple:
        return tuple(
            _clone_checkpoint_data(item, tensors_to_cpu=tensors_to_cpu)
            for item in cast(tuple[object, ...], value)
        )
    raise TypeError(
        "runtime checkpoint state contains unsupported rollback value type "
        f"'{value_type.__module__}.{value_type.__qualname__}'"
    )


def _checkpoint_values_equal(left: object, right: object) -> bool:
    """Compare two data-only values without ambiguous Tensor equality."""

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or left.layout != right.layout
        ):
            return False
        left_cpu = left.detach().to(device="cpu")
        right_cpu = right.detach().to(device="cpu")
        if torch.equal(left_cpu, right_cpu):
            return True
        if not (torch.is_floating_point(left_cpu) or torch.is_complex(left_cpu)):
            return False
        try:
            equal_or_matching_nan = torch.eq(left_cpu, right_cpu) | (
                torch.isnan(left_cpu) & torch.isnan(right_cpu)
            )
            return bool(torch.all(equal_or_matching_nan).item())
        except RuntimeError:
            return False
    if type(left) is not type(right):
        return False
    if type(left) is dict or type(left) is OrderedDict:
        left_mapping = cast(Mapping[object, object], left)
        right_mapping = cast(Mapping[object, object], right)
        return (
            set(left_mapping) == set(right_mapping)
            and all(
                _checkpoint_values_equal(
                    left_mapping[key],
                    right_mapping[key],
                )
                for key in left_mapping
            )
        )
    if type(left) is list or type(left) is tuple:
        left_sequence = cast(list[object] | tuple[object, ...], left)
        right_sequence = cast(list[object] | tuple[object, ...], right)
        return len(left_sequence) == len(right_sequence) and all(
            _checkpoint_values_equal(left_item, right_item)
            for left_item, right_item in zip(
                left_sequence,
                right_sequence,
                strict=True,
            )
        )
    return bool(left == right)


def validate_module_state_dict_compatibility(
    module: nn.Module,
    state_dict: Mapping[str, object],
    *,
    path: str,
    allow_lazy_state: bool = True,
) -> None:
    """Reject ordinary strict-load failures before touching a module."""

    if any(type(key) is not str for key in state_dict):
        raise TypeError(f"{path} keys must be exact strings")
    runtime_state = module.state_dict()
    actual_keys = cast(set[str], set(state_dict))
    expected_keys = set(runtime_state)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"{path} keys do not match runtime: "
            f"missing={missing or '<none>'}, "
            f"unexpected={unexpected or '<none>'}"
        )
    typed_state = cast(Mapping[str, object], state_dict)
    for name, runtime_value in runtime_state.items():
        checkpoint_value = typed_state[name]
        if not isinstance(runtime_value, torch.Tensor):
            continue
        if not isinstance(checkpoint_value, torch.Tensor):
            raise TypeError(f"{path}[{name!r}] must be a Tensor")
        if checkpoint_value.device.type == "meta":
            raise ValueError(
                f"{path}[{name!r}] cannot use the meta device"
            )
        if torch.nn.parameter.is_lazy(runtime_value):
            if not allow_lazy_state:
                raise ValueError(
                    f"{path}[{name!r}] runtime state is lazy; exact shape "
                    "validation requires initialized state"
                )
        elif checkpoint_value.shape != runtime_value.shape:
            raise ValueError(
                f"{path}[{name!r}] shape does not match runtime: "
                f"{tuple(checkpoint_value.shape)} != "
                f"{tuple(runtime_value.shape)}"
            )
        if checkpoint_value.dtype != runtime_value.dtype:
            raise ValueError(
                f"{path}[{name!r}] dtype does not match runtime: "
                f"{checkpoint_value.dtype} != {runtime_value.dtype}"
            )
        if checkpoint_value.layout != runtime_value.layout:
            raise ValueError(
                f"{path}[{name!r}] layout does not match runtime"
            )


def _validate_optimizer_state_dict_compatibility(
    optimizer: Optimizer,
    state_dict: Mapping[str, object],
) -> None:
    """Reject optimizer parameter-group topology changes before module loads."""

    if not isinstance(state_dict, dict):
        raise TypeError(
            "checkpoint optimizer_state_dict must be a dictionary"
        )
    optimizer_state = state_dict.get("state")
    if type(optimizer_state) is not dict:
        raise TypeError(
            "checkpoint optimizer_state_dict.state must be an exact dictionary"
        )
    param_groups = state_dict.get("param_groups")
    if type(param_groups) is not list:
        raise TypeError(
            "checkpoint optimizer_state_dict.param_groups must be an exact list"
        )
    expected_fields = {"state", "param_groups"}
    actual_fields = set(state_dict)
    if not expected_fields.issubset(actual_fields):
        missing = sorted(map(str, expected_fields - actual_fields))
        raise ValueError(
            "checkpoint optimizer_state_dict is missing required field(s): "
            + ", ".join(missing)
        )
    checkpoint_groups = cast(list[object], param_groups)
    runtime_groups = optimizer.state_dict().get("param_groups")
    if type(runtime_groups) is not list:
        raise TypeError("runtime optimizer state has invalid param_groups")
    typed_runtime_groups = cast(list[object], runtime_groups)
    if len(checkpoint_groups) != len(typed_runtime_groups):
        raise ValueError(
            "checkpoint optimizer parameter-group count does not match runtime"
        )
    for index, (checkpoint_group, runtime_group) in enumerate(
        zip(checkpoint_groups, typed_runtime_groups, strict=True)
    ):
        if type(checkpoint_group) is not dict:
            raise TypeError(
                "checkpoint optimizer_state_dict.param_groups"
                f"[{index}] must be an exact dictionary"
            )
        if type(runtime_group) is not dict:
            raise TypeError(
                f"runtime optimizer param_groups[{index}] is invalid"
            )
        checkpoint_params = cast(dict[object, object], checkpoint_group).get(
            "params"
        )
        runtime_params = cast(dict[object, object], runtime_group).get("params")
        if type(checkpoint_params) is not list:
            raise TypeError(
                "checkpoint optimizer_state_dict.param_groups"
                f"[{index}].params must be an exact list"
            )
        if type(runtime_params) is not list:
            raise TypeError(
                f"runtime optimizer param_groups[{index}].params is invalid"
            )
        if len(checkpoint_params) != len(runtime_params):
            raise ValueError(
                "checkpoint optimizer parameter count does not match runtime "
                f"for parameter group {index}"
            )
        checkpoint_group_keys = set(
            cast(dict[object, object], checkpoint_group)
        )
        required_group_keys = (
            set(cast(dict[object, object], runtime_group)) - {"params"}
        )
        missing_group_keys = required_group_keys - checkpoint_group_keys
        if missing_group_keys:
            raise ValueError(
                "checkpoint optimizer parameter group is missing runtime "
                f"key(s) for group {index}: "
                + ", ".join(sorted(map(str, missing_group_keys)))
            )


def _validate_lr_scheduler_state_dict_compatibility(
    lr_scheduler: LRScheduler,
    state_dict: Mapping[str, object],
) -> None:
    """Reject scheduler key-topology drift before applying any module state."""

    required_keys = set(lr_scheduler.state_dict())
    missing = required_keys - set(state_dict)
    if missing:
        raise ValueError(
            "checkpoint lr_scheduler_state_dict is missing runtime key(s): "
            + ", ".join(sorted(map(str, missing)))
        )


def _validate_serialized_ema_state_dict(
    value: object,
    *,
    model_state: Mapping[str, object],
) -> EMAStateDict:
    """Validate the runtime-independent EMA schema in a serialized payload."""

    if type(value) is not dict:
        raise TypeError("ema_state_dict must be an exact dictionary")
    state = cast(dict[str, object], value)
    required = {
        "decay",
        "update_after_step",
        "update_every",
        "num_updates",
        "shadow_params",
        "shadow_buffers",
    }
    if set(state) != required:
        missing = sorted(required - set(state))
        unexpected = sorted(set(state) - required)
        raise ValueError(
            "ema_state_dict has invalid fields: "
            f"missing={missing or '<none>'}, "
            f"unexpected={unexpected or '<none>'}"
        )
    decay = state["decay"]
    if (
        isinstance(decay, bool)
        or not isinstance(decay, (int, float))
        or not math.isfinite(float(decay))
        or not 0.0 <= float(decay) < 1.0
    ):
        raise ValueError("ema_state_dict.decay must satisfy 0 <= decay < 1")
    for field_name, minimum in (
        ("update_after_step", 0),
        ("update_every", 1),
        ("num_updates", 0),
    ):
        field_value = state[field_name]
        if type(field_value) is not int or cast(int, field_value) < minimum:
            raise ValueError(
                f"ema_state_dict.{field_name} must be an integer >= {minimum}"
            )

    shadow_names: set[str] = set()
    for field_name in ("shadow_params", "shadow_buffers"):
        raw_values = state[field_name]
        if type(raw_values) is not OrderedDict:
            raise TypeError(
                f"ema_state_dict.{field_name} must be an exact OrderedDict"
            )
        values = cast(OrderedDict[object, object], raw_values)
        if any(type(name) is not str for name in values):
            raise TypeError(
                f"ema_state_dict.{field_name} keys must be exact strings"
            )
        typed_names = cast(set[str], set(values))
        overlap = shadow_names.intersection(typed_names)
        if overlap:
            raise ValueError(
                "ema_state_dict shadow parameter and buffer names overlap: "
                + ", ".join(sorted(overlap))
            )
        shadow_names.update(typed_names)
        for name, raw_tensor in values.items():
            if name not in model_state:
                raise ValueError(
                    f"ema_state_dict.{field_name}[{name!r}] is missing "
                    "from model_state_dict"
                )
            if type(raw_tensor) is not torch.Tensor:
                raise TypeError(
                    f"ema_state_dict.{field_name}[{name!r}] must be an exact Tensor"
                )
            model_tensor = model_state[cast(str, name)]
            if not isinstance(model_tensor, torch.Tensor):
                raise TypeError(
                    f"model_state_dict[{name!r}] must be a Tensor for EMA"
                )
            tensor = cast(torch.Tensor, raw_tensor)
            if (
                tensor.shape != model_tensor.shape
                or tensor.dtype != model_tensor.dtype
                or tensor.layout != model_tensor.layout
            ):
                raise ValueError(
                    f"ema_state_dict.{field_name}[{name!r}] topology does "
                    "not match model_state_dict"
                )
    return cast(EMAStateDict, value)


def _validate_serialized_ema_projection(
    model_state: Mapping[str, object],
    ema_model_state: Mapping[str, object],
    ema_state: EMAStateDict,
) -> None:
    """Validate EMA projection values, including tied state-dict aliases."""

    if set(model_state) != set(ema_model_state):
        raise ValueError(
            "ema_model_state_dict keys must match model_state_dict"
        )
    shadow_values: dict[str, torch.Tensor] = {
        **ema_state["shadow_params"],
        **ema_state["shadow_buffers"],
    }
    canonical_model_tensors = {
        name: cast(torch.Tensor, model_state[name])
        for name in shadow_values
    }
    for key, raw_value in model_state.items():
        expected_value: object = raw_value
        if key in shadow_values:
            expected_value = shadow_values[key]
        elif isinstance(raw_value, torch.Tensor):
            alias_name = next(
                (
                    canonical_name
                    for canonical_name, canonical_tensor
                    in canonical_model_tensors.items()
                    if _tensor_state_alias(raw_value, canonical_tensor)
                ),
                None,
            )
            if alias_name is not None:
                expected_value = shadow_values[alias_name]
        if not _checkpoint_values_equal(
            ema_model_state[key],
            expected_value,
        ):
            raise ValueError(
                f"ema_model_state_dict[{key!r}] does not match EMA state"
            )


def _tensor_state_alias(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Return whether two state-dict tensors represent the same tensor view."""

    if left is right:
        return True
    if (
        left.layout is not torch.strided
        or right.layout is not torch.strided
        or left.device.type == "meta"
        or right.device.type == "meta"
    ):
        return False
    return (
        left.device == right.device
        and left.dtype == right.dtype
        and left.shape == right.shape
        and left.stride() == right.stride()
        and left.storage_offset() == right.storage_offset()
        and left.untyped_storage().data_ptr()
        == right.untyped_storage().data_ptr()
    )


def _validate_ema_state_dict(
    ema: ExponentialMovingAverage,
    value: object,
) -> EMAStateDict:
    """Validate the complete EMA state before managed modules are restored."""

    if type(value) is not dict:
        raise TypeError("ema_state_dict must be an exact dictionary")
    state = cast(dict[str, object], value)
    required = {
        "decay",
        "update_after_step",
        "update_every",
        "num_updates",
        "shadow_params",
        "shadow_buffers",
    }
    if set(state) != required:
        missing = sorted(required - set(state))
        unexpected = sorted(set(state) - required)
        raise ValueError(
            "ema_state_dict has invalid fields: "
            f"missing={missing or '<none>'}, "
            f"unexpected={unexpected or '<none>'}"
        )
    decay = state["decay"]
    if (
        isinstance(decay, bool)
        or not isinstance(decay, (int, float))
        or not math.isfinite(float(decay))
        or not 0.0 <= float(decay) < 1.0
    ):
        raise ValueError("ema_state_dict.decay must satisfy 0 <= decay < 1")
    for field_name, minimum in (
        ("update_after_step", 0),
        ("update_every", 1),
        ("num_updates", 0),
    ):
        field_value = state[field_name]
        if type(field_value) is not int or cast(int, field_value) < minimum:
            raise ValueError(
                f"ema_state_dict.{field_name} must be an integer >= {minimum}"
            )
    for field_name, runtime_values in (
        ("shadow_params", ema.shadow_params),
        ("shadow_buffers", ema.shadow_buffers),
    ):
        raw_values = state[field_name]
        if type(raw_values) is not OrderedDict:
            raise TypeError(
                f"ema_state_dict.{field_name} must be an exact OrderedDict"
            )
        values = cast(OrderedDict[object, object], raw_values)
        if any(type(name) is not str for name in values):
            raise TypeError(
                f"ema_state_dict.{field_name} keys must be exact strings"
            )
        actual_names = cast(set[str], set(values))
        expected_names = set(runtime_values)
        if actual_names != expected_names:
            raise ValueError(
                f"ema_state_dict.{field_name} keys do not match runtime"
            )
        for name, runtime_tensor in runtime_values.items():
            checkpoint_tensor = values[name]
            if type(checkpoint_tensor) is not torch.Tensor:
                raise TypeError(
                    f"ema_state_dict.{field_name}[{name!r}] must be an exact Tensor"
                )
            tensor = cast(torch.Tensor, checkpoint_tensor)
            if tensor.shape != runtime_tensor.shape:
                raise ValueError(
                    f"ema_state_dict.{field_name}[{name!r}] shape does not "
                    "match runtime"
                )
            if tensor.dtype != runtime_tensor.dtype:
                raise ValueError(
                    f"ema_state_dict.{field_name}[{name!r}] dtype does not "
                    "match runtime"
                )
    return cast(EMAStateDict, value)


def _prepare_ema_state_for_runtime(
    ema: ExponentialMovingAverage,
    value: EMAStateDict,
) -> EMAStateDict:
    """Detach an EMA payload and align its tensors with runtime shadows."""

    prepared = cast(
        EMAStateDict,
        _clone_checkpoint_data(value, tensors_to_cpu=False),
    )
    for field_name, runtime_values in (
        ("shadow_params", ema.shadow_params),
        ("shadow_buffers", ema.shadow_buffers),
    ):
        payload_values = prepared[field_name]
        prepared[field_name] = OrderedDict(
            (
                name,
                payload_values[name].to(
                    device=runtime_tensor.device,
                    dtype=runtime_tensor.dtype,
                    copy=True,
                ),
            )
            for name, runtime_tensor in runtime_values.items()
        )
    return prepared


def _validate_ema_model_consistency(
    model: nn.Module,
    model_state: Mapping[str, object],
    ema_model_state: Mapping[str, object],
    ema_state: EMAStateDict,
) -> None:
    """Keep the inference EMA projection identical to its canonical shadows."""

    if set(model_state) != set(ema_model_state):
        raise ValueError(
            "ema_model_state_dict keys must match model_state_dict"
    )
    shadow_params = ema_state["shadow_params"]
    shadow_buffers = ema_state["shadow_buffers"]
    canonical_shadow_names = {
        id(value): name
        for name, value in (
            *model.named_parameters(),
            *model.named_buffers(),
        )
        if name in shadow_params or name in shadow_buffers
    }
    runtime_state = model.state_dict(keep_vars=True)
    for key, raw_value in model_state.items():
        runtime_value = runtime_state[key]
        shadow_name = canonical_shadow_names.get(id(runtime_value))
        if shadow_name in shadow_params:
            expected_value = shadow_params[shadow_name]
        elif shadow_name in shadow_buffers:
            expected_value = shadow_buffers[shadow_name]
        else:
            expected_value = raw_value
        if not _checkpoint_values_equal(
            ema_model_state[key],
            expected_value,
        ):
            raise ValueError(
                f"ema_model_state_dict[{key!r}] does not match EMA state"
            )


def _ensure_parent_directory(path: Path) -> None:
    """Create the parent directory for a checkpoint path."""

    path.parent.mkdir(parents=True, exist_ok=True)


def _type_identity(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _clone_module_state(
    module: nn.Module,
    *,
    tensors_to_cpu: bool = False,
) -> OrderedDict[str, Any]:
    """Clone a module state so temporary EMA swaps cannot mutate a snapshot."""

    source = module.state_dict(keep_vars=True)
    tensor_clones: dict[int, torch.Tensor] = {}
    cloned: OrderedDict[str, Any] = OrderedDict()
    for name, value in source.items():
        if isinstance(value, torch.Tensor):
            tensor = tensor_clones.get(id(value))
            if tensor is None:
                detached = value.detach()
                tensor = (
                    detached.to(device="cpu", copy=True)
                    if tensors_to_cpu
                    else detached.clone()
                )
                tensor_clones[id(value)] = tensor
            cloned[name] = tensor
        else:
            cloned[name] = deepcopy(value)
    metadata = getattr(source, "_metadata", None)
    if metadata is not None:
        cast(StateDictWithMetadata, cloned)._metadata = deepcopy(metadata)
    return cloned


def _project_ema_module_state(
    module: nn.Module,
    model_state: Mapping[str, object],
    ema_state: EMAStateDict,
) -> OrderedDict[str, Any]:
    """Derive one host-resident projection from the serialized EMA snapshot."""

    shadow_params = ema_state["shadow_params"]
    shadow_buffers = ema_state["shadow_buffers"]
    shadow_values: dict[str, torch.Tensor] = {
        **shadow_params,
        **shadow_buffers,
    }
    canonical_shadow_names = {
        id(value): name
        for name, value in (
            *module.named_parameters(),
            *module.named_buffers(),
        )
        if name in shadow_values
    }
    runtime_state = module.state_dict(keep_vars=True)
    tensor_clones: dict[int, torch.Tensor] = {}
    projected: OrderedDict[str, Any] = OrderedDict()
    for name, raw_value in model_state.items():
        runtime_value = runtime_state[name]
        shadow_name = canonical_shadow_names.get(id(runtime_value))
        source_value: object = (
            shadow_values[shadow_name]
            if shadow_name is not None
            else raw_value
        )
        if isinstance(source_value, torch.Tensor):
            tensor = tensor_clones.get(id(source_value))
            if tensor is None:
                tensor = source_value.detach().to(device="cpu", copy=True)
                tensor_clones[id(source_value)] = tensor
            projected[name] = tensor
        else:
            projected[name] = deepcopy(source_value)
    metadata = getattr(model_state, "_metadata", None)
    if metadata is not None:
        cast(StateDictWithMetadata, projected)._metadata = deepcopy(metadata)
    return projected


def _find_named_checkpoint(root: str | Path, filename: str) -> Path:
    """Resolve a named checkpoint from a checkpoint dir, run dir, or output root."""

    root_path = Path(root)
    if root_path.is_file():
        if root_path.name != filename:
            raise FileNotFoundError(
                f"expected checkpoint file named '{filename}', got '{root_path}'"
            )
        return root_path

    if not root_path.exists():
        raise FileNotFoundError(f"checkpoint search root does not exist: {root_path}")
    if not root_path.is_dir():
        raise FileNotFoundError(f"checkpoint search root is not a directory: {root_path}")

    candidates = [
        candidate
        for candidate in (
            root_path / filename,
            root_path / "checkpoints" / filename,
        )
        if candidate.is_file()
    ]
    candidates.extend(
        candidate
        for candidate in root_path.rglob(filename)
        if candidate.is_file() and candidate.parent.name == "checkpoints"
    )
    matches = {
        candidate.resolve(): candidate
        for candidate in candidates
    }
    if not matches:
        raise FileNotFoundError(
            f"could not find '{filename}' under checkpoint search root: {root_path}"
        )
    return max(matches.values(), key=lambda path: path.stat().st_mtime)

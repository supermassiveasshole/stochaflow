"""Checkpoint save/load helpers."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

if TYPE_CHECKING:
    from stochaflow.training.ema import EMAStateDict, ExponentialMovingAverage
else:
    EMAStateDict = dict[str, Any]
    ExponentialMovingAverage = Any


CHECKPOINT_FORMAT_VERSION = 7


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
        }
        if self.process is not None:
            state["process_state_dict"] = _clone_module_state(self.process)
        if self.objective is not None:
            state["objective_state_dict"] = _clone_module_state(self.objective)
        state["training_assets_state_dict"] = {
            name: _clone_module_state(module)
            for name, module in self.auxiliary_modules.items()
        }
        if self.ema is not None:
            self.ema.store(self.model)
            try:
                self.ema.copy_to(self.model)
                state["ema_model_state_dict"] = _clone_module_state(self.model)
            finally:
                self.ema.restore(self.model)
        if self.optimizer is not None:
            state["optimizer_class"] = _type_identity(self.optimizer)
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
            state["lr_scheduler_class"] = _type_identity(self.lr_scheduler)
            state["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        if self.ema is not None:
            state["ema_state_dict"] = self.ema.state_dict()
        if epoch is not None:
            state["epoch"] = epoch
        if global_step is not None:
            state["global_step"] = global_step
        if config is not None:
            state["config"] = config
        if metrics is not None:
            state["metrics"] = metrics
        if metadata is not None:
            state["metadata"] = metadata
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
        _ensure_parent_directory(checkpoint_path)
        state = self.build_state(
            epoch=epoch,
            global_step=global_step,
            config=config,
            metrics=metrics,
            metadata=metadata,
        )
        torch.save(state, checkpoint_path)
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
        version = state.get("format_version")
        if version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"checkpoint format version {version!r} is unsupported; "
                f"expected version {CHECKPOINT_FORMAT_VERSION}"
            )
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

        self.model.load_state_dict(model_state_dict)
        if self.process is not None:
            assert validated_process_state is not None
            self.process.load_state_dict(validated_process_state)
        if self.objective is not None:
            assert validated_objective_state is not None
            self.objective.load_state_dict(validated_objective_state)
        for name, module in self.auxiliary_modules.items():
            module.load_state_dict(validated_assets[name])

        if self.lr_scheduler is not None:
            assert validated_lr_scheduler_state is not None
            self.lr_scheduler.load_state_dict(validated_lr_scheduler_state)
        if self.optimizer is not None:
            assert validated_optimizer_state is not None
            self.optimizer.load_state_dict(validated_optimizer_state)

        ema_state_dict = cast(object, state.get("ema_state_dict"))
        if self.ema is not None:
            if ema_state_dict is None:
                raise TypeError("checkpoint is missing ema_state_dict")
            if not isinstance(ema_state_dict, dict):
                raise TypeError("ema_state_dict must be a dictionary when provided")
            self.ema.load_state_dict(cast(EMAStateDict, ema_state_dict))

        epoch = cast(object, state.get("epoch"))
        if epoch is not None and not isinstance(epoch, int):
            raise TypeError("epoch must be an int when provided")
        global_step = cast(object, state.get("global_step"))
        if global_step is not None and not isinstance(global_step, int):
            raise TypeError("global_step must be an int when provided")

        config = cast(object, state.get("config"))
        if config is not None and not isinstance(config, dict):
            raise TypeError("config must be a dictionary when provided")
        metrics = cast(object, state.get("metrics"))
        if metrics is None:
            metrics = {}
        elif not isinstance(metrics, dict):
            raise TypeError("metrics must be a dictionary when provided")
        metadata = cast(object, state.get("metadata"))
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary when provided")

        return LoadedCheckpoint(
            path=checkpoint_path,
            epoch=epoch,
            global_step=global_step,
            config=config,
            metrics=metrics,
            metadata=metadata,
        )

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
            weights_only=False,
        )
        if not isinstance(raw_state, dict):
            raise TypeError(
                f"checkpoint at '{checkpoint_path}' must contain a dictionary payload"
            )
        return cast(CheckpointState, raw_state)

def _ensure_parent_directory(path: Path) -> None:
    """Create the parent directory for a checkpoint path."""

    path.parent.mkdir(parents=True, exist_ok=True)


def _type_identity(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _clone_module_state(module: nn.Module) -> OrderedDict[str, Any]:
    """Clone a module state so temporary EMA swaps cannot mutate a snapshot."""

    source = module.state_dict()
    cloned = OrderedDict(
        (
            name,
            (
                value.detach().clone()
                if isinstance(value, torch.Tensor)
                else deepcopy(value)
            ),
        )
        for name, value in source.items()
    )
    metadata = getattr(source, "_metadata", None)
    if metadata is not None:
        setattr(cloned, "_metadata", deepcopy(metadata))
    return cloned


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

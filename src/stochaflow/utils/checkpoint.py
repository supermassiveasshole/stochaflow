"""Checkpoint save/load helpers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, cast

import torch
import torch.nn as nn
from torch.optim import Optimizer

if TYPE_CHECKING:
    from stochaflow.training.ema import EMAStateDict, ExponentialMovingAverage
else:
    EMAStateDict = dict[str, Any]
    ExponentialMovingAverage = Any


CHECKPOINT_FORMAT_VERSION = 3


class CheckpointState(TypedDict, total=False):
    """Serialized checkpoint payload."""

    format_version: int
    epoch: int
    global_step: int
    model_state_dict: dict[str, torch.Tensor]
    denoiser_state_dict: dict[str, torch.Tensor]
    ema_denoiser_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
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
    denoiser: nn.Module | None = None
    optimizer: Optimizer | None = None
    lr_scheduler: Any | None = None
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
        if self.denoiser is not None:
            state["denoiser_state_dict"] = _clone_module_state(self.denoiser)
            if self.ema is not None:
                self.ema.store(self.model)
                try:
                    self.ema.copy_to(self.model)
                    state["ema_denoiser_state_dict"] = _clone_module_state(
                        self.denoiser
                    )
                finally:
                    self.ema.restore(self.model)
        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.lr_scheduler is not None:
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
        model_state_dict = state.get("model_state_dict")
        if not isinstance(model_state_dict, dict):
            raise TypeError("checkpoint is missing a valid model_state_dict")
        self.model.load_state_dict(model_state_dict)

        optimizer_state_dict = state.get("optimizer_state_dict")
        if self.optimizer is not None:
            if optimizer_state_dict is None:
                raise TypeError("checkpoint is missing optimizer_state_dict")
            if not isinstance(optimizer_state_dict, dict):
                raise TypeError("optimizer_state_dict must be a dictionary when provided")
            self.optimizer.load_state_dict(optimizer_state_dict)

        lr_scheduler_state_dict = state.get("lr_scheduler_state_dict")
        if self.lr_scheduler is not None:
            if lr_scheduler_state_dict is None:
                raise TypeError("checkpoint is missing lr_scheduler_state_dict")
            if not isinstance(lr_scheduler_state_dict, dict):
                raise TypeError(
                    "lr_scheduler_state_dict must be a dictionary when provided"
                )
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

        ema_state_dict = state.get("ema_state_dict")
        if self.ema is not None:
            if ema_state_dict is None:
                raise TypeError("checkpoint is missing ema_state_dict")
            if not isinstance(ema_state_dict, dict):
                raise TypeError("ema_state_dict must be a dictionary when provided")
            self.ema.load_state_dict(cast(EMAStateDict, ema_state_dict))

        epoch = state.get("epoch")
        if epoch is not None and not isinstance(epoch, int):
            raise TypeError("epoch must be an int when provided")
        global_step = state.get("global_step")
        if global_step is not None and not isinstance(global_step, int):
            raise TypeError("global_step must be an int when provided")

        config = state.get("config")
        if config is not None and not isinstance(config, dict):
            raise TypeError("config must be a dictionary when provided")
        metrics = state.get("metrics")
        if metrics is None:
            metrics = {}
        elif not isinstance(metrics, dict):
            raise TypeError("metrics must be a dictionary when provided")
        metadata = state.get("metadata")
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


def _clone_module_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a module state so temporary EMA swaps cannot mutate a snapshot."""

    return {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
    }


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

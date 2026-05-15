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


class CheckpointState(TypedDict, total=False):
    """Serialized checkpoint payload."""

    epoch: int
    global_step: int
    model_state_dict: dict[str, torch.Tensor]
    optimizer_state_dict: dict[str, Any]
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
    optimizer: Optimizer | None = None
    ema: ExponentialMovingAverage | None = None

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
            "model_state_dict": self.model.state_dict(),
        }
        if self.optimizer is not None:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
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
        raw_state = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )
        if not isinstance(raw_state, dict):
            raise TypeError(
                f"checkpoint at '{checkpoint_path}' must contain a dictionary payload"
            )

        state = raw_state
        model_state_dict = state.get("model_state_dict")
        if not isinstance(model_state_dict, dict):
            raise TypeError("checkpoint is missing a valid model_state_dict")
        self.model.load_state_dict(model_state_dict)

        optimizer_state_dict = state.get("optimizer_state_dict")
        if self.optimizer is not None and optimizer_state_dict is not None:
            if not isinstance(optimizer_state_dict, dict):
                raise TypeError("optimizer_state_dict must be a dictionary when provided")
            self.optimizer.load_state_dict(optimizer_state_dict)

        ema_state_dict = state.get("ema_state_dict")
        if self.ema is not None and ema_state_dict is not None:
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
    def find_latest(directory: str | Path, pattern: str = "*.pt") -> Path | None:
        """Return the most recently modified checkpoint file in a directory."""

        checkpoint_dir = Path(directory)
        matches = sorted(
            checkpoint_dir.glob(pattern),
            key=lambda path: path.stat().st_mtime,
        )
        if not matches:
            return None
        return matches[-1]


def _ensure_parent_directory(path: Path) -> None:
    """Create the parent directory for a checkpoint path."""

    path.parent.mkdir(parents=True, exist_ok=True)

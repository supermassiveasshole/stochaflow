"""Checkpoint save/load helpers."""

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path
import random
import tempfile
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from stochaflow.utils.plugins import parse_extension_plugin_provenance

if TYPE_CHECKING:
    from stochaflow.training.ema import EMAStateDict, ExponentialMovingAverage
else:
    EMAStateDict = dict[str, Any]
    ExponentialMovingAverage = Any


CHECKPOINT_FORMAT_VERSION = 8

_CHECKPOINT_LEAF_TYPES = (type(None), bool, int, float, complex, str, bytes)


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
        checkpoint_metadata = dict(metadata or {})
        checkpoint_metadata.setdefault("extension_plugins", [])
        state["metadata"] = checkpoint_metadata
        _validate_checkpoint_value(state, path="checkpoint")
        _validate_extension_plugin_metadata(state)
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
        _validate_checkpoint_header(state)
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
            os.replace(temporary_path, checkpoint_path)
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
        """Restore managed objects from an already loaded checkpoint payload."""

        checkpoint_path = Path(path)
        if type(payload) is not dict:
            raise TypeError(
                f"checkpoint at '{checkpoint_path}' must contain a dictionary payload"
        )
        _validate_checkpoint_value(payload, path="checkpoint")
        state = cast(CheckpointState, payload)
        _validate_checkpoint_header(state)
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
        if not isinstance(metadata, dict):
            raise TypeError("checkpoint is missing valid metadata")

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
            weights_only=True,
        )
        if type(raw_state) is not dict:
            raise TypeError(
                f"checkpoint at '{checkpoint_path}' must contain a dictionary payload"
            )
        _validate_checkpoint_value(raw_state, path="checkpoint")
        state = cast(CheckpointState, raw_state)
        _validate_checkpoint_header(state)
        return state


def capture_rng_state() -> dict[str, Any]:
    """Capture process-global RNG streams using the v8 data-only contract."""

    return _capture_rng_state(include_cuda=True)


def _capture_rng_state(*, include_cuda: bool) -> dict[str, Any]:
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
    }


def parse_rng_state(
    value: object,
    *,
    require_cuda_compatibility: bool = False,
) -> ParsedRNGState:
    """Validate and normalize one v8 RNG snapshot without changing global RNGs."""

    root = _exact_dict(
        value,
        path="checkpoint.rng_state",
        fields={"python", "numpy", "torch_cpu", "torch_cuda"},
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

    return ParsedRNGState(
        python=python_state,
        numpy=numpy_state,
        torch_cpu=torch_cpu,
        torch_cuda=torch_cuda,
    )


def restore_rng_state(
    state: object,
    *,
    restore_cuda: bool = True,
) -> None:
    """Restore a fully parsed RNG snapshot, rolling back on runtime failure."""

    if not isinstance(state, ParsedRNGState):
        raise TypeError("state must be a ParsedRNGState")
    parsed_state = state
    previous = parse_rng_state(
        _capture_rng_state(include_cuda=restore_cuda),
        require_cuda_compatibility=restore_cuda and bool(parsed_state.torch_cuda),
    )
    try:
        _apply_rng_state(parsed_state, restore_cuda=restore_cuda)
    except Exception:
        _apply_rng_state(previous, restore_cuda=restore_cuda)
        raise


def _apply_rng_state(state: ParsedRNGState, *, restore_cuda: bool) -> None:
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.random.set_rng_state(state.torch_cpu)
    if restore_cuda and state.torch_cuda:
        torch.cuda.set_rng_state_all(list(state.torch_cuda))


def _exact_dict(value: object, *, path: str, fields: set[str]) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{path} must be an exact dictionary")
    result = cast(dict[object, object], value)
    if set(result) != fields:
        missing = sorted(fields - set(result))
        unknown = sorted(set(result) - fields, key=str)
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


def _parse_rng_tensor(value: object, *, path: str) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        raise TypeError(f"{path} must be an exact Tensor")
    tensor = cast(torch.Tensor, value)
    if tensor.dtype is not torch.uint8 or tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError(f"{path} must be a non-empty one-dimensional uint8 Tensor")
    return tensor.detach().cpu().clone()


def _validate_checkpoint_header(state: CheckpointState) -> None:
    """Validate the versioned metadata required before runtime restoration."""

    version = state.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format version {version!r} is unsupported; "
            f"expected version {CHECKPOINT_FORMAT_VERSION}"
        )
    _validate_extension_plugin_metadata(state)
    parse_rng_state(state.get("rng_state"))


def _validate_extension_plugin_metadata(state: CheckpointState) -> None:
    """Require the v8 extension-provenance container on every checkpoint."""

    metadata = cast(object, state.get("metadata"))
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint is missing valid metadata")
    extension_plugins = cast(object, metadata.get("extension_plugins"))
    if not isinstance(extension_plugins, list):
        raise TypeError("checkpoint metadata.extension_plugins must be a list")
    parse_extension_plugin_provenance(extension_plugins)


def _validate_checkpoint_value(value: object, *, path: str) -> None:
    """Validate the recursively data-only v8 checkpoint value contract."""

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
        f"'{value_type.__module__}.{value_type.__qualname__}'; v8 checkpoints "
        "may contain only exact Tensor or Parameter values, primitive values, "
        "and plain dict, OrderedDict, list, or tuple containers"
    )


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

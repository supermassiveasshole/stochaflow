"""Fixed single-node process-group session and control collectives."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import timedelta
from types import TracebackType
from typing import Literal, Protocol, Self, cast

import torch
import torch.distributed as torch_distributed

from stochaflow.data.ranked import DataRankContext

from .contracts import DistributedCollectives, DistributedTopology

DistributedReduction = Literal["sum", "min", "max"]
DistributedDeviceBinder = Callable[[str, DistributedTopology], torch.device]


class ProcessGroupRuntime(Protocol):
    """Injectable process-group operations used by the session wrapper."""

    def is_available(self) -> bool:
        """Return whether distributed support is present in this Torch build."""

        ...

    def is_initialized(self) -> bool:
        """Return whether a process group is currently initialized."""

        ...

    def initialize(
        self,
        *,
        backend: str,
        rank: int,
        world_size: int,
        timeout: timedelta,
    ) -> None:
        """Initialize one process group from the launch environment."""

        ...

    def destroy(self) -> None:
        """Destroy the current process group without adding a barrier."""

        ...

    def rank(self) -> int:
        """Return the initialized process-group rank."""

        ...

    def world_size(self) -> int:
        """Return the initialized process-group world size."""

        ...

    def backend(self) -> str:
        """Return the initialized process-group backend name."""

        ...

    def broadcast_object_list(
        self,
        values: list[object],
        *,
        source_rank: int,
    ) -> None:
        """Broadcast a small object list in place."""

        ...

    def gather_object(
        self,
        value: object,
        output: list[object] | None,
        *,
        destination_rank: int,
    ) -> None:
        """Gather one small object per rank to one destination."""

        ...

    def all_gather_object(self, output: list[object], value: object) -> None:
        """Gather one small fixed-size control value to every rank."""

        ...

    def all_reduce(
        self,
        value: torch.Tensor,
        *,
        reduction: DistributedReduction,
    ) -> None:
        """Reduce one scalar tensor in place."""

        ...


class TorchProcessGroupRuntime:
    """Adapter around the default ``torch.distributed`` process group."""

    def is_available(self) -> bool:
        return torch_distributed.is_available()

    def is_initialized(self) -> bool:
        return torch_distributed.is_initialized()

    def initialize(
        self,
        *,
        backend: str,
        rank: int,
        world_size: int,
        timeout: timedelta,
    ) -> None:
        torch_distributed.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timeout,
        )

    def destroy(self) -> None:
        torch_distributed.destroy_process_group()

    def rank(self) -> int:
        return torch_distributed.get_rank()

    def world_size(self) -> int:
        return torch_distributed.get_world_size()

    def backend(self) -> str:
        return str(torch_distributed.get_backend())

    def broadcast_object_list(
        self,
        values: list[object],
        *,
        source_rank: int,
    ) -> None:
        torch_distributed.broadcast_object_list(values, src=source_rank)

    def gather_object(
        self,
        value: object,
        output: list[object] | None,
        *,
        destination_rank: int,
    ) -> None:
        torch_distributed.gather_object(
            value,
            object_gather_list=output,
            dst=destination_rank,
        )

    def all_gather_object(self, output: list[object], value: object) -> None:
        torch_distributed.all_gather_object(output, value)

    def all_reduce(
        self,
        value: torch.Tensor,
        *,
        reduction: DistributedReduction,
    ) -> None:
        operations = {
            "sum": torch_distributed.ReduceOp.SUM,
            "min": torch_distributed.ReduceOp.MIN,
            "max": torch_distributed.ReduceOp.MAX,
        }
        torch_distributed.all_reduce(value, op=operations[reduction])


def _environment_integer(
    environ: Mapping[str, str],
    name: str,
    *,
    minimum: int,
) -> int:
    raw = cast(object, environ.get(name))
    if raw is None:
        raise RuntimeError(f"torchrun environment is missing {name}")
    if not isinstance(raw, str) or not raw or not raw.isascii() or not raw.isdecimal():
        raise ValueError(
            f"torchrun environment {name} must be a canonical decimal integer"
        )
    value = int(raw)
    if value < minimum:
        raise ValueError(
            f"torchrun environment {name} must be at least {minimum}"
        )
    return value


def parse_torchrun_environment(
    environ: Mapping[str, str] | None = None,
) -> DistributedTopology:
    """Validate and return one fixed, non-elastic, single-node topology."""

    values = os.environ if environ is None else environ
    rank = _environment_integer(values, "RANK", minimum=0)
    local_rank = _environment_integer(values, "LOCAL_RANK", minimum=0)
    world_size = _environment_integer(values, "WORLD_SIZE", minimum=1)
    local_world_size = _environment_integer(
        values,
        "LOCAL_WORLD_SIZE",
        minimum=1,
    )
    master_address = cast(object, values.get("MASTER_ADDR"))
    if not isinstance(master_address, str) or not master_address.strip():
        raise RuntimeError("torchrun environment is missing a non-empty MASTER_ADDR")
    master_port = _environment_integer(values, "MASTER_PORT", minimum=1)
    if master_port > 65535:
        raise ValueError("torchrun environment MASTER_PORT must not exceed 65535")

    for name in ("TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS"):
        if _environment_integer(values, name, minimum=0) != 0:
            raise ValueError(
                "fixed distributed training does not accept elastic restarts; "
                f"{name} must be zero"
            )
    if _environment_integer(values, "GROUP_RANK", minimum=0) != 0:
        raise ValueError(
            "fixed single-node distributed training requires GROUP_RANK=0"
        )
    group_world_size = values.get("GROUP_WORLD_SIZE")
    if group_world_size is not None and (
        _environment_integer(values, "GROUP_WORLD_SIZE", minimum=1) != 1
    ):
        raise ValueError(
            "fixed single-node distributed training requires GROUP_WORLD_SIZE=1"
        )
    role_rank = values.get("ROLE_RANK")
    if role_rank is not None and (
        _environment_integer(values, "ROLE_RANK", minimum=0) != rank
    ):
        raise ValueError(
            "fixed distributed training requires ROLE_RANK to equal RANK"
        )
    role_world_size = values.get("ROLE_WORLD_SIZE")
    if role_world_size is not None and (
        _environment_integer(values, "ROLE_WORLD_SIZE", minimum=1)
        != world_size
    ):
        raise ValueError(
            "fixed distributed training requires ROLE_WORLD_SIZE to equal WORLD_SIZE"
        )

    return DistributedTopology(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
    )


def _default_device_binder(
    backend: str,
    topology: DistributedTopology,
) -> torch.device:
    if backend == "gloo":
        return torch.device("cpu")
    if backend != "nccl":
        raise ValueError("distributed backend must be 'nccl' or 'gloo'")
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL distributed training requires CUDA")
    device_count = torch.cuda.device_count()
    if topology.local_rank >= device_count:
        raise RuntimeError(
            "torchrun LOCAL_RANK does not identify an available CUDA device: "
            f"{topology.local_rank} >= {device_count}"
        )
    torch.cuda.set_device(topology.local_rank)
    return torch.device("cuda", topology.local_rank)


def _validate_bound_device(
    backend: str,
    topology: DistributedTopology,
    device: torch.device,
) -> None:
    device_value = cast(object, device)
    if not isinstance(device_value, torch.device):
        raise TypeError("distributed device binder must return torch.device")
    if backend == "gloo" and device.type != "cpu":
        raise ValueError("Gloo test sessions require a CPU device")
    if backend == "nccl" and (
        device.type != "cuda" or device.index != topology.local_rank
    ):
        raise ValueError("NCCL session device must be cuda:LOCAL_RANK")


def _control_value(value: object, *, path: str) -> object:
    value_type = type(value)
    if value is None:
        return ["none"]
    if value_type is bool:
        return ["bool", value]
    if value_type is int:
        return ["int", str(value)]
    if value_type is float:
        float_value = cast(float, value)
        if not math.isfinite(float_value):
            raise ValueError(f"{path} must not contain non-finite floats")
        return ["float", float_value.hex()]
    if value_type is str:
        return ["str", value]
    if value_type is bytes:
        encoded = base64.b64encode(cast(bytes, value)).decode("ascii")
        return ["bytes", encoded]
    if value_type is list:
        return [
            "list",
            [
                _control_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(cast(list[object], value))
            ],
        ]
    if value_type is tuple:
        return [
            "tuple",
            [
                _control_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(cast(tuple[object, ...], value))
            ],
        ]
    if value_type is dict:
        items: list[list[object]] = []
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise TypeError(f"{path} mapping keys must be exact strings")
            items.append(
                [
                    cast(str, key),
                    _control_value(item, path=f"{path}[{key!r}]")
                ]
            )
        items.sort(key=lambda pair: cast(str, pair[0]))
        return ["dict", items]
    raise TypeError(
        f"{path} must contain only deterministic data-only control values; "
        f"got {value_type.__module__}.{value_type.__qualname__}"
    )


def _control_digest(value: object) -> str:
    canonical = _control_value(value, path="distributed control value")
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ProcessGroupCollectives:
    """Control collectives backed by one active process-group runtime."""

    def __init__(
        self,
        *,
        runtime: ProcessGroupRuntime,
        topology: DistributedTopology,
        device: torch.device,
    ) -> None:
        self.runtime = runtime
        self.topology = topology
        self.device = device

    def _require_active(self) -> None:
        if not self.runtime.is_initialized():
            raise RuntimeError("distributed session is not active")

    def broadcast_from_primary(self, value: object) -> object:
        self._require_active()
        values = [value if self.topology.is_primary else None]
        self.runtime.broadcast_object_list(values, source_rank=0)
        return values[0]

    def gather_to_primary(self, value: object) -> tuple[object, ...] | None:
        self._require_active()
        output: list[object] | None = (
            [None for _ in range(self.topology.world_size)]
            if self.topology.is_primary
            else None
        )
        self.runtime.gather_object(value, output, destination_rank=0)
        return tuple(output) if output is not None else None

    def all_true(self, value: bool) -> bool:
        if type(value) is not bool:
            raise TypeError("all_true value must be a bool")
        return self.min_int(int(value)) == 1

    def all_equal(self, value: object) -> bool:
        self._require_active()
        digest = _control_digest(value)
        output: list[object] = [None for _ in range(self.topology.world_size)]
        self.runtime.all_gather_object(output, digest)
        if any(
            type(item) is not str or len(cast(str, item)) != 64
            for item in output
        ):
            raise RuntimeError(
                "distributed all_equal received an invalid control digest"
            )
        return all(item == output[0] for item in output[1:])

    def _reduce_int(self, value: int, reduction: DistributedReduction) -> int:
        self._require_active()
        if type(value) is not int:
            raise TypeError("distributed integer reduction requires an exact integer")
        tensor = torch.tensor(value, dtype=torch.int64, device=self.device)
        self.runtime.all_reduce(tensor, reduction=reduction)
        return int(tensor.item())

    def sum_int(self, value: int) -> int:
        return self._reduce_int(value, "sum")

    def sum_float(self, value: float) -> float:
        self._require_active()
        value_object = cast(object, value)
        if isinstance(value_object, bool) or not isinstance(
            value_object,
            (int, float),
        ):
            raise TypeError("distributed float reduction requires a finite number")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("distributed float reduction requires a finite number")
        tensor = torch.tensor(normalized, dtype=torch.float64, device=self.device)
        self.runtime.all_reduce(tensor, reduction="sum")
        result = float(tensor.item())
        if not math.isfinite(result):
            raise RuntimeError("distributed float sum produced a non-finite result")
        return result

    def min_int(self, value: int) -> int:
        return self._reduce_int(value, "min")

    def max_int(self, value: int) -> int:
        return self._reduce_int(value, "max")


class DistributedSession:
    """Own one fixed process group from initialization through teardown."""

    def __init__(
        self,
        topology: DistributedTopology,
        *,
        backend: str,
        timeout: timedelta,
        runtime: ProcessGroupRuntime,
        device_binder: DistributedDeviceBinder,
    ) -> None:
        topology_value = cast(object, topology)
        if not isinstance(topology_value, DistributedTopology):
            raise TypeError("topology must be DistributedTopology")
        if backend not in {"nccl", "gloo"}:
            raise ValueError("distributed backend must be 'nccl' or 'gloo'")
        timeout_value = cast(object, timeout)
        if not isinstance(timeout_value, timedelta) or timeout <= timedelta(0):
            raise ValueError("distributed timeout must be a positive timedelta")
        if not callable(device_binder):
            raise TypeError("device_binder must be callable")
        self.topology = topology
        self.backend = backend
        self.timeout = timeout
        self.runtime = runtime
        self.device_binder = device_binder
        self._device: torch.device | None = None
        self._collectives: DistributedCollectives | None = None
        self._entered = False
        self._owns_process_group = False

    @classmethod
    def from_environment(
        cls,
        *,
        backend: str = "nccl",
        timeout: timedelta = timedelta(minutes=10),
        environ: Mapping[str, str] | None = None,
        runtime: ProcessGroupRuntime | None = None,
        device_binder: DistributedDeviceBinder | None = None,
    ) -> DistributedSession:
        """Construct an unentered session from a validated torchrun environment."""

        return cls(
            parse_torchrun_environment(environ),
            backend=backend,
            timeout=timeout,
            runtime=(
                runtime if runtime is not None else TorchProcessGroupRuntime()
            ),
            device_binder=(
                device_binder
                if device_binder is not None
                else _default_device_binder
            ),
        )

    @property
    def is_primary(self) -> bool:
        """Return whether this process owns rank-zero side effects."""

        return self.topology.is_primary

    @property
    def device(self) -> torch.device:
        """Return the process-local device while the session is active."""

        if self._device is None or not self._owns_process_group:
            raise RuntimeError("distributed session is not active")
        return self._device

    @property
    def collectives(self) -> DistributedCollectives:
        """Return the narrow collective surface while the session is active."""

        if self._collectives is None or not self._owns_process_group:
            raise RuntimeError("distributed session is not active")
        return self._collectives

    @property
    def data_rank_context(self) -> DataRankContext:
        """Project only rank and world size into public Data composition."""

        return DataRankContext(
            rank=self.topology.rank,
            world_size=self.topology.world_size,
        )

    def __enter__(self) -> Self:
        if self._entered:
            raise RuntimeError("distributed session cannot be entered more than once")
        self._entered = True
        if not self.runtime.is_available():
            raise RuntimeError("this Torch build does not provide distributed support")
        if self.runtime.is_initialized():
            raise RuntimeError(
                "distributed session refuses to adopt an existing process group"
            )
        try:
            device = self.device_binder(self.backend, self.topology)
            _validate_bound_device(self.backend, self.topology, device)
            self._device = device
            try:
                self.runtime.initialize(
                    backend=self.backend,
                    rank=self.topology.rank,
                    world_size=self.topology.world_size,
                    timeout=self.timeout,
                )
            finally:
                self._owns_process_group = self.runtime.is_initialized()
            if not self._owns_process_group:
                raise RuntimeError(
                    "process-group initialization returned without an active group"
                )
            if self.runtime.rank() != self.topology.rank:
                raise RuntimeError(
                    "initialized process-group rank differs from torchrun RANK"
                )
            if self.runtime.world_size() != self.topology.world_size:
                raise RuntimeError(
                    "initialized process-group world size differs from torchrun "
                    "WORLD_SIZE"
                )
            if self.runtime.backend().casefold() != self.backend:
                raise RuntimeError(
                    "initialized process-group backend differs from the requested "
                    f"backend {self.backend!r}"
                )
            self._collectives = ProcessGroupCollectives(
                runtime=self.runtime,
                topology=self.topology,
                device=device,
            )
            return self
        except BaseException as error:
            self._destroy_after_failure(error)
            raise

    def _destroy_after_failure(self, error: BaseException) -> None:
        if not self._owns_process_group:
            return
        try:
            self.runtime.destroy()
        except BaseException as cleanup_error:  # noqa: BLE001
            with suppress(BaseException):
                BaseException.add_note(
                    error,
                    "distributed process-group teardown also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}",
                )
        finally:
            self._owns_process_group = False
            self._collectives = None
            self._device = None

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exception_type, traceback
        if not self._owns_process_group:
            return False
        try:
            self.runtime.destroy()
        except BaseException as cleanup_error:
            if exception is None:
                raise
            with suppress(BaseException):
                BaseException.add_note(
                    exception,
                    "distributed process-group teardown also failed: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}",
                )
        finally:
            self._owns_process_group = False
            self._collectives = None
            self._device = None
        return False


__all__ = [
    "DistributedSession",
    "parse_torchrun_environment",
]

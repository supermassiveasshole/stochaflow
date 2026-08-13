"""Deterministic fingerprints for fixed-DDP runtime state."""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn


def _storage_key(tensor: torch.Tensor) -> tuple[str, int]:
    """Return process-local storage identity, including zero-sized storage."""

    storage = tensor.untyped_storage()
    data_pointer = storage.data_ptr()
    return (str(tensor.device), data_pointer or storage._cdata)


def _storage_interval(tensor: torch.Tensor) -> tuple[str, int, int] | None:
    """Return the externally observable non-empty storage address interval."""

    storage = tensor.untyped_storage()
    start = storage.data_ptr()
    size = storage.nbytes()
    if start == 0 or size == 0:
        return None
    return (str(tensor.device), start, start + size)


def _require_disjoint_storage_interval(
    tensor: torch.Tensor,
    *,
    owner: tuple[str, int],
    intervals: list[tuple[str, int, int, str, int]],
    path: str,
) -> None:
    """Reject overlapping non-empty storage across distinct Tensor objects."""

    interval = _storage_interval(tensor)
    if interval is None:
        return
    device, start, stop = interval
    owner_name, owner_id = owner
    for prior_device, prior_start, prior_stop, prior_name, prior_id in intervals:
        if (
            device == prior_device
            and owner_id != prior_id
            and start < prior_stop
            and prior_start < stop
        ):
            raise ValueError(
                f"{path} tensors {prior_name!r} and {owner_name!r} use "
                "overlapping storage through distinct tensor objects"
            )
    intervals.append((device, start, stop, owner_name, owner_id))


def module_runtime_state(module: nn.Module) -> dict[str, object]:
    """Return persistent, extra, and non-persistent state for one module tree."""

    module_type = type(module)
    parameter_aliases: dict[int, int] = {}
    buffer_aliases: dict[int, int] = {}
    storage_aliases: dict[tuple[str, int], int] = {}
    module_aliases: dict[int, int] = {id(module): 0}
    pending_modules: list[nn.Module] = [module]
    module_schema: list[dict[str, object]] = []

    while pending_modules:
        parent = pending_modules.pop(0)
        slots: list[dict[str, object]] = []
        for name, child in parent._modules.items():
            if child is None:
                slots.append({"name": name, "child_alias": None})
                continue
            child_id = id(child)
            is_new = child_id not in module_aliases
            child_alias = module_aliases.setdefault(child_id, len(module_aliases))
            slots.append({"name": name, "child_alias": child_alias})
            if is_new:
                pending_modules.append(child)
        parent_type = type(parent)
        parameter_slots = [
            {
                "name": name,
                "tensor_alias": (
                    None
                    if value is None
                    else parameter_aliases.setdefault(
                        id(value),
                        len(parameter_aliases),
                    )
                ),
            }
            for name, value in parent._parameters.items()
        ]
        buffer_slots = [
            {
                "name": name,
                "tensor_alias": (
                    None
                    if value is None
                    else buffer_aliases.setdefault(
                        id(value),
                        len(buffer_aliases),
                    )
                ),
                "persistent": name not in parent._non_persistent_buffers_set,
            }
            for name, value in parent._buffers.items()
        ]
        module_schema.append(
            {
                "alias": module_aliases[id(parent)],
                "type": f"{parent_type.__module__}.{parent_type.__qualname__}",
                "slots": slots,
                "parameter_slots": parameter_slots,
                "buffer_slots": buffer_slots,
            }
        )

    def tensor_entry(
        name: str,
        value: torch.Tensor,
        *,
        aliases: dict[int, int],
    ) -> dict[str, object]:
        storage_alias: int | None = None
        if value.layout == torch.strided:
            storage_key = _storage_key(value)
            storage_alias = storage_aliases.setdefault(
                storage_key,
                len(storage_aliases),
            )
        return {
            "name": name,
            "alias": aliases.setdefault(id(value), len(aliases)),
            "storage_alias": storage_alias,
            "storage_offset": value.storage_offset(),
            "stride": value.stride(),
            "value": value,
        }

    state_dict = module.state_dict()
    state_dict_metadata = getattr(state_dict, "_metadata", None)
    return {
        "type": f"{module_type.__module__}.{module_type.__qualname__}",
        "module_schema": module_schema,
        "state_dict": state_dict,
        "state_dict_metadata": state_dict_metadata,
        "parameters": [
            {
                **tensor_entry(name, value, aliases=parameter_aliases),
                "requires_grad": value.requires_grad,
            }
            for name, value in module.named_parameters(remove_duplicate=False)
        ],
        "buffers": [
            {
                **tensor_entry(name, value, aliases=buffer_aliases),
                "persistent": name in module.state_dict(),
            }
            for name, value in module.named_buffers(remove_duplicate=False)
        ],
    }


def require_no_distinct_shared_storage_across_modules(
    modules: Mapping[str, nn.Module],
    *,
    path: str,
) -> None:
    """Reject distinct registered tensors backed by one storage allocation."""

    storage_owners: dict[tuple[str, int], tuple[str, int]] = {}
    storage_intervals: list[tuple[str, int, int, str, int]] = []
    tensor_owners: dict[int, tuple[str, str]] = {}
    for role, module in modules.items():
        state = [
            *module.named_parameters(recurse=True, remove_duplicate=False),
            *module.named_buffers(recurse=True, remove_duplicate=False),
        ]
        for name, tensor in state:
            qualified_name = f"{role}.{name}"
            tensor_id = id(tensor)
            previous_tensor = tensor_owners.setdefault(
                tensor_id,
                (qualified_name, role),
            )
            if previous_tensor[1] != role:
                raise ValueError(
                    f"{path} tensors {previous_tensor[0]!r} and "
                    f"{qualified_name!r} are one object shared across managed "
                    "roots; separate checkpoint state dictionaries cannot "
                    "preserve cross-root aliases"
                )
            if previous_tensor[0] != qualified_name:
                continue
            if tensor.layout != torch.strided:
                raise ValueError(
                    f"{path} tensor {qualified_name!r} must use strided storage"
                )
            _require_disjoint_storage_interval(
                tensor,
                owner=(qualified_name, tensor_id),
                intervals=storage_intervals,
                path=path,
            )
            storage_key = _storage_key(tensor)
            previous = storage_owners.setdefault(
                storage_key,
                (qualified_name, tensor_id),
            )
            if previous[1] != tensor_id:
                raise ValueError(
                    f"{path} tensors {previous[0]!r} and {qualified_name!r} "
                    "use shared storage through distinct tensor objects; fixed "
                    "DDP requires ordinary registered state or exact "
                    "tied-object aliases"
                )


def require_no_distinct_shared_storage(module: nn.Module, *, path: str) -> None:
    """Reject distinct registered tensors backed by one storage allocation."""

    require_no_distinct_shared_storage_across_modules(
        {"module": module},
        path=path,
    )


def runtime_state_fingerprint(value: object, *, path: str) -> str:
    """Hash one bounded data-only runtime-state value."""

    digest = hashlib.sha256()
    _update_runtime_state_fingerprint(
        digest,
        value,
        path=path,
        tensor_aliases={},
        storage_aliases={},
    )
    return digest.hexdigest()


def require_clone_safe_runtime_state(value: object, *, path: str) -> None:
    """Reject tensor aliases that the v12 data-only clone cannot preserve."""

    tensor_owners: dict[int, str] = {}
    storage_owners: dict[tuple[str, int], tuple[str, int]] = {}
    storage_intervals: list[tuple[str, int, int, str, int]] = []
    container_owners: dict[int, str] = {}

    def visit(item: object, *, item_path: str) -> None:
        if isinstance(item, torch.Tensor):
            if type(item) is not torch.Tensor:
                raise TypeError(
                    f"{path} tensor {item_path!r} must be an exact Tensor"
                )
            if item.requires_grad:
                raise ValueError(
                    f"{path} tensor {item_path!r} must not require gradients"
                )
            if item.layout != torch.strided:
                raise TypeError(
                    f"{path} tensor {item_path!r} must use strided storage"
                )
            tensor_id = id(item)
            previous_tensor = tensor_owners.setdefault(tensor_id, item_path)
            if previous_tensor != item_path:
                raise ValueError(
                    f"{path} tensors {previous_tensor!r} and {item_path!r} are "
                    "the same object; exact checkpoint cloning does not support "
                    "runtime tensor aliases"
                )
            _require_clone_preserved_tensor_layout(
                item,
                path=f"{path} tensor {item_path!r}",
            )
            _require_disjoint_storage_interval(
                item,
                owner=(item_path, tensor_id),
                intervals=storage_intervals,
                path=path,
            )
            storage_key = _storage_key(item)
            previous_storage = storage_owners.setdefault(
                storage_key,
                (item_path, tensor_id),
            )
            if previous_storage[1] != tensor_id:
                raise ValueError(
                    f"{path} tensors {previous_storage[0]!r} and {item_path!r} "
                    "share storage through distinct objects; exact checkpoint "
                    "cloning does not support runtime storage aliases"
                )
            return
        if type(item) in {dict, OrderedDict}:
            previous = container_owners.setdefault(id(item), item_path)
            if previous != item_path:
                raise ValueError(
                    f"{path} containers {previous!r} and {item_path!r} are "
                    "the same object; exact checkpoint cloning does not support "
                    "runtime container aliases"
                )
            for key, child in cast(Mapping[object, object], item).items():
                visit(child, item_path=f"{item_path}[{key!r}]")
            if type(item) is OrderedDict:
                attributes = vars(item)
                unexpected = set(attributes) - {"_metadata"}
                if unexpected:
                    names = ", ".join(sorted(unexpected))
                    raise TypeError(
                        f"{item_path} has unsupported OrderedDict attributes: {names}"
                    )
                if "_metadata" in attributes:
                    visit(
                        attributes["_metadata"],
                        item_path=f"{item_path}._metadata",
                    )
            return
        if type(item) in {tuple, list}:
            previous = container_owners.setdefault(id(item), item_path)
            if previous != item_path:
                raise ValueError(
                    f"{path} containers {previous!r} and {item_path!r} are "
                    "the same object; exact checkpoint cloning does not support "
                    "runtime container aliases"
                )
            sequence = cast(tuple[object, ...] | list[object], item)
            for index, child in enumerate(sequence):
                visit(child, item_path=f"{item_path}[{index}]")
            return
        if type(item) not in {
            type(None),
            bool,
            int,
            float,
            complex,
            str,
            bytes,
        }:
            raise TypeError(
                f"{item_path} contains unsupported checkpoint value type "
                f"{type(item).__module__}.{type(item).__qualname__}"
            )

    visit(value, item_path=path)


def _require_clone_preserved_tensor_layout(
    tensor: torch.Tensor,
    *,
    path: str,
) -> None:
    """Require ordinary cloning to preserve the Tensor's complete layout."""

    if tensor.layout != torch.strided:
        raise TypeError(f"{path} must use strided storage")
    if tensor.is_conj() or tensor.is_neg():
        raise ValueError(
            f"{path} uses a lazy conjugate or negative view that exact "
            "checkpoint cloning cannot preserve"
        )
    if tensor.is_pinned():
        raise ValueError(
            f"{path} uses pinned host memory that exact checkpoint cloning "
            "cannot preserve"
        )
    if tensor.is_shared():
        raise ValueError(
            f"{path} uses shared host memory that exact checkpoint cloning "
            "cannot preserve"
        )
    if tensor.is_inference():
        raise ValueError(
            f"{path} is an inference Tensor that exact checkpoint cloning "
            "cannot preserve"
        )
    if tensor.is_quantized:
        raise ValueError(
            f"{path} is quantized state whose quantization parameters are "
            "outside the first fixed-DDP checkpoint contract"
        )
    cloned = tensor.detach().clone()
    if (
        cloned.storage_offset() != tensor.storage_offset()
        or cloned.stride() != tensor.stride()
        or cloned.untyped_storage().nbytes()
        != tensor.untyped_storage().nbytes()
    ):
        raise ValueError(
            f"{path} has storage topology that exact checkpoint cloning "
            "cannot preserve"
        )


def require_runtime_state_disjoint_from_modules(
    value: object,
    modules: Mapping[str, nn.Module],
    *,
    path: str,
) -> None:
    """Reject runtime Tensor state that aliases any managed module Tensor."""

    module_tensor_ids: dict[int, str] = {}
    module_storage: dict[tuple[str, int], str] = {}
    module_storage_intervals: list[tuple[str, int, int, str, int]] = []
    module_container_ids: dict[int, str] = {}

    def record_module_tensor(tensor: torch.Tensor, *, tensor_path: str) -> None:
        module_tensor_ids.setdefault(id(tensor), tensor_path)
        if tensor.layout == torch.strided:
            _require_disjoint_storage_interval(
                tensor,
                owner=(tensor_path, id(tensor)),
                intervals=module_storage_intervals,
                path=path,
            )
            module_storage.setdefault(
                _storage_key(tensor),
                tensor_path,
            )

    def record_module_value(item: object, *, item_path: str) -> None:
        if isinstance(item, torch.Tensor):
            record_module_tensor(item, tensor_path=item_path)
            return
        if type(item) in {dict, OrderedDict}:
            module_container_ids.setdefault(id(item), item_path)
            for key, child in cast(Mapping[object, object], item).items():
                record_module_value(child, item_path=f"{item_path}[{key!r}]")
            if type(item) is OrderedDict and "_metadata" in vars(item):
                record_module_value(
                    vars(item)["_metadata"],
                    item_path=f"{item_path}._metadata",
                )
        elif type(item) in {tuple, list}:
            module_container_ids.setdefault(id(item), item_path)
            sequence = cast(tuple[object, ...] | list[object], item)
            for index, child in enumerate(sequence):
                record_module_value(child, item_path=f"{item_path}[{index}]")

    for role, module in modules.items():
        state = [
            *module.named_parameters(recurse=True, remove_duplicate=False),
            *module.named_buffers(recurse=True, remove_duplicate=False),
        ]
        for name, tensor in state:
            qualified_name = f"{role}.{name}"
            record_module_tensor(tensor, tensor_path=qualified_name)
        state_dict = module.state_dict(keep_vars=True)
        registered_ids = {id(tensor) for _, tensor in state}
        for name, item in state_dict.items():
            if isinstance(item, torch.Tensor) and id(item) in registered_ids:
                continue
            record_module_value(item, item_path=f"{role}.state_dict[{name!r}]")
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            record_module_value(metadata, item_path=f"{role}.state_dict._metadata")

    def visit(item: object, *, item_path: str) -> None:
        if isinstance(item, torch.Tensor):
            registered_name = module_tensor_ids.get(id(item))
            if registered_name is not None:
                raise ValueError(
                    f"{path} tensor {item_path!r} is managed module state "
                    f"{registered_name!r}; separate checkpoint sections cannot "
                    "preserve that alias"
                )
            if item.layout == torch.strided:
                registered_name = module_storage.get(_storage_key(item))
                if registered_name is not None:
                    raise ValueError(
                        f"{path} tensor {item_path!r} shares storage with managed "
                        f"module state {registered_name!r}"
                    )
                interval = _storage_interval(item)
                if interval is not None:
                    device, start, stop = interval
                    for (
                        prior_device,
                        prior_start,
                        prior_stop,
                        prior_name,
                        prior_id,
                    ) in module_storage_intervals:
                        if (
                            device == prior_device
                            and id(item) != prior_id
                            and start < prior_stop
                            and prior_start < stop
                        ):
                            raise ValueError(
                                f"{path} tensor {item_path!r} overlaps managed "
                                f"module state {prior_name!r}"
                            )
            return
        if type(item) in {dict, OrderedDict}:
            module_path = module_container_ids.get(id(item))
            if module_path is not None:
                raise ValueError(
                    f"{path} container {item_path!r} is managed module state "
                    f"{module_path!r}; separate checkpoint sections cannot "
                    "preserve that alias"
                )
            for key, child in cast(Mapping[object, object], item).items():
                visit(child, item_path=f"{item_path}[{key!r}]")
            if type(item) is OrderedDict and "_metadata" in vars(item):
                visit(
                    vars(item)["_metadata"],
                    item_path=f"{item_path}._metadata",
                )
        elif type(item) in {tuple, list}:
            module_path = module_container_ids.get(id(item))
            if module_path is not None:
                raise ValueError(
                    f"{path} container {item_path!r} is managed module state "
                    f"{module_path!r}; separate checkpoint sections cannot "
                    "preserve that alias"
                )
            sequence = cast(tuple[object, ...] | list[object], item)
            for index, child in enumerate(sequence):
                visit(child, item_path=f"{item_path}[{index}]")

    visit(value, item_path=path)


def require_tensor_free_runtime_state(value: object, *, path: str) -> None:
    """Reject tensor state whose device cannot be reconstructed generically."""

    if isinstance(value, torch.Tensor):
        raise TypeError(f"{path} must not contain tensor-valued state")
    if isinstance(value, Mapping):
        for key, item in value.items():
            require_tensor_free_runtime_state(item, path=f"{path}[{key!r}]")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            require_tensor_free_runtime_state(item, path=f"{path}[{index}]")


def require_relocatable_module_states(
    modules: Mapping[str, nn.Module],
    *,
    path: str,
) -> None:
    """Validate one complete managed-module graph for exact checkpoint cloning."""

    require_no_distinct_shared_storage_across_modules(modules, path=path)
    registered_storage: set[tuple[str, int]] = set()
    registered_intervals: list[tuple[str, int, int, str, int]] = []
    checked_registered_tensors: set[int] = set()
    for module in modules.values():
        parameters = tuple(module.parameters(recurse=True))
        buffers = tuple(module.buffers(recurse=True))
        if any(buffer.requires_grad for buffer in buffers):
            raise ValueError(f"{path} registered buffers must not require gradients")
        for tensor in (*parameters, *buffers):
            if id(tensor) not in checked_registered_tensors:
                _require_clone_preserved_tensor_layout(
                    tensor,
                    path=f"{path} registered tensor",
                )
                checked_registered_tensors.add(id(tensor))
            if tensor.layout != torch.strided:
                raise ValueError(f"{path} registered state must use strided storage")
            registered_storage.add(_storage_key(tensor))
            interval = _storage_interval(tensor)
            if interval is not None:
                device, start, stop = interval
                registered_intervals.append(
                    (device, start, stop, "registered module state", id(tensor))
                )
    extra_state: OrderedDict[str, object] = OrderedDict()

    def require_host_extra_state(value: object, *, value_path: str) -> None:
        if isinstance(value, torch.Tensor):
            if type(value) is not torch.Tensor:
                raise TypeError(f"{value_path} must be an exact Tensor")
            if value.requires_grad:
                raise ValueError(f"{value_path} must not require gradients")
            if value.device.type != "cpu":
                raise ValueError(
                    f"{value_path} must remain host-resident for exact restore"
                )
            if _storage_key(value) in registered_storage:
                raise ValueError(
                    f"{value_path} shares storage with registered module state"
                )
            interval = _storage_interval(value)
            if interval is not None:
                device, start, stop = interval
                for (
                    prior_device,
                    prior_start,
                    prior_stop,
                    _prior_name,
                    prior_id,
                ) in registered_intervals:
                    if (
                        device == prior_device
                        and id(value) != prior_id
                        and start < prior_stop
                        and prior_start < stop
                    ):
                        raise ValueError(
                            f"{value_path} overlaps registered module state"
                        )
            return
        if type(value) in {dict, OrderedDict}:
            for key, item in cast(Mapping[object, object], value).items():
                require_host_extra_state(
                    item,
                    value_path=f"{value_path}[{key!r}]",
                )
        elif type(value) in {tuple, list}:
            for index, item in enumerate(cast(tuple[object, ...] | list[object], value)):
                require_host_extra_state(
                    item,
                    value_path=f"{value_path}[{index}]",
                )
        elif type(value) not in {
            type(None),
            bool,
            int,
            float,
            complex,
            str,
            bytes,
        }:
            raise TypeError(
                f"{value_path} contains unsupported checkpoint value type "
                f"{type(value).__module__}.{type(value).__qualname__}"
            )

    for role, module in modules.items():
        parameter_ids = {id(tensor) for tensor in module.parameters(recurse=True)}
        canonical_state_entries: dict[str, int] = {}
        for submodule_path, submodule in module.named_modules(remove_duplicate=False):
            prefix = f"{submodule_path}." if submodule_path else ""
            for name, parameter in submodule._parameters.items():
                if parameter is not None:
                    canonical_state_entries[prefix + name] = id(parameter)
            for name, buffer in submodule._buffers.items():
                if (
                    buffer is not None
                    and name not in submodule._non_persistent_buffers_set
                ):
                    canonical_state_entries[prefix + name] = id(buffer)
        canonical_state_ids = set(canonical_state_entries.values())
        present_parameters: set[int] = set()
        state_dict = module.state_dict(keep_vars=True)
        for name, value in state_dict.items():
            canonical_id = canonical_state_entries.get(name)
            if canonical_id is not None:
                if not isinstance(value, torch.Tensor) or id(value) != canonical_id:
                    raise ValueError(
                        f"{path} {role} state entry {name!r} does not expose its "
                        "canonical registered tensor"
                    )
                if canonical_id in parameter_ids:
                    present_parameters.add(id(value))
                continue
            value_path = f"{path} {role} state entry {name!r}"
            if isinstance(value, torch.Tensor) and id(value) in canonical_state_ids:
                raise ValueError(
                    f"{value_path} reuses registered module state as extra state"
                )
            require_host_extra_state(value, value_path=value_path)
            extra_state[f"{role}.{name}"] = value
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            metadata_path = f"{path} {role} state_dict metadata"
            require_host_extra_state(metadata, value_path=metadata_path)
            extra_state[f"{role}.__metadata__"] = metadata
        if present_parameters != parameter_ids:
            raise ValueError(
                f"{path} {role} state_dict must contain every module parameter"
            )
    require_clone_safe_runtime_state(extra_state, path=f"{path} extra state")


def require_relocatable_module_state(module: nn.Module, *, path: str) -> None:
    """Validate one module through the complete managed-state admission."""

    require_relocatable_module_states({"module": module}, path=path)


def update_runtime_state_fingerprint(
    digest: Any,
    value: object,
    *,
    path: str,
) -> None:
    """Append one supported runtime-state value to an existing digest."""

    _update_runtime_state_fingerprint(
        digest,
        value,
        path=path,
        tensor_aliases={},
        storage_aliases={},
    )


def _update_runtime_state_fingerprint(
    digest: Any,
    value: object,
    *,
    path: str,
    tensor_aliases: dict[int, int],
    storage_aliases: dict[tuple[str, int], int],
) -> None:
    """Append a value while preserving tensor and storage alias topology."""

    if value is None:
        digest.update(b"none")
        return
    if type(value) is bool:
        digest.update(b"bool:1" if value else b"bool:0")
        return
    if type(value) is int:
        digest.update(f"int:{value}".encode())
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        digest.update(f"float:{value.hex()}".encode())
        return
    if type(value) is complex:
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError(f"{path} contains a non-finite complex number")
        digest.update(f"complex:{value.real.hex()}:{value.imag.hex()}".encode())
        return
    if type(value) is str:
        payload = value.encode("utf-8")
        digest.update(b"str:")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        return
    if type(value) is bytes:
        digest.update(b"bytes:")
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
        return
    if type(value) in {torch.Tensor, nn.Parameter}:
        tensor_value = cast(torch.Tensor, value)
        if tensor_value.layout != torch.strided:
            raise TypeError(f"{path} contains unsupported non-strided tensor state")
        tensor_alias = tensor_aliases.setdefault(id(tensor_value), len(tensor_aliases))
        storage_alias: int | None = None
        storage_nbytes: int | None = None
        if tensor_value.layout == torch.strided:
            storage_key = _storage_key(tensor_value)
            storage_alias = storage_aliases.setdefault(
                storage_key,
                len(storage_aliases),
            )
            storage_nbytes = tensor_value.untyped_storage().nbytes()
        tensor = tensor_value.detach().to(device="cpu").contiguous()
        tensor_kind = "parameter" if type(value) is nn.Parameter else "tensor"
        metadata = (
            f"{tensor_kind}:{tensor.dtype}:{tuple(tensor.shape)}:"
            f"device={tensor_value.device.type}:"
            f"pinned={tensor_value.is_pinned()}:"
            f"shared={tensor_value.is_shared()}:"
            f"inference={tensor_value.is_inference()}:"
            f"requires_grad={tensor_value.requires_grad}:"
            f"object={tensor_alias}:storage={storage_alias}:"
            f"storage_nbytes={storage_nbytes}:"
            f"offset={tensor_value.storage_offset()}:"
            f"stride={tuple(tensor_value.stride())}"
        ).encode()
        payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        return
    if type(value) in {dict, OrderedDict}:
        mapping = cast(Mapping[object, object], value)
        mapping_type = type(value)
        digest.update(
            f"mapping:{mapping_type.__module__}.{mapping_type.__qualname__}:"
            f"{len(mapping)}".encode()
        )
        for key, item in mapping.items():
            if type(key) not in {str, int}:
                raise TypeError(f"{path} has unsupported mapping key {key!r}")
            _update_runtime_state_fingerprint(
                digest,
                key,
                path=f"{path}.<key>",
                tensor_aliases=tensor_aliases,
                storage_aliases=storage_aliases,
            )
            _update_runtime_state_fingerprint(
                digest,
                item,
                path=f"{path}[{key!r}]",
                tensor_aliases=tensor_aliases,
                storage_aliases=storage_aliases,
            )
        if type(value) is OrderedDict:
            attributes = vars(value)
            unexpected = set(attributes) - {"_metadata"}
            if unexpected:
                names = ", ".join(sorted(unexpected))
                raise TypeError(
                    f"{path} has unsupported OrderedDict attributes: {names}"
                )
            digest.update(b"ordered-dict-metadata")
            _update_runtime_state_fingerprint(
                digest,
                attributes.get("_metadata"),
                path=f"{path}._metadata",
                tensor_aliases=tensor_aliases,
                storage_aliases=storage_aliases,
            )
        return
    if type(value) in {tuple, list}:
        sequence = cast(tuple[object, ...] | list[object], value)
        digest.update(f"{type(value).__name__}:{len(sequence)}".encode())
        for index, item in enumerate(sequence):
            _update_runtime_state_fingerprint(
                digest,
                item,
                path=f"{path}[{index}]",
                tensor_aliases=tensor_aliases,
                storage_aliases=storage_aliases,
            )
        return
    raise TypeError(f"{path} contains unsupported state type {type(value).__name__}")


__all__ = [
    "module_runtime_state",
    "require_clone_safe_runtime_state",
    "require_no_distinct_shared_storage",
    "require_no_distinct_shared_storage_across_modules",
    "require_relocatable_module_state",
    "require_relocatable_module_states",
    "require_runtime_state_disjoint_from_modules",
    "require_tensor_free_runtime_state",
    "runtime_state_fingerprint",
    "update_runtime_state_fingerprint",
]

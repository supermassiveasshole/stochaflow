"""Strict private configuration for the AFHQ-v2 class-image recipe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast


def _mapping(value: object, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} field names must be strings")
    return cast(dict[str, Any], dict(value))


def _check_fields(
    raw: dict[str, Any],
    *,
    allowed: set[str],
    required: set[str] | None = None,
    path: str,
) -> None:
    missing = sorted((required or set()) - set(raw))
    unknown = sorted(set(raw) - allowed)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{path} has invalid fields: {'; '.join(details)}")


def _boolean(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{path} must be boolean")
    return value


def _positive_integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class AFHQV2SourceMaterializationConfig:
    """Cache and verification policy for the AFHQ-v2 artifact."""

    cache_root: str = "./data"
    policy: Literal["require", "ensure"] = "ensure"
    verification: Literal["manifest", "full"] = "full"

    @classmethod
    def from_raw(
        cls,
        value: object,
        *,
        path: str,
    ) -> AFHQV2SourceMaterializationConfig:
        """Parse the source materialization envelope."""

        raw = _mapping(value, path=path)
        _check_fields(
            raw,
            allowed={"cache_root", "policy", "verification"},
            path=path,
        )
        cache_root = raw.get("cache_root", "./data")
        if not isinstance(cache_root, str) or not cache_root.strip():
            raise ValueError(f"{path}.cache_root must be a non-empty path")
        policy = raw.get("policy", "ensure")
        if policy not in {"require", "ensure"}:
            raise ValueError(f"{path}.policy must be require or ensure")
        verification = raw.get("verification", "full")
        if verification not in {"manifest", "full"}:
            raise ValueError(f"{path}.verification must be manifest or full")
        return cls(
            cache_root=cache_root,
            policy=cast(Literal["require", "ensure"], policy),
            verification=cast(Literal["manifest", "full"], verification),
        )


@dataclass(frozen=True, slots=True)
class AFHQV2SourceSelectionConfig:
    """Registered AFHQ-v2 source selection and private source parameters."""

    name: str
    params: dict[str, Any]
    materialization: AFHQV2SourceMaterializationConfig

    @classmethod
    def from_raw(
        cls,
        value: object,
        *,
        path: str,
    ) -> AFHQV2SourceSelectionConfig:
        """Parse one exact source selection."""

        raw = _mapping(value, path=path)
        _check_fields(
            raw,
            allowed={"name", "params", "materialization"},
            required={"name", "params", "materialization"},
            path=path,
        )
        name = raw["name"]
        if name != "afhq-v2.official":
            raise ValueError(
                f"{path}.name must be 'afhq-v2.official'"
            )
        return cls(
            name=cast(str, name),
            params=_mapping(raw["params"], path=f"{path}.params"),
            materialization=AFHQV2SourceMaterializationConfig.from_raw(
                raw["materialization"],
                path=f"{path}.materialization",
            ),
        )


@dataclass(frozen=True, slots=True)
class AFHQV2ImageRecipeConfig:
    """Exact prepared-image geometry and deterministic augmentation policy."""

    height: int
    width: int
    normalize: bool
    random_horizontal_flip: bool

    @classmethod
    def from_raw(
        cls,
        value: object,
        *,
        path: str,
    ) -> AFHQV2ImageRecipeConfig:
        """Parse exact-size RGB image parameters."""

        raw = _mapping(value, path=path)
        _check_fields(
            raw,
            allowed={
                "size",
                "channels",
                "require_exact_size",
                "normalize",
                "random_horizontal_flip",
            },
            required={"size"},
            path=path,
        )
        size = raw["size"]
        if not isinstance(size, list) or len(size) != 2:
            raise ValueError(f"{path}.size must contain [height, width]")
        height = _positive_integer(size[0], path=f"{path}.size[0]")
        width = _positive_integer(size[1], path=f"{path}.size[1]")
        channels = cast(object, raw.get("channels", 3))
        if isinstance(channels, bool) or not isinstance(channels, int) or channels != 3:
            raise ValueError(f"{path}.channels must be 3")
        exact = _boolean(
            raw.get("require_exact_size", True),
            path=f"{path}.require_exact_size",
        )
        if not exact:
            raise ValueError(
                f"{path}.require_exact_size must be true; AFHQ-v2 is "
                "resized during artifact preparation"
            )
        return cls(
            height=height,
            width=width,
            normalize=_boolean(
                raw.get("normalize", True),
                path=f"{path}.normalize",
            ),
            random_horizontal_flip=_boolean(
                raw.get("random_horizontal_flip", True),
                path=f"{path}.random_horizontal_flip",
            ),
        )


@dataclass(frozen=True, slots=True)
class AFHQV2LoaderRecipeConfig:
    """Finite DataLoader policy for prepared AFHQ-v2 partitions."""

    batch_size: int
    num_workers: int
    shuffle: bool
    drop_last: bool
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    steps_per_epoch: int | Literal["auto"]

    @classmethod
    def from_raw(
        cls,
        value: object,
        *,
        path: str,
    ) -> AFHQV2LoaderRecipeConfig:
        """Parse loader settings without a generic loader graph."""

        raw = _mapping(value, path=path)
        _check_fields(
            raw,
            allowed={
                "batch_size",
                "num_workers",
                "shuffle",
                "drop_last",
                "pin_memory",
                "persistent_workers",
                "prefetch_factor",
                "steps_per_epoch",
            },
            path=path,
        )
        num_workers = _nonnegative_integer(
            raw.get("num_workers", 2),
            path=f"{path}.num_workers",
        )
        persistent_workers = _boolean(
            raw.get("persistent_workers", True),
            path=f"{path}.persistent_workers",
        )
        if persistent_workers and num_workers == 0:
            raise ValueError(
                f"{path}.persistent_workers requires num_workers > 0"
            )
        prefetch_value = raw.get("prefetch_factor", 4)
        prefetch_factor = (
            None
            if prefetch_value is None
            else _positive_integer(
                prefetch_value,
                path=f"{path}.prefetch_factor",
            )
        )
        if prefetch_factor is not None and num_workers == 0:
            raise ValueError(
                f"{path}.prefetch_factor requires num_workers > 0"
            )
        steps_value = raw.get("steps_per_epoch", "auto")
        if steps_value == "auto":
            steps_per_epoch: int | Literal["auto"] = "auto"
        else:
            steps_per_epoch = _positive_integer(
                steps_value,
                path=f"{path}.steps_per_epoch",
            )
        return cls(
            batch_size=_positive_integer(
                raw.get("batch_size", 8),
                path=f"{path}.batch_size",
            ),
            num_workers=num_workers,
            shuffle=_boolean(
                raw.get("shuffle", True),
                path=f"{path}.shuffle",
            ),
            drop_last=_boolean(
                raw.get("drop_last", True),
                path=f"{path}.drop_last",
            ),
            pin_memory=_boolean(
                raw.get("pin_memory", True),
                path=f"{path}.pin_memory",
            ),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            steps_per_epoch=steps_per_epoch,
        )


@dataclass(frozen=True, slots=True)
class AFHQV2DataBuilderConfig:
    """Complete private AFHQ-v2 source-to-batch recipe."""

    source: AFHQV2SourceSelectionConfig
    image: AFHQV2ImageRecipeConfig
    loader: AFHQV2LoaderRecipeConfig

    @classmethod
    def from_params(cls, value: object) -> AFHQV2DataBuilderConfig:
        """Parse ``data.params`` and reject modality-wide graph fields."""

        path = "data.params"
        raw = _mapping(value, path=path)
        _check_fields(
            raw,
            allowed={"source", "image", "loader"},
            required={"source", "image", "loader"},
            path=path,
        )
        return cls(
            source=AFHQV2SourceSelectionConfig.from_raw(
                raw["source"],
                path=f"{path}.source",
            ),
            image=AFHQV2ImageRecipeConfig.from_raw(
                raw["image"],
                path=f"{path}.image",
            ),
            loader=AFHQV2LoaderRecipeConfig.from_raw(
                raw["loader"],
                path=f"{path}.loader",
            ),
        )


__all__ = [
    "AFHQV2DataBuilderConfig",
    "AFHQV2ImageRecipeConfig",
    "AFHQV2LoaderRecipeConfig",
    "AFHQV2SourceMaterializationConfig",
    "AFHQV2SourceSelectionConfig",
]

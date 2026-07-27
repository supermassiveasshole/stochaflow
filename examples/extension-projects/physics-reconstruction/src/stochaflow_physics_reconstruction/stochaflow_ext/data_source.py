"""Referenced trajectory artifacts for the physics extension."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from stochaflow.extensions import (
    DataArtifact,
    DataArtifactLoadContext,
    DataArtifactStore,
    DataArtifactValidationError,
    DataSource,
    DataSourceContext,
    ReferencedDataArtifactBuild,
    Registry,
    canonical_artifact_digest,
    canonical_artifact_json_bytes,
)

_SOURCE_NAME = "physics-reconstruction.numpy-trajectories"
_ARTIFACT_TYPE = "physics-reconstruction.numpy-trajectories.v1"
_MATERIALIZER_NAME = "physics-reconstruction.numpy-reference"
_SIDECAR_NAME = "array.json"
_DOMAIN_FIELDS = frozenset({"schema_version", "array"})
_ARRAY_FIELDS = frozenset({"size_bytes", "shape", "dtype"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def open_trajectory_array(path: Path) -> np.ndarray:
    """Open and validate one memory-mapped trajectory array."""

    if not path.is_file():
        raise FileNotFoundError(f"trajectory array does not exist: {path}")
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise TypeError(f"trajectory source must be a NumPy array: {path}")
    if value.ndim != 4:
        raise ValueError(
            "trajectory array must have shape [trajectory, time, height, width]"
        )
    if value.shape[1] < 3:
        raise ValueError("trajectory array must contain at least three time frames")
    if value.shape[2] < 2 or value.shape[3] < 2:
        raise ValueError("trajectory spatial dimensions must be at least 2x2")
    if value.shape[2] != value.shape[3] or value.shape[2] % 2:
        raise ValueError("trajectory fields must use an even, square spectral grid")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError("trajectory array must contain floating-point values")
    return value


@dataclass(frozen=True, slots=True)
class KolmogorovTrajectoryArtifactPayload:
    """Verified external array locator and authenticated array schema."""

    path: Path
    shape: tuple[int, int, int, int]
    dtype: str

    def __post_init__(self) -> None:
        path = Path(self.path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise FileNotFoundError(f"trajectory array does not exist: {path}")
        shape = cast(object, self.shape)
        if (
            not isinstance(shape, tuple)
            or len(shape) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in shape
            )
        ):
            raise ValueError("trajectory artifact shape must contain four dimensions")
        dtype = cast(object, self.dtype)
        if not isinstance(dtype, str) or not dtype:
            raise ValueError("trajectory artifact dtype must be non-empty")
        object.__setattr__(self, "path", path)


class KolmogorovDataSource(DataSource[KolmogorovTrajectoryArtifactPayload]):
    """Extension-local source contract for trajectory artifacts."""

    def __init__(self, params: dict[str, Any], *, config_path: str) -> None:
        params_value = cast(object, params)
        if not isinstance(params_value, dict):
            raise TypeError("physics data source params must be a mapping")
        self.params = deepcopy(params)
        self.config_path = config_path


PHYSICS_DATA_SOURCES: Registry[type[KolmogorovDataSource]] = Registry(
    "physics data source",
    expected_type=KolmogorovDataSource,
)


def _source_path(params: dict[str, Any], *, path: str) -> Path:
    raw = params.pop("path", None)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{path}.path must be a non-empty string")
    if params:
        raise ValueError(f"unknown {path} parameter(s): {', '.join(sorted(params))}")
    resolved = Path(raw).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"trajectory array does not exist: {resolved}")
    return resolved


def _array_record(path: Path, array: np.ndarray) -> dict[str, object]:
    return {
        "size_bytes": path.stat().st_size,
        "shape": [int(value) for value in array.shape],
        "dtype": array.dtype.str,
    }


def _validated_domain(domain: Mapping[str, object]) -> dict[str, object]:
    schema_version = domain.get("schema_version")
    if (
        set(domain) != _DOMAIN_FIELDS
        or type(schema_version) is not int
        or schema_version != 1
    ):
        raise DataArtifactValidationError(
            "trajectory artifact domain envelope is incompatible"
        )
    array = domain.get("array")
    if not isinstance(array, Mapping) or set(array) != _ARRAY_FIELDS:
        raise DataArtifactValidationError(
            "trajectory artifact array domain is incompatible"
        )
    size_bytes = array.get("size_bytes")
    shape = array.get("shape")
    dtype = array.get("dtype")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
        or not isinstance(shape, list)
        or len(shape) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in shape
        )
        or not isinstance(dtype, str)
        or not dtype
    ):
        raise DataArtifactValidationError(
            "trajectory artifact array domain is invalid"
        )
    return {
        "schema_version": 1,
        "array": {
            "size_bytes": size_bytes,
            "shape": list(shape),
            "dtype": dtype,
        },
    }


def _external_content_digest(
    *,
    size_bytes: int,
    sha256: str,
) -> str:
    return canonical_artifact_digest(
        {
            "files": [
                {
                    "path": "trajectory.npy",
                    "size_bytes": size_bytes,
                    "sha256": sha256,
                }
            ]
        }
    )


def _build_reference(
    data_root: Path,
    *,
    path: Path,
) -> ReferencedDataArtifactBuild:
    array = open_trajectory_array(path)
    sha256 = _sha256_file(path)
    domain = {
        "schema_version": 1,
        "array": _array_record(path, array),
    }
    (data_root / _SIDECAR_NAME).write_bytes(
        canonical_artifact_json_bytes(domain)
    )
    return ReferencedDataArtifactBuild(
        source_digest=sha256,
        materialization_digest=canonical_artifact_digest(
            {
                "schema_version": 1,
                "format": "npy",
                "layout": ["trajectory", "time", "height", "width"],
                "grid": "even-square",
                "dtype_family": "floating",
            }
        ),
        content_digest=_external_content_digest(
            size_bytes=path.stat().st_size,
            sha256=sha256,
        ),
        domain=domain,
    )


def _load_reference(
    context: DataArtifactLoadContext,
    *,
    path: Path,
) -> KolmogorovTrajectoryArtifactPayload:
    domain = _validated_domain(context.domain)
    sidecar = context.data_root / _SIDECAR_NAME
    if sidecar.read_bytes() != canonical_artifact_json_bytes(domain):
        raise DataArtifactValidationError(
            "trajectory artifact sidecar does not match its domain"
        )
    array_record = cast(dict[str, object], domain["array"])
    if path.stat().st_size != array_record["size_bytes"]:
        raise DataArtifactValidationError(
            "trajectory array size does not match its artifact"
        )
    try:
        array = open_trajectory_array(path)
    except (TypeError, ValueError) as exc:
        raise DataArtifactValidationError(
            "trajectory array schema is no longer valid"
        ) from exc
    shape = tuple(int(value) for value in array.shape)
    if list(shape) != array_record["shape"] or array.dtype.str != array_record["dtype"]:
        raise DataArtifactValidationError(
            "trajectory array schema does not match its artifact"
        )
    if context.verification == "full":
        sha256 = _sha256_file(path)
        if sha256 != context.identity.source_digest:
            raise DataArtifactValidationError(
                "trajectory array content does not match its source identity"
            )
        content_digest = _external_content_digest(
            size_bytes=path.stat().st_size,
            sha256=sha256,
        )
        if content_digest != context.identity.content_digest:
            raise DataArtifactValidationError(
                "trajectory array content does not match its artifact identity"
            )
    return KolmogorovTrajectoryArtifactPayload(
        path=path,
        shape=cast(tuple[int, int, int, int], shape),
        dtype=array.dtype.str,
    )


@PHYSICS_DATA_SOURCES.register(_SOURCE_NAME)
class NumpyTrajectoryDataSource(KolmogorovDataSource):
    """Index and verify an external mmap-ready NumPy trajectory array."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> DataArtifact[KolmogorovTrajectoryArtifactPayload]:
        path = _source_path(
            dict(self.params),
            path=f"{self.config_path}.params",
        )
        return DataArtifactStore(context).materialize_referenced(
            artifact_type=_ARTIFACT_TYPE,
            source_name=_SOURCE_NAME,
            materializer_name=_MATERIALIZER_NAME,
            locator_key={"schema_version": 1, "path": str(path)},
            referenced_roots={"trajectory_array": path},
            build=lambda data_root: _build_reference(data_root, path=path),
            load=lambda load_context: _load_reference(load_context, path=path),
        )


__all__ = [
    "PHYSICS_DATA_SOURCES",
    "KolmogorovDataSource",
    "KolmogorovTrajectoryArtifactPayload",
    "NumpyTrajectoryDataSource",
    "open_trajectory_array",
]

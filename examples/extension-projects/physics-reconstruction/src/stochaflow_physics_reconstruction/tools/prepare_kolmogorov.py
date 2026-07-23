"""Convert sparse PhysicsNeMo input into aligned mmap-ready Stochaflow data."""

from __future__ import annotations

import argparse
import json
import os
import struct
from math import prod
from pathlib import Path
from uuid import uuid4
from zipfile import BadZipFile, ZIP_STORED, ZipFile, ZipInfo

import numpy as np
import torch
import torch.nn.functional as F

from stochaflow_physics_reconstruction.stochaflow_ext._alignment import (
    write_alignment,
)


def _trajectory_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.ndim != 4:
        raise ValueError("reference must have shape [trajectory, time, height, width]")
    if not np.issubdtype(value.dtype, np.floating):
        raise TypeError("reference must be floating-point")
    return value


_LOCAL_FILE_HEADER = struct.Struct("<4s5H3I2H")
_EXTRA_FIELD_HEADER = struct.Struct("<HH")
_ZIP64_VALUE = struct.Struct("<Q")
_LOCAL_FILE_SIGNATURE = b"PK\x03\x04"
_ZIP64_SIZE_SENTINEL = 0xFFFFFFFF
_ZIP64_EXTRA_FIELD_ID = 0x0001
_CRC_CHUNK_BYTES = 8 * 1024 * 1024


def _validate_zip64_sizes(
    extra: bytes,
    *,
    member_name: str,
    local_compressed_size: int,
    local_file_size: int,
    expected_size: int,
) -> None:
    required: list[str] = []
    if local_file_size == _ZIP64_SIZE_SENTINEL:
        required.append("uncompressed")
    if local_compressed_size == _ZIP64_SIZE_SENTINEL:
        required.append("compressed")
    if not required:
        return

    cursor = 0
    zip64_payload: bytes | None = None
    while cursor < len(extra):
        if len(extra) - cursor < _EXTRA_FIELD_HEADER.size:
            raise ValueError(
                f"sparse member {member_name!r} has malformed ZIP extra metadata"
            )
        field_id, field_size = _EXTRA_FIELD_HEADER.unpack_from(extra, cursor)
        cursor += _EXTRA_FIELD_HEADER.size
        field_end = cursor + field_size
        if field_end > len(extra):
            raise ValueError(
                f"sparse member {member_name!r} has truncated ZIP extra metadata"
            )
        if field_id == _ZIP64_EXTRA_FIELD_ID:
            if zip64_payload is not None:
                raise ValueError(
                    f"sparse member {member_name!r} has duplicate ZIP64 metadata"
                )
            zip64_payload = extra[cursor:field_end]
        cursor = field_end

    if zip64_payload is None:
        raise ValueError(
            f"sparse member {member_name!r} is missing required ZIP64 metadata"
        )
    required_bytes = len(required) * _ZIP64_VALUE.size
    if len(zip64_payload) < required_bytes:
        raise ValueError(
            f"sparse member {member_name!r} has truncated ZIP64 size metadata"
        )
    for index, label in enumerate(required):
        value = _ZIP64_VALUE.unpack_from(
            zip64_payload,
            index * _ZIP64_VALUE.size,
        )[0]
        if value != expected_size:
            raise ValueError(
                f"sparse member {member_name!r} ZIP64 {label} size does not "
                "match its central directory"
            )


def _validate_member_crc(archive: ZipFile, info: ZipInfo) -> None:
    """Stream one member with bounded memory so ``zipfile`` verifies its CRC."""

    try:
        with archive.open(info, mode="r") as member:
            while member.read(_CRC_CHUNK_BYTES):
                pass
    except (BadZipFile, NotImplementedError, RuntimeError) as error:
        raise ValueError(
            f"sparse member {info.filename!r} failed ZIP integrity validation"
        ) from error


def _selected_stored_npy_member(
    path: Path,
    *,
    key: str,
    count: int,
) -> np.memmap:
    """Memory-map the final rows of one uncompressed NPY member in an NPZ."""

    if count <= 0:
        raise ValueError("selected sparse trajectory count must be positive")
    member_name = key if key.endswith(".npy") else f"{key}.npy"
    try:
        with path.open("rb") as source:
            initial_stat = os.fstat(source.fileno())
            with ZipFile(source) as archive:
                matches = [
                    info
                    for info in archive.infolist()
                    if info.filename == member_name
                ]
                if not matches:
                    raise KeyError(f"sparse archive has no key {key!r}")
                if len(matches) != 1:
                    raise ValueError(
                        f"sparse archive contains duplicate member {member_name!r}"
                    )
                info = matches[0]
                if info.compress_type != ZIP_STORED:
                    raise ValueError(
                        f"sparse member {member_name!r} must use ZIP_STORED for "
                        "memory-bounded access"
                    )
                if info.compress_size != info.file_size:
                    raise ValueError(
                        f"stored sparse member {member_name!r} has invalid sizes"
                    )
                if info.flag_bits & 0x1:
                    raise ValueError(
                        f"encrypted sparse member {member_name!r} is unsupported"
                    )
                if info.flag_bits & 0x8:
                    raise ValueError(
                        f"sparse member {member_name!r} uses a data descriptor, "
                        "which is unsupported for direct mapping"
                    )
                member_size = info.file_size
                header_offset = info.header_offset
                central_flags = info.flag_bits
                central_crc = info.CRC
                _validate_member_crc(archive, info)

            archive_size = initial_stat.st_size
            source.seek(header_offset)
            local_header = source.read(_LOCAL_FILE_HEADER.size)
            if len(local_header) != _LOCAL_FILE_HEADER.size:
                raise ValueError(
                    f"sparse member {member_name!r} has a truncated ZIP header"
                )
            (
                signature,
                _version,
                local_flags,
                local_compression,
                _modified_time,
                _modified_date,
                local_crc,
                local_compressed_size,
                local_file_size,
                filename_length,
                extra_length,
            ) = _LOCAL_FILE_HEADER.unpack(local_header)
            if signature != _LOCAL_FILE_SIGNATURE:
                raise ValueError(
                    f"sparse member {member_name!r} has an invalid ZIP header"
                )
            if local_flags != central_flags or local_compression != ZIP_STORED:
                raise ValueError(
                    f"sparse member {member_name!r} has inconsistent ZIP metadata"
                )
            if local_crc != central_crc:
                raise ValueError(
                    f"sparse member {member_name!r} has inconsistent ZIP CRC metadata"
                )
            valid_sizes = {member_size, _ZIP64_SIZE_SENTINEL}
            if local_compressed_size not in valid_sizes:
                raise ValueError(
                    f"sparse member {member_name!r} has an invalid local "
                    "compressed size"
                )
            if local_file_size not in valid_sizes:
                raise ValueError(
                    f"sparse member {member_name!r} has an invalid local file size"
                )
            filename = source.read(filename_length)
            if len(filename) != filename_length:
                raise ValueError(
                    f"sparse member {member_name!r} has a truncated filename"
                )
            encoding = "utf-8" if local_flags & 0x800 else "cp437"
            try:
                local_name = filename.decode(encoding)
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"sparse member {member_name!r} has an invalid encoded filename"
                ) from error
            if local_name != member_name:
                raise ValueError(
                    f"sparse member {member_name!r} has inconsistent local "
                    "filename metadata"
                )
            extra = source.read(extra_length)
            if len(extra) != extra_length:
                raise ValueError(
                    f"sparse member {member_name!r} has truncated ZIP metadata"
                )
            _validate_zip64_sizes(
                extra,
                member_name=member_name,
                local_compressed_size=local_compressed_size,
                local_file_size=local_file_size,
                expected_size=member_size,
            )
            member_offset = source.tell()
            if member_offset + member_size > archive_size:
                raise ValueError(
                    f"sparse member {member_name!r} extends beyond the archive"
                )

            try:
                version = np.lib.format.read_magic(source)
            except (EOFError, ValueError) as error:
                raise ValueError(
                    f"sparse member {member_name!r} has an invalid NPY header"
                ) from error
            if version not in {(1, 0), (2, 0)}:
                raise ValueError(
                    f"sparse member {member_name!r} uses unsupported NPY version "
                    f"{version[0]}.{version[1]}"
                )
            try:
                if version == (1, 0):
                    shape, fortran_order, dtype = (
                        np.lib.format.read_array_header_1_0(source)
                    )
                else:
                    shape, fortran_order, dtype = (
                        np.lib.format.read_array_header_2_0(source)
                    )
            except (EOFError, UnicodeError, ValueError) as error:
                raise ValueError(
                    f"sparse member {member_name!r} has an invalid NPY header"
                ) from error
            array_offset = source.tell()

            if fortran_order:
                raise ValueError(
                    f"sparse member {member_name!r} must be C-contiguous; "
                    "Fortran order is unsupported"
                )
            if len(shape) != 4 or not np.issubdtype(dtype, np.floating):
                raise ValueError(
                    "sparse data must be a floating [trajectory, time, H, W] array"
                )
            if any(dimension <= 0 for dimension in shape):
                raise ValueError("sparse data dimensions must all be positive")
            if count > shape[0]:
                raise ValueError(
                    "selected sparse trajectory count exceeds the archive "
                    "trajectory axis"
                )
            payload_size = prod(shape) * dtype.itemsize
            header_size = array_offset - member_offset
            if header_size + payload_size != member_size:
                raise ValueError(
                    f"sparse member {member_name!r} payload size does not match "
                    "its NPY header"
                )
            row_size = prod(shape[1:]) * dtype.itemsize
            selected_offset = array_offset + (shape[0] - count) * row_size
            mapped = np.memmap(
                source,
                mode="r",
                dtype=dtype,
                offset=selected_offset,
                shape=(count, *shape[1:]),
                order="C",
            )
            final_stat = os.fstat(source.fileno())
            if (
                final_stat.st_dev,
                final_stat.st_ino,
                final_stat.st_size,
                final_stat.st_mtime_ns,
                final_stat.st_ctime_ns,
            ) != (
                initial_stat.st_dev,
                initial_stat.st_ino,
                initial_stat.st_size,
                initial_stat.st_mtime_ns,
                initial_stat.st_ctime_ns,
            ):
                raise ValueError(
                    f"sparse archive changed while validating member {member_name!r}"
                )
            return mapped
    except BadZipFile as error:
        raise ValueError(f"sparse archive is not a valid NPZ file: {path}") from error


def _training_stats(reference: np.ndarray, stop: int) -> tuple[float, float]:
    count = 0
    total = 0.0
    squared = 0.0
    for index in range(stop):
        values = np.asarray(reference[index], dtype=np.float64)
        count += values.size
        total += float(values.sum(dtype=np.float64))
        squared += float(np.square(values).sum(dtype=np.float64))
    mean = total / count
    variance = max(squared / count - mean * mean, 0.0)
    scale = variance**0.5
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("reference training statistics are not finite and non-zero")
    return mean, scale


def _resize_and_smooth(
    sparse: np.ndarray,
    *,
    spatial_shape: tuple[int, int],
    smoothing_kernel: int,
) -> np.ndarray:
    tensor = torch.from_numpy(np.array(sparse, dtype=np.float32, copy=True))
    flat = tensor.reshape(-1, 1, *tensor.shape[-2:])
    if tuple(flat.shape[-2:]) != spatial_shape:
        flat = F.interpolate(
            flat,
            size=spatial_shape,
            mode="bicubic",
            align_corners=False,
        )
    if smoothing_kernel:
        if smoothing_kernel <= 0 or smoothing_kernel % 2 == 0:
            raise ValueError("smoothing_kernel must be zero or a positive odd integer")
        radius = smoothing_kernel // 2
        coordinate = torch.arange(-radius, radius + 1, dtype=flat.dtype)
        gaussian = torch.exp(-0.5 * (coordinate / smoothing_kernel).square())
        gaussian = gaussian / gaussian.sum()
        kernel = torch.outer(gaussian, gaussian).reshape(1, 1, smoothing_kernel, smoothing_kernel)
        flat = F.conv2d(F.pad(flat, (radius,) * 4, mode="circular"), kernel)
    return flat.reshape(*tensor.shape[:-2], *spatial_shape).numpy()


def prepare(
    *,
    reference_path: Path,
    sparse_path: Path,
    sparse_key: str,
    output_dir: Path,
    held_out_trajectories: int,
    smoothing_kernel: int,
) -> dict[str, Path]:
    """Prepare paired observation data and an audited alignment sidecar."""

    reference = _trajectory_array(reference_path)
    if held_out_trajectories <= 0 or held_out_trajectories >= reference.shape[0]:
        raise ValueError("held_out_trajectories must leave at least one training row")
    selected = _selected_stored_npy_member(
        sparse_path,
        key=sparse_key,
        count=held_out_trajectories,
    )
    reference_selected = reference[-held_out_trajectories:]
    if selected.shape[:2] != reference_selected.shape[:2]:
        raise ValueError("sparse and reference trajectory/time axes do not align")
    processed = _resize_and_smooth(
        selected,
        spatial_shape=(reference.shape[2], reference.shape[3]),
        smoothing_kernel=smoothing_kernel,
    )
    del selected
    output_dir.mkdir(parents=True, exist_ok=True)
    observation_path = output_dir / "kolmogorov_observations.npy"
    temporary = output_dir / f".{observation_path.name}.{uuid4().hex}.tmp"
    try:
        mapped = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=processed.shape,
        )
        mapped[:] = processed
        mapped.flush()
        del mapped
        os.replace(temporary, observation_path)
    finally:
        temporary.unlink(missing_ok=True)
    reference_range = (
        reference.shape[0] - held_out_trajectories,
        reference.shape[0],
    )
    observation_range = (0, held_out_trajectories)
    sample_count = held_out_trajectories * (reference.shape[1] - 2)
    alignment_path = output_dir / "kolmogorov-alignment.json"
    write_alignment(
        alignment_path,
        observation_path=observation_path,
        observation_range=observation_range,
        observation_shape=processed.shape,
        reference_path=reference_path,
        reference_range=reference_range,
        reference_shape=reference.shape,
        sample_count=sample_count,
    )
    mean, scale = _training_stats(reference, reference_range[0])
    stats_path = output_dir / "kolmogorov-stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "normalization_mean": mean,
                "normalization_scale": scale,
                "training_trajectories": [0, reference_range[0]],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "observations": observation_path,
        "alignment": alignment_path,
        "statistics": stats_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--sparse-key", default="u3232")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--held-out-trajectories", type=int, default=4)
    parser.add_argument("--smoothing-kernel", type=int, default=7)
    args = parser.parse_args()
    outputs = prepare(
        reference_path=args.reference,
        sparse_path=args.sparse,
        sparse_key=args.sparse_key,
        output_dir=args.output_dir,
        held_out_trajectories=args.held_out_trajectories,
        smoothing_kernel=args.smoothing_kernel,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

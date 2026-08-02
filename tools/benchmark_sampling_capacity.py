"""Measure materialized sampling-output and artifact-writer capacity.

Each measured repeat runs in a fresh child process. The worker constructs the
same ``SamplingOutput`` and invokes the same registered writers used by the
runtime, so the report captures the current public lifecycle rather than a
parallel benchmark-only implementation.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import torch
import yaml

from stochaflow.sampling import (
    SamplingArtifactContext,
    SamplingBatch,
    SamplingObservation,
    SamplingOutput,
    write_sampling_artifacts,
)
from stochaflow.sampling.runtime import validate_sampling_output
from stochaflow.utils.config import ComponentConfig


class ResourceUsage(Protocol):
    ru_maxrss: int


class ResourceModule(Protocol):
    RUSAGE_SELF: int

    def getrusage(self, who: int) -> ResourceUsage: ...


class SysconfProvider(Protocol):
    def __call__(self, name: str, /) -> int: ...


def _load_resource_module() -> ResourceModule | None:
    try:
        module = importlib.import_module("resource")
    except ImportError:  # pragma: no cover - exercised on Windows
        return None
    return cast(ResourceModule, module)


_resource = _load_resource_module()

PROFILE_FORMAT_VERSION = 1
DEFAULT_HOST_BUDGET_FRACTION = 0.70
DEFAULT_PREFLIGHT_FRACTION = 0.50
_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


class WindowsProcessMemoryCounters(ctypes.Structure):
    """ctypes representation of the Win32 PROCESS_MEMORY_COUNTERS structure."""

    _fields_ = [
        ("cb", wintypes.DWORD),
        ("page_fault_count", wintypes.DWORD),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    ]


@dataclass(frozen=True, slots=True)
class CapacityProfile:
    """One executable or projection-only sampling-capacity workload."""

    name: str
    description: str
    num_samples: int
    batch_size: int
    shape: tuple[int, ...]
    dtype: str
    trajectory_observations: int
    pattern: str
    seed: int
    writers: tuple[ComponentConfig, ...]
    execute: bool
    warmup_runs: int
    measured_runs: int
    timeout_seconds: int

    @property
    def element_size(self) -> int:
        """Return bytes per scalar for the configured dtype."""

        return torch.empty((), dtype=_DTYPES[self.dtype]).element_size()

    @property
    def state_bytes(self) -> int:
        """Return the raw bytes in all final sample states."""

        return self.num_samples * math.prod(self.shape) * self.element_size

    @property
    def trajectory_bytes(self) -> int:
        """Return raw bytes in retained trajectory observations."""

        return self.state_bytes * self.trajectory_observations

    def projection(self) -> dict[str, int]:
        """Return structural byte counts for the current materialized API."""

        final = self.state_bytes
        trajectory = self.trajectory_bytes
        tensor_writer_lower_bound = (
            2 * final if trajectory == 0 else final + 3 * trajectory
        )
        return {
            "one_sample_bytes": math.prod(self.shape) * self.element_size,
            "final_state_bytes": final,
            "trajectory_state_bytes": trajectory,
            "sampling_output_live_bytes": final + trajectory,
            "tensor_writer_structural_peak_lower_bound_bytes": (
                tensor_writer_lower_bound
            ),
            "raw_artifact_payload_bytes": final + trajectory,
        }


def load_profiles(path: str | Path) -> dict[str, CapacityProfile]:
    """Load and strictly validate a capacity-profile document."""

    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        raw = cast(object, yaml.safe_load(handle))
    if not isinstance(raw, dict):
        raise TypeError("capacity profile root must be a mapping")
    unknown_root = sorted(set(raw) - {"version", "profiles"})
    if unknown_root:
        raise ValueError(
            "unknown capacity profile root field(s): " + ", ".join(unknown_root)
        )
    version = raw.get("version")
    if isinstance(version, bool) or version != PROFILE_FORMAT_VERSION:
        raise ValueError(
            "capacity profile version must be " f"{PROFILE_FORMAT_VERSION}"
        )
    declarations = cast(object, raw.get("profiles"))
    if not isinstance(declarations, dict) or not declarations:
        raise ValueError("capacity profiles must be a non-empty mapping")
    profiles: dict[str, CapacityProfile] = {}
    for name, declaration in declarations.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("capacity profile names must be non-empty strings")
        profiles[name] = _parse_profile(name, declaration)
    return profiles


def _parse_profile(name: str, raw: object) -> CapacityProfile:
    if not isinstance(raw, dict):
        raise TypeError(f"capacity profile '{name}' must be a mapping")
    allowed = {
        "description",
        "num_samples",
        "batch_size",
        "shape",
        "dtype",
        "trajectory_observations",
        "pattern",
        "seed",
        "writers",
        "execute",
        "warmup_runs",
        "measured_runs",
        "timeout_seconds",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"unknown capacity profile '{name}' field(s): " + ", ".join(unknown)
        )
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise TypeError(f"capacity profile '{name}' description must be a string")
    num_samples = _positive_int(raw.get("num_samples"), f"{name}.num_samples")
    batch_size = _positive_int(raw.get("batch_size"), f"{name}.batch_size")
    shape_raw = cast(object, raw.get("shape"))
    if not isinstance(shape_raw, list) or not shape_raw:
        raise ValueError(f"capacity profile '{name}' shape must be a non-empty list")
    shape = tuple(
        _positive_int(value, f"{name}.shape[{index}]")
        for index, value in enumerate(shape_raw)
    )
    dtype = raw.get("dtype", "float32")
    if not isinstance(dtype, str) or dtype not in _DTYPES:
        raise ValueError(
            f"capacity profile '{name}' dtype must be one of "
            + ", ".join(sorted(_DTYPES))
        )
    observations = _non_negative_int(
        raw.get("trajectory_observations", 0),
        f"{name}.trajectory_observations",
    )
    if observations == 1:
        raise ValueError(
            f"capacity profile '{name}' trajectory observations must be 0 or at "
            "least 2"
        )
    pattern = raw.get("pattern", "uniform")
    if not isinstance(pattern, str) or pattern not in {"uniform", "random"}:
        raise ValueError(
            f"capacity profile '{name}' pattern must be uniform or random"
        )
    seed = _non_negative_int(raw.get("seed", 2026), f"{name}.seed")
    writers = _parse_writers(name, raw.get("writers", [{"name": "tensor"}]))
    execute = raw.get("execute", True)
    if not isinstance(execute, bool):
        raise TypeError(f"capacity profile '{name}' execute must be boolean")
    warmup_runs = _non_negative_int(
        raw.get("warmup_runs", 1), f"{name}.warmup_runs"
    )
    measured_runs = _positive_int(
        raw.get("measured_runs", 5), f"{name}.measured_runs"
    )
    timeout_seconds = _positive_int(
        raw.get("timeout_seconds", 1800), f"{name}.timeout_seconds"
    )
    if batch_size > num_samples:
        raise ValueError(f"capacity profile '{name}' batch_size exceeds num_samples")
    return CapacityProfile(
        name=name,
        description=description,
        num_samples=num_samples,
        batch_size=batch_size,
        shape=shape,
        dtype=dtype,
        trajectory_observations=observations,
        pattern=cast(str, pattern),
        seed=seed,
        writers=writers,
        execute=execute,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        timeout_seconds=timeout_seconds,
    )


def _parse_writers(name: str, raw: object) -> tuple[ComponentConfig, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"capacity profile '{name}' writers must be a non-empty list")
    writers: list[ComponentConfig] = []
    for index, declaration in enumerate(raw):
        if not isinstance(declaration, dict):
            raise TypeError(f"{name}.writers[{index}] must be a mapping")
        unknown = sorted(set(declaration) - {"name", "params"})
        if unknown:
            raise ValueError(
                f"unknown {name}.writers[{index}] field(s): " + ", ".join(unknown)
            )
        writer_name = declaration.get("name")
        params = declaration.get("params", {})
        if not isinstance(writer_name, str) or not writer_name.strip():
            raise ValueError(f"{name}.writers[{index}].name must be non-empty")
        if not isinstance(params, dict):
            raise TypeError(f"{name}.writers[{index}].params must be a mapping")
        writers.append(ComponentConfig(name=writer_name, params=dict(params)))
    return tuple(writers)


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _non_negative_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _peak_rss_bytes() -> int:
    if _resource is not None:
        peak = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    return _windows_peak_rss_bytes()


def _windows_peak_rss_bytes() -> int:
    """Return PeakWorkingSetSize without adding a benchmark dependency."""

    windows_ctypes = cast(Any, ctypes)
    kernel32 = windows_ctypes.WinDLL("kernel32", use_last_error=True)
    get_current_process = kernel32.GetCurrentProcess
    get_current_process.argtypes = ()
    get_current_process.restype = wintypes.HANDLE
    get_process_memory_info = kernel32.K32GetProcessMemoryInfo
    get_process_memory_info.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(WindowsProcessMemoryCounters),
        wintypes.DWORD,
    )
    get_process_memory_info.restype = wintypes.BOOL

    counters = WindowsProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    process = get_current_process()
    succeeded = get_process_memory_info(
        process,
        ctypes.byref(counters),
        counters.cb,
    )
    if not succeeded:
        error_code = windows_ctypes.get_last_error()
        raise OSError(error_code, "K32GetProcessMemoryInfo failed")
    return int(counters.peak_working_set_size)


def _host_memory_bytes() -> int | None:
    sysconf = getattr(os, "sysconf", None)
    if not callable(sysconf):
        return None
    typed_sysconf = cast(SysconfProvider, sysconf)
    try:
        physical = int(typed_sysconf("SC_PAGE_SIZE")) * int(
            typed_sysconf("SC_PHYS_PAGES")
        )
    except (OSError, TypeError, ValueError):
        return None
    limits = [physical]
    for candidate in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            raw = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw.isdecimal():
            limit = int(raw)
            if 0 < limit < 1 << 60:
                limits.append(limit)
    return min(limits)


def _available_host_memory_bytes() -> int | None:
    """Return MemAvailable where the host exposes it without a dependency."""

    candidates: list[int] = []
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdecimal():
                candidates.append(int(fields[1]) * 1024)
            break
    cgroup_available = _cgroup_available_memory_bytes(Path("/sys/fs/cgroup"))
    if cgroup_available is not None:
        candidates.append(cgroup_available)
    return min(candidates) if candidates else None


def _cgroup_available_memory_bytes(root: Path) -> int | None:
    """Return remaining cgroup v2/v1 memory when both values are readable."""

    pairs = (
        (root / "memory.max", root / "memory.current"),
        (
            root / "memory" / "memory.limit_in_bytes",
            root / "memory" / "memory.usage_in_bytes",
        ),
    )
    for limit_path, usage_path in pairs:
        limit = _read_bounded_decimal(limit_path)
        usage = _read_bounded_decimal(usage_path, allow_large=True)
        if limit is not None and usage is not None:
            return max(0, limit - usage)
    return None


def _read_bounded_decimal(path: Path, *, allow_large: bool = False) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw.isdecimal():
        return None
    value = int(raw)
    if value < 0 or (not allow_large and value >= 1 << 60):
        return None
    return value


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _materialize_state(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    fill_value: float,
    pattern: str,
    generator: torch.Generator,
) -> torch.Tensor:
    state = torch.empty(shape, dtype=dtype, device=device)
    if pattern == "random":
        state.uniform_(-1.0, 1.0, generator=generator)
    else:
        state.fill_(fill_value)
    _synchronize(device)
    return state.detach().to(device="cpu", copy=True)


def _materialize_output(
    profile: CapacityProfile,
    *,
    device: torch.device,
) -> SamplingOutput:
    dtype = _DTYPES[profile.dtype]
    generator = torch.Generator(device=device)
    generator.manual_seed(profile.seed)
    batches: list[SamplingBatch] = []
    remaining = profile.num_samples
    while remaining:
        count = min(profile.batch_size, remaining)
        batch_shape = (count, *profile.shape)
        samples = _materialize_state(
            batch_shape,
            dtype=dtype,
            device=device,
            fill_value=0.25,
            pattern=profile.pattern,
            generator=generator,
        )
        observations: tuple[SamplingObservation, ...] | None = None
        if profile.trajectory_observations:
            retained: list[SamplingObservation] = []
            last = profile.trajectory_observations - 1
            for step_index in range(profile.trajectory_observations):
                state = _materialize_state(
                    batch_shape,
                    dtype=dtype,
                    device=device,
                    fill_value=step_index / max(last, 1),
                    pattern=profile.pattern,
                    generator=generator,
                )
                retained.append(
                    SamplingObservation(
                        step_index=step_index,
                        coordinate=last - step_index,
                        state=state,
                        is_final=step_index == last,
                        diagnostics={},
                    )
                )
            observations = tuple(retained)
        batches.append(
            SamplingBatch(
                samples=samples,
                num_samples=count,
                trajectory=observations,
            )
        )
        remaining -= count
    return validate_sampling_output(
        SamplingOutput(
            batches=tuple(batches),
            metadata={"capacity_profile": profile.name},
        ),
        expected_num_samples=profile.num_samples,
    )


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _cuda_peaks(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    return {
        "allocated_bytes": torch.cuda.max_memory_allocated(device),
        "reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _run_worker(
    profile: CapacityProfile,
    *,
    device_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA capacity benchmark requested but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS capacity benchmark requested but MPS is unavailable")
    output_dir.mkdir(parents=True, exist_ok=False)
    baseline_rss = _peak_rss_bytes()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    output = _materialize_output(profile, device=device)
    after_output_rss = _peak_rss_bytes()
    after_output_cuda = _cuda_peaks(device)
    writer_results: list[dict[str, Any]] = []
    for index, writer in enumerate(profile.writers):
        writer_dir = output_dir / f"{index:02d}_{writer.name}"
        writer_started = time.perf_counter()
        artifacts = write_sampling_artifacts(
            [writer],
            SamplingArtifactContext(
                output_dir=writer_dir,
                batches=output.batches,
                metadata=output.metadata,
            ),
        )
        writer_results.append(
            {
                "name": writer.name,
                "params": dict(writer.params),
                "wall_seconds": time.perf_counter() - writer_started,
                "artifact_bytes": sum(path.stat().st_size for path in artifacts.values()),
                "artifact_keys": sorted(artifacts),
                "cuda_peak": _cuda_peaks(device),
                "cumulative_lifetime_peak_rss_bytes_after_writer": (
                    _peak_rss_bytes()
                ),
            }
        )
    _synchronize(device)
    return {
        "baseline_lifetime_peak_rss_bytes": baseline_rss,
        "after_output_lifetime_peak_rss_bytes": after_output_rss,
        "after_output_cuda_peak": after_output_cuda,
        "final_lifetime_peak_rss_bytes": _peak_rss_bytes(),
        "final_cuda_peak": _cuda_peaks(device),
        "artifact_bytes": _directory_bytes(output_dir),
        "wall_seconds": time.perf_counter() - started,
        "writers": writer_results,
    }


def _worker_command(
    *,
    profile_path: Path,
    profile_name: str,
    device_name: str,
    output_dir: Path,
    result_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--profiles",
        str(profile_path),
        "--profile",
        profile_name,
        "--device",
        device_name,
        "--worker-output-dir",
        str(output_dir),
        "--worker-result",
        str(result_path),
    ]


def _execute_repeat(
    *,
    profile_path: Path,
    profile_name: str,
    device_name: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"stochaflow-{profile_name}-") as raw:
        root = Path(raw)
        output_dir = root / "artifacts"
        result_path = root / "result.json"
        subprocess.run(
            _worker_command(
                profile_path=profile_path,
                profile_name=profile_name,
                device_name=device_name,
                output_dir=output_dir,
                result_path=result_path,
            ),
            check=True,
            timeout=timeout_seconds,
        )
        return cast(dict[str, Any], json.loads(result_path.read_text()))


def _summarize_repeats(
    repeats: list[dict[str, Any]],
    *,
    host_memory_bytes: int | None,
) -> dict[str, Any]:
    peaks = [
        int(repeat["final_lifetime_peak_rss_bytes"]) for repeat in repeats
    ]
    median = statistics.median(peaks)
    maximum = max(peaks)
    mean = statistics.fmean(peaks)
    coefficient_of_variation = (
        statistics.pstdev(peaks) / mean if len(peaks) > 1 and mean else 0.0
    )
    host_ratio = maximum / host_memory_bytes if host_memory_bytes else None
    cuda_reserved = [
        int(cuda_peak["reserved_bytes"])
        for repeat in repeats
        if isinstance((cuda_peak := repeat.get("final_cuda_peak")), dict)
    ]
    return {
        "median_peak_rss_bytes": median,
        "max_peak_rss_bytes": maximum,
        "peak_rss_coefficient_of_variation": coefficient_of_variation,
        "max_peak_rss_host_fraction": host_ratio,
        "passes_host_budget": (
            host_ratio <= DEFAULT_HOST_BUDGET_FRACTION
            if host_ratio is not None
            else None
        ),
        "max_cuda_reserved_bytes": max(cuda_reserved) if cuda_reserved else None,
    }


def run_benchmarks(
    *,
    profile_path: Path,
    selected_names: list[str],
    device_name: str,
    warmup_override: int | None,
    repeats_override: int | None,
    allow_over_budget: bool,
) -> dict[str, Any]:
    """Run selected profiles and return a machine-readable report."""

    profiles = load_profiles(profile_path)
    missing = sorted(set(selected_names) - set(profiles))
    if missing:
        raise ValueError("unknown capacity profile(s): " + ", ".join(missing))
    host_memory = _host_memory_bytes()
    available_memory = _available_host_memory_bytes()
    disk_free = shutil.disk_usage(tempfile.gettempdir()).free
    results: list[dict[str, Any]] = []
    for name in selected_names:
        profile = profiles[name]
        profile_result: dict[str, Any] = {
            "name": profile.name,
            "description": profile.description,
            "configuration": {
                **asdict(profile),
                "writers": [asdict(writer) for writer in profile.writers],
            },
            "projection": profile.projection(),
            "executed": profile.execute,
        }
        if profile.execute:
            _preflight_profile(
                profile,
                host_memory_bytes=host_memory,
                available_memory_bytes=available_memory,
                disk_free_bytes=disk_free,
                allow_over_budget=allow_over_budget,
            )
            warmups = (
                profile.warmup_runs
                if warmup_override is None
                else warmup_override
            )
            repeats_count = (
                profile.measured_runs
                if repeats_override is None
                else repeats_override
            )
            for _ in range(warmups):
                _execute_repeat(
                    profile_path=profile_path,
                    profile_name=name,
                    device_name=device_name,
                    timeout_seconds=profile.timeout_seconds,
                )
            repeats = [
                _execute_repeat(
                    profile_path=profile_path,
                    profile_name=name,
                    device_name=device_name,
                    timeout_seconds=profile.timeout_seconds,
                )
                for _ in range(repeats_count)
            ]
            profile_result["warmup_runs"] = warmups
            profile_result["measured_runs"] = repeats
            profile_result["summary"] = _summarize_repeats(
                repeats,
                host_memory_bytes=host_memory,
            )
        results.append(profile_result)
    return {
        "format_version": PROFILE_FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": device_name,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "effective_host_memory_bytes": host_memory,
            "available_host_memory_bytes": available_memory,
            "temporary_filesystem_free_bytes": disk_free,
        },
        "source": {
            "tool_sha256": _sha256_file(Path(__file__)),
            "profiles_sha256": _sha256_file(profile_path),
            "selected_profiles": list(selected_names),
            "device": device_name,
            "warmup_override": warmup_override,
            "repeats_override": repeats_override,
            "allow_over_budget": allow_over_budget,
        },
        "budgets": {
            "host_fraction": DEFAULT_HOST_BUDGET_FRACTION,
        },
        "profiles": results,
    }


def _preflight_profile(
    profile: CapacityProfile,
    *,
    host_memory_bytes: int | None,
    available_memory_bytes: int | None,
    disk_free_bytes: int,
    allow_over_budget: bool,
) -> None:
    projection = profile.projection()
    required_memory = max(
        projection["sampling_output_live_bytes"],
        projection["tensor_writer_structural_peak_lower_bound_bytes"],
    )
    memory_references = [
        value
        for value in (available_memory_bytes, host_memory_bytes)
        if value is not None
    ]
    memory_reference = min(memory_references) if memory_references else None
    if any(writer.name == "image" for writer in profile.writers):
        final = projection["final_state_bytes"]
        trajectory = projection["trajectory_state_bytes"]
        # Guard retained, concatenated, grid, normalization, and encoding storage.
        # This is deliberately conservative, not a cross-version upper bound.
        required_memory = max(required_memory, 4 * final + 6 * trajectory)
    failures: list[str] = []
    if (
        memory_reference is not None
        and required_memory > memory_reference * DEFAULT_PREFLIGHT_FRACTION
    ):
        failures.append(
            f"projected memory {required_memory} exceeds 50% of available/effective "
            f"host memory {memory_reference}"
        )
    # Each configured writer publishes into its own artifact set. Custom formats
    # may exceed this input-sized estimate, so this is a guard, not a disk bound.
    artifact_bytes = projection["raw_artifact_payload_bytes"] * len(profile.writers)
    if artifact_bytes > disk_free_bytes * DEFAULT_PREFLIGHT_FRACTION:
        failures.append(
            f"raw artifact payload {artifact_bytes} exceeds 50% of temporary "
            f"filesystem free space {disk_free_bytes}"
        )
    if failures and not allow_over_budget:
        raise ValueError(
            "capacity preflight rejected profile "
            f"'{profile.name}': "
            + "; ".join(failures)
            + "; pass --allow-over-budget only after reviewing the projection"
        )


def _sha256_file(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--profile", action="append", default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup-runs", type=int)
    parser.add_argument("--measured-runs", type=int)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--allow-over-budget", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI."""

    args = build_parser().parse_args(argv)
    profiles = load_profiles(args.profiles)
    if args.list:
        for name, profile in profiles.items():
            mode = "execute" if profile.execute else "projection"
            print(f"{name}\t{mode}\t{profile.description}")
        return 0
    if not args.profile:
        raise ValueError("at least one --profile is required")
    if args.warmup_runs is not None:
        _non_negative_int(args.warmup_runs, "--warmup-runs")
    if args.measured_runs is not None:
        _positive_int(args.measured_runs, "--measured-runs")
    if args.worker:
        if len(args.profile) != 1:
            raise ValueError("worker mode requires exactly one --profile")
        if args.worker_output_dir is None or args.worker_result is None:
            raise ValueError("worker mode requires output and result paths")
        profile = profiles[args.profile[0]]
        result = _run_worker(
            profile,
            device_name=args.device,
            output_dir=args.worker_output_dir,
        )
        args.worker_result.write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return 0
    report = run_benchmarks(
        profile_path=args.profiles.resolve(),
        selected_names=args.profile,
        device_name=args.device,
        warmup_override=args.warmup_runs,
        repeats_override=args.measured_runs,
        allow_over_budget=args.allow_over_budget,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.result is None:
        print(rendered)
    else:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

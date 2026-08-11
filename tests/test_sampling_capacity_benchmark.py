"""Contract tests for the repository sampling-capacity benchmark."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import runpy
import statistics
import subprocess
import sys
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "benchmark_sampling_capacity.py"
PROFILES = ROOT / "benchmarks" / "sampling_capacity_profiles.yaml"
_TOOL_API = runpy.run_path(str(TOOL))
_LOAD_PROFILES = cast(
    Callable[[Path], dict[str, Any]],
    _TOOL_API["load_profiles"],
)
_PREFLIGHT_PROFILE = cast(Callable[..., None], _TOOL_API["_preflight_profile"])
_CGROUP_AVAILABLE = cast(
    Callable[[Path], int | None],
    _TOOL_API["_cgroup_available_memory_bytes"],
)
_AVAILABLE_HOST_MEMORY = cast(
    Callable[[], int | None],
    _TOOL_API["_available_host_memory_bytes"],
)
_HOST_MEMORY = cast(Callable[[], int | None], _TOOL_API["_host_memory_bytes"])
_PEAK_RSS = cast(Callable[[], int], _TOOL_API["_peak_rss_bytes"])
_SUMMARIZE_REPEATS = cast(
    Callable[..., dict[str, Any]],
    _TOOL_API["_summarize_repeats"],
)
_SHA256_FILE = cast(Callable[[Path], str], _TOOL_API["_sha256_file"])
_WINDOWS_PEAK_RSS = cast(Callable[[], int], _TOOL_API["_windows_peak_rss_bytes"])
_WINDOWS_COUNTERS = cast(
    type[ctypes.Structure],
    _TOOL_API["WindowsProcessMemoryCounters"],
)


class FakeWin32Function:
    def __init__(self, implementation: Callable[..., object]) -> None:
        self._implementation = implementation
        self.argtypes: tuple[object, ...] | None = None
        self.restype: object = None

    def __call__(self, *args: object) -> object:
        return self._implementation(*args)


def _canonical_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _run_tool(*arguments: str, env: dict[str, str] | None = None):
    command_env = os.environ.copy()
    python_path = str(ROOT / "src")
    if existing := command_env.get("PYTHONPATH"):
        python_path = python_path + os.pathsep + existing
    command_env["PYTHONPATH"] = python_path
    if env:
        command_env.update(env)
    return subprocess.run(
        [sys.executable, str(TOOL), *arguments],
        cwd=ROOT,
        env=command_env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_tool_sha256_normalizes_checkout_line_endings(tmp_path: Path) -> None:
    unix_source = tmp_path / "unix.py"
    windows_source = tmp_path / "windows.py"
    canonical_payload = b"first line\nsecond line\n"
    unix_source.write_bytes(canonical_payload)
    windows_source.write_bytes(canonical_payload.replace(b"\n", b"\r\n"))

    expected = hashlib.sha256(canonical_payload).hexdigest()
    assert _SHA256_FILE(unix_source) == expected
    assert _SHA256_FILE(windows_source) == expected


def test_capacity_projection_reports_exact_generic_payloads(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "projection.json"

    completed = _run_tool(
        "--profiles",
        str(PROFILES),
        "--profile",
        "high_resolution_1024_projection",
        "--profile",
        "field3d_dense_projection",
        "--result",
        str(result_path),
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(result_path.read_text(encoding="utf-8"))
    assert report["source"]["selected_profiles"] == [
        "high_resolution_1024_projection",
        "field3d_dense_projection",
    ]
    assert report["source"]["tool_sha256"] == _canonical_sha256(TOOL)
    assert report["source"]["profiles_sha256"] == _canonical_sha256(PROFILES)
    high_resolution, field = report["profiles"]
    assert high_resolution["projection"] == {
        "one_sample_bytes": 12582912,
        "final_state_bytes": 201326592,
        "trajectory_state_bytes": 10267656192,
        "sampling_output_live_bytes": 10468982784,
        "tensor_writer_structural_peak_lower_bound_bytes": 31004295168,
        "raw_artifact_payload_bytes": 10468982784,
    }
    assert field["projection"]["final_state_bytes"] == 268435456
    assert field["projection"]["trajectory_state_bytes"] == 13690208256
    assert field["projection"]["sampling_output_live_bytes"] == 13958643712
    assert not high_resolution["executed"]
    assert not field["executed"]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"num_samples": True}, "num_samples must be a positive integer"),
        ({"batch_size": 0}, "batch_size must be a positive integer"),
        ({"batch_size": 3}, "batch_size exceeds num_samples"),
        ({"shape": [1, 0]}, r"shape\[1\] must be a positive integer"),
        ({"dtype": "int64"}, "dtype must be one of"),
        ({"trajectory_observations": 1}, "must be 0 or at least 2"),
        ({"pattern": []}, "pattern must be uniform or random"),
        ({"writers": []}, "writers must be a non-empty list"),
        ({"writers": [{"name": "tensor", "params": []}]}, "params must be"),
        ({"unknown": 1}, "unknown capacity profile 'bad' field"),
    ],
)
def test_capacity_profile_parser_rejects_invalid_schema(
    tmp_path: Path,
    override: dict[str, Any],
    message: str,
) -> None:
    invalid = tmp_path / "invalid.yaml"
    declaration: dict[str, Any] = {
        "num_samples": 2,
        "batch_size": 1,
        "shape": [1],
        "trajectory_observations": 0,
        "writers": [{"name": "tensor"}],
    }
    declaration.update(override)
    invalid.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "profiles": {"bad": declaration},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises((TypeError, ValueError), match=message):
        _LOAD_PROFILES(invalid)


def test_capacity_profile_parser_rejects_boolean_version(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid-version.yaml"
    invalid.write_text(
        yaml.safe_dump({"version": True, "profiles": {"unused": {}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="version must be 1"):
        _LOAD_PROFILES(invalid)


def test_capacity_smoke_uses_fresh_worker_and_cleans_artifacts(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "smoke.json"
    temporary_root = tmp_path / "worker-tmp"
    temporary_root.mkdir()

    completed = _run_tool(
        "--profiles",
        str(PROFILES),
        "--profile",
        "ci_smoke",
        "--result",
        str(result_path),
        env={"TMPDIR": str(temporary_root)},
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(result_path.read_text(encoding="utf-8"))
    profile = report["profiles"][0]
    repeat = profile["measured_runs"][0]
    assert profile["executed"]
    assert profile["summary"]["max_peak_rss_bytes"] > 0
    assert repeat["after_output_lifetime_peak_rss_bytes"] >= repeat[
        "baseline_lifetime_peak_rss_bytes"
    ]
    assert repeat["final_lifetime_peak_rss_bytes"] >= repeat[
        "after_output_lifetime_peak_rss_bytes"
    ]
    assert repeat["writers"][0]["artifact_keys"] == ["samples", "trajectory"]
    assert repeat["artifact_bytes"] > profile["projection"]["raw_artifact_payload_bytes"]
    assert list(temporary_root.iterdir()) == []


def test_capacity_preflight_rejects_unsafe_execution_without_override() -> None:
    profile = _LOAD_PROFILES(PROFILES)["high_resolution_1024_projection"]

    with pytest.raises(ValueError, match="--allow-over-budget"):
        _PREFLIGHT_PROFILE(
            profile,
            host_memory_bytes=2**30,
            available_memory_bytes=None,
            disk_free_bytes=2**40,
            allow_over_budget=False,
        )

    _PREFLIGHT_PROFILE(
        profile,
        host_memory_bytes=2**30,
        available_memory_bytes=None,
        disk_free_bytes=2**40,
        allow_over_budget=True,
    )

    field_profile = _LOAD_PROFILES(PROFILES)["field3d_preview"]
    image_profile = replace(
        field_profile,
        name="field3d_image_preview",
        writers=(replace(field_profile.writers[0], name="image"),),
    )
    with pytest.raises(ValueError, match="projected memory"):
        _PREFLIGHT_PROFILE(
            image_profile,
            host_memory_bytes=400 * 2**20,
            available_memory_bytes=None,
            disk_free_bytes=2**40,
            allow_over_budget=False,
        )

    final_profile = _LOAD_PROFILES(PROFILES)["field3d_dense_projection"]
    two_writers = replace(
        final_profile,
        writers=final_profile.writers + final_profile.writers,
    )
    with pytest.raises(ValueError, match="filesystem free space"):
        _PREFLIGHT_PROFILE(
            two_writers,
            host_memory_bytes=2**40,
            available_memory_bytes=None,
            disk_free_bytes=3 * 2**30,
            allow_over_budget=False,
        )


@pytest.mark.parametrize("version", [1, 2])
def test_cgroup_available_memory_uses_limit_minus_current(
    tmp_path: Path,
    version: int,
) -> None:
    if version == 2:
        (tmp_path / "memory.max").write_text(str(4 * 2**30), encoding="utf-8")
        (tmp_path / "memory.current").write_text(
            str(3 * 2**30), encoding="utf-8"
        )
    else:
        memory = tmp_path / "memory"
        memory.mkdir()
        (memory / "memory.limit_in_bytes").write_text(
            str(4 * 2**30), encoding="utf-8"
        )
        (memory / "memory.usage_in_bytes").write_text(
            str(3 * 2**30), encoding="utf-8"
        )

    assert _CGROUP_AVAILABLE(tmp_path) == 2**30


@pytest.mark.parametrize(
    ("platform_name", "expected"),
    [
        ("darwin", 123),
        ("linux", 123 * 1024),
    ],
)
def test_peak_rss_normalizes_posix_units(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
    expected: int,
) -> None:
    fake_resource = SimpleNamespace(
        RUSAGE_SELF=7,
        getrusage=lambda _who: SimpleNamespace(ru_maxrss=123),
    )
    monkeypatch.setitem(_PEAK_RSS.__globals__, "_resource", fake_resource)
    monkeypatch.setattr(sys, "platform", platform_name)

    assert _PEAK_RSS() == expected


def test_peak_rss_uses_windows_provider_without_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(_PEAK_RSS.__globals__, "_resource", None)
    monkeypatch.setitem(
        _PEAK_RSS.__globals__,
        "_windows_peak_rss_bytes",
        lambda: 123_456,
    )

    assert _PEAK_RSS() == 123_456


def test_host_memory_is_unavailable_without_sysconf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(os, "sysconf", raising=False)

    assert _HOST_MEMORY() is None


def test_available_host_memory_is_unavailable_without_procfs_or_cgroup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_path: Path, *args: object, **kwargs: object) -> str:
        del args, kwargs
        raise OSError("unavailable")

    monkeypatch.setattr(Path, "read_text", unavailable)

    assert _AVAILABLE_HOST_MEMORY() is None


def test_summary_marks_host_budget_unknown_without_host_memory() -> None:
    summary = _SUMMARIZE_REPEATS(
        [{"final_lifetime_peak_rss_bytes": 123, "final_cuda_peak": None}],
        host_memory_bytes=None,
    )

    assert summary["max_peak_rss_host_fraction"] is None
    assert summary["passes_host_budget"] is None


def test_summary_aggregates_repeat_statistics_and_device_peaks() -> None:
    peak_rss_values = [100, 200, 400]
    summary = _SUMMARIZE_REPEATS(
        [
            {
                "final_lifetime_peak_rss_bytes": peak_rss,
                "final_cuda_peak": {"reserved_bytes": cuda_reserved},
            }
            for peak_rss, cuda_reserved in zip(
                peak_rss_values,
                (11, 17, 13),
                strict=True,
            )
        ],
        host_memory_bytes=500,
    )

    assert summary["median_peak_rss_bytes"] == 200
    assert summary["max_peak_rss_bytes"] == 400
    assert summary["peak_rss_coefficient_of_variation"] == pytest.approx(
        statistics.pstdev(peak_rss_values) / statistics.fmean(peak_rss_values)
    )
    assert summary["max_peak_rss_host_fraction"] == pytest.approx(0.8)
    assert summary["passes_host_budget"] is False
    assert summary["max_cuda_reserved_bytes"] == 17


def test_windows_peak_rss_declares_pointer_sized_win32_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process_handle = object()
    get_current_process = FakeWin32Function(lambda: process_handle)

    def get_process_memory_info(
        received_process: object,
        counters_pointer: object,
        size: object,
    ) -> int:
        assert received_process is process_handle
        assert size == ctypes.sizeof(_WINDOWS_COUNTERS)
        counters = cast(Any, counters_pointer)._obj
        counters.peak_working_set_size = 123_456
        return 1

    get_memory_info = FakeWin32Function(get_process_memory_info)
    kernel32 = SimpleNamespace(
        GetCurrentProcess=get_current_process,
        K32GetProcessMemoryInfo=get_memory_info,
    )

    def load_library(name: str, *, use_last_error: bool) -> object:
        assert name == "kernel32"
        assert use_last_error
        return kernel32

    monkeypatch.setattr(ctypes, "WinDLL", load_library, raising=False)

    assert _WINDOWS_PEAK_RSS() == 123_456
    assert get_current_process.argtypes == ()
    assert get_current_process.restype is wintypes.HANDLE
    assert get_memory_info.argtypes == (
        wintypes.HANDLE,
        ctypes.POINTER(_WINDOWS_COUNTERS),
        wintypes.DWORD,
    )
    assert get_memory_info.restype is wintypes.BOOL


def test_windows_peak_rss_reports_win32_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel32 = SimpleNamespace(
        GetCurrentProcess=FakeWin32Function(object),
        K32GetProcessMemoryInfo=FakeWin32Function(lambda *_args: 0),
    )
    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda _name, *, use_last_error: kernel32,
        raising=False,
    )
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)

    with pytest.raises(OSError, match="K32GetProcessMemoryInfo failed") as error:
        _WINDOWS_PEAK_RSS()

    assert error.value.errno == 5

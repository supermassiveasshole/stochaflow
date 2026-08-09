"""Download the official AFHQ-v2 archive into private cache staging."""

from __future__ import annotations

import http.client
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from stochaflow.data.artifact_io import (
    cache_entry_exists,
    create_cache_file_exclusive,
    ensure_cache_directory,
    open_cache_file,
    publish_cache_file,
    quarantine_cache_entry,
    remove_cache_file,
)

from .contracts import PreparationError, SourceIntegrityError, SourceLock
from .safe_file import write_descriptor

_DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
_DOWNLOAD_PROGRESS_BYTES = 128 * 1024 * 1024
_CONTENT_RANGE_PATTERN = re.compile(r"^bytes ([0-9]+)-([0-9]+)/([0-9]+)$")

def _build_opener(proxy: str | None) -> urllib.request.OpenerDirector:
    handlers: list[urllib.request.BaseHandler] = []
    if proxy is not None:
        normalized = proxy.strip()
        if not normalized:
            raise PreparationError("proxy must not be empty")
        if "://" not in normalized:
            normalized = f"http://{normalized}"
        handlers.append(
            urllib.request.ProxyHandler(
                {
                    "http": normalized,
                    "https": normalized,
                }
            )
        )
    return urllib.request.build_opener(*handlers)


def _parse_content_range(value: str | None, *, expected_start: int) -> int:
    if value is None:
        raise SourceIntegrityError("resumed response has no Content-Range header")
    match = _CONTENT_RANGE_PATTERN.fullmatch(value)
    if match is None:
        raise SourceIntegrityError(f"invalid Content-Range header: {value!r}")
    start, end, total = (int(group) for group in match.groups())
    if start != expected_start or end < start or total <= end:
        raise SourceIntegrityError(
            f"unexpected Content-Range for offset {expected_start}: {value!r}"
        )
    return total


def _download_once(
    *,
    opener: urllib.request.OpenerDirector,
    url: str,
    cache_root: Path,
    partial_path: Path,
    expected_bytes: int,
) -> None:
    existing_bytes = _cache_file_size(
        cache_root,
        partial_path,
        label="AFHQ-v2 partial download",
    )
    if existing_bytes is None:
        existing_bytes = 0
    if existing_bytes > expected_bytes:
        raise SourceIntegrityError(
            f"partial download is larger than expected: {partial_path}"
        )
    if existing_bytes == expected_bytes:
        return

    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "Stochaflow-AFHQ-v2-preparer/1",
    }
    if existing_bytes:
        headers["Range"] = f"bytes={existing_bytes}-"
    request = urllib.request.Request(url, headers=headers)
    with opener.open(request, timeout=90) as response:
        status = response.getcode()
        append = existing_bytes > 0 and status == 206
        if existing_bytes > 0 and status == 200:
            existing_bytes = 0
        elif existing_bytes > 0 and status != 206:
            raise SourceIntegrityError(
                f"server returned HTTP {status} for a resumed download"
            )
        elif existing_bytes == 0 and status != 200:
            raise SourceIntegrityError(
                f"server returned unexpected HTTP {status} for download"
            )

        if append:
            response_total = _parse_content_range(
                response.headers.get("Content-Range"),
                expected_start=existing_bytes,
            )
            if response_total != expected_bytes:
                raise SourceIntegrityError(
                    "resumed response length does not match the source lock"
                )
        else:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != expected_bytes:
                raise SourceIntegrityError(
                    "response Content-Length does not match the source lock: "
                    f"{content_length} != {expected_bytes}"
                )

        mode = "ab" if append else "wb"
        downloaded = existing_bytes
        next_progress = (
            (downloaded // _DOWNLOAD_PROGRESS_BYTES) + 1
        ) * _DOWNLOAD_PROGRESS_BYTES
        started = time.monotonic()
        try:
            descriptor = open_cache_file(
                cache_root,
                partial_path,
                label="AFHQ-v2 partial download",
            )
        except (OSError, ValueError) as error:
            raise PreparationError(
                f"cannot open AFHQ-v2 partial download: {partial_path}"
            ) from error
        try:
            if append:
                observed_size = os.fstat(descriptor).st_size
                if observed_size != existing_bytes:
                    raise SourceIntegrityError(
                        "partial download changed before resume"
                    )
                os.lseek(descriptor, 0, os.SEEK_END)
            else:
                os.ftruncate(descriptor, 0)
                os.lseek(descriptor, 0, os.SEEK_SET)
            with os.fdopen(descriptor, mode, closefd=False) as destination:
                while chunk := response.read(_DOWNLOAD_CHUNK_BYTES):
                    destination.write(chunk)
                    downloaded += len(chunk)
                    if downloaded > expected_bytes:
                        raise SourceIntegrityError(
                            "download exceeded the byte count in the source lock"
                        )
                    if downloaded >= next_progress:
                        elapsed = max(time.monotonic() - started, 0.001)
                        transferred = downloaded - existing_bytes
                        rate_mib = transferred / elapsed / (1024 * 1024)
                        percent = downloaded * 100 / expected_bytes
                        print(
                            f"Downloaded {downloaded:,}/{expected_bytes:,} bytes "
                            f"({percent:.1f}%, {rate_mib:.1f} MiB/s)",
                            flush=True,
                        )
                        next_progress += _DOWNLOAD_PROGRESS_BYTES
                destination.flush()
                os.fsync(destination.fileno())
        finally:
            os.close(descriptor)


def _download_with_curl(
    *,
    lock: SourceLock,
    cache_root: Path,
    destination: Path,
    proxy: str | None,
) -> Path:
    executable = shutil.which("curl")
    if executable is None:
        raise PreparationError(
            "curl downloader was requested but curl is not on PATH"
        )
    staging = destination.with_name(
        f".{destination.name}.curl-{os.getpid()}-{uuid4().hex}.part"
    )
    command = [
        executable,
        "--fail",
        "--location",
        "--retry",
        "8",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--speed-time",
        "120",
        "--speed-limit",
        "1024",
        "--header",
        "Accept-Encoding: identity",
        "--user-agent",
        "Stochaflow-AFHQ-v2-preparer/1",
        "--silent",
        "--show-error",
    ]
    if proxy is not None:
        normalized_proxy = proxy.strip()
        if not normalized_proxy:
            raise PreparationError("proxy must not be empty")
        if "://" not in normalized_proxy:
            normalized_proxy = f"http://{normalized_proxy}"
        command.extend(("--proxy", normalized_proxy))
    command.append(lock.url)
    print("Downloading with curl into a private cache staging file", flush=True)
    descriptor: int | None = None
    process: subprocess.Popen[bytes] | None = None
    try:
        descriptor = create_cache_file_exclusive(
            cache_root,
            staging,
            label="AFHQ-v2 curl download staging",
        )
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
        )
        if process.stdout is None:
            raise PreparationError("curl did not expose its response stream")
        downloaded = 0
        started = time.monotonic()
        next_progress = _DOWNLOAD_PROGRESS_BYTES
        while chunk := process.stdout.read(_DOWNLOAD_CHUNK_BYTES):
            write_descriptor(descriptor, chunk)
            downloaded += len(chunk)
            if downloaded > lock.expected_bytes:
                process.kill()
                raise SourceIntegrityError(
                    "curl download exceeded the byte count in the source lock"
                )
            if downloaded >= next_progress:
                elapsed = max(time.monotonic() - started, 0.001)
                rate_mib = downloaded / elapsed / (1024 * 1024)
                percent = downloaded * 100 / lock.expected_bytes
                print(
                    f"Downloaded {downloaded:,}/{lock.expected_bytes:,} bytes "
                    f"({percent:.1f}%, {rate_mib:.1f} MiB/s)",
                    flush=True,
                )
                next_progress += _DOWNLOAD_PROGRESS_BYTES
        return_code = process.wait()
        process = None
        if return_code != 0:
            raise PreparationError(
                f"curl download failed with exit code {return_code}"
            )
        if downloaded != lock.expected_bytes:
            raise SourceIntegrityError(
                "curl download byte count does not match the source lock: "
                f"{downloaded} != {lock.expected_bytes}"
            )
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        publish_cache_file(
            cache_root,
            staging,
            destination,
            label="AFHQ-v2 completed curl download",
        )
        return destination
    except OSError as error:
        raise PreparationError(f"could not launch curl: {error}") from error
    finally:
        if process is not None:
            process.kill()
            process.wait()
        if descriptor is not None:
            os.close(descriptor)
        with suppress(FileNotFoundError, OSError, ValueError):
            remove_cache_file(
                cache_root,
                staging,
                label="AFHQ-v2 curl download staging cleanup",
            )


def _cache_file_size(
    cache_root: Path,
    path: Path,
    *,
    label: str,
) -> int | None:
    try:
        exists = cache_entry_exists(cache_root, path, label=label)
    except (OSError, ValueError) as error:
        raise PreparationError(f"cannot inspect {label}: {path}") from error
    if not exists:
        return None
    try:
        descriptor = open_cache_file(cache_root, path, label=label)
    except (OSError, ValueError) as error:
        raise PreparationError(f"cannot open {label}: {path}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PreparationError(f"{label} is not a regular file: {path}")
        return metadata.st_size
    finally:
        os.close(descriptor)


def _quarantine_invalid_download(
    path: Path,
    *,
    cache_root: Path,
    identity: str,
) -> Path:
    safe_identity = re.sub(r"[^0-9A-Za-z._-]", "-", identity)[:96]
    try:
        target = quarantine_cache_entry(
            cache_root,
            path,
            suffix=f"{safe_identity}.invalid",
            label="invalid AFHQ-v2 download",
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        raise PreparationError(
            f"cannot quarantine invalid AFHQ-v2 download: {path}"
        ) from error
    print(
        f"Quarantined invalid download at {target}; restarting from byte 0",
        file=sys.stderr,
        flush=True,
    )
    return target


def download_official_archive(
    *,
    lock: SourceLock,
    cache_root: Path,
    destination: Path,
    proxy: str | None,
    downloader: str = "auto",
    attempts: int = 6,
) -> Path:
    """Download the official archive atomically, resuming a partial transfer."""

    if attempts <= 0:
        raise ValueError("attempts must be positive")
    if downloader not in {"auto", "curl", "python"}:
        raise ValueError("downloader must be 'auto', 'curl', or 'python'")
    try:
        ensure_cache_directory(
            cache_root,
            destination.parent,
            label="AFHQ-v2 download directory",
        )
    except (OSError, ValueError) as error:
        raise PreparationError(
            f"cannot create AFHQ-v2 download directory: {destination.parent}"
        ) from error
    partial_path = destination.with_name(f"{destination.name}.part")
    existing_bytes = _cache_file_size(
        cache_root,
        partial_path,
        label="AFHQ-v2 partial download",
    )
    if existing_bytes is None:
        existing_bytes = 0
    if existing_bytes > lock.expected_bytes:
        _quarantine_invalid_download(
            partial_path,
            cache_root=cache_root,
            identity=f"bytes-{existing_bytes}",
        )
        existing_bytes = 0
    if existing_bytes == lock.expected_bytes:
        try:
            publish_cache_file(
                cache_root,
                partial_path,
                destination,
                label="AFHQ-v2 completed partial download",
            )
        except FileExistsError:
            remove_cache_file(
                cache_root,
                partial_path,
                label="redundant AFHQ-v2 partial download",
            )
        return destination
    required_bytes = max(lock.expected_bytes - existing_bytes, 0)
    free_bytes = shutil.disk_usage(destination.parent).free
    if free_bytes < required_bytes + 1024 * 1024 * 1024:
        raise PreparationError(
            "insufficient free space for the AFHQ-v2 archive plus a 1 GiB "
            f"safety margin: need {required_bytes + 1024 * 1024 * 1024:,}, "
            f"have {free_bytes:,}"
        )
    use_curl = downloader == "curl" or (
        downloader == "auto" and shutil.which("curl") is not None
    )
    if use_curl:
        return _download_with_curl(
            lock=lock,
            cache_root=cache_root,
            destination=destination,
            proxy=proxy,
        )

    opener = _build_opener(proxy)
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            _download_once(
                opener=opener,
                url=lock.url,
                cache_root=cache_root,
                partial_path=partial_path,
                expected_bytes=lock.expected_bytes,
            )
            completed_size = _cache_file_size(
                cache_root,
                partial_path,
                label="AFHQ-v2 partial download",
            )
            if completed_size != lock.expected_bytes:
                raise SourceIntegrityError(
                    "download ended before the source lock byte count"
                )
            try:
                publish_cache_file(
                    cache_root,
                    partial_path,
                    destination,
                    label="AFHQ-v2 completed download",
                )
            except FileExistsError:
                remove_cache_file(
                    cache_root,
                    partial_path,
                    label="redundant AFHQ-v2 partial download",
                )
            return destination
        except (
            OSError,
            SourceIntegrityError,
            http.client.HTTPException,
            urllib.error.URLError,
        ) as error:
            last_error = error
            if attempt + 1 == attempts:
                break
            delay = min(2**attempt, 16)
            print(
                f"Download attempt {attempt + 1}/{attempts} failed: {error}. "
                f"Retrying in {delay}s.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

    raise PreparationError(
        f"failed to download AFHQ-v2 after {attempts} attempts"
    ) from last_error

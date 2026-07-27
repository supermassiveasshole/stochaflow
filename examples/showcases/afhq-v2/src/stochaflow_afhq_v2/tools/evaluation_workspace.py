"""Private staging and atomic publication for AFHQ-v2 evaluation."""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from stochaflow.sampling.runtime import SamplingRunResult
from stochaflow_afhq_v2.tools.evaluation_result import default_output_dir


@dataclass(frozen=True, slots=True)
class EvaluationWorkspace:
    """A private staging tree published only after complete evaluation."""

    final_root: Path
    staging_root: Path

    @classmethod
    def create(
        cls,
        *,
        checkpoint_path: Path,
        output_dir: str | Path | None,
    ) -> EvaluationWorkspace:
        """Create a private sibling directory for one evaluation attempt."""

        final_root = (
            Path(output_dir).resolve()
            if output_dir is not None
            else default_output_dir(checkpoint_path).resolve()
        )
        if final_root.exists():
            raise FileExistsError(
                f"evaluation output directory already exists: {final_root}"
            )
        final_root.parent.mkdir(parents=True, exist_ok=True)
        staging_root = Path(
            tempfile.mkdtemp(
                prefix=f".{final_root.name}.staging-",
                dir=final_root.parent,
            )
        )
        return cls(final_root=final_root, staging_root=staging_root)

    def published_sampling_result(
        self,
        sampling: SamplingRunResult,
    ) -> SamplingRunResult:
        """Rewrite private sampling paths to their final published locations."""

        def published(path: Path) -> Path:
            try:
                relative = path.resolve().relative_to(self.staging_root.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"sampling artifact escaped private staging root: {path}"
                ) from exc
            return self.final_root / relative

        return replace(
            sampling,
            output_dir=published(sampling.output_dir),
            artifacts={
                name: published(path)
                for name, path in sampling.artifacts.items()
            },
        )

    def publish(self) -> None:
        """Atomically publish staging without replacing another result."""

        if self.final_root.exists():
            raise FileExistsError(
                "evaluation output directory appeared before publish: "
                f"{self.final_root}"
            )
        atomic_publish_directory(self.staging_root, self.final_root)

    def cleanup(self) -> None:
        """Remove an unpublished private tree after success or failure."""

        if self.staging_root.exists():
            shutil.rmtree(self.staging_root)


def atomic_publish_directory(source: Path, destination: Path) -> None:
    """Rename a private directory without replacing a concurrent destination."""

    if os.name == "nt":
        source.rename(destination)
        return
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux"):
        try:
            rename_no_replace = library.renameat2
        except AttributeError as exc:
            raise RuntimeError(
                "atomic no-replace publication requires renameat2 on Linux"
            ) from exc
        status = rename_no_replace(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            1,
        )
    elif sys.platform == "darwin":
        try:
            rename_no_replace = library.renamex_np
        except AttributeError as exc:
            raise RuntimeError(
                "atomic no-replace publication requires renamex_np on macOS"
            ) from exc
        status = rename_no_replace(source_bytes, destination_bytes, 4)
    else:
        raise RuntimeError(
            f"atomic no-replace publication is unsupported on {sys.platform}"
        )
    if status == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination,
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        destination,
    )


__all__ = ["EvaluationWorkspace", "atomic_publish_directory"]

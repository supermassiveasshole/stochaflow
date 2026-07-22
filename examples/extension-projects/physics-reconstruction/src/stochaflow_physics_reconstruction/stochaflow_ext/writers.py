"""Streaming NumPy and JSON artifacts for reconstruction output."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import torch

from stochaflow.extensions import (
    REGISTRIES,
    SamplingArtifactContext,
    SamplingArtifactWriter,
)


@REGISTRIES.sampling_artifact_writers.register(
    "physics-reconstruction.reconstruction-artifacts"
)
class ReconstructionArtifactWriter(SamplingArtifactWriter):
    """Write batches directly into one `.npy` memmap plus scalar metrics."""

    def __init__(
        self,
        *,
        filename: str = "reconstructions.npy",
        metrics_filename: str = "metrics.json",
    ) -> None:
        for name, value in {
            "filename": filename,
            "metrics_filename": metrics_filename,
        }.items():
            if not value or Path(value).name != value:
                raise ValueError(f"{name} must be a plain non-empty filename")
        if not filename.endswith(".npy"):
            raise ValueError("reconstruction filename must end with .npy")
        if not metrics_filename.endswith(".json"):
            raise ValueError("metrics filename must end with .json")
        if filename == metrics_filename:
            raise ValueError("reconstruction and metrics filenames must differ")
        self.filename = filename
        self.metrics_filename = metrics_filename

    def write(self, context: SamplingArtifactContext) -> Mapping[str, Path]:
        first_value = context.batches[0].samples
        if not isinstance(first_value, torch.Tensor) or first_value.ndim < 1:
            raise TypeError("reconstruction batches must contain batched Tensors")
        sample_shape = tuple(first_value.shape[1:])
        total = 0
        for index, batch in enumerate(context.batches):
            value = batch.samples
            if not isinstance(value, torch.Tensor) or value.ndim < 1:
                raise TypeError(
                    f"reconstruction batch {index} must contain a batched Tensor"
                )
            if tuple(value.shape[1:]) != sample_shape:
                raise ValueError("reconstruction batches must share sample shape")
            if not torch.is_floating_point(value):
                raise TypeError("reconstruction samples must be floating-point")
            total += value.shape[0]
        metrics_value: object = context.metadata.get("metrics")
        if not isinstance(metrics_value, Mapping):
            raise TypeError("reconstruction metadata must provide a metrics mapping")
        metrics: dict[str, Any] = dict(metrics_value)
        metrics_payload = json.dumps(metrics, indent=2, sort_keys=True) + "\n"

        output_path = context.output_dir / self.filename
        metrics_path = context.output_dir / self.metrics_filename
        existing = [path for path in (output_path, metrics_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "reconstruction writer refuses to replace existing artifact(s): "
                + ", ".join(str(path) for path in existing)
            )
        token = uuid4().hex
        output_temp = context.output_dir / f".{self.filename}.{token}.tmp"
        metrics_temp = context.output_dir / f".{self.metrics_filename}.{token}.tmp"
        committed: list[Path] = []
        try:
            mapped = np.lib.format.open_memmap(
                output_temp,
                mode="w+",
                dtype=np.float32,
                shape=(total, *sample_shape),
            )
            offset = 0
            for batch in context.batches:
                samples = (
                    cast(torch.Tensor, batch.samples)
                    .detach()
                    .cpu()
                    .to(torch.float32)
                )
                count = samples.shape[0]
                mapped[offset : offset + count] = samples.numpy()
                offset += count
            mapped.flush()
            del mapped
            with metrics_temp.open("w", encoding="utf-8") as handle:
                handle.write(metrics_payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(output_temp, output_path)
            committed.append(output_path)
            os.replace(metrics_temp, metrics_path)
            committed.append(metrics_path)
        except BaseException:
            for path in committed:
                path.unlink(missing_ok=True)
            raise
        finally:
            output_temp.unlink(missing_ok=True)
            metrics_temp.unlink(missing_ok=True)
        return {
            "reconstructions": output_path,
            "reconstruction_metrics": metrics_path,
        }


__all__ = ["ReconstructionArtifactWriter"]

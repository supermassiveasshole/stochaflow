"""Measurement normalization and comparison for capacity trials."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch

from stochaflow_afhq_v2.tools.capacity_config import DEFAULT_PRECISIONS


def cuda_peak_memory(device: torch.device) -> dict[str, int | None]:
    """Read peak CUDA memory or return null fields on other devices."""

    if device.type != "cuda":
        return {
            "peak_allocated_vram_bytes": None,
            "peak_reserved_vram_bytes": None,
        }
    return {
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
    }


def measurement_report(
    metrics: Mapping[str, float],
    *,
    micro_batch: int,
    accumulation: int,
    requested_updates: int,
    memory: Mapping[str, int | None],
) -> dict[str, Any]:
    """Convert trainer phase metrics into one capacity measurement."""

    successful_updates = int(metrics["optimizer_steps"])
    skipped_updates = int(metrics["skipped_optimizer_steps"])
    processed_images = int(metrics["micro_batches"]) * micro_batch
    duration_seconds = metrics["duration_seconds"]
    if successful_updates < requested_updates:
        raise RuntimeError(
            "capacity trial ended before the requested successful optimizer "
            f"updates: {successful_updates} < {requested_updates}"
        )
    return {
        "requested_optimizer_updates": requested_updates,
        "successful_optimizer_updates": successful_updates,
        "skipped_optimizer_updates": skipped_updates,
        "processed_micro_batches": int(metrics["micro_batches"]),
        "processed_images": processed_images,
        "effective_batch_size": micro_batch * accumulation,
        "loss": metrics["loss"],
        "duration_seconds": duration_seconds,
        "images_per_second": (
            processed_images / duration_seconds
            if duration_seconds > 0.0
            else 0.0
        ),
        "optimizer_updates_per_second": metrics[
            "optimizer_steps_per_second"
        ],
        "data_wait_seconds": metrics["data_wait_seconds"],
        "compute_seconds": metrics["compute_seconds"],
        "data_wait_compute_ratio": (
            metrics["data_wait_seconds"] / metrics["compute_seconds"]
            if metrics["compute_seconds"] > 0.0
            else None
        ),
        "forward_seconds": metrics["forward_seconds"],
        "backward_seconds": metrics["backward_seconds"],
        "optimizer_seconds": metrics["optimizer_seconds"],
        "non_finite_loss_count": int(metrics["non_finite_loss_count"]),
        "non_finite_gradient_count": int(
            metrics["non_finite_gradient_count"]
        ),
        **memory,
    }


def precision_comparisons(
    trials: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compare successful FP32 and BF16 trials at equal micro-batch sizes."""

    comparisons: list[dict[str, Any]] = []
    by_batch: dict[int, dict[str, Mapping[str, Any]]] = {}
    for trial in trials:
        if trial.get("status") not in {"ok", "non_finite"}:
            continue
        micro_batch = cast(int, trial["micro_batch_size"])
        precision = cast(str, trial["precision"])
        by_batch.setdefault(micro_batch, {})[precision] = trial
    for micro_batch in sorted(by_batch):
        pair = by_batch[micro_batch]
        if set(DEFAULT_PRECISIONS) - set(pair):
            continue
        fp32 = cast(Mapping[str, Any], pair["fp32"]["measurement"])
        bf16 = cast(
            Mapping[str, Any],
            pair["bf16-mixed"]["measurement"],
        )
        fp32_throughput = cast(float, fp32["images_per_second"])
        bf16_throughput = cast(float, bf16["images_per_second"])
        fp32_allocated = cast(
            int | None,
            fp32["peak_allocated_vram_bytes"],
        )
        bf16_allocated = cast(
            int | None,
            bf16["peak_allocated_vram_bytes"],
        )
        fp32_reserved = cast(
            int | None,
            fp32["peak_reserved_vram_bytes"],
        )
        bf16_reserved = cast(
            int | None,
            bf16["peak_reserved_vram_bytes"],
        )
        comparisons.append(
            {
                "micro_batch_size": micro_batch,
                "bf16_vs_fp32_images_per_second_delta": (
                    bf16_throughput - fp32_throughput
                ),
                "bf16_vs_fp32_throughput_ratio": (
                    bf16_throughput / fp32_throughput
                    if fp32_throughput > 0.0
                    else None
                ),
                "bf16_vs_fp32_peak_allocated_vram_delta_bytes": (
                    bf16_allocated - fp32_allocated
                    if bf16_allocated is not None
                    and fp32_allocated is not None
                    else None
                ),
                "bf16_vs_fp32_peak_reserved_vram_delta_bytes": (
                    bf16_reserved - fp32_reserved
                    if bf16_reserved is not None
                    and fp32_reserved is not None
                    else None
                ),
            }
        )
    return comparisons


__all__ = [
    "cuda_peak_memory",
    "measurement_report",
    "precision_comparisons",
]

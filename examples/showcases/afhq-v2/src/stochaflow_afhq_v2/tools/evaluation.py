"""Orchestrate AFHQ-v2 post-training quality evaluation."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import torch

from stochaflow.data import (
    DataArtifactBindings,
    DataLoaders,
    build_data_loaders,
)
from stochaflow.sampling.runtime import (
    ResolvedSamplingInputs,
    SamplingRunResult,
    resolve_sampling_inputs,
    run_resolved_sampling,
)
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.device import validate_execution_device
from stochaflow.utils.factory import resolve_device
from stochaflow.utils.plugins import (
    ExtensionVersionPolicy,
    ResolvedExtensions,
    activate_extension_plugins,
)
from stochaflow.utils.seed import set_seed
from stochaflow_afhq_v2.tools.evaluation_config import (
    SAMPLING_BUILDER_NAME,
    AFHQV2EvaluationDocument,
    AFHQV2EvaluationProtocol,
    AFHQV2MetricSpec,
    load_evaluation_document,
    sampling_overlay_bytes,
    sampling_parameters,
)
from stochaflow_afhq_v2.tools.evaluation_metrics import (
    MetricProviderFactory,
    collect_real_test_images,
    default_provider_factory,
    evaluate_reference_metrics,
    load_generated_samples,
    preflight_metric_providers,
    release_metric_device,
    split_fake_samples,
)
from stochaflow_afhq_v2.tools.evaluation_result import (
    SAMPLING_OVERLAY_NAME,
    AFHQV2EvaluationResult,
    default_output_dir,
    materialize_result,
    sha256_file,
    write_exclusive,
)

_BUILDER_NAME = "afhq-v2.class-images"
_SOURCE_NAME = "afhq-v2.official"


@dataclass(frozen=True, slots=True)
class CheckpointSnapshot:
    """One authenticated byte snapshot used by every evaluation lifecycle."""

    source_path: Path
    snapshot_path: Path
    sha256: str
    size_bytes: int


def _resolve_checkpoint_source(checkpoint: str | Path) -> Path:
    source = Path(checkpoint)
    if source.is_dir():
        return CheckpointManager.find_best(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    return source.resolve()


def _snapshot_checkpoint(source: Path, destination: Path) -> CheckpointSnapshot:
    digest = hashlib.sha256()
    copied = 0
    with source.open("rb") as source_handle:
        before = os.fstat(source_handle.fileno())
        with destination.open("xb") as snapshot_handle:
            for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                snapshot_handle.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            snapshot_handle.flush()
            os.fsync(snapshot_handle.fileno())
        after = os.fstat(source_handle.fileno())
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable_fields):
        raise RuntimeError("checkpoint changed while its evaluation snapshot was read")
    if copied != before.st_size:
        raise RuntimeError("checkpoint snapshot byte count changed during capture")
    return CheckpointSnapshot(
        source_path=source,
        snapshot_path=destination,
        sha256=digest.hexdigest(),
        size_bytes=copied,
    )


def _verify_snapshot(snapshot: CheckpointSnapshot) -> None:
    if snapshot.snapshot_path.stat().st_size != snapshot.size_bytes:
        raise RuntimeError("private checkpoint snapshot size changed")
    if sha256_file(snapshot.snapshot_path) != snapshot.sha256:
        raise RuntimeError("private checkpoint snapshot digest changed")


def _published_sampling_result(
    sampling: SamplingRunResult,
    *,
    staging_root: Path,
    final_root: Path,
) -> SamplingRunResult:
    def published(path: Path) -> Path:
        try:
            relative = path.resolve().relative_to(staging_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"sampling artifact escaped private staging root: {path}"
            ) from exc
        return final_root / relative

    return replace(
        sampling,
        output_dir=published(sampling.output_dir),
        artifacts={
            name: published(path)
            for name, path in sampling.artifacts.items()
        },
    )


def _atomic_publish_directory(source: Path, destination: Path) -> None:
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


def _checkpoint_data_bindings(
    inputs: ResolvedSamplingInputs,
) -> DataArtifactBindings:
    metadata = cast(object, inputs.checkpoint.get("metadata"))
    if not isinstance(metadata, dict):
        raise TypeError("checkpoint metadata must be a mapping")
    raw = metadata.get("data_artifacts")
    if raw is None:
        raise ValueError(
            "AFHQ-v2 evaluation requires checkpoint data artifact bindings"
        )
    bindings = DataArtifactBindings.from_dict(
        raw,
        path="checkpoint metadata.data_artifacts",
    )
    bindings.assert_ids(("source",))
    if bindings.identity_for("source").source_name != _SOURCE_NAME:
        raise ValueError(
            "checkpoint data artifact binding is not the official AFHQ-v2 source"
        )
    return bindings


def _checkpoint_progress(checkpoint_path: Path) -> dict[str, int]:
    payload = CheckpointManager.load_payload(checkpoint_path, map_location="cpu")
    epoch = cast(object, payload.get("epoch"))
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
        raise ValueError(
            "AFHQ-v2 evaluation checkpoint epoch must be positive"
        )
    global_step = cast(object, payload.get("global_step"))
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ValueError(
            "AFHQ-v2 evaluation checkpoint global_step must be non-negative"
        )
    return {"epoch": epoch, "global_step": global_step}


def _validate_data_config(inputs: ResolvedSamplingInputs) -> None:
    config = inputs.config
    if config.data.name != _BUILDER_NAME:
        raise ValueError(
            "AFHQ-v2 evaluation requires the afhq-v2.class-images DataBuilder"
        )
    params = cast(object, config.data.params)
    if not isinstance(params, dict):
        raise TypeError("checkpoint data.params must be a mapping")
    source = cast(object, params.get("source"))
    if not isinstance(source, dict) or source.get("name") != _SOURCE_NAME:
        raise ValueError(
            "AFHQ-v2 evaluation requires the official registered data source"
        )
    source_params = cast(object, source.get("params"))
    if not isinstance(source_params, dict):
        raise TypeError("checkpoint source params must be a mapping")
    if source_params.get("resolution") != 128:
        raise ValueError("AFHQ-v2 evaluation requires a 128x128 source artifact")
    materialization = cast(object, source.get("materialization"))
    if not isinstance(materialization, dict):
        raise TypeError("checkpoint source materialization must be a mapping")
    if materialization.get("policy") != "require":
        raise ValueError(
            "AFHQ-v2 evaluation requires source materialization policy: require"
        )
    if materialization.get("verification") != "full":
        raise ValueError(
            "AFHQ-v2 evaluation requires source verification: full"
        )
    image = cast(object, params.get("image"))
    if not isinstance(image, dict):
        raise TypeError("checkpoint data image recipe must be a mapping")
    if (
        image.get("size") != [128, 128]
        or image.get("channels") != 3
        or image.get("require_exact_size") is not True
        or image.get("normalize") is not True
    ):
        raise ValueError(
            "AFHQ-v2 evaluation requires exact normalized 128x128 RGB test images"
        )


def _build_strict_test_loaders(
    extensions: ResolvedExtensions,
    expected: DataArtifactBindings,
) -> DataLoaders:
    loaders = build_data_loaders(
        extensions.config.data,
        seed=extensions.config.experiment.seed,
        strict_resume=True,
        expected_artifacts=expected,
    )
    if loaders.artifact_bindings != expected:
        raise ValueError(
            "evaluation data artifact bindings changed during strict build"
        )
    if loaders.test is None:
        raise ValueError("AFHQ-v2 DataBuilder must expose the official test split")
    return loaders


def _validate_sampling_result(
    sampling: SamplingRunResult,
    document: AFHQV2EvaluationDocument,
    *,
    expected_device: torch.device,
) -> None:
    params = sampling_parameters(document)
    expected_conditions = cast(list[dict[str, int]], params["conditions"])
    metadata = sampling.metadata
    if sampling.device != expected_device:
        raise ValueError("sampling runtime device changed after execution preflight")
    if metadata.get("builder") != SAMPLING_BUILDER_NAME:
        raise ValueError("sampling metadata contains the wrong builder")
    if metadata.get("weights") != params["weights"]:
        raise ValueError("sampling did not use the explicitly frozen weight set")
    if metadata.get("guidance_scale") != params["guidance_scale"]:
        raise ValueError("sampling metadata guidance scale does not match protocol")
    if metadata.get("conditions") != expected_conditions:
        raise ValueError("sampling metadata class allocation does not match protocol")
    if metadata.get("sampler") != params["sampler"]:
        raise ValueError("sampling metadata sampler does not match protocol")
    expected_seed = cast(int, document.sampling_overlay["sampling"]["seed"])
    if sampling.seed != expected_seed:
        raise ValueError("sampling runtime seed does not match evaluation protocol")


def evaluate_checkpoint(
    *,
    config_path: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path | None = None,
    device_name: str | None = None,
    extension_version_policy: ExtensionVersionPolicy = ExtensionVersionPolicy.REJECT,
    extension_acceptance_method: str | None = None,
    provider_factory: MetricProviderFactory | None = None,
) -> AFHQV2EvaluationResult:
    """Evaluate one checkpoint through strict data and sampling lifecycles."""

    document = load_evaluation_document(config_path)
    overlay_encoded = sampling_overlay_bytes(document)
    source_checkpoint = _resolve_checkpoint_source(checkpoint)
    final_root = (
        Path(output_dir).resolve()
        if output_dir is not None
        else default_output_dir(source_checkpoint).resolve()
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
    try:
        snapshot = _snapshot_checkpoint(
            source_checkpoint,
            staging_root / "checkpoint.snapshot.pt",
        )
        overlay_path = staging_root / SAMPLING_OVERLAY_NAME
        write_exclusive(overlay_path, overlay_encoded)
        resolved_inputs = resolve_sampling_inputs(
            config_path=overlay_path,
            checkpoint=snapshot.snapshot_path,
        )
        _verify_snapshot(snapshot)
        checkpoint_progress = _checkpoint_progress(snapshot.snapshot_path)
        inputs = replace(
            resolved_inputs,
            checkpoint_path=source_checkpoint,
        )
        _validate_data_config(inputs)
        expected_bindings = _checkpoint_data_bindings(inputs)
        execution_device = resolve_device(
            device_name or inputs.config.trainer.device
        )
        validate_execution_device(execution_device)
        factory = provider_factory or default_provider_factory
        preflight_metric_providers(
            document.protocol,
            device=execution_device,
            factory=factory,
        )
        extensions = activate_extension_plugins(
            inputs.extension_plan,
            policy=extension_version_policy,
            acceptance_method=extension_acceptance_method,
        )
        loaders = _build_strict_test_loaders(extensions, expected_bindings)
        real_images, real_counts = collect_real_test_images(
            loaders,
            document.protocol,
        )
        sampling = run_resolved_sampling(
            inputs,
            extensions,
            output_dir=staging_root / "sampling",
            device_name=str(execution_device),
        )
        _validate_sampling_result(
            sampling,
            document,
            expected_device=execution_device,
        )
        _verify_snapshot(snapshot)
        samples = load_generated_samples(sampling, document.protocol)
        fake_images, fake_counts = split_fake_samples(
            samples,
            document.protocol,
        )
        release_metric_device(execution_device)
        set_seed(document.protocol.metric_seed)
        metrics, provider_identities = evaluate_reference_metrics(
            real_images=real_images,
            fake_images=fake_images,
            protocol=document.protocol,
            device=execution_device,
            factory=factory,
        )
        _verify_snapshot(snapshot)
        snapshot.snapshot_path.unlink()
        result_path, result_sha256, digest_path, manifest_path = (
            materialize_result(
                root=staging_root,
                document=document,
                inputs=inputs,
                extensions=extensions,
                sampling=sampling,
                expected_bindings=expected_bindings,
                real_counts=real_counts,
                fake_counts=fake_counts,
                metrics=metrics,
                provider_identities=provider_identities,
                checkpoint_sha256=snapshot.sha256,
                checkpoint_progress=checkpoint_progress,
                overlay_path=overlay_path,
            )
        )
        if final_root.exists():
            raise FileExistsError(
                f"evaluation output directory appeared before publish: {final_root}"
            )
        published_sampling = _published_sampling_result(
            sampling,
            staging_root=staging_root,
            final_root=final_root,
        )
        _atomic_publish_directory(staging_root, final_root)
        return AFHQV2EvaluationResult(
            output_dir=final_root,
            result_path=final_root / result_path.name,
            result_sha256=result_sha256,
            digest_path=final_root / digest_path.name,
            manifest_path=final_root / manifest_path.name,
            sampling=published_sampling,
        )
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)


__all__ = [
    "AFHQV2EvaluationDocument",
    "AFHQV2EvaluationProtocol",
    "AFHQV2EvaluationResult",
    "AFHQV2MetricSpec",
    "MetricProviderFactory",
    "evaluate_checkpoint",
    "load_evaluation_document",
]

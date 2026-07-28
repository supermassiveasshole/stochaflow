"""Orchestrate AFHQ-v2 post-training quality evaluation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import torch

from stochaflow.sampling.runtime import (
    SamplingRunResult,
    resolve_sampling_inputs,
    run_resolved_sampling,
)
from stochaflow.utils.device import validate_execution_device
from stochaflow.utils.factory import resolve_device
from stochaflow.utils.plugins import (
    ExtensionVersionPolicy,
    activate_extension_plugins,
)
from stochaflow.utils.seed import set_seed
from stochaflow_afhq_v2.tools import (
    evaluation_inputs,
    evaluation_workspace,
)
from stochaflow_afhq_v2.tools.evaluation_config import (
    SAMPLING_RECIPE_NAME,
    AFHQV2EvaluationDocument,
    AFHQV2EvaluationProtocol,
    AFHQV2MetricSpec,
    load_evaluation_document,
    sample_request_bytes,
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
    SAMPLE_REQUEST_NAME,
    AFHQV2EvaluationResult,
    materialize_result,
    write_exclusive,
)


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
    if sampling.recipe_name != SAMPLING_RECIPE_NAME:
        raise ValueError("sampling runtime selected the wrong inference recipe")
    if metadata.get("weights") != params["weights"]:
        raise ValueError("sampling did not use the explicitly frozen weight set")
    if metadata.get("guidance_scale") != params["guidance_scale"]:
        raise ValueError("sampling metadata guidance scale does not match protocol")
    if metadata.get("conditions") != expected_conditions:
        raise ValueError("sampling metadata class allocation does not match protocol")
    if metadata.get("sampler") != params["sampler"]:
        raise ValueError("sampling metadata sampler does not match protocol")
    expected_seed = cast(int, document.sample_request["sampling"]["seed"])
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
    request_encoded = sample_request_bytes(document)
    source_checkpoint = evaluation_inputs.resolve_checkpoint_source(checkpoint)
    workspace = evaluation_workspace.EvaluationWorkspace.create(
        checkpoint_path=source_checkpoint,
        output_dir=output_dir,
    )
    try:
        snapshot = evaluation_inputs.snapshot_checkpoint(
            source_checkpoint,
            workspace.staging_root / "checkpoint.snapshot.pt",
        )
        request_path = workspace.staging_root / SAMPLE_REQUEST_NAME
        write_exclusive(request_path, request_encoded)
        checkpoint_progress = evaluation_inputs.checkpoint_progress(
            snapshot.snapshot_path
        )
        evaluation_inputs.verify_checkpoint_snapshot(snapshot)
        inputs = resolve_sampling_inputs(
            config_path=request_path,
            checkpoint=snapshot.snapshot_path,
        )
        evaluation_inputs.verify_checkpoint_snapshot(snapshot)
        inputs = replace(
            inputs,
            checkpoint_path=source_checkpoint,
        )
        evaluation_inputs.validate_data_config(inputs)
        expected_bindings = evaluation_inputs.checkpoint_data_bindings(inputs)
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
        loaders = evaluation_inputs.build_strict_test_loaders(
            extensions,
            expected_bindings,
        )
        try:
            real_images, real_counts = collect_real_test_images(
                loaders,
                document.protocol,
            )
        finally:
            del loaders
        sampling = run_resolved_sampling(
            inputs,
            extensions,
            output_dir=workspace.staging_root / "sampling",
            device_name=str(execution_device),
        )
        _validate_sampling_result(
            sampling,
            document,
            expected_device=execution_device,
        )
        inputs = evaluation_inputs.retain_result_checkpoint_header(inputs)
        release_metric_device(execution_device)
        evaluation_inputs.verify_checkpoint_snapshot(snapshot)
        snapshot.snapshot_path.unlink()
        samples = load_generated_samples(sampling, document.protocol)
        fake_images, fake_counts = split_fake_samples(
            samples,
            document.protocol,
        )
        del samples
        set_seed(document.protocol.metric_seed)
        metrics, provider_identities = evaluate_reference_metrics(
            real_images=real_images,
            fake_images=fake_images,
            protocol=document.protocol,
            device=execution_device,
            factory=factory,
        )
        del real_images, fake_images
        result_path, result_sha256, digest_path, manifest_path = (
            materialize_result(
                root=workspace.staging_root,
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
            request_path=request_path,
            )
        )
        published_sampling = workspace.published_sampling_result(sampling)
        workspace.publish()
        return AFHQV2EvaluationResult(
            output_dir=workspace.final_root,
            result_path=workspace.final_root / result_path.name,
            result_sha256=result_sha256,
            digest_path=workspace.final_root / digest_path.name,
            manifest_path=workspace.final_root / manifest_path.name,
            sampling=published_sampling,
        )
    finally:
        workspace.cleanup()


__all__ = [
    "AFHQV2EvaluationDocument",
    "AFHQV2EvaluationProtocol",
    "AFHQV2EvaluationResult",
    "AFHQV2MetricSpec",
    "MetricProviderFactory",
    "evaluate_checkpoint",
    "load_evaluation_document",
]

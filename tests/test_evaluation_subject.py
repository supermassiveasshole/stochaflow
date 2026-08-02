"""Tests for checkpoint-backed formal evaluation subjects."""

from __future__ import annotations

import hashlib
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

import stochaflow.inference.checkpoint as checkpoint_module
from stochaflow.data.artifacts import DataArtifactBindings
from stochaflow.evaluation.config import CheckpointSubjectConfig
from stochaflow.evaluation.subject import (
    load_checkpoint_subject,
    resolve_checkpoint_subject,
)
from stochaflow.utils.checkpoint import CHECKPOINT_FORMAT_VERSION
from stochaflow.utils.config import ComponentConfig


def checkpoint_config() -> dict[str, Any]:
    """Return a minimal valid training authority for subject reconstruction."""

    return {
        "experiment": {
            "name": "evaluation-subject",
            "seed": 11,
            "output_dir": "outputs/evaluation-subject",
            "exp_id": "subject-run",
        },
        "data": {"name": "opaque_data", "params": {"split": "frozen"}},
        "model": {"name": "opaque_linear", "params": {"width": 2}},
        "training": {"name": "opaque_training", "params": {}},
    }


def data_artifact_identity() -> dict[str, Any]:
    """Return one strict checkpoint-bound artifact identity."""

    return {
        "schema_version": 2,
        "bindings": [
            {
                "id": "dataset",
                "identity": {
                    "schema_version": 2,
                    "kind": "managed",
                    "artifact_type": "opaque-records",
                    "source_name": "frozen-source",
                    "source_digest": "1" * 64,
                    "materializer_name": "frozen-materializer",
                    "materialization_digest": "2" * 64,
                    "content_digest": "3" * 64,
                    "artifact_digest": "4" * 64,
                    "manifest_sha256": "5" * 64,
                },
            }
        ],
    }


def checkpoint_payload(*, include_ema: bool = True) -> dict[str, Any]:
    """Return an inference-complete v12 payload without restore-only state."""

    raw_model = nn.Linear(2, 1)
    ema_model = nn.Linear(2, 1)
    with torch.no_grad():
        raw_model.weight.fill_(1.0)
        raw_model.bias.fill_(0.25)
        ema_model.weight.fill_(2.0)
        ema_model.bias.fill_(0.5)
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": 3,
        "global_step": 17,
        "config": checkpoint_config(),
        "model_state_dict": raw_model.state_dict(),
        "inference_recipe": None,
        "inference_asset_descriptors": {},
        "metadata": {
            "extension_plugins": [
                {
                    "name": "subject_extension",
                    "distribution": "subject-extension",
                    "version": "1.2.3",
                    "target": "subject_extension.plugin",
                }
            ],
            "selected_components": {
                "model": "opaque_linear",
                "training_builder": "opaque_training",
            },
            "lineage": {"resumed_from": None, "parents": ["seed-run"]},
            "data_artifacts": data_artifact_identity(),
            "nested": {"labels": ["a", "b"]},
        },
    }
    if include_ema:
        payload["ema_model_state_dict"] = ema_model.state_dict()
    return payload


def save_checkpoint(tmp_path: Path, payload: object) -> Path:
    """Write a subject fixture without invoking strict training restore checks."""

    path = tmp_path / "subject.pt"
    torch.save(payload, path)
    return path


def linear_model_factory(_declaration: ComponentConfig) -> nn.Module:
    """Build the primary model declared by the checkpoint fixture."""

    return nn.Linear(2, 1)


class FstatMutationProxy:
    """Return one altered second source snapshot without patching global os."""

    def __init__(self, real_fstat: Any, field: str) -> None:
        self.real_fstat = real_fstat
        self.field = field
        self.calls = 0

    def fstat(self, descriptor: int) -> object:
        """Mirror the first fstat and mutate one stable field on the second."""

        observed = self.real_fstat(descriptor)
        self.calls += 1
        values = {
            "st_dev": observed.st_dev,
            "st_ino": observed.st_ino,
            "st_size": observed.st_size,
            "st_mtime_ns": observed.st_mtime_ns,
        }
        if self.calls == 2:
            values[self.field] += 1
        return SimpleNamespace(**values)


def test_load_subject_is_safe_and_ignores_restore_only_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = checkpoint_payload()
    assert not {
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "grad_scaler_state_dict",
        "rng_state",
    } & set(payload)
    path = save_checkpoint(tmp_path, payload)
    observed: dict[str, object] = {}
    real_load = torch.load

    def recording_load(*args: Any, **kwargs: Any) -> Any:
        observed["map_location"] = kwargs.get("map_location")
        observed["weights_only"] = kwargs.get("weights_only")
        loaded = real_load(*args, **kwargs)
        observed["payload"] = loaded
        return loaded

    monkeypatch.setattr(checkpoint_module.torch, "load", recording_load)
    inputs = load_checkpoint_subject(
        CheckpointSubjectConfig(kind="checkpoint", path=path, weights="raw")
    )

    assert observed["map_location"] == "cpu"
    assert observed["weights_only"] is True
    assert inputs.path == path.resolve()
    assert inputs.path.is_absolute()
    assert inputs.content_digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert inputs.format_version == 12
    assert (inputs.epoch, inputs.global_step) == (3, 17)
    assert inputs.requested_weights == "raw"
    assert inputs.extension_provenance[0].name == "subject_extension"
    assert isinstance(inputs.data_artifacts, DataArtifactBindings)
    assert inputs.data_artifacts.ids == ("dataset",)

    loaded = cast(dict[str, Any], observed["payload"])
    loaded_keys = set(loaded)
    loaded_weight = loaded["model_state_dict"]["weight"].clone()
    loaded_metadata = deepcopy(loaded["metadata"])
    resolved = resolve_checkpoint_subject(
        inputs,
        device="cpu",
        model_factory=linear_model_factory,
    )
    resolved_model = cast(nn.Linear, resolved.model)

    assert set(loaded) == loaded_keys
    assert torch.equal(loaded["model_state_dict"]["weight"], loaded_weight)
    assert loaded["metadata"] == loaded_metadata
    assert torch.equal(resolved_model.weight, torch.ones_like(resolved_model.weight))


def test_subject_digest_and_torch_load_use_the_same_snapshot_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = save_checkpoint(tmp_path, checkpoint_payload())
    original_bytes = path.read_bytes()
    observed: dict[str, object] = {}
    real_load = torch.load

    def recording_load(
        source: Any,
        *,
        map_location: str,
        weights_only: bool,
    ) -> Any:
        position = source.tell()
        snapshot_bytes = source.read()
        source.seek(position)
        observed["snapshot_bytes"] = snapshot_bytes
        observed["position"] = position
        observed["map_location"] = map_location
        observed["weights_only"] = weights_only
        path.write_bytes(b"source changed after the stable snapshot")
        return real_load(
            source,
            map_location=map_location,
            weights_only=weights_only,
        )

    monkeypatch.setattr(checkpoint_module.torch, "load", recording_load)

    inputs = load_checkpoint_subject(
        CheckpointSubjectConfig(kind="checkpoint", path=path, weights="raw")
    )

    snapshot_bytes = cast(bytes, observed["snapshot_bytes"])
    assert observed["position"] == 0
    assert observed["map_location"] == "cpu"
    assert observed["weights_only"] is True
    assert snapshot_bytes == original_bytes
    assert inputs.content_digest == hashlib.sha256(snapshot_bytes).hexdigest()
    assert inputs.epoch == 3
    assert hashlib.sha256(path.read_bytes()).hexdigest() != inputs.content_digest


@pytest.mark.parametrize("changed_field", ["st_ino", "st_size", "st_mtime_ns"])
def test_subject_fails_closed_when_source_fstat_changes_during_snapshot(
    changed_field: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = save_checkpoint(tmp_path, checkpoint_payload())
    source_bytes = path.read_bytes()
    snapshot_dir = tmp_path / "temporary-snapshots"
    snapshot_dir.mkdir()
    real_temporary_file = checkpoint_module.tempfile.TemporaryFile

    def isolated_temporary_file(*args: Any, **kwargs: Any) -> Any:
        kwargs["dir"] = snapshot_dir
        return real_temporary_file(*args, **kwargs)

    load_calls = 0

    def forbidden_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal load_calls
        del args, kwargs
        load_calls += 1
        raise AssertionError("unstable checkpoint snapshot must not be deserialized")

    fstat_proxy = FstatMutationProxy(os.fstat, changed_field)
    monkeypatch.setattr(checkpoint_module, "os", fstat_proxy)
    monkeypatch.setattr(
        checkpoint_module.tempfile,
        "TemporaryFile",
        isolated_temporary_file,
    )
    monkeypatch.setattr(checkpoint_module.torch, "load", forbidden_load)

    with pytest.raises(RuntimeError, match=r"changed while.*snapshot was read"):
        load_checkpoint_subject(
            CheckpointSubjectConfig(kind="checkpoint", path=path, weights="raw")
        )

    assert fstat_proxy.calls == 2
    assert load_calls == 0
    assert path.read_bytes() == source_bytes
    assert list(snapshot_dir.iterdir()) == []


def test_subject_snapshots_config_metadata_and_portable_identity(
    tmp_path: Path,
) -> None:
    path = save_checkpoint(tmp_path, checkpoint_payload())
    inputs = load_checkpoint_subject(
        CheckpointSubjectConfig(kind="checkpoint", path=path, weights="ema")
    )

    nested_config = cast(dict[str, Any], inputs.config["data"])
    nested_metadata = cast(dict[str, Any], inputs.metadata["nested"])
    lineage = cast(dict[str, Any], inputs.lineage)
    with pytest.raises(TypeError):
        cast(dict[str, Any], inputs.config)["other"] = 1
    with pytest.raises(TypeError):
        nested_config["other"] = 1
    with pytest.raises(TypeError):
        nested_metadata["other"] = 1
    with pytest.raises(TypeError):
        lineage["other"] = 1
    assert nested_metadata["labels"] == ("a", "b")

    first_copy = inputs.training_config_copy()
    first_copy.experiment.name = "mutated"
    assert inputs.training_config_copy().experiment.name == "evaluation-subject"

    resolved = resolve_checkpoint_subject(
        inputs,
        device=torch.device("cpu"),
        model_factory=linear_model_factory,
    )
    resolved_model = cast(nn.Linear, resolved.model)
    assert resolved.kind == "checkpoint"
    assert resolved.requested_weights == "ema"
    assert resolved.resolved_weights == "ema"
    assert torch.equal(
        resolved_model.weight,
        torch.full_like(resolved_model.weight, 2.0),
    )
    assert not resolved.model.training
    assert all(not parameter.requires_grad for parameter in resolved.model.parameters())

    identity = resolved.identity
    assert identity["kind"] == "checkpoint"
    assert identity["path"] == str(path.resolve())
    assert identity["sha256"] == resolved.content_digest
    assert identity["format_version"] == 12
    assert identity["epoch"] == 3
    assert identity["global_step"] == 17
    assert identity["requested_weights"] == "ema"
    assert identity["resolved_weights"] == "ema"
    assert identity["selected_components"] == {
        "model": "opaque_linear",
        "training_builder": "opaque_training",
    }
    assert identity["lineage"] == {
        "resumed_from": None,
        "parents": ("seed-run",),
    }
    assert cast(dict[str, Any], identity["data_artifacts"])["schema_version"] == 2
    with pytest.raises(TypeError):
        cast(dict[str, Any], identity)["other"] = "value"


def test_raw_and_ema_are_resolved_before_the_builder_boundary(tmp_path: Path) -> None:
    path = save_checkpoint(tmp_path, checkpoint_payload())
    raw_inputs = load_checkpoint_subject(
        CheckpointSubjectConfig(kind="checkpoint", path=path, weights="raw")
    )
    ema_inputs = load_checkpoint_subject(
        CheckpointSubjectConfig(kind="checkpoint", path=path, weights="ema")
    )

    raw = resolve_checkpoint_subject(
        raw_inputs,
        device="cpu",
        model_factory=linear_model_factory,
    )
    ema = resolve_checkpoint_subject(
        ema_inputs,
        device="cpu",
        model_factory=linear_model_factory,
    )
    raw_model = cast(nn.Linear, raw.model)
    ema_model = cast(nn.Linear, ema.model)

    assert raw.resolved_weights == "raw"
    assert ema.resolved_weights == "ema"
    assert torch.equal(raw_model.weight, torch.full_like(raw_model.weight, 1.0))
    assert torch.equal(ema_model.weight, torch.full_like(ema_model.weight, 2.0))
    assert not hasattr(raw, "provider")
    assert not hasattr(raw, "optimizer")
    assert not hasattr(raw, "training_plan")


def test_explicit_ema_request_rejects_checkpoint_without_ema(tmp_path: Path) -> None:
    path = save_checkpoint(tmp_path, checkpoint_payload(include_ema=False))
    inputs = load_checkpoint_subject(
        CheckpointSubjectConfig(kind="checkpoint", path=path, weights="ema")
    )

    with pytest.raises(ValueError, match="EMA weights were requested"):
        resolve_checkpoint_subject(
            inputs,
            device="cpu",
            model_factory=linear_model_factory,
        )


@pytest.mark.parametrize(
    ("payload_update", "message"),
    [
        ({"format_version": 11}, "unsupported"),
        ({"format_version": True}, "exact integer"),
        ({"epoch": 0}, "epoch"),
        ({"global_step": -1}, "global_step"),
    ],
)
def test_subject_rejects_unsupported_or_invalid_checkpoint_identity(
    tmp_path: Path,
    payload_update: dict[str, object],
    message: str,
) -> None:
    payload = checkpoint_payload()
    payload.update(payload_update)
    path = save_checkpoint(tmp_path, payload)

    with pytest.raises((TypeError, ValueError), match=message):
        load_checkpoint_subject(
            CheckpointSubjectConfig(kind="checkpoint", path=path, weights="raw")
        )


def test_subject_rejects_non_mapping_checkpoint_payload(tmp_path: Path) -> None:
    path = save_checkpoint(tmp_path, ["not", "a", "mapping"])

    with pytest.raises(TypeError, match="dictionary payload"):
        load_checkpoint_subject(
            CheckpointSubjectConfig(kind="checkpoint", path=path, weights="raw")
        )


def test_relative_checkpoint_path_requires_explicit_config_base(tmp_path: Path) -> None:
    config_dir = tmp_path / "evaluation-configs"
    config_dir.mkdir()
    path = save_checkpoint(config_dir, checkpoint_payload())
    declaration = CheckpointSubjectConfig(
        kind="checkpoint",
        path=Path("subject.pt"),
        weights="raw",
    )

    with pytest.raises(ValueError, match="explicit base_dir"):
        load_checkpoint_subject(declaration)

    inputs = load_checkpoint_subject(declaration, base_dir=config_dir)
    assert inputs.path == path.resolve()

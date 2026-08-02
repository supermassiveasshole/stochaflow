"""Tests for reproducibility metadata shared by runtime manifests."""

from pathlib import Path

import pytest
import yaml

from stochaflow.utils import run_manifest
from stochaflow.utils.config import ComponentConfig, SampleConfig, load_config_dict
from stochaflow.utils.run_manifest import (
    selected_sampling_component_identities,
    selected_training_component_identities,
)


def test_manifest_replacement_is_atomic_and_cleans_temporary_file(
    monkeypatch,
    tmp_path,
) -> None:
    path = tmp_path / "run_manifest.yaml"
    run_manifest.write_yaml_manifest(path, {"status": "running"})

    def fail_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        run_manifest.write_yaml_manifest(path, {"status": "completed"})

    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
        "status": "running"
    }
    assert list(tmp_path.glob(".run_manifest.yaml.*.tmp")) == []


def test_selected_components_are_top_level_identities_only() -> None:
    config = load_config_dict(
        {
            "experiment": {"name": "selection-test"},
            "data": {
                "name": "project.data",
                "params": {"sampler": {"name": "private.data-sampler"}},
            },
            "model": {
                "name": "project.model",
                "params": {"backbone": {"name": "private.backbone"}},
            },
            "training": {
                "name": "project.training",
                "params": {"teacher": {"name": "private.teacher"}},
            },
            "objective": None,
            "process": None,
            "optimizer": {
                "name": "torch.optim.AdamW",
                "params": {"lr": 0.001},
            },
            "lr_scheduler": None,
            "logging": {
                "backends": [
                    {"name": "project.logger-b", "params": {}},
                    {"name": "project.logger-a", "params": {}},
                ]
            },
            "diagnostics": [
                {"name": "project.diagnostic-b", "params": {}},
                {"name": "project.diagnostic-a", "params": {}},
            ],
        }
    )

    assert selected_training_component_identities(
        config,
        inference_recipe="project.direct",
    ) == {
        "data_builder": "project.data",
        "model": "project.model",
        "training_builder": "project.training",
        "objective": None,
        "process": None,
        "optimizer": "torch.optim.AdamW",
        "lr_scheduler": None,
        "inference_recipe": "project.direct",
        "loggers": ["project.logger-b", "project.logger-a"],
        "diagnostics": [
            "project.diagnostic-b",
            "project.diagnostic-a",
        ],
        "metrics": [],
    }
    sample = SampleConfig(
        sampler=ComponentConfig("private.solver"),
        options={},
        shape=None,
        num_samples=2,
        batch_size=1,
        seed=7,
        writers=[
            ComponentConfig("project.writer-b"),
            ComponentConfig("project.writer-a"),
        ],
    )
    assert selected_sampling_component_identities(
        config,
        sample,
        inference_recipe="project.direct",
    ) == {
        "model": "project.model",
        "process": None,
        "inference_recipe": "project.direct",
        "sampler": "private.solver",
        "artifact_writers": ["project.writer-b", "project.writer-a"],
    }


def test_selected_components_include_present_optional_roles() -> None:
    config = load_config_dict(
        {
            "experiment": {"name": "selection-test"},
            "data": {"name": "project.data", "params": {}},
            "model": {"name": "project.model", "params": {}},
            "training": {"name": "project.training", "params": {}},
            "objective": {"name": "project.objective", "params": {}},
            "process": {"name": "project.process", "params": {}},
            "lr_scheduler": {
                "name": "torch.optim.lr_scheduler.StepLR",
                "interval": "epoch",
                "params": {"step_size": 1},
            },
        }
    )

    selected = selected_training_component_identities(config)
    assert selected["objective"] == "project.objective"
    assert selected["process"] == "project.process"
    assert selected["lr_scheduler"] == "torch.optim.lr_scheduler.StepLR"
    assert selected["inference_recipe"] is None
    assert "sampler" not in selected

"""Tests for reproducibility metadata shared by runtime manifests."""

from stochaflow.utils.config import load_config_dict
from stochaflow.utils.run_manifest import selected_component_identities


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
            "sampling": {
                "builder": {
                    "name": "project.direct",
                    "params": {"sampler": {"name": "private.solver"}},
                },
                "writers": [
                    {"name": "project.writer-b", "params": {}},
                    {"name": "project.writer-a", "params": {}},
                ],
            },
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

    assert selected_component_identities(config) == {
        "data_builder": "project.data",
        "model": "project.model",
        "training_builder": "project.training",
        "objective": None,
        "process": None,
        "optimizer": "torch.optim.AdamW",
        "lr_scheduler": None,
        "sampling_builder": "project.direct",
        "sampling_artifact_writers": [
            "project.writer-b",
            "project.writer-a",
        ],
        "loggers": ["project.logger-b", "project.logger-a"],
        "diagnostics": [
            "project.diagnostic-b",
            "project.diagnostic-a",
        ],
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
            "sampling": {"builder": None},
        }
    )

    selected = selected_component_identities(config)
    assert selected["objective"] == "project.objective"
    assert selected["process"] == "project.process"
    assert selected["lr_scheduler"] == "torch.optim.lr_scheduler.StepLR"
    assert selected["sampling_builder"] is None

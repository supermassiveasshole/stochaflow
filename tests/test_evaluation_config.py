"""Tests for the standalone evaluation configuration authority."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

import stochaflow.evaluation.config as evaluation_config_module
from stochaflow.evaluation import (
    CheckpointSubjectConfig,
    EvaluationConfig,
    EvaluationDataConfig,
    EvaluationExtensionsConfig,
    EvaluationProtocol,
    PredictionArtifactSubjectConfig,
    evaluation_config_to_dict,
    load_evaluation_config,
    load_evaluation_config_dict,
)
from stochaflow.metrics import MetricSpec
from stochaflow.utils.config import ComponentConfig, ConfigError


def valid_evaluation_config() -> dict[str, Any]:
    """Return one complete standalone evaluation declaration."""

    return {
        "version": 1,
        "name": "sr-x4-final-test",
        "purpose": "final_test",
        "extensions": {"plugins": ["example-evaluation"]},
        "subject": {
            "kind": "checkpoint",
            "path": "outputs/sr/checkpoints/best.pt",
            "weights": "ema",
        },
        "data": {"source": "checkpoint", "split": "test"},
        "evaluation": {
            "name": "super_resolution_paired",
            "params": {
                "inference": {"scale": 4, "tile": None},
                "sample_ids": ["image-001", "image-017"],
            },
        },
        "metrics": [
            {
                "id": "psnr_rgb",
                "name": "psnr",
                "channel": "sr.prediction_target",
                "params": {
                    "data_range": 1.0,
                    "preprocess": {"crop_border": 4, "bands": [0, 1, 2]},
                },
            }
        ],
        "protocol": {
            "id": "sr-x4-rgb-v1",
            "expected_examples": 100,
            "strict_complete": True,
        },
    }


def test_load_evaluation_config_dict_builds_standalone_authority() -> None:
    raw = valid_evaluation_config()

    config = load_evaluation_config_dict(raw)

    assert config == EvaluationConfig(
        version=1,
        name="sr-x4-final-test",
        purpose="final_test",
        extensions=EvaluationExtensionsConfig(("example-evaluation",)),
        subject=CheckpointSubjectConfig(
            kind="checkpoint",
            path=Path("outputs/sr/checkpoints/best.pt"),
            weights="ema",
        ),
        data=EvaluationDataConfig(source="checkpoint", split="test"),
        evaluation=ComponentConfig(
            name="super_resolution_paired",
            params=cast(dict[str, Any], config.evaluation.params),
        ),
        metrics=config.metrics,
        protocol=EvaluationProtocol(
            id="sr-x4-rgb-v1",
            expected_examples=100,
            strict_complete=True,
        ),
    )
    assert isinstance(config.metrics[0], MetricSpec)
    assert config.metrics[0].channel == "sr.prediction_target"
    assert config.extensions.plugins == ("example-evaluation",)


def test_loaded_config_snapshots_nested_mappings_and_sequences() -> None:
    raw = valid_evaluation_config()
    config = load_evaluation_config_dict(raw)
    raw["evaluation"]["params"]["inference"]["scale"] = 8
    raw["evaluation"]["params"]["sample_ids"].append("image-042")
    raw["metrics"][0]["params"]["preprocess"]["crop_border"] = 0

    inference = cast(dict[str, Any], config.evaluation.params["inference"])
    preprocess = cast(dict[str, Any], config.metrics[0].params["preprocess"])
    assert inference["scale"] == 4
    assert config.evaluation.params["sample_ids"] == ("image-001", "image-017")
    assert preprocess["crop_border"] == 4
    assert preprocess["bands"] == (0, 1, 2)

    with pytest.raises(TypeError):
        config.evaluation.params["other"] = 1
    with pytest.raises(TypeError):
        inference["scale"] = 8
    with pytest.raises(TypeError):
        config.metrics[0].params["other"] = 1
    with pytest.raises(TypeError):
        preprocess["crop_border"] = 0


def test_load_evaluation_config_rejects_duplicate_yaml_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(
        "\n".join(
            (
                "version: 1",
                "name: first",
                "name: second",
                "purpose: final_test",
                "subject: {kind: checkpoint, path: model.pt, weights: ema}",
                "data: {source: checkpoint, split: test}",
                "evaluation: {name: example, params: {}}",
                "metrics: []",
                "protocol:",
                "  id: example-v1",
                "  expected_examples: 1",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="duplicate field 'name'"):
        load_evaluation_config(config_path)


def test_config_snapshot_parses_the_same_bytes_after_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "evaluation.yaml"
    original = valid_evaluation_config()
    replacement = deepcopy(original)
    replacement["name"] = "replacement-authority"
    original_bytes = yaml.safe_dump(original, sort_keys=False).encode("utf-8")
    config_path.write_bytes(original_bytes)
    parse_snapshot = evaluation_config_module.load_evaluation_config_bytes

    def mutate_before_parse(encoded: bytes) -> EvaluationConfig:
        config_path.write_text(
            yaml.safe_dump(replacement, sort_keys=False),
            encoding="utf-8",
        )
        return parse_snapshot(encoded)

    monkeypatch.setattr(
        evaluation_config_module,
        "load_evaluation_config_bytes",
        mutate_before_parse,
    )

    config, encoded = evaluation_config_module.load_evaluation_config_snapshot(
        config_path
    )

    assert encoded == original_bytes
    assert config.name == original["name"]
    assert load_evaluation_config(config_path).name == replacement["name"]


@pytest.mark.parametrize(
    ("path", "unknown"),
    [
        ((), "top_level_extra"),
        (("extensions",), "extra"),
        (("subject",), "model"),
        (("data",), "batch_size"),
        (("evaluation",), "protocol"),
        (("metrics", 0), "phase"),
        (("protocol",), "seed"),
    ],
)
def test_config_rejects_unknown_fields(
    path: tuple[str | int, ...],
    unknown: str,
) -> None:
    raw = valid_evaluation_config()
    target: Any = raw
    for part in path:
        target = target[part]
    target[unknown] = True

    with pytest.raises(ConfigError, match="unknown field"):
        load_evaluation_config_dict(raw)


@pytest.mark.parametrize(
    ("purpose", "split", "valid"),
    [
        ("selection_candidate", "validation", True),
        ("selection_candidate", "test", False),
        ("final_test", "validation", False),
        ("final_test", "test", True),
        ("benchmark", "validation", True),
        ("benchmark", "test", True),
    ],
)
def test_config_enforces_purpose_split_governance(
    purpose: str,
    split: str,
    valid: bool,
) -> None:
    raw = valid_evaluation_config()
    raw["purpose"] = purpose
    raw["data"]["split"] = split

    if valid:
        config = load_evaluation_config_dict(raw)
        assert config.purpose == purpose
        assert config.data.split == split
    else:
        with pytest.raises(ConfigError, match=r"purpose.*split"):
            load_evaluation_config_dict(raw)


@pytest.mark.parametrize("weights", ["auto", "best", "raw_or_ema", None])
def test_checkpoint_subject_requires_explicit_raw_or_ema(weights: object) -> None:
    raw = valid_evaluation_config()
    raw["subject"]["weights"] = weights

    with pytest.raises(ConfigError, match=r"subject.weights.*raw.*ema"):
        load_evaluation_config_dict(raw)


def test_config_rejects_non_checkpoint_subject_and_data_source() -> None:
    raw = valid_evaluation_config()
    raw["subject"]["kind"] = "predictions"
    with pytest.raises(ConfigError, match=r"subject.kind.*checkpoint"):
        load_evaluation_config_dict(raw)

    raw = valid_evaluation_config()
    raw["data"]["source"] = "override"
    with pytest.raises(ConfigError, match=r"data.source.*checkpoint"):
        load_evaluation_config_dict(raw)


def test_prediction_artifact_subject_is_an_explicit_offline_authority() -> None:
    raw = valid_evaluation_config()
    raw["subject"] = {
        "kind": "prediction_artifact",
        "path": "producer/predictions/manifest.json",
    }
    raw["data"]["source"] = "prediction_artifact"

    config = load_evaluation_config_dict(raw)

    assert config.subject == PredictionArtifactSubjectConfig(
        kind="prediction_artifact",
        path=Path("producer/predictions/manifest.json"),
    )
    assert config.data.source == "prediction_artifact"
    assert evaluation_config_to_dict(config)["subject"] == {
        "kind": "prediction_artifact",
        "path": str(Path("producer/predictions/manifest.json")),
    }


@pytest.mark.parametrize(
    ("subject_kind", "data_source"),
    [
        ("checkpoint", "prediction_artifact"),
        ("prediction_artifact", "checkpoint"),
    ],
)
def test_subject_and_data_authorities_must_match(
    subject_kind: str,
    data_source: str,
) -> None:
    raw = valid_evaluation_config()
    if subject_kind == "prediction_artifact":
        raw["subject"] = {
            "kind": subject_kind,
            "path": "predictions/manifest.json",
        }
    raw["data"]["source"] = data_source

    with pytest.raises(ConfigError, match=r"subject.kind.*data.source"):
        load_evaluation_config_dict(raw)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.update(version=True), "version must be integer 1"),
        (lambda raw: raw.update(name=" spaced "), "name.*whitespace"),
        (
            lambda raw: raw["protocol"].update(expected_examples=0),
            "expected_examples.*positive",
        ),
        (
            lambda raw: raw["protocol"].update(strict_complete=1),
            "strict_complete.*bool",
        ),
        (lambda raw: raw.update(metrics={}), "metrics must be a list"),
        (
            lambda raw: raw["extensions"].update(plugins=["duplicate", "duplicate"]),
            "duplicate plugin",
        ),
    ],
)
def test_config_rejects_invalid_scalar_and_container_values(
    mutation: Any,
    message: str,
) -> None:
    raw = valid_evaluation_config()
    mutation(raw)

    with pytest.raises(ConfigError, match=message):
        load_evaluation_config_dict(raw)


def test_config_rejects_duplicate_metric_ids() -> None:
    raw = valid_evaluation_config()
    duplicate = deepcopy(raw["metrics"][0])
    duplicate["name"] = "another_metric"
    raw["metrics"].append(duplicate)

    with pytest.raises(ConfigError, match="duplicate metric id 'psnr_rgb'"):
        load_evaluation_config_dict(raw)

"""Open-closed tests for Strategy-level Gaussian training extension."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from fixtures.gaussian_training_extension import (
    MODEL_NAME as PLUGIN_MODEL_NAME,
)
from fixtures.gaussian_training_extension import (
    TRAINING_NAME as PLUGIN_TRAINING_NAME,
)
from fixtures.gaussian_training_extension import (
    PluginGaussianDenoiser,
    PluginScaledGaussianStrategy,
)
from torch.utils.data import DataLoader, Dataset

import stochaflow.training.gaussian as gaussian_training
from stochaflow.data import DataLoaders
from stochaflow.processes import DiscreteGaussianProcess
from stochaflow.scripts import experiment_runner
from stochaflow.scripts.cli import build_argument_parser
from stochaflow.training.builder import build_training_plan
from stochaflow.training.objectives import MSEObjective
from stochaflow.training.strategy import validate_train_step_output
from stochaflow.utils import plugins
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.config import ComponentConfig, load_config_dict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_NAME = "gaussian-strategy-lifecycle"
PLUGIN_DISTRIBUTION = "gaussian-strategy-lifecycle-tests"
PLUGIN_VERSION = "1.2.3"
PLUGIN_TARGET = "fixtures.gaussian_training_extension"


@dataclass(frozen=True, slots=True)
class InstalledPluginDistribution:
    """Distribution metadata exposed by the synthetic installed plugin."""

    name: str
    version: str

    @property
    def metadata(self) -> dict[str, str]:
        """Expose the canonical installed distribution name."""

        return {"Name": self.name}


@dataclass(frozen=True, slots=True)
class InstalledPluginEntryPoint:
    """Minimal importlib entry-point record used by plugin discovery."""

    name: str = PLUGIN_NAME
    value: str = PLUGIN_TARGET
    distribution: str = PLUGIN_DISTRIBUTION
    version: str = PLUGIN_VERSION

    @property
    def dist(self) -> InstalledPluginDistribution:
        """Return the owning installed distribution metadata."""

        return InstalledPluginDistribution(self.distribution, self.version)


class TensorOnlyDataset(Dataset[torch.Tensor]):
    """Return unwrapped tensors for the plugin Gaussian batch contract."""

    def __init__(self, values: torch.Tensor) -> None:
        self.values = values

    def __len__(self) -> int:
        """Return the synthetic sample count."""

        return self.values.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return one clean synthetic state."""

        return self.values[index]


def gaussian_process() -> DiscreteGaussianProcess:
    """Build a small model-free process for vertical tests."""

    return DiscreteGaussianProcess(
        {
            "name": "linear_beta",
            "params": {"num_timesteps": 4},
        }
    )


def test_gaussian_code_is_physically_grouped_by_layer_and_family() -> None:
    """Keep family implementations out of the generic layer roots."""

    source_root = REPOSITORY_ROOT / "src" / "stochaflow"
    expected_packages = (
        source_root / "families" / "gaussian",
        source_root / "processes" / "gaussian",
        source_root / "sampling" / "gaussian",
        source_root / "training" / "gaussian",
    )
    assert all(path.is_dir() for path in expected_packages)
    assert not (source_root / "training" / "gaussian_loss.py").exists()
    assert not (source_root / "training" / "gaussian_weighting.py").exists()
    assert not (source_root / "training" / "class_conditional_gaussian.py").exists()
    assert not (source_root / "sampling" / "ddpm.py").exists()
    assert not (source_root / "sampling" / "ddim.py").exists()
    assert not (source_root / "processes" / "discrete_gaussian.py").exists()


def test_gaussian_training_does_not_depend_on_sampling_or_policy_registry() -> None:
    training_root = (
        REPOSITORY_ROOT / "src" / "stochaflow" / "training" / "gaussian"
    )
    violations: list[str] = []
    source = ""
    for path in training_root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        source += text
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.startswith("stochaflow.sampling")
            ):
                violations.append(f"{path.name}:{node.lineno}")
            if isinstance(node, ast.Import) and any(
                alias.name.startswith("stochaflow.sampling")
                for alias in node.names
            ):
                violations.append(f"{path.name}:{node.lineno}")

    assert not violations
    assert "GaussianLossComposer" not in source
    assert "GaussianSimpleLossWeighting" not in source
    assert "register_gaussian_simple_loss_weighting" not in source


def test_gaussian_training_facade_keeps_strategy_internals_private() -> None:
    assert {
        "GaussianLossComputation",
        "gaussian_signal_to_noise_ratio",
        "learned_range_log_variance",
        "parse_gaussian_variance",
        "validate_gaussian_model_output_layout",
    }.isdisjoint(gaussian_training.__all__)


def test_namespaced_training_builder_composes_custom_strategy() -> None:
    process = gaussian_process()
    model = PluginGaussianDenoiser()
    plan = build_training_plan(
        ComponentConfig(
            name=PLUGIN_TRAINING_NAME,
            params={"scale": 0.25},
        ),
        primary_model=model,
        process=process,
        objective=MSEObjective(),
        model_factory=lambda config: (_ for _ in ()).throw(
            AssertionError(f"unexpected model construction: {config}")
        ),
        objective_factory=lambda config: (_ for _ in ()).throw(
            AssertionError(f"unexpected objective construction: {config}")
        ),
    )
    batch = torch.full((2, 1, 2, 2), 0.5)

    training = validate_train_step_output(plan.strategy.training_step(batch))
    evaluation = validate_train_step_output(plan.strategy.evaluation_step(batch))

    assert isinstance(plan.strategy, PluginScaledGaussianStrategy)
    for output in (training, evaluation):
        assert torch.isfinite(output.loss)
        assert output.loss_aggregation_weight == 2
        assert torch.equal(
            output.diagnostics["timestep_loss_weight"],
            torch.full((2,), 0.25),
        )


def test_custom_training_builder_config_survives_checkpoint_roundtrip(
    tmp_path: Path,
) -> None:
    raw_config: dict[str, Any] = {
        "experiment": {
            "name": "gaussian-strategy-ocp",
            "output_dir": str(tmp_path / "outputs"),
        },
        "extensions": {"plugins": [PLUGIN_NAME]},
        "data": {"name": "tests.synthetic", "params": {}},
        "model": {"name": PLUGIN_MODEL_NAME, "params": {}},
        "process": {
            "name": "discrete_gaussian",
            "params": {
                "schedule": {
                    "name": "linear_beta",
                    "params": {"num_timesteps": 4},
                }
            },
        },
        "training": {
            "name": PLUGIN_TRAINING_NAME,
            "params": {"scale": 0.25},
        },
        "objective": {"name": "mse", "params": {"reduction": "mean"}},
    }
    resolved = load_config_dict(raw_config)
    checkpoint = CheckpointManager(torch.nn.Linear(1, 1)).save(
        tmp_path / "custom-strategy.pt",
        config=resolved.to_dict(),
        metadata={},
    )

    payload = CheckpointManager.load_payload(checkpoint, map_location="cpu")
    checkpoint_config = payload.get("config")
    assert isinstance(checkpoint_config, dict)
    restored = load_config_dict(checkpoint_config)

    assert restored.training.name == PLUGIN_TRAINING_NAME
    assert restored.training.params == {"scale": 0.25}


def test_installed_custom_strategy_survives_strict_cli_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise YAML, plugin activation, checkpoints, and strict resume."""

    plugins._reset_extension_activation_state_for_testing()
    installed = [InstalledPluginEntryPoint()]

    def discover(*, group: str) -> tuple[InstalledPluginEntryPoint, ...]:
        assert group == "stochaflow.extensions"
        return tuple(installed)

    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    monkeypatch.setattr(plugins.metadata, "entry_points", discover)
    clean_samples = torch.full((4, 1, 2, 2), 0.25)
    dataset = TensorOnlyDataset(clean_samples)
    loaders = DataLoaders(
        train=DataLoader(dataset, batch_size=2),
        validation=DataLoader(dataset, batch_size=2),
        test=DataLoader(dataset, batch_size=2),
    )
    monkeypatch.setattr(
        experiment_runner,
        "build_data_loaders",
        lambda *args, **kwargs: loaders,
    )
    initial_root = tmp_path / "initial"
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {
                    "name": "gaussian-strategy-lifecycle",
                    "seed": 17,
                    "output_dir": str(initial_root),
                },
                "extensions": {"plugins": [PLUGIN_NAME]},
                "data": {"name": "tests.synthetic", "params": {}},
                "model": {"name": PLUGIN_MODEL_NAME, "params": {}},
                "process": {
                    "name": "discrete_gaussian",
                    "params": {
                        "schedule": {
                            "name": "linear_beta",
                            "params": {"num_timesteps": 4},
                        }
                    },
                },
                "training": {
                    "name": PLUGIN_TRAINING_NAME,
                    "params": {"scale": 0.25},
                },
                "objective": {
                    "name": "mse",
                    "params": {"reduction": "mean"},
                },
                "metrics": [],
                "optimizer": {
                    "name": "torch.optim.SGD",
                    "params": {"lr": 0.01},
                },
                "ema": {"enabled": False},
                "diagnostics": [],
                "trainer": {
                    "num_epochs": 1,
                    "device": "cpu",
                    "show_progress": False,
                    "early_stopping": {
                        "enabled": False,
                        "monitor": "valid/loss",
                        "mode": "min",
                    },
                },
                "logging": {
                    "log_every": 1,
                    "backends": [
                        {"name": "local", "params": {"console": False}}
                    ],
                },
                "artifacts": {"checkpoint_every": 1},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    parser = build_argument_parser()

    try:
        first_args = parser.parse_args(
            [
                "train",
                "--config",
                str(config_path),
                "--epochs",
                "1",
                "--limit-batches",
                "1",
                "--limit-validation-batches",
                "1",
                "--limit-test-batches",
                "1",
                "--skip-final-sample",
                "--no-progress",
            ]
        )
        experiment_runner.run_experiment_from_args(first_args)
        first_run = next(path for path in initial_root.iterdir() if path.is_dir())
        first_checkpoint = first_run / "checkpoints/latest.pt"
        first_payload = CheckpointManager.load_payload(
            first_checkpoint,
            map_location="cpu",
        )
        assert first_payload.get("epoch") == 1
        first_config = first_payload.get("config")
        assert isinstance(first_config, dict)
        assert first_config["training"] == {
            "name": PLUGIN_TRAINING_NAME,
            "params": {"scale": 0.25},
        }

        installed.clear()
        missing_args = parser.parse_args(
            ["train", "--resume", str(first_run), "--epochs", "2"]
        )
        with pytest.raises(plugins.ExtensionDiscoveryError, match="not installed"):
            experiment_runner._resolve_training_inputs(missing_args)

        installed.append(
            InstalledPluginEntryPoint(value="fixtures.metrics_extension")
        )
        with pytest.raises(plugins.ExtensionIdentityError, match="identity differs"):
            experiment_runner._resolve_training_inputs(missing_args)

        installed[:] = [InstalledPluginEntryPoint()]
        resumed_root = tmp_path / "resumed"
        resume_args = parser.parse_args(
            [
                "train",
                "--resume",
                str(first_run),
                "--output-dir",
                str(resumed_root),
                "--epochs",
                "2",
                "--limit-batches",
                "1",
                "--limit-validation-batches",
                "1",
                "--limit-test-batches",
                "1",
                "--skip-final-sample",
                "--no-progress",
            ]
        )
        experiment_runner.run_experiment_from_args(resume_args)
        resumed_run = next(
            path for path in resumed_root.iterdir() if path.is_dir()
        )
        resumed_payload = CheckpointManager.load_payload(
            resumed_run / "checkpoints/latest.pt",
            map_location="cpu",
        )

        assert resumed_payload.get("epoch") == 2
        assert resumed_payload.get("global_step") == 2
        resumed_metadata = resumed_payload.get("metadata")
        assert isinstance(resumed_metadata, dict)
        assert resumed_metadata["extension_plugins"] == [
            {
                "name": PLUGIN_NAME,
                "distribution": PLUGIN_DISTRIBUTION,
                "version": PLUGIN_VERSION,
                "target": PLUGIN_TARGET,
            }
        ]
        resumed_config = resumed_payload.get("config")
        assert isinstance(resumed_config, dict)
        assert resumed_config["training"] == {
            "name": PLUGIN_TRAINING_NAME,
            "params": {"scale": 0.25},
        }
    finally:
        plugins._reset_extension_activation_state_for_testing()

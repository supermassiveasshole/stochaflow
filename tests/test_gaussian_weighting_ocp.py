"""Open-closed and lifecycle tests for Gaussian simple-loss weighting."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stochaflow.data import DataLoaders
from stochaflow.extensions import (
    GaussianSimpleLossContext,
    GaussianSimpleLossWeighting,
    register_gaussian_simple_loss_weighting,
)
from stochaflow.families.gaussian import PredictionType
from stochaflow.processes import DiscreteGaussianProcess
from stochaflow.scripts import experiment_runner
from stochaflow.scripts.cli import build_argument_parser
from stochaflow.training.builder import TrainingPlan, build_training_plan
from stochaflow.training.objectives import MSEObjective
from stochaflow.training.strategy import validate_train_step_output
from stochaflow.utils import plugins
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.config import ComponentConfig, load_config_dict

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_WEIGHTING_NAME = "tests.gaussian-weighting.scaled"
CUSTOM_WEIGHTING_SPEC = {
    "name": CUSTOM_WEIGHTING_NAME,
    "params": {"scale": 0.25},
}
PLUGIN_NAME = "gaussian-weighting-lifecycle"
PLUGIN_DISTRIBUTION = "gaussian-weighting-lifecycle-tests"
PLUGIN_VERSION = "1.2.3"
PLUGIN_TARGET = "fixtures.gaussian_weighting_extension"
PLUGIN_MODEL_NAME = "tests.gaussian-weighting-plugin.denoiser"
PLUGIN_WEIGHTING_NAME = "tests.gaussian-weighting-plugin.scaled-snr"


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


@register_gaussian_simple_loss_weighting(CUSTOM_WEIGHTING_NAME)
class ScaledExtensionGaussianWeighting(GaussianSimpleLossWeighting):
    """Independent extension policy supporting every Gaussian prediction type."""

    def __init__(self, scale: float) -> None:
        self.scale = float(scale)

    @property
    def requires_per_sample_loss(self) -> bool:
        """Require the explicit reducer because samples receive custom weights."""

        return True

    def validate_contract(self, *, prediction_type: PredictionType) -> None:
        """Accept all family prediction representations without core dispatch."""

    def sample_weights(
        self,
        context: GaussianSimpleLossContext,
    ) -> torch.Tensor:
        """Return deterministic weights using only the narrow family context."""

        return torch.full_like(context.signal_to_noise_ratio, self.scale)


class OCPUnconditionalDenoiser(nn.Module):
    """Parameter-bearing model satisfying the built-in unconditional signature."""

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
    ) -> torch.Tensor:
        """Return one batch-aligned x0 prediction."""

        del model_time
        return torch.zeros_like(state) + self.offset


class OCPClassConditionalDenoiser(nn.Module):
    """Parameter-bearing structural implementation of class conditioning."""

    def __init__(self) -> None:
        super().__init__()
        self.offset = nn.Parameter(torch.zeros(()))

    @property
    def num_classes(self) -> int:
        """Expose two real classes."""

        return 2

    @property
    def null_class_id(self) -> int:
        """Reserve the first identifier after the real classes."""

        return 2

    def predict_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return one batch-aligned x0 prediction."""

        del model_time, class_labels
        return torch.zeros_like(state) + self.offset


class TensorOnlyDataset(Dataset[torch.Tensor]):
    """Return unwrapped tensors for the built-in Gaussian batch contract."""

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


def reject_model_factory(config: ComponentConfig) -> nn.Module:
    """Prove the built-in Gaussian builders do not construct hidden models."""

    del config
    raise AssertionError("Gaussian builder unexpectedly constructed a model")


def reject_objective_factory(config: ComponentConfig) -> nn.Module:
    """Prove the built-in Gaussian builders preserve the injected Objective."""

    del config
    raise AssertionError("Gaussian builder unexpectedly constructed an Objective")


def build_gaussian_plan(
    *,
    builder_name: str,
    model: nn.Module,
    process: DiscreteGaussianProcess,
    training_params: dict[str, Any] | None = None,
) -> TrainingPlan:
    """Build through the registered built-in TrainingBuilder boundary."""

    params: dict[str, Any] = {
        "prediction_type": "x0",
        "loss_weighting": CUSTOM_WEIGHTING_SPEC,
    }
    if training_params is not None:
        params.update(training_params)
    return build_training_plan(
        ComponentConfig(name=builder_name, params=params),
        primary_model=model,
        process=process,
        objective=MSEObjective(),
        model_factory=reject_model_factory,
        objective_factory=reject_objective_factory,
    )


def test_gaussian_training_core_has_no_sampling_or_p2_dispatch() -> None:
    core_paths = (
        REPOSITORY_ROOT / "src/stochaflow/training/gaussian.py",
        REPOSITORY_ROOT
        / "src/stochaflow/training/class_conditional_gaussian.py",
        REPOSITORY_ROOT / "src/stochaflow/training/gaussian_loss.py",
    )
    violations: list[str] = []
    for path in core_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                node.module == "stochaflow.sampling"
                or (
                    node.module is not None
                    and node.module.startswith("stochaflow.sampling.")
                )
            ):
                violations.append(f"{path.name}:{node.lineno}: sampling import")
            if isinstance(node, ast.Import) and any(
                alias.name == "stochaflow.sampling"
                or alias.name.startswith("stochaflow.sampling.")
                for alias in node.names
            ):
                violations.append(f"{path.name}:{node.lineno}: sampling import")
            if isinstance(node, ast.Constant) and node.value == "p2":
                violations.append(f"{path.name}:{node.lineno}: P2 name dispatch")

    assert not violations, "\n".join(violations)


def test_custom_policy_runs_both_unconditional_train_and_evaluation() -> None:
    process = gaussian_process()
    plan = build_gaussian_plan(
        builder_name="gaussian_denoising",
        model=OCPUnconditionalDenoiser(),
        process=process,
    )
    batch = torch.full((2, 1, 2, 2), 0.5)

    training = validate_train_step_output(plan.strategy.training_step(batch))
    evaluation = validate_train_step_output(plan.strategy.evaluation_step(batch))

    for output in (training, evaluation):
        assert torch.isfinite(output.loss)
        assert output.loss_aggregation_weight == 2
        assert torch.equal(
            output.diagnostics["timestep_loss_weight"],
            torch.full((2,), 0.25),
        )


def test_custom_policy_runs_both_conditional_train_and_evaluation() -> None:
    process = gaussian_process()
    plan = build_gaussian_plan(
        builder_name="class_conditional_gaussian_denoising",
        model=OCPClassConditionalDenoiser(),
        process=process,
        training_params={"condition_dropout": 0.0},
    )
    batch = (
        torch.full((2, 1, 2, 2), 0.5),
        {"class_label": torch.tensor([0, 1])},
    )

    training = validate_train_step_output(plan.strategy.training_step(batch))
    evaluation = validate_train_step_output(plan.strategy.evaluation_step(batch))

    for output in (training, evaluation):
        assert torch.isfinite(output.loss)
        assert output.loss_aggregation_weight == 2
        assert torch.equal(
            output.diagnostics["timestep_loss_weight"],
            torch.full((2,), 0.25),
        )


def test_custom_policy_config_survives_checkpoint_roundtrip(
    tmp_path: Path,
) -> None:
    raw_config: dict[str, Any] = {
        "experiment": {
            "name": "gaussian-weighting-ocp",
            "output_dir": str(tmp_path / "outputs"),
        },
        "extensions": {"plugins": ["tests-gaussian-weighting"]},
        "data": {"name": "tests.synthetic", "params": {}},
        "model": {"name": "tests.denoiser", "params": {}},
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
            "name": "gaussian_denoising",
            "params": {
                "prediction_type": "x0",
                "loss_weighting": CUSTOM_WEIGHTING_SPEC,
            },
        },
        "objective": {"name": "mse", "params": {"reduction": "mean"}},
    }
    resolved = load_config_dict(raw_config)
    serialized = resolved.to_dict()
    checkpoint = CheckpointManager(nn.Linear(1, 1)).save(
        tmp_path / "custom-weighting.pt",
        config=serialized,
        metadata={
            "extension_plugins": [
                {
                    "name": "tests-gaussian-weighting",
                    "distribution": "tests-gaussian-weighting",
                    "version": "1.0.0",
                    "target": "tests_gaussian_weighting.stochaflow_ext",
                }
            ]
        },
    )

    payload = CheckpointManager.load_payload(checkpoint, map_location="cpu")
    checkpoint_config = payload.get("config")
    assert checkpoint_config is not None
    restored = load_config_dict(cast(dict[str, Any], checkpoint_config))

    expected_spec = {
        "name": CUSTOM_WEIGHTING_NAME,
        "params": {"scale": 0.25},
    }
    assert restored.training.params["loss_weighting"] == expected_spec
    process = gaussian_process()
    plan = build_training_plan(
        restored.training,
        primary_model=OCPUnconditionalDenoiser(),
        process=process,
        objective=MSEObjective(),
        model_factory=reject_model_factory,
        objective_factory=reject_objective_factory,
    )
    output = validate_train_step_output(
        plan.strategy.evaluation_step(torch.ones(2, 1, 2, 2))
    )
    assert torch.isfinite(output.loss)
    assert torch.equal(
        output.diagnostics["timestep_loss_weight"],
        torch.full((2,), 0.25),
    )


def test_installed_custom_weighting_survives_strict_cli_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise YAML, plugin activation, checkpoints, and exact strict resume."""

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
                    "name": "gaussian-weighting-lifecycle",
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
                    "name": "gaussian_denoising",
                    "params": {
                        "prediction_type": "x0",
                        "loss_weighting": {
                            "name": PLUGIN_WEIGHTING_NAME,
                            "params": {"scale": 0.25},
                        },
                    },
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
        assert first_config["training"]["params"][
            "loss_weighting"
        ] == {
            "name": PLUGIN_WEIGHTING_NAME,
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
        assert resumed_config["training"]["params"][
            "loss_weighting"
        ] == {
            "name": PLUGIN_WEIGHTING_NAME,
            "params": {"scale": 0.25},
        }
    finally:
        plugins._reset_extension_activation_state_for_testing()

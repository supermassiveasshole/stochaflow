"""Tests for config-driven checkpoint sampling."""

from argparse import Namespace
from pathlib import Path

import pytest
import torch
import yaml

from stochaflow.sampling import runtime
from stochaflow.scripts import cli
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.config import ComponentConfig, load_config_dict
from stochaflow.utils.factory import build_diffusion, build_model, build_noise_schedule
from stochaflow.utils.registry import REGISTRIES


def _image_data_config(*, image_size: int = 8, channels: int = 1) -> dict:
    return {
        "name": "image",
        "params": {
            "source": {
                "kind": "torchvision",
                "dataset": "MNIST",
                "root": "./data",
                "download": False,
            },
            "image": {
                "size": [image_size, image_size],
                "channels": channels,
                "normalize": True,
            },
            "loader": {
                "batch_size": 2,
                "num_workers": 0,
                "shuffle": True,
                "drop_last": True,
                "pin_memory": False,
                "persistent_workers": False,
                "steps_per_epoch": "auto",
            },
            "partition": {"mode": "none"},
        },
    }


def _raw_config(*, ema: bool = False, trajectory: bool = False) -> dict:
    return {
        "experiment": {"name": "tiny", "seed": 7},
        "extensions": {"modules": []},
        "data": _image_data_config(),
        "model": {
            "name": "unet",
            "params": {
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
                "channel_multipliers": [1],
                "num_res_blocks": 1,
                "time_embedding_dim": 8,
                "dropout": 0.0,
            },
        },
        "diffusion": {
            "name": "ddpm",
            "noise_schedule": {
                "name": "linear_beta",
                "params": {"num_timesteps": 2},
            },
        },
        "objective": {"name": "ddpm_epsilon", "params": {}},
        "ema": {"enabled": ema, "use_for_sampling": ema},
        "sampling": {
            "shape": [1, 8, 8],
            "num_samples": 3,
            "batch_size": 2,
            "writers": [
                {"name": "tensor", "params": {}},
                {
                    "name": "image",
                    "params": {
                        "grid_nrow": 2,
                        "gif_fps": 4,
                        "denormalize": True,
                    },
                },
            ],
            "debug": {
                "trajectory": {
                    "enabled": trajectory,
                    "params": {"state_interval": 1} if trajectory else {},
                }
            },
        },
    }


def _save_checkpoint(path: Path, *, ema: bool = False, trajectory: bool = False):
    config = load_config_dict(_raw_config(ema=ema, trajectory=trajectory))
    model = build_model(config.model)
    schedule = build_noise_schedule(config.diffusion.noise_schedule)
    diffusion = build_diffusion(
        config.diffusion.name,
        model=model,
        noise_schedule=schedule,
        params=config.diffusion.params,
    )
    tracker = ExponentialMovingAverage(diffusion) if ema else None
    if tracker is not None:
        for shadow in tracker.shadow_params.values():
            shadow.fill_(0.25)
    CheckpointManager(
        model=diffusion,
        denoiser=model,
        ema=tracker,
    ).save(path, config=config.to_dict())
    return config


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["steps=10", "eta=0.5", "enabled=true"], {"steps": 10, "eta": 0.5, "enabled": True}),
        (["items=[1, 2]", "text=a=b"], {"items": [1, 2], "text": "a=b"}),
        (["value=1", "value=2"], {"value": 2}),
        (["value=null"], {"value": None}),
    ],
)
def test_parse_sampler_params(values, expected) -> None:
    assert runtime.parse_sampler_params(values) == expected


@pytest.mark.parametrize("value", ["missing", "bad-name=1", "nested={a: 1}"])
def test_parse_sampler_params_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError):
        runtime.parse_sampler_params([value])


def test_checkpoint_only_sampler_override_discards_old_params(tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    _save_checkpoint(checkpoint)

    resolved = runtime.resolve_sampling_inputs(
        config_path=None,
        checkpoint=checkpoint,
        sampler_name="ddim",
        sampler_params={"num_inference_steps": 1, "eta": 0.0},
    )

    assert resolved.sampler.name == "ddim"
    assert resolved.sampler.params == {"num_inference_steps": 1, "eta": 0.0}


def test_checkpoint_only_sampling_loads_custom_modules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "checkpoint_sampling_extension.py"
    module_path.write_text(
        """
from pathlib import Path

from stochaflow.sampling import SamplingArtifactWriter
from stochaflow.utils.registry import REGISTRIES


@REGISTRIES.sampling_artifact_writers.register("checkpoint_sampling_writer")
class CheckpointSamplingWriter(SamplingArtifactWriter):
    def write(self, context):
        path = context.output_dir / "custom.txt"
        path.write_text("ok")
        return {"custom": Path(path)}
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    checkpoint = tmp_path / "best.pt"
    _save_checkpoint(checkpoint)
    payload = CheckpointManager.load_payload(checkpoint)
    checkpoint_config = payload.get("config")
    assert isinstance(checkpoint_config, dict)
    checkpoint_config["extensions"]["modules"] = [
        "checkpoint_sampling_extension"
    ]
    checkpoint_config["sampling"]["writers"] = [
        {"name": "checkpoint_sampling_writer", "params": {}}
    ]
    torch.save(payload, checkpoint)

    resolved = runtime.resolve_sampling_inputs(
        config_path=None,
        checkpoint=checkpoint,
    )

    assert resolved.config.sampling.writers[0].name == "checkpoint_sampling_writer"
    assert resolved.config.extensions.modules == [
        "checkpoint_sampling_extension"
    ]
    assert (
        REGISTRIES.sampling_artifact_writers.resolve(
            "checkpoint_sampling_writer"
        ).__name__
        == "CheckpointSamplingWriter"
    )


def test_checkpoint_config_persists_global_extensions(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    _save_checkpoint(checkpoint)

    payload = CheckpointManager.load_payload(checkpoint)
    checkpoint_config = payload.get("config")

    assert isinstance(checkpoint_config, dict)
    assert payload.get("format_version") == 4
    assert checkpoint_config["extensions"] == {"modules": []}
    assert checkpoint_config["data"] == _image_data_config()
    assert "modules" not in checkpoint_config["data"]


def test_config_only_finds_best_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "run" / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    config = _save_checkpoint(checkpoint)
    config.experiment.output_dir = str(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")

    resolved = runtime.resolve_sampling_inputs(
        config_path=config_path,
        checkpoint=None,
    )

    assert resolved.checkpoint_path == checkpoint


def test_external_config_overrides_sampling_section(tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    config = _save_checkpoint(checkpoint)
    config.experiment.seed = 99
    config.data = ComponentConfig(name="external_data", params={"anything": True})
    config.sampling.sampler = ComponentConfig(
        name="ddim",
        params={"num_inference_steps": 1, "eta": 0.0},
    )
    config.sampling.num_samples = 5
    config.sampling.shape = [1, 12, 12]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")

    resolved = runtime.resolve_sampling_inputs(
        config_path=config_path,
        checkpoint=checkpoint,
    )

    assert resolved.sampler.name == "ddim"
    assert resolved.config.sampling.num_samples == 5
    assert resolved.config.experiment.seed == 7
    assert runtime.sample_shape(resolved.config, 1) == (1, 1, 12, 12)
    assert resolved.config.data.name == "image"


def test_external_config_rejects_incompatible_model(tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    config = _save_checkpoint(checkpoint)
    config.model.params["base_channels"] = 16
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")

    with pytest.raises(ValueError, match="model"):
        runtime.resolve_sampling_inputs(
            config_path=config_path,
            checkpoint=checkpoint,
        )


def test_same_sampler_override_preserves_config_params(tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    config = _save_checkpoint(checkpoint)
    config.sampling.sampler = ComponentConfig(
        name="ddpm",
        params={"clip_denoised": False},
    )
    payload = CheckpointManager.load_payload(checkpoint)
    payload["config"] = config.to_dict()
    torch.save(payload, checkpoint)

    resolved = runtime.resolve_sampling_inputs(
        config_path=None,
        checkpoint=checkpoint,
        sampler_name="ddpm",
        sampler_params={"clip_denoised": True},
    )

    assert resolved.sampler.params == {"clip_denoised": True}


def test_checkpoint_contains_raw_and_ema_denoiser_weights(tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    _save_checkpoint(checkpoint, ema=True)

    payload = CheckpointManager.load_payload(checkpoint)

    assert "denoiser_state_dict" in payload
    assert "ema_denoiser_state_dict" in payload
    first = next(iter(payload["ema_denoiser_state_dict"].values()))
    assert torch.allclose(first, torch.full_like(first, 0.25))


def test_run_sampling_writes_samples_trajectory_gif_and_manifest(tmp_path) -> None:
    checkpoint = tmp_path / "run" / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    _save_checkpoint(checkpoint, trajectory=True)
    output_dir = tmp_path / "samples"

    result = runtime.run_sampling(
        checkpoint=checkpoint,
        output_dir=output_dir,
        device_name="cpu",
    )

    assert torch.load(output_dir / "samples.pt", weights_only=False).shape == (
        3,
        1,
        8,
        8,
    )
    for name in (
        "samples.png",
        "trajectory.pt",
        "trajectory.png",
        "trajectory.gif",
        "resolved_sampling.yaml",
    ):
        assert (output_dir / name).is_file()
    manifest = yaml.safe_load((output_dir / "resolved_sampling.yaml").read_text())
    assert manifest["sampler"]["name"] == "ddpm"
    assert manifest["seed"] == 7
    assert result.artifacts["trajectory_gif"] == output_dir / "trajectory.gif"


def test_old_checkpoint_is_rejected_for_sampling(tmp_path) -> None:
    checkpoint = tmp_path / "old.pt"
    torch.save(
        {
            "format_version": 3,
            "model_state_dict": {},
            "config": _raw_config(),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="expected version 4"):
        runtime.run_sampling(checkpoint=checkpoint, device_name="cpu")


def test_cli_requires_config_or_checkpoint() -> None:
    with pytest.raises(SystemExit):
        cli.main(["sample"])


def test_cli_parses_sampler_switch(monkeypatch, tmp_path) -> None:
    observed = {}
    monkeypatch.setattr(cli, "run_sampling", lambda **kwargs: observed.update(kwargs) or Namespace(
        checkpoint_path=tmp_path / "best.pt",
        sampler_name="ddim",
        device="cpu",
        seed=1,
        used_ema=False,
        output_dir=tmp_path,
        artifacts={},
    ))

    cli.main(
        [
            "sample",
            "--checkpoint",
            str(tmp_path / "best.pt"),
            "--sampler",
            "ddim",
            "--sampler-param",
            "eta=0.0",
        ]
    )

    assert observed["sampler_name"] == "ddim"
    assert observed["sampler_param_values"] == ["eta=0.0"]

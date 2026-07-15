"""Tests for DDPM checkpoint sampling helpers."""

import os
from argparse import Namespace

import torch

from stochaflow.scripts import sample_ddpm
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.utils.checkpoint import CheckpointManager
from stochaflow.utils.config import load_config_dict
from stochaflow.utils.factory import (
    build_diffusion,
    build_model,
    build_noise_schedule,
)


def _image_data_config(*, image_size: int, channels: int) -> dict:
    return {
        "modules": [],
        "datasets": [
            {
                "id": "mnist",
                "factory": "mnist",
                "params": {"root": "./data", "download": False},
                "splits": {"train": "train"},
            }
        ],
        "image": {"channels": channels, "normalize": True},
        "batching": {
            "buckets": [
                {
                    "name": "sample",
                    "height": image_size,
                    "width": image_size,
                }
            ],
            "sample_bucket": "sample",
            "dynamic_batch_size": True,
            "steps_per_epoch": "auto",
        },
        "dataloader": {
            "batch_size": 4,
            "num_workers": 0,
            "shuffle": True,
            "drop_last": True,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "splits": {"mode": "none"},
    }


def test_load_sampling_config_reads_checkpoint_config(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    config = {
        "experiment": {"name": "ddpm_mnist"},
        "data": _image_data_config(image_size=32, channels=1),
        "model": {
            "name": "unet",
            "params": {
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 32,
                "channel_multipliers": [1],
                "num_res_blocks": 1,
                "time_embedding_dim": 32,
                "dropout": 0.0,
            },
        },
        "diffusion": {
            "name": "ddpm",
            "noise_schedule": {
                "name": "linear_beta",
                "params": {"num_timesteps": 10},
            },
        },
        "objective": {"name": "ddpm_epsilon", "params": {}},
    }
    torch.save(
        {
            "model_state_dict": {},
            "config": config,
        },
        checkpoint_path,
    )

    loaded = sample_ddpm._load_sampling_config(checkpoint_path)

    assert loaded.experiment.name == "ddpm_mnist"
    assert loaded.data.datasets[0].factory == "mnist"
    assert loaded.diffusion.name == "ddpm"


def test_resolve_sampling_checkpoint_defaults_to_newest_best(tmp_path) -> None:
    flat = tmp_path / "checkpoints" / "best.pt"
    older = tmp_path / "older" / "checkpoints" / "best.pt"
    newer = tmp_path / "newer" / "checkpoints" / "best.pt"
    flat.parent.mkdir(parents=True)
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    torch.save({"model_state_dict": {}}, flat)
    torch.save({"model_state_dict": {}}, older)
    torch.save({"model_state_dict": {}}, newer)
    os.utime(flat, (1, 1))
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    resolved = sample_ddpm._resolve_sampling_checkpoint(
        Namespace(checkpoint=None, search_dir=tmp_path)
    )

    assert resolved == newer


def test_resolve_sampling_checkpoint_accepts_run_directory(tmp_path) -> None:
    checkpoint_path = tmp_path / "run" / "checkpoints" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save({"model_state_dict": {}}, checkpoint_path)

    resolved = sample_ddpm._resolve_sampling_checkpoint(
        Namespace(checkpoint=tmp_path / "run", search_dir=tmp_path)
    )

    assert resolved == checkpoint_path


def test_build_checkpointed_ddpm_applies_ema_weights(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    raw_config = {
        "experiment": {"name": "ddpm_tiny"},
        "data": _image_data_config(image_size=8, channels=1),
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
                "params": {"num_timesteps": 10},
            },
        },
        "objective": {"name": "ddpm_epsilon", "params": {}},
        "ema": {"enabled": True, "decay": 0.9999, "use_for_sampling": True},
    }
    config = load_config_dict(raw_config)
    model = build_model(config.model)
    noise_schedule = build_noise_schedule(config.diffusion.noise_schedule)
    diffusion = build_diffusion(
        config.diffusion.name,
        model=model,
        noise_schedule=noise_schedule,
        params=config.diffusion.params,
    )
    ema = ExponentialMovingAverage(diffusion)
    for shadow in ema.shadow_params.values():
        shadow.fill_(0.25)
    CheckpointManager(model=diffusion, ema=ema).save(
        checkpoint_path,
        config=config.to_dict(),
    )

    diffusion, _, used_ema = sample_ddpm._build_checkpointed_ddpm(
        config,
        checkpoint_path,
        device=torch.device("cpu"),
    )

    assert used_ema
    first_parameter = next(diffusion.parameters())
    assert torch.allclose(first_parameter, torch.full_like(first_parameter, 0.25))

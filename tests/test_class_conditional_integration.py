"""Non-data integration tests for registered conditional training stacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from stochaflow.training.gaussian import (
    ClassConditionalGaussianDenoisingTrainingStrategy,
    ClassConditionalP2GaussianDenoisingTrainingStrategy,
)
from stochaflow.utils.checkpoint import (
    CheckpointManager,
    parse_rng_state,
    restore_rng_state,
)
from stochaflow.utils.config import load_config_dict
from stochaflow.utils.factory import build_training_components
from stochaflow.utils.seed import set_seed


def _raw_config(
    output_dir: Path,
    *,
    model_name: str,
    model_params: dict[str, Any],
    training_name: str,
    training_params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": {
            "name": f"conditional-{model_name}",
            "output_dir": str(output_dir),
            "seed": 123,
        },
        "extensions": {"plugins": []},
        "data": {"name": "not-built-in-this-test", "params": {}},
        "model": {"name": model_name, "params": model_params},
        "process": {
            "name": "discrete_gaussian",
            "params": {
                "schedule": {
                    "name": "linear_beta",
                    "params": {
                        "num_timesteps": 4,
                        "beta_start": 0.0001,
                        "beta_end": 0.02,
                    },
                }
            },
        },
        "training": {
            "name": training_name,
            "params": training_params,
        },
        "objective": {"name": "mse", "params": {"reduction": "mean"}},
        "optimizer": {
            "name": "torch.optim.AdamW",
            "params": {"lr": 0.001, "weight_decay": 0.0},
        },
        "lr_scheduler": {
            "name": "warmup_cosine",
            "interval": "step",
            "params": {
                "warmup_steps": 1,
                "total_steps": 4,
                "min_lr_ratio": 0.1,
            },
        },
        "ema": {
            "enabled": True,
            "decay": 0.9,
            "update_after_step": 0,
            "update_every": 1,
        },
        "trainer": {
            "num_epochs": 2,
            "device": "cpu",
            "precision": "fp32",
            "accumulate_grad_batches": 2,
            "show_progress": False,
        },
        "logging": {
            "log_every": 1,
            "backends": [{"name": "local", "params": {}}],
        },
    }


def _microbatches() -> list[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    return [
        (
            torch.linspace(-1.0, 1.0, steps=2 * 8 * 8).reshape(2, 1, 8, 8),
            {"class_label": torch.tensor([0, 1])},
        ),
        (
            torch.linspace(1.0, -1.0, steps=2 * 8 * 8).reshape(2, 1, 8, 8),
            {"class_label": torch.tensor([2, 0])},
        ),
    ]


@pytest.mark.parametrize(
    ("model_name", "model_params"),
    [
        (
            "adm_unet",
            {
                "input_size": 8,
                "in_channels": 1,
                "out_channels": 1,
                "base_channels": 8,
                "channel_multipliers": [1, 2],
                "num_res_blocks": 1,
                "attention_resolutions": [4],
                "attention_head_channels": 8,
                "num_classes": 3,
                "dropout": 0.0,
            },
        ),
        (
            "dit",
            {
                "input_size": [8, 8],
                "patch_size": 4,
                "in_channels": 1,
                "out_channels": 1,
                "hidden_size": 32,
                "depth": 2,
                "num_heads": 4,
                "mlp_ratio": 2.0,
                "num_classes": 3,
            },
        ),
    ],
)
@pytest.mark.parametrize(
    ("training_name", "training_params", "strategy_type"),
    [
        (
            "class_conditional_gaussian_denoising",
            {"prediction_type": "v", "condition_dropout": 0.5},
            ClassConditionalGaussianDenoisingTrainingStrategy,
        ),
        (
            "class_conditional_p2_gaussian_denoising",
            {
                "condition_dropout": 0.5,
                "k": 1.0,
                "gamma": 1.0,
                "variance": {"mode": "fixed"},
            },
            ClassConditionalP2GaussianDenoisingTrainingStrategy,
        ),
    ],
)
def test_registered_conditional_stack_resumes_at_epoch_boundary(
    tmp_path: Path,
    model_name: str,
    model_params: dict[str, Any],
    training_name: str,
    training_params: dict[str, Any],
    strategy_type: type[ClassConditionalGaussianDenoisingTrainingStrategy],
) -> None:
    config = load_config_dict(
        _raw_config(
            tmp_path / "uninterrupted",
            model_name=model_name,
            model_params=model_params,
            training_name=training_name,
            training_params=training_params,
        )
    )
    set_seed(config.experiment.seed)
    uninterrupted = build_training_components(config)
    assert isinstance(
        uninterrupted.plan.strategy,
        strategy_type,
    )
    uninterrupted.trainer.train_epoch(
        _microbatches(),
        epoch_index=1,
        show_progress=False,
    )
    checkpoint = uninterrupted.checkpoint_manager.save(
        tmp_path / f"{model_name}-{training_name}.pt",
        epoch=1,
        global_step=uninterrupted.trainer.global_step,
        config=config.to_dict(),
    )
    uninterrupted.trainer.train_epoch(
        _microbatches(),
        epoch_index=2,
        show_progress=False,
    )
    expected = uninterrupted.checkpoint_manager.build_state()

    resumed_config = load_config_dict(
        _raw_config(
            tmp_path / "resumed",
            model_name=model_name,
            model_params=model_params,
            training_name=training_name,
            training_params=training_params,
        )
    )
    resumed = build_training_components(resumed_config)
    payload = CheckpointManager.load_payload(checkpoint, map_location="cpu")
    loaded = resumed.checkpoint_manager.restore_payload(
        payload,
        path=checkpoint,
    )
    assert loaded.global_step == 1
    resumed.trainer.global_step = loaded.global_step
    rng_state = payload.get("rng_state")
    assert rng_state is not None
    restore_rng_state(
        parse_rng_state(rng_state),
        restore_cuda=False,
        restore_mps=False,
    )
    resumed.trainer.train_epoch(
        _microbatches(),
        epoch_index=2,
        show_progress=False,
    )
    actual = resumed.checkpoint_manager.build_state()

    assert resumed.trainer.global_step == uninterrupted.trainer.global_step == 2
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
        "ema_state_dict",
    ):
        torch.testing.assert_close(
            actual.get(key),
            expected.get(key),
            rtol=0.0,
            atol=0.0,
        )
    assert actual.get("precision_kind") == expected.get("precision_kind") == "fp32"
    assert actual.get("inference_asset_descriptors") == {}

    uninterrupted.logger.close()
    resumed.logger.close()

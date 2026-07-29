"""Class-conditional diagnostic capability and lifecycle tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
import torch
import yaml
from torch import nn

from stochaflow.processes import DiscreteGaussianProcess
from stochaflow.training import (
    ClassConditionalGaussianDenoisingTrainingStrategy,
    ManagedTrainingModule,
    MSEObjective,
    TrainingStrategy,
    TrainStepOutput,
)
from stochaflow.training.diagnostics import (
    ClassConditionalDiffusionQualityDiagnostic,
    FitStartEvent,
    TrainBatchEndEvent,
    TrainEpochEndEvent,
)

from .helpers import (
    RecordingLogger,
    fit_event,
    gaussian_system,
    provider_config,
    trainer,
)


class RecordingConditionalDenoiser(nn.Module):
    """Independent conditional model recording every diagnostic label call."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.label_calls: list[torch.Tensor] = []

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def null_class_id(self) -> int:
        return 2

    def predict_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        del model_time
        self.label_calls.append(class_labels.detach().cpu().clone())
        values = class_labels.to(state.dtype).reshape(
            class_labels.shape[0],
            *([1] * (state.ndim - 1)),
        )
        return torch.zeros_like(state) + values / 10.0 + self.bias


class CustomNullConditionalStrategy(TrainingStrategy):
    """Independent diagnostic capability with a non-derived null label."""

    def __init__(
        self,
        model: RecordingConditionalDenoiser,
        *,
        null_class_id: Any,
    ) -> None:
        self.model = model
        self._null_class_id = null_class_id

    @property
    def prediction_type(self) -> Literal["v"]:
        return "v"

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def null_class_id(self) -> Any:
        return self._null_class_id

    def predict_class_conditional_gaussian_model(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.predict_class_conditioned(
            state,
            model_time,
            class_labels,
        )

    def training_step(self, batch: Any) -> TrainStepOutput:
        del batch
        return TrainStepOutput(loss=self.model.bias.square())


def _custom_null_runtime(
    *,
    null_class_id: Any = 7,
) -> tuple[SimpleNamespace, RecordingConditionalDenoiser]:
    process = DiscreteGaussianProcess(
        {
            "name": "linear_beta",
            "params": {
                "num_timesteps": 2,
                "beta_start": 0.0001,
                "beta_end": 0.02,
            },
        }
    )
    model = RecordingConditionalDenoiser()
    strategy = CustomNullConditionalStrategy(
        model,
        null_class_id=null_class_id,
    )
    runtime = SimpleNamespace(
        model=model,
        process=process,
        strategy=strategy,
        managed_modules={
            "primary_model": ManagedTrainingModule(model),
            "process": ManagedTrainingModule(process),
        },
        ema_model=model,
        device=torch.device("cpu"),
        ema=None,
        global_step=1,
    )
    return runtime, model


def _conditional_runtime() -> tuple[SimpleNamespace, RecordingConditionalDenoiser]:
    process = DiscreteGaussianProcess(
        {
            "name": "linear_beta",
            "params": {
                "num_timesteps": 2,
                "beta_start": 0.0001,
                "beta_end": 0.02,
            },
        }
    )
    model = RecordingConditionalDenoiser()
    objective = MSEObjective()
    strategy = ClassConditionalGaussianDenoisingTrainingStrategy(
        model,
        process,
        objective,
        prediction_type="v",
        condition_dropout=1.0,
    )
    runtime = SimpleNamespace(
        model=model,
        process=process,
        strategy=strategy,
        managed_modules={
            "primary_model": ManagedTrainingModule(model),
            "process": ManagedTrainingModule(process),
            "objective": ManagedTrainingModule(objective),
        },
        ema_model=model,
        device=torch.device("cpu"),
        ema=None,
        global_step=1,
    )
    return runtime, model


def _diagnostic(
    tmp_path: Path,
    logger: RecordingLogger,
    **overrides: Any,
) -> ClassConditionalDiffusionQualityDiagnostic:
    params: dict[str, Any] = {
        "logger": logger,
        "output_dir": tmp_path,
        "conditions": [
            {"class_label": 0, "count": 1},
            {"class_label": 1, "count": 1},
        ],
        "guidance_scale": 2.0,
        "samplers": [
            {
                "id": "ddim_2",
                "name": "ddim",
                "params": {"num_inference_steps": 2, "eta": 0.0},
                "trajectory": {
                    "enabled": True,
                    "every_steps": 1,
                    "gif_fps": 4,
                },
            }
        ],
        "cadence": {"step_every": 1, "artifact_every_epochs": 1},
        "sampling": {
            "shape": [1, 4, 4],
            "sample_num": 2,
            "batch_size": 1,
            "seed": 123,
        },
        "providers": provider_config(),
        "use_ema": False,
        "failure_policy": "raise",
    }
    params.update(overrides)
    return ClassConditionalDiffusionQualityDiagnostic(**params)


def _events(runtime: SimpleNamespace):
    clean = torch.stack(
        (
            torch.full((1, 4, 4), -0.5),
            torch.full((1, 4, 4), 0.5),
        )
    )
    labels = torch.tensor([1, 0], dtype=torch.long)
    output = runtime.strategy.training_step(
        (clean, {"class_label": labels})
    )
    batch = TrainBatchEndEvent(
        trainer=runtime,
        batch=(clean, {"class_label": labels}),
        output=output,
        loss=float(output.loss.detach()),
        global_step=1,
        epoch_index=1,
    )
    epoch = TrainEpochEndEvent(
        trainer=runtime,
        epoch_index=1,
        metrics={},
    )
    return batch, epoch


def test_conditional_quality_preserves_original_labels_and_records_cfg(
    tmp_path: Path,
) -> None:
    logger = RecordingLogger()
    diagnostic = _diagnostic(tmp_path, logger)
    runtime, model = _conditional_runtime()
    batch, epoch = _events(runtime)
    model.label_calls.clear()

    diagnostic.on_fit_start(
        FitStartEvent(runtime, train_dataloader=[], validation_dataloader=None)
    )
    diagnostic.on_train_batch_end(batch)

    assert model.label_calls
    assert all(
        torch.equal(labels, torch.tensor([1, 0]))
        for labels in model.label_calls
    )
    model.label_calls.clear()

    diagnostic.on_train_epoch_end(epoch)

    assert any(torch.equal(labels, torch.tensor([1, 0])) for labels in model.label_calls)
    guided = [
        labels.tolist()
        for labels in model.label_calls
        if labels.numel() == 2 and 2 in labels.tolist()
    ]
    assert [0, 2] in guided
    assert [1, 2] in guided

    root = (
        tmp_path
        / "diagnostics"
        / "class_conditional_diffusion_quality"
        / "epoch_0001"
    )
    assert (root / "denoiser" / "reconstruction.png").is_file()
    assert (root / "ddim_2" / "samples.png").is_file()
    assert (root / "ddim_2" / "trajectory.gif").is_file()
    manifest = yaml.safe_load(
        (root / "manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest["conditioning"] == {
        "guidance_scale": 2.0,
        "allocations": [
            {"class_label": 0, "count": 1},
            {"class_label": 1, "count": 1},
        ],
    }
    evaluations = manifest["profiles"][0]["model_evaluations"]
    assert evaluations["forward_calls"] > 0
    assert evaluations["conditional_branches"] == evaluations["forward_calls"]
    assert evaluations["unconditional_branches"] == evaluations["forward_calls"]


def test_conditional_quality_cfg_uses_strategy_declared_null_class_id(
    tmp_path: Path,
) -> None:
    diagnostic = _diagnostic(
        tmp_path,
        RecordingLogger(),
        providers={
            "step_metrics": [],
            "sampler_metrics": [],
            "denoiser_artifacts": [],
            "sampler_artifacts": [
                {"name": "sample_grid", "params": {"nrow": 2}}
            ],
        },
    )
    runtime, model = _custom_null_runtime(null_class_id=7)

    diagnostic.on_fit_start(
        FitStartEvent(runtime, train_dataloader=[], validation_dataloader=None)
    )
    diagnostic.on_train_epoch_end(
        TrainEpochEndEvent(runtime, epoch_index=1, metrics={})
    )

    guided = [
        labels.tolist()
        for labels in model.label_calls
        if labels.numel() == 2 and 7 in labels.tolist()
    ]
    assert [0, 7] in guided
    assert [1, 7] in guided
    assert all(2 not in labels for labels in guided)


@pytest.mark.parametrize(
    ("null_class_id", "error", "message"),
    [
        (
            True,
            TypeError,
            "null_class_id must be an integer",
        ),
        (
            1,
            ValueError,
            "null_class_id must be outside the non-null class range",
        ),
    ],
)
def test_conditional_quality_validates_declared_null_class_id(
    tmp_path: Path,
    null_class_id: Any,
    error: type[Exception],
    message: str,
) -> None:
    diagnostic = _diagnostic(tmp_path, RecordingLogger())
    runtime, _ = _custom_null_runtime(null_class_id=null_class_id)

    with pytest.raises(error, match=message):
        diagnostic.on_fit_start(
            FitStartEvent(
                runtime,
                train_dataloader=[],
                validation_dataloader=None,
            )
        )


def test_conditional_quality_repeats_fixed_seed_samples(tmp_path: Path) -> None:
    diagnostic = _diagnostic(tmp_path, RecordingLogger())
    runtime, _ = _conditional_runtime()
    batch, _ = _events(runtime)
    diagnostic.on_fit_start(
        FitStartEvent(runtime, train_dataloader=[], validation_dataloader=None)
    )
    diagnostic.on_train_batch_end(batch)

    for epoch_index in (1, 2):
        diagnostic.on_train_epoch_end(
            TrainEpochEndEvent(runtime, epoch_index=epoch_index, metrics={})
        )

    root = tmp_path / "diagnostics" / "class_conditional_diffusion_quality"
    first = torch.load(
        root / "epoch_0001" / "ddim_2" / "samples.pt",
        weights_only=True,
    )
    second = torch.load(
        root / "epoch_0002" / "ddim_2" / "samples.pt",
        weights_only=True,
    )
    assert torch.equal(first, second)


def test_conditional_quality_warns_without_successful_batch_and_keeps_sampling(
    tmp_path: Path,
) -> None:
    logger = RecordingLogger()
    diagnostic = _diagnostic(
        tmp_path,
        logger,
        failure_policy="warn",
    )
    runtime, _ = _conditional_runtime()
    diagnostic.on_fit_start(
        FitStartEvent(runtime, train_dataloader=[], validation_dataloader=None)
    )

    diagnostic.on_train_epoch_end(
        TrainEpochEndEvent(runtime, epoch_index=1, metrics={})
    )

    root = (
        tmp_path
        / "diagnostics"
        / "class_conditional_diffusion_quality"
        / "epoch_0001"
    )
    assert (root / "ddim_2" / "samples.pt").is_file()
    manifest = yaml.safe_load(
        (root / "manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest["errors"] == [
        {
            "phase": "denoiser_artifact",
            "provider": "reconstruction_panel",
            "type": "RuntimeError",
            "message": (
                "conditional diagnostic has not captured a labeled batch"
            ),
        }
    ]
    assert manifest["profiles"][0]["model_evaluations"]["forward_calls"] > 0
    assert any(
        payload.get("diagnostics/system/error_count") == 1.0
        for _, payload in logger.metrics
    )


def test_conditional_quality_rejects_incompatible_runtime_and_config(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="condition counts"):
        _diagnostic(
            tmp_path,
            RecordingLogger(),
            conditions=[{"class_label": 0, "count": 1}],
        )
    with pytest.raises(ValueError, match="reference metrics are not supported"):
        _diagnostic(
            tmp_path,
            RecordingLogger(),
            reference={
                "enabled": True,
                "num_real": 2,
                "num_fake": 2,
                "batch_size": 1,
                "metrics": [{"name": "fid", "params": {}}],
            },
        )

    diagnostic = _diagnostic(tmp_path, RecordingLogger())
    unconditional = trainer(gaussian_system(num_timesteps=2))
    with pytest.raises(
        TypeError,
        match="ClassConditionalGaussianDiagnosticSemantics",
    ):
        diagnostic.on_fit_start(fit_event(unconditional))

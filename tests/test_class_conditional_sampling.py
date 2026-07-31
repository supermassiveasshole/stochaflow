"""Tests for built-in class-conditional Gaussian sampling and CFG."""

from typing import Any, cast

import pytest
import torch
from torch import nn

from stochaflow.processes import DiscreteGaussianProcess
from stochaflow.sampling import (
    InferenceModelProvider,
    PredictionType,
    SamplingBuilderContext,
)
from stochaflow.sampling.gaussian.class_conditional import (
    ClassConditionalDenoisingBuilder,
    ClassConditionalEvaluationCounts,
    ClassifierFreeGuidancePredictor,
)


class RecordingClassConditionalDenoiser(nn.Module):
    """Record structural capability calls without relying on a built-in model."""

    def __init__(self, *, num_classes: int = 2) -> None:
        super().__init__()
        self._num_classes = num_classes
        self.bias = nn.Parameter(torch.zeros(()))
        self.seen_labels: list[torch.Tensor] = []

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def null_class_id(self) -> int:
        return self.num_classes

    def predict_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        del model_time
        self.seen_labels.append(class_labels.detach().cpu().clone())
        label_shape = (class_labels.shape[0],) + (1,) * (state.ndim - 1)
        return (
            class_labels.to(device=state.device, dtype=state.dtype)
            .reshape(label_shape)
            .expand_as(state)
            + self.bias * 0.0
        )


class PrevalidatedRecordingClassConditionalDenoiser(
    RecordingClassConditionalDenoiser
):
    """Expose the optional no-repeat-validation prediction capability."""

    def __init__(self, *, num_classes: int = 2) -> None:
        super().__init__(num_classes=num_classes)
        self.prevalidated_calls = 0

    def predict_prevalidated_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        self.prevalidated_calls += 1
        return self.predict_class_conditioned(
            state,
            model_time,
            class_labels,
        )


class LearnedRangeClassConditionalDenoiser(RecordingClassConditionalDenoiser):
    """Return distinct prediction and variance heads for CFG assertions."""

    def predict_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        del model_time
        label_shape = (class_labels.shape[0],) + (1,) * (state.ndim - 1)
        labels = (
            class_labels.to(device=state.device, dtype=state.dtype)
            .reshape(label_shape)
            .expand_as(state)
        )
        return torch.cat((labels, labels + 10.0), dim=1)


class InvalidInferenceModel(nn.Module):
    """Model returned by a provider that lacks conditional capability."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state


class FixedModelProvider:
    """Select one deliberately malformed model for capability revalidation."""

    def __init__(self, model: nn.Module, label: str) -> None:
        self.model = model
        self.label = label

    def resolve(self, weights: str) -> tuple[nn.Module, str]:
        del weights
        return self.model, self.label


def _process() -> DiscreteGaussianProcess:
    return DiscreteGaussianProcess(
        {"name": "linear_beta", "params": {"num_timesteps": 2}}
    )


def _params(
    *,
    guidance_scale: object = 1.0,
    conditions: object = None,
    sampler: dict[str, Any] | None = None,
    weights: str = "raw",
    prediction_type: object = "epsilon",
) -> dict[str, Any]:
    return {
        "weights": weights,
        "prediction_type": prediction_type,
        "clip_denoised": True,
        "guidance_scale": guidance_scale,
        "conditions": (
            [
                {"class_label": 0, "count": 2},
                {"class_label": 1, "count": 3},
            ]
            if conditions is None
            else conditions
        ),
        "sampler": sampler
        or {"name": "ddim", "params": {"num_inference_steps": 1, "eta": 0.0}},
        "trajectory": {"enabled": True, "every_steps": 1},
    }


def _provider(
    instances: list[RecordingClassConditionalDenoiser],
    *,
    prefer_ema: bool = False,
) -> InferenceModelProvider:
    template = RecordingClassConditionalDenoiser()
    state = {
        name: value.detach().clone() for name, value in template.state_dict().items()
    }

    def factory() -> nn.Module:
        model = RecordingClassConditionalDenoiser()
        instances.append(model)
        return model

    return InferenceModelProvider(
        model_factory=factory,
        raw_state_dict=state,
        ema_state_dict=state,
        device=torch.device("cpu"),
        prefer_ema=prefer_ema,
    )


def _context(
    params: dict[str, Any],
    provider: InferenceModelProvider,
    *,
    num_samples: int = 5,
    batch_size: int = 3,
) -> SamplingBuilderContext:
    return SamplingBuilderContext(
        params=params,
        process=_process(),
        model_provider=provider,
        device=torch.device("cpu"),
        seed=17,
        shape=(1, 2, 2),
        num_samples=num_samples,
        batch_size=batch_size,
    )


@pytest.mark.parametrize(
    ("guidance_scale", "expected_labels", "conditional_count", "unconditional_count"),
    [
        (
            0.0,
            [torch.tensor([2, 2, 2]), torch.tensor([2, 2])],
            0,
            2,
        ),
        (
            1.0,
            [torch.tensor([0, 0, 1]), torch.tensor([1, 1])],
            2,
            0,
        ),
        (
            2.0,
            [
                torch.tensor([0, 0, 1, 2, 2, 2]),
                torch.tensor([1, 1, 2, 2]),
            ],
            2,
            2,
        ),
    ],
)
def test_builder_optimizes_cfg_and_preserves_cross_batch_label_order(
    guidance_scale: float,
    expected_labels: list[torch.Tensor],
    conditional_count: int,
    unconditional_count: int,
) -> None:
    instances: list[RecordingClassConditionalDenoiser] = []
    builder = ClassConditionalDenoisingBuilder(
        _context(_params(guidance_scale=guidance_scale), _provider(instances))
    )

    output = builder.run()

    assert len(output.batches) == 2
    assert [batch.samples.shape[0] for batch in output.batches] == [3, 2]
    assert all(batch.trajectory is not None for batch in output.batches)
    assert len(instances) == 1
    assert len(instances[0].seen_labels) == 2
    for actual, expected in zip(instances[0].seen_labels, expected_labels, strict=True):
        assert torch.equal(actual, expected)
    assert output.metadata["forward_call_count"] == 2
    assert output.metadata["conditional_branch_evaluation_count"] == conditional_count
    assert (
        output.metadata["unconditional_branch_evaluation_count"] == unconditional_count
    )
    assert output.metadata["conditions"] == [
        {"class_label": 0, "count": 2},
        {"class_label": 1, "count": 3},
    ]


@pytest.mark.parametrize(
    ("sampler", "expected_calls"),
    [
        ({"name": "ddpm", "params": {}}, 4),
        (
            {"name": "ddim", "params": {"num_inference_steps": 1, "eta": 0.0}},
            2,
        ),
    ],
)
def test_builder_reuses_registered_ddpm_and_ddim_samplers(
    sampler: dict[str, Any],
    expected_calls: int,
) -> None:
    instances: list[RecordingClassConditionalDenoiser] = []

    output = ClassConditionalDenoisingBuilder(
        _context(_params(guidance_scale=1.0, sampler=sampler), _provider(instances))
    ).run()

    assert output.metadata["forward_call_count"] == expected_calls
    assert output.metadata["sampler"]["name"] == sampler["name"]


@pytest.mark.parametrize("prediction_type", ["epsilon", "x0", "v", "score"])
def test_builder_supports_all_gaussian_prediction_types(
    prediction_type: PredictionType,
) -> None:
    instances: list[RecordingClassConditionalDenoiser] = []

    output = ClassConditionalDenoisingBuilder(
        _context(
            _params(prediction_type=prediction_type),
            _provider(instances),
        )
    ).run()

    assert output.metadata["prediction_type"] == prediction_type


@pytest.mark.parametrize(
    ("scale", "expected", "expected_labels"),
    [
        (0.0, torch.full((2, 1), 2.0), torch.tensor([2, 2])),
        (1.0, torch.tensor([[0.0], [1.0]]), torch.tensor([0, 1])),
        (
            2.0,
            torch.tensor([[-2.0], [0.0]]),
            torch.tensor([0, 1, 2, 2]),
        ),
    ],
)
def test_cfg_prediction_formula_uses_one_forward(
    scale: float,
    expected: torch.Tensor,
    expected_labels: torch.Tensor,
) -> None:
    model = RecordingClassConditionalDenoiser()
    counts = ClassConditionalEvaluationCounts()
    predict = ClassifierFreeGuidancePredictor(
        model,
        torch.tensor([0, 1]),
        guidance_scale=scale,
        counts=counts,
    )

    actual = predict(torch.zeros(2, 1), torch.zeros(2, dtype=torch.long))

    assert torch.equal(actual, expected)
    assert counts.forward_calls == 1
    assert len(model.seen_labels) == 1
    assert torch.equal(model.seen_labels[0], expected_labels)


@pytest.mark.parametrize(
    ("guidance_scale", "expected_mean", "expected_variance"),
    [
        (0.0, [2.0, 2.0], [12.0, 12.0]),
        (1.0, [0.0, 1.0], [10.0, 11.0]),
        (2.0, [-2.0, 0.0], [10.0, 11.0]),
    ],
)
def test_learned_range_cfg_guides_only_prediction_head(
    guidance_scale: float,
    expected_mean: list[float],
    expected_variance: list[float],
) -> None:
    model = LearnedRangeClassConditionalDenoiser()
    counts = ClassConditionalEvaluationCounts()
    predictor = ClassifierFreeGuidancePredictor(
        model,
        torch.tensor([0, 1]),
        guidance_scale=guidance_scale,
        variance_mode="learned_range",
        counts=counts,
    )

    output = predictor(
        torch.zeros(2, 1, 2, 2),
        torch.zeros(2, dtype=torch.long),
    )
    prediction, variance = output.chunk(2, dim=1)

    assert torch.equal(
        prediction[:, 0, 0, 0],
        torch.tensor(expected_mean),
    )
    assert torch.equal(
        variance[:, 0, 0, 0],
        torch.tensor(expected_variance),
    )


def test_cfg_predictor_uses_optional_prevalidated_model_path() -> None:
    model = PrevalidatedRecordingClassConditionalDenoiser()
    counts = ClassConditionalEvaluationCounts()
    predictor = ClassifierFreeGuidancePredictor(
        model,
        torch.tensor([0, 1]),
        guidance_scale=2.0,
        counts=counts,
    )

    predictor(
        torch.zeros(2, 1, 2, 2),
        torch.tensor([1, 1]),
    )

    assert model.prevalidated_calls == 1
    assert counts.forward_calls == 1


@pytest.mark.parametrize(
    ("guidance_scale", "error", "message"),
    [
        (True, TypeError, "numeric"),
        (-1.0, ValueError, "non-negative"),
        (float("inf"), ValueError, "finite"),
        (float("nan"), ValueError, "finite"),
    ],
)
def test_builder_rejects_invalid_guidance_scale(
    guidance_scale: object,
    error: type[Exception],
    message: str,
) -> None:
    instances: list[RecordingClassConditionalDenoiser] = []

    with pytest.raises(error, match=message):
        ClassConditionalDenoisingBuilder(
            _context(_params(guidance_scale=guidance_scale), _provider(instances))
        ).run()

    assert not instances


@pytest.mark.parametrize(
    ("conditions", "message"),
    [
        ([{"class_label": 0, "count": 4}], "must sum"),
        (
            [
                {"class_label": 0, "count": 2},
                {"class_label": 2, "count": 3},
            ],
            r"\[0, 2\)",
        ),
        ([{"class_label": True, "count": 5}], "must be an integer"),
        ([{"class_label": 0, "count": 0}], "must be positive"),
    ],
)
def test_builder_strictly_validates_condition_allocations(
    conditions: object,
    message: str,
) -> None:
    instances: list[RecordingClassConditionalDenoiser] = []

    with pytest.raises((TypeError, ValueError), match=message):
        ClassConditionalDenoisingBuilder(
            _context(_params(conditions=conditions), _provider(instances))
        ).run()


@pytest.mark.parametrize("weights", ["raw", "ema"])
def test_resolved_raw_or_ema_model_capability_is_revalidated(weights: str) -> None:
    provider = cast(
        InferenceModelProvider,
        FixedModelProvider(InvalidInferenceModel(), weights),
    )

    with pytest.raises(
        TypeError,
        match=rf"resolved {weights}.*ClassConditionalDenoiser",
    ):
        ClassConditionalDenoisingBuilder(
            _context(_params(weights=weights), provider)
        ).run()


def test_auto_weights_records_resolved_ema_selection() -> None:
    instances: list[RecordingClassConditionalDenoiser] = []

    output = ClassConditionalDenoisingBuilder(
        _context(
            _params(weights="auto"),
            _provider(instances, prefer_ema=True),
        )
    ).run()

    assert output.metadata["weights"] == "ema"

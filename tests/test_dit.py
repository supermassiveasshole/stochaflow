"""Focused contracts for the built-in Diffusion Transformer."""

from typing import Any

import pytest
import torch
from torch import nn

from stochaflow._builtin_activation import activate_model_builtins
from stochaflow.models import (
    ClassConditionalDenoiser,
    DiT,
    PrevalidatedClassConditionalDenoiser,
)
from stochaflow.models.dit import DiTBlock
from stochaflow.utils.registry import REGISTRIES


def _tiny_dit(**overrides: Any) -> DiT:
    parameters: dict[str, Any] = {
        "input_size": [8, 12],
        "patch_size": 4,
        "in_channels": 3,
        "out_channels": 3,
        "hidden_size": 16,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "num_classes": 3,
    }
    parameters.update(overrides)
    return DiT(**parameters)


def _release_final_zero_initialization(model: DiT) -> None:
    with torch.no_grad():
        nn.init.normal_(model.final_layer.modulation_projection.weight, std=0.02)
        nn.init.normal_(model.final_layer.output_projection.weight, std=0.02)


def test_dit_is_registered_and_implements_class_conditioning() -> None:
    activate_model_builtins()
    model = _tiny_dit()

    assert REGISTRIES.models.resolve("dit") is DiT
    assert isinstance(model, ClassConditionalDenoiser)
    assert isinstance(model, PrevalidatedClassConditionalDenoiser)
    assert model.num_classes == 3
    assert model.null_class_id == 3


def test_dit_s8_production_topology_stays_near_33m_parameters() -> None:
    with torch.device("meta"):
        model = DiT(
            input_size=128,
            patch_size=8,
            in_channels=3,
            out_channels=3,
            hidden_size=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            num_classes=3,
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert 32_000_000 <= parameter_count <= 34_000_000


def test_dit_preserves_rectangular_image_shape_and_dtype() -> None:
    model = _tiny_dit().double()
    state = torch.randn(2, 3, 8, 12, dtype=torch.float64)
    model_time = torch.tensor([0, 999])
    labels = torch.tensor([0, model.null_class_id])

    prediction = model.predict_class_conditioned(state, model_time, labels)

    assert prediction.shape == state.shape
    assert prediction.dtype == state.dtype
    assert prediction.device == state.device


def test_dit_prevalidated_hot_path_avoids_tensor_scalar_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_dit()
    state = torch.randn(2, 3, 8, 12)
    model_time = torch.tensor([1, 2])
    labels = torch.tensor([0, model.null_class_id])

    def fail_scalar_extraction(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("prevalidated model path synchronized a Tensor")

    monkeypatch.setattr(torch.Tensor, "item", fail_scalar_extraction)
    monkeypatch.setattr(torch.Tensor, "tolist", fail_scalar_extraction)
    monkeypatch.setattr(torch.Tensor, "__bool__", fail_scalar_extraction)

    prediction = model.predict_prevalidated_class_conditioned(
        state,
        model_time,
        labels,
    )

    assert prediction.shape == state.shape


def test_dit_forward_without_labels_uses_the_explicit_null_class() -> None:
    torch.manual_seed(1)
    model = _tiny_dit()
    _release_final_zero_initialization(model)
    state = torch.randn(2, 3, 8, 12)
    model_time = torch.tensor([1, 7])
    null_labels = torch.full((2,), model.null_class_id)

    implicit = model(state, model_time)
    explicit = model.predict_class_conditioned(
        state,
        model_time,
        null_labels,
    )

    assert torch.equal(implicit, explicit)


def test_dit_uses_fixed_position_state_and_zero_initialized_adaln() -> None:
    model = _tiny_dit()

    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    assert "position_embedding" not in parameters
    assert "position_embedding" in buffers
    assert model.position_embedding.shape == (1, 6, 16)
    assert not model.position_embedding.requires_grad

    for block_value in model.blocks:
        assert isinstance(block_value, DiTBlock)
        assert torch.count_nonzero(
            block_value.modulation_projection.weight
        ).item() == 0
        assert torch.count_nonzero(
            block_value.modulation_projection.bias
        ).item() == 0
    assert torch.count_nonzero(
        model.final_layer.modulation_projection.weight
    ).item() == 0
    assert torch.count_nonzero(
        model.final_layer.output_projection.weight
    ).item() == 0

    prediction = model(
        torch.randn(2, 3, 8, 12),
        torch.tensor([2, 4]),
        torch.tensor([0, 1]),
    )
    assert torch.count_nonzero(prediction).item() == 0


def test_dit_has_no_model_internal_class_dropout() -> None:
    torch.manual_seed(2)
    model = _tiny_dit()
    _release_final_zero_initialization(model)
    model.train()
    state = torch.randn(2, 3, 8, 12)
    model_time = torch.tensor([3, 8])
    labels = torch.tensor([1, 2])

    first = model.predict_class_conditioned(state, model_time, labels)
    second = model.predict_class_conditioned(state, model_time, labels)

    assert torch.equal(first, second)


def test_dit_real_and_null_class_embeddings_receive_gradients() -> None:
    torch.manual_seed(3)
    model = _tiny_dit()
    _release_final_zero_initialization(model)
    state = torch.randn(2, 3, 8, 12)
    model_time = torch.tensor([5, 6])
    labels = torch.tensor([0, model.null_class_id])

    prediction = model.predict_class_conditioned(state, model_time, labels)
    prediction.square().mean().backward()

    gradient = model.class_embedding.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient[0]).item() > 0
    assert torch.count_nonzero(gradient[model.null_class_id]).item() > 0


def test_dit_state_dict_round_trip_preserves_predictions() -> None:
    torch.manual_seed(4)
    first = _tiny_dit()
    _release_final_zero_initialization(first)
    state = torch.randn(2, 3, 8, 12)
    model_time = torch.tensor([11, 12])
    labels = torch.tensor([1, first.null_class_id])
    expected = first.predict_class_conditioned(state, model_time, labels)

    second = _tiny_dit()
    second.load_state_dict(first.state_dict())
    actual = second.predict_class_conditioned(state, model_time, labels)

    assert torch.equal(actual, expected)
    assert "position_embedding" in first.state_dict()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"input_size": [8]}, "input_size"),
        ({"input_size": [8, 10]}, "divisible by patch_size"),
        ({"patch_size": 0}, "patch_size"),
        ({"in_channels": True}, "in_channels"),
        ({"out_channels": 1}, "out_channels to equal in_channels"),
        ({"hidden_size": 16, "num_heads": 3}, "divisible by num_heads"),
        ({"hidden_size": 18, "num_heads": 3}, "divisible by 4"),
        ({"depth": 0}, "depth"),
        ({"mlp_ratio": float("inf")}, "mlp_ratio"),
        ({"num_classes": 0}, "num_classes"),
    ],
)
def test_dit_rejects_invalid_construction(
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _tiny_dit(**overrides)


def test_dit_rejects_invalid_state_and_time_tensors() -> None:
    model = _tiny_dit()
    valid_state = torch.randn(2, 3, 8, 12)
    valid_time = torch.tensor([1, 2])
    valid_labels = torch.tensor([0, 1])

    with pytest.raises(ValueError, match="NCHW"):
        model(torch.randn(2, 3, 8), valid_time, valid_labels)
    with pytest.raises(ValueError, match="shape after batch"):
        model(torch.randn(2, 3, 8, 8), valid_time, valid_labels)
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.ones(2, 3, 8, 12, dtype=torch.int64), valid_time, valid_labels)
    with pytest.raises(ValueError, match="model_time must be a 1D"):
        model(valid_state, valid_time[:, None], valid_labels)
    with pytest.raises(ValueError, match="model_time batch"):
        model(valid_state, torch.tensor([1]), valid_labels)
    with pytest.raises(TypeError, match="real numeric dtype"):
        model(valid_state, torch.tensor([True, False]), valid_labels)


def test_dit_rejects_invalid_class_labels() -> None:
    model = _tiny_dit()
    state = torch.randn(2, 3, 8, 12)
    model_time = torch.tensor([1, 2])

    with pytest.raises(ValueError, match="class_labels must be a 1D"):
        model(state, model_time, torch.tensor([[0], [1]]))
    with pytest.raises(ValueError, match="class_labels batch"):
        model(state, model_time, torch.tensor([0]))
    with pytest.raises(TypeError, match="int32 or int64"):
        model(state, model_time, torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        model(state, model_time, torch.tensor([-1, 0]))
    with pytest.raises(ValueError, match=r"\[0, 3\]"):
        model(state, model_time, torch.tensor([0, 4]))

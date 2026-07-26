"""Focused contracts for the built-in ADM-style UNet."""

from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.nn import functional

from stochaflow.models.adm_blocks import (
    ADMDownsample,
    ADMResidualBlock,
    ADMUpsample,
    SpatialTransformer,
    SpatialTransformerLayer,
)
from stochaflow.models.adm_unet import ADMUNet
from stochaflow.models.conditioning import (
    ClassConditionalDenoiser,
    PrevalidatedClassConditionalDenoiser,
)
from stochaflow.utils.registry import REGISTRIES


def _tiny_adm_unet(**overrides: Any) -> ADMUNet:
    parameters: dict[str, Any] = {
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 8,
        "channel_multipliers": [1, 2, 4],
        "num_res_blocks": 2,
        "transformer_depths": [0, 1, 2],
        "middle_transformer_depth": 1,
        "attention_head_dim": 8,
        "time_embedding_dim": 32,
        "num_classes": 3,
        "dropout": 0.0,
        "scale_shift_norm": True,
        "residual_resampling": True,
        "zero_init_residual": True,
        "zero_init_output": True,
    }
    parameters.update(overrides)
    return ADMUNet(**parameters)


def _release_zero_initialization(model: ADMUNet) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, ADMResidualBlock):
                nn.init.normal_(module.output_projection.weight, std=0.02)
        nn.init.normal_(model.output_projection.weight, std=0.02)


def test_adm_unet_is_registered_and_implements_class_conditioning() -> None:
    model = _tiny_adm_unet()

    assert REGISTRIES.models.resolve("adm_unet") is ADMUNet
    assert isinstance(model, ClassConditionalDenoiser)
    assert isinstance(model, PrevalidatedClassConditionalDenoiser)
    assert model.num_classes == 3
    assert model.null_class_id == 3


def test_adm_unet_preserves_rectangular_shape_dtype_and_device() -> None:
    model = _tiny_adm_unet().double()
    state = torch.randn(2, 3, 16, 24, dtype=torch.float64)
    model_time = torch.tensor([0, 999])
    labels = torch.tensor([0, model.null_class_id])

    prediction = model.predict_class_conditioned(state, model_time, labels)

    assert prediction.shape == state.shape
    assert prediction.dtype == state.dtype
    assert prediction.device == state.device


def test_adm_prevalidated_hot_path_avoids_tensor_scalar_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_adm_unet()
    state = torch.randn(2, 3, 16, 16)
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


def test_adm_unet_supports_minimum_divisible_spatial_size_at_batch_one() -> None:
    model = _tiny_adm_unet()
    state = torch.randn(1, 3, model.downsample_factor, model.downsample_factor)

    prediction = model(
        state,
        torch.tensor([0]),
        torch.tensor([model.null_class_id]),
    )

    assert prediction.shape == state.shape


def test_adm_unet_production_topology_stays_in_capacity_range() -> None:
    with torch.device("meta"):
        model = ADMUNet(
            in_channels=3,
            out_channels=3,
            base_channels=128,
            channel_multipliers=[1, 2, 3, 4],
            num_res_blocks=2,
            transformer_depths=[0, 0, 1, 2],
            middle_transformer_depth=1,
            attention_head_dim=64,
            time_embedding_dim=512,
            num_classes=3,
            dropout=0.1,
            scale_shift_norm=True,
            residual_resampling=True,
            zero_init_residual=True,
            zero_init_output=True,
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert 90_000_000 <= parameter_count <= 120_000_000


def test_adm_unet_places_independent_transformers_at_stage_ends() -> None:
    model = _tiny_adm_unet()

    assert isinstance(model.down_transformers[0], nn.Identity)
    assert isinstance(model.down_transformers[1], SpatialTransformer)
    assert isinstance(model.down_transformers[2], SpatialTransformer)
    assert isinstance(model.up_transformers[0], SpatialTransformer)
    assert isinstance(model.up_transformers[1], SpatialTransformer)
    assert isinstance(model.up_transformers[2], nn.Identity)
    assert isinstance(model.middle_transformer, SpatialTransformer)

    down_stage_one = cast(SpatialTransformer, model.down_transformers[1])
    down_stage_two = cast(SpatialTransformer, model.down_transformers[2])
    up_stage_two = cast(SpatialTransformer, model.up_transformers[0])
    up_stage_one = cast(SpatialTransformer, model.up_transformers[1])
    middle = cast(SpatialTransformer, model.middle_transformer)
    down_depths = [
        len(transformer.layers) for transformer in (down_stage_one, down_stage_two)
    ]
    up_depths = [
        len(transformer.layers) for transformer in (up_stage_two, up_stage_one)
    ]
    assert down_depths == [1, 2]
    assert up_depths == [2, 1]
    assert len(middle.layers) == 1
    assert model.down_transformers[2] is not model.up_transformers[0]

    assert len(model.down_blocks) == 3
    assert all(len(cast(nn.ModuleList, blocks)) == 2 for blocks in model.down_blocks)
    assert len(model.up_blocks) == 3
    assert all(len(cast(nn.ModuleList, blocks)) == 2 for blocks in model.up_blocks)


def test_adm_transformer_uses_scaled_dot_product_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = functional.scaled_dot_product_attention
    calls: list[tuple[torch.Size, torch.Size, torch.Size]] = []

    def recording_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        calls.append((query.shape, key.shape, value.shape))
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(
        functional,
        "scaled_dot_product_attention",
        recording_attention,
    )
    layer = SpatialTransformerLayer(16, attention_head_dim=4)

    output = layer(torch.randn(2, 5, 16))

    assert output.shape == (2, 5, 16)
    assert calls == [
        (
            torch.Size([2, 4, 5, 4]),
            torch.Size([2, 4, 5, 4]),
            torch.Size([2, 4, 5, 4]),
        )
    ]


def test_adm_unet_zero_initializes_residual_and_final_outputs() -> None:
    model = _tiny_adm_unet()

    residual_blocks = [
        module for module in model.modules() if isinstance(module, ADMResidualBlock)
    ]
    assert residual_blocks
    for block in residual_blocks:
        assert block.output_projection.bias is not None
        assert torch.count_nonzero(block.output_projection.weight).item() == 0
        assert torch.count_nonzero(block.output_projection.bias).item() == 0
    assert model.output_projection.bias is not None
    assert torch.count_nonzero(model.output_projection.weight).item() == 0
    assert torch.count_nonzero(model.output_projection.bias).item() == 0

    prediction = model(
        torch.randn(2, 3, 16, 16),
        torch.tensor([2, 4]),
        torch.tensor([0, model.null_class_id]),
    )
    assert torch.count_nonzero(prediction).item() == 0


def test_adm_unet_real_and_null_class_embeddings_receive_gradients() -> None:
    torch.manual_seed(3)
    model = _tiny_adm_unet()
    _release_zero_initialization(model)
    state = torch.randn(2, 3, 16, 16)
    model_time = torch.tensor([5, 6])
    labels = torch.tensor([0, model.null_class_id])

    prediction = model.predict_class_conditioned(state, model_time, labels)
    prediction.square().mean().backward()

    gradient = model.class_embedding.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient[0]).item() > 0
    assert torch.count_nonzero(gradient[model.null_class_id]).item() > 0


def test_adm_unet_state_dict_round_trip_preserves_predictions() -> None:
    torch.manual_seed(4)
    first = _tiny_adm_unet()
    _release_zero_initialization(first)
    first.eval()
    state = torch.randn(2, 3, 16, 16)
    model_time = torch.tensor([11, 12])
    labels = torch.tensor([1, first.null_class_id])
    expected = first.predict_class_conditioned(state, model_time, labels)

    second = _tiny_adm_unet()
    second.load_state_dict(first.state_dict())
    second.eval()
    actual = second.predict_class_conditioned(state, model_time, labels)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("residual_resampling", [False, True])
def test_adm_unet_supports_both_resampling_policies(
    residual_resampling: bool,
) -> None:
    model = _tiny_adm_unet(
        residual_resampling=residual_resampling,
        scale_shift_norm=False,
    )

    if residual_resampling:
        assert all(isinstance(module, ADMResidualBlock) for module in model.downsamples)
        assert all(isinstance(module, ADMResidualBlock) for module in model.upsamples)
    else:
        assert all(isinstance(module, ADMDownsample) for module in model.downsamples)
        assert all(isinstance(module, ADMUpsample) for module in model.upsamples)

    state = torch.randn(2, 3, 16, 24)
    prediction = model(
        state,
        torch.tensor([1, 2]),
        torch.tensor([0, model.null_class_id]),
    )
    assert prediction.shape == state.shape


def test_adm_plain_resamplers_validate_their_forward_contract() -> None:
    modules = (ADMDownsample(4, 8), ADMUpsample(4, 8))

    for module in modules:
        with pytest.raises(TypeError, match=r"torch\.Tensor"):
            module(cast(Any, "not a tensor"))
        with pytest.raises(ValueError, match="4D NCHW"):
            module(torch.randn(1, 4, 8))
        with pytest.raises(TypeError, match="floating-point"):
            module(torch.ones(1, 4, 8, 8, dtype=torch.int64))
        with pytest.raises(ValueError, match="4 channels"):
            module(torch.randn(1, 3, 8, 8))
        with pytest.raises(ValueError, match="spatial dimensions"):
            module(torch.randn(1, 4, 0, 8))


@pytest.mark.parametrize(
    ("overrides", "exception_type", "message"),
    [
        ({"in_channels": True}, TypeError, "in_channels"),
        ({"out_channels": 0}, ValueError, "out_channels"),
        ({"out_channels": 1}, ValueError, "equal in_channels"),
        ({"base_channels": 0}, ValueError, "base_channels"),
        ({"base_channels": 1}, ValueError, "at least 2"),
        ({"channel_multipliers": []}, ValueError, "channel_multipliers"),
        ({"channel_multipliers": [1, 0]}, ValueError, "channel_multipliers"),
        ({"num_res_blocks": 0}, ValueError, "num_res_blocks"),
        ({"transformer_depths": [0, 1]}, ValueError, "one value"),
        (
            {"transformer_depths": [0, -1, 2]},
            ValueError,
            "transformer_depths",
        ),
        (
            {"middle_transformer_depth": -1},
            ValueError,
            "middle_transformer_depth",
        ),
        ({"attention_head_dim": 6}, ValueError, "divisible"),
        ({"time_embedding_dim": 1}, ValueError, "at least 2"),
        ({"num_classes": 0}, ValueError, "num_classes"),
        ({"dropout": float("nan")}, ValueError, "dropout"),
        ({"dropout": 1.0}, ValueError, "dropout"),
        ({"scale_shift_norm": 1}, TypeError, "scale_shift_norm"),
        ({"residual_resampling": 1}, TypeError, "residual_resampling"),
        ({"zero_init_residual": 1}, TypeError, "zero_init_residual"),
        ({"zero_init_output": 1}, TypeError, "zero_init_output"),
    ],
)
def test_adm_unet_rejects_invalid_construction(
    overrides: dict[str, Any],
    exception_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception_type, match=message):
        _tiny_adm_unet(**overrides)


def test_adm_unet_rejects_invalid_state_and_time_tensors() -> None:
    model = _tiny_adm_unet()
    valid_state = torch.randn(2, 3, 16, 16)
    valid_time = torch.tensor([1, 2])
    valid_labels = torch.tensor([0, 1])

    with pytest.raises(ValueError, match="4D NCHW"):
        model(torch.randn(2, 3, 16), valid_time, valid_labels)
    with pytest.raises(ValueError, match="3 channels"):
        model(torch.randn(2, 2, 16, 16), valid_time, valid_labels)
    with pytest.raises(TypeError, match="floating-point"):
        model(
            torch.ones(2, 3, 16, 16, dtype=torch.int64),
            valid_time,
            valid_labels,
        )
    with pytest.raises(ValueError, match="divisible by 4"):
        model(torch.randn(2, 3, 14, 16), valid_time, valid_labels)
    with pytest.raises(ValueError, match="model_time must be a 1D"):
        model(valid_state, valid_time[:, None], valid_labels)
    with pytest.raises(ValueError, match="model_time batch"):
        model(valid_state, torch.tensor([1]), valid_labels)
    with pytest.raises(TypeError, match="int32 or int64"):
        model(valid_state, torch.tensor([1.0, 2.0]), valid_labels)
    with pytest.raises(ValueError, match="non-negative"):
        model(valid_state, torch.tensor([-1, 2]), valid_labels)


def test_adm_unet_rejects_invalid_class_labels() -> None:
    model = _tiny_adm_unet()
    state = torch.randn(2, 3, 16, 16)
    model_time = torch.tensor([1, 2])

    with pytest.raises(ValueError, match="class_labels must be a 1D"):
        model(state, model_time, torch.tensor([[0], [1]]))
    with pytest.raises(ValueError, match="class_labels batch"):
        model(state, model_time, torch.tensor([0]))
    with pytest.raises(TypeError, match="int32 or int64"):
        model(state, model_time, torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="null class identifier 3"):
        model(state, model_time, torch.tensor([-1, 0]))
    with pytest.raises(ValueError, match="null class identifier 3"):
        model(state, model_time, torch.tensor([0, 4]))

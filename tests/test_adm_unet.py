"""Focused contracts for the canonical built-in ADM U-Net."""

import json
import math
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.nn import functional

from stochaflow.models.adm_blocks import (
    ADMAttentionBlock,
    ADMResidualBlock,
)
from stochaflow.models.adm_unet import ADMConditionedSequential, ADMUNet
from stochaflow.models.conditioning import (
    ClassConditionalDenoiser,
    PrevalidatedClassConditionalDenoiser,
)
from stochaflow.processes import DiscreteGaussianProcess
from stochaflow.sampling import (
    ClassConditionalDenoisingBuilder,
    InferenceModelProvider,
    SamplingBuilderContext,
    StandardDenoisingBuilder,
)
from stochaflow.training.builder import TrainingBuilderContext
from stochaflow.training.gaussian import (
    GaussianDenoisingTrainingBuilder,
    GaussianDenoisingTrainingStrategy,
)
from stochaflow.training.objectives import MSEObjective
from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.registry import REGISTRIES

_REFERENCE = Path(__file__).parent / "fixtures" / "adm" / "reference.json"


def _tiny_adm_unet(**overrides: Any) -> ADMUNet:
    parameters: dict[str, Any] = {
        "input_size": 16,
        "in_channels": 3,
        "out_channels": 3,
        "base_channels": 8,
        "channel_multipliers": [1, 2, 4],
        "num_res_blocks": 1,
        "attention_resolutions": [8, 4],
        "attention_head_channels": 8,
        "num_classes": 3,
        "dropout": 0.0,
    }
    parameters.update(overrides)
    return ADMUNet(**parameters)


def _unexpected_component_factory(config: ComponentConfig) -> nn.Module:
    del config
    raise AssertionError("the built-in builder must preserve its injected assets")


def _adm_inference_provider(
    model: ADMUNet,
    *,
    num_classes: int | None,
) -> InferenceModelProvider:
    state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    return InferenceModelProvider(
        model_factory=lambda: _tiny_adm_unet(num_classes=num_classes),
        raw_state_dict=state,
        ema_state_dict=None,
        device=torch.device("cpu"),
    )


def _release_zero_initialization(model: ADMUNet) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (ADMResidualBlock, ADMAttentionBlock)):
                nn.init.normal_(module.output_projection.weight, std=0.02)
        nn.init.normal_(model.output_projection.weight, std=0.02)


def _fill_reference_parameters(model: nn.Module) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.arange(
                parameter.numel(),
                dtype=torch.float64,
            ).reshape(parameter.shape)
            phase = (index + 1) * 0.173
            if parameter.ndim == 1:
                initialized = 0.15 + torch.sin(values * 0.013 + phase) * 0.025
            else:
                scale = 0.025 / math.sqrt(parameter[0].numel())
                initialized = torch.sin(values * 0.013 + phase) * scale
            parameter.copy_(
                initialized.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )


def _block_residuals(
    block: nn.Module,
) -> list[ADMResidualBlock]:
    if not isinstance(block, ADMConditionedSequential):
        raise TypeError("expected an ADM conditioned block")
    return [layer for layer in block.layers if isinstance(layer, ADMResidualBlock)]


def _block_attentions(
    block: nn.Module,
) -> list[ADMAttentionBlock]:
    if not isinstance(block, ADMConditionedSequential):
        raise TypeError("expected an ADM conditioned block")
    return [layer for layer in block.layers if isinstance(layer, ADMAttentionBlock)]


def _manual_legacy_attention(
    block: ADMAttentionBlock,
    state: torch.Tensor,
) -> torch.Tensor:
    query_key_value_bias = block.query_key_value.bias
    output_projection_bias = block.output_projection.bias
    assert query_key_value_bias is not None
    assert output_projection_bias is not None
    normalized = functional.group_norm(
        state,
        block.normalization.num_groups,
        block.normalization.weight.detach(),
        block.normalization.bias.detach(),
        block.normalization.eps,
    )
    query_key_value = functional.conv2d(
        normalized,
        block.query_key_value.weight.detach(),
        query_key_value_bias.detach(),
    )
    batch_size, channels, height, width = state.shape
    token_count = height * width
    query_key_value = query_key_value.reshape(
        batch_size,
        block.num_heads,
        3,
        block.attention_head_channels,
        token_count,
    )
    query, key, value = query_key_value.unbind(dim=2)
    query = query.transpose(-2, -1)
    key = key.transpose(-2, -1)
    value = value.transpose(-2, -1)
    scale = 1.0 / math.sqrt(math.sqrt(block.attention_head_channels))
    logits = torch.matmul(query * scale, (key * scale).transpose(-2, -1))
    weights = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
    attended = torch.matmul(weights, value)
    attended = attended.transpose(-2, -1).reshape(
        batch_size,
        channels,
        height,
        width,
    )
    projected = functional.conv2d(
        attended,
        block.output_projection.weight.detach(),
        output_projection_bias.detach(),
    )
    return state + projected


def test_adm_reference_pins_and_characterization_are_frozen() -> None:
    reference = json.loads(_REFERENCE.read_text(encoding="utf-8"))

    assert reference["guided_diffusion"]["commit"] == (
        "22e0df8183507e13a7813f8d38d51b072ca1e67c"
    )
    assert reference["p2_weighting"]["commit"] == (
        "3da0947ac350072e457c211401218175bc94e137"
    )
    assert reference["legacy_stochaflow"]["parameter_count"] == 91_300_867
    assert reference["canonical_adm_128"]["parameter_count"] == 105_197_187
    assert reference["p2_afhq_256"]["parameter_count"] == 93_563_910


def test_adm_tiny_forward_and_input_gradient_match_pinned_upstream_fixture() -> None:
    fixture = json.loads(_REFERENCE.read_text(encoding="utf-8"))[
        "tiny_forward_gradient"
    ]
    model = ADMUNet(
        input_size=fixture["input_size"],
        in_channels=fixture["in_channels"],
        out_channels=fixture["out_channels"],
        base_channels=fixture["base_channels"],
        channel_multipliers=fixture["channel_multipliers"],
        num_res_blocks=fixture["num_res_blocks"],
        attention_resolutions=fixture["attention_resolutions"],
        attention_head_channels=fixture["attention_head_channels"],
        num_classes=None,
        dropout=0.0,
    )
    _fill_reference_parameters(model)
    state = torch.tensor(fixture["input"]).reshape(1, 1, 4, 4)
    state.requires_grad_(True)

    output = model(state, torch.tensor(fixture["model_time"]))
    loss = output.square().sum() + 0.125 * output.sum()
    loss.backward()

    expected_output = torch.tensor(fixture["output"]).reshape_as(output)
    expected_gradient = torch.tensor(fixture["input_gradient"]).reshape_as(state)
    torch.testing.assert_close(output, expected_output, rtol=2e-6, atol=1e-7)
    assert state.grad is not None
    torch.testing.assert_close(
        state.grad,
        expected_gradient,
        rtol=2e-4,
        atol=2e-9,
    )
    assert loss.item() == pytest.approx(fixture["loss"], rel=2e-6, abs=2e-7)


def test_adm_unet_is_registered_and_exposes_conditional_capabilities() -> None:
    model = _tiny_adm_unet()

    assert REGISTRIES.models.resolve("adm_unet") is ADMUNet
    assert isinstance(model, ClassConditionalDenoiser)
    assert isinstance(model, PrevalidatedClassConditionalDenoiser)
    assert model.num_classes == 3
    assert model.null_class_id == 3
    assert model.class_embedding is not None


def test_unconditional_adm_uses_two_argument_forward_and_independent_output_width() -> (
    None
):
    model = _tiny_adm_unet(num_classes=None, out_channels=6)
    state = torch.randn(2, 3, 16, 16)
    model_time = torch.tensor([0, 999])

    prediction = model(state, model_time)

    assert prediction.shape == (2, 6, 16, 16)
    assert model.num_classes is None
    assert model.null_class_id is None
    assert model.class_embedding is None
    with pytest.raises(ValueError, match="does not accept class_labels"):
        model(state, model_time, torch.tensor([0, 0]))
    with pytest.raises(RuntimeError, match="no class-conditioned prediction"):
        model.predict_class_conditioned(state, model_time, torch.tensor([0, 0]))


def test_unconditional_adm_runs_builtin_training_and_sampling_builders() -> None:
    torch.manual_seed(9)
    model = _tiny_adm_unet(num_classes=None)
    process = DiscreteGaussianProcess(
        {
            "name": "linear_beta",
            "params": {
                "num_timesteps": 4,
                "beta_start": 0.0001,
                "beta_end": 0.02,
            },
        }
    )
    plan = GaussianDenoisingTrainingBuilder(
        TrainingBuilderContext(
            params={"prediction_type": "epsilon"},
            primary_model=model,
            process=process,
            objective=MSEObjective(),
            model_factory=_unexpected_component_factory,
            objective_factory=_unexpected_component_factory,
        )
    ).build()

    assert isinstance(plan.strategy, GaussianDenoisingTrainingStrategy)
    assert plan.primary_model is model
    assert plan.inference_recipe is not None
    assert plan.inference_recipe.name == "standard_denoising"
    training_output = plan.strategy.training_step(torch.randn(2, 3, 16, 16))
    training_output.loss.backward()

    assert training_output.loss.shape == ()
    assert model.output_projection.weight.grad is not None
    sampling_output = StandardDenoisingBuilder(
        SamplingBuilderContext(
            params={
                "weights": "raw",
                "prediction_type": "epsilon",
                "variance": {"mode": "fixed"},
                "clip_denoised": True,
                "sampler": {
                    "name": "ddim",
                    "params": {"num_inference_steps": 1, "eta": 0.0},
                },
                "trajectory": {"enabled": True, "every_steps": 1},
            },
            process=process,
            model_provider=_adm_inference_provider(model, num_classes=None),
            device=torch.device("cpu"),
            seed=10,
            shape=(3, 16, 16),
            num_samples=2,
            batch_size=2,
        )
    ).run()

    assert len(sampling_output.batches) == 1
    assert sampling_output.batches[0].samples.shape == (2, 3, 16, 16)
    assert sampling_output.batches[0].trajectory is not None
    assert sampling_output.metadata["weights"] == "raw"
    assert sampling_output.metadata["sampler"] == {
        "name": "ddim",
        "params": {"num_inference_steps": 1, "eta": 0.0},
    }
    assert sampling_output.metadata["solver_diagnostics"] == [
        {"num_dynamics_evaluations": 1}
    ]


def test_class_conditioned_adm_runs_builtin_cfg_sampling_builder() -> None:
    torch.manual_seed(10)
    model = _tiny_adm_unet(num_classes=3)
    state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    seen_embedding_labels: list[torch.Tensor] = []

    def record_embedding_labels(
        module: nn.Module,
        inputs: tuple[Any, ...],
    ) -> None:
        del module
        labels = inputs[0]
        assert isinstance(labels, torch.Tensor)
        seen_embedding_labels.append(labels.detach().cpu().clone())

    def model_factory() -> nn.Module:
        inference_model = _tiny_adm_unet(num_classes=3)
        assert inference_model.class_embedding is not None
        inference_model.class_embedding.register_forward_pre_hook(
            record_embedding_labels
        )
        return inference_model

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

    output = ClassConditionalDenoisingBuilder(
        SamplingBuilderContext(
            params={
                "weights": "raw",
                "prediction_type": "epsilon",
                "variance": {"mode": "fixed"},
                "clip_denoised": True,
                "guidance_scale": 2.0,
                "conditions": [
                    {"class_label": 0, "count": 1},
                    {"class_label": 2, "count": 1},
                ],
                "sampler": {
                    "name": "ddim",
                    "params": {"num_inference_steps": 1, "eta": 0.0},
                },
                "trajectory": {"enabled": False, "every_steps": 1},
            },
            process=process,
            model_provider=InferenceModelProvider(
                model_factory=model_factory,
                raw_state_dict=state,
                ema_state_dict=None,
                device=torch.device("cpu"),
            ),
            device=torch.device("cpu"),
            seed=11,
            shape=(3, 16, 16),
            num_samples=2,
            batch_size=2,
        )
    ).run()

    assert len(output.batches) == 1
    assert output.batches[0].samples.shape == (2, 3, 16, 16)
    assert output.metadata["weights"] == "raw"
    assert output.metadata["guidance_scale"] == 2.0
    assert output.metadata["conditions"] == [
        {"class_label": 0, "count": 1},
        {"class_label": 2, "count": 1},
    ]
    assert output.metadata["forward_call_count"] == 1
    assert output.metadata["conditional_branch_evaluation_count"] == 1
    assert output.metadata["unconditional_branch_evaluation_count"] == 1
    assert len(seen_embedding_labels) == 1
    assert torch.equal(seen_embedding_labels[0], torch.tensor([0, 2, 3, 3]))


def test_adm_unet_preserves_configured_shape_dtype_and_device() -> None:
    model = _tiny_adm_unet().double()
    state = torch.randn(2, 3, 16, 16, dtype=torch.float64)
    model_time = torch.tensor([0, 999])
    labels = torch.tensor([0, cast(int, model.null_class_id)])

    prediction = model.predict_class_conditioned(state, model_time, labels)

    assert prediction.shape == state.shape
    assert prediction.dtype == state.dtype
    assert prediction.device == state.device
    with pytest.raises(ValueError, match="configured input_size 16"):
        model(
            torch.randn(2, 3, 16, 24, dtype=torch.float64),
            model_time,
            labels,
        )


def test_adm_prevalidated_hot_path_avoids_tensor_scalar_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _tiny_adm_unet()
    state = torch.randn(2, 3, 16, 16)
    model_time = torch.tensor([1, 2])
    labels = torch.tensor([0, cast(int, model.null_class_id)])

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


def test_adm_input_ledger_and_decoder_consume_every_skip_once() -> None:
    model = _tiny_adm_unet()
    expected_blocks = len(model.channel_multipliers) * (model.num_res_blocks + 1)

    assert len(model.input_blocks) == expected_blocks
    assert len(model.output_blocks) == expected_blocks
    assert model.input_block_resolutions == (16, 16, 8, 8, 4, 4)
    assert model.output_block_input_resolutions == (4, 4, 8, 8, 16, 16)
    assert model.output_skip_channels == tuple(reversed(model.input_block_channels))

    channels = model.level_channels[-1]
    for block, skip_channels in zip(
        model.output_blocks,
        model.output_skip_channels,
        strict=True,
    ):
        assert isinstance(block, ADMConditionedSequential)
        first = block.layers[0]
        assert isinstance(first, ADMResidualBlock)
        assert first.in_channels == channels + skip_channels
        channels = first.out_channels

    calls: list[int] = []
    handles = [
        block.register_forward_hook(
            lambda module, inputs, output, index=index: calls.append(index)
        )
        for index, block in enumerate(model.output_blocks)
    ]
    try:
        model(
            torch.randn(2, 3, 16, 16),
            torch.tensor([1, 2]),
            torch.tensor([0, cast(int, model.null_class_id)]),
        )
    finally:
        for handle in handles:
            handle.remove()

    assert calls == list(range(expected_blocks))


def test_adm_resampling_preserves_level_channels() -> None:
    model = _tiny_adm_unet()
    downsamplers = [
        residual
        for block in model.input_blocks
        for residual in _block_residuals(block)
        if residual.resample == "down"
    ]
    upsamplers = [
        residual
        for block in model.output_blocks
        for residual in _block_residuals(block)
        if residual.resample == "up"
    ]

    assert len(downsamplers) == len(model.channel_multipliers) - 1
    assert len(upsamplers) == len(model.channel_multipliers) - 1
    assert all(
        block.in_channels == block.out_channels
        for block in (*downsamplers, *upsamplers)
    )

    input_residuals = [
        residual for block in model.input_blocks for residual in _block_residuals(block)
    ]
    first_level_one = next(
        block
        for block in input_residuals
        if block.resample == "none" and block.out_channels == 16
    )
    assert first_level_one.in_channels == 8


def test_adm_attention_placement_matches_spatial_resolutions() -> None:
    model = _tiny_adm_unet()
    active = set(model.attention_resolutions)

    for block, resolution in zip(
        model.input_blocks,
        model.input_block_resolutions,
        strict=True,
    ):
        residuals = _block_residuals(block)
        expected = bool(
            residuals and residuals[0].resample == "none" and resolution in active
        )
        assert len(_block_attentions(block)) == int(expected)

    for block, resolution in zip(
        model.output_blocks,
        model.output_block_input_resolutions,
        strict=True,
    ):
        assert len(_block_attentions(block)) == int(resolution in active)

    assert len(_block_attentions(model.middle_block)) == 1


def test_adm_production_topology_has_exact_count_depth_and_attention() -> None:
    with torch.device("meta"):
        model = ADMUNet(
            input_size=128,
            in_channels=3,
            out_channels=3,
            base_channels=128,
            channel_multipliers=[1, 1, 2, 3, 4],
            num_res_blocks=2,
            attention_resolutions=[32, 16, 8],
            attention_head_channels=64,
            num_classes=3,
            dropout=0.1,
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    attention_count = sum(
        isinstance(module, ADMAttentionBlock) for module in model.modules()
    )

    assert parameter_count == 105_197_187
    assert len(model.input_blocks) == len(model.output_blocks) == 15
    assert model.level_resolutions == (128, 64, 32, 16, 8)
    assert model.downsample_factor == 16
    assert attention_count == 16


def test_p2_afhq_topology_matches_pinned_parameter_golden() -> None:
    with torch.device("meta"):
        model = ADMUNet(
            input_size=256,
            in_channels=3,
            out_channels=6,
            base_channels=128,
            channel_multipliers=[1, 1, 2, 2, 4, 4],
            num_res_blocks=1,
            attention_resolutions=[16],
            attention_head_channels=64,
            num_classes=None,
            dropout=0.1,
        )

    assert sum(parameter.numel() for parameter in model.parameters()) == 93_563_910


def test_adm_attention_uses_group_norm_qkv_and_zero_projection() -> None:
    block = ADMAttentionBlock(16, attention_head_channels=4)

    assert isinstance(block.normalization, nn.GroupNorm)
    assert isinstance(block.query_key_value, nn.Conv2d)
    assert block.query_key_value.kernel_size == (1, 1)
    assert isinstance(block.output_projection, nn.Conv2d)
    assert block.output_projection.kernel_size == (1, 1)
    assert not any(isinstance(module, nn.LayerNorm) for module in block.modules())
    assert not any(isinstance(module, nn.Linear) for module in block.modules())
    assert not any(isinstance(module, nn.Dropout) for module in block.modules())
    assert torch.count_nonzero(block.output_projection.weight).item() == 0
    assert block.output_projection.bias is not None
    assert torch.count_nonzero(block.output_projection.bias).item() == 0

    state = torch.randn(2, 16, 2, 3)
    assert torch.equal(block(state), state)


def test_adm_attention_matches_independent_forward_and_input_gradient() -> None:
    torch.manual_seed(17)
    block = ADMAttentionBlock(8, attention_head_channels=4).double()
    with torch.no_grad():
        nn.init.normal_(block.output_projection.weight, std=0.04)
        output_projection_bias = block.output_projection.bias
        assert output_projection_bias is not None
        nn.init.normal_(output_projection_bias, std=0.02)
    actual_input = torch.randn(2, 8, 2, 3, dtype=torch.float64, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)

    actual = block(actual_input)
    reference = _manual_legacy_attention(block, reference_input)
    actual.square().sum().backward()
    reference.square().sum().backward()

    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(
        actual_input.grad,
        reference_input.grad,
        rtol=1e-6,
        atol=1e-8,
    )


def test_adm_attention_uses_scaled_dot_product_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = functional.scaled_dot_product_attention
    calls: list[tuple[torch.Size, torch.Size, torch.Size, float]] = []

    def recording_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        calls.append(
            (
                query.shape,
                key.shape,
                value.shape,
                float(kwargs["dropout_p"]),
            )
        )
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(
        functional,
        "scaled_dot_product_attention",
        recording_attention,
    )
    block = ADMAttentionBlock(16, attention_head_channels=4)

    output = block(torch.randn(2, 16, 2, 3))

    assert output.shape == (2, 16, 2, 3)
    expected = torch.Size([2, 4, 6, 4])
    assert calls == [(expected, expected, expected, 0.0)]


def test_adm_time_embedding_is_fixed_to_four_times_base_width() -> None:
    model = _tiny_adm_unet()
    first = model.time_embedding[0]
    second = model.time_embedding[2]

    assert isinstance(first, nn.Linear)
    assert isinstance(second, nn.Linear)
    assert (first.in_features, first.out_features) == (8, 32)
    assert (second.in_features, second.out_features) == (32, 32)
    assert model.time_embedding_dim == 32
    assert model.class_embedding is not None
    assert model.class_embedding.embedding_dim == 32


def test_adm_unet_zero_initializes_residual_attention_and_final_outputs() -> None:
    model = _tiny_adm_unet()

    residual_blocks = [
        module for module in model.modules() if isinstance(module, ADMResidualBlock)
    ]
    attention_blocks = [
        module for module in model.modules() if isinstance(module, ADMAttentionBlock)
    ]
    assert residual_blocks
    assert attention_blocks
    for block in (*residual_blocks, *attention_blocks):
        assert block.output_projection.bias is not None
        assert torch.count_nonzero(block.output_projection.weight).item() == 0
        assert torch.count_nonzero(block.output_projection.bias).item() == 0
    assert model.output_projection.bias is not None
    assert torch.count_nonzero(model.output_projection.weight).item() == 0
    assert torch.count_nonzero(model.output_projection.bias).item() == 0

    prediction = model(
        torch.randn(2, 3, 16, 16),
        torch.tensor([2, 4]),
        torch.tensor([0, cast(int, model.null_class_id)]),
    )
    assert torch.count_nonzero(prediction).item() == 0


def test_adm_unet_forward_backward_and_class_embeddings() -> None:
    torch.manual_seed(3)
    model = _tiny_adm_unet()
    _release_zero_initialization(model)
    state = torch.randn(2, 3, 16, 16, requires_grad=True)
    model_time = torch.tensor([5, 6])
    labels = torch.tensor([0, cast(int, model.null_class_id)])

    prediction = model.predict_class_conditioned(state, model_time, labels)
    prediction.square().mean().backward()

    assert state.grad is not None
    assert torch.count_nonzero(state.grad).item() > 0
    assert model.class_embedding is not None
    gradient = model.class_embedding.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient[0]).item() > 0
    assert torch.count_nonzero(gradient[cast(int, model.null_class_id)]).item() > 0


def test_adm_unet_state_dict_round_trip_preserves_predictions() -> None:
    torch.manual_seed(4)
    first = _tiny_adm_unet()
    _release_zero_initialization(first)
    first.eval()
    state = torch.randn(2, 3, 16, 16)
    model_time = torch.tensor([11, 12])
    labels = torch.tensor([1, cast(int, first.null_class_id)])
    expected = first.predict_class_conditioned(state, model_time, labels)

    second = _tiny_adm_unet()
    second.load_state_dict(first.state_dict())
    second.eval()
    actual = second.predict_class_conditioned(state, model_time, labels)

    assert torch.equal(actual, expected)


def test_adm_unet_supports_cpu_bfloat16_autocast() -> None:
    model = _tiny_adm_unet()
    state = torch.randn(2, 3, 16, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = model(
            state,
            torch.tensor([1, 2]),
            torch.tensor([0, cast(int, model.null_class_id)]),
        )

    assert prediction.shape == state.shape
    assert prediction.dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("overrides", "exception_type", "message"),
    [
        ({"input_size": True}, TypeError, "input_size"),
        ({"input_size": 0}, ValueError, "input_size"),
        ({"out_channels": 0}, ValueError, "out_channels"),
        ({"base_channels": 1}, ValueError, "at least 2"),
        ({"channel_multipliers": []}, ValueError, "channel_multipliers"),
        ({"channel_multipliers": [2, 4]}, ValueError, "first channel multiplier"),
        (
            {"input_size": 14},
            ValueError,
            "divisible by the ADM downsample factor",
        ),
        (
            {"attention_resolutions": [8, 8]},
            ValueError,
            "must not contain duplicates",
        ),
        (
            {"attention_resolutions": [6]},
            ValueError,
            "must name ADM spatial levels",
        ),
        (
            {"attention_head_channels": 6},
            ValueError,
            "divisible by attention_head_channels",
        ),
        ({"num_classes": 0}, ValueError, "num_classes"),
        ({"dropout": float("nan")}, ValueError, "dropout"),
        ({"dropout": 1.0}, ValueError, "dropout"),
    ],
)
def test_adm_unet_rejects_invalid_construction(
    overrides: dict[str, Any],
    exception_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception_type, match=message):
        _tiny_adm_unet(**overrides)


@pytest.mark.parametrize(
    "legacy_field",
    [
        "transformer_depths",
        "middle_transformer_depth",
        "attention_head_dim",
        "time_embedding_dim",
        "scale_shift_norm",
        "residual_resampling",
        "zero_init_residual",
        "zero_init_output",
    ],
)
def test_adm_unet_rejects_legacy_constructor_fields(legacy_field: str) -> None:
    with pytest.raises(TypeError, match=legacy_field):
        _tiny_adm_unet(**{legacy_field: 1})


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
    with pytest.raises(ValueError, match="configured input_size 16"):
        model(torch.randn(2, 3, 8, 8), valid_time, valid_labels)
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

    with pytest.raises(TypeError, match=r"class_labels must be a torch\.Tensor"):
        model(state, model_time)
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

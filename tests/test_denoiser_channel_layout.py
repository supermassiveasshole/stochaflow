"""Tests for the optional static denoiser channel-layout capability."""

import pytest

from stochaflow.extensions import (
    DenoiserChannelLayout as ExtensionDenoiserChannelLayout,
)
from stochaflow.models import ADMUNet, DenoiserChannelLayout, UNet


def test_denoiser_channel_layout_is_an_extension_public_contract() -> None:
    assert ExtensionDenoiserChannelLayout is DenoiserChannelLayout


@pytest.mark.parametrize(
    "model",
    [
        UNet(
            in_channels=2,
            out_channels=4,
            base_channels=4,
            channel_multipliers=(1,),
            num_res_blocks=1,
            time_embedding_dim=4,
        ),
        ADMUNet(
            input_size=4,
            in_channels=2,
            out_channels=4,
            base_channels=4,
            channel_multipliers=(1,),
            num_res_blocks=1,
            attention_resolutions=(),
            attention_head_channels=2,
        ),
    ],
    ids=["unet", "adm_unet"],
)
def test_builtin_unets_declare_static_denoiser_channel_layout(
    model: object,
) -> None:
    assert isinstance(model, DenoiserChannelLayout)
    assert model.in_channels == 2
    assert model.out_channels == 4

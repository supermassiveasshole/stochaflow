"""Tests for the diffusion UNet."""

import torch

from stochaflow.models.unet import UNet


def test_unet_preserves_spatial_shape() -> None:
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
        channel_multipliers=(1, 2),
        num_res_blocks=2,
        time_embedding_dim=32,
    )
    x = torch.randn(4, 3, 32, 32)
    timesteps = torch.randint(0, 1000, (4,))
    y = model(x, timesteps)
    assert y.shape == x.shape


def test_unet_supports_single_channel_inputs() -> None:
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        channel_multipliers=(1, 2, 2),
        num_res_blocks=1,
        time_embedding_dim=32,
    )
    x = torch.randn(2, 1, 28, 28)
    timesteps = torch.randint(0, 1000, (2,))
    y = model(x, timesteps)
    assert y.shape == x.shape

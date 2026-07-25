"""Building blocks for diffusion UNets."""

import torch
from torch import nn


def _group_norm_groups(num_channels: int, max_groups: int = 32) -> int:
    """Choose a valid GroupNorm group count for the given channel width."""

    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    """Residual block with optional time-conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embedding_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_norm_groups(in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(_group_norm_groups(out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        """Apply a residual block conditioned on a time embedding."""

        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block over spatial tokens."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.norm = nn.GroupNorm(_group_norm_groups(channels), channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.projection = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)
        h, _ = self.attention(h, h, h, need_weights=False)
        h = self.projection(h)
        h = h.transpose(1, 2).reshape(batch_size, channels, height, width)
        return x + h


class Downsample(nn.Module):
    """Spatial downsampling by a factor of 2."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """Spatial upsampling by a factor of 2."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

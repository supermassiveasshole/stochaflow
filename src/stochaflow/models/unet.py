"""UNet backbone for diffusion models."""

from typing import cast

import torch
import torch.nn as nn

from stochaflow.models.blocks import AttentionBlock, Downsample, ResidualBlock, Upsample
from stochaflow.models.embeddings import TimeEmbedding
from stochaflow.utils.registry import REGISTRIES


@REGISTRIES.models.register("unet")
class UNet(nn.Module):
    """A compact time-conditioned UNet suitable for DDPM-style denoisers."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_embedding_dim: int = 128,
        dropout: float = 0.0,
        attention_levels: tuple[int, ...] | list[int] | None = None,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()

        model_out_channels = out_channels
        attention_level_set = set(attention_levels or ())

        self.time_embedding = TimeEmbedding(
            embedding_dim=time_embedding_dim,
            hidden_dim=base_channels * 4,
        )
        time_dim = self.time_embedding.output_dim

        self.input_projection = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.skip_channels: list[int] = []

        channels = base_channels
        level_channels = [base_channels * mult for mult in channel_multipliers]

        for level, block_out_channels in enumerate(level_channels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ResidualBlock(
                        channels,
                        block_out_channels,
                        time_embedding_dim=time_dim,
                        dropout=dropout,
                    )
                )
                channels = block_out_channels
                self.skip_channels.append(channels)
                if level in attention_level_set:
                    blocks.append(AttentionBlock(channels, num_heads=attention_heads))

            self.down_blocks.append(blocks)

            if level < len(level_channels) - 1:
                self.downsamples.append(Downsample(channels))

        self.mid_block1 = ResidualBlock(
            channels,
            channels,
            time_embedding_dim=time_dim,
            dropout=dropout,
        )
        self.mid_attention = (
            AttentionBlock(channels, num_heads=attention_heads)
            if len(level_channels) - 1 in attention_level_set
            else nn.Identity()
        )
        self.mid_block2 = ResidualBlock(
            channels,
            channels,
            time_embedding_dim=time_dim,
            dropout=dropout,
        )

        skip_stack = list(self.skip_channels)

        for level in reversed(range(len(level_channels))):
            blocks = nn.ModuleList()
            block_out_channels = level_channels[level]

            for _ in range(num_res_blocks):
                skip_channels = skip_stack.pop()
                blocks.append(
                    ResidualBlock(
                        channels + skip_channels,
                        block_out_channels,
                        time_embedding_dim=time_dim,
                        dropout=dropout,
                    )
                )
                channels = block_out_channels
                if level in attention_level_set:
                    blocks.append(AttentionBlock(channels, num_heads=attention_heads))

            self.up_blocks.append(blocks)

            if level > 0:
                self.upsamples.append(Upsample(channels))

        self.output_norm = nn.GroupNorm(32 if channels % 32 == 0 else 1, channels)
        self.output_act = nn.SiLU()
        self.output_projection = nn.Conv2d(
            channels,
            model_out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Denoise an input batch conditioned on discrete diffusion timesteps."""

        if timesteps.ndim != 1:
            raise ValueError("timesteps must be a 1D tensor")
        if timesteps.shape[0] != x.shape[0]:
            raise ValueError("timesteps batch dimension must match input batch")

        time_embedding = self.time_embedding(timesteps)
        h = self.input_projection(x)
        skips: list[torch.Tensor] = []

        for level in range(len(self.down_blocks)):
            blocks = cast(nn.ModuleList, self.down_blocks[level])
            block_index = 0
            while block_index < len(blocks):
                block = blocks[block_index]
                if not isinstance(block, ResidualBlock):
                    h = block(h)
                    block_index += 1
                    continue

                h = block(h, time_embedding)
                block_index += 1
                if block_index < len(blocks) and isinstance(
                    blocks[block_index],
                    AttentionBlock,
                ):
                    h = blocks[block_index](h)
                    block_index += 1
                skips.append(h)

            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        h = self.mid_block1(h, time_embedding)
        h = self.mid_attention(h)
        h = self.mid_block2(h, time_embedding)

        for level in range(len(self.up_blocks)):
            blocks = cast(nn.ModuleList, self.up_blocks[level])

            for block in blocks:
                if isinstance(block, ResidualBlock):
                    skip = skips.pop()
                    h = torch.cat([h, skip], dim=1)
                    h = block(h, time_embedding)
                else:
                    h = block(h)

            if level < len(self.upsamples):
                h = self.upsamples[level](h)

        h = self.output_projection(self.output_act(self.output_norm(h)))
        return h

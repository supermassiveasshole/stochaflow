"""A compact condition-aware denoiser with model-owned physics state."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from stochaflow.extensions import REGISTRIES


def _normalization_groups(channels: int) -> int:
    return next(group for group in range(min(8, channels), 0, -1) if channels % group == 0)


class _TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 4 or embedding_dim % 2:
            raise ValueError("time_embedding_dim must be an even integer of at least 4")
        self.embedding_dim = embedding_dim

    def forward(self, model_time: torch.Tensor) -> torch.Tensor:
        half = self.embedding_dim // 2
        exponent = -math.log(10_000.0) * torch.arange(
            half,
            device=model_time.device,
            dtype=torch.float32,
        ) / max(half - 1, 1)
        frequencies = exponent.exp()
        arguments = model_time.to(torch.float32).reshape(-1, 1) * frequencies.reshape(1, -1)
        return torch.cat((arguments.sin(), arguments.cos()), dim=1)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, time_embedding_dim: int) -> None:
        super().__init__()
        groups = _normalization_groups(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time = nn.Linear(time_embedding_dim, channels)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()

    def forward(self, state: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(self.activation(self.norm1(state)))
        hidden = hidden + self.time(time_embedding).reshape(state.shape[0], -1, 1, 1)
        hidden = self.conv2(self.activation(self.norm2(hidden)))
        return state + hidden


@REGISTRIES.models.register("physics-reconstruction.conditional-denoiser")
class ConditionalDenoiser(nn.Module):
    """Predict Gaussian semantics from a state and detached physics condition."""

    normalization_mean: torch.Tensor
    normalization_scale: torch.Tensor
    reynolds_number: torch.Tensor
    time_delta: torch.Tensor
    linear_damping: torch.Tensor
    forcing_wavenumber: torch.Tensor
    forcing_amplitude: torch.Tensor

    def __init__(
        self,
        *,
        channels: int = 3,
        hidden_channels: int = 32,
        num_blocks: int = 4,
        time_embedding_dim: int = 64,
        normalization_mean: float = 0.0,
        normalization_scale: float = 1.0,
        reynolds_number: float = 1000.0,
        time_delta: float = 0.03125,
        linear_damping: float = 0.1,
        forcing_wavenumber: int = 4,
        forcing_amplitude: float = -4.0,
    ) -> None:
        super().__init__()
        for name, value in {
            "channels": channels,
            "hidden_channels": hidden_channels,
            "num_blocks": num_blocks,
            "time_embedding_dim": time_embedding_dim,
            "forcing_wavenumber": forcing_wavenumber,
        }.items():
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if channels != 3:
            raise ValueError("physics reconstruction requires exactly three time channels")
        for name, value in {
            "normalization_mean": normalization_mean,
            "normalization_scale": normalization_scale,
            "reynolds_number": reynolds_number,
            "time_delta": time_delta,
            "linear_damping": linear_damping,
            "forcing_amplitude": forcing_amplitude,
        }.items():
            if isinstance(value, bool):
                raise TypeError(f"{name} must be numeric")
        if normalization_scale <= 0:
            raise ValueError("normalization_scale must be positive")
        if reynolds_number <= 0:
            raise ValueError("reynolds_number must be positive")
        if time_delta <= 0:
            raise ValueError("time_delta must be positive")

        self.channels = channels
        self.time_embedding = _TimeEmbedding(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        self.input = nn.Conv2d(channels * 2, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            _ResidualBlock(hidden_channels, time_embedding_dim)
            for _ in range(num_blocks)
        )
        self.output = nn.Sequential(
            nn.GroupNorm(_normalization_groups(hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
        )
        self.register_buffer(
            "normalization_mean",
            torch.tensor(float(normalization_mean), dtype=torch.float32),
        )
        self.register_buffer(
            "normalization_scale",
            torch.tensor(float(normalization_scale), dtype=torch.float32),
        )
        self.register_buffer(
            "reynolds_number",
            torch.tensor(float(reynolds_number), dtype=torch.float32),
        )
        self.register_buffer(
            "time_delta",
            torch.tensor(float(time_delta), dtype=torch.float32),
        )
        self.register_buffer(
            "linear_damping",
            torch.tensor(float(linear_damping), dtype=torch.float32),
        )
        self.register_buffer(
            "forcing_wavenumber",
            torch.tensor(forcing_wavenumber, dtype=torch.long),
        )
        self.register_buffer(
            "forcing_amplitude",
            torch.tensor(float(forcing_amplitude), dtype=torch.float32),
        )

    def normalize(self, physical_state: torch.Tensor) -> torch.Tensor:
        """Normalize physical vorticity with checkpointed statistics."""

        return (physical_state - self.normalization_mean) / self.normalization_scale

    def denormalize(self, normalized_state: torch.Tensor) -> torch.Tensor:
        """Restore physical vorticity units."""

        return normalized_state * self.normalization_scale + self.normalization_mean

    def forward(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim != 4 or state.shape[1] != self.channels:
            raise ValueError("denoiser state must have shape [batch, 3, height, width]")
        if condition.shape != state.shape:
            raise ValueError("physics condition must match the denoiser state")
        if condition.device != state.device or condition.dtype != state.dtype:
            raise ValueError("physics condition must share state device and dtype")
        if model_time.ndim != 1 or model_time.shape[0] != state.shape[0]:
            raise ValueError("model_time must be a 1D tensor matching the batch")
        embedded = self.time_mlp(self.time_embedding(model_time)).to(dtype=state.dtype)
        hidden = self.input(torch.cat((state, condition), dim=1))
        for block in self.blocks:
            hidden = block(hidden, embedded)
        return self.output(hidden)


__all__ = ["ConditionalDenoiser"]

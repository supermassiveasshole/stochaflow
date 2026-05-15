"""Embedding layers for diffusion models."""

import math

import torch
import torch.nn as nn


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    max_period: int = 10_000,
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings for a batch of discrete times."""

    if timesteps.ndim != 1:
        raise ValueError("timesteps must be a 1D tensor")
    half_dim = embedding_dim // 2
    if half_dim == 0:
        raise ValueError("embedding_dim must be at least 2")

    exponent = -math.log(max_period) * torch.arange(
        half_dim,
        dtype=torch.float32,
        device=timesteps.device,
    ) / half_dim
    frequencies = torch.exp(exponent)
    args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)

    if embedding_dim % 2 == 1:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])],
            dim=1,
        )
    return embedding


class TimeEmbedding(nn.Module):
    """Project sinusoidal timestep features into a learned embedding space."""

    def __init__(self, embedding_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embedding_dim * 4
        self.embedding_dim = embedding_dim
        self._output_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @property
    def output_dim(self) -> int:
        """Return the output dimensionality of the learned time embedding."""

        return self._output_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed a batch of discrete timesteps."""

        base = sinusoidal_timestep_embedding(timesteps, self.embedding_dim)
        return self.mlp(base)

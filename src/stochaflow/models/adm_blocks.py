"""Canonical ADM residual and spatial-attention building blocks."""

import math
from typing import Literal, cast

import torch
from torch import nn
from torch.nn import functional

ResidualResampleKind = Literal["none", "down", "up"]


def validate_adm_positive_integer(name: str, value: object) -> int:
    """Validate one positive integer used by ADM components."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_adm_dropout(dropout: object) -> float:
    """Validate and normalize an ADM dropout probability."""

    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
        raise TypeError("dropout must be a real number")
    value = float(dropout)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("dropout must be a finite number in [0, 1)")
    return value


def adm_group_norm_groups(channels: int, maximum_groups: int = 32) -> int:
    """Choose the largest valid ADM GroupNorm group count up to 32."""

    channels = validate_adm_positive_integer("channels", channels)
    maximum_groups = validate_adm_positive_integer(
        "maximum_groups",
        maximum_groups,
    )
    for groups in range(min(channels, maximum_groups), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def zero_adm_module(module: nn.Module) -> None:
    """Zero every parameter owned by one ADM projection module."""

    for parameter in module.parameters():
        nn.init.zeros_(parameter)


def _validated_spatial_state(
    state: object,
    *,
    channels: int,
) -> torch.Tensor:
    if not isinstance(state, torch.Tensor):
        raise TypeError("state must be a torch.Tensor")
    if state.ndim != 4:
        raise ValueError("state must be a 4D NCHW tensor")
    if not state.is_floating_point():
        raise TypeError("state must have a floating-point dtype")
    if state.shape[0] <= 0:
        raise ValueError("state batch dimension must be positive")
    if state.shape[1] != channels:
        raise ValueError(f"state must have {channels} channels, got {state.shape[1]}")
    if state.shape[2] <= 0 or state.shape[3] <= 0:
        raise ValueError("state spatial dimensions must be positive")
    return state


def _validate_group_norm_input(
    state: torch.Tensor,
    normalization: nn.GroupNorm,
) -> None:
    values_per_group = (
        normalization.num_channels
        // normalization.num_groups
        * state.shape[2]
        * state.shape[3]
    )
    if values_per_group < 2:
        raise ValueError("state must provide at least two values per GroupNorm group")


def _resampling_module(kind: ResidualResampleKind) -> nn.Module:
    if kind == "up":
        return nn.Upsample(scale_factor=2, mode="nearest")
    if kind == "down":
        return nn.AvgPool2d(kernel_size=2, stride=2)
    return nn.Identity()


class ADMResidualBlock(nn.Module):
    """Apply one scale-shift ADM residual block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        dropout: float = 0.0,
        *,
        resample: ResidualResampleKind = "none",
    ) -> None:
        super().__init__()
        validate_adm_positive_integer("in_channels", in_channels)
        validate_adm_positive_integer("out_channels", out_channels)
        validate_adm_positive_integer("embedding_dim", embedding_dim)
        dropout_value = validate_adm_dropout(dropout)
        if resample not in {"none", "down", "up"}:
            raise ValueError("resample must be one of: none, down, up")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.resample = resample

        self.input_norm = nn.GroupNorm(
            adm_group_norm_groups(in_channels),
            in_channels,
        )
        self.input_activation = nn.SiLU()
        self.main_resampling = _resampling_module(resample)
        self.skip_resampling = _resampling_module(resample)
        self.input_projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.embedding_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, out_channels * 2),
        )
        self.output_norm = nn.GroupNorm(
            adm_group_norm_groups(out_channels),
            out_channels,
        )
        self.output_activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout_value)
        self.output_projection = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        zero_adm_module(self.output_projection)

        self.skip_projection: nn.Module
        if in_channels == out_channels:
            self.skip_projection = nn.Identity()
        else:
            self.skip_projection = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            )

    def forward(
        self,
        state: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Transform a spatial state using one conditioning embedding."""

        state = _validated_spatial_state(
            cast(object, state),
            channels=self.in_channels,
        )
        if self.resample == "down" and (state.shape[2] < 2 or state.shape[3] < 2):
            raise ValueError("downsampling requires spatial dimensions of at least 2")
        _validate_group_norm_input(state, self.input_norm)
        embedding_value = cast(object, embedding)
        if not isinstance(embedding_value, torch.Tensor):
            raise TypeError("embedding must be a torch.Tensor")
        embedding = embedding_value
        if embedding.ndim != 2:
            raise ValueError("embedding must be a 2D tensor")
        if not embedding.is_floating_point():
            raise TypeError("embedding must have a floating-point dtype")
        if embedding.shape != (state.shape[0], self.embedding_dim):
            raise ValueError(
                f"embedding shape must be ({state.shape[0]}, {self.embedding_dim})"
            )
        if embedding.device != state.device:
            raise ValueError("state and embedding must be on the same device")

        residual = self.skip_projection(self.skip_resampling(state))
        hidden = self.input_activation(self.input_norm(state))
        hidden = self.input_projection(self.main_resampling(hidden))
        _validate_group_norm_input(hidden, self.output_norm)

        projected = self.embedding_projection(embedding).to(dtype=hidden.dtype)
        scale, shift = projected.chunk(2, dim=1)
        hidden = self.output_norm(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None])
        hidden = hidden + shift[:, :, None, None]
        hidden = self.output_activation(hidden)
        hidden = self.output_projection(self.dropout(hidden))
        return residual + hidden


class ADMAttentionBlock(nn.Module):
    """Apply canonical ADM QKV self-attention over spatial positions."""

    def __init__(
        self,
        channels: int,
        attention_head_channels: int,
    ) -> None:
        super().__init__()
        validate_adm_positive_integer("channels", channels)
        validate_adm_positive_integer(
            "attention_head_channels",
            attention_head_channels,
        )
        if channels % attention_head_channels != 0:
            raise ValueError("channels must be divisible by attention_head_channels")

        self.channels = channels
        self.attention_head_channels = attention_head_channels
        self.num_heads = channels // attention_head_channels
        self.normalization = nn.GroupNorm(
            adm_group_norm_groups(channels),
            channels,
        )
        self.query_key_value = nn.Conv2d(
            channels,
            channels * 3,
            kernel_size=1,
        )
        self.output_projection = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
        )
        zero_adm_module(self.output_projection)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Attend over a feature map while preserving its NCHW shape."""

        state = _validated_spatial_state(
            cast(object, state),
            channels=self.channels,
        )
        _validate_group_norm_input(state, self.normalization)
        batch_size, channels, height, width = state.shape
        token_count = height * width
        query_key_value = self.query_key_value(self.normalization(state))

        # guided-diffusion's original QKVAttentionLegacy assigns one contiguous
        # Q/K/V channel group to each head. SDPA supplies the equivalent scaled
        # dot-product backend once the tensors are arranged as B,H,N,D.
        query_key_value = query_key_value.reshape(
            batch_size,
            self.num_heads,
            3,
            self.attention_head_channels,
            token_count,
        )
        query, key, value = query_key_value.unbind(dim=2)
        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        value = value.transpose(-2, -1)
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
        )
        attended = attended.transpose(-2, -1).reshape(
            batch_size,
            channels,
            height,
            width,
        )
        return state + self.output_projection(attended)


__all__ = [
    "ADMAttentionBlock",
    "ADMResidualBlock",
    "ResidualResampleKind",
]

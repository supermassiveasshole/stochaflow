"""ADM-style residual and spatial Transformer building blocks."""

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


def _validate_boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def adm_group_norm_groups(channels: int, maximum_groups: int = 32) -> int:
    """Choose groups while retaining two channels per group when possible."""

    channels = validate_adm_positive_integer("channels", channels)
    maximum_groups = validate_adm_positive_integer(
        "maximum_groups",
        maximum_groups,
    )
    group_limit = max(channels // 2, 1)
    for groups in range(min(group_limit, maximum_groups), 0, -1):
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
    """Apply an ADM residual block with embedding-driven normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        dropout: float = 0.0,
        *,
        scale_shift_norm: bool = True,
        resample: ResidualResampleKind = "none",
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        validate_adm_positive_integer("in_channels", in_channels)
        validate_adm_positive_integer("out_channels", out_channels)
        validate_adm_positive_integer("embedding_dim", embedding_dim)
        dropout_value = validate_adm_dropout(dropout)
        _validate_boolean("scale_shift_norm", cast(object, scale_shift_norm))
        if resample not in {"none", "down", "up"}:
            raise ValueError("resample must be one of: none, down, up")
        _validate_boolean(
            "zero_init_residual",
            cast(object, zero_init_residual),
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_dim = embedding_dim
        self.scale_shift_norm = scale_shift_norm
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

        projected_embedding_dim = out_channels * 2 if scale_shift_norm else out_channels
        self.embedding_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, projected_embedding_dim),
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
        if zero_init_residual:
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

        projected = self.embedding_projection(embedding)
        projected = projected.to(dtype=hidden.dtype)
        if self.scale_shift_norm:
            scale, shift = projected.chunk(2, dim=1)
            hidden = self.output_norm(hidden)
            hidden = hidden * (1.0 + scale[:, :, None, None])
            hidden = hidden + shift[:, :, None, None]
        else:
            hidden = hidden + projected[:, :, None, None]
            hidden = self.output_norm(hidden)

        hidden = self.output_activation(hidden)
        hidden = self.output_projection(self.dropout(hidden))
        return residual + hidden


class ADMDownsample(nn.Module):
    """Downsample while adapting the channel width without a residual path."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        validate_adm_positive_integer("in_channels", in_channels)
        validate_adm_positive_integer("out_channels", out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Downsample a spatial state."""

        state = _validated_spatial_state(
            cast(object, state),
            channels=self.in_channels,
        )
        if state.shape[2] < 2 or state.shape[3] < 2:
            raise ValueError("downsampling requires spatial dimensions of at least 2")
        return self.projection(state)


class ADMUpsample(nn.Module):
    """Upsample while adapting the channel width without a residual path."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        validate_adm_positive_integer("in_channels", in_channels)
        validate_adm_positive_integer("out_channels", out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Upsample a spatial state."""

        state = _validated_spatial_state(
            cast(object, state),
            channels=self.in_channels,
        )
        state = functional.interpolate(state, scale_factor=2, mode="nearest")
        return self.projection(state)


class SpatialTransformerLayer(nn.Module):
    """Pre-normalized spatial self-attention and MLP residual layer."""

    def __init__(
        self,
        channels: int,
        attention_head_dim: int,
        *,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        validate_adm_positive_integer("channels", channels)
        validate_adm_positive_integer(
            "attention_head_dim",
            attention_head_dim,
        )
        if channels % attention_head_dim != 0:
            raise ValueError("channels must be divisible by attention_head_dim")
        dropout_value = validate_adm_dropout(dropout)
        mlp_ratio_value_input = cast(object, mlp_ratio)
        if isinstance(mlp_ratio_value_input, bool) or not isinstance(
            mlp_ratio_value_input,
            (int, float),
        ):
            raise TypeError("mlp_ratio must be a real number")
        mlp_ratio_value = float(mlp_ratio_value_input)
        if not math.isfinite(mlp_ratio_value) or mlp_ratio_value <= 0.0:
            raise ValueError("mlp_ratio must be a finite positive number")
        mlp_channels = int(channels * mlp_ratio_value)
        if mlp_channels <= 0:
            raise ValueError("mlp_ratio produces an empty MLP")

        self.channels = channels
        self.attention_head_dim = attention_head_dim
        self.num_heads = channels // attention_head_dim
        self.attention_dropout = dropout_value

        self.attention_norm = nn.LayerNorm(channels)
        self.query_key_value = nn.Linear(channels, channels * 3)
        self.attention_projection = nn.Linear(channels, channels)
        self.attention_output_dropout = nn.Dropout(dropout_value)

        self.mlp_norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_channels),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(mlp_channels, channels),
            nn.Dropout(dropout_value),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and an MLP to a batch of spatial tokens."""

        tokens_value = cast(object, tokens)
        if not isinstance(tokens_value, torch.Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        tokens = tokens_value
        if tokens.ndim != 3:
            raise ValueError("tokens must be a 3D batch-first tensor")
        if not tokens.is_floating_point():
            raise TypeError("tokens must have a floating-point dtype")
        if tokens.shape[-1] != self.channels:
            raise ValueError(
                f"tokens must have feature dimension {self.channels}, "
                f"got {tokens.shape[-1]}"
            )

        batch_size, token_count, channels = tokens.shape
        normalized = self.attention_norm(tokens)
        query_key_value = self.query_key_value(normalized)
        query_key_value = query_key_value.reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.attention_head_dim,
        )
        query_key_value = query_key_value.permute(2, 0, 3, 1, 4)
        query, key, value = query_key_value.unbind(dim=0)
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            token_count,
            channels,
        )
        tokens = tokens + self.attention_output_dropout(
            self.attention_projection(attended)
        )
        return tokens + self.mlp(self.mlp_norm(tokens))


class SpatialTransformer(nn.Module):
    """Apply a stack of Transformer layers over an NCHW feature map."""

    def __init__(
        self,
        channels: int,
        attention_head_dim: int,
        depth: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        validate_adm_positive_integer("channels", channels)
        validate_adm_positive_integer(
            "attention_head_dim",
            attention_head_dim,
        )
        validate_adm_positive_integer("depth", depth)
        if channels % attention_head_dim != 0:
            raise ValueError("channels must be divisible by attention_head_dim")

        self.channels = channels
        self.layers = nn.ModuleList(
            [
                SpatialTransformerLayer(
                    channels,
                    attention_head_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Flatten a feature map to tokens, transform it, and restore NCHW."""

        state = _validated_spatial_state(
            cast(object, state),
            channels=self.channels,
        )

        batch_size, channels, height, width = state.shape
        tokens = state.flatten(2).transpose(1, 2)
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens.transpose(1, 2).reshape(
            batch_size,
            channels,
            height,
            width,
        )


__all__ = [
    "ADMDownsample",
    "ADMResidualBlock",
    "ADMUpsample",
    "ResidualResampleKind",
    "SpatialTransformer",
    "SpatialTransformerLayer",
]

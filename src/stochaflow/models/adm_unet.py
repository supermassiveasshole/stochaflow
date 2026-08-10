"""Canonical ADM U-Net with optional class conditioning."""

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from stochaflow.models.adm_blocks import (
    ADMAttentionBlock,
    ADMResidualBlock,
    adm_group_norm_groups,
    validate_adm_dropout,
    validate_adm_positive_integer,
    zero_adm_module,
)
from stochaflow.models.embeddings import sinusoidal_timestep_embedding


def _validate_integer_sequence(
    name: str,
    values: object,
    *,
    allow_empty: bool,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of integers")
    resolved = tuple(values)
    if not resolved and not allow_empty:
        raise ValueError(f"{name} must be a non-empty sequence of integers")
    return tuple(
        validate_adm_positive_integer(f"each {name} value", value) for value in resolved
    )


class ADMConditionedSequential(nn.Module):
    """Run one ADM input, middle, or output block."""

    def __init__(self, *layers: nn.Module) -> None:
        super().__init__()
        if not layers:
            raise ValueError("an ADM block must contain at least one layer")
        allowed_types = (nn.Conv2d, ADMResidualBlock, ADMAttentionBlock)
        if any(not isinstance(layer, allowed_types) for layer in layers):
            raise TypeError(
                "ADM blocks only accept convolution, residual, or attention"
            )
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        state: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Run the block, injecting the time/class embedding into ResBlocks."""

        hidden = state
        for layer in self.layers:
            if isinstance(layer, ADMResidualBlock):
                hidden = layer(hidden, embedding)
            else:
                hidden = layer(hidden)
        return hidden


class ADMUNet(nn.Module):
    """Predict denoising targets with the canonical ADM input/output graph."""

    def __init__(
        self,
        input_size: int = 128,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: Sequence[int] = (1, 1, 2, 3, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Sequence[int] = (32, 16, 8),
        attention_head_channels: int = 64,
        num_classes: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        validate_adm_positive_integer("input_size", input_size)
        validate_adm_positive_integer("in_channels", in_channels)
        validate_adm_positive_integer("out_channels", out_channels)
        validate_adm_positive_integer("base_channels", base_channels)
        if base_channels < 2:
            raise ValueError("base_channels must be at least 2")
        validate_adm_positive_integer("num_res_blocks", num_res_blocks)
        validate_adm_positive_integer(
            "attention_head_channels",
            attention_head_channels,
        )
        dropout_value = validate_adm_dropout(dropout)
        if num_classes is not None:
            validate_adm_positive_integer("num_classes", num_classes)

        multipliers = _validate_integer_sequence(
            "channel_multipliers",
            channel_multipliers,
            allow_empty=False,
        )
        if multipliers[0] != 1:
            raise ValueError("the first channel multiplier must be 1")
        resolutions = _validate_integer_sequence(
            "attention_resolutions",
            attention_resolutions,
            allow_empty=True,
        )
        if len(set(resolutions)) != len(resolutions):
            raise ValueError("attention_resolutions must not contain duplicates")

        downsample_factor = 2 ** (len(multipliers) - 1)
        if input_size % downsample_factor != 0:
            raise ValueError(
                "input_size must be divisible by the ADM downsample factor "
                f"{downsample_factor}"
            )
        level_resolutions = tuple(
            input_size // (2**level) for level in range(len(multipliers))
        )
        invalid_resolutions = sorted(set(resolutions) - set(level_resolutions))
        if invalid_resolutions:
            raise ValueError(
                "attention_resolutions must name ADM spatial levels; "
                f"invalid values: {invalid_resolutions}"
            )

        level_channels = tuple(base_channels * value for value in multipliers)
        attention_levels = set(resolutions)
        for level, (channels, resolution) in enumerate(
            zip(level_channels, level_resolutions, strict=True)
        ):
            if (
                resolution in attention_levels
                and channels % attention_head_channels != 0
            ):
                raise ValueError(
                    f"level {level} channels ({channels}) must be divisible by "
                    "attention_head_channels "
                    f"({attention_head_channels})"
                )
        if level_channels[-1] % attention_head_channels != 0:
            raise ValueError(
                "middle channels must be divisible by attention_head_channels"
            )

        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_multipliers = multipliers
        self.level_channels = level_channels
        self.level_resolutions = level_resolutions
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = resolutions
        self.attention_head_channels = attention_head_channels
        self.time_embedding_dim = base_channels * 4
        self.downsample_factor = downsample_factor
        self._num_classes = num_classes
        self._null_class_id = num_classes

        self.time_embedding = nn.Sequential(
            nn.Linear(base_channels, self.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )
        self.class_embedding = (
            None
            if num_classes is None
            else nn.Embedding(num_classes + 1, self.time_embedding_dim)
        )

        input_blocks: list[ADMConditionedSequential] = []
        input_block_channels: list[int] = []
        input_block_resolutions: list[int] = []
        channels = base_channels
        current_resolution = input_size
        input_blocks.append(
            ADMConditionedSequential(
                nn.Conv2d(
                    in_channels,
                    base_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
        )
        input_block_channels.append(channels)
        input_block_resolutions.append(current_resolution)

        for level, stage_channels in enumerate(level_channels):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [
                    ADMResidualBlock(
                        channels,
                        stage_channels,
                        self.time_embedding_dim,
                        dropout_value,
                    )
                ]
                channels = stage_channels
                if current_resolution in attention_levels:
                    layers.append(
                        ADMAttentionBlock(
                            channels,
                            attention_head_channels,
                        )
                    )
                input_blocks.append(ADMConditionedSequential(*layers))
                input_block_channels.append(channels)
                input_block_resolutions.append(current_resolution)

            if level < len(level_channels) - 1:
                input_blocks.append(
                    ADMConditionedSequential(
                        ADMResidualBlock(
                            channels,
                            channels,
                            self.time_embedding_dim,
                            dropout_value,
                            resample="down",
                        )
                    )
                )
                current_resolution //= 2
                input_block_channels.append(channels)
                input_block_resolutions.append(current_resolution)

        self.input_blocks = nn.ModuleList(input_blocks)
        self.input_block_channels = tuple(input_block_channels)
        self.input_block_resolutions = tuple(input_block_resolutions)

        self.middle_block = ADMConditionedSequential(
            ADMResidualBlock(
                channels,
                channels,
                self.time_embedding_dim,
                dropout_value,
            ),
            ADMAttentionBlock(
                channels,
                attention_head_channels,
            ),
            ADMResidualBlock(
                channels,
                channels,
                self.time_embedding_dim,
                dropout_value,
            ),
        )

        remaining_skip_channels = list(input_block_channels)
        remaining_skip_resolutions = list(input_block_resolutions)
        output_blocks: list[ADMConditionedSequential] = []
        output_skip_channels: list[int] = []
        output_block_input_resolutions: list[int] = []
        output_block_output_resolutions: list[int] = []
        for level in reversed(range(len(level_channels))):
            stage_channels = level_channels[level]
            for block_index in range(num_res_blocks + 1):
                skip_channels = remaining_skip_channels.pop()
                skip_resolution = remaining_skip_resolutions.pop()
                if skip_resolution != current_resolution:
                    raise RuntimeError(
                        "ADM topology construction produced a mismatched skip "
                        "resolution"
                    )
                layers = [
                    ADMResidualBlock(
                        channels + skip_channels,
                        stage_channels,
                        self.time_embedding_dim,
                        dropout_value,
                    )
                ]
                channels = stage_channels
                if current_resolution in attention_levels:
                    layers.append(
                        ADMAttentionBlock(
                            channels,
                            attention_head_channels,
                        )
                    )
                output_resolution = current_resolution
                if level > 0 and block_index == num_res_blocks:
                    layers.append(
                        ADMResidualBlock(
                            channels,
                            channels,
                            self.time_embedding_dim,
                            dropout_value,
                            resample="up",
                        )
                    )
                    output_resolution *= 2
                output_blocks.append(ADMConditionedSequential(*layers))
                output_skip_channels.append(skip_channels)
                output_block_input_resolutions.append(current_resolution)
                output_block_output_resolutions.append(output_resolution)
                current_resolution = output_resolution

        if remaining_skip_channels or remaining_skip_resolutions:
            raise RuntimeError("ADM topology construction left unmatched skips")
        if current_resolution != input_size:
            raise RuntimeError("ADM topology construction did not restore input size")

        self.output_blocks = nn.ModuleList(output_blocks)
        self.output_skip_channels = tuple(output_skip_channels)
        self.output_block_input_resolutions = tuple(output_block_input_resolutions)
        self.output_block_output_resolutions = tuple(output_block_output_resolutions)
        self.output_norm = nn.GroupNorm(
            adm_group_norm_groups(channels),
            channels,
        )
        self.output_activation = nn.SiLU()
        self.output_projection = nn.Conv2d(
            channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        zero_adm_module(self.output_projection)

    @property
    def input_projection(self) -> nn.Conv2d:
        """Return the initial image projection."""

        input_block = cast(ADMConditionedSequential, self.input_blocks[0])
        projection = input_block.layers[0]
        if not isinstance(projection, nn.Conv2d):
            raise TypeError("the first ADM input block is not a convolution")
        return projection

    @property
    def num_classes(self) -> int | None:
        """Return the number of real classes, or null for unconditional ADM."""

        return self._num_classes

    @property
    def null_class_id(self) -> int | None:
        """Return the classifier-free null id, or null when unconditional."""

        return self._null_class_id

    def predict_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return one prediction for explicit real or null class labels."""

        if self._num_classes is None:
            raise RuntimeError("unconditional ADM has no class-conditioned prediction")
        return self._predict(
            state,
            model_time,
            class_labels,
            validate_values=True,
        )

    def predict_prevalidated_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Predict after a composition boundary has certified value ranges."""

        if self._num_classes is None:
            raise RuntimeError("unconditional ADM has no class-conditioned prediction")
        return self._predict(
            state,
            model_time,
            class_labels,
            validate_values=False,
        )

    def forward(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict a denoising target from state, discrete time, and condition."""

        return self._predict(
            state,
            model_time,
            class_labels,
            validate_values=True,
        )

    def _predict(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor | None,
        *,
        validate_values: bool,
    ) -> torch.Tensor:
        self._validate_forward_inputs(
            state,
            model_time,
            class_labels,
            validate_values=validate_values,
        )
        embedding = self._conditioning_embedding(
            state,
            model_time,
            class_labels,
        )

        hidden = state
        skips: list[torch.Tensor] = []
        for block in self.input_blocks:
            hidden = block(hidden, embedding)
            skips.append(hidden)

        hidden = self.middle_block(hidden, embedding)

        for block in self.output_blocks:
            skip = skips.pop()
            if hidden.shape[2:] != skip.shape[2:]:
                raise RuntimeError("mirrored ADM features have incompatible shapes")
            hidden = block(torch.cat([hidden, skip], dim=1), embedding)

        if skips:
            raise RuntimeError("ADM skip features were not consumed")
        # Keep the canonical ADM output head at the input precision. Under
        # mixed precision the U-Net torso may produce BF16/FP16 activations,
        # while guided-diffusion restores the original input dtype before the
        # final normalization and projection head.
        hidden = hidden.to(dtype=state.dtype)
        output = self.output_projection(
            self.output_activation(self.output_norm(hidden))
        )
        expected_shape = (
            state.shape[0],
            self.out_channels,
            self.input_size,
            self.input_size,
        )
        if output.shape != expected_shape:
            raise RuntimeError(
                f"ADMUNet produced shape {tuple(output.shape)}, "
                f"expected {expected_shape}"
            )
        return output

    def _conditioning_embedding(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor | None,
    ) -> torch.Tensor:
        base_embedding = sinusoidal_timestep_embedding(
            model_time,
            self.base_channels,
        )
        time_weight = cast(nn.Linear, self.time_embedding[0]).weight
        base_embedding = base_embedding.to(dtype=time_weight.dtype)
        embedding = self.time_embedding(base_embedding)
        if self.class_embedding is not None:
            if class_labels is None:
                raise RuntimeError("conditional ADM is missing class labels")
            class_embedding = self.class_embedding(class_labels.to(dtype=torch.long))
            embedding = embedding + class_embedding.to(dtype=embedding.dtype)
        if embedding.device != state.device:
            raise RuntimeError("conditioning embedding was created on the wrong device")
        return embedding

    def _validate_forward_inputs(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor | None,
        *,
        validate_values: bool,
    ) -> None:
        state_value = cast(object, state)
        if not isinstance(state_value, torch.Tensor):
            raise TypeError("state must be a torch.Tensor")
        state = state_value
        if state.ndim != 4:
            raise ValueError("state must be a 4D NCHW tensor")
        if not state.is_floating_point():
            raise TypeError("state must have a floating-point dtype")
        if state.shape[0] <= 0:
            raise ValueError("state batch dimension must be positive")
        if state.shape[1] != self.in_channels:
            raise ValueError(
                f"state must have {self.in_channels} channels, got {state.shape[1]}"
            )
        expected_spatial_shape = (self.input_size, self.input_size)
        if state.shape[2:] != expected_spatial_shape:
            raise ValueError(
                "state spatial dimensions must equal configured input_size "
                f"{self.input_size}"
            )
        if state.device != self.input_projection.weight.device:
            raise ValueError("state and model parameters must be on the same device")

        model_time_value = cast(object, model_time)
        if not isinstance(model_time_value, torch.Tensor):
            raise TypeError("model_time must be a torch.Tensor")
        model_time = model_time_value
        if model_time.ndim != 1:
            raise ValueError("model_time must be a 1D tensor")
        if model_time.shape[0] != state.shape[0]:
            raise ValueError("model_time batch dimension must match state")
        if model_time.dtype not in {torch.int32, torch.int64}:
            raise TypeError("model_time must have an int32 or int64 dtype")
        if model_time.device != state.device:
            raise ValueError("model_time and state must be on the same device")

        if self._num_classes is None:
            if class_labels is not None:
                raise ValueError("unconditional ADM does not accept class_labels")
        else:
            class_labels_value = cast(object, class_labels)
            if not isinstance(class_labels_value, torch.Tensor):
                raise TypeError("class_labels must be a torch.Tensor")
            class_labels = class_labels_value
            if class_labels.ndim != 1:
                raise ValueError("class_labels must be a 1D tensor")
            if class_labels.shape[0] != state.shape[0]:
                raise ValueError("class_labels batch dimension must match state")
            if class_labels.dtype not in {torch.int32, torch.int64}:
                raise TypeError("class_labels must have an int32 or int64 dtype")
            if class_labels.device != state.device:
                raise ValueError("class_labels and state must be on the same device")

        if not validate_values:
            return
        invalid_time = torch.any(model_time < 0)
        invalid_labels = torch.tensor(False, device=state.device)
        if class_labels is not None:
            assert self._null_class_id is not None
            invalid_labels = torch.any(
                (class_labels < 0) | (class_labels > self._null_class_id)
            )
        invalid_value_flags = torch.stack((invalid_time, invalid_labels)).tolist()
        if invalid_value_flags[0]:
            raise ValueError("model_time values must be non-negative")
        if invalid_value_flags[1]:
            raise ValueError(
                "class_labels values must be real class identifiers or "
                f"the null class identifier {self._null_class_id}"
            )


__all__ = ["ADMUNet"]

"""Class-conditioned ADM-style UNet with low-resolution Transformers."""

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from stochaflow.models.adm_blocks import (
    ADMDownsample,
    ADMResidualBlock,
    ADMUpsample,
    SpatialTransformer,
    adm_group_norm_groups,
    validate_adm_dropout,
    validate_adm_positive_integer,
    zero_adm_module,
)
from stochaflow.models.conditioning import ClassConditionalDenoiser
from stochaflow.models.embeddings import sinusoidal_timestep_embedding
from stochaflow.utils.registry import REGISTRIES


def _validate_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_integer_sequence(
    name: str,
    values: object,
    *,
    allow_zero: bool,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of integers")
    resolved = tuple(values)
    if not resolved:
        raise ValueError(f"{name} must be a non-empty sequence of integers")
    validator = (
        _validate_nonnegative_int if allow_zero else validate_adm_positive_integer
    )
    return tuple(validator(f"each {name} value", value) for value in resolved)


def _validate_boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


@REGISTRIES.models.register("adm_unet")
class ADMUNet(nn.Module, ClassConditionalDenoiser):
    """Predict class-conditioned denoising targets with an ADM-style UNet."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: Sequence[int] = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        transformer_depths: Sequence[int] = (0, 0, 1, 2),
        middle_transformer_depth: int = 1,
        attention_head_dim: int = 64,
        time_embedding_dim: int = 512,
        num_classes: int = 3,
        dropout: float = 0.0,
        *,
        scale_shift_norm: bool = True,
        residual_resampling: bool = True,
        zero_init_residual: bool = True,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        validate_adm_positive_integer("in_channels", in_channels)
        validate_adm_positive_integer("out_channels", out_channels)
        if out_channels != in_channels:
            raise ValueError(
                "class-conditioned ADM denoising requires out_channels "
                "to equal in_channels"
            )
        validate_adm_positive_integer("base_channels", base_channels)
        if base_channels < 2:
            raise ValueError("base_channels must be at least 2")
        validate_adm_positive_integer("num_res_blocks", num_res_blocks)
        _validate_nonnegative_int(
            "middle_transformer_depth",
            middle_transformer_depth,
        )
        validate_adm_positive_integer(
            "attention_head_dim",
            attention_head_dim,
        )
        validate_adm_positive_integer(
            "time_embedding_dim",
            time_embedding_dim,
        )
        if time_embedding_dim < 2:
            raise ValueError("time_embedding_dim must be at least 2")
        validate_adm_positive_integer("num_classes", num_classes)
        dropout_value = validate_adm_dropout(dropout)
        for name, value in (
            ("scale_shift_norm", scale_shift_norm),
            ("residual_resampling", residual_resampling),
            ("zero_init_residual", zero_init_residual),
            ("zero_init_output", zero_init_output),
        ):
            _validate_boolean(name, value)

        multipliers = _validate_integer_sequence(
            "channel_multipliers",
            channel_multipliers,
            allow_zero=False,
        )
        depths = _validate_integer_sequence(
            "transformer_depths",
            transformer_depths,
            allow_zero=True,
        )
        if len(depths) != len(multipliers):
            raise ValueError(
                "transformer_depths must have one value per channel multiplier"
            )

        level_channels = tuple(base_channels * value for value in multipliers)
        for level, (channels, depth) in enumerate(
            zip(level_channels, depths, strict=True)
        ):
            if depth > 0 and channels % attention_head_dim != 0:
                raise ValueError(
                    f"stage {level} channels ({channels}) must be divisible by "
                    f"attention_head_dim ({attention_head_dim})"
                )
        if (
            middle_transformer_depth > 0
            and level_channels[-1] % attention_head_dim != 0
        ):
            raise ValueError("middle channels must be divisible by attention_head_dim")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_multipliers = multipliers
        self.level_channels = level_channels
        self.num_res_blocks = num_res_blocks
        self.transformer_depths = depths
        self.middle_transformer_depth = middle_transformer_depth
        self.attention_head_dim = attention_head_dim
        self.time_embedding_dim = time_embedding_dim
        self.residual_resampling = residual_resampling
        self._num_classes = num_classes
        self._null_class_id = num_classes
        self.downsample_factor = 2 ** (len(level_channels) - 1)

        self.time_embedding = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )
        self.class_embedding = nn.Embedding(
            num_classes + 1,
            time_embedding_dim,
        )
        self.input_projection = nn.Conv2d(
            in_channels,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        self.down_blocks = nn.ModuleList()
        self.down_transformers = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        channels = base_channels
        for level, stage_channels in enumerate(level_channels):
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ADMResidualBlock(
                        channels,
                        stage_channels,
                        time_embedding_dim,
                        dropout_value,
                        scale_shift_norm=scale_shift_norm,
                        zero_init_residual=zero_init_residual,
                    )
                )
                channels = stage_channels
            self.down_blocks.append(blocks)
            self.down_transformers.append(
                self._build_transformer(
                    channels,
                    depths[level],
                    attention_head_dim,
                    dropout_value,
                )
            )

            if level < len(level_channels) - 1:
                next_channels = level_channels[level + 1]
                if residual_resampling:
                    downsample: nn.Module = ADMResidualBlock(
                        channels,
                        next_channels,
                        time_embedding_dim,
                        dropout_value,
                        scale_shift_norm=scale_shift_norm,
                        resample="down",
                        zero_init_residual=zero_init_residual,
                    )
                else:
                    downsample = ADMDownsample(channels, next_channels)
                self.downsamples.append(downsample)
                channels = next_channels

        self.middle_block_before = ADMResidualBlock(
            channels,
            channels,
            time_embedding_dim,
            dropout_value,
            scale_shift_norm=scale_shift_norm,
            zero_init_residual=zero_init_residual,
        )
        self.middle_transformer = self._build_transformer(
            channels,
            middle_transformer_depth,
            attention_head_dim,
            dropout_value,
        )
        self.middle_block_after = ADMResidualBlock(
            channels,
            channels,
            time_embedding_dim,
            dropout_value,
            scale_shift_norm=scale_shift_norm,
            zero_init_residual=zero_init_residual,
        )

        self.up_blocks = nn.ModuleList()
        self.up_transformers = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for stage_index in reversed(range(len(level_channels))):
            stage_channels = level_channels[stage_index]
            blocks = nn.ModuleList()
            blocks.append(
                ADMResidualBlock(
                    channels + stage_channels,
                    stage_channels,
                    time_embedding_dim,
                    dropout_value,
                    scale_shift_norm=scale_shift_norm,
                    zero_init_residual=zero_init_residual,
                )
            )
            channels = stage_channels
            for _ in range(num_res_blocks - 1):
                blocks.append(
                    ADMResidualBlock(
                        channels,
                        stage_channels,
                        time_embedding_dim,
                        dropout_value,
                        scale_shift_norm=scale_shift_norm,
                        zero_init_residual=zero_init_residual,
                    )
                )
            self.up_blocks.append(blocks)
            self.up_transformers.append(
                self._build_transformer(
                    channels,
                    depths[stage_index],
                    attention_head_dim,
                    dropout_value,
                )
            )

            if stage_index > 0:
                next_channels = level_channels[stage_index - 1]
                if residual_resampling:
                    upsample: nn.Module = ADMResidualBlock(
                        channels,
                        next_channels,
                        time_embedding_dim,
                        dropout_value,
                        scale_shift_norm=scale_shift_norm,
                        resample="up",
                        zero_init_residual=zero_init_residual,
                    )
                else:
                    upsample = ADMUpsample(channels, next_channels)
                self.upsamples.append(upsample)
                channels = next_channels

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
        if zero_init_output:
            zero_adm_module(self.output_projection)

    @staticmethod
    def _build_transformer(
        channels: int,
        depth: int,
        attention_head_dim: int,
        dropout: float,
    ) -> nn.Module:
        if depth == 0:
            return nn.Identity()
        return SpatialTransformer(
            channels,
            attention_head_dim,
            depth,
            dropout=dropout,
        )

    @property
    def num_classes(self) -> int:
        """Return the number of real class identifiers."""

        return self._num_classes

    @property
    def null_class_id(self) -> int:
        """Return the reserved identifier used for unconditional prediction."""

        return self._null_class_id

    def predict_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return one prediction for explicit real or null class labels."""

        return self.forward(state, model_time, class_labels)

    def predict_prevalidated_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Predict after a composition boundary has certified value ranges."""

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
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a denoising target from state, discrete time, and class."""

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
        class_labels: torch.Tensor,
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

        hidden = self.input_projection(state)
        skips: list[torch.Tensor] = []
        for level, blocks_module in enumerate(self.down_blocks):
            blocks = cast(nn.ModuleList, blocks_module)
            for block_module in blocks:
                block = cast(ADMResidualBlock, block_module)
                hidden = block(hidden, embedding)
            hidden = self.down_transformers[level](hidden)
            skips.append(hidden)
            if level < len(self.downsamples):
                downsample = self.downsamples[level]
                if self.residual_resampling:
                    hidden = cast(ADMResidualBlock, downsample)(
                        hidden,
                        embedding,
                    )
                else:
                    hidden = cast(ADMDownsample, downsample)(hidden)

        hidden = self.middle_block_before(hidden, embedding)
        hidden = self.middle_transformer(hidden)
        hidden = self.middle_block_after(hidden, embedding)

        for level, blocks_module in enumerate(self.up_blocks):
            skip = skips.pop()
            if hidden.shape[2:] != skip.shape[2:]:
                raise RuntimeError("mirrored UNet features have incompatible shapes")
            hidden = torch.cat([hidden, skip], dim=1)
            blocks = cast(nn.ModuleList, blocks_module)
            for block_module in blocks:
                block = cast(ADMResidualBlock, block_module)
                hidden = block(hidden, embedding)
            hidden = self.up_transformers[level](hidden)
            if level < len(self.upsamples):
                upsample = self.upsamples[level]
                if self.residual_resampling:
                    hidden = cast(ADMResidualBlock, upsample)(
                        hidden,
                        embedding,
                    )
                else:
                    hidden = cast(ADMUpsample, upsample)(hidden)

        if skips:
            raise RuntimeError("UNet skip features were not consumed")
        output = self.output_projection(
            self.output_activation(self.output_norm(hidden))
        )
        expected_shape = (
            state.shape[0],
            self.out_channels,
            state.shape[2],
            state.shape[3],
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
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        base_embedding = sinusoidal_timestep_embedding(
            model_time,
            self.time_embedding_dim,
        )
        time_weight = cast(nn.Linear, self.time_embedding[0]).weight
        base_embedding = base_embedding.to(dtype=time_weight.dtype)
        time_embedding = self.time_embedding(base_embedding)
        class_embedding = self.class_embedding(class_labels.to(dtype=torch.long))
        class_embedding = class_embedding.to(dtype=time_embedding.dtype)
        embedding = time_embedding + class_embedding
        if embedding.device != state.device:
            raise RuntimeError("conditioning embedding was created on the wrong device")
        return embedding

    def _validate_forward_inputs(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
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
        if state.shape[2] <= 0 or state.shape[3] <= 0:
            raise ValueError("state spatial dimensions must be positive")
        if (
            state.shape[2] % self.downsample_factor != 0
            or state.shape[3] % self.downsample_factor != 0
        ):
            raise ValueError(
                "state spatial dimensions must be divisible by "
                f"{self.downsample_factor}"
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
        invalid_labels = (class_labels < 0) | (class_labels > self.null_class_id)
        invalid_value_flags = torch.stack(
            (
                torch.any(model_time < 0),
                torch.any(invalid_labels),
            )
        ).tolist()
        if invalid_value_flags[0]:
            raise ValueError("model_time values must be non-negative")
        if invalid_value_flags[1]:
            raise ValueError(
                "class_labels values must be real class identifiers or "
                f"the null class identifier {self.null_class_id}"
            )


__all__ = ["ADMUNet"]

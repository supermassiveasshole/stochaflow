"""Diffusion Transformer backbone with explicit class conditioning."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn
from torch.nn import functional

from stochaflow.models.embeddings import sinusoidal_timestep_embedding
from stochaflow.utils.registry import REGISTRIES

_LABEL_DTYPES = frozenset((torch.int32, torch.int64))
_TIME_DTYPES = frozenset(
    (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
)


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _spatial_size(
    value: int | tuple[int, int] | list[int],
) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        dimension = _positive_integer(value, name="input_size")
        return dimension, dimension
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("input_size must be a positive integer or [height, width]")
    height = _positive_integer(value[0], name="input_size[0]")
    width = _positive_integer(value[1], name="input_size[1]")
    return height, width


def _one_dimensional_position_embedding(
    coordinates: torch.Tensor,
    embedding_dim: int,
) -> torch.Tensor:
    frequency_indices = torch.arange(
        embedding_dim // 2,
        dtype=torch.float64,
    )
    frequencies = torch.exp(
        -math.log(10_000.0) * frequency_indices / (embedding_dim // 2)
    )
    phases = coordinates.to(torch.float64).reshape(-1, 1) * frequencies.reshape(
        1,
        -1,
    )
    return torch.cat((torch.sin(phases), torch.cos(phases)), dim=1)


def _two_dimensional_position_embedding(
    grid_size: tuple[int, int],
    hidden_size: int,
) -> torch.Tensor:
    if hidden_size % 4 != 0:
        raise ValueError(
            "hidden_size must be divisible by 4 for fixed 2D position embeddings"
        )
    height, width = grid_size
    rows, columns = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    coordinate_embedding_dim = hidden_size // 2
    row_embedding = _one_dimensional_position_embedding(
        rows.reshape(-1),
        coordinate_embedding_dim,
    )
    column_embedding = _one_dimensional_position_embedding(
        columns.reshape(-1),
        coordinate_embedding_dim,
    )
    return torch.cat((row_embedding, column_embedding), dim=1).to(torch.float32)


def _modulate(
    value: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return value * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTTimestepEmbedding(nn.Module):
    """Project sinusoidal model times into the DiT conditioning width."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.SiLU()
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, model_time: torch.Tensor) -> torch.Tensor:
        """Embed one model time per sample."""

        embedding = sinusoidal_timestep_embedding(model_time, self.hidden_size)
        embedding = embedding.to(dtype=self.input_projection.weight.dtype)
        embedding = self.input_projection(embedding)
        return self.output_projection(self.activation(embedding))


class DiTSelfAttention(nn.Module):
    """Multi-head self-attention implemented with PyTorch SDPA."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query_key_value = nn.Linear(hidden_size, hidden_size * 3)
        self.output_projection = nn.Linear(hidden_size, hidden_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply non-causal self-attention to a token sequence."""

        batch_size, sequence_length, hidden_size = tokens.shape
        query_key_value = self.query_key_value(tokens)
        query_key_value = query_key_value.reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = query_key_value.unbind(dim=0)
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            hidden_size,
        )
        return self.output_projection(attended)


class DiTFeedForward(nn.Module):
    """Two-layer Transformer feed-forward network."""

    def __init__(self, hidden_size: int, mlp_hidden_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(hidden_size, mlp_hidden_size)
        self.activation = nn.GELU(approximate="tanh")
        self.output_projection = nn.Linear(mlp_hidden_size, hidden_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply the token-wise feed-forward network."""

        hidden = self.activation(self.input_projection(tokens))
        return self.output_projection(hidden)


class DiTBlock(nn.Module):
    """Transformer block conditioned through adaLN-Zero modulation."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_hidden_size: int,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.attention = DiTSelfAttention(hidden_size, num_heads)
        self.feed_forward_norm = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.feed_forward = DiTFeedForward(hidden_size, mlp_hidden_size)
        self.modulation_activation = nn.SiLU()
        self.modulation_projection = nn.Linear(hidden_size, hidden_size * 6)

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one adaLN-Zero attention and feed-forward update."""

        modulation = self.modulation_projection(
            self.modulation_activation(condition)
        )
        (
            attention_shift,
            attention_scale,
            attention_gate,
            feed_forward_shift,
            feed_forward_scale,
            feed_forward_gate,
        ) = modulation.chunk(6, dim=1)
        attention_input = _modulate(
            self.attention_norm(tokens),
            attention_shift,
            attention_scale,
        )
        tokens = tokens + attention_gate.unsqueeze(1) * self.attention(
            attention_input
        )
        feed_forward_input = _modulate(
            self.feed_forward_norm(tokens),
            feed_forward_shift,
            feed_forward_scale,
        )
        return tokens + feed_forward_gate.unsqueeze(1) * self.feed_forward(
            feed_forward_input
        )


class DiTFinalLayer(nn.Module):
    """adaLN-modulated projection from tokens back to image patches."""

    def __init__(self, hidden_size: int, patch_output_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(
            hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.modulation_activation = nn.SiLU()
        self.modulation_projection = nn.Linear(hidden_size, hidden_size * 2)
        self.output_projection = nn.Linear(hidden_size, patch_output_size)

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Project conditioned tokens into flattened output patches."""

        shift, scale = self.modulation_projection(
            self.modulation_activation(condition)
        ).chunk(2, dim=1)
        tokens = _modulate(self.norm(tokens), shift, scale)
        return self.output_projection(tokens)


@REGISTRIES.models.register("dit")
class DiT(nn.Module):
    """Fixed-variance class-conditional Diffusion Transformer denoiser."""

    position_embedding: torch.Tensor

    def __init__(
        self,
        input_size: int | tuple[int, int] | list[int],
        patch_size: int,
        in_channels: int = 3,
        out_channels: int = 3,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int = 1_000,
    ) -> None:
        super().__init__()
        resolved_input_size = _spatial_size(input_size)
        patch_size = _positive_integer(patch_size, name="patch_size")
        in_channels = _positive_integer(in_channels, name="in_channels")
        out_channels = _positive_integer(out_channels, name="out_channels")
        hidden_size = _positive_integer(hidden_size, name="hidden_size")
        depth = _positive_integer(depth, name="depth")
        num_heads = _positive_integer(num_heads, name="num_heads")
        num_classes = _positive_integer(num_classes, name="num_classes")
        mlp_ratio_value = cast(object, mlp_ratio)
        if (
            isinstance(mlp_ratio_value, bool)
            or not isinstance(mlp_ratio_value, (int, float))
            or not math.isfinite(mlp_ratio_value)
            or mlp_ratio_value <= 0
        ):
            raise ValueError("mlp_ratio must be a finite positive number")
        if any(dimension % patch_size != 0 for dimension in resolved_input_size):
            raise ValueError("input_size dimensions must be divisible by patch_size")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if hidden_size % 4 != 0:
            raise ValueError(
                "hidden_size must be divisible by 4 for fixed 2D position embeddings"
            )
        if out_channels != in_channels:
            raise ValueError(
                "fixed-variance DiT requires out_channels to equal in_channels"
            )

        mlp_hidden_size = int(hidden_size * float(mlp_ratio_value))
        if mlp_hidden_size <= 0:
            raise ValueError("mlp_ratio produces an empty feed-forward width")

        self.input_size = resolved_input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self._num_classes = num_classes
        self.grid_size = (
            resolved_input_size[0] // patch_size,
            resolved_input_size[1] // patch_size,
        )

        self.patch_projection = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        position_embedding = _two_dimensional_position_embedding(
            self.grid_size,
            hidden_size,
        ).unsqueeze(0)
        self.register_buffer(
            "position_embedding",
            position_embedding,
            persistent=True,
        )
        self.time_embedding = DiTTimestepEmbedding(hidden_size)
        self.class_embedding = nn.Embedding(num_classes + 1, hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_hidden_size,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = DiTFinalLayer(
            hidden_size,
            patch_size * patch_size * out_channels,
        )
        self._initialize_weights()

    @property
    def num_classes(self) -> int:
        """Return the number of real class identifiers."""

        return self._num_classes

    @property
    def null_class_id(self) -> int:
        """Return the reserved unconditional class identifier."""

        return self.num_classes

    def forward(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict with explicit labels or the null class when labels are omitted."""

        state, model_time = self._validate_state_and_time(state, model_time)
        if class_labels is None:
            class_labels = torch.full(
                (state.shape[0],),
                self.null_class_id,
                dtype=torch.long,
                device=state.device,
            )
        class_labels = self._validate_class_labels(class_labels, state)
        return self._predict_validated(state, model_time, class_labels)

    def predict_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return a prediction for real or explicitly reserved null labels."""

        state, model_time = self._validate_state_and_time(state, model_time)
        class_labels = self._validate_class_labels(class_labels, state)
        return self._predict_validated(state, model_time, class_labels)

    def predict_prevalidated_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Predict after a composition boundary has certified label values."""

        state, model_time = self._validate_state_and_time(state, model_time)
        class_labels = self._validate_class_labels(
            class_labels,
            state,
            validate_values=False,
        )
        return self._predict_validated(state, model_time, class_labels)

    def _predict_validated(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.patch_projection(state).flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding.to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        condition = self.time_embedding(model_time)
        class_condition = self.class_embedding(class_labels)
        condition = condition + class_condition.to(dtype=condition.dtype)
        for block in self.blocks:
            tokens = block(tokens, condition)
        patches = self.final_layer(tokens, condition)
        return self._unpatchify(patches)

    def _validate_state_and_time(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_value = cast(object, state)
        if not isinstance(state_value, torch.Tensor):
            raise TypeError("DiT state must be a Tensor")
        state = state_value
        if state.ndim != 4:
            raise ValueError("DiT state must be an NCHW tensor")
        if state.shape[0] <= 0:
            raise ValueError("DiT state batch must not be empty")
        expected = (self.in_channels, *self.input_size)
        if tuple(state.shape[1:]) != expected:
            raise ValueError(
                f"DiT state shape after batch must be {expected}, "
                f"got {tuple(state.shape[1:])}"
            )
        if not torch.is_floating_point(state):
            raise TypeError("DiT state must be floating-point")
        if state.device != self.patch_projection.weight.device:
            raise ValueError("DiT state must share the model device")

        model_time_value = cast(object, model_time)
        if not isinstance(model_time_value, torch.Tensor):
            raise TypeError("DiT model_time must be a Tensor")
        model_time = model_time_value
        if model_time.ndim != 1:
            raise ValueError("DiT model_time must be a 1D tensor")
        if model_time.shape[0] != state.shape[0]:
            raise ValueError("DiT model_time batch must match the state batch")
        if model_time.device != state.device:
            raise ValueError("DiT model_time must share the state device")
        if model_time.dtype not in _TIME_DTYPES:
            raise TypeError("DiT model_time must have a real numeric dtype")
        return state, model_time

    def _validate_class_labels(
        self,
        class_labels: torch.Tensor,
        state: torch.Tensor,
        *,
        validate_values: bool = True,
    ) -> torch.Tensor:
        labels_value = cast(object, class_labels)
        if not isinstance(labels_value, torch.Tensor):
            raise TypeError("DiT class_labels must be a Tensor")
        class_labels = labels_value
        if class_labels.ndim != 1:
            raise ValueError("DiT class_labels must be a 1D tensor")
        if class_labels.shape[0] != state.shape[0]:
            raise ValueError("DiT class_labels batch must match the state batch")
        if class_labels.device != state.device:
            raise ValueError("DiT class_labels must share the state device")
        if class_labels.dtype not in _LABEL_DTYPES:
            raise TypeError("DiT class_labels must use int32 or int64")
        invalid_labels = (class_labels < 0) | (
            class_labels > self.null_class_id
        )
        if validate_values and bool(torch.any(invalid_labels)):
            raise ValueError(
                "DiT class_labels must be in "
                f"[0, {self.null_class_id}] including the null class"
            )
        return class_labels.to(dtype=torch.long)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size = patches.shape[0]
        grid_height, grid_width = self.grid_size
        patches = patches.reshape(
            batch_size,
            grid_height,
            grid_width,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(
            batch_size,
            self.out_channels,
            *self.input_size,
        )

    def _initialize_weights(self) -> None:
        def initialize_module(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize_module)
        nn.init.xavier_uniform_(
            self.patch_projection.weight.reshape(
                self.patch_projection.weight.shape[0],
                -1,
            )
        )
        if self.patch_projection.bias is not None:
            nn.init.zeros_(self.patch_projection.bias)
        nn.init.normal_(self.class_embedding.weight, std=0.02)
        nn.init.normal_(self.time_embedding.input_projection.weight, std=0.02)
        nn.init.normal_(self.time_embedding.output_projection.weight, std=0.02)
        for block_value in self.blocks:
            block = cast(DiTBlock, block_value)
            nn.init.zeros_(block.modulation_projection.weight)
            nn.init.zeros_(block.modulation_projection.bias)
        nn.init.zeros_(self.final_layer.modulation_projection.weight)
        nn.init.zeros_(self.final_layer.modulation_projection.bias)
        nn.init.zeros_(self.final_layer.output_projection.weight)
        nn.init.zeros_(self.final_layer.output_projection.bias)


__all__ = ["DiT"]

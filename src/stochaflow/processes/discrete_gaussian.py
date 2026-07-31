"""Model-free discrete Gaussian probability process."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch

from stochaflow.utils.config import ComponentConfig
from stochaflow.utils.registry import REGISTRIES

from .gaussian import (
    DiscreteGaussianDenoisingProcess,
    GaussianLogVarianceBounds,
    GaussianMarginalCoefficientSnapshot,
)
from .noise_schedules import DiscreteVPSchedule, GaussianScales


@REGISTRIES.processes.register("discrete_gaussian")
class DiscreteGaussianProcess(DiscreteGaussianDenoisingProcess):
    r"""Model-free discrete VP Gaussian process with fixed coefficients."""

    reference_alpha_bar_t: torch.Tensor
    marginal_signal_t: torch.Tensor
    marginal_noise_t: torch.Tensor
    sqrt_posterior_variance_t: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor

    def __init__(self, schedule: ComponentConfig | dict[str, object]) -> None:
        super().__init__()
        declaration = self._coerce_schedule(schedule)
        instance = REGISTRIES.noise_schedules.create(
            declaration.name, **declaration.params
        )
        if not isinstance(instance, DiscreteVPSchedule):
            raise TypeError("discrete_gaussian requires a DiscreteVPSchedule")
        parameter_names = [name for name, _ in instance.named_parameters()]
        if parameter_names:
            raise TypeError(
                "DiscreteGaussianProcess requires an immutable schedule without "
                "Parameters; found: "
                + ", ".join(parameter_names)
            )
        num_timesteps = cast(object, instance.num_timesteps)
        if isinstance(num_timesteps, bool) or not isinstance(num_timesteps, int):
            raise TypeError("schedule num_timesteps must be an integer")
        if num_timesteps <= 0:
            raise ValueError("schedule num_timesteps must be positive")
        self._num_timesteps = num_timesteps
        self._register_coefficient_snapshot(instance)

    @staticmethod
    def _coerce_schedule(schedule: object) -> ComponentConfig:
        if isinstance(schedule, ComponentConfig):
            return schedule
        if not isinstance(schedule, dict):
            raise TypeError("process schedule must be a component mapping")
        unknown = set(schedule) - {"name", "params"}
        if unknown:
            raise ValueError(
                "unknown process schedule field(s): " + ", ".join(sorted(unknown))
            )
        name = schedule.get("name")
        params = schedule.get("params", {})
        if not isinstance(name, str) or not name.strip():
            raise ValueError("process schedule.name must be a non-empty string")
        if not isinstance(params, dict):
            raise TypeError("process schedule.params must be a mapping")
        return ComponentConfig(name=name, params=dict(params))

    @property
    def num_timesteps(self) -> int:
        """Return the number of discrete forward transitions."""

        return self._num_timesteps

    @property
    def terminal_time(self) -> int:
        """Return terminal public state time ``T``."""

        return self.num_timesteps

    @property
    def clean_time(self) -> int:
        """Return clean public state time ``0``."""

        return 0

    def sample_terminal_prior(
        self,
        shape: torch.Size | tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample the standard Gaussian terminal prior."""

        return torch.randn(shape, device=device, dtype=dtype, generator=generator)

    def sample_marginal(
        self,
        clean: torch.Tensor,
        state_times: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample ``q(x_t | x_0)`` at public state times."""

        state_times = self._validate_state_times(state_times)
        if noise is None:
            noise = torch.randn(
                clean.shape,
                device=clean.device,
                dtype=clean.dtype,
                generator=generator,
            )
        if noise.shape != clean.shape:
            raise ValueError("marginal noise must match the clean sample shape")
        scales = self.marginal_scales(state_times, clean.size())
        return scales.signal * clean + scales.noise * noise, noise

    def marginal_scales(
        self, state_times: torch.Tensor, broadcast_shape: torch.Size
    ) -> GaussianScales:
        """Return process-owned marginal signal and noise scales."""

        state_times = self._validate_state_times(state_times)
        signal = self._gather(self.marginal_signal_t, state_times)
        noise = self._gather(self.marginal_noise_t, state_times)
        return GaussianScales(
            self._append_dimensions(
                signal,
                state_times=state_times,
                broadcast_shape=broadcast_shape,
            ),
            self._append_dimensions(
                noise,
                state_times=state_times,
                broadcast_shape=broadcast_shape,
            ),
        )

    def marginal_coefficient_snapshot(
        self,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
        broadcast_shape: torch.Size,
    ) -> GaussianMarginalCoefficientSnapshot:
        """Return reference-precision coefficients for selected reverse pairs.

        A subclass that replaces this process's marginal path must override
        this selected-pair capability (and learned bounds) coherently.
        """

        source_times, target_times = self._validate_selected_pair_times(
            source_times,
            target_times,
        )
        source_alpha_bar = self._append_dimensions(
            self._gather(self.reference_alpha_bar_t, source_times),
            state_times=source_times,
            broadcast_shape=broadcast_shape,
        )
        target_alpha_bar = self._append_dimensions(
            self._gather(self.reference_alpha_bar_t, target_times),
            state_times=target_times,
            broadcast_shape=broadcast_shape,
        )
        return GaussianMarginalCoefficientSnapshot(
            source_alpha_bar=source_alpha_bar,
            target_alpha_bar=target_alpha_bar,
            transition_alpha=source_alpha_bar / target_alpha_bar,
        )

    def reverse_log_variance_bounds(
        self,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
        broadcast_shape: torch.Size,
        *,
        clean_target_reference_times: (
            tuple[torch.Tensor, torch.Tensor] | None
        ) = None,
    ) -> GaussianLogVarianceBounds:
        """Return learned-range bounds for one selected reverse transition."""

        source_times, target_times = self._validate_selected_pair_times(
            source_times,
            target_times,
        )
        snapshot = self.marginal_coefficient_snapshot(
            source_times,
            target_times,
            broadcast_shape,
        )
        transition_variance = 1.0 - snapshot.transition_alpha
        posterior_variance = self._selected_pair_posterior_variance(snapshot)

        clean_target = self._append_dimensions(
            target_times == self.clean_time,
            state_times=target_times,
            broadcast_shape=broadcast_shape,
        )
        if clean_target_reference_times is not None:
            clipped_source, clipped_target = clean_target_reference_times
            clipped_source, clipped_target = self._validate_selected_pair_times(
                clipped_source,
                clipped_target,
            )
            if clipped_source.shape != source_times.shape:
                raise ValueError(
                    "clean-target variance reference times must match source times"
                )
            if bool(
                torch.any(
                    clean_target
                    & self._append_dimensions(
                        clipped_target != source_times,
                        state_times=source_times,
                        broadcast_shape=broadcast_shape,
                    )
                )
            ):
                raise ValueError(
                    "clean-target variance reference must end at the final "
                    "transition source"
                )
            clipped_snapshot = self.marginal_coefficient_snapshot(
                clipped_source,
                clipped_target,
                broadcast_shape,
            )
            clipped_lower = self._selected_pair_posterior_variance(
                clipped_snapshot
            )
        elif self.num_timesteps > 1:
            clipped_source = torch.full_like(
                source_times,
                self.clean_time + 2,
            )
            clipped_target = torch.full_like(
                target_times,
                self.clean_time + 1,
            )
            clipped_snapshot = self.marginal_coefficient_snapshot(
                clipped_source,
                clipped_target,
                broadcast_shape,
            )
            clipped_lower = self._selected_pair_posterior_variance(
                clipped_snapshot
            )
        else:
            clipped_lower = transition_variance
        posterior_variance = torch.where(
            clean_target,
            clipped_lower,
            posterior_variance,
        )
        storage_dtype = self.marginal_signal_t.dtype
        return GaussianLogVarianceBounds(
            lower=posterior_variance.log().to(dtype=storage_dtype),
            upper=transition_variance.log().to(dtype=storage_dtype),
        )

    def validate_noisy_state_times(self, state_times: torch.Tensor) -> torch.Tensor:
        """Validate source states in ``[1, T]``."""

        result = self._validate_state_times(state_times)
        if torch.any(result == 0):
            raise ValueError("source state times must lie in [1, T]")
        return result

    def _validate_selected_pair_times(
        self,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_times = self.validate_noisy_state_times(source_times)
        target_times = self._validate_state_times(target_times)
        if source_times.shape != target_times.shape:
            raise ValueError(
                "selected-pair source and target times must share a shape"
            )
        if source_times.device != target_times.device:
            raise ValueError(
                "selected-pair source and target times must share a device"
            )
        if torch.any(target_times >= source_times):
            raise ValueError(
                "selected-pair target times must be smaller than source times"
            )
        return source_times, target_times

    @staticmethod
    def _selected_pair_posterior_variance(
        snapshot: GaussianMarginalCoefficientSnapshot,
    ) -> torch.Tensor:
        return (1.0 - snapshot.transition_alpha) * (
            1.0 - snapshot.target_alpha_bar
        ) / (1.0 - snapshot.source_alpha_bar)

    def posterior_mean(
        self,
        state: torch.Tensor,
        state_times: torch.Tensor,
        clean_prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Return the adjacent DDPM posterior mean."""

        return self._at(self.posterior_mean_coef1, state_times, state.size()) * (
            clean_prediction
        ) + self._at(self.posterior_mean_coef2, state_times, state.size()) * state

    def posterior_standard_deviation(
        self, state_times: torch.Tensor, broadcast_shape: torch.Size
    ) -> torch.Tensor:
        """Return the adjacent DDPM posterior standard deviation."""

        return self._at(self.sqrt_posterior_variance_t, state_times, broadcast_shape)

    def _register_coefficient_snapshot(self, schedule: DiscreteVPSchedule) -> None:
        state_times = torch.arange(0, self.num_timesteps + 1)
        transition_times = state_times[1:]
        snapshot = schedule.coefficient_snapshot(state_times, transition_times)
        scales = snapshot.marginal_scales
        coefficients = snapshot.transition_coefficients
        signal, noise = self._validate_snapshot_tensors(
            (scales.signal, scales.noise),
            expected_shape=state_times.shape,
            labels=("marginal signal", "marginal noise"),
        )
        beta, alpha, alpha_bar, previous = self._validate_snapshot_tensors(
            (
                coefficients.beta,
                coefficients.alpha,
                coefficients.alpha_bar,
                coefficients.previous_alpha_bar,
            ),
            expected_shape=transition_times.shape,
            labels=(
                "transition beta",
                "transition alpha",
                "transition alpha_bar",
                "transition previous_alpha_bar",
            ),
        )
        if not torch.allclose(signal[:1], torch.ones_like(signal[:1])):
            raise ValueError("schedule clean marginal signal must equal one")
        if not torch.allclose(noise[:1], torch.zeros_like(noise[:1])):
            raise ValueError("schedule clean marginal noise must equal zero")
        denominator = 1.0 - alpha_bar
        if torch.any(denominator <= 0):
            raise ValueError("schedule noisy alpha_bar must be less than one")
        variance = beta * (1.0 - previous) / denominator
        if torch.any(variance < 0):
            raise ValueError("schedule posterior variance must be non-negative")
        posterior_values = self._validate_snapshot_tensors(
            (
                variance.sqrt(),
                beta * previous.sqrt() / denominator,
                alpha.sqrt() * (1.0 - previous) / denominator,
            ),
            expected_shape=transition_times.shape,
            labels=(
                "posterior standard deviation",
                "posterior mean coefficient 1",
                "posterior mean coefficient 2",
            ),
        )
        storage_dtype = snapshot.storage_dtype
        reference_alpha_bar = torch.cat(
            (torch.ones_like(alpha_bar[:1]), alpha_bar),
        )
        self.register_buffer(
            "reference_alpha_bar_t",
            reference_alpha_bar,
            persistent=False,
        )
        signal = signal.to(dtype=storage_dtype)
        noise = noise.to(dtype=storage_dtype)
        posterior_values = tuple(
            value.to(dtype=storage_dtype) for value in posterior_values
        )
        self.register_buffer("marginal_signal_t", signal)
        self.register_buffer("marginal_noise_t", noise)
        self.register_buffer("sqrt_posterior_variance_t", posterior_values[0])
        self.register_buffer("posterior_mean_coef1", posterior_values[1])
        self.register_buffer("posterior_mean_coef2", posterior_values[2])

    @staticmethod
    def _validate_snapshot_tensors(
        values: Sequence[torch.Tensor],
        *,
        expected_shape: torch.Size,
        labels: Sequence[str],
    ) -> tuple[torch.Tensor, ...]:
        snapshots: list[torch.Tensor] = []
        for value, label in zip(values, labels, strict=True):
            if value.shape != expected_shape:
                raise ValueError(
                    f"schedule {label} must have shape {tuple(expected_shape)}"
                )
            if not torch.is_floating_point(value):
                raise TypeError(f"schedule {label} must be floating-point")
            if value.requires_grad:
                raise TypeError(f"schedule {label} must not require gradients")
            if not torch.all(torch.isfinite(value)):
                raise ValueError(f"schedule {label} must contain only finite values")
            snapshots.append(value.detach().clone())
        return tuple(snapshots)

    def _validate_state_times(self, state_times: torch.Tensor) -> torch.Tensor:
        if state_times.ndim != 1:
            raise ValueError("state_times must be a 1D tensor")
        if (
            state_times.dtype == torch.bool
            or torch.is_floating_point(state_times)
            or torch.is_complex(state_times)
        ):
            raise TypeError("state_times must contain integer mathematical states")
        normalized = state_times.to(dtype=torch.long)
        if torch.any(normalized < 0) or torch.any(normalized > self.num_timesteps):
            raise ValueError("state_times must lie in [0, T]")
        return normalized

    @staticmethod
    def _append_dimensions(
        values: torch.Tensor,
        *,
        state_times: torch.Tensor,
        broadcast_shape: torch.Size,
    ) -> torch.Tensor:
        if values.shape != state_times.shape:
            raise ValueError("process coefficients must match requested state times")
        if not broadcast_shape or broadcast_shape[0] != state_times.shape[0]:
            raise ValueError("broadcast_shape batch dimension must match state_times")
        return values.reshape(
            (state_times.shape[0],) + (1,) * (len(broadcast_shape) - 1)
        )

    @staticmethod
    def _gather(values: torch.Tensor, state_times: torch.Tensor) -> torch.Tensor:
        return values.gather(0, state_times.to(values.device))

    def _at(
        self,
        values: torch.Tensor,
        state_times: torch.Tensor,
        broadcast_shape: torch.Size,
    ) -> torch.Tensor:
        state_times = self.validate_noisy_state_times(state_times)
        if not broadcast_shape or broadcast_shape[0] != state_times.shape[0]:
            raise ValueError("broadcast_shape batch dimension must match state_times")
        gathered = values.gather(0, (state_times - 1).to(values.device))
        return gathered.reshape(
            (state_times.shape[0],) + (1,) * (len(broadcast_shape) - 1)
        )


__all__ = ["DiscreteGaussianProcess"]

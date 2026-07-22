"""Differentiable Kolmogorov-vorticity residuals used by project code."""

from __future__ import annotations

import math

import torch

from .model import ConditionalDenoiser


def vorticity_residual(
    physical_state: torch.Tensor,
    model: ConditionalDenoiser,
) -> torch.Tensor:
    """Return the centered-time 2D vorticity equation residual."""

    if physical_state.ndim != 4 or physical_state.shape[1] != 3:
        raise ValueError("physics state must have shape [batch, 3, height, width]")
    height, width = physical_state.shape[-2:]
    if height != width:
        raise ValueError("spectral Kolmogorov residual requires square spatial fields")
    if height < 2:
        raise ValueError("spectral Kolmogorov residual requires width at least two")
    center = physical_state[:, 1:2]
    spectrum = torch.fft.fft2(center, dim=(-2, -1))
    modes = torch.fft.fftfreq(
        height,
        d=1.0 / height,
        device=physical_state.device,
        dtype=physical_state.dtype,
    )
    # Match the reference convention exactly: x varies along dim 2 and y along
    # dim 3, while velocity is (d psi / dy, -d psi / dx).
    kx = modes.reshape(1, 1, height, 1).expand(1, 1, height, width)
    ky = modes.reshape(1, 1, 1, width).expand(1, 1, height, width)
    laplacian_modes = kx.square() + ky.square()
    inverse_laplacian = torch.where(
        laplacian_modes == 0,
        torch.zeros_like(laplacian_modes),
        laplacian_modes.reciprocal(),
    )
    streamfunction = spectrum * inverse_laplacian
    velocity_x = torch.fft.ifft2(1j * ky * streamfunction).real
    velocity_y = torch.fft.ifft2(-1j * kx * streamfunction).real
    gradient_x = torch.fft.ifft2(1j * kx * spectrum).real
    gradient_y = torch.fft.ifft2(1j * ky * spectrum).real
    laplacian = torch.fft.ifft2(-laplacian_modes * spectrum).real
    time_derivative = (
        physical_state[:, 2:3] - physical_state[:, 0:1]
    ) / (2.0 * model.time_delta)
    coordinate = torch.arange(
        height,
        device=physical_state.device,
        dtype=physical_state.dtype,
    ) * (2.0 * math.pi / height)
    forcing = model.forcing_amplitude.to(dtype=physical_state.dtype) * torch.cos(
        model.forcing_wavenumber.to(dtype=physical_state.dtype) * coordinate
    )
    forcing = forcing.reshape(1, 1, 1, width)
    advection = velocity_x * gradient_x + velocity_y * gradient_y
    return (
        time_derivative
        + advection
        - laplacian / model.reynolds_number.to(dtype=physical_state.dtype)
        + model.linear_damping.to(dtype=physical_state.dtype) * center
        - forcing
    )


def correction_residual_loss(
    physical_state: torch.Tensor,
    model: ConditionalDenoiser,
) -> torch.Tensor:
    """Return the energy-normalized residual used for step correction."""

    residual = vorticity_residual(physical_state, model)
    center_energy = physical_state[:, 1].square().sum(dim=(-2, -1))
    residual_energy = residual.square().sum(dim=(-3, -2, -1))
    return (residual_energy / (center_energy + 1.0e-6)).mean()


def conditioning_residual_loss(
    physical_state: torch.Tensor,
    model: ConditionalDenoiser,
) -> torch.Tensor:
    """Return the plain residual MSE used as a denoiser condition."""

    return vorticity_residual(physical_state, model).square().mean()


def _gradient(
    normalized_state: torch.Tensor,
    model: ConditionalDenoiser,
    *,
    strength: float,
    normalized_energy: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a detached reference-compatible guidance direction.

    The local grad-enabled block deliberately works inside the sampling runtime's
    outer ``no_grad`` scope. The returned direction never retains the residual
    graph, so denoiser training does not introduce second-order derivatives.
    """

    if strength < 0:
        raise ValueError("physics gradient strength must be non-negative")
    with torch.enable_grad():
        physical = model.denormalize(normalized_state.detach()).requires_grad_(True)
        loss = (
            correction_residual_loss(physical, model)
            if normalized_energy
            else conditioning_residual_loss(physical, model)
        )
        gradient = torch.autograd.grad(loss, physical, create_graph=False)[0]
        direction = gradient / model.normalization_scale.to(dtype=gradient.dtype)
    return direction.detach() * strength, loss.detach()


def conditioning_gradient(
    normalized_state: torch.Tensor,
    model: ConditionalDenoiser,
    *,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the plain-residual gradient supplied to the conditional model."""

    return _gradient(
        normalized_state,
        model,
        strength=strength,
        normalized_energy=False,
    )


def correction_gradient(
    normalized_state: torch.Tensor,
    model: ConditionalDenoiser,
    *,
    strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the energy-normalized gradient subtracted after a solver step."""

    return _gradient(
        normalized_state,
        model,
        strength=strength,
        normalized_energy=True,
    )


__all__ = [
    "conditioning_gradient",
    "conditioning_residual_loss",
    "correction_gradient",
    "correction_residual_loss",
    "vorticity_residual",
]

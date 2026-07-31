"""Built-in denoiser and sampler artifact providers."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from stochaflow.sampling.grid import (
    save_image_grid,
    save_trajectory_gif,
    save_trajectory_grid,
)
from stochaflow.training.diagnostics.contracts import (
    ArtifactRecord,
    DenoiserArtifactContext,
    DenoiserArtifactProvider,
    ProviderValidationContext,
    SamplerArtifactContext,
    SamplerArtifactProvider,
)
from stochaflow.training.diagnostics.providers.denoiser import (
    parse_timesteps,
    validate_timesteps,
)
from stochaflow.training.diagnostics.registry import DIAGNOSTIC_PROVIDERS


def _positive_int(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


@DIAGNOSTIC_PROVIDERS.denoiser_artifacts.register("reconstruction_panel")
class ReconstructionPanelProvider(DenoiserArtifactProvider):
    """Write clean/noisy/predicted reconstruction tensors and a PNG panel."""

    def __init__(
        self,
        *,
        timesteps: Sequence[int],
        max_samples: int = 16,
    ) -> None:
        self.timesteps = parse_timesteps(
            timesteps,
            provider="reconstruction_panel",
        )
        self.max_samples = _positive_int(
            max_samples,
            path="reconstruction_panel max_samples",
        )

    def validate(self, context: ProviderValidationContext) -> None:
        validate_timesteps(
            self.timesteps,
            context,
            provider="reconstruction_panel",
        )

    def render(
        self,
        context: DenoiserArtifactContext,
    ) -> Sequence[ArtifactRecord]:
        if context.clean_samples is None:
            raise ValueError("reconstruction_panel requires clean_samples diagnostics")
        result = context.reconstruct(
            clean_samples=context.clean_samples,
            timesteps=self.timesteps,
            max_samples=self.max_samples,
            use_ema=context.use_ema,
        )
        if not result.frames:
            raise ValueError("reconstruction_panel produced no reconstruction frames")
        rows: list[torch.Tensor] = []
        for frame in result.frames:
            rows.extend([frame.clean, frame.noisy, frame.predicted_clean])
        reconstruction = torch.cat(rows, dim=0)
        tensor_path = context.store.reserve("denoiser/reconstruction.pt")
        grid_path = context.store.reserve("denoiser/reconstruction.png")
        torch.save(reconstruction, tensor_path)
        save_image_grid(
            reconstruction,
            grid_path,
            nrow=result.frames[0].clean.shape[0],
            denormalize=True,
        )
        return (
            ArtifactRecord(kind="reconstruction_tensor", path=tensor_path),
            ArtifactRecord(
                kind="reconstruction_image",
                path=grid_path,
                image_tag="diagnostics/denoiser/reconstruction",
                caption=(
                    "Rows repeat clean, noisy, and reconstructed samples per "
                    "timestep."
                ),
            ),
        )


@DIAGNOSTIC_PROVIDERS.sampler_artifacts.register("sample_grid")
class SampleGridProvider(SamplerArtifactProvider):
    """Write final sampler tensors and a fixed-noise image grid."""

    def __init__(self, *, nrow: int = 4) -> None:
        self.nrow = _positive_int(nrow, path="sample_grid nrow")

    def render(
        self,
        context: SamplerArtifactContext,
    ) -> Sequence[ArtifactRecord]:
        tensor_path = context.store.reserve(f"{context.profile_id}/samples.pt")
        grid_path = context.store.reserve(f"{context.profile_id}/samples.png")
        torch.save(context.result.samples, tensor_path)
        save_image_grid(
            context.result.samples,
            grid_path,
            nrow=self.nrow,
            denormalize=True,
        )
        return (
            ArtifactRecord(kind="sample_tensor", path=tensor_path),
            ArtifactRecord(
                kind="sample_image",
                path=grid_path,
                image_tag=f"diagnostics/samplers/{context.profile_id}/samples",
                caption=(
                    f"Sampler profile {context.profile_id} "
                    f"({context.profile_name})."
                ),
            ),
        )


@DIAGNOSTIC_PROVIDERS.sampler_artifacts.register("trajectory")
class TrajectoryArtifactProvider(SamplerArtifactProvider):
    """Write trajectory tensor, static grid, and GIF when enabled."""

    def __init__(self, *, nrow: int = 4) -> None:
        self.nrow = _positive_int(nrow, path="trajectory nrow")

    def render(
        self,
        context: SamplerArtifactContext,
    ) -> Sequence[ArtifactRecord]:
        if not context.trajectory_enabled:
            return ()
        trajectory = context.result.trajectory
        if trajectory is None:
            raise ValueError(
                f"sampler '{context.profile_id}' did not return a trajectory"
            )
        tensor_path = context.store.reserve(f"{context.profile_id}/trajectory.pt")
        grid_path = context.store.reserve(f"{context.profile_id}/trajectory.png")
        gif_path = context.store.reserve(f"{context.profile_id}/trajectory.gif")
        states = tuple(snapshot.state for snapshot in trajectory)
        torch.save(
            {
                "step_indices": [snapshot.step_index for snapshot in trajectory],
                "coordinates": [snapshot.coordinate for snapshot in trajectory],
                "states": torch.stack(states, dim=0),
            },
            tensor_path,
        )
        save_trajectory_grid(states, grid_path, denormalize=True)
        save_trajectory_gif(
            states,
            gif_path,
            nrow=self.nrow,
            fps=context.trajectory_gif_fps,
            denormalize=True,
        )
        return (
            ArtifactRecord(kind="trajectory_tensor", path=tensor_path),
            ArtifactRecord(
                kind="trajectory_image",
                path=grid_path,
                image_tag=(
                    f"diagnostics/samplers/{context.profile_id}/trajectory"
                ),
                caption=(
                    f"Reverse trajectory for sampler profile {context.profile_id}."
                ),
            ),
            ArtifactRecord(kind="trajectory_gif", path=gif_path),
        )


__all__ = [
    "ReconstructionPanelProvider",
    "SampleGridProvider",
    "TrajectoryArtifactProvider",
]

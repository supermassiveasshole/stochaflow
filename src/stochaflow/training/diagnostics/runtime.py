"""Runtime services shared by diagnostic providers and orchestrators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
import hashlib
import time
from typing import Any

import torch
import torch.nn as nn

from stochaflow.diffusion import GaussianDiffusion
from stochaflow.sampling import SamplingTrace
from stochaflow.training.diagnostics.config import SamplerProfileConfig
from stochaflow.training.diagnostics.contracts import (
    ReconstructionFrame,
    ReconstructionResult,
    SamplingResult,
)
from stochaflow.utils.registry import REGISTRIES


def first_tensor_from_batch(batch: Any) -> torch.Tensor | None:
    """Find the first tensor in a common training batch structure."""

    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, Mapping):
        for value in batch.values():
            tensor = first_tensor_from_batch(value)
            if tensor is not None:
                return tensor
        return None
    if isinstance(batch, (tuple, list)):
        for value in batch:
            tensor = first_tensor_from_batch(value)
            if tensor is not None:
                return tensor
    return None


def clean_samples_from_event(output: Any, batch: Any) -> torch.Tensor | None:
    """Prefer train-step clean samples, then fall back to the input batch."""

    diagnostics = getattr(output, "diagnostics", None)
    if isinstance(diagnostics, Mapping):
        clean = diagnostics.get("clean_samples")
        if isinstance(clean, torch.Tensor):
            return clean
    return first_tensor_from_batch(batch)


def prepare_reference_images(images: torch.Tensor) -> torch.Tensor:
    """Convert normalized grayscale/RGB images to float RGB in ``[0, 1]``."""

    images = images.detach().float()
    if images.ndim != 4:
        raise ValueError("reference metric images must have shape (N, C, H, W)")
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    elif images.shape[1] != 3:
        raise ValueError("reference metrics support one-channel or RGB images")
    return ((images.clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()


def _fork_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    return [torch.cuda.current_device() if device.index is None else device.index]


def _manual_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class SeedPolicy:
    """Stable seed derivation and fixed terminal-noise generation."""

    def __init__(self, base_seed: int) -> None:
        self.base_seed = base_seed

    def profile_seed(self, profile_id: str) -> int:
        """Derive a stable reverse-process seed for one profile."""

        digest = hashlib.sha256(profile_id.encode("utf-8")).digest()
        offset = int.from_bytes(digest[:4], byteorder="little")
        return (self.base_seed + offset) % (2**31 - 1)

    def initial_noise(
        self,
        count: int,
        sample_shape: tuple[int, int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Create the common fixed terminal-noise batch on the target device."""

        generator = torch.Generator(device="cpu").manual_seed(self.base_seed)
        return torch.randn(
            (count, *sample_shape),
            generator=generator,
            device="cpu",
        ).to(device)

    @contextmanager
    def fork_rng(self, device: torch.device, *, offset: int = 0):
        """Run with a fixed seed and restore global RNG state on exit."""

        with torch.random.fork_rng(devices=_fork_rng_devices(device)):
            _manual_seed(self.base_seed + offset, device)
            yield


class EvaluationGuard:
    """Protect inference mode, model mode, EMA weights, and global RNG state."""

    def __init__(
        self,
        trainer: Any,
        *,
        seed: int,
        use_ema: bool,
        evaluation_modules: Sequence[nn.Module] = (),
    ) -> None:
        self.trainer = trainer
        self.seed = seed
        self.use_ema = use_ema
        self.model = trainer.model
        self._stack = ExitStack()
        self._was_training = bool(self.model.training)
        self._ema = trainer.ema if use_ema else None
        self._ema_stored = False
        self._evaluation_module_modes = tuple(
            (module, bool(module.training))
            for module in evaluation_modules
            if module is not self.model
        )

    def __enter__(self) -> nn.Module:
        self._stack.__enter__()
        try:
            self._stack.enter_context(torch.inference_mode())
            self._stack.enter_context(
                torch.random.fork_rng(
                    devices=_fork_rng_devices(self.trainer.device)
                )
            )
            _manual_seed(self.seed, self.trainer.device)
            if self._ema is not None:
                self._ema.store(self.model)
                self._ema_stored = True
                self._ema.copy_to(self.model)
            self.model.eval()
            for module, _ in self._evaluation_module_modes:
                module.eval()
            return self.model
        except BaseException:
            self._restore()
            self._stack.close()
            raise

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, exc, traceback
        try:
            self._restore()
        finally:
            self._stack.close()
        return False

    def _restore(self) -> None:
        try:
            if self._ema is not None and self._ema_stored:
                self._ema.restore(self.model)
                self._ema_stored = False
        finally:
            for module, was_training in self._evaluation_module_modes:
                module.train(was_training)
            self.model.train(self._was_training)


class SamplerPool:
    """Build and retain inference-only samplers sharing one training denoiser."""

    def __init__(
        self,
        training_diffusion: GaussianDiffusion,
        profiles: Sequence[SamplerProfileConfig],
        *,
        device: torch.device,
    ) -> None:
        self._samplers: dict[str, nn.Module] = {}
        for profile in profiles:
            sampler_cls = REGISTRIES.diffusions.resolve(profile.name)
            if not callable(getattr(sampler_cls, "sample_from_noise", None)):
                raise TypeError(
                    f"sampler '{profile.name}' does not provide "
                    "sample_from_noise(initial_noise)"
                )
            if profile.trajectory.enabled and not callable(
                getattr(sampler_cls, "sample_trajectory_from_noise", None)
            ):
                raise TypeError(
                    f"sampler '{profile.name}' does not provide "
                    "sample_trajectory_from_noise()"
                )
            sampler = REGISTRIES.diffusions.create(
                profile.name,
                noise_schedule=training_diffusion.noise_schedule,
                model=training_diffusion.model,
                **profile.params,
            )
            if not isinstance(sampler, nn.Module):
                raise TypeError(
                    f"diagnostic sampler '{profile.id}' did not produce an nn.Module"
                )
            sampler.to(device)
            self._samplers[profile.id] = sampler

    def get(self, profile_id: str) -> nn.Module:
        """Return a previously validated sampler by profile ID."""

        try:
            return self._samplers[profile_id]
        except KeyError as exc:
            raise RuntimeError(
                f"diagnostic sampler profile '{profile_id}' was not initialized"
            ) from exc


class SamplerRunner:
    """Execute batched sample or trajectory generation exactly once."""

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def run(
        self,
        sampler: nn.Module,
        profile: SamplerProfileConfig,
        initial_noise: torch.Tensor,
    ) -> SamplingResult:
        """Generate a profile result while preserving trajectory batch alignment."""

        sample_parts: list[torch.Tensor] = []
        frame_parts: dict[int, list[torch.Tensor]] = {}
        expected_times: list[int] | None = None
        _synchronize(initial_noise.device)
        started_at = time.perf_counter()
        for start in range(0, initial_noise.shape[0], self.batch_size):
            noise_batch = initial_noise[start : start + self.batch_size]
            if profile.trajectory.enabled:
                trace_fn = getattr(sampler, "sample_trajectory_from_noise")
                trace = trace_fn(noise_batch, **profile.trajectory.params)
                if not isinstance(trace, SamplingTrace):
                    raise TypeError(
                        f"sampler '{profile.id}' trajectory must return SamplingTrace"
                    )
                state_times = [frame.state_time for frame in trace.frames]
                if expected_times is None:
                    expected_times = state_times
                elif state_times != expected_times:
                    raise ValueError(
                        f"sampler '{profile.id}' trajectory frame times changed "
                        "between batches"
                    )
                for frame in trace.frames:
                    frame_parts.setdefault(frame.state_time, []).append(
                        frame.samples.detach().cpu()
                    )
                sampled = trace.samples
            else:
                sample_fn = getattr(sampler, "sample_from_noise")
                sampled = sample_fn(noise_batch)
            if not isinstance(sampled, torch.Tensor):
                raise TypeError(
                    f"sampler '{profile.id}' sample_from_noise must return a Tensor"
                )
            if sampled.shape != noise_batch.shape:
                raise ValueError(
                    f"sampler '{profile.id}' returned shape {tuple(sampled.shape)}, "
                    f"expected {tuple(noise_batch.shape)}"
                )
            sample_parts.append(sampled.detach().cpu())
        _synchronize(initial_noise.device)
        trajectory = (
            {
                state_time: torch.cat(parts, dim=0)
                for state_time, parts in frame_parts.items()
            }
            if frame_parts
            else None
        )
        return SamplingResult(
            samples=torch.cat(sample_parts, dim=0),
            trajectory=trajectory,
            duration_seconds=time.perf_counter() - started_at,
        )


class ReconstructionEvaluator:
    """Evaluate fixed-timestep ``x0`` reconstruction under a protected model."""

    def __init__(self, trainer: Any, seed_policy: SeedPolicy) -> None:
        self.trainer = trainer
        self.seed_policy = seed_policy

    def __call__(
        self,
        *,
        clean_samples: torch.Tensor,
        timesteps: Sequence[int],
        max_samples: int,
        use_ema: bool,
    ) -> ReconstructionResult:
        x0 = clean_samples[:max_samples].to(self.trainer.device)
        if x0.ndim != 4:
            raise ValueError("reconstruction samples must have shape (N, C, H, W)")
        frames: list[ReconstructionFrame] = []
        with EvaluationGuard(
            self.trainer,
            seed=self.seed_policy.base_seed,
            use_ema=use_ema,
        ) as model:
            if not isinstance(model, GaussianDiffusion):
                raise TypeError("reconstruction requires a GaussianDiffusion model")
            for timestep in timesteps:
                times = torch.full(
                    (x0.shape[0],),
                    timestep,
                    dtype=torch.long,
                    device=self.trainer.device,
                )
                noise = torch.randn_like(x0)
                noisy, _ = model.add_noise(x0, times, noise=noise)
                predicted_noise = model._predict_noise(noisy, times)
                predicted_clean = model._estimate_x0_from_epsilon(
                    noisy,
                    times,
                    predicted_noise=predicted_noise,
                    clip_denoised=model.clip_denoised,
                )
                mse = (predicted_clean - x0).square().mean()
                psnr = 10.0 * torch.log10(
                    torch.tensor(4.0, device=self.trainer.device)
                    / mse.clamp_min(1e-12)
                )
                frames.append(
                    ReconstructionFrame(
                        timestep=timestep,
                        clean=x0.detach().cpu(),
                        noisy=noisy.detach().cpu(),
                        predicted_clean=predicted_clean.detach().cpu(),
                        mse=float(mse),
                        psnr=float(psnr),
                    )
                )
        return ReconstructionResult(frames=tuple(frames))


__all__ = [
    "EvaluationGuard",
    "ReconstructionEvaluator",
    "SamplerPool",
    "SamplerRunner",
    "SeedPolicy",
    "clean_samples_from_event",
    "first_tensor_from_batch",
    "prepare_reference_images",
]

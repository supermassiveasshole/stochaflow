"""Runtime services shared by diagnostic providers and orchestrators."""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from stochaflow.processes import DiscreteGaussianDenoisingProcess
from stochaflow.sampling import (
    GaussianDenoisingDynamics,
    GaussianModelDynamics,
    PredictionType,
    Sampler,
    SamplerResult,
    SamplingObservation,
)
from stochaflow.training.diagnostics.config import SamplerProfileConfig
from stochaflow.training.diagnostics.contracts import (
    ReconstructionFrame,
    ReconstructionResult,
    SamplingResult,
)
from stochaflow.training.gaussian import GaussianDiagnosticSemantics
from stochaflow.utils.registry import REGISTRIES
from stochaflow.utils.seed import preserve_global_rng_state


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


def _manual_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _cpu_tensor_snapshot(state: Any) -> torch.Tensor:
    if not isinstance(state, torch.Tensor):
        raise TypeError("diagnostic trajectory states must be tensors")
    return state.detach().cpu().clone()


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

        with preserve_global_rng_state(device):
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
        self.ema_model = getattr(trainer, "ema_model", self.model)
        self._stack = ExitStack()
        self._ema = trainer.ema if use_ema else None
        self._ema_stored = False
        discovered: list[nn.Module] = [self.model, *evaluation_modules]
        managed = getattr(trainer, "managed_modules", None)
        if isinstance(managed, Mapping):
            for asset in managed.values():
                module = getattr(asset, "module", asset)
                if isinstance(module, nn.Module):
                    discovered.append(module)
        seen: set[int] = set()
        evaluation_module_modes: list[tuple[nn.Module, bool]] = []
        for module in discovered:
            if id(module) in seen:
                continue
            seen.add(id(module))
            evaluation_module_modes.append((module, bool(module.training)))
        self._evaluation_module_modes = tuple(evaluation_module_modes)

    def __enter__(self) -> nn.Module:
        self._stack.__enter__()
        try:
            self._stack.enter_context(torch.inference_mode())
            self._stack.enter_context(
                preserve_global_rng_state(self.trainer.device)
            )
            _manual_seed(self.seed, self.trainer.device)
            if self._ema is not None:
                self._ema.store(self.ema_model)
                self._ema_stored = True
                self._ema.copy_to(self.ema_model)
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
                self._ema.restore(self.ema_model)
                self._ema_stored = False
        finally:
            for module, was_training in self._evaluation_module_modes:
                module.train(was_training)


@dataclass(frozen=True, slots=True)
class GaussianTrainingRuntime:
    """Gaussian process and task-adapted prediction used by diagnostics."""

    process: DiscreteGaussianDenoisingProcess
    prediction_type: PredictionType
    predict_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def gaussian_training_runtime(trainer: Any) -> GaussianTrainingRuntime:
    """Resolve the narrow Gaussian training capability from a Trainer."""

    model = getattr(trainer, "model", None)
    if not isinstance(model, nn.Module):
        raise TypeError("Gaussian diagnostics require a primary nn.Module model")
    process = getattr(trainer, "process", None)
    if not isinstance(process, DiscreteGaussianDenoisingProcess):
        raise TypeError(
            "Gaussian diagnostics require DiscreteGaussianDenoisingProcess"
        )
    strategy = getattr(trainer, "strategy", None)
    if not isinstance(strategy, GaussianDiagnosticSemantics):
        raise TypeError(
            "Gaussian diagnostics require GaussianDiagnosticSemantics strategy"
        )
    return GaussianTrainingRuntime(
        process,
        strategy.prediction_type,
        strategy.predict_gaussian_model,
    )


@dataclass(frozen=True, slots=True)
class BoundSampler:
    """A solver bound to model-aware Gaussian dynamics."""

    sampler: Sampler
    dynamics: GaussianDenoisingDynamics


class SamplerPool:
    """Build and retain inference-only samplers sharing one training denoiser."""

    def __init__(
        self,
        training_runtime: GaussianTrainingRuntime,
        profiles: Sequence[SamplerProfileConfig],
        *,
        device: torch.device,
    ) -> None:
        del device
        self._samplers: dict[str, BoundSampler] = {}
        process = training_runtime.process
        dynamics = GaussianModelDynamics(
            process,
            training_runtime.predict_fn,
            prediction_type=training_runtime.prediction_type,
            clip_denoised=True,
        )
        for profile in profiles:
            sampler = cast(
                Sampler,
                REGISTRIES.samplers.create(profile.name, **profile.params),
            )
            self._samplers[profile.id] = BoundSampler(sampler, dynamics)

    def get(self, profile_id: str) -> BoundSampler:
        """Return a previously validated sampler by profile ID."""

        try:
            return self._samplers[profile_id]
        except KeyError as exc:
            raise RuntimeError(
                f"diagnostic sampler profile '{profile_id}' was not initialized"
            ) from exc


class DiagnosticSamplingObserver:
    """Validate a diagnostic denoising lifecycle and optionally retain it."""

    def __init__(
        self,
        *,
        process: DiscreteGaussianDenoisingProcess,
        expected_shape: torch.Size,
        retain: bool,
        every_steps: int,
    ) -> None:
        self._process = process
        self._expected_shape = expected_shape
        self._retain = retain
        self._every_steps = every_steps
        self._previous_step: int | None = None
        self._final_seen = False
        self._observations: list[SamplingObservation] = []

    @property
    def observations(self) -> tuple[SamplingObservation, ...] | None:
        if not self._retain:
            return None
        return tuple(self._observations)

    def observe(self, observation: object) -> None:
        if not isinstance(observation, SamplingObservation):
            raise TypeError("diagnostic sampler events must be SamplingObservation")
        if self._final_seen:
            raise ValueError("diagnostic sampler emitted an event after final")
        step_index = cast(object, observation.step_index)
        if isinstance(step_index, bool) or not isinstance(step_index, int):
            raise TypeError("diagnostic observation step_index must be an integer")
        if self._previous_step is None:
            if step_index != 0:
                raise ValueError("diagnostic sampler must start at step index 0")
            if observation.coordinate != self._process.terminal_time:
                raise ValueError(
                    "diagnostic sampler must start at process terminal time"
                )
        elif step_index <= self._previous_step:
            raise ValueError("diagnostic observation step indices must increase")
        is_final = cast(object, observation.is_final)
        if not isinstance(is_final, bool):
            raise TypeError("diagnostic observation is_final must be boolean")
        diagnostics = cast(object, observation.diagnostics)
        if not isinstance(diagnostics, Mapping):
            raise TypeError("diagnostic observation diagnostics must be a mapping")
        state = observation.state
        if not isinstance(state, torch.Tensor):
            raise TypeError("diagnostic trajectory states must be tensors")
        if state.shape != self._expected_shape:
            raise ValueError(
                "diagnostic observation state shape must match its initial noise"
            )
        if observation.is_final:
            if observation.coordinate != self._process.clean_time:
                raise ValueError(
                    "diagnostic sampler must end at process clean time"
                )
            self._final_seen = True
        self._previous_step = step_index
        if self._retain and (
            step_index == 0
            or observation.is_final
            or step_index % self._every_steps == 0
        ):
            self._observations.append(
                SamplingObservation(
                    step_index=step_index,
                    coordinate=observation.coordinate,
                    state=_cpu_tensor_snapshot(state),
                    is_final=observation.is_final,
                    diagnostics=dict(observation.diagnostics),
                )
            )

    def validate_complete(self, result: SamplerResult) -> None:
        if self._previous_step is None:
            raise ValueError("diagnostic sampler emitted no observations")
        if not self._final_seen:
            raise ValueError("diagnostic sampler emitted no final observation")
        if self._previous_step != result.num_steps:
            raise ValueError(
                "diagnostic final observation must match SamplerResult.num_steps"
            )


class SamplerRunner:
    """Execute batched sample or trajectory generation exactly once."""

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size

    def run(
        self,
        sampler: BoundSampler,
        profile: SamplerProfileConfig,
        initial_noise: torch.Tensor,
    ) -> SamplingResult:
        """Generate a profile result while preserving trajectory batch alignment."""

        sample_parts: list[torch.Tensor] = []
        frame_parts: list[list[torch.Tensor]] = []
        expected_identity: tuple[tuple[int, int | float, bool], ...] | None = None
        template_observations: tuple[SamplingObservation, ...] | None = None
        _synchronize(initial_noise.device)
        started_at = time.perf_counter()
        for start in range(0, initial_noise.shape[0], self.batch_size):
            noise_batch = initial_noise[start : start + self.batch_size]
            lifecycle = DiagnosticSamplingObserver(
                process=sampler.dynamics.process,
                expected_shape=noise_batch.shape,
                retain=profile.trajectory.enabled,
                every_steps=profile.trajectory.every_steps,
            )
            result_value = cast(
                object,
                sampler.sampler.sample(
                    sampler.dynamics,
                    noise_batch,
                    observer=lifecycle,
                ),
            )
            if not isinstance(result_value, SamplerResult):
                raise TypeError(
                    f"sampler '{profile.id}' must return SamplerResult"
                )
            lifecycle.validate_complete(result_value)
            sampled = result_value.final_state
            observations = lifecycle.observations
            if observations is not None:
                identity = tuple(
                    (
                        observation.step_index,
                        observation.coordinate,
                        observation.is_final,
                    )
                    for observation in observations
                )
                if expected_identity is None:
                    expected_identity = identity
                    template_observations = observations
                    frame_parts = [[] for _ in identity]
                elif identity != expected_identity:
                    raise ValueError(
                        f"sampler '{profile.id}' trajectory lifecycle changed "
                        "between batches"
                    )
                for index, observation in enumerate(observations):
                    state = observation.state
                    if not isinstance(state, torch.Tensor):
                        raise TypeError("diagnostic trajectory states must be tensors")
                    frame_parts[index].append(state)
            if not isinstance(sampled, torch.Tensor):
                raise TypeError(
                    f"sampler '{profile.id}' must return a Tensor final_state"
                )
            if sampled.shape != noise_batch.shape:
                raise ValueError(
                    f"sampler '{profile.id}' returned shape {tuple(sampled.shape)}, "
                    f"expected {tuple(noise_batch.shape)}"
                )
            sample_parts.append(sampled.detach().cpu())
        _synchronize(initial_noise.device)
        trajectory = None
        if frame_parts and template_observations is not None:
            trajectory = tuple(
                SamplingObservation(
                    step_index=template.step_index,
                    coordinate=template.coordinate,
                    state=torch.cat(parts, dim=0),
                    is_final=template.is_final,
                    diagnostics=dict(template.diagnostics),
                )
                for template, parts in zip(
                    template_observations, frame_parts, strict=True
                )
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
        runtime = gaussian_training_runtime(self.trainer)
        with EvaluationGuard(
            self.trainer,
            seed=self.seed_policy.base_seed,
            use_ema=use_ema,
        ):
            process = runtime.process
            dynamics = GaussianModelDynamics(
                process,
                runtime.predict_fn,
                prediction_type=runtime.prediction_type,
                clip_denoised=True,
            )
            for timestep in timesteps:
                times = torch.full(
                    (x0.shape[0],),
                    timestep,
                    dtype=torch.long,
                    device=self.trainer.device,
                )
                noise = torch.randn_like(x0)
                noisy, _ = process.sample_marginal(x0, times, noise=noise)
                predicted_clean = dynamics.predict(noisy, times).clean
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
    "BoundSampler",
    "EvaluationGuard",
    "GaussianTrainingRuntime",
    "ReconstructionEvaluator",
    "SamplerPool",
    "SamplerRunner",
    "SeedPolicy",
    "clean_samples_from_event",
    "first_tensor_from_batch",
    "gaussian_training_runtime",
    "prepare_reference_images",
]

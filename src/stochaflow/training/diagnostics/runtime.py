"""Runtime services shared by diagnostic providers and orchestrators."""

from __future__ import annotations

import hashlib
import random
import threading
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from stochaflow._builtin_activation import activate_sampling_component_builtins
from stochaflow.families.gaussian.prediction import PredictionType
from stochaflow.processes.gaussian.contracts import (
    DiscreteGaussianDenoisingProcess,
)
from stochaflow.sampling.gaussian.class_conditional import (
    ClassConditionalEvaluationCounts,
    ClassifierFreeGuidancePredictor,
)
from stochaflow.sampling.gaussian.dynamics import (
    GaussianDenoisingDynamics,
    GaussianModelDynamics,
    VarianceMode,
)
from stochaflow.sampling.sampler import (
    Sampler,
    SamplerResult,
    SamplingObservation,
)
from stochaflow.training.diagnostics.config import SamplerProfileConfig
from stochaflow.training.diagnostics.contracts import (
    DiagnosticModelAccess,
    ReconstructionFrame,
    ReconstructionResult,
    SamplingResult,
)
from stochaflow.training.ema import ExponentialMovingAverage
from stochaflow.training.gaussian.contracts import (
    ClassConditionalGaussianDiagnosticSemantics,
    GaussianDiagnosticSemantics,
)
from stochaflow.utils.registry import REGISTRIES
from stochaflow.utils.seed import preserve_global_rng_state


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
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            "diagnostic evaluation supports CPU, CUDA, or MPS devices"
        )
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)


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
    def fork_rng(
        self,
        device: torch.device,
        *,
        offset: int = 0,
    ) -> Generator[None, None, None]:
        """Run with a fixed seed and restore global RNG state on exit."""

        with preserve_global_rng_state(device):
            _manual_seed(self.base_seed + offset, device)
            yield


class DiagnosticModelAccessCleanupError(RuntimeError):
    """Report that protected diagnostic model state could not be restored."""


type ModuleMode = tuple[str, nn.Module, bool, int, int]
type ModuleRootMode = tuple[str, nn.Module, bool]
type ModuleModeTree = tuple[ModuleRootMode, frozenset[int]]


def _capture_module_mode_trees(
    managed_modules: Sequence[tuple[str, nn.Module]],
) -> tuple[tuple[ModuleModeTree, ...], tuple[ModuleMode, ...]]:
    """Capture unique managed modules and a parent-before-child restore order."""

    modes: dict[int, ModuleMode] = {}
    edges: dict[int, set[int]] = {}
    mode_trees: list[ModuleModeTree] = []
    capture_order = 0
    for name, root in managed_modules:
        root_id = id(root)
        if root_id in modes:
            continue
        reachable: set[int] = set()
        pending: list[tuple[str, nn.Module]] = [(name, root)]
        while pending:
            label, module = pending.pop()
            module_id = id(module)
            if module_id in reachable:
                continue
            reachable.add(module_id)
            existing = modes.get(module_id)
            if existing is None:
                modes[module_id] = (
                    label,
                    module,
                    bool(module.training),
                    0,
                    capture_order,
                )
                capture_order += 1
            children = tuple(module.named_children())
            edges.setdefault(module_id, set()).update(
                id(child) for _, child in children
            )
            for child_name, child in reversed(children):
                pending.append((f"{label}.{child_name}", child))
        root_mode = modes[root_id]
        mode_trees.append(
            ((root_mode[0], root_mode[1], root_mode[2]), frozenset(reachable))
        )

    indegrees = dict.fromkeys(modes, 0)
    for children in edges.values():
        for child_id in children:
            indegrees[child_id] += 1
    longest_depth = dict.fromkeys(modes, 0)
    ready = [module_id for module_id, degree in indegrees.items() if degree == 0]
    visited = 0
    while ready:
        module_id = ready.pop()
        visited += 1
        for child_id in edges.get(module_id, ()):
            longest_depth[child_id] = max(
                longest_depth[child_id], longest_depth[module_id] + 1
            )
            indegrees[child_id] -= 1
            if indegrees[child_id] == 0:
                ready.append(child_id)
    if visited != len(modes):
        raise ValueError(
            "managed Diagnostic modules must form an acyclic module graph"
        )
    for module_id, mode in tuple(modes.items()):
        modes[module_id] = (
            mode[0],
            mode[1],
            mode[2],
            longest_depth[module_id],
            mode[4],
        )
    ordered_modes = tuple(
        sorted(modes.values(), key=lambda item: (item[3], -item[4]))
    )
    return tuple(mode_trees), ordered_modes


class TrainingDiagnosticModelAccess:
    """Serialize protected diagnostic access to one managed training model."""

    def __init__(
        self,
        *,
        device: torch.device,
        model: nn.Module,
        ema: ExponentialMovingAverage | None,
        managed_modules: Sequence[tuple[str, nn.Module]],
    ) -> None:
        self._device = torch.device(device)
        self._model = model
        self._ema = ema
        discovered = (("primary_model", model), *managed_modules)
        seen: set[int] = set()
        modules: list[tuple[str, nn.Module]] = []
        for name, module in discovered:
            if id(module) in seen:
                continue
            seen.add(id(module))
            modules.append((name, module))
        self._managed_modules = tuple(modules)
        self._lock = threading.RLock()
        self._active = False

    @property
    def device(self) -> torch.device:
        """Return the device used by the managed training model."""

        return self._device

    @property
    def ema_available(self) -> bool:
        """Report whether this training run owns an EMA snapshot."""

        return self._ema is not None

    def evaluation(
        self,
        *,
        seed: int,
        prefer_ema: bool,
    ) -> AbstractContextManager[None]:
        """Return one serialized, fully restored diagnostic evaluation scope."""

        seed_value = cast(object, seed)
        if isinstance(seed_value, bool) or not isinstance(seed_value, int):
            raise TypeError("diagnostic evaluation seed must be an integer")
        prefer_ema_value = cast(object, prefer_ema)
        if not isinstance(prefer_ema_value, bool):
            raise TypeError("diagnostic prefer_ema must be a boolean")
        return DiagnosticEvaluationContext(self, seed_value, prefer_ema_value)


class DiagnosticEvaluationContext:
    """Implement one strict DiagnosticModelAccess evaluation scope."""

    def __init__(
        self,
        owner: TrainingDiagnosticModelAccess,
        seed: int,
        prefer_ema: bool,
    ) -> None:
        self._owner = owner
        self._seed = seed
        self._prefer_ema = prefer_ema
        self._stack = ExitStack()
        self._module_mode_trees: list[ModuleModeTree] = []
        self._captured_module_modes: tuple[ModuleMode, ...] = ()
        self._ema_stored = False
        self._entered = False

    def __enter__(self) -> None:
        owner = self._owner
        owner._lock.acquire()
        if owner._active:
            owner._lock.release()
            raise RuntimeError("diagnostic model evaluation cannot be nested")
        owner._active = True
        self._entered = True
        try:
            self._stack.__enter__()
            self._stack.enter_context(torch.inference_mode())
            self._stack.enter_context(preserve_global_rng_state(owner.device))
            _manual_seed(self._seed, owner.device)
            if self._prefer_ema and owner._ema is not None:
                owner._ema.store(owner._model)
                self._ema_stored = True
                owner._ema.copy_to(owner._model)
            mode_trees, self._captured_module_modes = _capture_module_mode_trees(
                owner._managed_modules
            )
            for mode_tree in mode_trees:
                (_, module, _), _ = mode_tree
                self._module_mode_trees.append(mode_tree)
                module.eval()
        except BaseException as error:
            self._finish(error)
            raise
        return

    def __exit__(self, exc_type, exc, traceback) -> bool:
        del exc_type, traceback
        self._finish(exc)
        return False

    def _finish(self, body_error: BaseException | None) -> None:
        failures: list[tuple[str, BaseException]] = []
        affected_modules: set[int] = set()
        for _, reachable in self._module_mode_trees:
            affected_modules.update(reachable)
        for name, module, was_training, _, _ in self._captured_module_modes:
            if id(module) not in affected_modules:
                continue
            try:
                module.train(was_training)
            except BaseException as error:  # noqa: BLE001
                failures.append((f"restore module '{name}' mode", error))
        self._module_mode_trees.clear()
        self._captured_module_modes = ()
        owner = self._owner
        if self._ema_stored:
            assert owner._ema is not None
            try:
                owner._ema.restore(owner._model)
            except BaseException as error:  # noqa: BLE001
                failures.append(("restore raw model weights", error))
            self._ema_stored = False
        try:
            self._stack.close()
        except BaseException as error:  # noqa: BLE001
            failures.append(("restore inference and RNG state", error))
        if self._entered:
            owner._active = False
            owner._lock.release()
            self._entered = False
        if failures:
            cleanup_error = DiagnosticModelAccessCleanupError(
                "diagnostic model state restoration failed"
            )
            for label, error in failures:
                try:
                    detail = str(error)
                except BaseException:  # noqa: BLE001
                    detail = "<exception text unavailable>"
                BaseException.add_note(
                    cleanup_error,
                    f"{label}: {type(error).__name__}: {detail}"
                )
            cause = body_error if body_error is not None else failures[0][1]
            raise cleanup_error from cause


@dataclass(frozen=True, slots=True)
class GaussianTrainingRuntime:
    """Gaussian process and task-adapted prediction used by diagnostics."""

    process: DiscreteGaussianDenoisingProcess
    prediction_type: PredictionType
    predict_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    variance_mode: VarianceMode = "fixed"


@dataclass(frozen=True, slots=True)
class ClassConditionalGaussianTrainingRuntime:
    """Conditional Gaussian process and task-adapted diagnostic invocation."""

    process: DiscreteGaussianDenoisingProcess
    prediction_type: PredictionType
    num_classes: int
    null_class_id: int
    predict_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]
    variance_mode: VarianceMode = "fixed"


def gaussian_training_runtime(
    process: object,
    strategy: object,
) -> GaussianTrainingRuntime:
    """Bind Gaussian diagnostic semantics from explicit narrow capabilities."""

    if not isinstance(process, DiscreteGaussianDenoisingProcess):
        raise TypeError(
            "Gaussian diagnostics require DiscreteGaussianDenoisingProcess"
        )
    if not isinstance(strategy, GaussianDiagnosticSemantics):
        raise TypeError(
            "Gaussian diagnostics require GaussianDiagnosticSemantics strategy"
        )
    return GaussianTrainingRuntime(
        process,
        strategy.prediction_type,
        strategy.predict_gaussian_model,
        strategy.variance_mode,
    )


def class_conditional_gaussian_training_runtime(
    process: object,
    strategy: object,
) -> ClassConditionalGaussianTrainingRuntime:
    """Bind conditional diagnostics from explicit narrow capabilities."""

    if not isinstance(process, DiscreteGaussianDenoisingProcess):
        raise TypeError(
            "class-conditional Gaussian diagnostics require "
            "DiscreteGaussianDenoisingProcess"
        )
    if not isinstance(strategy, ClassConditionalGaussianDiagnosticSemantics):
        raise TypeError(
            "class-conditional Gaussian diagnostics require "
            "ClassConditionalGaussianDiagnosticSemantics strategy"
        )
    num_classes = cast(object, strategy.num_classes)
    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes <= 0
    ):
        raise ValueError(
            "class-conditional Gaussian diagnostic num_classes must be positive"
        )
    null_class_id = cast(object, strategy.null_class_id)
    if isinstance(null_class_id, bool) or not isinstance(
        null_class_id,
        int,
    ):
        raise TypeError(
            "class-conditional Gaussian diagnostic null_class_id must be "
            "an integer"
        )
    if null_class_id < num_classes:
        raise ValueError(
            "class-conditional Gaussian diagnostic null_class_id must be "
            "outside the non-null class range"
        )
    return ClassConditionalGaussianTrainingRuntime(
        process,
        strategy.prediction_type,
        num_classes,
        null_class_id,
        strategy.predict_class_conditional_gaussian_model,
        strategy.variance_mode,
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
        activate_sampling_component_builtins()
        self._samplers: dict[str, BoundSampler] = {}
        process = training_runtime.process
        dynamics = GaussianModelDynamics(
            process,
            training_runtime.predict_fn,
            prediction_type=training_runtime.prediction_type,
            variance_mode=training_runtime.variance_mode,
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


class DiagnosticClassConditionalDenoiser:
    """Adapt a diagnostic strategy capability to the CFG model contract."""

    def __init__(self, runtime: ClassConditionalGaussianTrainingRuntime) -> None:
        self.runtime = runtime

    @property
    def num_classes(self) -> int:
        """Return the number of real class identifiers."""

        return self.runtime.num_classes

    @property
    def null_class_id(self) -> int:
        """Return the reserved unconditional class identifier."""

        return self.runtime.null_class_id

    def predict_class_conditioned(
        self,
        state: torch.Tensor,
        model_time: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Invoke the training strategy's conditional diagnostic capability."""

        return self.runtime.predict_fn(state, model_time, class_labels)


@dataclass(frozen=True, slots=True)
class ClassConditionalBoundSampler:
    """A Gaussian solver awaiting per-batch conditional dynamics."""

    sampler: Sampler


class ClassConditionalSamplerPool:
    """Build inference-only solvers for a conditional diagnostic."""

    def __init__(
        self,
        profiles: Sequence[SamplerProfileConfig],
        *,
        device: torch.device,
    ) -> None:
        del device
        activate_sampling_component_builtins()
        self._samplers: dict[str, ClassConditionalBoundSampler] = {}
        for profile in profiles:
            sampler_value = cast(
                object,
                REGISTRIES.samplers.create(profile.name, **profile.params),
            )
            if not isinstance(sampler_value, Sampler):
                raise TypeError(
                    f"diagnostic sampler '{profile.name}' must satisfy Sampler"
                )
            self._samplers[profile.id] = ClassConditionalBoundSampler(
                sampler_value
            )

    def get(self, profile_id: str) -> ClassConditionalBoundSampler:
        """Return a previously validated conditional sampler."""

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


def _run_sampler_batches(
    *,
    sampler: Sampler,
    profile: SamplerProfileConfig,
    initial_noise: torch.Tensor,
    batch_size: int,
    dynamics_for_batch: Callable[
        [int, int],
        GaussianDenoisingDynamics,
    ],
) -> SamplingResult:
    sample_parts: list[torch.Tensor] = []
    frame_parts: list[list[torch.Tensor]] = []
    expected_identity: tuple[tuple[int, int | float, bool], ...] | None = None
    template_observations: tuple[SamplingObservation, ...] | None = None
    _synchronize(initial_noise.device)
    started_at = time.perf_counter()
    for start in range(0, initial_noise.shape[0], batch_size):
        end = min(start + batch_size, initial_noise.shape[0])
        noise_batch = initial_noise[start:end]
        dynamics = dynamics_for_batch(start, end)
        lifecycle = DiagnosticSamplingObserver(
            process=dynamics.process,
            expected_shape=noise_batch.shape,
            retain=profile.trajectory.enabled,
            every_steps=profile.trajectory.every_steps,
        )
        result_value = cast(
            object,
            sampler.sample(
                dynamics,
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
                    raise TypeError(
                        "diagnostic trajectory states must be tensors"
                    )
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
                template_observations,
                frame_parts,
                strict=True,
            )
        )
    return SamplingResult(
        samples=torch.cat(sample_parts, dim=0),
        trajectory=trajectory,
        duration_seconds=time.perf_counter() - started_at,
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

        return _run_sampler_batches(
            sampler=sampler.sampler,
            profile=profile,
            initial_noise=initial_noise,
            batch_size=self.batch_size,
            dynamics_for_batch=lambda start, end: sampler.dynamics,
        )


class ClassConditionalSamplerRunner:
    """Execute conditional diagnostic sampling with fixed labels and CFG."""

    def __init__(
        self,
        batch_size: int,
        *,
        runtime: ClassConditionalGaussianTrainingRuntime,
        class_labels: torch.Tensor,
        guidance_scale: float,
    ) -> None:
        if class_labels.ndim != 1 or class_labels.dtype != torch.long:
            raise ValueError(
                "conditional diagnostic class_labels must be a 1D long Tensor"
            )
        if bool(torch.any(class_labels < 0)) or bool(
            torch.any(class_labels >= runtime.num_classes)
        ):
            raise ValueError(
                "conditional diagnostic class labels are outside the model range"
            )
        self.batch_size = batch_size
        self.runtime = runtime
        self.class_labels = class_labels
        self.guidance_scale = guidance_scale
        self.model = DiagnosticClassConditionalDenoiser(runtime)
        self._counts: dict[str, ClassConditionalEvaluationCounts] = {}

    def run(
        self,
        sampler: ClassConditionalBoundSampler,
        profile: SamplerProfileConfig,
        initial_noise: torch.Tensor,
    ) -> SamplingResult:
        """Generate one class-aligned profile and retain branch counts."""

        if initial_noise.shape[0] != self.class_labels.shape[0]:
            raise ValueError(
                "conditional diagnostic labels must match initial noise"
            )
        if initial_noise.device != self.class_labels.device:
            raise ValueError(
                "conditional diagnostic labels and noise must share a device"
            )
        counts = ClassConditionalEvaluationCounts()

        def dynamics_for_batch(
            start: int,
            end: int,
        ) -> GaussianDenoisingDynamics:
            predictor = ClassifierFreeGuidancePredictor(
                self.model,
                self.class_labels[start:end],
                guidance_scale=self.guidance_scale,
                variance_mode=self.runtime.variance_mode,
                counts=counts,
            )
            return GaussianModelDynamics(
                self.runtime.process,
                predictor,
                prediction_type=self.runtime.prediction_type,
                variance_mode=self.runtime.variance_mode,
                clip_denoised=True,
            )

        result = _run_sampler_batches(
            sampler=sampler.sampler,
            profile=profile,
            initial_noise=initial_noise,
            batch_size=self.batch_size,
            dynamics_for_batch=dynamics_for_batch,
        )
        self._counts[profile.id] = counts
        return result

    def counts_for(
        self,
        profile_id: str,
    ) -> ClassConditionalEvaluationCounts:
        """Return model evaluation counts from the latest profile run."""

        try:
            return self._counts[profile_id]
        except KeyError as exc:
            raise RuntimeError(
                f"conditional diagnostic profile '{profile_id}' has not run"
            ) from exc


def _evaluate_reconstruction(
    *,
    model_access: DiagnosticModelAccess,
    seed_policy: SeedPolicy,
    clean_samples: torch.Tensor,
    timesteps: Sequence[int],
    max_samples: int,
    use_ema: bool,
    dynamics: GaussianDenoisingDynamics,
) -> ReconstructionResult:
    x0 = clean_samples[:max_samples].to(model_access.device)
    if x0.ndim != 4:
        raise ValueError("reconstruction samples must have shape (N, C, H, W)")
    frames: list[ReconstructionFrame] = []
    with model_access.evaluation(
        seed=seed_policy.base_seed,
        prefer_ema=use_ema,
    ):
        process = dynamics.process
        for timestep in timesteps:
            times = torch.full(
                (x0.shape[0],),
                timestep,
                dtype=torch.long,
                device=model_access.device,
            )
            noise = torch.randn_like(x0)
            noisy, _ = process.sample_marginal(x0, times, noise=noise)
            predicted_clean = dynamics.predict(noisy, times).clean
            mse = (predicted_clean - x0).square().mean()
            psnr = 10.0 * torch.log10(
                torch.tensor(4.0, device=model_access.device)
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


class ReconstructionEvaluator:
    """Evaluate fixed-timestep ``x0`` reconstruction under a protected model."""

    def __init__(
        self,
        model_access: DiagnosticModelAccess,
        seed_policy: SeedPolicy,
        runtime: GaussianTrainingRuntime,
    ) -> None:
        self.model_access = model_access
        self.seed_policy = seed_policy
        self.runtime = runtime

    def __call__(
        self,
        *,
        clean_samples: torch.Tensor,
        timesteps: Sequence[int],
        max_samples: int,
        use_ema: bool,
    ) -> ReconstructionResult:
        dynamics = GaussianModelDynamics(
            self.runtime.process,
            self.runtime.predict_fn,
            prediction_type=self.runtime.prediction_type,
            variance_mode=self.runtime.variance_mode,
            clip_denoised=True,
        )
        return _evaluate_reconstruction(
            model_access=self.model_access,
            seed_policy=self.seed_policy,
            clean_samples=clean_samples,
            timesteps=timesteps,
            max_samples=max_samples,
            use_ema=use_ema,
            dynamics=dynamics,
        )


class ClassConditionalReconstructionEvaluator:
    """Evaluate reconstruction with labels aligned to retained clean samples."""

    def __init__(
        self,
        model_access: DiagnosticModelAccess,
        seed_policy: SeedPolicy,
        runtime: ClassConditionalGaussianTrainingRuntime,
        class_labels: torch.Tensor,
    ) -> None:
        if class_labels.ndim != 1:
            raise ValueError(
                "conditional reconstruction labels must be a 1D Tensor"
            )
        self.model_access = model_access
        self.seed_policy = seed_policy
        self.runtime = runtime
        self.class_labels = class_labels.detach().cpu()

    def __call__(
        self,
        *,
        clean_samples: torch.Tensor,
        timesteps: Sequence[int],
        max_samples: int,
        use_ema: bool,
    ) -> ReconstructionResult:
        count = min(max_samples, clean_samples.shape[0])
        if self.class_labels.shape[0] < count:
            raise ValueError(
                "conditional reconstruction labels do not match clean samples"
            )
        labels = self.class_labels[:count].to(
            self.model_access.device,
            dtype=torch.long,
        )
        dynamics = GaussianModelDynamics(
            self.runtime.process,
            lambda state, model_time: self.runtime.predict_fn(
                state,
                model_time,
                labels,
            ),
            prediction_type=self.runtime.prediction_type,
            variance_mode=self.runtime.variance_mode,
            clip_denoised=True,
        )
        return _evaluate_reconstruction(
            model_access=self.model_access,
            seed_policy=self.seed_policy,
            clean_samples=clean_samples,
            timesteps=timesteps,
            max_samples=max_samples,
            use_ema=use_ema,
            dynamics=dynamics,
        )


__all__ = [
    "BoundSampler",
    "ClassConditionalBoundSampler",
    "ClassConditionalGaussianTrainingRuntime",
    "ClassConditionalReconstructionEvaluator",
    "ClassConditionalSamplerPool",
    "ClassConditionalSamplerRunner",
    "GaussianTrainingRuntime",
    "ReconstructionEvaluator",
    "SamplerPool",
    "SamplerRunner",
    "SeedPolicy",
    "class_conditional_gaussian_training_runtime",
    "gaussian_training_runtime",
    "prepare_reference_images",
]

"""Optimizer and learning-rate scheduler construction boundaries."""

from __future__ import annotations

import inspect
import math
from copy import deepcopy
from typing import Any, cast

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from stochaflow.utils.config import ComponentConfig, LRSchedulerConfig
from stochaflow.utils.registry import REGISTRIES, RegistryError

_TORCH_OPTIMIZER_PREFIX = "torch.optim."
_TORCH_LR_SCHEDULER_PREFIX = "torch.optim.lr_scheduler."


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@REGISTRIES.lr_schedulers.register("warmup_cosine")
class WarmupCosineLR(LRScheduler):
    """Linearly warm up before decaying with a half cosine."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_steps = _validate_positive_int(
            warmup_steps,
            name="warmup_steps",
        )
        self.total_steps = _validate_positive_int(total_steps, name="total_steps")
        if self.total_steps <= self.warmup_steps:
            raise ValueError("total_steps must be greater than warmup_steps")
        min_lr_ratio_value = cast(object, min_lr_ratio)
        if isinstance(min_lr_ratio_value, bool) or not isinstance(
            min_lr_ratio_value,
            (int, float),
        ):
            raise TypeError("min_lr_ratio must be numeric")
        self.min_lr_ratio = float(min_lr_ratio_value)
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be between 0 and 1")
        super().__init__(optimizer, last_epoch=last_epoch)

    def _lr_factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step + 1) / float(self.warmup_steps)
        progress = min(
            1.0,
            float(step + 1 - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps),
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay

    def get_lr(self) -> list[float]:
        """Compute learning rates for the current scheduler step."""

        factor = self._lr_factor(self.last_epoch)
        return [float(base_lr) * factor for base_lr in self.base_lrs]


REGISTRIES.optimizers.require_base(Optimizer)
REGISTRIES.lr_schedulers.require_base(LRScheduler)


def _resolve_native_class(
    name: str,
    *,
    prefix: str,
    namespace: object,
    expected_base: type[Any],
    kind: str,
) -> type[Any]:
    class_name = name.removeprefix(prefix)
    if (
        not class_name
        or "." in class_name
        or not class_name.isidentifier()
        or class_name.startswith("_")
    ):
        raise RegistryError(
            f"{kind} target '{name}' must name a direct class in '{prefix[:-1]}'"
        )
    try:
        candidate = getattr(namespace, class_name)
    except AttributeError as exc:
        raise RegistryError(f"unknown native {kind} target '{name}'") from exc
    if not isinstance(candidate, type) or not issubclass(candidate, expected_base):
        raise RegistryError(
            f"native {kind} target '{name}' must inherit "
            f"{expected_base.__module__}.{expected_base.__qualname__}"
        )
    return candidate


def _resolve_optimizer_class(name: str) -> type[Optimizer]:
    if name.startswith(_TORCH_OPTIMIZER_PREFIX):
        return _resolve_native_class(
            name,
            prefix=_TORCH_OPTIMIZER_PREFIX,
            namespace=torch.optim,
            expected_base=Optimizer,
            kind="optimizer",
        )
    return REGISTRIES.optimizers.resolve(name)


def _resolve_lr_scheduler_class(name: str) -> type[LRScheduler]:
    if name.startswith(_TORCH_LR_SCHEDULER_PREFIX):
        return _resolve_native_class(
            name,
            prefix=_TORCH_LR_SCHEDULER_PREFIX,
            namespace=torch.optim.lr_scheduler,
            expected_base=LRScheduler,
            kind="lr scheduler",
        )
    return REGISTRIES.lr_schedulers.resolve(name)


def _validate_zero_argument_step(component: object, *, kind: str, name: str) -> None:
    step = getattr(component, "step", None)
    if not callable(step):
        raise RegistryError(f"{kind} '{name}' must provide step()")
    try:
        step_signature = inspect.signature(step)
    except (TypeError, ValueError) as exc:
        raise RegistryError(
            f"{kind} '{name}' has no inspectable step() signature"
        ) from exc
    try:
        step_signature.bind()
    except TypeError as exc:
        raise RegistryError(
            f"{kind} '{name}' step() must be callable without arguments"
        ) from exc


def build_optimizer(config: ComponentConfig, parameters: Any) -> Optimizer:
    """Build an optimizer while injecting the selected trainable parameters."""

    constructor_params = deepcopy(config.params)
    if "params" in constructor_params:
        raise RegistryError("optimizer.params cannot override runtime parameter 'params'")
    optimizer_class = _resolve_optimizer_class(config.name)
    try:
        optimizer = optimizer_class(parameters, **constructor_params)
    except Exception as exc:
        raise RegistryError(
            f"failed to initialize optimizer '{config.name}' with params "
            f"{constructor_params}: {exc}"
        ) from exc
    _validate_zero_argument_step(optimizer, kind="optimizer", name=config.name)
    return optimizer


def build_lr_scheduler(
    config: LRSchedulerConfig | None,
    optimizer: Optimizer,
) -> LRScheduler | None:
    """Build a zero-argument-step scheduler while injecting its optimizer."""

    if config is None:
        return None
    constructor_params = deepcopy(config.params)
    if "optimizer" in constructor_params:
        raise RegistryError(
            "lr_scheduler.params cannot override runtime parameter 'optimizer'"
        )
    scheduler_class = _resolve_lr_scheduler_class(config.name)
    try:
        scheduler = scheduler_class(optimizer, **constructor_params)
    except Exception as exc:
        raise RegistryError(
            f"failed to initialize lr scheduler '{config.name}' with params "
            f"{constructor_params}: {exc}"
        ) from exc
    if getattr(scheduler, "optimizer", None) is not optimizer:
        raise RegistryError(
            f"lr scheduler '{config.name}' must retain the injected optimizer"
        )
    _validate_zero_argument_step(
        scheduler,
        kind="lr scheduler",
        name=config.name,
    )
    return scheduler


__all__ = ["WarmupCosineLR", "build_lr_scheduler", "build_optimizer"]

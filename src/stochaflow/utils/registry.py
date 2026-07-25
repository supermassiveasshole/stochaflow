"""Typed, object-oriented component registries."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, TypeVar, cast

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

U = TypeVar("U")


class RegistryError(ValueError):
    """Raised when a registry operation fails."""


def _component_name(component: object) -> str:
    module = getattr(component, "__module__", type(component).__module__)
    name = getattr(component, "__qualname__", type(component).__qualname__)
    return f"{module}.{name}"


class Registry[T](Mapping[str, T]):
    """Store and construct named components of one semantic kind.

    A registry is deliberately a small object instead of a public dictionary:
    it owns duplicate checks, optional base-class validation, and consistent
    construction errors.  Distribution discovery and plugin activation
    deliberately live outside the registry layer.
    """

    def __init__(
        self,
        kind: str,
        *,
        expected_type: type[Any] | None = None,
        reserved_prefixes: Sequence[str] = (),
    ) -> None:
        self.kind = kind
        self._expected_type = expected_type
        self._reserved_prefixes = tuple(reserved_prefixes)
        self._components: dict[str, T] = {}

    def __getitem__(self, name: str) -> T:
        return self.resolve(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._components)

    def __len__(self) -> int:
        return len(self._components)

    def require_base(self, expected_type: type[Any]) -> None:
        """Require subsequently registered components to inherit a base class."""

        if self._expected_type is not None and self._expected_type is not expected_type:
            raise RegistryError(
                f"{self.kind} registry already requires "
                f"{self._expected_type.__module__}.{self._expected_type.__qualname__}"
            )
        invalid = [
            name
            for name, component in self._components.items()
            if not isinstance(component, type) or not issubclass(component, expected_type)
        ]
        if invalid:
            raise RegistryError(
                f"existing {self.kind} registration(s) do not inherit "
                f"{expected_type.__name__}: {', '.join(sorted(invalid))}"
            )
        self._expected_type = expected_type

    def _validate(self, component: T) -> None:
        if self._expected_type is None:
            return
        if not isinstance(component, type) or not issubclass(
            component,
            self._expected_type,
        ):
            raise RegistryError(
                f"{self.kind} registrations must inherit "
                f"{self._expected_type.__module__}.{self._expected_type.__qualname__}"
            )

    def add(self, name: str, component: T) -> T:
        """Register a component imperatively and return it unchanged."""

        name_value = cast(object, name)
        if not isinstance(name_value, str) or not name_value.strip():
            raise RegistryError(f"{self.kind} registry name must be non-empty")
        reserved_prefix = next(
            (prefix for prefix in self._reserved_prefixes if name.startswith(prefix)),
            None,
        )
        if reserved_prefix is not None:
            raise RegistryError(
                f"{self.kind} registry names cannot use reserved namespace "
                f"'{reserved_prefix}'"
            )
        self._validate(component)
        existing = self._components.get(name)
        if existing is not None:
            if existing is component:
                return component
            raise RegistryError(
                f"{self.kind} '{name}' already registered by "
                f"{_component_name(existing)}"
            )
        self._components[name] = component
        return component

    def register(self, name: str) -> Callable[[U], U]:
        """Return a decorator that preserves the concrete component type."""

        def decorator(component: U) -> U:
            self.add(name, cast(T, component))
            return component

        return decorator

    def resolve(self, name: str) -> T:
        """Resolve a component or raise an error that lists valid choices."""

        try:
            return self._components[name]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<empty>"
            raise RegistryError(
                f"unknown {self.kind} '{name}'. Available: {available}"
            ) from exc

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Resolve and instantiate or invoke a registered component."""

        component = self.resolve(name)
        if not callable(component):
            raise RegistryError(
                f"registered {self.kind} '{name}' is not callable: "
                f"{_component_name(component)}"
            )
        try:
            return cast(Callable[..., Any], component)(*args, **kwargs)
        except TypeError as exc:
            raise RegistryError(
                f"failed to initialize {self.kind} '{name}' with params "
                f"{kwargs}: {exc}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        """Return registered names in stable order."""

        return tuple(sorted(self._components))

class RegistryCatalog:
    """Application-wide collection of typed component registries."""

    def __init__(self) -> None:
        self.models: Registry[type[Any]] = Registry(
            "model",
            expected_type=nn.Module,
        )
        self.data_builders: Registry[type[Any]] = Registry("data builder")
        self.sampling_artifact_writers: Registry[type[Any]] = Registry(
            "sampling artifact writer"
        )
        self.noise_schedules: Registry[type[Any]] = Registry("noise schedule")
        self.processes: Registry[type[Any]] = Registry("process")
        self.samplers: Registry[type[Any]] = Registry("sampler")
        self.sampling_builders: Registry[type[Any]] = Registry("sampling builder")
        self.training_builders: Registry[type[Any]] = Registry("training builder")
        self.objectives: Registry[type[Any]] = Registry(
            "objective",
            expected_type=nn.Module,
        )
        self.optimizers: Registry[type[Optimizer]] = Registry(
            "optimizer",
            expected_type=Optimizer,
            reserved_prefixes=("torch.optim.",),
        )
        self.lr_schedulers: Registry[type[LRScheduler]] = Registry(
            "lr scheduler",
            expected_type=LRScheduler,
            reserved_prefixes=("torch.optim.lr_scheduler.", "torch.optim."),
        )
        self.loggers: Registry[type[Any]] = Registry("logger")
        self.diagnostics: Registry[type[Any]] = Registry("diagnostic")


REGISTRIES = RegistryCatalog()


__all__ = ["REGISTRIES", "Registry", "RegistryCatalog", "RegistryError"]

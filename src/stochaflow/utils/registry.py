"""Typed, object-oriented component registries."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from importlib import import_module
from typing import Any, Generic, TypeVar, cast

T = TypeVar("T")
U = TypeVar("U")


class RegistryError(ValueError):
    """Raised when a registry operation fails."""


def _component_name(component: object) -> str:
    module = getattr(component, "__module__", type(component).__module__)
    name = getattr(component, "__qualname__", type(component).__qualname__)
    return f"{module}.{name}"


class Registry(Mapping[str, T], Generic[T]):
    """Store and construct named components of one semantic kind.

    A registry is deliberately a small object instead of a public dictionary:
    it owns duplicate checks, optional base-class validation, discovery, module
    loading, and consistent construction errors.
    """

    def __init__(self, kind: str, *, expected_type: type[Any] | None = None) -> None:
        self.kind = kind
        self._expected_type = expected_type
        self._components: dict[str, T] = {}
        self._loaded_modules: set[str] = set()

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

        if not isinstance(name, str) or not name.strip():
            raise RegistryError(f"{self.kind} registry name must be non-empty")
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

    def load_modules(self, modules: Sequence[str]) -> None:
        """Import extension modules once so their decorators can register."""

        for module_name in modules:
            if not isinstance(module_name, str) or not module_name.strip():
                raise RegistryError("registry module names must be non-empty strings")
            if module_name in self._loaded_modules:
                continue
            try:
                import_module(module_name)
            except (ImportError, ModuleNotFoundError) as exc:
                raise RegistryError(
                    f"failed to import registry module '{module_name}': {exc}"
                ) from exc
            self._loaded_modules.add(module_name)


class RegistryCatalog:
    """Application-wide collection of typed component registries."""

    def __init__(self) -> None:
        self.models: Registry[type[Any]] = Registry("model")
        self.dataset_factories: Registry[type[Any]] = Registry("dataset factory")
        self.split_strategies: Registry[type[Any]] = Registry("split strategy")
        self.noise_schedules: Registry[type[Any]] = Registry("noise schedule")
        self.diffusions: Registry[type[Any]] = Registry("diffusion")
        self.objectives: Registry[type[Any]] = Registry("objective")
        self.optimizers: Registry[type[Any]] = Registry("optimizer")
        self.lr_schedulers: Registry[Callable[..., Any]] = Registry("lr scheduler")
        self.loggers: Registry[type[Any]] = Registry("logger")
        self.diagnostics: Registry[type[Any]] = Registry("diagnostic")
        self._loaded_modules: set[str] = set()

    def load_modules(self, modules: Sequence[str]) -> None:
        """Import component modules once for the entire catalog."""

        for module_name in modules:
            if not isinstance(module_name, str) or not module_name.strip():
                raise RegistryError("registry module names must be non-empty strings")
            if module_name in self._loaded_modules:
                continue
            try:
                import_module(module_name)
            except (ImportError, ModuleNotFoundError) as exc:
                raise RegistryError(
                    f"failed to import registry module '{module_name}': {exc}"
                ) from exc
            self._loaded_modules.add(module_name)


REGISTRIES = RegistryCatalog()


__all__ = ["REGISTRIES", "Registry", "RegistryCatalog", "RegistryError"]

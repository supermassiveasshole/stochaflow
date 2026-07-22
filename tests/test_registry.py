"""Tests for generic object-oriented registries."""

import pytest

from stochaflow.utils.registry import Registry, RegistryError


class Base:
    pass


class Component(Base):
    def __init__(self, value: int) -> None:
        self.value = value


def test_registry_decorator_resolve_create_and_names() -> None:
    registry: Registry[type[Base]] = Registry("component", expected_type=Base)
    registered = registry.register("example")(Component)

    assert registered is Component
    assert registry.resolve("example") is Component
    assert registry.names() == ("example",)
    created = registry.create("example", 7)
    assert isinstance(created, Component)
    assert created.value == 7


def test_registry_rejects_duplicate_and_wrong_base() -> None:
    registry: Registry[type[Base]] = Registry("component", expected_type=Base)
    registry.add("example", Component)

    with pytest.raises(RegistryError, match="already registered"):
        registry.add("example", type("Other", (Base,), {}))
    with pytest.raises(RegistryError, match="must inherit"):
        registry.add("wrong", str)  # type: ignore[arg-type]


def test_registry_unknown_error_lists_available_names() -> None:
    registry: Registry[type[Base]] = Registry("component", expected_type=Base)
    registry.add("example", Component)

    with pytest.raises(RegistryError, match="Available: example"):
        registry.resolve("missing")

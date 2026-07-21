"""Family-neutral generative-dynamics semantic root."""

from abc import ABC


class GenerativeDynamics(ABC):
    """Identify an assembled generation direction without prescribing math.

    Algorithm families define their own narrow subclasses. This root has no
    Registry, configuration identity, or universal evaluation method.
    """


__all__ = [
    "GenerativeDynamics",
]

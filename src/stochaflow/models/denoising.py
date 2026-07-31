"""Narrow structural capabilities for denoising models."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class DenoiserChannelLayout(Protocol):
    """Declare a denoiser's static input and raw-output channel counts."""

    @property
    def in_channels(self) -> int:
        """Return the number of channels in one denoising state."""

        ...

    @property
    def out_channels(self) -> int:
        """Return the number of channels in one raw model prediction."""

        ...


__all__ = ["DenoiserChannelLayout"]

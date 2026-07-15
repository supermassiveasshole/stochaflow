"""Abstract contracts for forward noise paths."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class NoiseSchedule(nn.Module, ABC):
    r"""Define a Gaussian forward noise path through its marginal scales.

    A noise schedule answers the process-level question "how much signal and
    noise are present at state time ``t``?". Every implementation represents
    its forward marginal as

    ```math
    x_t = a(t)x_0 + s(t)\epsilon,
    \qquad \epsilon \sim \mathcal{N}(0, I)
    ```

    where ``a(t)`` is the signal scale and ``s(t)`` is the noise scale. The
    contract deliberately does not prescribe betas, a discrete time grid, or
    a particular diffusion algorithm. Those details belong to specialized
    schedule families and diffusion processes.

    Public state time begins at the clean endpoint ``0``. Implementations own
    the remaining time-domain rules and must return scales that broadcast over
    a batch-shaped tensor.
    """

    @property
    def clean_time(self) -> int | float:
        """Return the public clean-state time, fixed by convention to zero."""

        return 0

    @property
    @abstractmethod
    def terminal_time(self) -> int | float:
        """Return the final public state time of the forward noise path."""

    @abstractmethod
    def validate_state_times(self, state_times: torch.Tensor) -> torch.Tensor:
        """Validate and normalize public mathematical state times.

        Args:
            state_times: One state-time value per batch element.

        Returns:
            The normalized state-time tensor used by this schedule.

        Raises:
            TypeError: If the values use an unsupported representation.
            ValueError: If the tensor shape or time domain is invalid.
        """

    @abstractmethod
    def marginal_scales(
        self,
        state_times: torch.Tensor,
        broadcast_shape: torch.Size,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return batch-broadcastable signal and noise scales.

        Args:
            state_times: One public mathematical state time per batch element.
            broadcast_shape: Shape of the sample tensor to which the returned
                scales will be applied. Its leading dimension must match the
                number of state times.

        Returns:
            A pair ``(signal_scale, noise_scale)`` shaped as
            ``(batch, 1, ..., 1)`` for broadcasting over ``broadcast_shape``.
        """

    def signal_to_noise_ratio(
        self,
        state_times: torch.Tensor,
        broadcast_shape: torch.Size,
    ) -> torch.Tensor:
        """Return the marginal signal-to-noise ratio ``a(t)^2 / s(t)^2``."""

        signal_scale, noise_scale = self.marginal_scales(
            state_times,
            broadcast_shape,
        )
        return signal_scale.square() / noise_scale.square()

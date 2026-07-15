"""Linear-beta variance-preserving schedules."""

import torch

from stochaflow.utils.registry import REGISTRIES

from .discrete_vp import DiscreteVPSchedule


@REGISTRIES.noise_schedules.register("linear_beta")
class LinearBetaSchedule(DiscreteVPSchedule):
    r"""Build a discrete VP path from linearly interpolated betas.

    The class owns the beta-native construction policy, while
    :class:`DiscreteVPSchedule` owns validation, coefficient storage, and
    marginal queries. Array index ``i`` stores ``β_{i+1}`` for the
    forward transition into mathematical state ``i + 1``.

    Args:
        num_timesteps: Number ``T`` of forward transitions.
        beta_start: Beta assigned to the first transition ``x_0 -> x_1``.
        beta_end: Beta assigned to the final transition ``x_{T-1} -> x_T``.
        dtype: Floating-point dtype used for all coefficient tables.
    """

    def __init__(
        self,
        num_timesteps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._validate_num_timesteps(num_timesteps)
        self._validate_dtype(dtype)
        self._validate_beta_range(beta_start, beta_end)

        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        super().__init__(
            torch.linspace(
                self.beta_start,
                self.beta_end,
                num_timesteps,
                dtype=dtype,
            )
        )

    @staticmethod
    def _validate_beta_range(beta_start: float, beta_end: float) -> None:
        """Validate the endpoints of the linear beta parameterization."""

        if isinstance(beta_start, bool) or not isinstance(beta_start, (int, float)):
            raise TypeError("beta_start must be numeric")
        if isinstance(beta_end, bool) or not isinstance(beta_end, (int, float)):
            raise TypeError("beta_end must be numeric")
        if not 0 < beta_start < beta_end < 1:
            raise ValueError("expected 0 < beta_start < beta_end < 1")

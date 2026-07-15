"""Object-oriented forward noise paths organized by responsibility.

``NoiseSchedule`` defines the process-level marginal contract,
``DiscreteVPSchedule`` owns the canonical storage and queries for discrete
variance-preserving paths, and concrete subclasses own individual path
construction policies. The public package intentionally exports classes only;
it does not maintain a parallel free-function construction API.
"""

from .base import NoiseSchedule
from .cosine_alpha_bar import CosineAlphaBarSchedule
from .discrete_vp import DiscreteVPSchedule
from .linear_beta import LinearBetaSchedule

__all__ = [
    "CosineAlphaBarSchedule",
    "DiscreteVPSchedule",
    "LinearBetaSchedule",
    "NoiseSchedule",
]

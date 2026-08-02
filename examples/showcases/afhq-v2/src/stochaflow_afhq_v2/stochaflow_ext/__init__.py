"""Register AFHQ-v2 capabilities supported by the installed core runtime."""

from importlib import import_module
from importlib.util import find_spec

from . import source

__all__ = ["source"]

# The published AFHQ v0.1.0 distribution still resolves the released core
# v0.1.0, which predates standalone Evaluation.  Keep its original DataSource
# activation usable, while source-checkout/current-core installs additionally
# register the formal Evaluation Builder and Metric.  The next coordinated core
# and showcase release can make this import unconditional.
if find_spec("stochaflow.evaluation") is not None:
    evaluation = import_module(".evaluation", __package__)
    __all__ += ["evaluation"]

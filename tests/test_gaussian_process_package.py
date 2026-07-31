"""Import-boundary tests for the Gaussian Process package."""

from stochaflow import processes
from stochaflow.processes import gaussian


def test_gaussian_process_package_preserves_root_export_identity() -> None:
    exported_names = (
        "CosineAlphaBarSchedule",
        "DiscreteGaussianDenoisingProcess",
        "DiscreteGaussianProcess",
        "DiscreteVPCoefficients",
        "DiscreteVPSchedule",
        "DiscreteVPScheduleSnapshot",
        "GaussianLogVarianceBounds",
        "GaussianMarginalCoefficientSnapshot",
        "GaussianNoiseSchedule",
        "GaussianScales",
        "LearnedRangeGaussianVarianceProcess",
        "LinearBetaSchedule",
        "SelectedPairGaussianProcess",
        "TabulatedDiscreteVPSchedule",
    )

    for name in exported_names:
        assert getattr(processes, name) is getattr(gaussian, name)


def test_gaussian_process_implementations_have_canonical_module_paths() -> None:
    assert processes.DiscreteGaussianProcess.__module__ == (
        "stochaflow.processes.gaussian.discrete"
    )
    assert processes.LinearBetaSchedule.__module__ == (
        "stochaflow.processes.gaussian.noise_schedules.linear_beta"
    )
    assert processes.DiscreteGaussianDenoisingProcess.__module__ == (
        "stochaflow.processes.gaussian.contracts"
    )

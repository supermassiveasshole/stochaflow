"""Tests for global random-seed policy."""

import pytest
import torch

from stochaflow.utils.seed import set_cpu_seed, set_local_seed, set_seed


def test_deterministic_seed_enables_strict_algorithms_without_disabling_them() -> None:
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        set_seed(11, deterministic=True)

        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark

        set_seed(12, deterministic=False)

        assert torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(
            previous_algorithms,
            warn_only=previous_warn_only,
        )
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark


def test_cpu_seed_does_not_use_all_device_torch_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch,
        "manual_seed",
        lambda unused: pytest.fail("CPU-only seeding used torch.manual_seed"),
    )

    set_cpu_seed(17)


def test_local_cpu_seed_does_not_initialize_accelerators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda unused: pytest.fail("local seeding touched every CUDA device"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda *unused: pytest.fail("CPU seeding touched CUDA state"),
    )

    set_local_seed(19, device="cpu")

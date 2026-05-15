"""Tests for diffusion schedulers."""

import inspect

import torch

from stochaflow.diffusion.schedules import (
    DiffusionScheduler,
    cosine_beta_schedule,
    linear_beta_schedule,
)


def test_linear_beta_schedule_length_matches_timesteps() -> None:
    betas = linear_beta_schedule(10)
    assert betas.shape == (10,)


def test_cosine_beta_schedule_stays_in_valid_range() -> None:
    betas = cosine_beta_schedule(32)
    assert torch.all(betas > 0)
    assert torch.all(betas < 1)


class ToyScheduler(DiffusionScheduler):
    @property
    def coefficient_names(self) -> tuple[str, ...]:
        return ("weights",)

    def __init__(self) -> None:
        super().__init__(num_timesteps=8)
        self.register_coefficient("weights", torch.linspace(0.1, 0.8, self.num_timesteps))


def test_scheduler_is_abstract() -> None:
    assert inspect.isabstract(DiffusionScheduler)
    assert "coefficient_names" in DiffusionScheduler.__abstractmethods__


def test_scheduler_tracks_num_timesteps() -> None:
    scheduler = ToyScheduler()
    assert scheduler.num_timesteps == 8


def test_scheduler_coefficients_broadcast_to_target_shape() -> None:
    scheduler = ToyScheduler()
    timesteps = torch.tensor([0, 3, 7], dtype=torch.long)
    coeffs = scheduler.coefficients_at("weights", timesteps, torch.Size([3, 2, 4, 4]))
    assert coeffs.shape == (3, 1, 1, 1)


def test_extract_preserves_batch_broadcast_shape() -> None:
    scheduler = ToyScheduler()
    values = torch.linspace(0.1, 0.8, scheduler.num_timesteps)
    timesteps = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    extracted = scheduler.extract(values, timesteps, torch.Size([4, 3, 16, 16]))
    assert extracted.shape == (4, 1, 1, 1)


def test_scheduler_rejects_unknown_coefficients() -> None:
    scheduler = ToyScheduler()
    timesteps = torch.tensor([0, 1], dtype=torch.long)
    try:
        scheduler.coefficients_at("missing", timesteps, torch.Size([2, 3]))
    except ValueError as exc:
        assert "Unsupported coefficient name" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown coefficient")

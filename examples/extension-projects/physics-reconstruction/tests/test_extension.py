"""Focused contracts for the physics reconstruction extension."""

from __future__ import annotations

from io import BytesIO
from mmap import mmap
from pathlib import Path
import struct
from typing import Any, cast
from zipfile import ZIP_STORED, ZipFile

import numpy as np
import pytest
import torch

from stochaflow.extensions import (
    DDIMSampler,
    DataBuilderContext,
    DiscreteGaussianProcess,
    GaussianPrediction,
    GaussianTransition,
    MSEObjective,
    PredictionType,
    REGISTRIES,
    SamplingArtifactContext,
    SamplingBatch,
    TrajectoryObserver,
    gaussian_training_target,
)
from stochaflow_physics_reconstruction.stochaflow_ext.data import (
    EpochShuffleSampler,
    KolmogorovDataBuilder,
    TrajectoryTripletDataset,
)
from stochaflow_physics_reconstruction.stochaflow_ext.model import (
    ConditionalDenoiser,
)
from stochaflow_physics_reconstruction.stochaflow_ext.physics import (
    conditioning_gradient,
    correction_gradient,
    vorticity_residual,
)
from stochaflow_physics_reconstruction.stochaflow_ext.sampling import (
    GuidedDDIMSampler,
    PhysicsCorrectionDynamics,
    PhysicsGaussianDynamics,
)
from stochaflow_physics_reconstruction.stochaflow_ext.training import (
    PhysicsDenoisingStrategy,
)
from stochaflow_physics_reconstruction.stochaflow_ext.writers import (
    ReconstructionArtifactWriter,
)
from stochaflow_physics_reconstruction.tools.prepare_kolmogorov import (
    _selected_stored_npy_member,
    prepare,
)
from stochaflow_physics_reconstruction.tools import capacity_check
from stochaflow_physics_reconstruction.tools.prepare_tiny_data import (
    write_tiny_data,
)


def _process(timesteps: int = 4) -> DiscreteGaussianProcess:
    return DiscreteGaussianProcess(
        {
            "name": "linear_beta",
            "params": {
                "num_timesteps": timesteps,
                "beta_start": 0.0001,
                "beta_end": 0.02,
            },
        }
    )


def _model() -> ConditionalDenoiser:
    return ConditionalDenoiser(
        hidden_channels=8,
        num_blocks=1,
        time_embedding_dim=8,
    )


def test_registered_components_are_namespaced() -> None:
    assert "physics-reconstruction.kolmogorov-trajectories" in REGISTRIES.data_builders
    assert "physics-reconstruction.conditional-denoiser" in REGISTRIES.models
    assert "physics-reconstruction.gaussian-denoising" in REGISTRIES.training_builders
    assert "physics-reconstruction.reconstruction" in REGISTRIES.sampling_builders
    assert "physics-reconstruction.guided-ddim" in REGISTRIES.samplers
    assert (
        "physics-reconstruction.reconstruction-artifacts"
        in REGISTRIES.sampling_artifact_writers
    )


def test_capacity_auto_device_prefers_cuda_then_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert capacity_check._device("auto") == torch.device("mps")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert capacity_check._device("auto") == torch.device("cuda")


def test_triplet_dataset_is_raw_stable_and_memory_mapped(tmp_path: Path) -> None:
    paths = write_tiny_data(tmp_path)
    dataset = TrajectoryTripletDataset(paths["trajectories"], (0, 4))
    expected = np.load(paths["trajectories"], mmap_mode="r")[0, :3]

    assert len(dataset) == 16
    assert dataset.sample_shape == (3, 8, 8)
    assert torch.equal(dataset[0], torch.from_numpy(np.array(expected, copy=True)))
    assert isinstance(dataset._data(), np.memmap)


@pytest.mark.parametrize("shape", [(2, 5, 8, 6), (2, 5, 7, 7)])
def test_triplet_dataset_rejects_non_spectral_grids(
    tmp_path: Path,
    shape: tuple[int, int, int, int],
) -> None:
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros(shape, dtype=np.float32))
    with pytest.raises(ValueError, match="even, square"):
        TrajectoryTripletDataset(path, (0, 1))


def test_epoch_shuffle_sampler_rebuilds_resume_permutation() -> None:
    dataset = torch.utils.data.TensorDataset(torch.arange(12))
    uninterrupted = EpochShuffleSampler(len(dataset), seed=19)
    uninterrupted.set_epoch(2)
    expected = list(uninterrupted)

    rebuilt = EpochShuffleSampler(len(dataset), seed=19)
    rebuilt.set_epoch(2)
    assert list(rebuilt) == expected
    assert list(rebuilt) == expected
    wrapped = EpochShuffleSampler(len(dataset), seed=-(2**70))
    wrapped.set_epoch(2**70)
    assert list(wrapped) == list(wrapped)


def test_data_builder_rejects_overlapping_trajectory_partitions(tmp_path: Path) -> None:
    paths = write_tiny_data(tmp_path)
    builder = KolmogorovDataBuilder(
        DataBuilderContext(
            params={
                "path": str(paths["trajectories"]),
                "train_trajectories": [0, 4],
                "validation_trajectories": [3, 5],
                "test_trajectories": [5, 6],
                "loader": {"batch_size": 2},
            },
            seed=1,
        )
    )
    with pytest.raises(ValueError, match="overlap"):
        builder.build()


def _reference_residual(state: torch.Tensor, model: ConditionalDenoiser) -> torch.Tensor:
    _, _, height, width = state.shape
    modes = torch.cat((torch.arange(0, height // 2), torch.arange(-height // 2, 0)))
    kx = modes.reshape(1, 1, height, 1).expand(1, 1, height, width)
    ky = modes.reshape(1, 1, 1, width).expand(1, 1, height, width)
    lap = kx.square() + ky.square()
    safe_lap = lap.clone()
    safe_lap[..., 0, 0] = 1
    spectrum = torch.fft.fft2(state[:, 1:2], dim=(-2, -1))
    psi = spectrum / safe_lap
    velocity_x = torch.fft.ifft2(1j * ky * psi).real
    velocity_y = torch.fft.ifft2(-1j * kx * psi).real
    gradient_x = torch.fft.ifft2(1j * kx * spectrum).real
    gradient_y = torch.fft.ifft2(1j * ky * spectrum).real
    laplacian = torch.fft.ifft2(-lap * spectrum).real
    coordinate = torch.arange(width) * (2.0 * torch.pi / width)
    forcing = -4.0 * torch.cos(4.0 * coordinate).reshape(1, 1, 1, width)
    return (
        (state[:, 2:3] - state[:, 0:1]) / (2.0 * model.time_delta)
        + velocity_x * gradient_x
        + velocity_y * gradient_y
        - laplacian / model.reynolds_number
        + model.linear_damping * state[:, 1:2]
        - forcing
    )


def test_spectral_residual_matches_reference_axis_convention() -> None:
    generator = torch.Generator().manual_seed(8)
    state = torch.randn(2, 3, 8, 8, generator=generator)
    model = _model()
    assert torch.allclose(
        vorticity_residual(state, model),
        _reference_residual(state, model),
        atol=1.0e-5,
        rtol=1.0e-5,
    )


def test_condition_and_correction_gradients_are_separate_and_detached() -> None:
    state = torch.randn(2, 3, 8, 8)
    model = _model()
    with torch.no_grad():
        condition, condition_loss = conditioning_gradient(state, model, strength=0.5)
        correction, correction_loss = correction_gradient(state, model, strength=0.5)
    assert condition.shape == correction.shape == state.shape
    assert not condition.requires_grad and not correction.requires_grad
    assert condition_loss.ndim == correction_loss.ndim == 0
    assert not torch.allclose(condition, correction)


def test_training_strategy_reuses_objective_and_backpropagates() -> None:
    model = _model()
    strategy = PhysicsDenoisingStrategy(
        model,
        _process(),
        MSEObjective(),
        prediction_type="epsilon",
        conditioning_strength=0.0001,
    )
    output = strategy.training_step(torch.randn(2, 3, 8, 8))
    output.loss.backward()
    assert output.loss.ndim == 0
    assert "per_sample_loss" in output.diagnostics
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("prediction_type", ["epsilon", "x0", "v", "score"])
def test_all_gaussian_training_targets_follow_public_semantics(
    prediction_type: str,
) -> None:
    process = _process()
    clean = torch.randn(2, 3, 8, 8)
    noise = torch.randn_like(clean)
    times = torch.tensor([1, 3])
    target = gaussian_training_target(
        process,
        clean=clean,
        noise=noise,
        state_times=times,
        prediction_type=cast(PredictionType, prediction_type),
    )
    scales = process.marginal_scales(times, clean.size())
    expected = {
        "epsilon": noise,
        "x0": clean,
        "v": scales.signal * noise - scales.noise * clean,
        "score": -noise / scales.noise,
    }[prediction_type]
    assert torch.allclose(target, expected)


class _AtomicDynamics(PhysicsCorrectionDynamics):
    def __init__(self, process: DiscreteGaussianProcess) -> None:
        self._process = process
        self.calls = 0

    @property
    def process(self) -> DiscreteGaussianProcess:
        return self._process

    def predict(
        self, state: torch.Tensor, state_times: torch.Tensor
    ) -> GaussianPrediction:
        del state_times
        return GaussianPrediction(state, torch.zeros_like(state), torch.zeros_like(state))

    def evaluate(
        self, state: torch.Tensor, state_times: torch.Tensor
    ) -> tuple[GaussianPrediction, torch.Tensor, dict[str, float]]:
        del state_times
        self.calls += 1
        prediction = GaussianPrediction(
            state,
            torch.zeros_like(state),
            torch.zeros_like(state),
        )
        return prediction, torch.full_like(state, 0.25), {"correction_residual": 1.0}


class _BroadcastCorrectionDynamics(_AtomicDynamics):
    def evaluate(
        self, state: torch.Tensor, state_times: torch.Tensor
    ) -> tuple[GaussianPrediction, torch.Tensor, dict[str, float]]:
        prediction, _, diagnostics = super().evaluate(state, state_times)
        return prediction, torch.tensor(0.25), diagnostics


def test_guided_ddim_applies_atomic_correction_after_public_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dynamics = _AtomicDynamics(_process(2))
    sampler = GuidedDDIMSampler(schedule=[2, 0], eta=0.0)

    def transition(
        self: Any,
        process: Any,
        state: torch.Tensor,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
        prediction: GaussianPrediction,
    ) -> GaussianTransition:
        del self, process, source_times, target_times, prediction
        return GaussianTransition(state + 1.0, torch.zeros_like(state))

    monkeypatch.setattr(type(sampler._ddim), "transition", transition)
    result = sampler.sample(dynamics, torch.zeros(1, 3, 8, 8))
    assert dynamics.calls == 1
    assert torch.equal(result.final_state, torch.full((1, 3, 8, 8), 0.75))


def test_guided_ddim_rejects_broadcast_correction() -> None:
    dynamics = _BroadcastCorrectionDynamics(_process(2))
    sampler = GuidedDDIMSampler(schedule=[2, 0], eta=0.0)
    with pytest.raises(ValueError, match="match the source state shape"):
        sampler.sample(dynamics, torch.zeros(1, 3, 8, 8))


def test_zero_correction_guided_ddim_matches_builtin_and_observer_lifecycle() -> None:
    process = _process()
    model = _model().eval()
    dynamics = PhysicsGaussianDynamics(
        process,
        model,
        prediction_type="epsilon",
        clip_denoised=False,
        conditioning_strength=0.0001,
        correction_strength=0.0,
    )
    initial = torch.randn(1, 3, 8, 8, generator=torch.Generator().manual_seed(5))
    baseline_observer = TrajectoryObserver()
    guided_observer = TrajectoryObserver()
    baseline_generator = torch.Generator().manual_seed(7)
    guided_generator = torch.Generator().manual_seed(7)
    baseline_before = baseline_generator.get_state().clone()
    guided_before = guided_generator.get_state().clone()
    with torch.no_grad():
        baseline = DDIMSampler(schedule=[4, 2, 0], eta=0.0).sample(
            dynamics,
            initial.clone(),
            generator=baseline_generator,
            observer=baseline_observer,
        )
        guided = GuidedDDIMSampler(schedule=[4, 2, 0], eta=0.0).sample(
            dynamics,
            initial.clone(),
            generator=guided_generator,
            observer=guided_observer,
        )
    assert torch.equal(baseline.final_state, guided.final_state)
    assert torch.equal(baseline_generator.get_state(), baseline_before)
    assert torch.equal(guided_generator.get_state(), guided_before)
    expected_lifecycle = [(0, 4, False), (1, 2, False), (2, 0, True)]
    assert [
        (item.step_index, item.coordinate, item.is_final)
        for item in baseline_observer.observations
    ] == expected_lifecycle
    assert [
        (item.step_index, item.coordinate, item.is_final)
        for item in guided_observer.observations
    ] == expected_lifecycle


def test_writer_publishes_memmap_and_cleans_partial_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stochaflow_physics_reconstruction.stochaflow_ext.writers as writers

    first = torch.arange(2 * 3 * 8 * 8, dtype=torch.float64).reshape(2, 3, 8, 8)
    second = torch.full((1, 3, 8, 8), -3.0, dtype=torch.float64)
    context = SamplingArtifactContext(
        tmp_path,
        (SamplingBatch(first), SamplingBatch(second)),
        {"metrics": {"num_samples": 3}},
    )
    writer = ReconstructionArtifactWriter()
    artifacts = writer.write(context)
    saved = np.load(artifacts["reconstructions"], mmap_mode="r")
    assert isinstance(saved, np.memmap)
    assert saved.shape == (3, 3, 8, 8)
    assert saved.dtype == np.float32
    assert np.array_equal(saved[:2], first.numpy().astype(np.float32))
    assert np.array_equal(saved[2:], second.numpy().astype(np.float32))
    with pytest.raises(FileExistsError, match="refuses to replace"):
        writer.write(context)

    mapping = saved.base
    assert isinstance(mapping, mmap)
    mapping.close()
    artifacts["reconstructions"].unlink()
    artifacts["reconstruction_metrics"].unlink()
    replace = writers.os.replace
    calls = 0

    def fail_second(source: Any, destination: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated metrics publication failure")
        replace(source, destination)

    monkeypatch.setattr(writers.os, "replace", fail_second)
    with pytest.raises(OSError, match="simulated"):
        writer.write(context)
    assert not (tmp_path / "reconstructions.npy").exists()
    assert not (tmp_path / "metrics.json").exists()
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_prepare_tool_writes_resized_pairs_stats_and_alignment(tmp_path: Path) -> None:
    reference = np.arange(6 * 5 * 8 * 8, dtype=np.float32).reshape(6, 5, 8, 8)
    sparse = reference[:, :, ::2, ::2]
    reference_path = tmp_path / "reference.npy"
    sparse_path = tmp_path / "sparse.npz"
    np.save(reference_path, reference)
    np.savez(sparse_path, u3232=sparse)
    outputs = prepare(
        reference_path=reference_path,
        sparse_path=sparse_path,
        sparse_key="u3232",
        output_dir=tmp_path / "prepared",
        held_out_trajectories=2,
        smoothing_kernel=0,
    )
    observations = np.load(outputs["observations"], mmap_mode="r")
    assert observations.shape == (2, 5, 8, 8)
    assert outputs["alignment"].is_file()
    assert outputs["statistics"].is_file()


def test_sparse_npz_reader_maps_only_selected_tail(tmp_path: Path) -> None:
    sparse = np.arange(6 * 5 * 4 * 4, dtype=np.float64).reshape(6, 5, 4, 4)
    archive_path = tmp_path / "sparse.npz"
    np.savez(archive_path, ignored=np.ones((2,), dtype=np.float32), u3232=sparse)

    selected = _selected_stored_npy_member(
        archive_path,
        key="u3232",
        count=2,
    )

    assert isinstance(selected, np.memmap)
    assert selected.shape == (2, 5, 4, 4)
    assert np.array_equal(selected, sparse[-2:])


def test_sparse_npz_reader_rejects_compressed_and_fortran_members(
    tmp_path: Path,
) -> None:
    sparse = np.arange(4 * 3 * 2 * 2, dtype=np.float64).reshape(4, 3, 2, 2)
    compressed_path = tmp_path / "compressed.npz"
    fortran_path = tmp_path / "fortran.npz"
    np.savez_compressed(compressed_path, u3232=sparse)
    np.savez(fortran_path, u3232=np.asfortranarray(sparse))

    with pytest.raises(ValueError, match="ZIP_STORED"):
        _selected_stored_npy_member(compressed_path, key="u3232", count=1)
    with pytest.raises(ValueError, match="Fortran order"):
        _selected_stored_npy_member(fortran_path, key="u3232", count=1)


def test_sparse_npz_reader_rejects_malformed_npy_payload(tmp_path: Path) -> None:
    sparse = np.arange(2 * 3 * 4 * 4, dtype=np.float64).reshape(2, 3, 4, 4)
    buffer = BytesIO()
    np.save(buffer, sparse, allow_pickle=False)
    payload = buffer.getvalue().replace(
        b"(2, 3, 4, 4)",
        b"(9, 3, 4, 4)",
        1,
    )
    archive_path = tmp_path / "malformed.npz"
    with ZipFile(archive_path, mode="w", compression=ZIP_STORED) as archive:
        archive.writestr("u3232.npy", payload)

    with pytest.raises(ValueError, match="payload size does not match"):
        _selected_stored_npy_member(archive_path, key="u3232", count=1)


def test_sparse_npz_reader_rejects_crc_corruption(tmp_path: Path) -> None:
    sparse = np.arange(2 * 3 * 4 * 4, dtype=np.float64).reshape(2, 3, 4, 4)
    archive_path = tmp_path / "corrupted.npz"
    np.savez(archive_path, u3232=sparse)
    with ZipFile(archive_path) as archive:
        info = archive.getinfo("u3232.npy")
    local_header = struct.Struct("<4s5H3I2H")
    payload = bytearray(archive_path.read_bytes())
    fields = local_header.unpack_from(payload, info.header_offset)
    member_offset = info.header_offset + local_header.size + fields[-2] + fields[-1]
    payload[member_offset + info.file_size - 1] ^= 0x01
    archive_path.write_bytes(payload)

    with pytest.raises(ValueError, match="integrity validation"):
        _selected_stored_npy_member(archive_path, key="u3232", count=1)


def test_sparse_npz_reader_rejects_inconsistent_local_crc(tmp_path: Path) -> None:
    sparse = np.arange(2 * 3 * 4 * 4, dtype=np.float64).reshape(2, 3, 4, 4)
    archive_path = tmp_path / "crc-metadata.npz"
    np.savez(archive_path, u3232=sparse)
    payload = bytearray(archive_path.read_bytes())
    struct.pack_into("<I", payload, 14, 0)
    archive_path.write_bytes(payload)

    with pytest.raises(ValueError, match="inconsistent ZIP CRC metadata"):
        _selected_stored_npy_member(archive_path, key="u3232", count=1)


def test_sparse_npz_reader_accepts_and_validates_zip64_sizes(
    tmp_path: Path,
) -> None:
    sparse = np.arange(2 * 3 * 4 * 4, dtype=np.float64).reshape(2, 3, 4, 4)
    npy_buffer = BytesIO()
    np.save(npy_buffer, sparse, allow_pickle=False)
    archive_path = tmp_path / "zip64.npz"
    with ZipFile(archive_path, mode="w", compression=ZIP_STORED) as archive:
        with archive.open("u3232.npy", mode="w", force_zip64=True) as member:
            member.write(npy_buffer.getvalue())

    selected = _selected_stored_npy_member(
        archive_path,
        key="u3232",
        count=1,
    )
    assert np.array_equal(selected, sparse[-1:])
    del selected

    local_header = struct.Struct("<4s5H3I2H")
    payload = bytearray(archive_path.read_bytes())
    fields = local_header.unpack_from(payload, 0)
    extra_offset = local_header.size + fields[-2]
    field_id, field_size = struct.unpack_from("<HH", payload, extra_offset)
    assert (field_id, field_size) == (0x0001, 16)
    struct.pack_into("<Q", payload, extra_offset + 4, len(npy_buffer.getvalue()) + 1)
    archive_path.write_bytes(payload)

    with pytest.raises(ValueError, match="ZIP64 uncompressed size"):
        _selected_stored_npy_member(archive_path, key="u3232", count=1)

"""Regression tests for the built-in class-labeled image runtime."""

from __future__ import annotations

import hashlib
import os
import platform
import random
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from stochaflow.data.dataloaders import (
    build_class_labeled_image_data_loader,
    collate_class_labeled_image_batch,
)
from stochaflow.data.datasets import ClassLabeledImageDataset
from stochaflow.data.image_contracts import (
    ClassLabeledImageFileRecord,
    ImageFileRecord,
)
from stochaflow.data.recipe_config import LoaderRecipeConfig
from stochaflow.data.samplers import EpochTaggedIndexSampler
from stochaflow.data.transforms import ImageTransform

LEGACY_INTEL_MACOS_TORCH_22 = (
    sys.platform == "darwin"
    and platform.machine() == "x86_64"
    and torch.__version__.startswith("2.2.")
)
LEGACY_WORKER_CHILD_ENV = "STOCHAFLOW_TEST_LEGACY_WORKER_CHILD"


def patterned_image(*, offset: int = 0) -> Image.Image:
    """Build an asymmetric image whose crops and flips are observable."""

    image = Image.new("RGB", (13, 9))
    for y_coordinate in range(9):
        for x_coordinate in range(13):
            image.putpixel(
                (x_coordinate, y_coordinate),
                (
                    (17 * x_coordinate + offset) % 256,
                    (29 * y_coordinate + offset) % 256,
                    (11 * (x_coordinate + y_coordinate) + offset) % 256,
                ),
            )
    return image


def write_labeled_records(
    root: Path,
    *,
    count: int,
) -> tuple[ClassLabeledImageFileRecord, ...]:
    """Write authenticated fixture images and return their records."""

    records: list[ClassLabeledImageFileRecord] = []
    for index in range(count):
        relative_path = f"sample-{index}.png"
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        patterned_image(offset=index * 7).save(path)
        encoded = path.read_bytes()
        records.append(
            ClassLabeledImageFileRecord(
                image=ImageFileRecord(
                    tree="train",
                    path=relative_path,
                    size_bytes=len(encoded),
                    sha256=hashlib.sha256(encoded).hexdigest(),
                    width=13,
                    height=9,
                ),
                class_label=index % 3,
            )
        )
    return tuple(records)


def training_transform() -> ImageTransform:
    """Return the stochastic transform used by runtime fixtures."""

    return ImageTransform(
        (6, 6),
        role="train",
        channels=3,
        normalize=False,
        random_horizontal_flip=True,
    )


def collect_epoch(
    loader: DataLoader[Any],
    *,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect one explicitly selected loader epoch."""

    sampler = loader.sampler
    assert isinstance(sampler, EpochTaggedIndexSampler)
    sampler.set_epoch(epoch)
    batches = list(loader)
    return (
        torch.cat([batch[0] for batch in batches]),
        torch.cat([batch[1]["class_label"] for batch in batches]),
    )


def process_group_exists(process_group_id: int) -> bool:
    """Return whether a POSIX process group still has a live member."""

    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_process_group_exit(
    process_group_id: int,
    *,
    timeout_seconds: float,
) -> bool:
    """Wait boundedly for a POSIX process group to disappear."""

    deadline = time.monotonic() + timeout_seconds
    while process_group_exists(process_group_id):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def cleanup_process_group(
    process_group_id: int,
    *,
    natural_exit_seconds: float,
) -> str:
    """Boundedly reap an isolated test process group."""

    if wait_for_process_group_exit(
        process_group_id,
        timeout_seconds=natural_exit_seconds,
    ):
        return "exited"
    with suppress(ProcessLookupError):
        os.killpg(process_group_id, signal.SIGTERM)
    if wait_for_process_group_exit(process_group_id, timeout_seconds=5.0):
        return "terminated"
    with suppress(ProcessLookupError):
        os.killpg(process_group_id, signal.SIGKILL)
    if wait_for_process_group_exit(process_group_id, timeout_seconds=5.0):
        return "killed"
    return "survived-sigkill"


def test_seeded_image_transform_is_stateless_and_domain_separated() -> None:
    image = patterned_image()
    transform = training_transform()

    random.seed(101)
    torch.manual_seed(202)
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state().clone()
    first = transform(image, random_seed=303)
    assert random.getstate() == python_state
    assert torch.equal(torch.random.get_rng_state(), torch_state)

    random.seed(404)
    torch.manual_seed(505)
    repeated = transform(image, random_seed=303)
    assert torch.equal(repeated, first)
    variants = [transform(image, random_seed=seed) for seed in range(16)]
    assert any(not torch.equal(variant, variants[0]) for variant in variants[1:])

    crop_only = ImageTransform(
        (6, 6),
        role="train",
        channels=3,
        normalize=False,
        random_horizontal_flip=False,
    )(image, random_seed=303)
    assert torch.equal(first, crop_only) or torch.equal(
        first,
        torch.flip(crop_only, dims=(-1,)),
    )

    random.seed(606)
    python_state = random.getstate()
    crop_only_transform = ImageTransform(
        (6, 6),
        role="train",
        channels=3,
        normalize=False,
        random_horizontal_flip=False,
    )
    crop_only_transform(image)
    assert random.getstate() == python_state
    with pytest.raises(TypeError, match="random_seed"):
        transform(image, random_seed=True)


def test_dataset_derives_epoch_seed_and_reuses_hash_verification(
    tmp_path: Path,
) -> None:
    records = write_labeled_records(tmp_path, count=1)
    dataset = ClassLabeledImageDataset(
        roots={"train": tmp_path},
        records=records,
        transform=training_transform(),
        seed=41,
    )

    first_image, first_label = dataset[(2, 0)]
    repeated_image, repeated_label = dataset[(2, 0)]
    assert first_label == repeated_label == 0
    assert torch.equal(first_image, repeated_image)
    assert dataset[0][0].shape == (3, 6, 6)

    path = tmp_path / records[0].image.path
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="size changed"):
        dataset[(2, 0)]

    with pytest.raises(ValueError, match="non-negative"):
        dataset[(-1, 0)]
    with pytest.raises(IndexError):
        dataset[(0, 1)]


def test_epoch_tagged_sampler_always_propagates_epoch() -> None:
    dataset = torch.utils.data.TensorDataset(torch.arange(5))
    ordered = EpochTaggedIndexSampler(
        dataset,
        seed=17,
        shuffle=False,
    )
    assert list(ordered) == [(0, index) for index in range(5)]
    ordered.set_epoch(3)
    assert list(ordered) == [(3, index) for index in range(5)]

    shuffled = EpochTaggedIndexSampler(
        dataset,
        seed=17,
        shuffle=True,
    )
    initial = list(shuffled)
    shuffled.set_epoch(2)
    selected = list(shuffled)
    assert sorted(index for _, index in selected) == list(range(5))
    assert all(epoch == 2 for epoch, _ in selected)
    assert selected != initial
    with pytest.raises(ValueError, match="non-negative"):
        shuffled.set_epoch(True)


@pytest.mark.skipif(
    LEGACY_INTEL_MACOS_TORCH_22
    and os.environ.get(LEGACY_WORKER_CHILD_ENV) != "1",
    reason="runs in the bounded fresh-interpreter regression on this platform",
)
def test_class_labeled_loader_is_worker_count_independent(
    tmp_path: Path,
) -> None:
    records = write_labeled_records(tmp_path, count=8)
    single_dataset = ClassLabeledImageDataset(
        roots={"train": tmp_path},
        records=records,
        transform=training_transform(),
        seed=29,
    )
    worker_dataset = ClassLabeledImageDataset(
        roots={"train": tmp_path},
        records=records,
        transform=training_transform(),
        seed=29,
    )
    single_config = LoaderRecipeConfig(
        batch_size=3,
        num_workers=0,
        shuffle=True,
        drop_last=False,
        pin_memory=False,
        persistent_workers=False,
    )
    worker_config = LoaderRecipeConfig(
        batch_size=3,
        num_workers=1,
        shuffle=True,
        drop_last=False,
        pin_memory=False,
        # Recreate and synchronously reap workers at each epoch. PyTorch 2.2 on
        # Intel macOS can retain a persistent worker during interpreter exit;
        # policy forwarding is covered without starting workers below.
        persistent_workers=False,
    )
    single_loader = build_class_labeled_image_data_loader(
        single_dataset,
        single_config,
        training=True,
        seed=29,
    )
    worker_loader = build_class_labeled_image_data_loader(
        worker_dataset,
        worker_config,
        training=True,
        seed=29,
    )
    assert single_loader is not None
    assert worker_loader is not None

    for epoch in (2, 3):
        single_images, single_labels = collect_epoch(
            single_loader,
            epoch=epoch,
        )
        worker_images, worker_labels = collect_epoch(
            worker_loader,
            epoch=epoch,
        )
        assert torch.equal(worker_labels, single_labels)
        assert torch.equal(worker_images, single_images)

    evaluation_loader = build_class_labeled_image_data_loader(
        single_dataset,
        single_config,
        training=False,
        seed=29,
    )
    assert evaluation_loader is not None
    assert list(evaluation_loader.sampler) == list(range(len(single_dataset)))

    images, conditions = collate_class_labeled_image_batch(
        [single_dataset[0], single_dataset[1]]
    )
    assert images.shape == (2, 3, 6, 6)
    assert conditions["class_label"].dtype == torch.long
    with pytest.raises(ValueError, match="must not be empty"):
        collate_class_labeled_image_batch([])


def test_class_labeled_loader_preserves_persistent_worker_policy(
    tmp_path: Path,
) -> None:
    dataset = ClassLabeledImageDataset(
        roots={"train": tmp_path},
        records=write_labeled_records(tmp_path, count=1),
        transform=training_transform(),
        seed=29,
    )
    loader = build_class_labeled_image_data_loader(
        dataset,
        LoaderRecipeConfig(
            batch_size=1,
            num_workers=1,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
            persistent_workers=True,
        ),
        training=True,
        seed=29,
    )

    assert loader is not None
    assert loader.num_workers == 1
    assert loader.persistent_workers is True


@pytest.mark.skipif(
    not LEGACY_INTEL_MACOS_TORCH_22,
    reason="targets the legacy Intel macOS PyTorch worker lifecycle",
)
def test_legacy_macos_worker_loader_exits_in_fresh_interpreter(
    tmp_path: Path,
) -> None:
    """Bound the PyTorch 2.2 worker-lifecycle regression in its own process."""

    target = (
        f"{Path(__file__).resolve()}::"
        "test_class_labeled_loader_is_worker_count_independent"
    )
    output_path = tmp_path / "worker-exit-regression.log"
    timed_out = False
    leader_survived = False
    with output_path.open("w", encoding="utf-8") as output_stream:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "pytest", "-q", target],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, LEGACY_WORKER_CHILD_ENV: "1"},
            stdout=output_stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            try:
                process.wait(timeout=60.0)
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            group_cleanup = cleanup_process_group(
                process.pid,
                natural_exit_seconds=0.0 if timed_out else 5.0,
            )
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    process.kill()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    leader_survived = True

    output = output_path.read_text(encoding="utf-8")
    if timed_out:
        pytest.fail(
            "Intel macOS PyTorch worker test did not exit within 60 seconds; "
            f"the isolated process group was terminated.\n{output}",
            pytrace=False,
        )

    assert not leader_survived, (
        "isolated Intel macOS pytest leader survived SIGKILL\n" + output
    )
    assert group_cleanup == "exited", (
        "Intel macOS PyTorch worker test leaked its isolated process group; "
        f"cleanup={group_cleanup!r}\n{output}"
    )
    assert process.returncode == 0, output

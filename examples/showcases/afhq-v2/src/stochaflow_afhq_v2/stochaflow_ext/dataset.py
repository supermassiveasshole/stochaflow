"""Verified labeled datasets and stateless augmentation for AFHQ-v2."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import cast

import torch
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import functional as vision_functional

from stochaflow.extensions import ImageFileRecord
from stochaflow_afhq_v2.preparation import load_verified_prepared_image
from stochaflow_afhq_v2.stochaflow_ext.config import AFHQV2ImageRecipeConfig

type AFHQV2SampleIndex = int | tuple[int, int]


def _class_labels(
    records: Sequence[ImageFileRecord],
    *,
    role: str,
    class_mapping: Mapping[str, int],
    image: AFHQV2ImageRecipeConfig,
) -> tuple[int, ...]:
    if not records:
        raise ValueError(f"AFHQ-v2 {role} partition must not be empty")
    labels: list[int] = []
    seen_classes: set[str] = set()
    for record in records:
        if record.tree != role:
            raise ValueError(
                f"AFHQ-v2 {role} record uses tree {record.tree!r}"
            )
        parts = PurePosixPath(record.path).parts
        if (
            len(parts) != 2
            or PurePosixPath(record.path).suffix.lower() != ".png"
            or parts[0] not in class_mapping
        ):
            raise ValueError(
                f"AFHQ-v2 {role} record has an invalid class path: "
                f"{record.path!r}"
            )
        if record.width != image.width or record.height != image.height:
            raise ValueError(
                f"AFHQ-v2 {role} record {record.path!r} has authenticated "
                f"size {record.width}x{record.height}, expected "
                f"{image.width}x{image.height}"
            )
        seen_classes.add(parts[0])
        labels.append(class_mapping[parts[0]])
    missing = sorted(set(class_mapping) - seen_classes)
    if missing:
        raise ValueError(
            f"AFHQ-v2 {role} partition is missing classes: "
            + ", ".join(missing)
        )
    return tuple(labels)


def _augmentation_bit(
    *,
    seed: int,
    epoch: int,
    record: ImageFileRecord,
) -> bool:
    identity = (
        f"stochaflow.afhq-v2.hflip.v1\0{seed}\0{epoch}\0"
        f"{record.tree}\0{record.path}"
    ).encode()
    return bool(hashlib.sha256(identity).digest()[0] & 1)


class AFHQV2ClassDataset(
    Dataset[tuple[torch.Tensor, int]]
):
    """Read one authenticated partition and emit image/class pairs."""

    def __init__(
        self,
        *,
        roots: Mapping[str, Path],
        records: Sequence[ImageFileRecord],
        role: str,
        class_mapping: Mapping[str, int],
        image: AFHQV2ImageRecipeConfig,
        seed: int,
        augment: bool,
    ) -> None:
        seed_value = cast(object, seed)
        if isinstance(seed_value, bool) or not isinstance(seed_value, int):
            raise TypeError("AFHQ-v2 data seed must be an integer")
        self.roots = dict(roots)
        self.records = tuple(records)
        self.role = role
        self.image = image
        self.seed = seed
        self.augment = augment
        self.labels = _class_labels(
            self.records,
            role=role,
            class_mapping=class_mapping,
            image=image,
        )
        if role not in self.roots:
            raise ValueError(f"AFHQ-v2 payload is missing the {role} root")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: AFHQV2SampleIndex,
    ) -> tuple[torch.Tensor, int]:
        epoch = 0
        sample_index: object = index
        if isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError("AFHQ-v2 epoch-tagged index must contain two values")
            epoch, sample_index = index
        epoch_value = cast(object, epoch)
        if (
            isinstance(epoch_value, bool)
            or not isinstance(epoch_value, int)
            or epoch_value < 0
        ):
            raise ValueError("AFHQ-v2 sample epoch must be non-negative")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise TypeError("AFHQ-v2 sample index must be an integer")
        if sample_index < 0 or sample_index >= len(self.records):
            raise IndexError(sample_index)
        record = self.records[sample_index]
        source = load_verified_prepared_image(
            self.roots[record.tree],
            record.path,
            expected_size_bytes=record.size_bytes,
            expected_sha256=record.sha256,
            expected_width=record.width,
            expected_height=record.height,
        )
        tensor = vision_functional.pil_to_tensor(source)
        tensor = vision_functional.convert_image_dtype(
            tensor,
            torch.float32,
        )
        if (
            self.augment
            and self.image.random_horizontal_flip
            and _augmentation_bit(seed=self.seed, epoch=epoch, record=record)
        ):
            tensor = torch.flip(tensor, dims=(-1,))
        if self.image.normalize:
            tensor = tensor.mul(2.0).sub(1.0)
        expected_shape = (3, self.image.height, self.image.width)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"AFHQ-v2 tensor shape {tuple(tensor.shape)} does not match "
                f"{expected_shape}"
            )
        return tensor.contiguous(), self.labels[sample_index]


class AFHQV2EpochSampler(Sampler[tuple[int, int]]):
    """Emit deterministic epoch-tagged indices for persistent workers."""

    def __init__(self, size: int, *, seed: int, shuffle: bool) -> None:
        size_value = cast(object, size)
        if (
            isinstance(size_value, bool)
            or not isinstance(size_value, int)
            or size_value <= 0
        ):
            raise ValueError("AFHQ-v2 sampler size must be positive")
        seed_value = cast(object, seed)
        if isinstance(seed_value, bool) or not isinstance(seed_value, int):
            raise TypeError("AFHQ-v2 sampler seed must be an integer")
        shuffle_value = cast(object, shuffle)
        if not isinstance(shuffle_value, bool):
            raise TypeError("AFHQ-v2 sampler shuffle must be boolean")
        self.size = size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self) -> Iterator[tuple[int, int]]:
        if self.shuffle:
            encoded = (
                f"stochaflow.afhq-v2.shuffle.v1\0{self.seed}\0{self.epoch}"
            ).encode()
            epoch_seed = int.from_bytes(
                hashlib.sha256(encoded).digest()[:8],
                byteorder="little",
            )
            generator = torch.Generator().manual_seed(epoch_seed)
            indices = cast(list[int], torch.randperm(
                self.size,
                generator=generator,
            ).tolist())
        else:
            indices = list(range(self.size))
        yield from ((self.epoch, index) for index in indices)

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Select the permutation and augmentation identity for an epoch."""

        epoch_value = cast(object, epoch)
        if (
            isinstance(epoch_value, bool)
            or not isinstance(epoch_value, int)
            or epoch_value < 0
        ):
            raise ValueError("AFHQ-v2 sampler epoch must be non-negative")
        self.epoch = epoch


def collate_afhq_v2_class_batch(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Stack AFHQ-v2 images and preserve class labels as one condition."""

    if not batch:
        raise ValueError("AFHQ-v2 batch must not be empty")
    return (
        torch.stack([image for image, _ in batch]),
        {
            "class_label": torch.tensor(
                [label for _, label in batch],
                dtype=torch.long,
            )
        },
    )


__all__ = [
    "AFHQV2ClassDataset",
    "AFHQV2EpochSampler",
    "AFHQV2SampleIndex",
    "collate_afhq_v2_class_batch",
]

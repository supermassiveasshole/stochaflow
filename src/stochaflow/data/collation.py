"""Image-only batch collation for heterogeneous dataset sources."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


class ImageBatchCollator:
    """Normalize valid dataset samples into one homogeneous image batch.

    Dataset factories may expose a bare image tensor or retain source-specific
    values after the image in a tuple/list. Diffusion training only consumes
    images, so the loader boundary deliberately removes those auxiliary values
    before samples from different sources are stacked together.
    """

    @staticmethod
    def image_from_sample(sample: Any) -> torch.Tensor:
        """Extract the image tensor required by the dataset factory contract."""

        image = sample[0] if isinstance(sample, (tuple, list)) and sample else sample
        if not isinstance(image, torch.Tensor):
            raise TypeError(
                "dataset samples must be image tensors or non-empty tuple/list "
                "values whose first element is an image tensor"
            )
        return image

    def __call__(self, samples: Sequence[Any]) -> torch.Tensor:
        """Stack images while intentionally discarding source-specific payloads."""

        if not samples:
            raise ValueError("cannot collate an empty image batch")
        images = [self.image_from_sample(sample) for sample in samples]
        return torch.stack(images, dim=0)


__all__ = ["ImageBatchCollator"]

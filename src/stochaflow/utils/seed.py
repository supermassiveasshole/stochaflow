"""Random seed helpers."""

from collections.abc import Iterator
from contextlib import contextmanager
import random

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch RNG state."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@contextmanager
def preserve_global_rng_state(
    device: torch.device | str | None = None,
) -> Iterator[None]:
    """Restore Python, NumPy, and Torch global RNG streams after a callback."""

    target = None if device is None else torch.device(device)
    if torch.cuda.is_initialized():
        cuda_devices = list(range(torch.cuda.device_count()))
    elif target is not None and target.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device() if target.index is None else target.index
        ]
    else:
        cuda_devices = []
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

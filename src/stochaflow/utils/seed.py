"""Random seed helpers."""

import random
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch RNG state."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    _configure_determinism(deterministic)


def set_cpu_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed host RNG streams without initializing accelerator runtimes."""

    random.seed(seed)
    np.random.seed(seed)
    torch.default_generator.manual_seed(seed)

    _configure_determinism(deterministic)


def set_local_seed(
    seed: int,
    *,
    device: torch.device | str,
    deterministic: bool = False,
) -> None:
    """Seed host streams and only the explicitly process-owned accelerator."""

    selected = torch.device(device)
    set_cpu_seed(seed, deterministic=deterministic)
    if selected.type == "cuda":
        if selected.index is None:
            raise ValueError("local CUDA seeding requires an indexed device")
        generator = torch.Generator(device=selected)
        generator.manual_seed(seed)
        torch.cuda.set_rng_state(generator.get_state(), selected)
    elif selected.type == "mps":
        torch.mps.manual_seed(seed)
    elif selected.type != "cpu":
        raise ValueError(f"unsupported local seed device type: {selected.type}")


def _configure_determinism(deterministic: bool) -> None:
    """Enable the process-wide deterministic policy when requested."""

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@contextmanager
def preserve_global_rng_state(
    device: torch.device | str | None = None,
) -> Generator[None, None, None]:
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
    preserve_mps = (
        target is not None
        and target.type == "mps"
        and torch.backends.mps.is_available()
    )
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state().clone()
    cuda_states = tuple(
        (device_index, torch.cuda.get_rng_state(device_index).clone())
        for device_index in cuda_devices
    )
    mps_state = torch.mps.get_rng_state().clone() if preserve_mps else None

    def restore_states() -> list[tuple[str, BaseException]]:
        failures: list[tuple[str, BaseException]] = []

        def attempt(label: str, action: Callable[[], None]) -> None:
            try:
                action()
            except BaseException as error:  # noqa: BLE001
                failures.append((label, error))

        attempt("restore Python RNG state", lambda: random.setstate(python_state))
        attempt(
            "restore NumPy RNG state",
            lambda: np.random.set_state(numpy_state),
        )
        attempt(
            "restore CPU Torch RNG state",
            lambda: torch.random.set_rng_state(cpu_state),
        )
        for device_index, cuda_state in cuda_states:
            attempt(
                f"restore CUDA device {device_index} RNG state",
                lambda index=device_index, state=cuda_state: torch.cuda.set_rng_state(
                    state, index
                ),
            )
        if mps_state is not None:
            saved_mps_state = mps_state
            attempt(
                "restore MPS RNG state",
                lambda: torch.mps.set_rng_state(saved_mps_state),
            )
        return failures

    def add_failure_notes(
        primary: BaseException,
        failures: list[tuple[str, BaseException]],
    ) -> None:
        for label, failure in failures:
            try:
                detail = str(failure)
            except BaseException:  # noqa: BLE001
                detail = "<exception text unavailable>"
            with suppress(BaseException):
                BaseException.add_note(
                    primary,
                    f"{label}: {type(failure).__name__}: {detail}",
                )

    try:
        yield
    except BaseException as error:
        add_failure_notes(error, restore_states())
        raise
    failures = restore_states()
    if failures:
        primary = failures[0][1]
        add_failure_notes(primary, failures[1:])
        raise primary

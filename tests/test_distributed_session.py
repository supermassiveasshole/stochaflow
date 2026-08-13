"""Tests for fixed-topology distributed session ownership."""

from __future__ import annotations

from datetime import timedelta
from typing import Literal

import pytest
import torch

from stochaflow.data.ranked import DataRankContext
from stochaflow.training.distributed import (
    DistributedSession,
    DistributedTopology,
    parse_torchrun_environment,
)

DistributedReduction = Literal["sum", "min", "max"]


class FakeProcessGroupRuntime:
    """Deterministic in-process stand-in for Gloo control operations."""

    def __init__(self, *, topology: DistributedTopology) -> None:
        self.topology = topology
        self.available = True
        self.initialized = False
        self.destroy_error: BaseException | None = None
        self.initialize_error: BaseException | None = None
        self.reported_rank: int | None = None
        self.events: list[str] = []
        self.gathered_digests: list[object] | None = None

    def is_available(self) -> bool:
        return self.available

    def is_initialized(self) -> bool:
        return self.initialized

    def initialize(
        self,
        *,
        backend: str,
        rank: int,
        world_size: int,
        timeout: timedelta,
    ) -> None:
        assert backend == "gloo"
        assert rank == self.topology.rank
        assert world_size == self.topology.world_size
        assert timeout > timedelta(0)
        self.events.append("initialize")
        self.initialized = True
        if self.initialize_error is not None:
            raise self.initialize_error

    def destroy(self) -> None:
        self.events.append("destroy")
        self.initialized = False
        if self.destroy_error is not None:
            raise self.destroy_error

    def rank(self) -> int:
        return (
            self.reported_rank
            if self.reported_rank is not None
            else self.topology.rank
        )

    def world_size(self) -> int:
        return self.topology.world_size

    def backend(self) -> str:
        return "gloo"

    def broadcast_object_list(
        self,
        values: list[object],
        *,
        source_rank: int,
    ) -> None:
        assert source_rank == 0
        self.events.append("broadcast")

    def gather_object(
        self,
        value: object,
        output: list[object] | None,
        *,
        destination_rank: int,
    ) -> None:
        assert destination_rank == 0
        self.events.append("gather")
        if output is not None:
            output[:] = [value for _ in range(self.topology.world_size)]

    def all_gather_object(self, output: list[object], value: object) -> None:
        self.events.append("all_gather")
        output[:] = (
            list(self.gathered_digests)
            if self.gathered_digests is not None
            else [value for _ in range(self.topology.world_size)]
        )

    def all_reduce(
        self,
        value: torch.Tensor,
        *,
        reduction: DistributedReduction,
    ) -> None:
        self.events.append(f"all_reduce:{reduction}")
        if reduction == "sum":
            value.mul_(self.topology.world_size)


def torchrun_environment(*, rank: int = 0, world_size: int = 2) -> dict[str, str]:
    """Return one valid fixed single-node launch environment."""

    return {
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "LOCAL_WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        "GROUP_RANK": "0",
        "GROUP_WORLD_SIZE": "1",
        "ROLE_RANK": str(rank),
        "ROLE_WORLD_SIZE": str(world_size),
        "TORCHELASTIC_RESTART_COUNT": "0",
        "TORCHELASTIC_MAX_RESTARTS": "0",
    }


def test_parse_torchrun_environment_accepts_fixed_single_node() -> None:
    topology = parse_torchrun_environment(torchrun_environment(rank=1))

    assert topology == DistributedTopology(
        rank=1,
        local_rank=1,
        world_size=2,
        local_world_size=2,
    )
    assert not topology.is_primary


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"WORLD_SIZE": "4", "ROLE_WORLD_SIZE": "4"},
            "world_size to equal local_world_size",
        ),
        ({"LOCAL_RANK": "1"}, "rank to equal local_rank"),
        ({"GROUP_RANK": "1"}, "GROUP_RANK=0"),
        ({"GROUP_WORLD_SIZE": "2"}, "GROUP_WORLD_SIZE=1"),
        ({"ROLE_RANK": "1"}, "ROLE_RANK to equal RANK"),
        ({"ROLE_WORLD_SIZE": "4"}, "ROLE_WORLD_SIZE to equal WORLD_SIZE"),
        ({"TORCHELASTIC_MAX_RESTARTS": "1"}, "does not accept elastic restarts"),
        ({"MASTER_PORT": "70000"}, "must not exceed 65535"),
        ({"RANK": "+0"}, "canonical decimal integer"),
    ],
)
def test_parse_torchrun_environment_rejects_unsupported_layouts(
    updates: dict[str, str],
    message: str,
) -> None:
    environ = torchrun_environment()
    environ.update(updates)

    with pytest.raises(ValueError, match=message):
        parse_torchrun_environment(environ)


def test_session_binds_device_before_initializing_and_destroys() -> None:
    topology = parse_torchrun_environment(torchrun_environment())
    runtime = FakeProcessGroupRuntime(topology=topology)

    def bind_device(backend: str, value: DistributedTopology) -> torch.device:
        assert backend == "gloo"
        assert value == topology
        assert not runtime.initialized
        runtime.events.append("bind")
        return torch.device("cpu")

    session = DistributedSession.from_environment(
        backend="gloo",
        environ=torchrun_environment(),
        runtime=runtime,
        device_binder=bind_device,
    )
    with session as active:
        assert active is session
        assert active.device == torch.device("cpu")
        assert active.is_primary
        assert active.data_rank_context == DataRankContext(rank=0, world_size=2)
        assert runtime.initialized

    assert runtime.events == ["bind", "initialize", "destroy"]
    with pytest.raises(RuntimeError, match="not active"):
        _ = session.collectives
    assert not runtime.initialized


def test_control_collectives_are_narrow_and_validate_values() -> None:
    topology = parse_torchrun_environment(torchrun_environment())
    runtime = FakeProcessGroupRuntime(topology=topology)
    with DistributedSession.from_environment(
        backend="gloo",
        environ=torchrun_environment(),
        runtime=runtime,
    ) as session:
        collectives = session.collectives
        assert collectives.broadcast_from_primary({"decision": "continue"}) == {
            "decision": "continue"
        }
        assert collectives.gather_to_primary("ready") == ("ready", "ready")
        assert collectives.all_true(True)
        assert collectives.all_equal({"b": [2], "a": (1,)})
        assert collectives.sum_int(3) == 6
        assert collectives.sum_float(1.25) == 2.5
        assert collectives.min_int(3) == 3
        assert collectives.max_int(3) == 3
        with pytest.raises(TypeError, match="exact integer"):
            collectives.sum_int(True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="non-finite"):
            collectives.all_equal(float("nan"))


def test_all_equal_returns_false_for_a_different_rank_digest() -> None:
    topology = parse_torchrun_environment(torchrun_environment())
    runtime = FakeProcessGroupRuntime(topology=topology)
    runtime.gathered_digests = ["a" * 64, "b" * 64]
    with DistributedSession.from_environment(
        backend="gloo",
        environ=torchrun_environment(),
        runtime=runtime,
    ) as session:
        assert not session.collectives.all_equal("same locally")


def test_body_error_remains_primary_when_destroy_also_fails() -> None:
    topology = parse_torchrun_environment(torchrun_environment())
    runtime = FakeProcessGroupRuntime(topology=topology)
    runtime.destroy_error = RuntimeError("destroy failed")

    with (
        pytest.raises(ValueError, match="training failed") as captured,
        DistributedSession.from_environment(
            backend="gloo",
            environ=torchrun_environment(),
            runtime=runtime,
        ),
    ):
        raise ValueError("training failed")

    assert captured.value.__notes__ == [
        "distributed process-group teardown also failed: RuntimeError: destroy failed"
    ]


def test_session_destroys_a_partially_initialized_group() -> None:
    topology = parse_torchrun_environment(torchrun_environment())
    runtime = FakeProcessGroupRuntime(topology=topology)
    runtime.initialize_error = ValueError("initialization failed")

    with (
        pytest.raises(ValueError, match="initialization failed"),
        DistributedSession.from_environment(
            backend="gloo",
            environ=torchrun_environment(),
            runtime=runtime,
        ),
    ):
        pytest.fail("an initialization failure cannot enter the body")

    assert runtime.events == ["initialize", "destroy"]
    assert not runtime.initialized


def test_session_destroys_a_group_with_mismatched_runtime_identity() -> None:
    topology = parse_torchrun_environment(torchrun_environment())
    runtime = FakeProcessGroupRuntime(topology=topology)
    runtime.reported_rank = 1

    with (
        pytest.raises(RuntimeError, match="rank differs"),
        DistributedSession.from_environment(
            backend="gloo",
            environ=torchrun_environment(),
            runtime=runtime,
        ),
    ):
        pytest.fail("a mismatched group cannot enter the body")

    assert runtime.events == ["initialize", "destroy"]
    assert not runtime.initialized


def test_destroy_failure_is_primary_after_a_successful_body() -> None:
    topology = parse_torchrun_environment(torchrun_environment())
    runtime = FakeProcessGroupRuntime(topology=topology)
    runtime.destroy_error = RuntimeError("destroy failed")

    with (
        pytest.raises(RuntimeError, match="destroy failed"),
        DistributedSession.from_environment(
            backend="gloo",
            environ=torchrun_environment(),
            runtime=runtime,
        ),
    ):
        pass


def test_session_rejects_an_existing_process_group_without_destroying_it() -> None:
    topology = parse_torchrun_environment(torchrun_environment())
    runtime = FakeProcessGroupRuntime(topology=topology)
    runtime.initialized = True
    session = DistributedSession.from_environment(
        backend="gloo",
        environ=torchrun_environment(),
        runtime=runtime,
    )

    with pytest.raises(RuntimeError, match="refuses to adopt"):
        session.__enter__()

    assert runtime.events == []
    assert runtime.initialized

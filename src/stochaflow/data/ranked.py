"""Narrow public contracts for rank-aware runtime data execution."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable


def _require_integer(value: object, *, path: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{path} must be a {qualifier} integer")
    return cast(int, value)


def _require_digest(value: object, *, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class DataRankContext:
    """Immutable global-rank projection injected into one DataBuilder."""

    rank: int
    world_size: int

    def __post_init__(self) -> None:
        rank = _require_integer(self.rank, path="data rank", minimum=0)
        world_size = _require_integer(
            self.world_size,
            path="data world_size",
            minimum=1,
        )
        if rank >= world_size:
            raise ValueError("data rank must be smaller than world_size")


@dataclass(frozen=True, slots=True)
class RankedEpochDataIdentity:
    """Bounded semantic identity for reconstructing ranked epoch data."""

    provider: str
    digest: str

    def __post_init__(self) -> None:
        provider = cast(object, self.provider)
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("ranked data identity provider must be non-empty")
        _require_digest(self.digest, path="ranked data identity digest")

    def to_dict(self) -> dict[str, str]:
        """Serialize the identity without runtime handles or receipts."""

        return {"provider": self.provider, "digest": self.digest}


@dataclass(frozen=True, slots=True)
class RankedBatchFacts:
    """Builder-authenticated facts for one opaque training microbatch."""

    ordinal: int
    sample_count: int
    loss_weight: float
    assignment_token: str

    def __post_init__(self) -> None:
        _require_integer(self.ordinal, path="ranked batch ordinal", minimum=0)
        _require_integer(
            self.sample_count,
            path="ranked batch sample_count",
            minimum=1,
        )
        loss_weight = cast(object, self.loss_weight)
        if (
            isinstance(loss_weight, bool)
            or not isinstance(loss_weight, (int, float))
            or not math.isfinite(loss_weight)
            or loss_weight <= 0
        ):
            raise ValueError("ranked batch loss_weight must be finite and positive")
        _require_digest(
            self.assignment_token,
            path="ranked batch assignment_token",
        )


@dataclass(frozen=True, slots=True)
class RankedTrainEpochPlan:
    """Exact equal-window assignment for one rank and one training epoch."""

    data_identity: RankedEpochDataIdentity
    plan_digest: str
    expected_terminal_token: str
    epoch: int
    rank: int
    world_size: int
    microbatches_per_window: int
    window_count: int
    samples_per_microbatch: int
    local_assigned_samples: int
    global_assigned_samples: int
    global_dropped_samples: int
    assignment_digest: str
    requested_max_microbatches: int | None = None

    def __post_init__(self) -> None:
        data_identity = cast(object, self.data_identity)
        if not isinstance(data_identity, RankedEpochDataIdentity):
            raise TypeError("ranked train data_identity has the wrong type")
        _require_digest(self.plan_digest, path="ranked train plan_digest")
        _require_digest(
            self.expected_terminal_token,
            path="ranked train expected_terminal_token",
        )
        _require_integer(self.epoch, path="ranked train epoch", minimum=0)
        rank_context = DataRankContext(self.rank, self.world_size)
        microbatches = _require_integer(
            self.microbatches_per_window,
            path="ranked train microbatches_per_window",
            minimum=1,
        )
        windows = _require_integer(
            self.window_count,
            path="ranked train window_count",
            minimum=1,
        )
        samples = _require_integer(
            self.samples_per_microbatch,
            path="ranked train samples_per_microbatch",
            minimum=1,
        )
        local_samples = _require_integer(
            self.local_assigned_samples,
            path="ranked train local_assigned_samples",
            minimum=1,
        )
        global_samples = _require_integer(
            self.global_assigned_samples,
            path="ranked train global_assigned_samples",
            minimum=1,
        )
        _require_integer(
            self.global_dropped_samples,
            path="ranked train global_dropped_samples",
            minimum=0,
        )
        _require_digest(
            self.assignment_digest,
            path="ranked train assignment_digest",
        )
        expected_local = windows * microbatches * samples
        if local_samples != expected_local:
            raise ValueError(
                "ranked train local_assigned_samples does not match the window plan"
            )
        if global_samples != expected_local * rank_context.world_size:
            raise ValueError(
                "ranked train global_assigned_samples does not match world_size"
            )
        maximum = cast(object, self.requested_max_microbatches)
        if maximum is not None:
            maximum_value = _require_integer(
                maximum,
                path="ranked train requested_max_microbatches",
                minimum=1,
            )
            if maximum_value % microbatches != 0:
                raise ValueError(
                    "ranked train requested_max_microbatches must contain complete "
                    "accumulation windows"
                )
            if windows * microbatches > maximum_value:
                raise ValueError(
                    "ranked train microbatch_count exceeds "
                    "requested_max_microbatches"
                )

    @property
    def microbatch_count(self) -> int:
        """Return the exact number of local microbatches in this plan."""

        return self.window_count * self.microbatches_per_window


@dataclass(frozen=True, slots=True)
class RankedTrainWindow:
    """One complete optimizer window containing still-opaque task batches."""

    ordinal: int
    batches: tuple[Any, ...]
    batch_facts: tuple[RankedBatchFacts, ...]

    def __post_init__(self) -> None:
        _require_integer(self.ordinal, path="ranked train window ordinal", minimum=0)
        batches = cast(object, self.batches)
        if not isinstance(batches, tuple) or not batches:
            raise ValueError("ranked train window batches must be a non-empty tuple")
        batch_facts = cast(object, self.batch_facts)
        if not isinstance(batch_facts, tuple) or len(batch_facts) != len(batches):
            raise ValueError(
                "ranked train window facts must match the opaque batch count"
            )
        if any(
            not isinstance(fact, RankedBatchFacts)
            for fact in cast(tuple[object, ...], self.batch_facts)
        ):
            raise TypeError("ranked train window facts have the wrong type")


@dataclass(frozen=True, slots=True)
class RankedEpochCompletion:
    """Terminal proof that one rank consumed its complete train plan."""

    plan_digest: str
    rank: int
    observed_windows: int
    observed_microbatches: int
    observed_samples: int
    assignment_digest: str
    terminal_token: str

    def __post_init__(self) -> None:
        _require_digest(self.plan_digest, path="ranked completion plan_digest")
        _require_integer(self.rank, path="ranked completion rank", minimum=0)
        _require_integer(
            self.observed_windows,
            path="ranked completion observed_windows",
            minimum=1,
        )
        _require_integer(
            self.observed_microbatches,
            path="ranked completion observed_microbatches",
            minimum=1,
        )
        _require_integer(
            self.observed_samples,
            path="ranked completion observed_samples",
            minimum=1,
        )
        _require_digest(
            self.assignment_digest,
            path="ranked completion assignment_digest",
        )
        _require_digest(self.terminal_token, path="ranked completion terminal_token")


@runtime_checkable
class RankedTrainEpochReader(Protocol):
    """Read one already planned rank-local training epoch."""

    @property
    def plan(self) -> RankedTrainEpochPlan:
        """Return the immutable plan this reader is consuming."""

        ...

    def read_window(self) -> RankedTrainWindow | None:
        """Return the next complete optimizer window, or ``None`` at its end."""

        ...

    def finish(self) -> RankedEpochCompletion:
        """Verify exhaustion and return terminal consumption facts."""

        ...

    def close(self) -> None:
        """Release reader-owned iterator and worker resources."""

        ...


@runtime_checkable
class RankedTrainExecution(Protocol):
    """Builder-owned training assignment and reader factory."""

    @property
    def batches(self) -> Iterable[Any]:
        """Return the exact re-iterable exposed through ``DataLoaders.train``."""

        ...

    @property
    def resume_identity(self) -> RankedEpochDataIdentity:
        """Return the semantic identity required for epoch-boundary resume."""

        ...

    def plan_epoch(
        self,
        epoch: int,
        *,
        microbatches_per_window: int,
        max_microbatches: int | None,
    ) -> RankedTrainEpochPlan:
        """Plan equal complete windows without opening data workers."""

        ...

    def open_epoch(self, plan: RankedTrainEpochPlan) -> RankedTrainEpochReader:
        """Open a reader for an exact plan issued by this execution."""

        ...


@dataclass(frozen=True, slots=True)
class ExactCoverageSpan:
    """Half-open interval in one Builder-authenticated validation universe."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        start = _require_integer(self.start, path="coverage span start", minimum=0)
        stop = _require_integer(self.stop, path="coverage span stop", minimum=1)
        if stop <= start:
            raise ValueError("coverage span stop must be greater than start")


@dataclass(frozen=True, slots=True)
class ExactValidationEpochPlan:
    """Exact rank-local share of one complete validation coverage universe."""

    coverage_identity: RankedEpochDataIdentity
    plan_digest: str
    epoch: int
    rank: int
    world_size: int
    global_expected_samples: int
    primary_batch_count: int
    local_expected_samples: int
    local_spans: tuple[ExactCoverageSpan, ...]

    def __post_init__(self) -> None:
        coverage_identity = cast(object, self.coverage_identity)
        if not isinstance(coverage_identity, RankedEpochDataIdentity):
            raise TypeError("validation coverage_identity has the wrong type")
        _require_digest(self.plan_digest, path="validation plan_digest")
        _require_integer(self.epoch, path="validation epoch", minimum=0)
        DataRankContext(self.rank, self.world_size)
        global_samples = _require_integer(
            self.global_expected_samples,
            path="validation global_expected_samples",
            minimum=1,
        )
        _require_integer(
            self.primary_batch_count,
            path="validation primary_batch_count",
            minimum=1,
        )
        local_samples = _require_integer(
            self.local_expected_samples,
            path="validation local_expected_samples",
            minimum=0,
        )
        local_spans = cast(object, self.local_spans)
        if not isinstance(local_spans, tuple) or any(
            not isinstance(span, ExactCoverageSpan)
            for span in cast(tuple[object, ...], local_spans)
        ):
            raise TypeError("validation local_spans must contain coverage spans")
        span_samples = sum(span.stop - span.start for span in self.local_spans)
        if local_samples != span_samples:
            raise ValueError(
                "validation local_expected_samples does not match local_spans"
            )
        if local_samples > global_samples:
            raise ValueError(
                "validation local_expected_samples exceeds global coverage"
            )


@dataclass(frozen=True, slots=True)
class ExactValidationBatch:
    """One opaque validation batch plus exact Builder-owned coverage facts."""

    batch: Any
    facts: RankedBatchFacts
    coverage_span: ExactCoverageSpan

    def __post_init__(self) -> None:
        facts = cast(object, self.facts)
        if not isinstance(facts, RankedBatchFacts):
            raise TypeError("exact validation batch facts have the wrong type")
        coverage_span = cast(object, self.coverage_span)
        if not isinstance(coverage_span, ExactCoverageSpan):
            raise TypeError("exact validation batch span has the wrong type")
        if self.facts.sample_count != self.coverage_span.stop - self.coverage_span.start:
            raise ValueError(
                "exact validation sample_count does not match its coverage span"
            )


@dataclass(frozen=True, slots=True)
class ExactCoverageReceipt:
    """Terminal proof for one rank's declared validation coverage."""

    plan_digest: str
    rank: int
    completed_spans: tuple[ExactCoverageSpan, ...]
    observed_samples: int

    def __post_init__(self) -> None:
        _require_digest(self.plan_digest, path="coverage receipt plan_digest")
        _require_integer(self.rank, path="coverage receipt rank", minimum=0)
        completed_spans = cast(object, self.completed_spans)
        if not isinstance(completed_spans, tuple) or any(
            not isinstance(span, ExactCoverageSpan)
            for span in cast(tuple[object, ...], completed_spans)
        ):
            raise TypeError("coverage receipt spans have the wrong type")
        observed = _require_integer(
            self.observed_samples,
            path="coverage receipt observed_samples",
            minimum=0,
        )
        if observed != sum(
            span.stop - span.start for span in self.completed_spans
        ):
            raise ValueError(
                "coverage receipt observed_samples does not match completed_spans"
            )


@runtime_checkable
class ExactValidationEpochReader(Protocol):
    """Read one rank's exact validation coverage assignment."""

    @property
    def plan(self) -> ExactValidationEpochPlan:
        """Return the immutable plan this reader is consuming."""

        ...

    def read_batch(self) -> ExactValidationBatch | None:
        """Return the next validation batch, or ``None`` at its end."""

        ...

    def finish(self) -> ExactCoverageReceipt:
        """Verify exhaustion and return exact coverage facts."""

        ...

    def close(self) -> None:
        """Release reader-owned iterator and worker resources."""

        ...


@runtime_checkable
class ExactValidationExecution(Protocol):
    """Builder-owned exact validation coverage and reader factory."""

    @property
    def batches(self) -> Iterable[Any]:
        """Return the exact iterable exposed through ``DataLoaders.validation``."""

        ...

    @property
    def coverage_identity(self) -> RankedEpochDataIdentity:
        """Return the identity of the complete validation universe."""

        ...

    def plan_epoch(self, epoch: int) -> ExactValidationEpochPlan:
        """Plan validation coverage without reading any sample."""

        ...

    def open_epoch(
        self,
        plan: ExactValidationEpochPlan,
    ) -> ExactValidationEpochReader:
        """Open a reader for an exact plan issued by this execution."""

        ...


@dataclass(frozen=True, slots=True)
class RankedDataExecution:
    """Ranked train and validation capabilities bound to one loader bundle."""

    rank_context: DataRankContext
    train: RankedTrainExecution
    validation: ExactValidationExecution | None

    def __post_init__(self) -> None:
        rank_context = cast(object, self.rank_context)
        if not isinstance(rank_context, DataRankContext):
            raise TypeError("ranked execution rank_context has the wrong type")
        train = cast(object, self.train)
        if not isinstance(train, RankedTrainExecution):
            raise TypeError("ranked execution train capability is missing")
        validation = cast(object, self.validation)
        if validation is not None and not isinstance(
            validation,
            ExactValidationExecution,
        ):
            raise TypeError("ranked execution validation capability is invalid")


__all__ = [
    "DataRankContext",
    "ExactCoverageReceipt",
    "ExactCoverageSpan",
    "ExactValidationBatch",
    "ExactValidationEpochPlan",
    "ExactValidationEpochReader",
    "ExactValidationExecution",
    "RankedBatchFacts",
    "RankedDataExecution",
    "RankedEpochCompletion",
    "RankedEpochDataIdentity",
    "RankedTrainEpochPlan",
    "RankedTrainEpochReader",
    "RankedTrainExecution",
    "RankedTrainWindow",
]

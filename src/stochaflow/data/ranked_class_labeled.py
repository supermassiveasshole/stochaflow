"""Rank-aware execution for the built-in class-labeled image recipe."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence, Sized
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.utils.data import DataLoader, Sampler

from stochaflow.data.artifact_store import canonical_artifact_digest
from stochaflow.data.artifacts import DataArtifactBindings
from stochaflow.data.dataloaders import (
    collate_class_labeled_image_batch,
    data_loader_kwargs,
)
from stochaflow.data.datasets import ClassLabeledImageDataset
from stochaflow.data.image_contracts import ClassLabeledImageFileRecord
from stochaflow.data.ranked import (
    DataRankContext,
    ExactCoverageReceipt,
    ExactCoverageSpan,
    ExactValidationBatch,
    ExactValidationEpochPlan,
    RankedBatchFacts,
    RankedDataExecution,
    RankedEpochCompletion,
    RankedEpochDataIdentity,
    RankedTrainEpochPlan,
    RankedTrainWindow,
)
from stochaflow.data.recipe_config import ClassLabeledImageDataBuilderConfig

_TRAIN_PROVIDER = "stochaflow.class_labeled_image.ranked_train.v1"
_TRAIN_TERMINAL_PROVIDER = (
    "stochaflow.class_labeled_image.ranked_train.terminal.v1"
)
_VALIDATION_PROVIDER = "stochaflow.class_labeled_image.rank0_validation.v1"
_SHUFFLE_DOMAIN = b"stochaflow.class-labeled-ranked.shuffle.v1"


def _integer(value: object, *, path: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{path} must be a {qualifier} integer")
    return cast(int, value)


def _record_profile(
    records: Sequence[ClassLabeledImageFileRecord],
) -> list[dict[str, object]]:
    return [
        {
            "tree": record.image.tree,
            "path": record.image.path,
            "sha256": record.image.sha256,
            "class_label": record.class_label,
        }
        for record in records
    ]


def _recipe_profile(config: ClassLabeledImageDataBuilderConfig) -> dict[str, object]:
    return {
        "image": {
            "size": list(config.image.size),
            "channels": config.image.channels,
            "normalize": config.image.normalize,
            "random_horizontal_flip": config.image.random_horizontal_flip,
        },
        "partition": {
            "validation_per_class": config.partition.validation_per_class,
            "seed": config.partition.seed,
        },
        "loader": {
            "batch_size": config.loader.batch_size,
            "shuffle": config.loader.shuffle,
            "drop_last": config.loader.drop_last,
            "steps_per_epoch": config.loader.steps_per_epoch,
        },
        "sample_randomness": "stochaflow.class-labeled-image.sample.v1",
    }


def _data_identity(
    *,
    provider: str,
    records: Sequence[ClassLabeledImageFileRecord],
    config: ClassLabeledImageDataBuilderConfig,
    artifact_bindings: DataArtifactBindings,
    seed: int,
    world_size: int | None,
) -> RankedEpochDataIdentity:
    body: dict[str, object] = {
        "provider": provider,
        "artifacts": artifact_bindings.to_dict(),
        "records": _record_profile(records),
        "recipe": _recipe_profile(config),
        "run_seed": seed,
    }
    if world_size is not None:
        body["world_size"] = world_size
    return RankedEpochDataIdentity(
        provider=provider,
        digest=canonical_artifact_digest(body),
    )


def _epoch_seed(seed: int, epoch: int) -> int:
    digest = hashlib.sha256()
    for value in (_SHUFFLE_DOMAIN, str(seed).encode(), str(epoch).encode()):
        digest.update(len(value).to_bytes(8, byteorder="big"))
        digest.update(value)
    return int.from_bytes(digest.digest()[:8], byteorder="little")


def _expected_train_terminal_token(
    *,
    plan_digest: str,
    rank: int,
    assignment_digest: str,
) -> str:
    return canonical_artifact_digest(
        {
            "provider": _TRAIN_TERMINAL_PROVIDER,
            "plan_digest": plan_digest,
            "rank": rank,
            "assignment_digest": assignment_digest,
        }
    )


def _assignment_digest(
    *,
    plan_digest: str,
    rank: int,
    assignment_tokens: Sequence[str],
) -> str:
    return canonical_artifact_digest(
        {
            "provider": "stochaflow.class_labeled_image.assignment.v1",
            "plan_digest": plan_digest,
            "rank": rank,
            "tokens": list(assignment_tokens),
        }
    )


def _ranked_loader_kwargs(
    config: ClassLabeledImageDataBuilderConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    kwargs = data_loader_kwargs(config.loader, seed=seed)
    if config.loader.num_workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
    return kwargs


@dataclass(frozen=True, slots=True)
class RankedClassLabeledAssignment:
    """Internal exact index assignment for one opaque microbatch."""

    indices: tuple[int, ...]
    facts: RankedBatchFacts


class RankedClassLabeledBatchSampler(Sampler[list[tuple[int, int]]]):
    """Emit one epoch-tagged batch for each planned local assignment."""

    def __init__(
        self,
        assignments: tuple[RankedClassLabeledAssignment, ...],
        *,
        epoch: int,
    ) -> None:
        self.assignments = assignments
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        for assignment in self.assignments:
            yield [(self.epoch, index) for index in assignment.indices]

    def __len__(self) -> int:
        return len(self.assignments)


class ClassLabeledRankedTrainEpochReader:
    """Consume exactly one planned class-labeled rank-local epoch."""

    def __init__(
        self,
        *,
        plan: RankedTrainEpochPlan,
        assignments: tuple[RankedClassLabeledAssignment, ...],
        loader: DataLoader[Any],
    ) -> None:
        self._plan = plan
        self.assignments = assignments
        self.loader: DataLoader[Any] | None = loader
        self.iterator: Iterator[Any] | None = iter(loader)
        self.next_window = 0
        self.finished = False
        self.closed = False

    @property
    def plan(self) -> RankedTrainEpochPlan:
        """Return the exact plan accepted by this reader."""

        return self._plan

    def _require_open(self) -> Iterator[Any]:
        if self.closed:
            raise RuntimeError("ranked train reader is closed")
        if self.finished:
            raise RuntimeError("ranked train reader is already finished")
        assert self.iterator is not None
        return self.iterator

    def read_window(self) -> RankedTrainWindow | None:
        """Read one complete accumulation window without inspecting batches."""

        iterator = self._require_open()
        if self.next_window >= self.plan.window_count:
            return None
        start = self.next_window * self.plan.microbatches_per_window
        stop = start + self.plan.microbatches_per_window
        batches: list[Any] = []
        for ordinal in range(start, stop):
            try:
                batches.append(next(iterator))
            except StopIteration as error:
                raise RuntimeError(
                    "ranked train loader ended before its authenticated plan"
                ) from error
            if self.assignments[ordinal].facts.ordinal != ordinal:
                raise RuntimeError("ranked train assignment ordinal is inconsistent")
        window = RankedTrainWindow(
            ordinal=self.next_window,
            batches=tuple(batches),
            batch_facts=tuple(
                assignment.facts for assignment in self.assignments[start:stop]
            ),
        )
        self.next_window += 1
        return window

    def finish(self) -> RankedEpochCompletion:
        """Require exact exhaustion and issue terminal rank-local facts."""

        iterator = self._require_open()
        if self.next_window != self.plan.window_count:
            raise RuntimeError(
                "ranked train reader cannot finish before every planned window"
            )
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise RuntimeError("ranked train loader yielded beyond its plan")
        terminal_token = _expected_train_terminal_token(
            plan_digest=self.plan.plan_digest,
            rank=self.plan.rank,
            assignment_digest=self.plan.assignment_digest,
        )
        if terminal_token != self.plan.expected_terminal_token:
            raise RuntimeError(
                "ranked train terminal token does not match its issued plan"
            )
        self.finished = True
        return RankedEpochCompletion(
            plan_digest=self.plan.plan_digest,
            rank=self.plan.rank,
            observed_windows=self.plan.window_count,
            observed_microbatches=self.plan.microbatch_count,
            observed_samples=self.plan.local_assigned_samples,
            assignment_digest=self.plan.assignment_digest,
            terminal_token=terminal_token,
        )

    def close(self) -> None:
        """Release the loader iterator even after an incomplete read."""

        self.iterator = None
        self.loader = None
        self.closed = True


class ClassLabeledRankedTrainExecution:
    """Build deterministic equal-window assignments for one global rank."""

    def __init__(
        self,
        *,
        dataset: ClassLabeledImageDataset,
        config: ClassLabeledImageDataBuilderConfig,
        rank_context: DataRankContext,
        seed: int,
        resume_identity: RankedEpochDataIdentity,
    ) -> None:
        if not config.loader.drop_last:
            raise ValueError(
                "ranked class-labeled training requires loader.drop_last=true"
            )
        self.dataset = dataset
        self.config = config
        self.rank_context = rank_context
        self.seed = seed
        self._resume_identity = resume_identity
        self.default_epoch = 0
        natural_local_microbatches = len(cast(Sized, dataset)) // (
            rank_context.world_size * config.loader.batch_size
        )
        if natural_local_microbatches <= 0:
            raise ValueError(
                "ranked class-labeled training has no complete per-rank batch"
            )
        configured = config.loader.steps_per_epoch
        if isinstance(configured, int) and configured > natural_local_microbatches:
            raise ValueError(
                "loader.steps_per_epoch exceeds ranked natural microbatches"
            )
        self.natural_local_microbatches = natural_local_microbatches

    @property
    def batches(self) -> ClassLabeledRankedTrainExecution:
        """Return the exact ordinary iterable bound into DataLoaders."""

        return self

    @property
    def resume_identity(self) -> RankedEpochDataIdentity:
        """Return the semantic epoch reconstruction identity."""

        return self._resume_identity

    def __len__(self) -> int:
        configured = self.config.loader.steps_per_epoch
        return min(
            self.natural_local_microbatches,
            configured if isinstance(configured, int) else self.natural_local_microbatches,
        )

    def set_epoch(self, epoch: int) -> None:
        """Select the epoch used by the compatibility iterable view."""

        self.default_epoch = _integer(epoch, path="ranked train epoch", minimum=0)

    def __iter__(self) -> Iterator[Any]:
        plan = self.plan_epoch(
            self.default_epoch,
            microbatches_per_window=1,
            max_microbatches=None,
        )
        reader = self.open_epoch(plan)
        try:
            while True:
                window = reader.read_window()
                if window is None:
                    break
                yield window.batches[0]
            reader.finish()
        finally:
            reader.close()

    def plan_epoch(
        self,
        epoch: int,
        *,
        microbatches_per_window: int,
        max_microbatches: int | None,
    ) -> RankedTrainEpochPlan:
        """Plan equal complete windows without opening the DataLoader."""

        epoch_value = _integer(epoch, path="ranked train epoch", minimum=0)
        accumulation = _integer(
            microbatches_per_window,
            path="ranked train microbatches_per_window",
            minimum=1,
        )
        requested_max: int | None = None
        if max_microbatches is not None:
            requested_max = _integer(
                max_microbatches,
                path="ranked train max_microbatches",
                minimum=1,
            )
            if requested_max % accumulation != 0:
                raise ValueError(
                    "ranked train max_microbatches must contain complete "
                    "accumulation windows"
                )
        configured = self.config.loader.steps_per_epoch
        configured_max = configured if isinstance(configured, int) else None
        if configured_max is not None and configured_max % accumulation != 0:
            raise ValueError(
                "loader.steps_per_epoch must contain complete accumulation windows"
            )
        available = self.natural_local_microbatches
        if configured_max is not None:
            available = min(available, configured_max)
        if requested_max is not None:
            available = min(available, requested_max)
        window_count = available // accumulation
        if window_count <= 0:
            raise ValueError(
                "ranked class-labeled training has no complete accumulation window"
            )
        local_samples = (
            window_count * accumulation * self.config.loader.batch_size
        )
        global_samples = local_samples * self.rank_context.world_size
        body = {
            "data_identity": self.resume_identity.to_dict(),
            "epoch": epoch_value,
            "world_size": self.rank_context.world_size,
            "microbatches_per_window": accumulation,
            "window_count": window_count,
            "samples_per_microbatch": self.config.loader.batch_size,
            "global_assigned_samples": global_samples,
            "global_dropped_samples": len(cast(Sized, self.dataset)) - global_samples,
            "requested_max_microbatches": requested_max,
        }
        plan_digest = canonical_artifact_digest(body)
        assignment_tokens = self._assignment_tokens(
            plan_digest=plan_digest,
            epoch=epoch_value,
            microbatch_count=window_count * accumulation,
        )
        assignment_digest = _assignment_digest(
            plan_digest=plan_digest,
            rank=self.rank_context.rank,
            assignment_tokens=assignment_tokens,
        )
        return RankedTrainEpochPlan(
            data_identity=self.resume_identity,
            plan_digest=plan_digest,
            expected_terminal_token=_expected_train_terminal_token(
                plan_digest=plan_digest,
                rank=self.rank_context.rank,
                assignment_digest=assignment_digest,
            ),
            epoch=epoch_value,
            rank=self.rank_context.rank,
            world_size=self.rank_context.world_size,
            microbatches_per_window=accumulation,
            window_count=window_count,
            samples_per_microbatch=self.config.loader.batch_size,
            local_assigned_samples=local_samples,
            global_assigned_samples=global_samples,
            global_dropped_samples=len(cast(Sized, self.dataset)) - global_samples,
            assignment_digest=assignment_digest,
            requested_max_microbatches=requested_max,
        )

    def _assignment_tokens(
        self,
        *,
        plan_digest: str,
        epoch: int,
        microbatch_count: int,
    ) -> tuple[str, ...]:
        dataset_size = len(cast(Sized, self.dataset))
        if self.config.loader.shuffle:
            generator = torch.Generator().manual_seed(_epoch_seed(self.seed, epoch))
            permutation = cast(
                list[int],
                torch.randperm(dataset_size, generator=generator).tolist(),
            )
        else:
            permutation = list(range(dataset_size))
        per_global_microbatch = (
            self.rank_context.world_size * self.config.loader.batch_size
        )
        tokens: list[str] = []
        for ordinal in range(microbatch_count):
            start = (
                ordinal * per_global_microbatch
                + self.rank_context.rank * self.config.loader.batch_size
            )
            indices = permutation[start : start + self.config.loader.batch_size]
            if len(indices) != self.config.loader.batch_size:
                raise RuntimeError("ranked training assignment is unexpectedly short")
            tokens.append(
                canonical_artifact_digest(
                    {
                        "plan_digest": plan_digest,
                        "rank": self.rank_context.rank,
                        "ordinal": ordinal,
                        "indices": indices,
                    }
                )
            )
        return tuple(tokens)

    def _assignments(
        self,
        plan: RankedTrainEpochPlan,
    ) -> tuple[RankedClassLabeledAssignment, ...]:
        dataset_size = len(cast(Sized, self.dataset))
        if self.config.loader.shuffle:
            generator = torch.Generator().manual_seed(
                _epoch_seed(self.seed, plan.epoch)
            )
            permutation = cast(
                list[int],
                torch.randperm(dataset_size, generator=generator).tolist(),
            )
        else:
            permutation = list(range(dataset_size))
        per_global_microbatch = plan.world_size * plan.samples_per_microbatch
        assignments: list[RankedClassLabeledAssignment] = []
        for ordinal in range(plan.microbatch_count):
            start = (
                ordinal * per_global_microbatch
                + plan.rank * plan.samples_per_microbatch
            )
            indices = tuple(
                permutation[start : start + plan.samples_per_microbatch]
            )
            if len(indices) != plan.samples_per_microbatch:
                raise RuntimeError("ranked training assignment is unexpectedly short")
            assignment_token = canonical_artifact_digest(
                {
                    "plan_digest": plan.plan_digest,
                    "rank": plan.rank,
                    "ordinal": ordinal,
                    "indices": list(indices),
                }
            )
            assignments.append(
                RankedClassLabeledAssignment(
                    indices=indices,
                    facts=RankedBatchFacts(
                        ordinal=ordinal,
                        sample_count=len(indices),
                        loss_weight=float(len(indices)),
                        assignment_token=assignment_token,
                    ),
                )
            )
        result = tuple(assignments)
        observed_assignment_digest = _assignment_digest(
            plan_digest=plan.plan_digest,
            rank=plan.rank,
            assignment_tokens=tuple(
                assignment.facts.assignment_token for assignment in result
            ),
        )
        if observed_assignment_digest != plan.assignment_digest:
            raise RuntimeError(
                "ranked training assignment digest does not match its issued plan"
            )
        return result

    def open_epoch(
        self,
        plan: RankedTrainEpochPlan,
    ) -> ClassLabeledRankedTrainEpochReader:
        """Open a DataLoader for a plan issued by this exact execution."""

        plan_value = cast(object, plan)
        if not isinstance(plan_value, RankedTrainEpochPlan):
            raise TypeError("ranked train plan has the wrong type")
        expected = self.plan_epoch(
            plan.epoch,
            microbatches_per_window=plan.microbatches_per_window,
            max_microbatches=plan.requested_max_microbatches,
        )
        if plan != expected:
            raise ValueError("ranked train plan does not belong to this execution")
        assignments = self._assignments(plan)
        sampler = RankedClassLabeledBatchSampler(assignments, epoch=plan.epoch)
        loader = DataLoader(
            self.dataset,
            batch_sampler=sampler,
            collate_fn=collate_class_labeled_image_batch,
            **_ranked_loader_kwargs(self.config, seed=self.seed + plan.epoch),
        )
        return ClassLabeledRankedTrainEpochReader(
            plan=plan,
            assignments=assignments,
            loader=loader,
        )


@dataclass(frozen=True, slots=True)
class ExactValidationAssignment:
    """Internal sequential batch and exact coverage span."""

    indices: tuple[int, ...]
    facts: RankedBatchFacts
    span: ExactCoverageSpan


class ExactValidationBatchSampler(Sampler[list[int]]):
    """Emit sequential indices for rank-zero full validation."""

    def __init__(self, assignments: tuple[ExactValidationAssignment, ...]) -> None:
        self.assignments = assignments

    def __iter__(self) -> Iterator[list[int]]:
        for assignment in self.assignments:
            yield list(assignment.indices)

    def __len__(self) -> int:
        return len(self.assignments)


class ClassLabeledExactValidationEpochReader:
    """Consume one rank's exact validation coverage assignment."""

    def __init__(
        self,
        *,
        plan: ExactValidationEpochPlan,
        assignments: tuple[ExactValidationAssignment, ...],
        loader: DataLoader[Any] | None,
    ) -> None:
        self._plan = plan
        self.assignments = assignments
        self.loader: DataLoader[Any] | None = loader
        self.iterator: Iterator[Any] | None = iter(loader) if loader is not None else None
        self.next_batch = 0
        self.finished = False
        self.closed = False

    @property
    def plan(self) -> ExactValidationEpochPlan:
        """Return the exact validation plan accepted by this reader."""

        return self._plan

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("exact validation reader is closed")
        if self.finished:
            raise RuntimeError("exact validation reader is already finished")

    def read_batch(self) -> ExactValidationBatch | None:
        """Return the next opaque validation batch and authenticated facts."""

        self._require_open()
        if self.next_batch >= len(self.assignments):
            return None
        if self.iterator is None:
            raise RuntimeError("validation assignment has no DataLoader iterator")
        assignment = self.assignments[self.next_batch]
        try:
            batch = next(self.iterator)
        except StopIteration as error:
            raise RuntimeError(
                "validation loader ended before its authenticated coverage"
            ) from error
        self.next_batch += 1
        return ExactValidationBatch(
            batch=batch,
            facts=assignment.facts,
            coverage_span=assignment.span,
        )

    def finish(self) -> ExactCoverageReceipt:
        """Require exact exhaustion and issue the rank's coverage receipt."""

        self._require_open()
        if self.next_batch != len(self.assignments):
            raise RuntimeError(
                "validation reader cannot finish before all assigned coverage"
            )
        if self.iterator is not None:
            try:
                next(self.iterator)
            except StopIteration:
                pass
            else:
                raise RuntimeError("validation loader yielded beyond its coverage")
        self.finished = True
        return ExactCoverageReceipt(
            plan_digest=self.plan.plan_digest,
            rank=self.plan.rank,
            completed_spans=self.plan.local_spans,
            observed_samples=self.plan.local_expected_samples,
        )

    def close(self) -> None:
        """Release the loader iterator even after an incomplete read."""

        self.iterator = None
        self.loader = None
        self.closed = True


class ClassLabeledExactValidationExecution:
    """Expose full validation on rank zero and exact emptiness elsewhere."""

    def __init__(
        self,
        *,
        dataset: ClassLabeledImageDataset,
        config: ClassLabeledImageDataBuilderConfig,
        rank_context: DataRankContext,
        seed: int,
        coverage_identity: RankedEpochDataIdentity,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.rank_context = rank_context
        self.seed = seed
        self._coverage_identity = coverage_identity

    @property
    def batches(self) -> ClassLabeledExactValidationExecution:
        """Return the exact ordinary iterable bound into DataLoaders."""

        return self

    @property
    def coverage_identity(self) -> RankedEpochDataIdentity:
        """Return the identity of the complete validation universe."""

        return self._coverage_identity

    def __len__(self) -> int:
        if self.rank_context.rank != 0:
            return 0
        return math.ceil(
            len(cast(Sized, self.dataset)) / self.config.loader.batch_size
        )

    def __iter__(self) -> Iterator[Any]:
        reader = self.open_epoch(self.plan_epoch(0))
        try:
            while True:
                selected = reader.read_batch()
                if selected is None:
                    break
                yield selected.batch
            reader.finish()
        finally:
            reader.close()

    def plan_epoch(self, epoch: int) -> ExactValidationEpochPlan:
        """Plan rank-zero full coverage without opening the DataLoader."""

        epoch_value = _integer(epoch, path="validation epoch", minimum=0)
        dataset_size = len(cast(Sized, self.dataset))
        primary_batch_count = math.ceil(
            dataset_size / self.config.loader.batch_size
        )
        local_spans = (
            (ExactCoverageSpan(0, dataset_size),)
            if self.rank_context.rank == 0
            else ()
        )
        plan_digest = canonical_artifact_digest(
            {
                "coverage_identity": self.coverage_identity.to_dict(),
                "epoch": epoch_value,
                "world_size": self.rank_context.world_size,
                "global_expected_samples": dataset_size,
                "primary_batch_count": primary_batch_count,
                "assignment": "rank0-full",
            }
        )
        return ExactValidationEpochPlan(
            coverage_identity=self.coverage_identity,
            plan_digest=plan_digest,
            epoch=epoch_value,
            rank=self.rank_context.rank,
            world_size=self.rank_context.world_size,
            global_expected_samples=dataset_size,
            primary_batch_count=primary_batch_count,
            local_expected_samples=(
                dataset_size if self.rank_context.rank == 0 else 0
            ),
            local_spans=local_spans,
        )

    def _assignments(
        self,
        plan: ExactValidationEpochPlan,
    ) -> tuple[ExactValidationAssignment, ...]:
        if plan.rank != 0:
            return ()
        assignments: list[ExactValidationAssignment] = []
        dataset_size = len(cast(Sized, self.dataset))
        batch_size = self.config.loader.batch_size
        for ordinal, start in enumerate(range(0, dataset_size, batch_size)):
            stop = min(start + batch_size, dataset_size)
            span = ExactCoverageSpan(start, stop)
            indices = tuple(range(start, stop))
            assignments.append(
                ExactValidationAssignment(
                    indices=indices,
                    facts=RankedBatchFacts(
                        ordinal=ordinal,
                        sample_count=len(indices),
                        loss_weight=float(len(indices)),
                        assignment_token=canonical_artifact_digest(
                            {
                                "plan_digest": plan.plan_digest,
                                "ordinal": ordinal,
                                "span": [start, stop],
                            }
                        ),
                    ),
                    span=span,
                )
            )
        return tuple(assignments)

    def open_epoch(
        self,
        plan: ExactValidationEpochPlan,
    ) -> ClassLabeledExactValidationEpochReader:
        """Open the rank's exact validation view for one issued plan."""

        plan_value = cast(object, plan)
        if not isinstance(plan_value, ExactValidationEpochPlan):
            raise TypeError("exact validation plan has the wrong type")
        expected = self.plan_epoch(plan.epoch)
        if plan != expected:
            raise ValueError("validation plan does not belong to this execution")
        assignments = self._assignments(plan)
        loader: DataLoader[Any] | None = None
        if assignments:
            loader = DataLoader(
                self.dataset,
                batch_sampler=ExactValidationBatchSampler(assignments),
                collate_fn=collate_class_labeled_image_batch,
                **_ranked_loader_kwargs(
                    self.config,
                    seed=self.seed + plan.epoch,
                ),
            )
        return ClassLabeledExactValidationEpochReader(
            plan=plan,
            assignments=assignments,
            loader=loader,
        )


def build_class_labeled_ranked_execution(
    *,
    train: ClassLabeledImageDataset,
    validation: ClassLabeledImageDataset,
    train_records: Sequence[ClassLabeledImageFileRecord],
    validation_records: Sequence[ClassLabeledImageFileRecord],
    config: ClassLabeledImageDataBuilderConfig,
    rank_context: DataRankContext,
    seed: int,
    artifact_bindings: DataArtifactBindings,
) -> RankedDataExecution:
    """Build the first fixed-topology ranked class-labeled data path."""

    train_identity = _data_identity(
        provider=_TRAIN_PROVIDER,
        records=train_records,
        config=config,
        artifact_bindings=artifact_bindings,
        seed=seed,
        world_size=rank_context.world_size,
    )
    validation_identity = _data_identity(
        provider=_VALIDATION_PROVIDER,
        records=validation_records,
        config=config,
        artifact_bindings=artifact_bindings,
        seed=seed,
        world_size=None,
    )
    train_execution = ClassLabeledRankedTrainExecution(
        dataset=train,
        config=config,
        rank_context=rank_context,
        seed=seed,
        resume_identity=train_identity,
    )
    validation_execution = ClassLabeledExactValidationExecution(
        dataset=validation,
        config=config,
        rank_context=rank_context,
        seed=seed + 1,
        coverage_identity=validation_identity,
    )
    return RankedDataExecution(
        rank_context=rank_context,
        train=train_execution,
        validation=validation_execution,
    )


__all__ = ["build_class_labeled_ranked_execution"]

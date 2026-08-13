"""Focused contracts for fixed-topology rank-aware data execution."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from stochaflow.data import (
    DataLoaders,
    DataRankContext,
    RankedDataExecution,
    RankedEpochDataIdentity,
    RankedTrainEpochPlan,
)
from stochaflow.data.datasets import ClassLabeledImageDataset
from stochaflow.data.ranked_class_labeled import (
    ClassLabeledExactValidationExecution,
    ClassLabeledRankedTrainExecution,
)
from stochaflow.data.recipe_config import (
    ClassLabeledImageDataBuilderConfig,
    ClassStratifiedPartitionRecipeConfig,
    DataSourceMaterializationConfig,
    ImageRecipeConfig,
    ImageSourceConfig,
    LoaderRecipeConfig,
)


class SyntheticClassLabeledDataset(ClassLabeledImageDataset):
    """Small identity-preserving dataset that avoids image filesystem I/O."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(
        self,
        index: int | tuple[int, int],
    ) -> tuple[torch.Tensor, int]:
        sample_index = index[1] if isinstance(index, tuple) else index
        return torch.tensor([float(sample_index)]), sample_index


def ranked_config(
    *,
    batch_size: int = 2,
    drop_last: bool = True,
    steps_per_epoch: int | str = "auto",
) -> ClassLabeledImageDataBuilderConfig:
    """Build one validated-enough in-memory ranked recipe fixture."""

    return ClassLabeledImageDataBuilderConfig(
        source=ImageSourceConfig(
            name="tests.synthetic",
            materialization=DataSourceMaterializationConfig(),
        ),
        image=ImageRecipeConfig(
            size=[1, 1],
            channels=1,
            normalize=False,
            random_horizontal_flip=False,
        ),
        partition=ClassStratifiedPartitionRecipeConfig(
            validation_per_class=1,
        ),
        loader=LoaderRecipeConfig(
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
            drop_last=drop_last,
            pin_memory=False,
            persistent_workers=False,
            steps_per_epoch=steps_per_epoch,
        ),
    )


def ranked_identity() -> RankedEpochDataIdentity:
    """Return one stable identity shared by the synthetic ranks."""

    return RankedEpochDataIdentity(
        provider="tests.synthetic.ranked.v1",
        digest="a" * 64,
    )


@pytest.mark.parametrize(
    ("rank", "world_size"),
    [
        (-1, 2),
        (2, 2),
        (0, 0),
        (True, 2),
        (0, True),
    ],
)
def test_data_rank_context_rejects_invalid_topology(
    rank: object,
    world_size: object,
) -> None:
    with pytest.raises(ValueError, match=r"data rank|world_size"):
        DataRankContext(rank=rank, world_size=world_size)  # type: ignore[arg-type]


def test_ranked_training_emits_equal_complete_disjoint_windows() -> None:
    dataset = SyntheticClassLabeledDataset(19)
    config = ranked_config()
    executions = [
        ClassLabeledRankedTrainExecution(
            dataset=dataset,
            config=config,
            rank_context=DataRankContext(rank=rank, world_size=2),
            seed=31,
            resume_identity=ranked_identity(),
        )
        for rank in range(2)
    ]
    plans = [
        execution.plan_epoch(
            3,
            microbatches_per_window=2,
            max_microbatches=None,
        )
        for execution in executions
    ]

    assert plans[0].plan_digest == plans[1].plan_digest
    assert plans[0].expected_terminal_token != plans[1].expected_terminal_token
    assert plans[0].window_count == plans[1].window_count == 2
    assert plans[0].local_assigned_samples == 8
    assert plans[0].global_assigned_samples == 16
    assert plans[0].global_dropped_samples == 3

    rank_samples: list[set[int]] = []
    for execution, plan in zip(executions, plans, strict=True):
        reader = execution.open_epoch(plan)
        observed: set[int] = set()
        try:
            for window_ordinal in range(plan.window_count):
                window = reader.read_window()
                assert window is not None
                assert window.ordinal == window_ordinal
                assert len(window.batches) == 2
                assert all(fact.sample_count == 2 for fact in window.batch_facts)
                assert all(fact.loss_weight == 2.0 for fact in window.batch_facts)
                for _, conditions in window.batches:
                    observed.update(conditions["class_label"].tolist())
            assert reader.read_window() is None
            completion = reader.finish()
            assert completion.observed_windows == 2
            assert completion.observed_microbatches == 4
            assert completion.observed_samples == 8
            assert completion.terminal_token == plan.expected_terminal_token
        finally:
            reader.close()
        rank_samples.append(observed)

    assert rank_samples[0].isdisjoint(rank_samples[1])
    assert rank_samples[0] | rank_samples[1] == set(range(16))


def test_ranked_training_terminal_token_commits_config_epoch_and_rank() -> None:
    dataset = SyntheticClassLabeledDataset(24)
    identity = ranked_identity()

    def plan_for(
        *,
        rank: int = 0,
        epoch: int = 0,
        batch_size: int = 2,
    ) -> tuple[ClassLabeledRankedTrainExecution, RankedTrainEpochPlan]:
        execution = ClassLabeledRankedTrainExecution(
            dataset=dataset,
            config=ranked_config(batch_size=batch_size),
            rank_context=DataRankContext(rank=rank, world_size=2),
            seed=41,
            resume_identity=identity,
        )
        return execution, execution.plan_epoch(
            epoch,
            microbatches_per_window=2,
            max_microbatches=None,
        )

    execution, baseline = plan_for()
    _, changed_config = plan_for(batch_size=3)
    _, changed_epoch = plan_for(epoch=1)
    _, changed_rank = plan_for(rank=1)
    tokens = {
        baseline.expected_terminal_token,
        changed_config.expected_terminal_token,
        changed_epoch.expected_terminal_token,
        changed_rank.expected_terminal_token,
    }
    assert len(tokens) == 4
    assert all(
        len(token) == 64
        and token == token.lower()
        and set(token) <= set("0123456789abcdef")
        for token in tokens
    )

    forged_token = "0" * 64
    if forged_token == baseline.expected_terminal_token:
        forged_token = "1" * 64
    with pytest.raises(ValueError, match="does not belong"):
        execution.open_epoch(
            replace(baseline, expected_terminal_token=forged_token)
        )
    with pytest.raises(ValueError, match="expected_terminal_token"):
        replace(baseline, expected_terminal_token="A" * 64)


def test_ranked_training_rejects_partial_windows_and_foreign_plans() -> None:
    execution = ClassLabeledRankedTrainExecution(
        dataset=SyntheticClassLabeledDataset(20),
        config=ranked_config(),
        rank_context=DataRankContext(rank=0, world_size=2),
        seed=11,
        resume_identity=ranked_identity(),
    )

    with pytest.raises(ValueError, match="complete accumulation windows"):
        execution.plan_epoch(
            0,
            microbatches_per_window=2,
            max_microbatches=3,
        )

    plan = execution.plan_epoch(
        0,
        microbatches_per_window=2,
        max_microbatches=None,
    )
    assert plan.microbatch_count > 2
    with pytest.raises(ValueError, match="microbatch_count exceeds"):
        replace(plan, requested_max_microbatches=2)
    with pytest.raises(ValueError, match="does not belong"):
        execution.open_epoch(replace(plan, rank=1))

    configured = ClassLabeledRankedTrainExecution(
        dataset=SyntheticClassLabeledDataset(20),
        config=ranked_config(steps_per_epoch=3),
        rank_context=DataRankContext(rank=0, world_size=2),
        seed=11,
        resume_identity=ranked_identity(),
    )
    with pytest.raises(ValueError, match=r"steps_per_epoch.*complete"):
        configured.plan_epoch(
            0,
            microbatches_per_window=2,
            max_microbatches=None,
        )


def test_rank_zero_validation_covers_every_sample_once() -> None:
    dataset = SyntheticClassLabeledDataset(5)
    config = ranked_config(batch_size=2)
    executions = [
        ClassLabeledExactValidationExecution(
            dataset=dataset,
            config=config,
            rank_context=DataRankContext(rank=rank, world_size=2),
            seed=19,
            coverage_identity=ranked_identity(),
        )
        for rank in range(2)
    ]
    plans = [execution.plan_epoch(4) for execution in executions]

    assert plans[0].plan_digest == plans[1].plan_digest
    assert plans[0].global_expected_samples == 5
    assert plans[0].primary_batch_count == 3
    assert plans[1].primary_batch_count == 3
    assert plans[0].local_expected_samples == 5
    assert plans[1].local_expected_samples == 0
    assert len(executions[0]) == 3
    assert len(executions[1]) == 0

    rank_zero_reader = executions[0].open_epoch(plans[0])
    observed: list[int] = []
    observed_batch_sizes: list[int] = []
    try:
        while selected := rank_zero_reader.read_batch():
            _, conditions = selected.batch
            observed.extend(conditions["class_label"].tolist())
            observed_batch_sizes.append(selected.facts.sample_count)
            assert (
                selected.facts.sample_count
                == selected.coverage_span.stop - selected.coverage_span.start
            )
        rank_zero_receipt = rank_zero_reader.finish()
    finally:
        rank_zero_reader.close()

    empty_reader = executions[1].open_epoch(plans[1])
    try:
        assert empty_reader.read_batch() is None
        empty_receipt = empty_reader.finish()
    finally:
        empty_reader.close()

    assert observed == list(range(5))
    assert observed_batch_sizes == [2, 2, 1]
    assert rank_zero_receipt.observed_samples == 5
    assert empty_receipt.observed_samples == 0
    assert rank_zero_receipt.completed_spans == plans[0].local_spans
    assert empty_receipt.completed_spans == ()


def test_rank_zero_validation_declares_batch_count_when_samples_are_fewer_than_ranks(
) -> None:
    dataset = SyntheticClassLabeledDataset(3)
    config = ranked_config(batch_size=2)
    plans = [
        ClassLabeledExactValidationExecution(
            dataset=dataset,
            config=config,
            rank_context=DataRankContext(rank=rank, world_size=8),
            seed=19,
            coverage_identity=ranked_identity(),
        ).plan_epoch(0)
        for rank in range(8)
    ]

    assert {plan.plan_digest for plan in plans} == {plans[0].plan_digest}
    assert {plan.primary_batch_count for plan in plans} == {2}
    assert plans[0].local_expected_samples == 3
    assert all(plan.local_expected_samples == 0 for plan in plans[1:])
    with pytest.raises(ValueError, match="primary_batch_count"):
        replace(plans[0], primary_batch_count=0)


def test_data_loaders_bind_ranked_capabilities_by_object_identity() -> None:
    config = ranked_config()
    context = DataRankContext(rank=0, world_size=2)
    train = ClassLabeledRankedTrainExecution(
        dataset=SyntheticClassLabeledDataset(8),
        config=config,
        rank_context=context,
        seed=7,
        resume_identity=ranked_identity(),
    )
    validation = ClassLabeledExactValidationExecution(
        dataset=SyntheticClassLabeledDataset(3),
        config=config,
        rank_context=context,
        seed=8,
        coverage_identity=ranked_identity(),
    )
    ranked = RankedDataExecution(
        rank_context=context,
        train=train,
        validation=validation,
    )

    loaders = DataLoaders(
        train=train.batches,
        validation=validation.batches,
        ranked_execution=ranked,
    )
    assert loaders.ranked_execution is ranked

    with pytest.raises(ValueError, match="exact train iterable"):
        DataLoaders(
            train=[object()],
            validation=validation.batches,
            ranked_execution=ranked,
        )


def test_ranked_training_requires_explicit_no_padding_policy() -> None:
    with pytest.raises(ValueError, match="drop_last=true"):
        ClassLabeledRankedTrainExecution(
            dataset=SyntheticClassLabeledDataset(8),
            config=ranked_config(drop_last=False),
            rank_context=DataRankContext(rank=0, world_size=2),
            seed=7,
            resume_identity=ranked_identity(),
        )

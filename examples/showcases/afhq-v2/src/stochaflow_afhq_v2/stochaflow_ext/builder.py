"""AFHQ-v2 source-to-class-batch composition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.data import DataLoader

from stochaflow.extensions import (
    IMAGE_DATA_SOURCES,
    REGISTRIES,
    DataArtifactBinding,
    DataArtifactBindings,
    DataArtifactIdentity,
    DataBuilder,
    DataBuilderContext,
    DataLoaders,
    DataSourceContext,
    ManagedDataArtifact,
)
from stochaflow_afhq_v2.stochaflow_ext.config import (
    AFHQV2DataBuilderConfig,
    AFHQV2LoaderRecipeConfig,
)
from stochaflow_afhq_v2.stochaflow_ext.data import (
    AFHQV2ImageFolderArtifactPayload,
)
from stochaflow_afhq_v2.stochaflow_ext.dataset import (
    AFHQV2ClassDataset,
    AFHQV2EpochSampler,
    collate_afhq_v2_class_batch,
)

_BUILDER_NAME = "afhq-v2.class-images"
_BINDING_ID = "source"


def _expected_identity(
    builder: AFHQV2DataBuilder,
) -> DataArtifactIdentity | None:
    if not builder.context.strict_resume:
        return None
    assert builder.context.expected_artifacts is not None
    expected = builder.context.expected_artifacts.identity_for(_BINDING_ID)
    if expected.source_name != builder.config.source.name:
        raise ValueError(
            "AFHQ-v2 strict resume expected a different registered source"
        )
    return expected


def _loader_kwargs(
    config: AFHQV2LoaderRecipeConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "generator": torch.Generator().manual_seed(seed),
    }
    if config.num_workers > 0:
        kwargs["persistent_workers"] = config.persistent_workers
        if config.prefetch_factor is not None:
            kwargs["prefetch_factor"] = config.prefetch_factor
    return kwargs


def _evaluation_loader(
    dataset: AFHQV2ClassDataset,
    config: AFHQV2LoaderRecipeConfig,
    *,
    seed: int,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_afhq_v2_class_batch,
        **_loader_kwargs(config, seed=seed),
    )


@REGISTRIES.data_builders.register(_BUILDER_NAME)
class AFHQV2DataBuilder(DataBuilder):
    """Build deterministic labeled AFHQ-v2 train/validation/test loaders."""

    def __init__(self, context: DataBuilderContext) -> None:
        super().__init__(context)
        self.config = AFHQV2DataBuilderConfig.from_params(context.params)

    def build(self) -> DataLoaders:
        """Materialize one artifact and expose its official labeled splits."""

        self.context.require_artifact_ids((_BINDING_ID,))
        expected = _expected_identity(self)
        source = IMAGE_DATA_SOURCES.create(
            self.config.source.name,
            self.config.source.params,
            config_path="data.params.source",
        )
        materialization = self.config.source.materialization
        artifact_value = cast(
            object,
            source.materialize(
                DataSourceContext(
                    cache_root=Path(materialization.cache_root),
                    policy=materialization.policy,
                    verification=materialization.verification,
                    expected_identity=expected,
                )
            ),
        )
        if not isinstance(artifact_value, ManagedDataArtifact):
            raise TypeError(
                "AFHQ-v2 class-images source must return ManagedDataArtifact"
            )
        artifact = artifact_value
        if artifact.identity.source_name != self.config.source.name:
            raise ValueError(
                "AFHQ-v2 source returned an identity for a different source"
            )
        if expected is not None and artifact.identity != expected:
            raise ValueError(
                "AFHQ-v2 strict resume data artifact identity does not match"
            )
        payload = artifact.payload
        if not isinstance(payload, AFHQV2ImageFolderArtifactPayload):
            raise TypeError(
                "AFHQ-v2 class-images source returned an incompatible payload"
            )
        bindings = DataArtifactBindings(
            (
                DataArtifactBinding(
                    id=_BINDING_ID,
                    identity=artifact.identity,
                ),
            )
        )
        self.context.verify_artifacts(bindings)

        train = AFHQV2ClassDataset(
            roots=payload.roots,
            records=payload.train,
            role="train",
            class_mapping=payload.class_mapping,
            image=self.config.image,
            seed=self.context.seed,
            augment=True,
        )
        if payload.validation is None or payload.test is None:
            raise ValueError(
                "AFHQ-v2 official source must provide validation and test splits"
            )
        validation = AFHQV2ClassDataset(
            roots=payload.roots,
            records=payload.validation,
            role="validation",
            class_mapping=payload.class_mapping,
            image=self.config.image,
            seed=self.context.seed,
            augment=False,
        )
        test = AFHQV2ClassDataset(
            roots=payload.roots,
            records=payload.test,
            role="test",
            class_mapping=payload.class_mapping,
            image=self.config.image,
            seed=self.context.seed,
            augment=False,
        )
        loader = self.config.loader
        train_loader = DataLoader(
            train,
            batch_size=loader.batch_size,
            shuffle=False,
            sampler=AFHQV2EpochSampler(
                len(train),
                seed=self.context.seed,
                shuffle=loader.shuffle,
            ),
            drop_last=loader.drop_last,
            collate_fn=collate_afhq_v2_class_batch,
            **_loader_kwargs(loader, seed=self.context.seed),
        )
        return DataLoaders(
            train=train_loader,
            validation=_evaluation_loader(
                validation,
                loader,
                seed=self.context.seed + 1,
            ),
            test=_evaluation_loader(
                test,
                loader,
                seed=self.context.seed + 2,
            ),
            steps_per_epoch=(
                None
                if loader.steps_per_epoch == "auto"
                else loader.steps_per_epoch
            ),
            artifact_bindings=bindings,
        )


__all__ = ["AFHQV2DataBuilder"]

"""Built-in referenced image-folder source providers."""

from __future__ import annotations

import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from stochaflow.data.artifact_io import canonical_directory
from stochaflow.data.artifacts import (
    DataSourceContext,
    ReferencedDataArtifact,
)
from stochaflow.data.image_contracts import (
    IMAGE_DATA_SOURCES,
    ImageDataSource,
    ImageFilePair,
    ImageFileRecord,
    ImageFolderArtifactPayload,
    PairedImageFolderArtifactPayload,
)
from stochaflow.data.reference_artifacts import materialize_reference
from stochaflow.utils.config import ConfigError, coerce_config_section


@dataclass(slots=True)
class ImageFolderSourceConfig:
    """Provider parameters for an external image folder."""

    root: str
    layout: str = "flat"

    def validate(self, *, path: str) -> None:
        """Validate folder locator and native split layout."""

        root = cast(object, self.root)
        if not isinstance(root, str) or not root.strip():
            raise ConfigError(f"{path}.root must be a non-empty string")
        layout = cast(object, self.layout)
        if not isinstance(layout, str) or layout not in {"flat", "split"}:
            raise ConfigError(f"{path}.layout must be flat or split")


@dataclass(slots=True)
class PairedImageFolderSourceConfig:
    """Provider parameters for external aligned HR/LR image folders."""

    high_resolution_root: str
    low_resolution_root: str
    layout: str = "flat"

    def validate(self, *, path: str) -> None:
        """Validate paired folder locators and native split layout."""

        for name in ("high_resolution_root", "low_resolution_root"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"{path}.{name} must be a non-empty string")
        layout = cast(object, self.layout)
        if not isinstance(layout, str) or layout not in {"flat", "split"}:
            raise ConfigError(f"{path}.layout must be flat or split")


def absolute_directory(value: object, *, path: str) -> Path:
    """Resolve one configured directory with a pathful error."""

    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty path string")
    try:
        return canonical_directory(Path(value), label=path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{path} does not exist: {value}") from exc


def is_link_or_reparse(path: Path) -> bool:
    """Return whether one path is a symlink or Windows reparse point."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return path.is_symlink() or bool(attributes & reparse_flag)


def partition_roots(root: Path, layout: str) -> dict[str, Path]:
    """Resolve flat or native-split roots without following split links."""

    if layout == "flat":
        return {"train": root}
    if layout != "split":
        raise ConfigError("image source params.layout must be flat or split")
    train = root / "train"
    try:
        canonical_train = canonical_directory(
            train,
            label="split image folder train directory",
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"split image folder requires a train directory: {train}"
        ) from None
    roots = {"train": canonical_train}
    for candidate in (root / "validation", root / "val", root / "test"):
        if is_link_or_reparse(candidate):
            raise ValueError(
                "split image folder must not use linked split directories: "
                f"{candidate}"
            )
    validation = [
        candidate
        for candidate in (root / "validation", root / "val")
        if candidate.is_dir()
    ]
    if len(validation) > 1:
        raise ValueError(
            "split image folder cannot contain both validation and val"
        )
    if validation:
        roots["validation"] = canonical_directory(
            validation[0],
            label="split image folder validation directory",
        )
    if (root / "test").is_dir():
        roots["test"] = canonical_directory(
            root / "test",
            label="split image folder test directory",
        )
    return roots


def records_by_tree(
    records: Sequence[ImageFileRecord],
) -> dict[str, tuple[ImageFileRecord, ...]]:
    """Group a sorted inventory by its declared root tree."""

    grouped: dict[str, list[ImageFileRecord]] = {}
    for record in records:
        grouped.setdefault(record.tree, []).append(record)
    return {
        tree: tuple(selected)
        for tree, selected in grouped.items()
    }


@IMAGE_DATA_SOURCES.register("image_folder")
class ImageFolderDataSource(ImageDataSource):
    """Index an external image directory without copying its content."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> ReferencedDataArtifact[ImageFolderArtifactPayload]:
        config = cast(
            ImageFolderSourceConfig,
            coerce_config_section(
                ImageFolderSourceConfig,
                self.params,
                f"{self.config_path}.params",
            ),
        )
        config.validate(path=f"{self.config_path}.params")
        root = absolute_directory(
            config.root,
            path=f"{self.config_path}.params.root",
        )
        roots = partition_roots(root, config.layout)
        layout = {
            "type": "image_folder",
            "mode": config.layout,
            "trees": sorted(roots),
        }
        index_root, identity, records = materialize_reference(
            context,
            source_name="image_folder",
            artifact_type="stochaflow.image-folder-reference.v2",
            roots=roots,
            layout=layout,
        )
        grouped = records_by_tree(records)
        return ReferencedDataArtifact(
            index_root=index_root,
            manifest_path=index_root / "manifest.json",
            identity=identity,
            payload=ImageFolderArtifactPayload(
                roots=roots,
                train=grouped["train"],
                validation=grouped.get("validation"),
                test=grouped.get("test"),
            ),
        )


def paired_partition_roots(
    high_root: Path,
    low_root: Path,
    layout: str,
) -> tuple[dict[str, Path], tuple[str, ...]]:
    """Resolve aligned HR/LR split roots."""

    high = partition_roots(high_root, layout)
    low = partition_roots(low_root, layout)
    if set(high) != set(low):
        raise ValueError(
            "paired image folders must expose the same native splits"
        )
    roots = {
        f"{role}.high_resolution": high[role]
        for role in sorted(high)
    }
    roots.update(
        {
            f"{role}.low_resolution": low[role]
            for role in sorted(low)
        }
    )
    return roots, tuple(sorted(high))


def pair_records(
    high: Sequence[ImageFileRecord],
    low: Sequence[ImageFileRecord],
    *,
    role: str,
) -> tuple[ImageFilePair, ...]:
    """Pair HR/LR records by unique relative stem."""

    def by_stem(
        records: Sequence[ImageFileRecord],
        label: str,
    ) -> dict[str, ImageFileRecord]:
        result: dict[str, ImageFileRecord] = {}
        for record in records:
            stem = PurePosixPath(record.path).with_suffix("").as_posix()
            if stem in result:
                raise ValueError(
                    f"duplicate {label} relative image stem '{stem}' in {role}"
                )
            result[stem] = record
        return result

    high_by_stem = by_stem(high, "high-resolution")
    low_by_stem = by_stem(low, "low-resolution")
    missing_low = sorted(set(high_by_stem) - set(low_by_stem))
    missing_high = sorted(set(low_by_stem) - set(high_by_stem))
    if missing_low or missing_high:
        details: list[str] = []
        if missing_low:
            details.append("missing LR: " + ", ".join(missing_low))
        if missing_high:
            details.append("missing HR: " + ", ".join(missing_high))
        raise ValueError(
            "paired image folders do not align; " + "; ".join(details)
        )
    return tuple(
        ImageFilePair(high_by_stem[key], low_by_stem[key])
        for key in sorted(high_by_stem)
    )


@IMAGE_DATA_SOURCES.register("paired_image_folders")
class PairedImageFolderDataSource(ImageDataSource):
    """Index aligned external HR/LR directories without copying content."""

    def materialize(
        self,
        context: DataSourceContext,
    ) -> ReferencedDataArtifact[PairedImageFolderArtifactPayload]:
        config = cast(
            PairedImageFolderSourceConfig,
            coerce_config_section(
                PairedImageFolderSourceConfig,
                self.params,
                f"{self.config_path}.params",
            ),
        )
        config.validate(path=f"{self.config_path}.params")
        high_root = absolute_directory(
            config.high_resolution_root,
            path=f"{self.config_path}.params.high_resolution_root",
        )
        low_root = absolute_directory(
            config.low_resolution_root,
            path=f"{self.config_path}.params.low_resolution_root",
        )
        if high_root == low_root:
            raise ValueError("paired image roots must be distinct")
        roots, roles = paired_partition_roots(
            high_root,
            low_root,
            config.layout,
        )
        layout = {
            "type": "paired_image_folders",
            "mode": config.layout,
            "roles": list(roles),
        }
        index_root, identity, records = materialize_reference(
            context,
            source_name="paired_image_folders",
            artifact_type="stochaflow.paired-image-folder-reference.v2",
            roots=roots,
            layout=layout,
        )
        grouped = records_by_tree(records)
        pairs = {
            role: pair_records(
                grouped[f"{role}.high_resolution"],
                grouped[f"{role}.low_resolution"],
                role=role,
            )
            for role in roles
        }
        return ReferencedDataArtifact(
            index_root=index_root,
            manifest_path=index_root / "manifest.json",
            identity=identity,
            payload=PairedImageFolderArtifactPayload(
                roots=roots,
                train=pairs["train"],
                validation=pairs.get("validation"),
                test=pairs.get("test"),
            ),
        )


__all__ = [
    "ImageFolderDataSource",
    "ImageFolderSourceConfig",
    "PairedImageFolderDataSource",
    "PairedImageFolderSourceConfig",
    "absolute_directory",
    "is_link_or_reparse",
    "pair_records",
    "paired_partition_roots",
    "partition_roots",
    "records_by_tree",
]

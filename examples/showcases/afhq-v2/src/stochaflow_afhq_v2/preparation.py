"""Public facade for AFHQ-v2 artifact preparation."""

from ._preparation.archive import inspect_archive
from ._preparation.contracts import (
    DatasetContract,
    PreparationError,
    PreparationPlan,
    PreparedArtifact,
    PreparedImageRecord,
    SourceArchive,
    SourceImage,
    SourceIntegrityError,
    SourceLock,
)
from ._preparation.downloading import download_official_archive
from ._preparation.locking import ArtifactPreparationLock
from ._preparation.planning import build_preparation_plan
from ._preparation.prepared_artifact import (
    load_prepared_image_records,
    require_prepared_artifact,
    verify_prepared_artifact,
)
from ._preparation.publication import prepare_archive
from ._preparation.safe_file import load_verified_prepared_image, sha256_file
from ._preparation.source_acquisition import acquire_official_archive
from ._preparation.source_lock import load_source_lock
from ._preparation.source_session import SourceArchiveSession

__all__ = [
    "ArtifactPreparationLock",
    "DatasetContract",
    "PreparationError",
    "PreparationPlan",
    "PreparedArtifact",
    "PreparedImageRecord",
    "SourceArchive",
    "SourceArchiveSession",
    "SourceImage",
    "SourceIntegrityError",
    "SourceLock",
    "acquire_official_archive",
    "build_preparation_plan",
    "download_official_archive",
    "inspect_archive",
    "load_prepared_image_records",
    "load_source_lock",
    "load_verified_prepared_image",
    "prepare_archive",
    "require_prepared_artifact",
    "sha256_file",
    "verify_prepared_artifact",
]

"""Atomic no-replace publication contracts for sampling bundles."""

from pathlib import Path

import pytest

from stochaflow.sampling.publication import (
    abort_sampling_staging,
    create_sampling_publication_staging,
    publish_sampling_staging,
)


def test_sampling_staging_publishes_one_complete_absent_directory(
    tmp_path: Path,
) -> None:
    publication = create_sampling_publication_staging(tmp_path / "samples")
    (publication.staging / "resolved_sampling.yaml").write_text(
        "kind: sampling\n",
        encoding="utf-8",
    )

    published = publish_sampling_staging(publication)

    assert published == (tmp_path / "samples").resolve()
    assert not publication.staging.exists()
    assert (published / "resolved_sampling.yaml").read_text(encoding="utf-8") == (
        "kind: sampling\n"
    )


def test_sampling_staging_rejects_existing_target_and_cleans_abort(
    tmp_path: Path,
) -> None:
    target = tmp_path / "samples"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        create_sampling_publication_staging(target)
    assert sentinel.read_text(encoding="utf-8") == "preserve"

    publication = create_sampling_publication_staging(tmp_path / "other")
    (publication.staging / "partial.txt").write_text("partial", encoding="utf-8")
    abort_sampling_staging(publication)
    assert not publication.staging.exists()
    assert not publication.destination.exists()

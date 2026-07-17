"""Named real-pipeline archive fixtures shared by composition consumers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from tests.infra.workload_artifacts import (
    SeededArchiveArtifact,
    SeededArchiveClone,
    build_seeded_archive,
    clone_seeded_archive,
    named_corpus_specs,
    schema_coverage_corpus_specs,
)


@pytest.fixture(scope="session")
def seeded_archive() -> SeededArchiveArtifact:
    """Shared immutable named schema-coverage archive for read-only consumers."""
    return build_seeded_archive(schema_coverage_corpus_specs())


@pytest.fixture
def seeded_archive_writable(seeded_archive: SeededArchiveArtifact, tmp_path: Path) -> SeededArchiveClone:
    """Private full-root clone for a mutating consumer."""
    return clone_seeded_archive(seeded_archive, tmp_path / "seeded-archive-clone")


@pytest.fixture
def named_seeded_archive(
    workspace_env: dict[str, Path],
) -> Callable[[str], Path]:
    """Clone one registered immutable workload into this test's archive root."""
    archive_root = workspace_env["archive_root"]

    def seed(name: str) -> Path:
        artifact = build_seeded_archive(named_corpus_specs(name))
        clone_seeded_archive(artifact, archive_root)
        return archive_root / "index.db"

    return seed

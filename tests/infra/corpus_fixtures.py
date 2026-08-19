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


@pytest.fixture(scope="session")
def corpus_fidelity_archive(seeded_archive: SeededArchiveArtifact) -> SeededArchiveArtifact:
    """Real production-route archive used by corpus acceptance gate tests."""
    return seeded_archive


@pytest.fixture
def seeded_archive_writable(seeded_archive: SeededArchiveArtifact, tmp_path: Path) -> SeededArchiveClone:
    """Private full-root clone for a mutating consumer."""
    return clone_seeded_archive(seeded_archive, tmp_path / "seeded-archive-clone")


@pytest.fixture
def named_seeded_archive(
    workspace_env: dict[str, Path],
) -> Callable[[str], Path]:
    """Clone one registered immutable workload into this test's archive root.

    For consumers that MUTATE the archive (ingest, insight rebuild, marks,
    maintenance). A non-mutating consumer should take
    :func:`named_seeded_archive_ro` instead and skip the clone entirely.
    """
    archive_root = workspace_env["archive_root"]

    def seed(name: str) -> Path:
        artifact = build_seeded_archive(named_corpus_specs(name))
        clone_seeded_archive(artifact, archive_root)
        return archive_root / "index.db"

    return seed


@pytest.fixture
def named_seeded_archive_ro(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], Path]:
    """Point this test's archive root straight at an immutable shared artifact.

    Published artifacts are already chmod-read-only with their WAL collapsed
    into a DELETE journal, so any number of processes and xdist workers can
    open them read-only at once. A read-only consumer therefore does not need
    its own copy, and the per-test reflink clone it was paying for is pure
    cost: ``tests/unit/cli/test_plain_cli_snapshots.py`` spent 95% of its
    worker-seconds in fixture setup for tests that only run query verbs.

    This intentionally does not depend on ``workspace_env``: the autouse
    ``_clear_polylogue_env`` fixture already strips inherited ``POLYLOGUE_*``
    configuration and redirects the XDG roots into ``tmp_path``, so the only
    thing left to set is where the archive lives. Taking ``workspace_env``
    would additionally clone an empty archive template per test that nothing
    then reads.

    A consumer that writes must take ``named_seeded_archive`` instead. Writes
    here fail loudly on the artifact's read-only mode bits rather than
    silently mutating the cache every other test shares.
    """

    def seed(name: str) -> Path:
        artifact = build_seeded_archive(named_corpus_specs(name))
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(artifact.root))
        return artifact.root / "index.db"

    return seed

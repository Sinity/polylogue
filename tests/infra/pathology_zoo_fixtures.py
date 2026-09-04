"""Session-scoped fixture reuse for the production-ingested pathology zoo.

:func:`tests.infra.pathology_zoo.build_pathology_zoo` runs a full two-pass
production ingest (real parsing, real writes, real revision-governance
replay). Several test modules each need that same manifest-backed archive,
some only to read it and some to mutate a private copy of it. Building it
once per test session and handing out cheap clones is the same shape as
``tests.infra.corpus_fixtures.seeded_archive`` for the schema-coverage corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.infra.pathology_zoo import PathologyZoo, build_pathology_zoo_ro, clone_pathology_zoo


@pytest.fixture(scope="session")
def pathology_zoo_archive(tmp_path_factory: pytest.TempPathFactory) -> PathologyZoo:
    """Build the production pathology zoo once for the whole test session.

    Every consumer treats ``archive_root`` as read-only: a test that only
    verifies the manifest, reads the archive, or copies it into its own
    ``tmp_path`` before mutating may depend on this directly. A test that
    needs to mutate the zoo's own root in place should depend on
    :func:`pathology_zoo_writable` instead.
    """
    canonical = build_pathology_zoo_ro()
    return clone_pathology_zoo(canonical, tmp_path_factory.mktemp("pathology-zoo-session"))


@pytest.fixture
def pathology_zoo_writable(pathology_zoo_archive: PathologyZoo, tmp_path: Path) -> PathologyZoo:
    """Private per-test writable copy of the shared zoo for mutation tests."""
    return clone_pathology_zoo(pathology_zoo_archive, tmp_path / "pathology-zoo-clone")


__all__ = ["pathology_zoo_archive", "pathology_zoo_writable"]

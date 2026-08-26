"""Named real-pipeline archive fixtures shared by composition consumers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from tests.infra.pathology_zoo import (
    PathologyZoo,
    build_pathology_zoo,
    build_pathology_zoo_ro,
)
from tests.infra.workload_artifacts import (
    SeededArchiveArtifact,
    SeededArchiveClone,
    SeededArchiveQueryLease,
    acquire_query_only_seeded_archive,
    build_seeded_archive,
    clone_seeded_archive,
    default_cache_root,
    named_corpus_specs,
    schema_coverage_corpus_specs,
    seeded_archive_key,
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
def seeded_archive_writable(seeded_archive: SeededArchiveArtifact, tmp_path: Path) -> Iterator[SeededArchiveClone]:
    """Private full-root clone for a mutating consumer."""
    clone = clone_seeded_archive(seeded_archive, tmp_path / "seeded-archive-clone")
    try:
        yield clone
    finally:
        clone.close()


@pytest.fixture
def named_seeded_archive(
    workspace_env: dict[str, Path],
    request: pytest.FixtureRequest,
) -> Callable[[str], SeededArchiveClone]:
    """Clone one registered immutable workload into this test's archive root.

    For consumers that MUTATE the archive (ingest, insight rebuild, marks,
    maintenance). A non-mutating consumer should take
    :func:`named_seeded_archive_ro` instead and skip the clone entirely.
    """
    archive_root = workspace_env["archive_root"]
    clones: list[SeededArchiveClone] = []

    def close_clones() -> None:
        errors: list[RuntimeError] = []
        for clone in reversed(clones):
            try:
                clone.close()
            except RuntimeError as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    request.addfinalizer(close_clones)

    def seed(name: str) -> SeededArchiveClone:
        artifact = build_seeded_archive(named_corpus_specs(name))
        clone = clone_seeded_archive(artifact, archive_root)
        clones.append(clone)
        return clone

    return seed


@pytest.fixture
def named_seeded_archive_ro(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    workspace_env: dict[str, Path],
) -> Callable[[str], SeededArchiveQueryLease]:
    """Give read consumers an authenticated, query-only artifact lease.

    CLI and completion tests open ``POLYLOGUE_ARCHIVE_ROOT`` by ordinary
    filesystem paths.  The artifact is already sealed and content-addressed,
    so a read-only consumer can share it directly; the lease remains held for
    the fixture lifetime and is closed by the fixture finalizer. Only a
    mutating consumer receives a private clone through
    :func:`named_seeded_archive`.
    """

    leases: list[SeededArchiveQueryLease] = []

    def close_leases() -> None:
        for lease in reversed(leases):
            lease.close()

    request.addfinalizer(close_leases)

    def seed(name: str) -> SeededArchiveQueryLease:
        specs = named_corpus_specs(name)
        artifact = build_seeded_archive(specs)
        lease = acquire_query_only_seeded_archive(artifact, seeded_archive_key(specs))
        leases.append(lease)
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(artifact.root))
        monkeypatch.setattr("polylogue.daemon.api_auth.load_or_mint_api_auth_token", lambda *_args, **_kwargs: None)
        return lease

    return seed


@pytest.fixture(scope="session")
def pathology_zoo_artifact() -> PathologyZoo:
    """Shared read-only aggregate pathology evidence."""
    return build_pathology_zoo_ro()


@pytest.fixture
def pathology_zoo_writable(pathology_zoo_artifact: PathologyZoo, tmp_path: Path) -> PathologyZoo:
    """Private writable clone; mutations never touch the aggregate cache."""
    return build_pathology_zoo(tmp_path / "pathology-zoo", cache_root=default_cache_root())

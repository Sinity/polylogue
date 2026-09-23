"""Demand-driven pytest resources for the small workload-artifact pilot.

These are ordinary pytest fixtures over the existing provider generator and
artifact capability owners.  The fixtures stay lazy: importing this module
does not generate bytes, open a database, or start a daemon.  A parser-only
test requests ``pilot_provider_packages``; archive and transport tests opt in
to the heavier resources explicitly.

Nothing here owns construction, caching or cloning. The archive resources go
through :func:`tests.infra.integration_profile.build_integration_archive`,
which is the declared owner of this recipe and publishes into the shared
artifact cache under the ``default/integration`` reachability key -- so a
second module, worker or run reuses the published artifact instead of
rebuilding one in a private cache root.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from polylogue.sources.parsers.base import ParsedSession
from tests.infra.source_builders import ProviderSourcePackage, provider_source_package
from tests.infra.workload_artifacts import (
    SeededArchiveArtifact,
    SeededArchiveClone,
    SeededArchiveQueryLease,
    acquire_query_only_seeded_archive,
    clone_seeded_archive,
    seeded_archive_key,
)

if TYPE_CHECKING:
    from tests.infra.daemon_operations import DaemonOperationStack
    from tests.infra.integration_profile import IntegrationSelection


def _pilot_selection() -> IntegrationSelection:
    """Resolve the existing heterogeneous integration recipe lazily."""
    from tests.infra.integration_profile import default_integration_selection

    return default_integration_selection()


def build_pilot_provider_packages(root: Path) -> tuple[ProviderSourcePackage, ...]:
    """Write the pilot selection's provider bytes under *root*.

    Exposed as a plain function, not only as a fixture, so a test can measure
    or instrument one acquisition (bytes written, tiers opened) without
    depending on whether a module-scoped fixture happened to run first.
    """
    from polylogue.schemas.synthetic import SyntheticCorpus

    packages: list[ProviderSourcePackage] = []
    for index, spec in enumerate(_pilot_selection().corpus_specs()):
        written = SyntheticCorpus.write_spec_artifacts(spec, root / spec.provider, prefix=f"pilot-{index:02d}")
        packages.append(
            provider_source_package(
                spec.provider,
                written.files,
                generator_id="synthetic-corpus:integration-pilot:v1",
                schema_inputs=(spec.package_version, spec.element_kind or "default"),
                schedule_digest=f"seed:{spec.seed}:style:{spec.style}",
            )
        )
    return tuple(packages)


@pytest.fixture(scope="session")
def pilot_provider_packages(tmp_path_factory: pytest.TempPathFactory) -> tuple[ProviderSourcePackage, ...]:
    """Generate provider bytes once per pytest worker on first parser use."""
    return build_pilot_provider_packages(tmp_path_factory.mktemp("pilot-provider-bytes"))


@pytest.fixture(scope="session")
def pilot_parsed_sessions(pilot_provider_packages: tuple[ProviderSourcePackage, ...]) -> tuple[ParsedSession, ...]:
    """Parse provider bytes without acquiring any archive/database resource."""
    from polylogue.sources import iter_source_sessions

    return tuple(
        session
        for package in pilot_provider_packages
        for source in package.admitted_sources()
        for session in iter_source_sessions(source)
    )


@pytest.fixture(scope="session")
def pilot_artifact() -> SeededArchiveArtifact:
    """Acquire the shared multi-provider artifact once per pytest worker."""
    from tests.infra.integration_profile import build_integration_archive

    return build_integration_archive()


@pytest.fixture(scope="session")
def pilot_query_archive(
    pilot_artifact: SeededArchiveArtifact,
) -> Iterator[SeededArchiveQueryLease]:
    """Share one authenticated read-only artifact lease across pilot reads."""
    selection = _pilot_selection()
    lease = acquire_query_only_seeded_archive(
        pilot_artifact,
        seeded_archive_key(selection.corpus_specs()),
    )
    try:
        yield lease
    finally:
        lease.close()


@pytest.fixture
def pilot_writable_archive(
    pilot_artifact: SeededArchiveArtifact,
    tmp_path: Path,
) -> Iterator[SeededArchiveClone]:
    """Provide a private clone for tests that must commit archive mutations."""
    clone = clone_seeded_archive(pilot_artifact, tmp_path / "pilot-writable-archive")
    try:
        yield clone
    finally:
        clone.close()


@pytest.fixture
def pilot_daemon_operations(
    pilot_artifact: SeededArchiveArtifact,
    tmp_path: Path,
) -> Iterator[DaemonOperationStack]:
    """Start the real UDS operation transport only for transport tests."""
    from tests.infra.daemon_operations import running_daemon_operations

    archive_root = tmp_path / "pilot-daemon-archive"

    def seed(root: Path) -> None:
        clone = clone_seeded_archive(pilot_artifact, root)
        clone.close()

    with running_daemon_operations(archive_root, seed_archive=seed) as stack:
        yield stack


__all__ = [
    "build_pilot_provider_packages",
    "pilot_artifact",
    "pilot_daemon_operations",
    "pilot_parsed_sessions",
    "pilot_provider_packages",
    "pilot_query_archive",
    "pilot_writable_archive",
]

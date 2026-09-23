"""Small demand-driven resource pilot.

The expectations below are authored independently of the artifact manifest:
the integration recipe supplies provider/native identity, while the tests
assert parser, read, mutation-isolation, and transport behavior separately.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import pytest

from tests.infra.pilot_scenarios import pilot_scenarios

if TYPE_CHECKING:
    from polylogue.sources.parsers.base import ParsedSession
    from tests.infra.daemon_operations import DaemonOperationStack
    from tests.infra.pilot_scenarios import PilotScenario
    from tests.infra.source_builders import ProviderSourcePackage
    from tests.infra.workload_artifacts import SeededArchiveArtifact, SeededArchiveClone, SeededArchiveQueryLease

pytest_plugins = ("tests.infra.pilot_resources",)


def _provider_native_ids(package: ProviderSourcePackage) -> tuple[str, ...]:
    """Return the native ids from provider bytes, independent of archive rows."""
    from polylogue.sources import iter_source_sessions

    return tuple(
        str(session.provider_session_id)
        for source in package.admitted_sources()
        for session in iter_source_sessions(source)
    )


def test_parser_resource_needs_only_provider_bytes(
    pilot_provider_packages: tuple[ProviderSourcePackage, ...],
    pilot_parsed_sessions: tuple[ParsedSession, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parser bytes are sufficient; no SQLite tier or daemon is needed."""

    def unexpected_database_open(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("parser-only pilot must not open a database tier")

    monkeypatch.setattr(sqlite3, "connect", unexpected_database_open)
    observed = tuple(str(session.provider_session_id) for session in pilot_parsed_sessions)
    expected = tuple(native_id for package in pilot_provider_packages for native_id in _provider_native_ids(package))
    assert observed == expected
    assert {package.provider for package in pilot_provider_packages} == {"chatgpt", "codex"}
    assert all(package.wire_hashes and package.generator_id for package in pilot_provider_packages)


def test_pilot_reuses_the_declared_shared_artifact_instead_of_rebuilding_one(
    pilot_artifact: SeededArchiveArtifact,
) -> None:
    """The pilot consumes the artifact the cache already protects and reuses it.

    This is the migration's whole point, so it is asserted rather than left
    to a fixture body: a pilot that builds into a private cache root (as this
    one used to) republishes the same recipe per module and can never hit the
    published artifact, while the GC reachability entry protecting that
    recipe names a key nothing builds.
    """
    from tests.infra.integration_profile import build_integration_archive, default_integration_selection
    from tests.infra.workload_artifacts import (
        current_seeded_archive_reachability,
        default_cache_root,
        seeded_archive_key,
    )

    key = seeded_archive_key(default_integration_selection().corpus_specs())
    declared = {(entry.kind, entry.name): entry.key.value for entry in current_seeded_archive_reachability().entries}
    assert declared[("default", "integration")] == key.value

    # Published into the shared cache, not a per-module temporary directory.
    assert pilot_artifact.root.parent.name == "artifacts"
    assert pilot_artifact.root.parent.parent == default_cache_root()
    assert pilot_artifact.root.name == key.value.rsplit(":", 1)[-1]

    reacquired = build_integration_archive()
    assert reacquired.root == pilot_artifact.root
    assert reacquired.manifest.manifest_id == pilot_artifact.manifest.manifest_id


def test_pilot_repeated_provider_build_reports_setup_and_byte_cost(tmp_path: Path) -> None:
    """Keep a small, reproducible cost receipt for the demand-driven pilot.

    This deliberately measures provider-byte construction only: parser tests
    must not pay for archive tiers, and the immutable archive reuse assertion
    above covers the separate cache/build path.  The receipt is diagnostic,
    while equal identities and byte counts make the comparison deterministic.
    """
    from tests.infra.pilot_resources import build_pilot_provider_packages

    measurements: list[tuple[float, int, tuple[str, ...]]] = []
    for attempt in range(2):
        root = tmp_path / f"build-{attempt}"
        started = perf_counter()
        packages = build_pilot_provider_packages(root)
        elapsed = perf_counter() - started
        bytes_written = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        measurements.append((elapsed, bytes_written, tuple(package.identity for package in packages)))

    first, second = measurements
    assert first[1] > 0
    assert second[1] == first[1]
    assert second[2] == first[2]
    print(
        "pilot-provider-build-cost="
        f"{{'builds': 2, 'bytes': {first[1]}, 'setup_seconds': "
        f"[{first[0]:.6f}, {second[0]:.6f}], 'identities_equal': true}}"
    )


def test_pilot_archive_reuse_reports_cold_warm_setup_and_bytes(tmp_path: Path) -> None:
    """Measure the canonical archive build once, then verify warm reuse.

    A second request with the same recipe must hit the workload-artifact
    identity/memoization path.  The assertion on the artifact root and
    manifest prevents a test-only cache from merely producing equivalent
    output while repeating the expensive construction.
    """
    from tests.infra.integration_profile import build_integration_archive

    cache_root = tmp_path / "pilot-artifact-cache"
    timings: list[float] = []
    artifacts = []
    for _ in range(2):
        started = perf_counter()
        artifact = build_integration_archive(cache_root=cache_root)
        timings.append(perf_counter() - started)
        artifacts.append(artifact)

    cold, warm = artifacts
    assert cold.root == warm.root
    assert cold.manifest.manifest_id == warm.manifest.manifest_id
    assert cold.manifest.resources.total_bytes > 0
    assert warm.manifest.resources.total_bytes == cold.manifest.resources.total_bytes
    print(
        "pilot-archive-build-cost="
        f"{{'builds': 2, 'bytes': {cold.manifest.resources.total_bytes}, "
        f"'setup_seconds': [{timings[0]:.6f}, {timings[1]:.6f}], "
        "'same_artifact': true}}"
    )


def test_parser_resource_acquisition_opens_no_archive_tier(tmp_path: Path) -> None:
    """Acquiring AND parsing the pilot's bytes opens no SQLite tier at all.

    ``test_parser_resource_needs_only_provider_bytes`` guards the parse; the
    generation step ran inside a module fixture before any guard was armed.
    This measures one complete cold acquisition instead, so a generator that
    starts bootstrapping archive tiers to produce wire bytes is caught.
    """
    from tests.infra.pilot_resources import build_pilot_provider_packages
    from tests.infra.sqlite_work_counter import sqlite_work_counter

    with sqlite_work_counter() as counter:
        packages = build_pilot_provider_packages(tmp_path)
        from polylogue.sources import iter_source_sessions

        sessions = tuple(
            session
            for package in packages
            for source in package.admitted_sources()
            for session in iter_source_sessions(source)
        )

    assert sessions
    assert sum(counter.connections_by_database.values()) == 0, (
        f"parser-only pilot acquisition opened database connections: {dict(counter.connections_by_database)}"
    )
    written_bytes = sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())
    assert written_bytes > 0
    print(
        "pilot-parser-acquisition="
        f"{{'packages': {len(packages)}, 'sessions': {len(sessions)}, 'written_bytes': {written_bytes}}}"
    )


def test_query_resource_reuses_one_authenticated_immutable_artifact(
    pilot_query_archive: SeededArchiveQueryLease,
) -> None:
    """Read-only consumers share the artifact and cannot write through it."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    with ArchiveStore.open_existing(pilot_query_archive.root, read_only=True) as archive:
        sessions = archive.list_summaries(limit=20)
        tags = archive.list_user_tags()

    assert sessions
    assert tags == {}
    assert pilot_query_archive.root == pilot_query_archive.artifact.root
    assert pilot_query_archive.artifact.manifest.resources.total_bytes > 0


def test_mutation_resource_isolated_from_artifact_and_siblings(
    pilot_artifact: SeededArchiveArtifact,
    pilot_writable_archive: SeededArchiveClone,
    tmp_path: Path,
) -> None:
    """A committed mutation remains in its clone, not the immutable sibling."""
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.workload_artifacts import clone_seeded_archive

    session_id = next(fact.expected_session_id for fact in pilot_artifact.facts if fact.expected_session_id)
    sibling = clone_seeded_archive(pilot_artifact, tmp_path / "pilot-sibling-archive")
    with ArchiveStore(pilot_writable_archive.root) as archive:
        assert archive.add_user_tags((session_id,), ("pilot-isolated",)) == 1
        assert archive.list_user_tags() == {"pilot-isolated": 1}

    with ArchiveStore.open_existing(pilot_artifact.root, read_only=True) as archive:
        assert archive.list_user_tags() == {}
    try:
        with ArchiveStore.open_existing(sibling.root, read_only=True) as archive:
            assert archive.list_user_tags() == {}
    finally:
        sibling.close()


def test_transport_resource_uses_the_real_daemon_operation_route(
    pilot_daemon_operations: DaemonOperationStack,
) -> None:
    """The transport pilot reaches the production UDS operation endpoint."""
    response = pilot_daemon_operations.client.operation(
        "cli.query",
        {"params": {"limit": 20}},
        archive_root=str(pilot_daemon_operations.archive_root),
    )
    assert response is not None
    assert response["outcome"] == "completed"
    assert response["authority"]["writes"] == "daemon-owned"
    assert response["result"]["total"] == 2


@pytest.mark.parametrize(
    ("provider", "minimum_messages"),
    (("chatgpt", 2), ("codex", 2)),
)
def test_pilot_facts_keep_provider_and_multiplicity_independent(
    provider: str,
    minimum_messages: int,
    pilot_provider_packages: tuple[ProviderSourcePackage, ...],
    pilot_parsed_sessions: tuple[ParsedSession, ...],
) -> None:
    """Schema/provider variation does not change the authored truth bounds."""
    package = next(package for package in pilot_provider_packages if package.provider == provider)
    sessions = tuple(
        session
        for session in pilot_parsed_sessions
        if session.source_name.value == provider or session.source_name.value.endswith(provider)
    )
    assert sessions
    assert len(_provider_native_ids(package)) == 1
    assert all(len(session.messages) >= minimum_messages for session in sessions)


@pytest.mark.parametrize("case", pilot_scenarios(), ids=lambda case: case.name)
def test_compact_pilot_scenarios_keep_independent_facts_across_wire_shapes(case: PilotScenario) -> None:
    """Wire-shape variation changes representation, never the authored truth."""

    from tests.infra.pilot_scenarios import project_representation

    observed = tuple(project_representation(case, representation) for representation in case.representations)
    assert observed == (case.expected,) * len(case.representations)


def test_multiplicity_scenario_is_not_a_set_sum() -> None:
    """The independent expectation catches the classic duplicate-collapse bug."""

    from tests.infra.pilot_scenarios import pilot_scenarios, project_representation

    case = next(case for case in pilot_scenarios() if case.name == "multiplicity-sensitive-sum")
    collapsed = project_representation(case, {"amounts": sorted({2, 3})})
    assert collapsed.weighted_total != case.expected.weighted_total


def test_payload_tail_and_revision_order_are_independent_facts() -> None:
    """Tail bytes and revision selection remain observable beyond row counts."""

    from tests.infra.pilot_scenarios import pilot_scenarios, project_representation

    cases = {case.name: case for case in pilot_scenarios()}
    tail = cases["payload-tail-preservation"]
    revisions = cases["ordering-revision-relations"]

    assert project_representation(tail, tail.representations[0]) == tail.expected
    assert project_representation(revisions, revisions.representations[0]) == revisions.expected

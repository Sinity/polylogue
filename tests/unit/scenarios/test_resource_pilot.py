"""Small demand-driven resource pilot.

The expectations below are authored independently of the artifact manifest:
the integration recipe supplies provider/native identity, while the tests
assert parser, read, mutation-isolation, and transport behavior separately.
"""

from __future__ import annotations

import sqlite3

import pytest

from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.daemon_operations import DaemonOperationStack
from tests.infra.pilot_scenarios import PilotScenario, pilot_scenarios, project_representation
from tests.infra.source_builders import ProviderSourcePackage
from tests.infra.workload_artifacts import SeededArchiveArtifact, SeededArchiveClone, SeededArchiveQueryLease

pytest_plugins = ("tests.infra.corpus_fixtures", "tests.infra.pilot_resources")


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


def test_query_resource_reuses_one_authenticated_immutable_artifact(
    pilot_query_archive: SeededArchiveQueryLease,
) -> None:
    """Read-only consumers share the artifact and cannot write through it."""
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
) -> None:
    """A committed mutation remains in its clone, not the immutable sibling."""
    session_id = next(fact.expected_session_id for fact in pilot_artifact.facts if fact.expected_session_id)
    with ArchiveStore(pilot_writable_archive.root) as archive:
        assert archive.add_user_tags((session_id,), ("pilot-isolated",)) == 1
        assert archive.list_user_tags() == {"pilot-isolated": 1}

    with ArchiveStore.open_existing(pilot_artifact.root, read_only=True) as archive:
        assert archive.list_user_tags() == {}


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

    observed = tuple(project_representation(case, representation) for representation in case.representations)
    assert observed == (case.expected,) * len(case.representations)


def test_multiplicity_scenario_is_not_a_set_sum() -> None:
    """The independent expectation catches the classic duplicate-collapse bug."""

    case = next(case for case in pilot_scenarios() if case.name == "multiplicity-sensitive-sum")
    collapsed = project_representation(case, {"amounts": sorted({2, 3})})
    assert collapsed.weighted_total != case.expected.weighted_total

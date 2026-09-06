"""Antigravity source-level acquisition contracts.

The periodic daemon reconciler covers a working language-server export. These
tests exercise the shared source iterator when that supported export surface
is unavailable, which is where brain metadata used to bypass artifact
classification and become synthetic conversation sessions.
"""

from __future__ import annotations

import logging
import os
import socket
from pathlib import Path

import pytest

from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.sources.parsers import antigravity
from polylogue.sources.parsers.antigravity import AntigravityBinaryUnavailableError
from polylogue.sources.source_parsing import iter_antigravity_language_server_sessions
from polylogue.sources.source_walk import census_source_root


def _write_brain_sidecar(root: Path) -> Path:
    metadata_path = root / "brain" / "work-session" / "plan.md.metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.with_name("plan.md").write_text("# Plan\n\nInspect the archive.\n", encoding="utf-8")
    metadata_path.write_text(
        '{"artifactType":"ARTIFACT_TYPE_OTHER","summary":"Plan","updatedAt":"2026-08-04T08:00:00Z"}',
        encoding="utf-8",
    )
    return metadata_path


def test_unavailable_language_server_never_promotes_brain_sidecars_to_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing exporter leaves the conversation coverage gap visible.

    This enters the real Antigravity source iterator that batch import uses.
    Before the fix, its fallback parsed this sidecar into a degraded session,
    bypassing the same taxonomy rule that excludes the file from live ingest
    and schema inference.
    """
    root = tmp_path / "antigravity"
    (root / "conversations").mkdir(parents=True)
    _write_brain_sidecar(root)

    def unavailable_export(*_args: object, **_kwargs: object) -> object:
        raise AntigravityBinaryUnavailableError("test language server unavailable")

    monkeypatch.setattr(
        "polylogue.sources.source_parsing.antigravity.iter_language_server_export_results",
        unavailable_export,
    )
    caplog.set_level(logging.WARNING, logger="polylogue.sources.source_parsing")

    sessions = list(iter_antigravity_language_server_sessions(Source(name="antigravity", path=root)))

    assert sessions == []
    assert "antigravity_coverage_gap" in caplog.messages[-1]


def test_source_census_accounts_for_all_roles_and_unknown_items(tmp_path: Path) -> None:
    root = tmp_path / "antigravity"
    (root / "conversations").mkdir(parents=True)
    (root / "conversations" / "cascade.pb").write_bytes(b"opaque")
    (root / "brain" / "work").mkdir(parents=True)
    (root / "brain" / "work" / "plan.md").write_text("# plan", encoding="utf-8")
    (root / "brain" / "work" / "plan.md.metadata.json").write_text("{}", encoding="utf-8")
    (root / "settings" / "opaque.bin").parent.mkdir()
    (root / "settings" / "opaque.bin").write_bytes(b"unknown")

    source_census = antigravity.census_source(root)
    assert source_census.counts == {
        antigravity.AntigravitySourceRole.CONVERSATION_PROTOBUF: 1,
        antigravity.AntigravitySourceRole.BRAIN_DOCUMENT: 1,
        antigravity.AntigravitySourceRole.METADATA_SIDECAR: 1,
        antigravity.AntigravitySourceRole.UNKNOWN: 1,
    }
    assert source_census.unknown_count == 1
    assert source_census.inspection_counts == {
        antigravity.AntigravitySourceInspection.REGULAR: 4,
        antigravity.AntigravitySourceInspection.NON_REGULAR: 0,
        antigravity.AntigravitySourceInspection.UNREADABLE: 0,
    }
    assert source_census.unexplained_items == ()
    source_census.assert_conserved()

    root_census = census_source_root(root, provider=Provider.ANTIGRAVITY)
    assert root_census.candidate_count == 4
    assert root_census.disposition_counts == {"session": 1, "non_session": 2, "unsupported": 1}
    assert root_census.is_complete


def test_source_census_rejects_mutation_during_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "antigravity"
    root.mkdir()
    source = root / "settings.bin"
    source.write_bytes(b"before")
    original_digest = antigravity._file_digest

    def digest_then_mutate(path: Path) -> str:
        digest = original_digest(path)
        path.write_bytes(b"after")
        return digest

    monkeypatch.setattr(antigravity, "_file_digest", digest_then_mutate)
    with pytest.raises(antigravity.AntigravitySourceMutationError):
        antigravity.census_source(root)


def test_source_census_rejects_an_unclassified_item(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "antigravity"
    root.mkdir()
    (root / "settings.bin").write_bytes(b"unknown")

    monkeypatch.setattr(antigravity, "classify_source_path", lambda _path: None)

    with pytest.raises(ValueError, match="unexplained"):
        antigravity.census_source(root)


def test_source_census_counts_non_regular_and_unreadable_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "antigravity"
    root.mkdir()
    readable = root / "readable.bin"
    readable.write_bytes(b"readable")
    fifo = root / "pipe"
    os.mkfifo(fifo)
    dangling = root / "dangling"
    dangling.symlink_to(root / "missing")
    socket_path = root / "socket"
    server = socket.socket(socket.AF_UNIX)
    original_cwd = Path.cwd()
    os.chdir(root)
    try:
        server.bind("socket")
    finally:
        os.chdir(original_cwd)
    unreadable = root / "unreadable.bin"
    unreadable.write_bytes(b"unreadable")
    original_digest = antigravity._file_digest

    def fail_one(path: Path) -> str:
        if path == unreadable:
            raise PermissionError("synthetic unreadable source")
        return original_digest(path)

    monkeypatch.setattr(antigravity, "_file_digest", fail_one)
    try:
        census = antigravity.census_source(root)
    finally:
        server.close()
        socket_path.unlink()

    assert len(census.items) == 5
    assert census.inspection_counts == {
        antigravity.AntigravitySourceInspection.REGULAR: 1,
        antigravity.AntigravitySourceInspection.NON_REGULAR: 3,
        antigravity.AntigravitySourceInspection.UNREADABLE: 1,
    }
    assert census.unknown_count == 5
    assert {item.classification.reason for item in census.items if item.classification is not None} == {
        "non-regular Antigravity source item",
        "source item is unreadable: synthetic unreadable source",
        "unrecognized Antigravity source item",
    }
    assert census.unexplained_items == ()


def _write_brain_population(root: Path, cascade_id: str, *names: str) -> None:
    """Write one brain directory in the shape a real source root carries.

    A real ``brain/<cascade-id>/`` holds a document per artifact plus a
    ``<name>.metadata.json`` sidecar beside it. Every one of those files was
    once minted as a one-message session; none of them is conversation content.
    """
    brain = root / "brain" / cascade_id
    brain.mkdir(parents=True)
    for name in names:
        (brain / f"{name}.md").write_text(f"# {name}\n\nBody for {name}.\n", encoding="utf-8")
        (brain / f"{name}.md.metadata.json").write_text(
            '{"artifactType":"ARTIFACT_TYPE_OTHER","summary":"' + name + '"}',
            encoding="utf-8",
        )


@pytest.mark.asyncio
async def test_brain_population_types_as_non_session_artifact_not_as_a_session(
    tmp_path: Path, workspace_env: dict[str, Path]
) -> None:
    """Reacquiring a brain population owes the conservation equation a term.

    Every ``antigravity-session`` row in the pre-rebuild index was minted from
    one of these metadata files. On a fresh acquisition each one must land in
    ``raw_artifacts`` under its declared artifact kind, so the equation types it
    as ``non_session_artifact`` instead of leaving a session with no lineage.

    Anti-vacuity: a metadata payload admitted without its artifact
    classification lands in ``unexplained`` (blocking), and one that still mints
    a session lands in ``phantom_declared_non_session_lineage`` (blocking).
    """
    import sqlite3

    from polylogue.maintenance.source_conservation import audit_source_conservation
    from polylogue.pipeline.services.archive_ingest import parse_sources_archive

    root = tmp_path / "antigravity"
    (root / "conversations").mkdir(parents=True)
    _write_brain_population(root, "aaaaaaaa-0000-4000-8000-000000000001", "plan", "report")

    archive_root = workspace_env["archive_root"]
    result = await parse_sources_archive(archive_root, [Source(name="antigravity", path=root)], parse_workers=1)

    assert result.parse_failures == 0
    assert result.counts.get("sessions", 0) == 0

    conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    try:
        conn.execute("ATTACH DATABASE ? AS idx_tier", (f"file:{archive_root / 'index.db'}?mode=ro",))
        kinds = dict(
            conn.execute(
                "SELECT artifact_kind, COUNT(*) FROM raw_artifacts WHERE origin = ? GROUP BY 1",
                ("antigravity-session",),
            ).fetchall()
        )
        report = audit_source_conservation(conn, archive_root=archive_root)
    finally:
        conn.close()

    assert kinds == {"agent_sidecar_meta": 2, "metadata_document": 2}
    terms = {term.name: term for term in report.terms}
    assert terms["non_session_artifact"].count == 4
    assert terms["non_session_artifact"].breakdown == {
        "antigravity-session:agent_sidecar_meta": 2,
        "antigravity-session:metadata_document": 2,
    }
    assert report.session_total == 0
    assert [term.name for term in report.terms if term.blocking and term.count] == []


def test_real_conversation_files_are_claimed_by_the_declared_session_route(tmp_path: Path) -> None:
    """A conversation protobuf on disk resolves to the export route, not a guess.

    Anti-vacuity: a ``conversations/*.pb`` file no detector claims reports
    ``unsupported``/``UNKNOWN`` here, which is the coverage failure this pins --
    the origin's only conversation content is those protobufs, and nothing else
    in the tree may be promoted in their place.
    """
    from polylogue.sources.origin_specs import artifact_rule_for_path, recognize_source_class

    root = tmp_path / "antigravity"
    (root / "conversations").mkdir(parents=True)
    conversation = root / "conversations" / "aaaaaaaa-0000-4000-8000-000000000001.pb"
    conversation.write_bytes(b"\x08\x01opaque-trajectory")
    _write_brain_population(root, "aaaaaaaa-0000-4000-8000-000000000001", "plan")

    assert antigravity.conversation_pb_paths(root) == [conversation]

    recognition = recognize_source_class(Provider.ANTIGRAVITY, str(conversation), source_only=True)
    assert recognition is not None and recognition.source_class == "session"

    rule = artifact_rule_for_path(Provider.ANTIGRAVITY, str(conversation))
    assert rule is not None
    assert rule.parse_policy == "session"
    assert rule.parser_path == "polylogue/sources/parsers/antigravity.py:iter_language_server_exports"

    # Nothing else in the tree is a session candidate.
    for other in sorted(p for p in root.rglob("*") if p.is_file() and p != conversation):
        assert antigravity.classify_source_path(other).parse_as_session is False

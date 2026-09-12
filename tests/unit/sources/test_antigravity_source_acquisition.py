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
import sqlite3
from pathlib import Path

import pytest

import polylogue.sources.live.watcher as live_watcher
from polylogue import Polylogue
from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.sources.live import WatchSource
from polylogue.sources.live.batch import LiveBatchProcessor
from polylogue.sources.live.cursor import CursorStore
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


def test_symlinked_brain_document_is_censused_but_not_admitted(tmp_path: Path) -> None:
    root = tmp_path / "antigravity"
    target = tmp_path / "real.md"
    target.write_text("# linked", encoding="utf-8")
    linked = root / "brain" / "w" / "linked.md"
    linked.parent.mkdir(parents=True)
    linked.symlink_to(target)

    census = antigravity.census_source(root)

    assert census.inspection_counts[antigravity.AntigravitySourceInspection.NON_REGULAR] == 1
    assert census.unknown_count == 1
    assert linked in {item.path for item in census.items}
    assert linked not in antigravity._conversation_pb_paths(root)


def test_skip_directories_are_shared_by_census_and_production_walk(tmp_path: Path) -> None:
    root = tmp_path / "antigravity"
    skipped = root / "conversations" / "analysis"
    skipped.mkdir(parents=True)
    (skipped / "hidden.pb").write_bytes(b"opaque")

    census = antigravity.census_source(root)

    assert census.items == ()
    assert antigravity._conversation_pb_paths(root) == []


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


def _write_trajectory_store(path: Path) -> sqlite3.Connection:
    """Create a neutral trajectory whose committed turns remain in its WAL."""
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(
            """
            CREATE TABLE trajectory_meta (trajectory_id TEXT, cascade_id TEXT);
            CREATE TABLE steps (
                idx INTEGER, step_type TEXT, step_format TEXT, step_payload TEXT
            );
            CREATE TABLE conversation_summaries (cascade_id TEXT, title TEXT, last_modified_time TEXT);
            """
        )

    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    writer.execute("INSERT INTO trajectory_meta VALUES (?, ?)", ("trajectory-e2e", "cascade-e2e"))
    writer.execute(
        "INSERT INTO conversation_summaries VALUES (?, ?, ?)",
        ("cascade-e2e", "SQLite trajectory", "2026-09-12T08:00:00Z"),
    )
    writer.executemany(
        "INSERT INTO steps VALUES (?, ?, ?, ?)",
        [
            (10, "message", "v1", '{"role":"user","text":"inspect the workspace"}'),
            (
                20,
                "terminal_command",
                "v1",
                '{"tool_name":"shell","tool_id":"command-1","command":"git status --short"}',
            ),
            (
                30,
                "tool_result",
                "v1",
                '{"tool_name":"shell","tool_id":"command-1","output":" M README.md","status":"success"}',
            ),
            (
                40,
                "file_edit",
                "v1",
                '{"tool_name":"edit_file","tool_id":"edit-1","path":"README.md"}',
            ),
            (
                45,
                "tool_result",
                "v1",
                '{"tool_name":"edit_file","tool_id":"edit-1","output":"edited README.md",'
                '"path":"README.md","old_string":"old","new_string":"new"}',
            ),
            (50, "plan", "v1", '{"plan":["inspect","edit"]}'),
        ],
    )
    writer.commit()
    return writer


@pytest.mark.asyncio
async def test_trajectory_sqlite_wal_reaches_the_daemon_owned_public_read_route(
    workspace_env: dict[str, Path],
) -> None:
    """The ordinary live batch route snapshots WAL content before parsing it.

    Anti-vacuity: bypassing ``snapshot_sqlite_to_blob`` loses the committed
    rows below because they remain in the WAL while the writer stays open.
    """
    root = workspace_env["data_root"] / "antigravity"
    root.mkdir(parents=True)
    source_path = root / "unpredictable-name.sqlite"
    writer = _write_trajectory_store(source_path)
    archive = Polylogue(archive_root=workspace_env["archive_root"], db_path=workspace_env["data_root"] / "cursor.db")
    processor = LiveBatchProcessor(
        archive,
        (WatchSource(name="antigravity", root=root, suffixes=(".sqlite", ".db")),),
        cursor=CursorStore(workspace_env["data_root"] / "cursor.db"),
        parser_fingerprint=live_watcher._PARSER_FINGERPRINT,
    )
    try:
        metrics = await processor.ingest_files([source_path], emit_event=False)
        assert metrics.failed_file_count == 0
        assert metrics.ingested_session_count == 1

        replay = await processor.ingest_files([source_path], emit_event=False)
        assert replay.failed_file_count == 0
        assert replay.ingested_session_count == 1

        session = await archive.get_session("antigravity-session:trajectory-e2e")
        assert session is not None
        assert session.title == "SQLite trajectory"
        assert [message.text for message in session.messages] == [
            "inspect the workspace",
            None,
            " M README.md",
            None,
            "edited README.md",
            '["inspect", "edit"]',
        ]
        terminal_message = session.messages[1]
        result_message = session.messages[2]
        assert not isinstance(terminal_message, list)
        assert not isinstance(result_message, list)
        terminal = terminal_message.blocks[0]
        result = result_message.blocks[0]
        assert terminal["tool_name"] == "shell"
        assert terminal["tool_input"] == {"command": "git status --short"}
        assert result["tool_id"] == "command-1"
        assert result["text"] == " M README.md"
        assert result["tool_outcome"] == "ok"

        with sqlite3.connect(workspace_env["archive_root"] / "index.db") as index:
            session_count = index.execute(
                "SELECT COUNT(*) FROM sessions WHERE session_id = ?",
                ("antigravity-session:trajectory-e2e",),
            ).fetchone()
            actions = index.execute(
                "SELECT tool_name, tool_input, output_text, result_state FROM actions WHERE session_id = ?",
                ("antigravity-session:trajectory-e2e",),
            ).fetchall()
            edits = index.execute(
                "SELECT file_path, old_string, new_string FROM file_edits WHERE session_id = ?",
                ("antigravity-session:trajectory-e2e",),
            ).fetchall()
        assert actions == [
            ("shell", '{"command":"git status --short"}', " M README.md", "outcome_success"),
            ("edit_file", '{"path":"README.md"}', "edited README.md", "outcome_unknown"),
        ]
        assert session_count == (1,)
        assert edits == [("README.md", "old", "new")]
        assert source_path.with_name(f"{source_path.name}-wal").exists()
    finally:
        writer.close()
        await archive.close()

"""Hermes ``state.db`` tool-outcome extraction contracts.

Hermes stores tool results as a JSON envelope in ``messages.content``
(``{"output": ...}`` plus ``exit_code`` / ``success`` / ``error`` depending on
tool family) rather than dedicated outcome columns. These tests pin the
mapping from that envelope onto ``ParsedContentBlock.is_error`` /
``exit_code``, verified against real shapes observed in a live Hermes
``state.db`` (see ``polylogue-uwlu``).
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

import polylogue.sources.parsers.hermes_state as hermes_state
import polylogue.sources.sqlite_export as sqlite_export
from polylogue.core.enums import BlockType, TitleSource
from polylogue.core.json import JSONDocument
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage
from polylogue.sources.parsers.hermes_state import parse_state_db, parse_state_db_payload
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import search_archive_blocks, write_parsed_session_to_archive
from tests.infra.logical_source_probe import open_reconstruction_handles, record_logical_source_connections


def _write_state_db(path: Path, *, tool_contents: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (16);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                model TEXT,
                model_config TEXT,
                parent_session_id TEXT,
                started_at REAL,
                ended_at REAL,
                title TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_name TEXT,
                tool_calls TEXT,
                timestamp REAL NOT NULL,
                observed INTEGER DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                compacted INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO sessions (id, source, model, model_config, parent_session_id, started_at, ended_at, title)
            VALUES ('s1', 'hermes', 'test-model', '{}', NULL, 1775000000.0, 1775000010.0, 'Outcome fixture');
            """
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, 'user', 'go', ?)",
            ("s1", 1775000001.0),
        )
        for index, content in enumerate(tool_contents):
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, tool_call_id, tool_name, timestamp)
                VALUES ('s1', 'tool', ?, ?, 'shell', ?)
                """,
                (content, f"call-{index}", 1775000002.0 + index),
            )
        conn.commit()


def _tool_result_blocks(path: Path, *, tool_contents: list[str]) -> list[ParsedContentBlock]:
    _write_state_db(path, tool_contents=tool_contents)
    sessions = parse_state_db(path)
    assert len(sessions) == 1
    return [
        block for message in sessions[0].messages for block in message.blocks if block.type is BlockType.TOOL_RESULT
    ]


def test_exit_code_zero_is_not_an_error(tmp_path: Path) -> None:
    blocks = _tool_result_blocks(tmp_path / "state.db", tool_contents=[json.dumps({"output": "ok", "exit_code": 0})])
    assert blocks[0].is_error is False
    assert blocks[0].exit_code == 0


def test_state_db_explicit_title_is_provider_provenance(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    _write_state_db(path, tool_contents=[json.dumps({"output": "ok", "exit_code": 0})])

    sessions = parse_state_db(path)

    assert sessions[0].title == "Outcome fixture"
    assert sessions[0].title_source is TitleSource.ORIGIN


def test_nonzero_exit_code_is_an_error(tmp_path: Path) -> None:
    blocks = _tool_result_blocks(
        tmp_path / "state.db", tool_contents=[json.dumps({"output": "boom", "exit_code": 127})]
    )
    assert blocks[0].is_error is True
    assert blocks[0].exit_code == 127


def test_success_true_is_not_an_error(tmp_path: Path) -> None:
    blocks = _tool_result_blocks(tmp_path / "state.db", tool_contents=[json.dumps({"output": "done", "success": True})])
    assert blocks[0].is_error is False
    assert blocks[0].exit_code is None


def test_success_false_with_error_message_is_an_error(tmp_path: Path) -> None:
    blocks = _tool_result_blocks(
        tmp_path / "state.db",
        tool_contents=[json.dumps({"output": "", "success": False, "error": "not found"})],
    )
    assert blocks[0].is_error is True
    assert blocks[0].exit_code is None


def test_bare_error_message_with_no_exit_code_or_success_is_an_error(tmp_path: Path) -> None:
    blocks = _tool_result_blocks(
        tmp_path / "state.db", tool_contents=[json.dumps({"output": None, "error": "invalid sort: recent"})]
    )
    assert blocks[0].is_error is True
    assert blocks[0].exit_code is None


def test_plain_output_with_no_outcome_signal_is_unknown_not_guessed(tmp_path: Path) -> None:
    blocks = _tool_result_blocks(tmp_path / "state.db", tool_contents=[json.dumps({"output": "just text"})])
    assert blocks[0].is_error is None
    assert blocks[0].exit_code is None


def test_exit_code_and_error_together_prefer_error_but_keep_exit_code(tmp_path: Path) -> None:
    blocks = _tool_result_blocks(
        tmp_path / "state.db",
        tool_contents=[json.dumps({"output": "boom", "exit_code": 1, "error": "denied"})],
    )
    assert blocks[0].is_error is True
    assert blocks[0].exit_code == 1


def _write_reasoning_state_db(path: Path) -> None:
    """A minimal state.db with a reasoning-bearing assistant message.

    ``reasoning_details``/``codex_reasoning_items``/``codex_message_items``
    are Hermes's captured Codex-native reasoning-trace evidence
    (bd polylogue-9x22): the ``blocks`` table has no metadata column, so this
    would be silently dropped at write time if it only reached
    ``ParsedContentBlock.metadata``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (16);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                model TEXT,
                model_config TEXT,
                parent_session_id TEXT,
                started_at REAL,
                ended_at REAL,
                title TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_name TEXT,
                tool_calls TEXT,
                reasoning_content TEXT,
                reasoning_details TEXT,
                codex_reasoning_items TEXT,
                codex_message_items TEXT,
                timestamp REAL NOT NULL,
                observed INTEGER DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                compacted INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO sessions (id, source, model, model_config, parent_session_id, started_at, ended_at, title)
            VALUES ('s1', 'hermes', 'test-model', '{}', NULL, 1775000000.0, 1775000010.0, 'Reasoning fixture');
            """
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, 'user', 'solve it', ?)",
            ("s1", 1775000001.0),
        )
        conn.execute(
            """
            INSERT INTO messages (
                session_id, role, content, reasoning_content, reasoning_details,
                codex_reasoning_items, codex_message_items, timestamp
            )
            VALUES ('s1', 'assistant', 'done', 'thinking it through', ?, ?, ?, 1775000002.0)
            """,
            (
                json.dumps([{"type": "text", "text": "step one"}]),
                json.dumps(["reasoning-item-1"]),
                json.dumps(["message-item-1"]),
            ),
        )
        conn.commit()


def test_reasoning_evidence_routes_to_session_events_not_only_block_metadata(tmp_path: Path) -> None:
    """Production route: sqlite state.db row -> ``parse_state_db`` -> ``session_events``.

    This fails if ``_reasoning_evidence_events`` (or its wiring into
    ``_parse_session_row``) is removed, since the reasoning payload would
    then only live on ``ParsedContentBlock.metadata`` -- a field the archive
    write path never persists (only ``language`` is read back out of it).
    """
    path = tmp_path / "state.db"
    _write_reasoning_state_db(path)
    sessions = parse_state_db(path)
    assert len(sessions) == 1
    session = sessions[0]

    thinking_blocks = [
        block for message in session.messages for block in message.blocks if block.type is BlockType.THINKING
    ]
    assert len(thinking_blocks) == 1
    assert thinking_blocks[0].text == "thinking it through"

    reasoning_events = [event for event in session.session_events if event.event_type == "hermes_reasoning_evidence"]
    assert len(reasoning_events) == 1
    event = reasoning_events[0]
    assert event.payload == {
        "reasoning_details": [{"type": "text", "text": "step one"}],
        "codex_reasoning_items": ["reasoning-item-1"],
        "codex_message_items": ["message-item-1"],
    }
    assistant_message = next(message for message in session.messages if message.role.value == "assistant")
    assert event.source_message_provider_id == assistant_message.provider_message_id


# Prose that exists only inside ``codex_message_items``: no other fixture row
# carries it, so an FTS hit on it can come from nowhere but the projection.
_CODEX_ONLY_PROSE = "The wrapper stayed untouched while the stale config migrated."
# Prose the ``content`` column already carries, which the item then repeats.
_DUPLICATED_PROSE = "Verification passed on both affected suites."


def _codex_message_item(prose: str) -> list[dict[str, object]]:
    return [
        {
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": prose}],
            "id": "msg_0887",
            "phase": "commentary",
        }
    ]


def _write_codex_message_items_state_db(path: Path) -> None:
    """A state.db with the two shapes ``codex_message_items`` takes.

    Row 2 is the sharp one: a Codex-compatible backend left ``content`` empty
    and put the assistant's prose only in the structured response item, beside
    reasoning and a tool call. Row 3 is the ordinary case, where the item
    repeats prose ``content`` already carries.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version(version) VALUES (16);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                model TEXT,
                model_config TEXT,
                parent_session_id TEXT,
                started_at REAL,
                ended_at REAL,
                title TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_name TEXT,
                tool_calls TEXT,
                reasoning_content TEXT,
                reasoning_details TEXT,
                codex_reasoning_items TEXT,
                codex_message_items TEXT,
                timestamp REAL NOT NULL,
                observed INTEGER DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                compacted INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO sessions (id, source, model, model_config, parent_session_id, started_at, ended_at, title)
            VALUES ('s1', 'hermes', 'test-model', '{}', NULL, 1775000000.0, 1775000010.0, 'Codex items fixture');
            INSERT INTO messages (session_id, role, content, timestamp)
            VALUES ('s1', 'user', 'migrate the config', 1775000001.0);
            """
        )
        conn.execute(
            """
            INSERT INTO messages (
                session_id, role, content, reasoning_content, tool_calls,
                codex_reasoning_items, codex_message_items, timestamp
            )
            VALUES ('s1', 'assistant', '', 'weighing the wrapper', ?, ?, ?, 1775000002.0)
            """,
            (
                json.dumps([{"id": "call-1", "function": {"name": "shell", "arguments": "{}"}}]),
                json.dumps(
                    [
                        {
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": "Inspect the wrapper."}],
                            "content": [{"type": "reasoning_text", "text": "The stale value is isolated."}],
                        }
                    ]
                ),
                json.dumps(_codex_message_item(_CODEX_ONLY_PROSE)),
            ),
        )
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, codex_message_items, timestamp)
            VALUES ('s1', 'assistant', ?, ?, 1775000003.0)
            """,
            (_DUPLICATED_PROSE, json.dumps(_codex_message_item(_DUPLICATED_PROSE))),
        )
        conn.commit()


def _text_block_texts(message: ParsedMessage) -> list[str | None]:
    return [block.text for block in message.blocks if block.type is BlockType.TEXT]


def test_codex_message_items_prose_reaches_a_block_when_content_is_empty(tmp_path: Path) -> None:
    """``codex_message_items`` is assistant output, not reasoning evidence.

    Goes red if the projection in ``_parse_message_row`` is removed: the row's
    only prose would then live solely in the ``hermes_reasoning_evidence``
    event payload, which no block and therefore no FTS surface reads.
    """
    path = tmp_path / "state.db"
    _write_codex_message_items_state_db(path)
    [session] = parse_state_db(path)

    content_less_message = session.messages[1]
    assert _text_block_texts(content_less_message) == [_CODEX_ONLY_PROSE]
    assert content_less_message.text == _CODEX_ONLY_PROSE

    evidence = [event for event in session.session_events if event.event_type == "hermes_reasoning_evidence"]
    assert len(evidence) == 2


def test_codex_message_items_do_not_duplicate_prose_content_already_carries(tmp_path: Path) -> None:
    """Goes red if the projection stops skipping segments ``content`` covers,
    which would double the turn in both the display text and the FTS index."""
    path = tmp_path / "state.db"
    _write_codex_message_items_state_db(path)
    [session] = parse_state_db(path)

    assert _text_block_texts(session.messages[2]) == [_DUPLICATED_PROSE]


def test_codex_reasoning_items_project_summary_and_content_to_thinking(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    _write_codex_message_items_state_db(path)
    [session] = parse_state_db(path)

    thinking = [block.text for block in session.messages[1].blocks if block.type is BlockType.THINKING]
    assert thinking == [
        "weighing the wrapper",
        "Inspect the wrapper.",
        "The stale value is isolated.",
    ]


def test_codex_message_items_prose_is_findable_by_search(tmp_path: Path) -> None:
    """Production route: state.db -> parse -> archive write -> message FTS.

    Goes red without the projection: no other fixture row carries
    ``_CODEX_ONLY_PROSE``, so the match count drops to zero.
    """
    path = tmp_path / "state.db"
    _write_codex_message_items_state_db(path)
    [session] = parse_state_db(path)

    db = tmp_path / "index.db"
    initialize_archive_database(db, ArchiveTier.INDEX)
    conn = sqlite3.connect(db)
    try:
        write_parsed_session_to_archive(conn, session)
        conn.commit()
        matched = search_archive_blocks(conn, "wrapper untouched")
        block_texts = [
            row[0]
            for row in conn.execute(
                f"SELECT text FROM blocks WHERE block_id IN ({','.join('?' * len(matched))})",
                matched,
            )
        ]
    finally:
        conn.close()

    assert block_texts == [_CODEX_ONLY_PROSE]


# MARKER PATH CONFINEMENT


def test_state_db_marker_refuses_a_database_it_does_not_declare(tmp_path: Path) -> None:
    """An imported JSON document cannot steer this parser at another database.

    The Hermes state.db marker is an ordinary JSON object recognised by shape,
    so any imported document can carry it and name a local path. Anti-vacuity:
    removing the ``require_declared_export`` call from
    ``parse_state_db_payload`` makes this call succeed and return the victim
    database's session -- the exact confused-deputy read -- so the ``ValueError``
    asserted here disappears and the test goes red.
    """
    victim = tmp_path / "victim" / "state.db"
    _write_state_db(victim, tool_contents=[json.dumps({"output": "secret", "exit_code": 0})])
    # Sanity: the victim database really is parseable, so the refusal below is
    # the guard firing and not an unrelated failure to read the file.
    assert parse_state_db(victim)

    payload: JSONDocument = {
        "polylogue_artifact": "hermes_state_db",
        "state_db_path": str(victim),
    }

    with pytest.raises(ValueError, match="declared logical export"):
        parse_state_db_payload(payload, "fallback", source_path=str(tmp_path / "innocent_export.json"))


def test_state_db_marker_without_a_source_path_is_refused(tmp_path: Path) -> None:
    """A marker with no owning source cannot be trusted to name a database.

    Anti-vacuity: defaulting the guard to "allow when ``source_path`` is None"
    would leave every caller that does not thread the source path open to the
    same steered read. This pins the fail-closed default.
    """
    victim = tmp_path / "victim" / "state.db"
    _write_state_db(victim, tool_contents=[json.dumps({"output": "secret", "exit_code": 0})])
    payload: JSONDocument = {"polylogue_artifact": "hermes_state_db", "state_db_path": str(victim)}

    with pytest.raises(ValueError, match="declared logical export"):
        parse_state_db_payload(payload, "fallback")


def test_state_db_and_atof_from_one_install_agree_on_the_profile_key(tmp_path: Path) -> None:
    """polylogue-q5j3o: Hermes writes ``state.db`` at its install root and
    NeMo Relay ATOF events at ``<root>/observability/nemo-relay/atof/``. Both
    families must hash the SAME install root, or one logical session gets two
    profile keys and lands as two archive sessions.

    Anti-vacuity: reverting either family's profile root to the artifact's
    immediate parent (``Path(source_path).parent``) makes the ATOF key hash
    ``.../atof`` instead of ``<root>`` and the equality assertion goes red.
    """
    from polylogue.core.enums import Provider
    from polylogue.sources.dispatch import parse_stream_payload
    from polylogue.sources.parsers import hermes_spans
    from polylogue.sources.parsers.hermes_identity import profile_key, split_qualified_session_id

    root = tmp_path / ".hermes"
    state_path = root / "state.db"
    _write_state_db(state_path, tool_contents=[json.dumps({"output": "ok", "exit_code": 0})])

    state_sessions = parse_state_db(state_path)
    assert len(state_sessions) == 1
    _raw_id, state_key = split_qualified_session_id(state_sessions[0].provider_session_id)

    atof_path = root / "observability" / "nemo-relay" / "atof" / "events.jsonl"
    atof_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[JSONDocument] = [
        {
            "atof_version": "0.1",
            "kind": "scope",
            "category": "agent",
            "scope_category": "start",
            "uuid": "scope-parent",
            "timestamp": "2026-04-01T00:00:00Z",
            "name": "hermes-session",
            "metadata": {"session_id": "s1", "platform": "hermes"},
        }
    ]
    assert hermes_spans.looks_like_atof_payload(records[0])
    atof_path.write_text("\n".join(json.dumps(record) for record in records))

    atof_sessions = parse_stream_payload(
        Provider.HERMES,
        iter(records),
        "fallback-id",
        source_path=str(atof_path),
    )
    assert atof_sessions
    _atof_raw, atof_key = split_qualified_session_id(atof_sessions[0].provider_session_id)

    assert state_key == profile_key(root)
    assert atof_key == state_key


# THE PRIVATE READER: read plan, output parity, and connection lifetime


def _state_db_export(tmp_path: Path) -> tuple[Path, Path]:
    """Return ``(live state.db, its retained logical export)`` in one profile root.

    Both files share a directory, so they resolve to the same Hermes profile
    root and any identity difference between the two reads is a real parser
    difference rather than a path artefact.
    """
    root = tmp_path / ".hermes"
    live = root / "state.db"
    _write_state_db(live, tool_contents=[json.dumps({"output": "ok", "exit_code": index}) for index in range(3)])
    export = root / "state.db.export"
    export.write_bytes(sqlite_export.logical_export_bytes(live))
    return live, export


def test_the_per_session_message_read_is_answered_by_the_reconstruction_index(tmp_path: Path) -> None:
    """The parser's one per-session query must not scan and sort the whole table.

    A retained export reconstructs with no index at all, so this parser --
    which issues ``WHERE session_id = ? ORDER BY id`` once per session -- used
    to re-scan and re-sort every reconstructed message for every session.

    Anti-vacuity: drop the ``read_indexes=`` argument at
    ``hermes_state._connect_readonly`` (or delete the ``CREATE INDEX`` loop in
    ``materialize_export``) and the plan returns to the ``SCAN messages`` plus
    ``USE TEMP B-TREE FOR ORDER BY`` this asserts against.
    """
    _live, export = _state_db_export(tmp_path)

    with closing(hermes_state._connect_readonly(export)) as conn:
        plan = [
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM messages WHERE session_id = ? ORDER BY id",
                ("s1",),
            )
        ]
        indexes = [str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")]

    assert indexes == ["polylogue_read_messages_session_id_id"], indexes
    assert any("SEARCH" in step and indexes[0] in step for step in plan), plan
    assert not any("TEMP B-TREE" in step.upper() for step in plan), plan


def test_the_read_index_changes_no_parsed_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The hint is a read plan, so normalized output must be identical without it.

    Anti-vacuity: make the hint change what is read -- for example build it as
    a UNIQUE index, or let it reorder the result -- and the serialized
    comparison below separates immediately. Parity alone would be vacuous, so
    the sibling test asserts the index is really there.
    """
    live, export = _state_db_export(tmp_path)
    real_open = sqlite_export.open_logical_source

    def _open_without_hint(path: Path, **kwargs: object) -> sqlite3.Connection:
        kwargs.pop("read_indexes", None)
        return real_open(path, **kwargs)  # type: ignore[arg-type]

    hinted = [session.model_dump_json() for session in parse_state_db(export)]
    from_live = [session.model_dump_json() for session in parse_state_db(live)]
    monkeypatch.setattr(hermes_state, "open_logical_source", _open_without_hint)
    unhinted = [session.model_dump_json() for session in parse_state_db(export)]

    assert hinted, "sanity: the fixture really parses"
    assert hinted == unhinted, "the read index changed the parsed output"
    # Replay parity: the retained export is the material for the live member,
    # so parsing either must produce the same normalized sessions.
    assert hinted == from_live, "the retained export and its live source parsed differently"


def test_parse_state_db_closes_its_private_reader_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A leaked connection is invisible to any assertion about rows.

    ``sqlite3``'s own context manager commits or rolls back and never closes,
    so ``with _connect_readonly(...)`` returned with the connection open and
    the already-unlinked reconstruction still backed by that handle.

    Anti-vacuity: revert ``closing(_connect_readonly(...))`` in
    ``parse_state_db`` to a bare ``with _connect_readonly(...)`` and
    ``probe.closed`` is ``False`` while every parsed session stays correct.
    """
    _live, export = _state_db_export(tmp_path)

    with record_logical_source_connections(monkeypatch, hermes_state) as opened:
        sessions = parse_state_db(export)
        assert sessions, "sanity: the parse really ran"
        assert [probe.closed for probe in opened] == [True]


def test_parse_state_db_closes_its_private_reader_when_it_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal path owns the connection too.

    ``parse_state_db`` raises before its first row read when the file is not
    a Hermes state.db, and that early return must still release the
    reconstruction. Anti-vacuity: revert to a bare ``with`` and this probe
    reports ``closed is False`` while the ``ValueError`` still raises.
    """
    stranger = tmp_path / ".hermes" / "other.db"
    stranger.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(stranger) as conn:
        conn.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
    export = tmp_path / ".hermes" / "other.db.export"
    export.write_bytes(sqlite_export.logical_export_bytes(stranger))

    with record_logical_source_connections(monkeypatch, hermes_state) as opened:
        with pytest.raises(ValueError, match="not a Hermes state.db file"):
            parse_state_db(export)
        assert [probe.closed for probe in opened] == [True]


def test_parse_state_db_leaves_no_handle_on_the_unlinked_reconstruction(tmp_path: Path) -> None:
    """The production route, with nothing patched, strands no inode.

    ``open_logical_source`` unlinks the reconstruction while it is open, so an
    unclosed connection holds a deleted file's inode -- and the only place
    that is visible is this process's own descriptor table.

    Anti-vacuity: revert ``closing(_connect_readonly(...))`` in
    ``parse_state_db`` to a bare ``with`` and the descriptor count rises by
    one per parse and never falls.
    """
    _live, export = _state_db_export(tmp_path)

    before = open_reconstruction_handles()
    sessions = parse_state_db(export)
    after = open_reconstruction_handles()

    assert sessions, "sanity: the parse really ran"
    assert after == before, f"a reconstruction handle survived the parse: {before} -> {after}"

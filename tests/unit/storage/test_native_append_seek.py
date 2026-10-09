"""Native append admission must seek an identity without changing row membership."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.sources.parsers.base import ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers import write
from tests.infra.index_writer import fixture_index_connection, write_fixture_index_session


def _previous_probe(conn: sqlite3.Connection, session_id: str, native_id: object) -> bool:
    return (
        isinstance(native_id, str)
        and bool(native_id)
        and conn.execute(
            "SELECT 1 FROM messages WHERE session_id = ? AND native_id = ? LIMIT 1",
            (session_id, native_id),
        ).fetchone()
        is not None
    )


def _message(native: str, text: str) -> ParsedMessage:
    return ParsedMessage(provider_message_id=native, role=Role.USER, text=text)


def _rows(conn: sqlite3.Connection) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    return (
        [tuple(row) for row in conn.execute("SELECT * FROM messages ORDER BY session_id,position,variant_index")],
        [tuple(row) for row in conn.execute("SELECT * FROM blocks ORDER BY message_id,position")],
    )


def test_append_seek_preserves_production_rows_and_duplicate_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare both predicates through prepared publication, including NULL fallback identities."""
    native_ids = (" padded ", "native:n:colon", "żółć", "native\x00suffix")
    initial = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="opaque-native-append",
        messages=[
            *(_message(native, f"body {index}") for index, native in enumerate(native_ids)),
            _message("", "empty identity"),
            _message("   ", "blank identity"),
        ],
    )
    extension = initial.model_copy(
        update={
            "messages": [
                _message(native_ids[0], "duplicate body must stay unchanged"),
                _message("fresh:é", "new native body"),
                _message("", "new fallback body"),
            ]
        }
    )
    duplicate = initial.model_copy(update={"messages": [_message("fresh:é", "ignored duplicate body")]})
    current_probe = write._stored_native_id_exists
    snapshots = []
    outcomes = []
    for label, probe in (("previous", _previous_probe), ("seek", current_probe)):
        with monkeypatch.context() as patched:
            patched.setattr(write, "_stored_native_id_exists", probe)
            with fixture_index_connection(tmp_path / label / "index.db") as conn:
                session_id = write_fixture_index_session(conn, initial)
                natives = [
                    row[0]
                    for row in conn.execute(
                        "SELECT native_id FROM messages WHERE session_id=? ORDER BY position", (session_id,)
                    )
                ]
                assert natives == ["padded", *native_ids[1:], None, None]
                for native in (None, "", 0, " padded ", "padded", *native_ids[1:], "absent"):
                    assert probe(conn, session_id, native) == _previous_probe(conn, session_id, native)
                for row in conn.execute(
                    "SELECT session_id,native_id,message_id FROM messages WHERE native_id IS NOT NULL"
                ):
                    assert row[2] == f"{row[0]}:n:{row[1]}"
                write_fixture_index_session(conn, extension, merge_append=True)
                before_duplicate = _rows(conn)
                result: list[write.ArchiveWriteOutcome] = []
                write_fixture_index_session(conn, duplicate, merge_append=True, write_outcome=result)
                assert _rows(conn) == before_duplicate
                assert result[-1].wrote is False
                snapshots.append(_rows(conn))
                outcomes.append(result)
    assert snapshots[0] == snapshots[1]
    assert outcomes[0] == outcomes[1]


def _steps(operation: Callable[[], object], conn: sqlite3.Connection) -> int:
    steps = 0

    def progress() -> int:
        nonlocal steps
        steps += 1
        return 0

    conn.set_progress_handler(progress, 1)
    try:
        operation()
    finally:
        conn.set_progress_handler(None, 0)
    return steps


def test_native_probe_work_does_not_follow_session_length(tmp_path: Path) -> None:
    """Restoring the old session-range scan grows VM work eightfold in this control."""
    current_steps = []
    previous_steps = []
    for size in (64, 512):
        with fixture_index_connection(tmp_path / str(size) / "index.db") as conn:
            session = ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id="native-probe-cost",
                messages=[_message(f"native-{index}", f"body {index}") for index in range(size)],
            )
            session_id = write_fixture_index_session(conn, session)
            current_steps.append(_steps(partial(write._stored_native_id_exists, conn, session_id, "absent"), conn))
            previous_steps.append(_steps(partial(_previous_probe, conn, session_id, "absent"), conn))
            statements: list[str] = []
            conn.set_trace_callback(statements.append)
            try:
                assert write._stored_native_id_exists(conn, session_id, "absent") is False
            finally:
                conn.set_trace_callback(None)
            plan = conn.execute("EXPLAIN QUERY PLAN " + statements[-1]).fetchall()
            assert any("message_id=?" in str(row[3]) for row in plan), plan
    assert previous_steps[1] > previous_steps[0] * 4, previous_steps
    assert current_steps[1] <= current_steps[0] * 2, current_steps
    assert current_steps[1] * 10 < previous_steps[1], (current_steps, previous_steps)

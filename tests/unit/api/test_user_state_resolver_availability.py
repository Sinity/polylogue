"""An unreadable index tier refuses a target check; it never denies the target.

polylogue-p707n's class at the user-state write boundary: "the tier could not
be read" and "the target is not materialized" are different answers, and only
the second one is a fact about the target.

Anti-vacuity: restore ``except sqlite3.Error: return False`` in
``_row_exists``/``_index_db_path`` and the corrupt-tier case reports the target
as not materialized, matching the genuinely-empty case exactly.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from polylogue.api.user_state_resolver import (
    _resolve_attachment_in_connections,
    bind_attachment_source_guard,
    resolve_insight_target,
)
from polylogue.core.user_state_targets import TARGET_SESSION


def _materialized_index(root: Path) -> Path:
    index_db = root / "index.db"
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions(session_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE session_profiles(session_id TEXT PRIMARY KEY)")
    return index_db


def _resolve(root: Path) -> None:
    asyncio.run(
        resolve_insight_target(
            root,
            target_type=TARGET_SESSION,
            target_id="origin:native",
            session_id="origin:native",
        )
    )


def test_absent_profile_is_reported_as_not_materialized(tmp_path: Path) -> None:
    _materialized_index(tmp_path)

    with pytest.raises(ValueError, match="is not materialized"):
        _resolve(tmp_path)


def test_unreadable_index_refuses_instead_of_denying_the_target(tmp_path: Path) -> None:
    _materialized_index(tmp_path)
    (tmp_path / "index.db").write_bytes(b"this is not a sqlite database")

    with pytest.raises(ValueError, match="could not be checked") as excinfo:
        _resolve(tmp_path)

    assert "is not materialized" not in str(excinfo.value)


def test_missing_index_is_absence_not_unavailability(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is not materialized"):
        _resolve(tmp_path)


def test_complete_existence_probe_runs_on_original_bounded_creator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from polylogue.api import user_state_resolver
    from polylogue.core.compute import current_cancellation
    from polylogue.core.evidence import Evidence

    _materialized_index(tmp_path)
    creator = threading.get_ident()
    actual_probe = user_state_resolver._index_db_path
    workers: list[int] = []

    def probe(root: Path) -> Evidence[Path]:
        workers.append(threading.get_ident())
        assert current_cancellation() is not None
        return actual_probe(root)

    monkeypatch.setattr(user_state_resolver, "_index_db_path", probe)
    with pytest.raises(ValueError, match="not materialized"):
        _resolve(tmp_path)
    assert len(workers) == 1
    assert workers[0] != creator


def test_positional_block_input_returns_stable_block_id_for_persistence(tmp_path: Path) -> None:
    index_db = _materialized_index(tmp_path)
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE blocks(session_id TEXT, message_id TEXT, position INTEGER, block_id TEXT)")
        conn.execute(
            "INSERT INTO blocks VALUES (?, ?, ?, ?)",
            (
                "origin:session",
                "origin:session:n:message",
                2,
                "origin:session:n:message:b:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:0",
            ),
        )

    resolved = asyncio.run(
        resolve_insight_target(
            tmp_path,
            target_type="block",
            target_id="origin:session:n:message:2",
            session_id="origin:session",
        )
    )

    assert (
        resolved["target_id"]
        == "origin:session:n:message:b:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:0"
    )
    assert resolved["message_id"] == "origin:session:n:message"


def test_stable_block_target_is_resolved_by_opaque_id(tmp_path: Path) -> None:
    index_db = _materialized_index(tmp_path)
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE blocks(session_id TEXT, message_id TEXT, position INTEGER, block_id TEXT)")
        conn.execute(
            "INSERT INTO blocks VALUES (?, ?, ?, ?)",
            (
                "origin:session",
                "origin:session:n:message",
                2,
                "origin:session:n:message:b:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:0",
            ),
        )

    resolved = asyncio.run(
        resolve_insight_target(
            tmp_path,
            target_type="block",
            target_id="origin:session:n:message:b:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:0",
            session_id="origin:session",
        )
    )

    assert (
        resolved["target_id"]
        == "origin:session:n:message:b:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:0"
    )
    assert resolved["message_id"] == "origin:session:n:message"


def test_attachment_target_uses_stable_reference_and_source_supplier() -> None:
    index = sqlite3.connect(":memory:")
    source = sqlite3.connect(":memory:")
    try:
        index.executescript(
            "CREATE TABLE messages(message_id TEXT, session_id TEXT);"
            "CREATE TABLE attachments(attachment_id TEXT, blob_hash BLOB, byte_count INTEGER);"
            "CREATE TABLE attachment_refs(ref_id TEXT, attachment_id TEXT, session_id TEXT, message_id TEXT, "
            "supplying_raw_id TEXT);"
            "CREATE TABLE attachment_native_ids(ref_id TEXT, id_kind TEXT, native_id TEXT);"
        )
        source.executescript(
            "CREATE TABLE raw_sessions(raw_id TEXT, blob_hash BLOB);"
            "CREATE TABLE blob_refs(ref_id TEXT, ref_type TEXT, source_path TEXT, blob_hash BLOB, size_bytes INTEGER);"
        )
        index.execute("INSERT INTO messages VALUES ('message-a', 'session-a')")
        index.execute("INSERT INTO attachments VALUES ('content-version-1', X'aa', 3)")
        index.execute(
            "INSERT INTO attachment_refs VALUES ('message-a:attachment:0', 'content-version-1', 'session-a', "
            "'message-a', 'raw-a')"
        )
        source.execute("INSERT INTO raw_sessions VALUES ('raw-a', X'01')")
        index.execute("INSERT INTO attachment_native_ids VALUES ('message-a:attachment:0', 'file', 'file-a')")
        source.execute("INSERT INTO blob_refs VALUES ('raw-a', 'attachment', 'attachment:file-a', X'aa', 3)")
        assert _resolve_attachment_in_connections(
            index, source, session_id="session-a", reference_id="message-a:attachment:0"
        ) == ("message-a:attachment:0", "message-a")
        resolved = asyncio.run(
            resolve_insight_target(
                Path("/unused"),
                target_type="attachment",
                target_id="message-a:attachment:0",
                session_id="session-a",
                index_connection=index,
                source_connection=source,
            )
        )
        assert resolved["target_id"] == "message-a:attachment:0"
        assert resolved["message_id"] == "message-a"

        guard = bind_attachment_source_guard(
            SimpleNamespace(archive=SimpleNamespace(_conn=index, source_connection=source)),
            session_id="session-a",
            ref_id="message-a:attachment:0",
            current_index_connection=index,
            current_source_connection=source,
        )
        guard()

        # Acquisition can replace the content-addressed attachment row while
        # preserving the stable message reference used by new user state.
        index.execute("UPDATE attachments SET attachment_id='content-version-2'")
        index.execute("UPDATE attachment_refs SET attachment_id='content-version-2'")
        assert _resolve_attachment_in_connections(
            index, source, session_id="session-a", reference_id="message-a:attachment:0"
        ) == ("message-a:attachment:0", "message-a")
        assert (
            _resolve_attachment_in_connections(index, source, session_id="session-a", reference_id="content-version-2")
            is None
        )

        assert _resolve_attachment_in_connections(
            index, source, session_id="session-a", reference_id="message-a:attachment:0"
        ) == ("message-a:attachment:0", "message-a")
        source.execute("UPDATE blob_refs SET source_path='attachment:other-file'")
        assert (
            _resolve_attachment_in_connections(
                index, source, session_id="session-a", reference_id="message-a:attachment:0"
            )
            is None
        )

        # A guard bound before the supplier's Source evidence moves refuses at
        # the durable-apply boundary.
        with pytest.raises(ValueError, match="changed before durable apply"):
            guard()

        source.execute("DELETE FROM raw_sessions WHERE raw_id='raw-a'")
        assert (
            _resolve_attachment_in_connections(
                index, source, session_id="session-a", reference_id="message-a:attachment:0"
            )
            is None
        )
    finally:
        index.close()
        source.close()

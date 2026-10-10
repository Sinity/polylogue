"""Retire a profile binding once without losing later input obligations."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest

from polylogue.archive.query.transaction import archive_snapshot_epoch
from polylogue.storage.derived.session.derivation import SessionProfileDerivation, publish_session_profile
from polylogue.storage.derived.session.input_binding import session_input_bindings
from polylogue.storage.derived.session.usage_rollup import (
    publish_session_usage_rollup,
    session_usage_rollup_recipe_version,
)
from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.connection_profile import open_connection
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.session_profiles import write_session_profile
from tests.infra.storage_records import SessionBuilder


@pytest.fixture
def archive(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    builder = SessionBuilder(root / "index.db", "binding-retirement")
    builder.add_message(role="user", text="retained first input")
    builder.add_message(role="assistant", text="retained second input")
    builder.save()
    yield root, builder.native_session_id()


def _connection(root: Path) -> sqlite3.Connection:
    conn = open_connection(root / "index.db")
    conn.row_factory = sqlite3.Row
    return conn


def _revision(conn: sqlite3.Connection, session_id: str) -> int:
    row = conn.execute("SELECT revision FROM session_profile_demand WHERE session_id=?", (session_id,)).fetchone()
    return 0 if row is None else int(row[0])


def _profile_epoch(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT epoch FROM query_unit_frame_state WHERE relation='session_profiles'").fetchone()[0])


def _seed_related(conn: sqlite3.Connection, session_id: str) -> None:
    message_id = conn.execute(
        "SELECT message_id FROM messages WHERE session_id=? ORDER BY position LIMIT 1", (session_id,)
    ).fetchone()[0]
    for position in range(2):
        conn.execute("INSERT INTO attachments(attachment_id) VALUES (?)", (f"attachment-{position}",))
        conn.execute(
            "INSERT INTO attachment_refs(attachment_id, session_id, message_id, native_identity, position) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"attachment-{position}", session_id, message_id, f"{position:02x}", position),
        )
        conn.execute(
            "INSERT INTO session_events(session_id, position, event_type) VALUES (?, ?, 'compaction')",
            (session_id, position),
        )
        conn.execute(
            "INSERT INTO session_working_dirs(session_id, path, position) VALUES (?, ?, ?)",
            (session_id, f"/synthetic/before/{position}", position),
        )
        conn.execute(
            "INSERT INTO session_provider_usage_events(session_id, position, provider_event_type) "
            "VALUES (?, ?, 'token_count')",
            (session_id, position),
        )


def _mutate(conn: sqlite3.Connection, session_id: str, relation: str, operation: str, position: int) -> None:
    if relation == "sessions":
        conn.execute("UPDATE sessions SET title=? WHERE session_id=?", (f"changed-{position}", session_id))
    elif relation == "attachments":
        if operation == "update":
            conn.execute(
                "UPDATE attachments SET display_name=? WHERE attachment_id=?", (f"changed-{position}", "attachment-0")
            )
        else:
            conn.execute("DELETE FROM attachments WHERE attachment_id=?", (f"attachment-{position}",))
    elif operation == "update":
        expressions = {
            "messages": "word_count = word_count + 1",
            "attachment_refs": "caption = COALESCE(caption, '') || 'changed'",
            "session_events": "occurred_at_ms = COALESCE(occurred_at_ms, 0) + 1",
            "session_working_dirs": "path = path || '/changed'",
            "session_provider_usage_events": "last_input_tokens = COALESCE(last_input_tokens, 0) + 1",
        }
        conn.execute(f"UPDATE {relation} SET {expressions[relation]} WHERE session_id=? AND position=0", (session_id,))
    elif operation == "delete":
        conn.execute(f"DELETE FROM {relation} WHERE session_id=? AND position=?", (session_id, position))
    elif relation == "messages":
        conn.execute(
            "INSERT INTO messages(session_id, native_id, position, role, content_hash) "
            "VALUES (?, ?, ?, 'user', zeroblob(32))",
            (session_id, f"new-{position}", position + 2),
        )
    elif relation == "attachment_refs":
        message_id = conn.execute(
            "SELECT message_id FROM messages WHERE session_id=? AND position=0", (session_id,)
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO attachment_refs(attachment_id, session_id, message_id, native_identity, position) "
            "VALUES ('attachment-0', ?, ?, ?, ?)",
            (session_id, message_id, f"{position + 10:02x}", position + 2),
        )
    elif relation == "session_events":
        conn.execute(
            "INSERT INTO session_events(session_id, position, event_type) VALUES (?, ?, 'compaction')",
            (session_id, position + 2),
        )
    elif relation == "session_working_dirs":
        conn.execute(
            "INSERT INTO session_working_dirs(session_id, path, position) VALUES (?, ?, ?)",
            (session_id, f"/synthetic/new/{position}", position + 2),
        )
    else:
        assert relation == "session_provider_usage_events"
        conn.execute(
            "INSERT INTO session_provider_usage_events(session_id, position, provider_event_type) "
            "VALUES (?, ?, 'token_count')",
            (session_id, position + 2),
        )


_MUTATIONS = tuple(
    (relation, operation)
    for relation in (
        "messages",
        "attachment_refs",
        "session_events",
        "session_working_dirs",
        "session_provider_usage_events",
    )
    for operation in ("insert", "update", "delete")
) + (("sessions", "update"), ("attachments", "update"), ("attachments", "delete"))


@pytest.mark.parametrize(("relation", "operation"), _MUTATIONS)
def test_repeated_input_mutations_retire_one_binding_but_keep_every_demand(
    archive: tuple[Path, str], relation: str, operation: str
) -> None:
    root, session_id = archive
    with write_lease("test.binding-retirement", archive_root=root), closing(_connection(root)) as conn:
        _seed_related(conn, session_id)
        write_session_profile(conn, session_id, input_content_hash="published-binding")
        conn.execute("DELETE FROM session_profile_demand")
        conn.commit()
        before = _profile_epoch(conn)
        for position in range(2):
            _mutate(conn, session_id, relation, operation, position)
            assert (
                conn.execute(
                    "SELECT input_content_hash FROM session_profiles WHERE session_id=?", (session_id,)
                ).fetchone()[0]
                is None
            )
            assert _profile_epoch(conn) == before + 1
        # OLD and NEW each enqueue on relational UPDATE, even with the same
        # owner. Attachment metadata and session-row UPDATE enqueue once.
        increments = 2 if operation == "update" and relation not in {"sessions", "attachments"} else 1
        expected = increments * 2
        if relation == "messages" and operation == "delete":
            # The first message owns two attachment refs whose FK cascades
            # retain their own demands, in addition to both message deletes.
            expected += 2
        assert _revision(conn, session_id) == expected
        conn.rollback()
        assert (
            conn.execute(
                "SELECT input_content_hash FROM session_profiles WHERE session_id=?", (session_id,)
            ).fetchone()[0]
            == "published-binding"
        )
        assert _profile_epoch(conn) == before
        assert _revision(conn, session_id) == 0


def test_thousand_message_inserts_keep_demand_and_query_frame_after_first_retirement(
    archive: tuple[Path, str],
) -> None:
    root, session_id = archive
    with write_lease("test.binding-batch", archive_root=root), closing(_connection(root)) as conn:
        write_session_profile(conn, session_id, input_content_hash="published-binding")
        conn.execute("DELETE FROM session_profile_demand")
        before = _profile_epoch(conn)
        messages_before = conn.execute("SELECT epoch FROM query_unit_frame_state WHERE relation='messages'").fetchone()[
            0
        ]
        conn.commit()
        with ArchiveStore.open_existing(root) as reader:
            first_frame = archive_snapshot_epoch(reader, relations=("messages",))
        conn.executemany(
            "INSERT INTO messages(session_id, native_id, position, role, content_hash) "
            "VALUES (?, ?, ?, 'user', zeroblob(32))",
            ((session_id, f"batch-{position}", position + 2) for position in range(1000)),
        )
        assert _profile_epoch(conn) == before + 1
        assert _revision(conn, session_id) == 1000
        assert (
            conn.execute("SELECT epoch FROM query_unit_frame_state WHERE relation='messages'").fetchone()[0]
            == messages_before + 1000
        )
        conn.commit()
        with ArchiveStore.open_existing(root) as reader:
            last_frame = archive_snapshot_epoch(reader, relations=("messages",))
        assert last_frame != first_frame


def test_a_preparation_cannot_publish_after_a_repeat_mutation_of_a_null_binding(
    archive: tuple[Path, str],
) -> None:
    root, session_id = archive
    with write_lease("test.publish-initial", archive_root=root), closing(_connection(root)) as conn:
        binding = session_input_bindings(conn, (session_id,))[session_id]
        publish_session_usage_rollup(
            conn, session_id, input_binding=binding, recipe_version=session_usage_rollup_recipe_version()
        )
        assert publish_session_profile(conn, session_id, input_binding=binding)
    adapter = SessionProfileDerivation(
        lambda: sqlite3.connect(f"file:{root / 'index.db'}?mode=ro", uri=True),
        lambda: _connection(root),
        materializer_version=SESSION_INSIGHT_MATERIALIZER_VERSION,
        session_scope=lambda _frame: (session_id,),
        archive_root=root,
    )
    with write_lease("test.first-mutation", archive_root=root), closing(_connection(root)) as conn:
        _mutate(conn, session_id, "messages", "update", 0)
        conn.commit()
    prepared = adapter.compute(object(), session_id)
    with write_lease("test.repeat-mutation", archive_root=root), closing(_connection(root)) as conn:
        before = _profile_epoch(conn)
        _mutate(conn, session_id, "messages", "update", 1)
        assert _profile_epoch(conn) == before
        assert _revision(conn, session_id) == prepared.demand_revision + 2
        conn.commit()
    assert adapter.publish(object(), prepared) is False
    with closing(_connection(root)) as conn:
        assert _revision(conn, session_id) == prepared.demand_revision + 2

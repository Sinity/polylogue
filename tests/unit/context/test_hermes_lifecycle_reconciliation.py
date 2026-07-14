"""fs1.7: reconcile a Hermes session's drained lifecycle-event stream (review fix).

The reviewer found ``hermes_lifecycle.reconcile_lifecycle_events`` unit-tested
but unreachable from any production surface -- nothing read the durable
``raw_hook_events`` spool and the ingested snapshot and fed them through it.
This module is that read-side join; these tests exercise it directly against
real ``source.db``/``index.db`` connections (the facade-level integration
test lives in ``tests/unit/api/test_facade_contracts.py``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.context.hermes_lifecycle_reconciliation import reconcile_hermes_session_lifecycle
from polylogue.sources.hooks import drain_hook_event_spool, enqueue_hook_event
from polylogue.sources.parsers.hermes_lifecycle import DURABLE_FINALIZE, TOOL_FINISH, TOOL_START
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.index import INDEX_DDL, INDEX_SCHEMA_VERSION
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_HASH = b"x" * 32


def _index_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(INDEX_DDL)
    conn.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
    return conn


def _source_conn(archive_root: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _seed_snapshot(index_conn: sqlite3.Connection, *, qualified_native_id: str, message_native_id: str) -> None:
    index_conn.execute(
        "INSERT INTO sessions (native_id, origin, title, content_hash, message_count) VALUES (?, ?, ?, ?, ?)",
        (qualified_native_id, "hermes-session", "test", _HASH, 1),
    )
    index_conn.execute(
        "INSERT INTO messages (session_id, native_id, position, role, message_type, content_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (f"hermes-session:{qualified_native_id}", message_native_id, 0, "assistant", "message", _HASH),
    )
    index_conn.commit()


def test_complete_paired_stream_against_known_snapshot_message_reconciles_clean(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    spool_root = tmp_path / "hooks"
    index_conn = _index_conn()
    _seed_snapshot(index_conn, qualified_native_id="conv-1@profile-abc", message_native_id="m1")

    enqueue_hook_event(
        event_id="e1",
        provider="hermes",
        event_type=TOOL_START,
        session_id="conv-1",
        timestamp="2026-07-12T10:00:00Z",
        payload={"tool_call_id": "call-1", "message_id": "m1"},
        root=spool_root,
    )
    enqueue_hook_event(
        event_id="e2",
        provider="hermes",
        event_type=TOOL_FINISH,
        session_id="conv-1",
        timestamp="2026-07-12T10:00:01Z",
        payload={"tool_call_id": "call-1", "message_id": "m1"},
        root=spool_root,
    )
    enqueue_hook_event(
        event_id="e3",
        provider="hermes",
        event_type=DURABLE_FINALIZE,
        session_id="conv-1",
        timestamp="2026-07-12T10:00:02Z",
        payload={},
        root=spool_root,
    )
    assert drain_hook_event_spool(archive_root, root=spool_root).acknowledged == 3

    source_conn = _source_conn(archive_root)
    report = reconcile_hermes_session_lifecycle(source_conn, index_conn, hermes_session_native_id="conv-1")

    assert report.total_events == 3
    assert report.complete
    assert report.finalized
    assert report.caveats == ()


def test_unpaired_event_and_unknown_message_reference_are_both_visible(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    spool_root = tmp_path / "hooks"
    index_conn = _index_conn()
    _seed_snapshot(index_conn, qualified_native_id="conv-2@profile-xyz", message_native_id="m1")

    enqueue_hook_event(
        event_id="e1",
        provider="hermes",
        event_type=TOOL_START,
        session_id="conv-2",
        timestamp="2026-07-12T10:00:00Z",
        payload={"tool_call_id": "call-1", "message_id": "message-not-ingested"},
        root=spool_root,
    )
    assert drain_hook_event_spool(archive_root, root=spool_root).acknowledged == 1

    source_conn = _source_conn(archive_root)
    report = reconcile_hermes_session_lifecycle(source_conn, index_conn, hermes_session_native_id="conv-2")

    assert not report.complete
    assert report.unpaired_event_ids == ("hook:e1",)
    assert report.events_referencing_unknown_messages == ("hook:e1",)
    assert not report.finalized


def test_raw_unqualified_session_id_resolves_against_profile_qualified_snapshot(tmp_path: Path) -> None:
    """A caller only ever holds the raw Hermes session id (see module docstring);
    the snapshot join must still find messages under the qualified native_id."""

    archive_root = tmp_path / "archive"
    spool_root = tmp_path / "hooks"
    index_conn = _index_conn()
    _seed_snapshot(index_conn, qualified_native_id="conv-3@profile-anything", message_native_id="only-message")

    enqueue_hook_event(
        event_id="e1",
        provider="hermes",
        event_type=TOOL_START,
        session_id="conv-3",
        timestamp="2026-07-12T10:00:00Z",
        payload={"tool_call_id": "call-1", "message_id": "only-message"},
        root=spool_root,
    )
    assert drain_hook_event_spool(archive_root, root=spool_root).acknowledged == 1

    source_conn = _source_conn(archive_root)
    report = reconcile_hermes_session_lifecycle(source_conn, index_conn, hermes_session_native_id="conv-3")

    assert report.events_referencing_unknown_messages == ()


def test_session_with_no_drained_events_reconciles_as_a_well_formed_empty_report(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(archive_root / "source.db") as init_conn:
        initialize_archive_tier(init_conn, ArchiveTier.SOURCE)
        init_conn.commit()
    index_conn = _index_conn()

    source_conn = _source_conn(archive_root)
    report = reconcile_hermes_session_lifecycle(source_conn, index_conn, hermes_session_native_id="conv-none")

    assert report.total_events == 0
    assert report.complete
    assert not report.finalized

"""polylogue-tfzw0: blob GC liveness must be a per-ref_type JOIN, not membership.

Regression coverage for the fix: ``blob_gc.py``'s prior ``_archive_reference_
surfaces`` treated ANY row in ``blob_refs`` matching a candidate blob hash as
proof the blob is still "referenced" -- a tautology, since the row being
asked about is itself the "evidence". A hook event's own durable blob ref
(``write_source_hook_event``) never mints a ``raw_sessions`` row
(polylogue-31r1), so its ref could never join to a live referent under the
corrected check and was retained forever, uncounted, under the old one
(measured: 73,427 orphaned rows / ~1.94 GiB on the live archive).

These tests exercise the real production write path
(``ArchiveStore.write_hook_event`` -> ``write_source_hook_event`` ->
``_insert_blob_ref``/``_insert_hook_event``) and the real GC entrypoint
(``run_blob_gc``), not a synthetic reimplementation -- reverting
``_blob_refs_still_live`` back to a bare membership check on ``blob_refs``
makes ``test_hook_payload_ref_survives_gc_while_hook_event_row_exists``'s
final assertion fail (the second GC pass would report ``deleted == 0``
instead of ``1``, because the stale ref alone would still "prove" liveness).
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pytest

from polylogue.core.enums import Origin, Provider
from polylogue.storage.blob_gc import census_orphaned_blob_refs, run_blob_gc
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent

pytestmark = pytest.mark.uses_real_clock(
    "backdates a real blob mtime via os.utime; blob_gc.py's age gate compares it against a real time.time() call"
)


def _backdate(blob_store: BlobStore, blob_hash: str, *, seconds: float = 3600) -> None:
    path = blob_store.blob_path(blob_hash)
    past = time.time() - seconds
    os.utime(path, (past, past))


def _write_hook_event(archive_root: Path, *, hook_event_id: str, source_path: str, payload: bytes) -> None:
    with ArchiveStore(archive_root) as archive:
        archive.write_hook_event(
            provider=Provider.CODEX,
            payload=payload,
            source_path=source_path,
            acquired_at_ms=1_700_000_000_000,
            hook_event=ArchiveHookEvent(
                hook_event_id=hook_event_id,
                origin=Origin.CODEX_SESSION,
                source_path=source_path,
                event_type="PostToolUse",
                payload={"event": "PostToolUse"},
                observed_at_ms=1_700_000_000_000,
                native_id=f"{hook_event_id}:native",
                session_native_id="session-native-1",
            ),
        )


def test_hook_payload_ref_written_as_hook_payload_not_raw_payload(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    _write_hook_event(archive_root, hook_event_id="hook-1", source_path="/hooks/a.jsonl", payload=b'{"a":1}')

    with sqlite3.connect(archive_root / "source.db") as conn:
        rows = conn.execute("SELECT ref_type, ref_id FROM blob_refs").fetchall()
        assert rows == [("hook_payload", "hook-1")]
        blob_hash = conn.execute("SELECT blob_hash FROM raw_hook_events WHERE hook_event_id = 'hook-1'").fetchone()[0]
        assert blob_hash is not None


def test_hook_payload_ref_survives_gc_while_hook_event_row_exists_then_is_reclaimed_once_deleted(
    tmp_path: Path,
) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    payload = b'{"event":"PostToolUse","n":1}'
    _write_hook_event(archive_root, hook_event_id="hook-live", source_path="/hooks/b.jsonl", payload=payload)

    blob_store = BlobStore(archive_root / "blob")
    with sqlite3.connect(archive_root / "source.db") as conn:
        stored_blob_hash = conn.execute(
            "SELECT blob_hash FROM raw_hook_events WHERE hook_event_id = 'hook-live'"
        ).fetchone()[0]
    blob_hash = stored_blob_hash.hex()
    _backdate(blob_store, blob_hash)

    # Live: the hook event row exists, so the 'hook_payload' ref joins.
    deleted = run_blob_gc(archive_root / "source.db", archive_root / "blob", max_batch=10)
    assert deleted == 0
    assert blob_store.exists(blob_hash)

    # Simulate the hook event row being gone (a since-deleted/repaired row --
    # exactly the shape a genuinely dead ref takes): the blob_refs row is now
    # a true orphan and must no longer protect the blob.
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("DELETE FROM raw_hook_events WHERE hook_event_id = 'hook-live'")
        conn.commit()

    deleted = run_blob_gc(archive_root / "source.db", archive_root / "blob", max_batch=10)
    assert deleted == 1
    assert not blob_store.exists(blob_hash)


def test_attachment_ref_type_joins_against_raw_sessions_not_raw_artifacts(tmp_path: Path) -> None:
    """Verified against every production call site (polylogue-tfzw0): an
    'attachment' blob ref's ref_id is always the parent session's raw_id
    (``write_source_raw_session``/``write_source_raw_session_blob_ref`` pass
    ``ref.raw_id=resolved_raw_id`` for every entry in ``additional_blob_refs``
    regardless of ``ref_type``), never a ``raw_artifacts.artifact_id``. This
    proves the GC join treats it that way: an attachment ref with no matching
    raw_sessions row is reclaimed.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    blob_store = BlobStore(archive_root / "blob")
    orphaned_hash, _size = blob_store.write_from_bytes(b"orphaned attachment bytes")
    _backdate(blob_store, orphaned_hash)

    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'raw-that-does-not-exist', 'attachment', '/tmp/a.json', 10, 1)
            """,
            (bytes.fromhex(orphaned_hash),),
        )
        conn.commit()

    deleted = run_blob_gc(archive_root / "source.db", archive_root / "blob", max_batch=10)
    assert deleted == 1
    assert not blob_store.exists(orphaned_hash)


def test_census_orphaned_blob_refs_counts_by_ref_type(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    _write_hook_event(archive_root, hook_event_id="hook-live", source_path="/hooks/c.jsonl", payload=b'{"c":1}')

    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.row_factory = sqlite3.Row
        # A live hook event contributes zero to the census.
        census = census_orphaned_blob_refs(conn)
        assert census.total == 0
        assert census.by_ref_type == {}

        # Two orphaned refs: one raw_payload (no raw_sessions row), one
        # hook_payload (no raw_hook_events row).
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'raw-gone', 'raw_payload', '/tmp/gone.json', 10, 1)
            """,
            (b"\x11" * 32,),
        )
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'hook-gone', 'hook_payload', '/tmp/gone2.json', 10, 1)
            """,
            (b"\x22" * 32,),
        )
        conn.commit()

        census = census_orphaned_blob_refs(conn)
        assert census.total == 2
        assert census.by_ref_type == {"raw_payload": 1, "hook_payload": 1}
        assert census.to_dict() == {"total": 2, "by_ref_type": {"raw_payload": 1, "hook_payload": 1}}

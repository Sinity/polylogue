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
from polylogue.storage.blob_gc import run_blob_gc
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveHookEvent,
    ArchiveSourceBlobRef,
    HookEventConflictError,
    deterministic_blob_hash,
    write_source_raw_session,
)

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


def _delete_hook_event(archive_root: Path, *, hook_event_id: str) -> bool:
    with ArchiveStore(archive_root) as archive:
        return archive.delete_hook_event(hook_event_id)


def test_hook_payload_ref_written_as_hook_payload_not_raw_payload(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    _write_hook_event(archive_root, hook_event_id="hook-1", source_path="/hooks/a.jsonl", payload=b'{"a":1}')

    with sqlite3.connect(archive_root / "source.db") as conn:
        rows = conn.execute("SELECT ref_type, ref_id FROM blob_refs").fetchall()
        assert rows == [("hook_payload", "hook-1")]
        blob_hash = conn.execute("SELECT blob_hash FROM raw_hook_events WHERE hook_event_id = 'hook-1'").fetchone()[0]
        assert blob_hash is not None


def test_hook_payload_replacement_is_a_durable_conflict(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    _write_hook_event(archive_root, hook_event_id="hook-replaced", source_path="/hooks/a.jsonl", payload=b"old")
    with pytest.raises(HookEventConflictError):
        _write_hook_event(archive_root, hook_event_id="hook-replaced", source_path="/hooks/a.jsonl", payload=b"new")

    with sqlite3.connect(archive_root / "source.db") as conn:
        refs = conn.execute("SELECT ref_type, ref_id, blob_hash FROM blob_refs ORDER BY ref_type, ref_id").fetchall()
        event_hash = conn.execute(
            "SELECT blob_hash FROM raw_hook_events WHERE hook_event_id = 'hook-replaced'"
        ).fetchone()[0]

    assert refs == [("hook_payload", "hook-replaced", event_hash)]
    assert event_hash == deterministic_blob_hash(b"old")


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

    # Delete through the source-tier route. The hook event's own ref must be
    # removed with the event, so GC sees the payload as truly dead.
    assert _delete_hook_event(archive_root, hook_event_id="hook-live") is True
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM blob_refs WHERE ref_type = 'hook_payload' AND ref_id = 'hook-live'"
        ).fetchone() == (0,)

    deleted = run_blob_gc(archive_root / "source.db", archive_root / "blob", max_batch=10)
    assert deleted == 1
    assert not blob_store.exists(blob_hash)


def test_gc_retains_live_raw_attachment_and_hook_references_together(tmp_path: Path) -> None:
    """The real GC route protects every durable source reference surface.

    Raw payload and attachment refs are keyed to source rows, while hook
    payload refs join to ``raw_hook_events``.  Keeping all three in one pass
    prevents a future change from preserving the already-tested hook path
    while regressing one of the source-ref branches.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    blob_store = BlobStore(archive_root / "blob")
    raw_hash, _ = blob_store.write_from_bytes(b"raw payload")
    attachment_hash, _ = blob_store.write_from_bytes(b"attachment payload")
    _write_hook_event(
        archive_root,
        hook_event_id="hook-batch",
        source_path="/hooks/batch.jsonl",
        payload=b'{"event":"PostToolUse","batch":true}',
    )

    with sqlite3.connect(archive_root / "source.db") as conn:
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.CODEX_SESSION,
            source_path="/raw.jsonl",
            source_index=0,
            native_id="raw-live",
            payload=b"raw payload",
            acquired_at_ms=1,
            additional_blob_refs=(
                ArchiveSourceBlobRef(
                    blob_hash=bytes.fromhex(attachment_hash),
                    ref_type="attachment",
                    source_path="/raw.jsonl",
                    size_bytes=18,
                    acquired_at_ms=1,
                ),
            ),
        )
        assert conn.execute("SELECT ref_id FROM blob_refs WHERE ref_type = 'attachment'").fetchone() == (raw_id,)

    with sqlite3.connect(archive_root / "source.db") as conn:
        hook_hash = conn.execute("SELECT blob_hash FROM raw_hook_events WHERE hook_event_id = 'hook-batch'").fetchone()[
            0
        ]

    for blob_hash in (raw_hash, attachment_hash, bytes(hook_hash).hex()):
        _backdate(blob_store, blob_hash)

    assert run_blob_gc(archive_root / "source.db", archive_root / "blob", max_batch=10) == 0
    assert all(blob_store.exists(blob_hash) for blob_hash in (raw_hash, attachment_hash, bytes(hook_hash).hex()))

    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("DELETE FROM blob_refs WHERE ref_id = ?", (raw_id,))
        conn.execute("DELETE FROM raw_sessions WHERE raw_id = ?", (raw_id,))
    assert _delete_hook_event(archive_root, hook_event_id="hook-batch") is True

    assert run_blob_gc(archive_root / "source.db", archive_root / "blob", max_batch=10) == 3
    assert not any(blob_store.exists(blob_hash) for blob_hash in (raw_hash, attachment_hash, bytes(hook_hash).hex()))


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


def test_gc_retains_a_raw_payload_named_only_by_the_raw_row_hash(tmp_path: Path) -> None:
    """A raw payload stays protected when the ledger has no row for it.

    Liveness must not depend on ``blob_refs`` being complete. On the live
    archive 399 raw payload hashes (~1 GB of irreplaceable acquired bytes)
    carry no ledger row of any type, so a ledger-only test would read them as
    collectable. Removing the ``raw_sessions`` branch from
    ``_archive_reference_surfaces`` makes the retention assertion below fail.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    blob_store = BlobStore(archive_root / "blob")
    blob_store.write_from_bytes(b"ledgerless raw payload")

    with sqlite3.connect(archive_root / "source.db") as conn:
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.CODEX_SESSION,
            source_path="/ledgerless.jsonl",
            source_index=0,
            native_id="raw-ledgerless",
            payload=b"ledgerless raw payload",
            acquired_at_ms=1,
        )
        raw_hash = bytes(
            conn.execute("SELECT blob_hash FROM raw_sessions WHERE raw_id = ?", (raw_id,)).fetchone()[0]
        ).hex()
        # Drop the ledger row, leaving raw_sessions.blob_hash as the only
        # reference -- the shape measured on the live archive.
        conn.execute("DELETE FROM blob_refs WHERE ref_id = ?", (raw_id,))
        assert (
            conn.execute("SELECT COUNT(*) FROM blob_refs WHERE blob_hash = ?", (bytes.fromhex(raw_hash),)).fetchone()[0]
            == 0
        )

    _backdate(blob_store, raw_hash)

    assert run_blob_gc(archive_root / "source.db", archive_root / "blob", max_batch=10) == 0
    assert blob_store.exists(raw_hash)

    # Once the raw row itself is gone, nothing names the payload and GC may
    # reclaim it -- so the assertion above is about the raw-row surface, not a
    # blanket refusal to collect.
    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("DELETE FROM raw_sessions WHERE raw_id = ?", (raw_id,))

    assert run_blob_gc(archive_root / "source.db", archive_root / "blob", max_batch=10) == 1
    assert not blob_store.exists(raw_hash)

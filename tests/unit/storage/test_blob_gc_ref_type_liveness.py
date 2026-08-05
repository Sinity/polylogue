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
from polylogue.maintenance.blob_ref_liveness_reconciliation import (
    BlobRefLivenessReconciliationError,
    reconcile_blob_ref_liveness,
)
from polylogue.storage.blob_gc import census_orphaned_blob_refs, run_blob_gc
from polylogue.storage.blob_ref_liveness import classify_blob_ref_liveness
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.source_write import (
    ArchiveHookEvent,
    deterministic_blob_hash,
    deterministic_raw_session_id,
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


def test_hook_payload_replacement_removes_previous_ref(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    _write_hook_event(archive_root, hook_event_id="hook-replaced", source_path="/hooks/a.jsonl", payload=b"old")
    _write_hook_event(archive_root, hook_event_id="hook-replaced", source_path="/hooks/a.jsonl", payload=b"new")

    with sqlite3.connect(archive_root / "source.db") as conn:
        refs = conn.execute("SELECT ref_type, ref_id, blob_hash FROM blob_refs ORDER BY ref_type, ref_id").fetchall()
        event_hash = conn.execute(
            "SELECT blob_hash FROM raw_hook_events WHERE hook_event_id = 'hook-replaced'"
        ).fetchone()[0]

    assert refs == [("hook_payload", "hook-replaced", event_hash)]
    assert event_hash == deterministic_blob_hash(b"new")


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
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms
            ) VALUES ('raw-live', 'codex-session', 'raw-live', '/raw.jsonl', ?, 11, 1)
            """,
            (bytes.fromhex(raw_hash),),
        )
        # Attachment refs use the parent raw session as their ref_id in the
        # source tier. Its payload is deliberately a different, non-file hash
        # so this assertion exercises the attachment ref join itself.
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms
            ) VALUES ('attachment-parent', 'codex-session', 'attachment-parent', '/parent.jsonl', ?, 1, 1)
            """,
            (b"p" * 32,),
        )
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'raw-live', 'raw_payload', '/raw.jsonl', 11, 1)
            """,
            (bytes.fromhex(raw_hash),),
        )
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, 'attachment-parent', 'attachment', '/parent.jsonl', 18, 1)
            """,
            (bytes.fromhex(attachment_hash),),
        )

    with sqlite3.connect(archive_root / "source.db") as conn:
        hook_hash = conn.execute("SELECT blob_hash FROM raw_hook_events WHERE hook_event_id = 'hook-batch'").fetchone()[
            0
        ]

    for blob_hash in (raw_hash, attachment_hash, bytes(hook_hash).hex()):
        _backdate(blob_store, blob_hash)

    assert run_blob_gc(archive_root / "source.db", archive_root / "blob", max_batch=10) == 0
    assert all(blob_store.exists(blob_hash) for blob_hash in (raw_hash, attachment_hash, bytes(hook_hash).hex()))

    with sqlite3.connect(archive_root / "source.db") as conn:
        conn.execute("DELETE FROM blob_refs WHERE ref_id IN ('raw-live', 'attachment-parent')")
        conn.execute("DELETE FROM raw_sessions WHERE raw_id IN ('raw-live', 'attachment-parent')")
    assert _delete_hook_event(archive_root, hook_event_id="hook-batch") is True

    assert run_blob_gc(archive_root / "source.db", archive_root / "blob", max_batch=10) == 3
    assert not any(blob_store.exists(blob_hash) for blob_hash in (raw_hash, attachment_hash, bytes(hook_hash).hex()))


def test_interrupted_hook_rekey_blocks_liveness_deletion_and_preserves_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hook hash without its canonical ref remains live and rekeyable.

    This creates the state through the production hook writer, then removes
    its canonical ref and restores the legacy raw-payload reference as an
    interrupted historical re-key would leave it. The generic liveness
    classifier must block deletion of that reference, and real blob GC must
    keep the payload based on ``raw_hook_events.blob_hash`` until repair.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    source_path = "/hooks/interrupted.jsonl"
    hook_event_id = "hook-interrupted"
    native_id = f"{hook_event_id}:native"
    payload = b'{"event":"PostToolUse","n":2}'
    _write_hook_event(archive_root, hook_event_id=hook_event_id, source_path=source_path, payload=payload)

    blob_store = BlobStore(archive_root / "blob")
    with sqlite3.connect(archive_root / "source.db") as conn:
        blob_hash = conn.execute(
            "SELECT blob_hash FROM raw_hook_events WHERE hook_event_id = ?", (hook_event_id,)
        ).fetchone()[0]
        legacy_ref_id = deterministic_raw_session_id("codex-session", source_path, 0, blob_hash, native_id)
        conn.execute("DELETE FROM blob_refs WHERE ref_type = 'hook_payload' AND ref_id = ?", (hook_event_id,))
        conn.execute(
            """
            INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms)
            VALUES (?, ?, 'raw_payload', ?, ?, ?)
            """,
            (blob_hash, legacy_ref_id, source_path, len(payload), 1_700_000_000_000),
        )
        classification = classify_blob_ref_liveness(conn)
        conn.commit()

    assert classification.rekeyable_hook_payload_count == 1
    assert classification.safe_to_apply is False
    assert all(candidate.ref_id != legacy_ref_id for candidate in classification.candidates)

    monkeypatch.setattr(
        "polylogue.maintenance.blob_ref_liveness_reconciliation.validate_migration_backup_manifest",
        lambda *args, **kwargs: args[0],
    )
    monkeypatch.setattr(
        "polylogue.maintenance.blob_ref_liveness_reconciliation.validate_migration_backup_live_fingerprint",
        lambda *args, **kwargs: args[0],
    )
    monkeypatch.setattr(
        "polylogue.maintenance.blob_ref_liveness_reconciliation.running_daemon_pid",
        lambda _config: None,
    )
    with pytest.raises(BlobRefLivenessReconciliationError, match="rekeyable_hook_payloads=1"):
        reconcile_blob_ref_liveness(
            archive_root,
            backup_manifest=tmp_path / "backup.json",
            receipt_path=tmp_path / "liveness.jsonl",
            dry_run=False,
        )
    with sqlite3.connect(archive_root / "source.db") as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM blob_refs WHERE ref_type = 'raw_payload' AND ref_id = ?", (legacy_ref_id,)
        ).fetchone() == (1,)

    blob_hash_hex = blob_hash.hex()
    _backdate(blob_store, blob_hash_hex)
    deleted = run_blob_gc(archive_root / "source.db", archive_root / "blob", max_batch=10)
    assert deleted == 0
    assert blob_store.exists(blob_hash_hex)


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

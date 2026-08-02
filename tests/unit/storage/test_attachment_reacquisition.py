"""polylogue-pfdf: backfilling historically-unfetched attachments.

Exercises the real production entry points end to end -- ``ingest_record``
(the live ingest worker's per-raw parse entry point) parses genuine raw bytes
out of a real ``source.db``/blob store, and ``apply_attachment_reacquisition``
mutates a real ``index.db`` -- no mocking of the parse or classification logic
under test.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.pipeline.services.ingest_worker import ingest_record
from polylogue.storage.attachment_reacquisition import (
    AttachmentReacquisitionError,
    apply_attachment_reacquisition,
    plan_attachment_reacquisition,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime.raw.records import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive

_CLAUDE_AI_PAYLOAD = {
    "uuid": "reacq-session-1",
    "name": "Reacquisition test session",
    "chat_messages": [
        {
            "uuid": "m0",
            "sender": "human",
            "text": "here is a file",
            "attachments": [
                {
                    "file_name": "notes.md",
                    "file_type": "text/markdown",
                    "file_size": 11,
                    "extracted_content": "hello notes",
                }
            ],
        }
    ],
}


def _index_conn(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _source_conn(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    initialize_archive_tier(conn, ArchiveTier.SOURCE)
    return conn


def _write_raw_row(
    source_conn: sqlite3.Connection, blob_store: BlobStore, payload: object, *, raw_id: str, source_path: str
) -> RawSessionRecord:
    content = json.dumps(payload).encode("utf-8")
    blob_hash, blob_size = blob_store.write_from_bytes(content)
    source_conn.execute(
        """
        INSERT INTO raw_sessions (raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms)
        VALUES (?, 'claude-ai-export', ?, 0, ?, ?, 0)
        """,
        (raw_id, source_path, bytes.fromhex(blob_hash), blob_size),
    )
    source_conn.commit()
    return RawSessionRecord(
        raw_id=raw_id,
        source_name=Provider.CLAUDE_AI.value,
        payload_provider=Provider.CLAUDE_AI,
        source_path=source_path,
        source_index=0,
        blob_size=blob_size,
        blob_hash=blob_hash,
        acquired_at="2026-01-01T00:00:00+00:00",
    )


def _write_unfetched_extracted_content_attachment(
    archive_root: Path,
    index_conn: sqlite3.Connection,
    source_conn: sqlite3.Connection,
    blob_store: BlobStore,
) -> tuple[str, str]:
    """Write a session whose attachment has recoverable inline bytes *today*,
    but persist it exactly as an unfetched row -- simulating a session
    ingested before the parser fix (60d93b618) landed: the raw payload was,
    and still is, durable in source.db; only the acquisition never ran.
    """
    raw_record = _write_raw_row(
        source_conn, blob_store, _CLAUDE_AI_PAYLOAD, raw_id="raw-1", source_path="conversations.json"
    )
    result = ingest_record(raw_record, str(archive_root), "advisory", blob_root_str=str(blob_store.root))
    assert result.error is None, result.error
    write_payload = result.sessions[0]
    assert write_payload.parsed_session.attachments[0].inline_bytes == b"hello notes"

    # Simulate "ingested before the extracted_content parser fix landed":
    # write the SAME session/attachment identity (attachment_id hashes only
    # content-identity fields, never inline_bytes -- see write.py's
    # _attachment_id) but with inline_bytes stripped, exactly what the
    # pre-60d93b618 parser would have produced. The raw payload written to
    # source.db above is untouched and still carries the real
    # extracted_content -- only the historical write never saw it.
    historical_attachment = write_payload.parsed_session.attachments[0].model_copy(update={"inline_bytes": None})
    historical_session = write_payload.parsed_session.model_copy(update={"attachments": [historical_attachment]})
    write_parsed_session_to_archive(index_conn, historical_session, raw_id=raw_record.raw_id)

    row = index_conn.execute(
        "SELECT attachment_id, acquisition_status FROM attachments WHERE display_name = 'notes.md'"
    ).fetchone()
    assert row is not None
    assert row["acquisition_status"] == "unfetched"
    return str(row["attachment_id"]), write_payload.session_id


def _write_bare_unfetched_attachment(
    index_conn: sqlite3.Connection, *, attachment_id: str, source_url: str | None
) -> None:
    index_conn.execute(
        "INSERT INTO attachments (attachment_id, display_name, media_type, byte_count, acquisition_status, ref_count) "
        "VALUES (?, 'bare.bin', 'application/octet-stream', 42, 'unfetched', 0)",
        (attachment_id,),
    )
    if source_url is not None:
        dummy_hash = bytes(32)
        index_conn.execute(
            "INSERT INTO sessions (origin, native_id, content_hash) VALUES ('unknown-export', 'bare-session', ?)",
            (dummy_hash,),
        )
        index_conn.execute(
            "INSERT INTO messages (session_id, native_id, position, variant_index, role, content_hash) "
            "VALUES ('unknown-export:bare-session', 'm0', 0, 0, 'user', ?)",
            (dummy_hash,),
        )
        index_conn.execute(
            "INSERT INTO attachment_refs (attachment_id, session_id, message_id, position, source_url) "
            "VALUES (?, 'unknown-export:bare-session', 'unknown-export:bare-session:m0', 0, ?)",
            (attachment_id, source_url),
        )
    index_conn.commit()


def _fake_valid_backup(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, object]]:
    validated: list[tuple[Path, object]] = []

    def _fake_validate(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        validated.append((manifest, tier))
        assert connection.execute("SELECT 1").fetchone() == (1,)
        return manifest.with_name("verification-receipt.json")

    monkeypatch.setattr(
        "polylogue.storage.attachment_reacquisition.validate_migration_backup_manifest",
        _fake_validate,
    )
    return validated


def test_plan_finds_reacquirable_and_leaves_ghost_undetermined(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blob")
    index_conn = _index_conn(tmp_path / "index.db")
    source_conn = _source_conn(tmp_path / "source.db")

    recoverable_id, session_id = _write_unfetched_extracted_content_attachment(
        tmp_path, index_conn, source_conn, blob_store
    )
    _write_bare_unfetched_attachment(index_conn, attachment_id="ghost-attachment-id", source_url=None)

    plan = plan_attachment_reacquisition(index_conn, source_conn, archive_root=tmp_path, blob_root=blob_store.root)

    assert plan.unfetched_count == 2
    assert len(plan.reacquirable) == 1
    assert plan.reacquirable[0].attachment_id == recoverable_id
    assert plan.reacquirable[0].session_id == session_id
    assert plan.reacquirable[0].byte_count == len(b"hello notes")
    assert plan.unrecoverable == ()
    assert plan.undetermined_count == 1  # the ghost: no raw reproduces it, not sandbox-shaped either


def test_sandbox_output_classified_unrecoverable_without_reparse(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blob")
    index_conn = _index_conn(tmp_path / "index.db")
    source_conn = _source_conn(tmp_path / "source.db")

    _write_bare_unfetched_attachment(
        index_conn, attachment_id="sandbox-attachment-id", source_url="sandbox:/mnt/data/report.json"
    )

    plan = plan_attachment_reacquisition(index_conn, source_conn, archive_root=tmp_path, blob_root=blob_store.root)

    assert plan.unfetched_count == 1
    assert plan.reacquirable == ()
    assert len(plan.unrecoverable) == 1
    assert plan.unrecoverable[0].attachment_id == "sandbox-attachment-id"
    assert "sandbox" in plan.unrecoverable[0].reason
    assert plan.undetermined_count == 0
    # Zero raw rows should have been scanned -- the sandbox verdict never
    # needed a reparse.
    assert plan.raw_rows_scanned == 0


def test_dry_run_is_default_and_makes_zero_mutation(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blob")
    index_conn = _index_conn(tmp_path / "index.db")
    source_conn = _source_conn(tmp_path / "source.db")
    _write_unfetched_extracted_content_attachment(tmp_path, index_conn, source_conn, blob_store)
    _write_bare_unfetched_attachment(
        index_conn, attachment_id="sandbox-attachment-id", source_url="sandbox:/mnt/data/report.json"
    )
    index_conn.close()
    source_conn.close()

    result = apply_attachment_reacquisition(tmp_path)

    assert result.applied is False
    assert result.unfetched_count == 2
    assert result.reacquirable_count == 1
    assert result.unrecoverable_count == 1
    assert result.reacquired_count == 0
    assert result.marked_unavailable_count == 0

    statuses = _status_by_id(tmp_path)
    assert set(statuses.values()) == {"unfetched"}


def _status_by_id(archive_root: Path) -> dict[str, str]:
    conn = sqlite3.connect(archive_root / "index.db")
    try:
        rows = conn.execute("SELECT attachment_id, blob_hash, acquisition_status FROM attachments").fetchall()
    finally:
        conn.close()
    return {attachment_id: status for attachment_id, _blob_hash, status in rows}


def test_apply_reacquires_and_marks_unavailable_with_verified_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blob_store = BlobStore(tmp_path / "blob")
    index_conn = _index_conn(tmp_path / "index.db")
    source_conn = _source_conn(tmp_path / "source.db")
    recoverable_id, _session_id = _write_unfetched_extracted_content_attachment(
        tmp_path, index_conn, source_conn, blob_store
    )
    _write_bare_unfetched_attachment(
        index_conn, attachment_id="sandbox-attachment-id", source_url="sandbox:/mnt/data/report.json"
    )
    _write_bare_unfetched_attachment(index_conn, attachment_id="ghost-attachment-id", source_url=None)
    index_conn.close()
    source_conn.close()

    validated = _fake_valid_backup(monkeypatch)
    manifest_path = tmp_path / "manifest.jsonl"
    backup_manifest = tmp_path / "verified-backup" / "manifest.json"

    result = apply_attachment_reacquisition(
        tmp_path, manifest_path=manifest_path, backup_manifest=backup_manifest, dry_run=False
    )

    assert result.applied is True
    assert result.reacquired_count == 1
    assert result.reacquired_bytes == len(b"hello notes")
    assert result.marked_unavailable_count == 1
    assert result.errors == ()
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    assert validated == [(backup_manifest, ArchiveTier.INDEX), (backup_manifest, ArchiveTier.INDEX)]

    statuses = _status_by_id(tmp_path)
    assert statuses[recoverable_id] == "acquired"
    assert statuses["sandbox-attachment-id"] == "unavailable"
    assert statuses["ghost-attachment-id"] == "unfetched"  # never touched

    conn = sqlite3.connect(tmp_path / "index.db")
    try:
        blob_hash, byte_count = conn.execute(
            "SELECT blob_hash, byte_count FROM attachments WHERE attachment_id = ?", (recoverable_id,)
        ).fetchone()
    finally:
        conn.close()
    assert blob_hash is not None
    assert bytes(blob_hash).hex() == __import__("hashlib").sha256(b"hello notes").hexdigest()
    assert byte_count == len(b"hello notes")
    assert blob_store.exists(bytes(blob_hash).hex())

    manifest_rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    actions = {row["attachment_id"]: row["action"] for row in manifest_rows}
    assert actions[recoverable_id] == "reacquired"
    assert actions["sandbox-attachment-id"] == "marked_unavailable"
    assert "ghost-attachment-id" not in actions


def test_apply_refuses_without_backup_manifest(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blob")
    index_conn = _index_conn(tmp_path / "index.db")
    source_conn = _source_conn(tmp_path / "source.db")
    _write_unfetched_extracted_content_attachment(tmp_path, index_conn, source_conn, blob_store)
    index_conn.close()
    source_conn.close()

    with pytest.raises(AttachmentReacquisitionError, match="backup manifest"):
        apply_attachment_reacquisition(
            tmp_path, manifest_path=tmp_path / "manifest.jsonl", backup_manifest=None, dry_run=False
        )

    statuses = _status_by_id(tmp_path)
    assert set(statuses.values()) == {"unfetched"}


def test_apply_refuses_without_manifest_path(tmp_path: Path) -> None:
    blob_store = BlobStore(tmp_path / "blob")
    index_conn = _index_conn(tmp_path / "index.db")
    source_conn = _source_conn(tmp_path / "source.db")
    _write_unfetched_extracted_content_attachment(tmp_path, index_conn, source_conn, blob_store)
    index_conn.close()
    source_conn.close()

    with pytest.raises(AttachmentReacquisitionError, match="manifest-path"):
        apply_attachment_reacquisition(
            tmp_path,
            manifest_path=None,
            backup_manifest=tmp_path / "verified-backup" / "manifest.json",
            dry_run=False,
        )

    statuses = _status_by_id(tmp_path)
    assert set(statuses.values()) == {"unfetched"}


def test_apply_refuses_when_backup_manifest_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blob_store = BlobStore(tmp_path / "blob")
    index_conn = _index_conn(tmp_path / "index.db")
    source_conn = _source_conn(tmp_path / "source.db")
    _write_unfetched_extracted_content_attachment(tmp_path, index_conn, source_conn, blob_store)
    index_conn.close()
    source_conn.close()

    def _reject(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        raise ValueError("backup manifest does not match live index.db")

    monkeypatch.setattr("polylogue.storage.attachment_reacquisition.validate_migration_backup_manifest", _reject)

    with pytest.raises(ValueError, match="does not match"):
        apply_attachment_reacquisition(
            tmp_path,
            manifest_path=tmp_path / "manifest.jsonl",
            backup_manifest=tmp_path / "stale-backup" / "manifest.json",
            dry_run=False,
        )

    statuses = _status_by_id(tmp_path)
    assert set(statuses.values()) == {"unfetched"}

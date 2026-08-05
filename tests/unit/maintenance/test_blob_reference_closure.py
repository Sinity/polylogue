"""Production-route proofs for acquired blob-reference closure repair."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import Provider
from polylogue.core.outcomes import OutcomeStatus
from polylogue.maintenance.archive_verification import ArchiveVerificationCheck
from polylogue.maintenance.blob_reference_closure import (
    BlobReferenceBlockerKind,
    BlobReferenceClosureError,
    plan_blob_reference_closure,
    reconcile_blob_reference_closure,
)
from polylogue.pipeline.services.ingest_worker import IngestRecordResult, SessionWritePayload, ingest_record
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession
from polylogue.storage.attachment_relink import RawSessionParser
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.runtime.raw.records import RawSessionRecord
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import _attachment_id, write_parsed_session_to_archive

_PAYLOAD = {
    "uuid": "closure-session-1",
    "name": "Closure test",
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


def _conn(path: Path, tier: ArchiveTier) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, tier)
    return conn


def _fixture(tmp_path: Path) -> tuple[Path, sqlite3.Connection, sqlite3.Connection, str]:
    root = tmp_path
    blob_store = BlobStore(root / "blob")
    source = _conn(root / "source.db", ArchiveTier.SOURCE)
    index = _conn(root / "index.db", ArchiveTier.INDEX)
    payload = json.dumps(_PAYLOAD).encode()
    blob_hash, blob_size = blob_store.write_from_bytes(payload)
    source.execute(
        "INSERT INTO raw_sessions (raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms) "
        "VALUES (?, 'claude-ai-export', ?, 0, ?, ?, 100)",
        ("raw-closure", "conversations.json", bytes.fromhex(blob_hash), blob_size),
    )
    source.commit()
    record = RawSessionRecord(
        raw_id="raw-closure",
        source_name=Provider.CLAUDE_AI.value,
        payload_provider=Provider.CLAUDE_AI,
        source_path="conversations.json",
        source_index=0,
        blob_size=blob_size,
        blob_hash=blob_hash,
        acquired_at="2026-01-01T00:00:00+00:00",
    )
    parsed = ingest_record(record, str(root), "advisory", blob_root_str=str(blob_store.root))
    assert parsed.error is None
    session = parsed.sessions[0]
    attachment = session.parsed_session.attachments[0]
    attachment_hash, attachment_size = blob_store.write_from_bytes(attachment.inline_bytes or b"")
    write_parsed_session_to_archive(
        index,
        session.parsed_session,
        raw_id=record.raw_id,
        preacquired_attachment_blobs={id(attachment): (bytes.fromhex(attachment_hash), attachment_size, "acquired")},
    )
    attachment_id = str(index.execute("SELECT attachment_id FROM attachments").fetchone()[0])
    index.execute("DELETE FROM attachment_refs WHERE attachment_id = ?", (attachment_id,))
    index.execute("UPDATE attachments SET ref_count = 0 WHERE attachment_id = ?", (attachment_id,))
    index.commit()
    return root, source, index, attachment_id


def _mapping_fixture(
    tmp_path: Path,
    *,
    messages: list[ParsedMessage],
    attachment: ParsedAttachment,
    append_only: bool = False,
    existing_messages: list[ParsedMessage] | None = None,
    orphan_attachment: bool = True,
) -> tuple[Path, str, str, RawSessionParser]:
    """Build a real closure archive with a parser result for one session."""
    root = tmp_path
    blob_store = BlobStore(root / "blob")
    source = _conn(root / "source.db", ArchiveTier.SOURCE)
    index = _conn(root / "index.db", ArchiveTier.INDEX)
    raw_bytes = b"closure mapping raw fixture"
    raw_hash, raw_size = blob_store.write_from_bytes(raw_bytes)
    source.execute(
        "INSERT INTO raw_sessions (raw_id, origin, source_path, source_index, blob_hash, blob_size, acquired_at_ms) "
        "VALUES ('raw-mapping', 'claude-ai-export', 'mapping.json', 0, ?, ?, 100)",
        (bytes.fromhex(raw_hash), raw_size),
    )
    source.commit()

    session = ParsedSession(
        source_name=Provider.CLAUDE_AI,
        provider_session_id="closure-mapping",
        title="Closure mapping",
        messages=messages,
        attachments=[attachment],
    )
    attachment_hash, attachment_size = blob_store.write_from_bytes(attachment.inline_bytes or b"")
    if append_only:
        if existing_messages is None:
            raise AssertionError("append fixtures require existing messages")
        write_parsed_session_to_archive(
            index,
            session.model_copy(update={"messages": existing_messages, "attachments": []}),
            raw_id="raw-mapping-base",
        )
    session_id = write_parsed_session_to_archive(
        index,
        session,
        raw_id="raw-mapping",
        merge_append=append_only,
        preacquired_attachment_blobs={id(attachment): (bytes.fromhex(attachment_hash), attachment_size, "acquired")},
    )
    attachment_id = _attachment_id(session_id, attachment)
    index.execute(
        """
        INSERT OR IGNORE INTO attachments (
            attachment_id, display_name, media_type, byte_count, blob_hash, acquisition_status, ref_count
        ) VALUES (?, ?, ?, ?, ?, 'acquired', 0)
        """,
        (
            attachment_id,
            attachment.name,
            attachment.mime_type,
            attachment_size,
            bytes.fromhex(attachment_hash),
        ),
    )
    if orphan_attachment:
        index.execute("DELETE FROM attachment_refs WHERE attachment_id = ?", (attachment_id,))
        index.execute("UPDATE attachments SET ref_count = 0 WHERE attachment_id = ?", (attachment_id,))
    index.commit()
    source.close()
    index.close()

    def parse(_raw_record: RawSessionRecord) -> IngestRecordResult:
        return IngestRecordResult(
            raw_id="raw-mapping",
            sessions=[
                SessionWritePayload(
                    session_id=session_id,
                    content_hash="mapping-fixture",
                    parsed_session=session,
                    append_only=append_only,
                )
            ],
        )

    return root, session_id, attachment_id, parse


def _accept_backup(manifest: Path, _tier: ArchiveTier, *, connection: sqlite3.Connection) -> Path:
    assert connection.execute("SELECT 1").fetchone() == (1,)
    return manifest


def _apply_mapping_fixture(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    parser: RawSessionParser,
) -> None:
    manifest = root / "verified-backup" / "manifest.json"
    receipt = root / "receipts" / "closure.jsonl"
    monkeypatch.setattr(
        "polylogue.maintenance.blob_reference_closure.validate_migration_backup_manifest", _accept_backup
    )
    monkeypatch.setattr(
        "polylogue.maintenance.blob_reference_closure.validate_backup_manifest_covers_derived_tier", _accept_backup
    )
    reconcile_blob_reference_closure(
        root,
        backup_manifest=manifest,
        receipt_path=receipt,
        dry_run=False,
        raw_session_parser=parser,
    )


def test_closure_repairs_idless_attachment_by_authoritative_message_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attachment = ParsedAttachment(
        provider_attachment_id="idless-position",
        message_position=1,
        name="position.txt",
        mime_type="text/plain",
        size_bytes=8,
        inline_bytes=b"position",
    )
    root, session_id, attachment_id, parser = _mapping_fixture(
        tmp_path,
        messages=[
            ParsedMessage(provider_message_id="first", role=Role.USER, text="first", position=0),
            ParsedMessage(provider_message_id="second", role=Role.ASSISTANT, text="second", position=1),
        ],
        attachment=attachment,
    )

    dry = reconcile_blob_reference_closure(root, raw_session_parser=parser)
    assert dry.applied is False
    assert dry.plan.attachment_candidates[0].message_id == f"{session_id}:second"
    with sqlite3.connect(root / "index.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM attachment_refs").fetchone()[0] == 0

    _apply_mapping_fixture(monkeypatch, root, parser)
    with sqlite3.connect(root / "index.db") as conn:
        ref = conn.execute(
            "SELECT message_id FROM attachment_refs WHERE attachment_id = ?", (attachment_id,)
        ).fetchone()
    assert ref == (f"{session_id}:second",)


def test_production_append_attaches_idless_message_at_max_position_plus_one(tmp_path: Path) -> None:
    attachment = ParsedAttachment(
        provider_attachment_id="append-position",
        message_position=0,
        name="append.txt",
        mime_type="text/plain",
        size_bytes=6,
        inline_bytes=b"append",
    )
    root, session_id, attachment_id, _parser = _mapping_fixture(
        tmp_path,
        messages=[ParsedMessage(provider_message_id="", role=Role.ASSISTANT, text="appended", position=0)],
        existing_messages=[ParsedMessage(provider_message_id="", role=Role.USER, text="older", position=0)],
        attachment=attachment,
        append_only=True,
        orphan_attachment=False,
    )

    with sqlite3.connect(root / "index.db") as conn:
        messages = conn.execute(
            "SELECT message_id, native_id, position FROM messages WHERE session_id = ? ORDER BY position",
            (session_id,),
        ).fetchall()
        assert messages == [
            (f"{session_id}:0.0", None, 0),
            (f"{session_id}:1.0", None, 1),
        ]
        ref = conn.execute(
            "SELECT message_id FROM attachment_refs WHERE attachment_id = ?", (attachment_id,)
        ).fetchone()
    assert ref == (f"{session_id}:1.0",)


def test_closure_relinks_idless_append_attachment_to_existing_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closure must reuse the already-materialized append tail position.

    The raw append payload still carries its relative position ``0``. The
    current index already contains that message at position ``1``. Applying
    ``MAX(position) + 1`` again would probe position ``2`` and incorrectly
    report the acquired attachment as unrecoverable.
    """
    attachment = ParsedAttachment(
        provider_attachment_id="append-orphan",
        message_position=0,
        name="append-orphan.txt",
        mime_type="text/plain",
        size_bytes=6,
        inline_bytes=b"append",
    )
    root, session_id, attachment_id, parser = _mapping_fixture(
        tmp_path,
        messages=[ParsedMessage(provider_message_id="", role=Role.ASSISTANT, text="appended", position=0)],
        existing_messages=[ParsedMessage(provider_message_id="", role=Role.USER, text="older", position=0)],
        attachment=attachment,
        append_only=True,
    )

    plan = plan_blob_reference_closure(root, raw_session_parser=parser)
    assert plan.attachment_candidates[0].attachment_id == attachment_id
    assert plan.attachment_candidates[0].message_id == f"{session_id}:1.0"

    _apply_mapping_fixture(monkeypatch, root, parser)
    with sqlite3.connect(root / "index.db") as conn:
        ref = conn.execute(
            "SELECT message_id FROM attachment_refs WHERE attachment_id = ?", (attachment_id,)
        ).fetchone()
    assert ref == (f"{session_id}:1.0",)


def test_closure_fails_closed_for_whitespace_duplicate_native_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attachment = ParsedAttachment(
        provider_attachment_id="duplicate-native-position",
        message_provider_id=" duplicate ",
        message_position=None,
        name="duplicate.txt",
        mime_type="text/plain",
        size_bytes=9,
        inline_bytes=b"duplicate",
    )
    root, session_id, attachment_id, parser = _mapping_fixture(
        tmp_path,
        messages=[
            ParsedMessage(provider_message_id="duplicate", role=Role.USER, text="first", position=0),
            ParsedMessage(provider_message_id=" duplicate ", role=Role.ASSISTANT, text="second", position=1),
        ],
        attachment=attachment,
    )

    plan = plan_blob_reference_closure(root, raw_session_parser=parser)
    assert not plan.attachment_candidates
    assert any(blocker.object_id == attachment_id for blocker in plan.blockers)

    _apply_mapping_fixture(monkeypatch, root, parser)
    with sqlite3.connect(root / "index.db") as conn:
        ref = conn.execute(
            "SELECT message_id FROM attachment_refs WHERE attachment_id = ?", (attachment_id,)
        ).fetchone()
    assert ref is None


def test_plan_is_complete_and_typed_for_deterministic_and_irreparable_rows(tmp_path: Path) -> None:
    root, source, index, attachment_id = _fixture(tmp_path)
    index.execute(
        "INSERT INTO attachments (attachment_id, display_name, byte_count, blob_hash, acquisition_status, ref_count) "
        "VALUES ('irreparable', 'ghost.bin', 5, ?, 'acquired', 0)",
        (b"g" * 32,),
    )
    index.commit()
    plan = plan_blob_reference_closure(root)
    assert [candidate.raw_id for candidate in plan.raw_candidates] == ["raw-closure"]
    assert [candidate.attachment_id for candidate in plan.attachment_candidates] == [attachment_id]
    assert any(
        blocker.object_id == "irreparable" and blocker.kind is BlobReferenceBlockerKind.ATTACHMENT_NO_AUTHORITATIVE_RAW
        for blocker in plan.blockers
    )
    source.close()
    index.close()


def test_plan_excludes_unfetched_orphans_from_closure_scope(tmp_path: Path) -> None:
    root, source, index, _attachment_id = _fixture(tmp_path)
    index.execute(
        "INSERT INTO attachments (attachment_id, display_name, byte_count, blob_hash, acquisition_status, ref_count) "
        "VALUES (?, ?, ?, ?, 'unfetched', 0)",
        ("f" * 64, "not acquired", 0, b"z" * 32),
    )
    index.commit()

    plan = plan_blob_reference_closure(root)

    assert plan.attachment_orphan_count == 1
    assert all(blocker.object_id != "f" * 64 for blocker in plan.blockers)
    assert all(candidate.attachment_id != "f" * 64 for candidate in plan.attachment_candidates)
    source.close()
    index.close()


def test_dry_run_does_not_write_and_apply_requires_both_tier_backup(tmp_path: Path) -> None:
    root, source, index, _attachment_id = _fixture(tmp_path)
    dry = reconcile_blob_reference_closure(root)
    assert dry.applied is False
    assert dry.plan.candidate_count == 2
    assert source.execute("SELECT COUNT(*) FROM blob_refs").fetchone()[0] == 0
    assert index.execute("SELECT COUNT(*) FROM attachment_refs").fetchone()[0] == 0
    with pytest.raises(BlobReferenceClosureError, match="backup manifest"):
        reconcile_blob_reference_closure(root, dry_run=False, receipt_path=tmp_path / "receipt.jsonl")
    source.close()
    index.close()


def test_apply_writes_only_exact_canonical_refs_and_is_receipted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, source, index, attachment_id = _fixture(tmp_path)
    manifest = tmp_path / "verified-backup" / "manifest.json"
    receipt = tmp_path / "receipts" / "closure.jsonl"
    validated: list[ArchiveTier] = []

    def accept(_manifest: Path, tier: ArchiveTier, *, connection: sqlite3.Connection) -> Path:
        assert connection.execute("SELECT 1").fetchone() == (1,)
        validated.append(tier)
        return _manifest

    monkeypatch.setattr("polylogue.maintenance.blob_reference_closure.validate_migration_backup_manifest", accept)
    monkeypatch.setattr(
        "polylogue.maintenance.blob_reference_closure.validate_backup_manifest_covers_derived_tier", accept
    )
    report = reconcile_blob_reference_closure(root, backup_manifest=manifest, receipt_path=receipt, dry_run=False)
    assert report.raw_repaired_count == 1
    assert report.attachment_repaired_count == 1
    assert validated == [ArchiveTier.SOURCE, ArchiveTier.INDEX]
    assert source.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_type = 'raw_payload'").fetchone()[0] == 1
    assert (
        index.execute("SELECT COUNT(*) FROM attachment_refs WHERE attachment_id = ?", (attachment_id,)).fetchone()[0]
        == 1
    )
    assert receipt.exists()
    source.close()
    index.close()


def test_integrity_check_fails_when_exact_raw_reference_is_tampered(tmp_path: Path) -> None:
    root, source, index, _attachment_id = _fixture(tmp_path)
    source.execute(
        "INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms) "
        "SELECT blob_hash, raw_id, 'raw_payload', source_path, blob_size, acquired_at_ms FROM raw_sessions"
    )
    source.execute(
        "INSERT INTO blob_refs (blob_hash, ref_id, ref_type, source_path, size_bytes, acquired_at_ms) "
        "SELECT ?, raw_id, 'raw_payload', source_path, blob_size, acquired_at_ms FROM raw_sessions",
        (b"x" * 32,),
    )
    source.commit()
    from polylogue.maintenance.archive_verification import verify_archive

    check = next(
        check
        for check in verify_archive(root, checks=("blob-reference-closure",)).checks
        if check.name == "blob-reference-closure"
    )
    assert isinstance(check, ArchiveVerificationCheck)
    assert check.status is OutcomeStatus.ERROR
    assert check.evidence["raw_missing_exact_count"] == 1
    source.close()
    index.close()

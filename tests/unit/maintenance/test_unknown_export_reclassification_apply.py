"""Real-storage tests for the ChatGPT unknown-export repair actuator."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.maintenance.unknown_export_reclassification_apply import (
    TOOL_VERSION,
    UnknownExportReclassificationApplyError,
    apply_unknown_export_reclassification,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _browser_capture_payload(provider: str, provider_session_id: str) -> bytes:
    return json.dumps(
        {
            "polylogue_capture_kind": "browser_llm_session",
            "schema_version": 1,
            "capture_id": f"{provider}:{provider_session_id}",
            "raw_provider_payload": {"padding": "x" * (1024 * 1024 + 64 * 1024)},
            "session": {
                "provider": provider,
                "provider_session_id": provider_session_id,
                "turns": [{"provider_turn_id": "u1", "role": "user", "text": "hi"}],
            },
        }
    ).encode()


def _write_unknown_raw(archive: ArchiveStore, *, raw_id: str, payload: bytes, source_path: str) -> None:
    archive.write_raw_payload(
        provider=Provider.UNKNOWN,
        payload=payload,
        source_path=source_path,
        source_index=-1,
        acquired_at_ms=1_700_000_000_000,
        raw_id=raw_id,
        revision=RawRevisionEnvelope(
            logical_source_key=f"unknown-export:{raw_id}",
            kind=RawRevisionKind.FULL,
            source_revision=f"{raw_id}-revision",
            acquisition_generation=0,
            authority=RawRevisionAuthority.QUARANTINED,
        ),
    )


def _build_archive(tmp_path: Path) -> Path:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        _write_unknown_raw(
            archive,
            raw_id="raw-chatgpt",
            payload=_browser_capture_payload("chatgpt", "chatgpt-session"),
            source_path="/spool/browser-capture/chatgpt/chatgpt.json",
        )
        _write_unknown_raw(
            archive,
            raw_id="raw-claude",
            payload=_browser_capture_payload("claude-ai", "claude-session"),
            source_path="/spool/browser-capture/chatgpt/wrong-provider.json",
        )
        _write_unknown_raw(
            archive,
            raw_id="raw-top-level-provider",
            payload=json.dumps({"provider": "chatgpt", "messages": []}).encode(),
            source_path="/spool/browser-capture/chatgpt/no-session-marker.json",
        )
        _write_unknown_raw(
            archive,
            raw_id="raw-outside-default-scope",
            payload=_browser_capture_payload("chatgpt", "outside-default-scope"),
            source_path="/spool/browser-capture/claude-ai/outside.json",
        )
        archive.commit()
    return archive_root


def _raw_rows(archive_root: Path) -> dict[str, tuple[str, str | None]]:
    with sqlite3.connect(archive_root / "source.db") as conn:
        return {
            str(raw_id): (str(origin), str(capture_mode) if capture_mode is not None else None)
            for raw_id, origin, capture_mode in conn.execute(
                "SELECT raw_id, origin, capture_mode FROM raw_sessions ORDER BY raw_id"
            )
        }


def _receipt_rows(archive_root: Path) -> list[tuple[object, ...]]:
    with sqlite3.connect(archive_root / "source.db") as conn:
        return conn.execute(
            """
            SELECT raw_id, previous_origin, new_origin, previous_capture_mode,
                   new_capture_mode, embedded_provider, source_path, blob_hash,
                   blob_size, tool_version, index_reparse_required, detail
            FROM raw_unknown_export_reclassification_receipts
            ORDER BY raw_id
            """
        ).fetchall()


def test_dry_run_uses_real_storage_and_mutates_nothing(tmp_path: Path) -> None:
    archive_root = _build_archive(tmp_path)
    before = _raw_rows(archive_root)
    index_before = (archive_root / "index.db").read_bytes()

    report = apply_unknown_export_reclassification(archive_root, dry_run=True)

    assert report.applied is False
    assert report.scanned_count == 3
    assert report.reclassifiable_count == 2
    assert report.chatgpt_reclassifiable_count == 1
    assert report.non_chatgpt_reclassifiable_count == 1
    assert report.still_unknown_count == 1
    assert report.blob_missing_count == 0
    assert report.reclassified_count == 1
    assert report.reclassified_raw_ids == ()
    assert report.index_reparse_required is True
    assert report.index_rows_touched == 0
    assert _raw_rows(archive_root) == before
    assert _receipt_rows(archive_root) == []
    assert (archive_root / "index.db").read_bytes() == index_before


def test_apply_reclassifies_only_chatgpt_and_records_identity_safe_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = _build_archive(tmp_path)
    index_before = (archive_root / "index.db").read_bytes()
    manifest = tmp_path / "verified-source-backup" / "manifest.json"
    validated: list[tuple[Path, object]] = []

    def _fake_validate(path: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        validated.append((path, tier))
        return path.with_name("verification-receipt.json")

    monkeypatch.setattr(
        "polylogue.maintenance.unknown_export_reclassification_apply.validate_migration_backup_manifest",
        _fake_validate,
    )

    report = apply_unknown_export_reclassification(archive_root, backup_manifest=manifest, dry_run=False)

    assert report.applied is True
    assert report.scanned_count == 3
    assert report.reclassifiable_count == 2
    assert report.chatgpt_reclassifiable_count == 1
    assert report.non_chatgpt_reclassifiable_count == 1
    assert report.reclassified_count == 1
    assert report.reclassified_raw_ids == ("raw-chatgpt",)
    assert report.backup_manifest == manifest
    assert report.index_reparse_required is True
    assert report.index_rows_touched == 0
    assert validated == [(manifest, ArchiveTier.SOURCE), (manifest, ArchiveTier.SOURCE)]

    rows = _raw_rows(archive_root)
    assert rows["raw-chatgpt"] == ("chatgpt-export", "chatgpt")
    assert rows["raw-claude"] == ("unknown-export", "unknown")
    assert rows["raw-top-level-provider"] == ("unknown-export", "unknown")
    assert rows["raw-outside-default-scope"] == ("unknown-export", "unknown")

    receipts = _receipt_rows(archive_root)
    assert len(receipts) == 1
    (
        raw_id,
        previous_origin,
        new_origin,
        previous_capture_mode,
        new_capture_mode,
        embedded_provider,
        source_path,
        blob_hash,
        blob_size,
        tool_version,
        index_reparse_required,
        detail,
    ) = receipts[0]
    assert (raw_id, previous_origin, new_origin) == ("raw-chatgpt", "unknown-export", "chatgpt-export")
    assert (previous_capture_mode, new_capture_mode, embedded_provider) == ("unknown", "chatgpt", "chatgpt")
    assert source_path == "/spool/browser-capture/chatgpt/chatgpt.json"
    assert len(cast(bytes, blob_hash)) == 32
    assert int(cast(int, blob_size)) > 1024 * 1024
    assert tool_version == TOOL_VERSION
    assert index_reparse_required == 1
    assert detail == "generated index identity deferred to normal reparse"
    assert (archive_root / "index.db").read_bytes() == index_before


def test_apply_requires_verified_backup_before_mutation(tmp_path: Path) -> None:
    archive_root = _build_archive(tmp_path)
    before = _raw_rows(archive_root)

    with pytest.raises(UnknownExportReclassificationApplyError, match="backup manifest"):
        apply_unknown_export_reclassification(archive_root, backup_manifest=None, dry_run=False)

    assert _raw_rows(archive_root) == before
    assert _receipt_rows(archive_root) == []


def test_apply_can_scan_beyond_default_path_but_still_requires_chatgpt_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = _build_archive(tmp_path)
    manifest = tmp_path / "verified-source-backup" / "manifest.json"

    monkeypatch.setattr(
        "polylogue.maintenance.unknown_export_reclassification_apply.validate_migration_backup_manifest",
        lambda path, tier, *, connection: path,
    )
    report = apply_unknown_export_reclassification(
        archive_root,
        backup_manifest=manifest,
        source_path_like=None,
        dry_run=False,
    )

    assert report.scanned_count == 4
    assert report.chatgpt_reclassifiable_count == 2
    assert report.reclassified_raw_ids == ("raw-chatgpt", "raw-outside-default-scope")
    assert _raw_rows(archive_root)["raw-claude"] == ("unknown-export", "unknown")

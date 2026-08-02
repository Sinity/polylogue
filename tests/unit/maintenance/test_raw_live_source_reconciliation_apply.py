"""polylogue-u19l: the actuator that acts on live-source-verification classification.

Builds a fixture archive with one row of each classification
(exact_match / codex_header_strip_match / diverged / source_missing) and
proves:

* dry-run (the default) never mutates anything;
* --apply promotes exactly the two safe verdicts (exact_match,
  codex_header_strip_match) to revision_authority='byte_proven' with
  revision_authority_evidence='live_source_verification_v1', and writes an
  immutable receipt for each;
* diverged and source_missing rows are left untouched, no exceptions;
* applying without a backup manifest is refused before anything is touched.
"""

from __future__ import annotations

import sqlite3
from json import dumps as json_dumps
from pathlib import Path

import pytest

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.maintenance.raw_live_source_reconciliation_apply import (
    TOOL_VERSION,
    RawLiveSourceReconciliationApplyError,
    apply_raw_live_source_reconciliation,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _write_quarantined_raw(
    archive: ArchiveStore,
    *,
    raw_id: str,
    provider: Provider,
    payload: bytes,
    source_path: str,
    kind: RawRevisionKind,
    logical_source_key: str,
    append_start_offset: int | None = None,
    append_end_offset: int | None = None,
) -> None:
    predecessor_source_revision = f"{raw_id}-predecessor" if kind is RawRevisionKind.APPEND else None
    archive.write_raw_payload(
        provider=provider,
        payload=payload,
        source_path=source_path,
        source_index=-1,
        acquired_at_ms=1_700_000_000_000,
        raw_id=raw_id,
        revision=RawRevisionEnvelope(
            logical_source_key=logical_source_key,
            kind=kind,
            source_revision=f"{raw_id}-revision",
            acquisition_generation=0,
            predecessor_source_revision=predecessor_source_revision,
            append_start_offset=append_start_offset,
            append_end_offset=append_end_offset,
            authority=RawRevisionAuthority.QUARANTINED,
        ),
    )


def _build_fixture_archive(tmp_path: Path) -> Path:
    """One row per verdict: exact_match, codex_header_strip_match, diverged, source_missing."""
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    exact_live = tmp_path / "exact.jsonl"
    exact_live.write_bytes(b'{"a":1}\n')

    diverged_live = tmp_path / "diverged.jsonl"
    diverged_live.write_bytes(b'{"a":"CHANGED"}\n')

    missing_live_path = str(tmp_path / "gone.jsonl")

    codex_live = tmp_path / "rollout-header-strip.jsonl"
    codex_prefix = b'{"type":"session_meta","payload":{"id":"abc"}}\n'
    codex_delta = b'{"type":"response_item","payload":{"id":"m1"}}\n'
    codex_live.write_bytes(codex_prefix + codex_delta)
    codex_synthetic_header = json_dumps(
        {"type": "session_meta", "payload": {"id": "abc"}}, separators=(",", ":")
    ).encode()
    codex_archived_payload = codex_synthetic_header + b"\n" + codex_delta

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        _write_quarantined_raw(
            archive,
            raw_id="raw-exact",
            provider=Provider.CLAUDE_CODE,
            payload=b'{"a":1}\n',
            source_path=str(exact_live),
            kind=RawRevisionKind.FULL,
            logical_source_key="claude-code:exact",
        )
        _write_quarantined_raw(
            archive,
            raw_id="raw-diverged",
            provider=Provider.CLAUDE_CODE,
            payload=b'{"a":1}\n',
            source_path=str(diverged_live),
            kind=RawRevisionKind.FULL,
            logical_source_key="claude-code:diverged",
        )
        _write_quarantined_raw(
            archive,
            raw_id="raw-missing",
            provider=Provider.CLAUDE_CODE,
            payload=b'{"a":1}\n',
            source_path=missing_live_path,
            kind=RawRevisionKind.FULL,
            logical_source_key="claude-code:missing",
        )
        _write_quarantined_raw(
            archive,
            raw_id="raw-codex-header-strip",
            provider=Provider.CODEX,
            payload=codex_archived_payload,
            source_path=str(codex_live),
            kind=RawRevisionKind.APPEND,
            logical_source_key="codex:header-strip",
            append_start_offset=len(codex_prefix),
            append_end_offset=len(codex_prefix) + len(codex_delta),
        )
        archive.commit()

    return archive_root


def _revision_authority_rows(archive_root: Path) -> dict[str, tuple[str, str | None]]:
    conn = sqlite3.connect(archive_root / "source.db")
    try:
        rows = conn.execute(
            "SELECT raw_id, revision_authority, revision_authority_evidence FROM raw_sessions"
        ).fetchall()
    finally:
        conn.close()
    return {raw_id: (authority, evidence) for raw_id, authority, evidence in rows}


def _receipt_rows(archive_root: Path) -> dict[str, tuple[str, str]]:
    conn = sqlite3.connect(archive_root / "source.db")
    try:
        rows = conn.execute(
            "SELECT raw_id, verdict, tool_version FROM raw_live_source_reconciliation_receipts"
        ).fetchall()
    finally:
        conn.close()
    return {raw_id: (verdict, tool_version) for raw_id, verdict, tool_version in rows}


def test_dry_run_makes_zero_mutations(tmp_path: Path) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before = _revision_authority_rows(archive_root)
    assert all(authority == "quarantined" for authority, _evidence in before.values())

    report = apply_raw_live_source_reconciliation(archive_root, dry_run=True)

    assert report.applied is False
    assert report.scanned_count == 4
    assert report.promoted_count == 2  # raw-exact + raw-codex-header-strip
    assert report.codex_header_strip_match_count == 1
    assert report.diverged_count == 1
    assert report.source_missing_count == 1
    assert report.promoted_raw_ids == ()

    after = _revision_authority_rows(archive_root)
    assert after == before  # zero mutation
    assert _receipt_rows(archive_root) == {}


def test_apply_promotes_only_safe_verdicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = _build_fixture_archive(tmp_path)

    validated: list[tuple[Path, object]] = []

    def _fake_validate(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        validated.append((manifest, tier))
        assert connection.execute("SELECT 1").fetchone() == (1,)
        return manifest.with_name("verification-receipt.json")

    monkeypatch.setattr(
        "polylogue.maintenance.raw_live_source_reconciliation_apply.validate_migration_backup_manifest",
        _fake_validate,
    )

    manifest = tmp_path / "verified-backup" / "manifest.json"
    report = apply_raw_live_source_reconciliation(archive_root, backup_manifest=manifest, dry_run=False)

    assert report.applied is True
    assert report.scanned_count == 4
    assert report.promoted_count == 2
    assert set(report.promoted_raw_ids) == {"raw-exact", "raw-codex-header-strip"}
    assert report.codex_header_strip_match_count == 1
    assert report.diverged_count == 1
    assert report.source_missing_count == 1
    assert report.backup_manifest == manifest
    # Validated twice: lock-free precheck, then authoritative re-validation
    # once the write lock is held (mirrors migrate_archive_tier's pattern).
    assert validated == [(manifest, ArchiveTier.SOURCE), (manifest, ArchiveTier.SOURCE)]

    rows = _revision_authority_rows(archive_root)
    assert rows["raw-exact"] == ("byte_proven", "live_source_verification_v1")
    assert rows["raw-codex-header-strip"] == ("byte_proven", "live_source_verification_v1")
    # Never touched.
    assert rows["raw-diverged"] == ("quarantined", None)
    assert rows["raw-missing"] == ("quarantined", None)

    receipts = _receipt_rows(archive_root)
    assert receipts["raw-exact"] == ("exact_match", TOOL_VERSION)
    assert receipts["raw-codex-header-strip"] == ("codex_header_strip_match", TOOL_VERSION)
    assert set(receipts) == {"raw-exact", "raw-codex-header-strip"}


def test_apply_refuses_without_backup_manifest(tmp_path: Path) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before = _revision_authority_rows(archive_root)

    with pytest.raises(RawLiveSourceReconciliationApplyError, match="backup manifest"):
        apply_raw_live_source_reconciliation(archive_root, backup_manifest=None, dry_run=False)

    assert _revision_authority_rows(archive_root) == before
    assert _receipt_rows(archive_root) == {}


def test_apply_refuses_when_backup_manifest_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A backup manifest that fails validation (stale/missing/wrong tier) must
    refuse before any mutation -- not merely log a warning."""
    archive_root = _build_fixture_archive(tmp_path)
    before = _revision_authority_rows(archive_root)

    def _reject(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        raise ValueError("backup manifest does not match live source.db")

    monkeypatch.setattr(
        "polylogue.maintenance.raw_live_source_reconciliation_apply.validate_migration_backup_manifest",
        _reject,
    )

    manifest = tmp_path / "stale-backup" / "manifest.json"
    with pytest.raises(ValueError, match="does not match"):
        apply_raw_live_source_reconciliation(archive_root, backup_manifest=manifest, dry_run=False)

    assert _revision_authority_rows(archive_root) == before
    assert _receipt_rows(archive_root) == {}

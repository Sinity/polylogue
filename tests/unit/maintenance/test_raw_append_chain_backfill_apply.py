"""polylogue-lb39z (Phase 1, item 3): the actuator for membershipless append backfill.

Builds a fixture archive with one row of each classification (exact_match /
diverged / source_missing) plus a control row that already has a membership
row, and proves:

* dry-run (the default) never mutates anything;
* --apply promotes only the exact_match row to revision_authority=
  'byte_proven' with revision_authority_evidence='live_source_verification_v1',
  and writes an immutable receipt;
* diverged and source_missing rows are left untouched, no exceptions;
* a row that already has a raw_session_memberships row is never touched here
  even when its bytes match exactly -- proving the NOT EXISTS scope is
  respected end-to-end through the actuator, not just the classifier;
* applying without a backup manifest is refused before anything is touched;
* predecessor_raw_id / baseline_raw_id / acquisition_generation are never
  touched -- that revision-graph linkage belongs to classify_raw_revision_cohort
  / _promote_contiguous_append_evidence, exactly like polylogue-u19l's actuator.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.maintenance.raw_append_chain_backfill_apply import (
    TOOL_VERSION,
    RawAppendChainBackfillApplyError,
    apply_raw_append_chain_backfill,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_PREFIX = b"PREFIX-BYTES"


def _write_append_raw(
    archive: ArchiveStore,
    *,
    raw_id: str,
    payload: bytes,
    source_path: str,
    logical_source_key: str,
    start_offset: int,
    end_offset: int,
) -> None:
    archive.write_raw_payload(
        provider=Provider.CLAUDE_CODE,
        payload=payload,
        source_path=source_path,
        source_index=-1,
        acquired_at_ms=1_700_000_000_000,
        raw_id=raw_id,
        revision=RawRevisionEnvelope(
            logical_source_key=logical_source_key,
            kind=RawRevisionKind.APPEND,
            source_revision=f"{raw_id}-revision",
            acquisition_generation=0,
            predecessor_source_revision=f"{raw_id}-predecessor",
            append_start_offset=start_offset,
            append_end_offset=end_offset,
            authority=RawRevisionAuthority.QUARANTINED,
        ),
    )


def _write_membership(conn: sqlite3.Connection, *, raw_id: str, logical_source_key: str) -> None:
    conn.execute(
        """
        INSERT INTO raw_session_memberships (
            raw_id, logical_source_key, provider_session_id, source_revision,
            normalized_content_hash, message_count, revision_authority
        ) VALUES (?, ?, 'session-1', 'rev-1', ?, 1, 'quarantined')
        """,
        (raw_id, logical_source_key, b"\x00" * 32),
    )


def _build_fixture_archive(tmp_path: Path) -> Path:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    delta = b'{"delta":1}\n'

    exact_live = tmp_path / "exact.jsonl"
    exact_live.write_bytes(_PREFIX + delta)

    diverged_live = tmp_path / "diverged.jsonl"
    diverged_live.write_bytes(_PREFIX + b'{"CHANGED":true}\n')

    missing_live_path = str(tmp_path / "gone.jsonl")

    has_membership_live = tmp_path / "has-membership.jsonl"
    has_membership_live.write_bytes(_PREFIX + delta)

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        _write_append_raw(
            archive,
            raw_id="raw-exact",
            payload=delta,
            source_path=str(exact_live),
            logical_source_key="claude-code:exact",
            start_offset=len(_PREFIX),
            end_offset=len(_PREFIX) + len(delta),
        )
        _write_append_raw(
            archive,
            raw_id="raw-diverged",
            payload=delta,
            source_path=str(diverged_live),
            logical_source_key="claude-code:diverged",
            start_offset=len(_PREFIX),
            end_offset=len(_PREFIX) + len(delta),
        )
        _write_append_raw(
            archive,
            raw_id="raw-missing",
            payload=delta,
            source_path=missing_live_path,
            logical_source_key="claude-code:missing",
            start_offset=0,
            end_offset=len(delta),
        )
        _write_append_raw(
            archive,
            raw_id="raw-has-membership",
            payload=delta,
            source_path=str(has_membership_live),
            logical_source_key="claude-code:has-membership",
            start_offset=len(_PREFIX),
            end_offset=len(_PREFIX) + len(delta),
        )
        archive.commit()

    conn = sqlite3.connect(archive_root / "source.db")
    try:
        _write_membership(conn, raw_id="raw-has-membership", logical_source_key="claude-code:has-membership")
        conn.commit()
    finally:
        conn.close()

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


def _linkage_rows(archive_root: Path) -> dict[str, tuple[object, object, object]]:
    conn = sqlite3.connect(archive_root / "source.db")
    try:
        rows = conn.execute(
            "SELECT raw_id, predecessor_raw_id, baseline_raw_id, acquisition_generation FROM raw_sessions"
        ).fetchall()
    finally:
        conn.close()
    return {raw_id: (pred, baseline, gen) for raw_id, pred, baseline, gen in rows}


def _receipt_rows(archive_root: Path) -> dict[str, str]:
    conn = sqlite3.connect(archive_root / "source.db")
    try:
        rows = conn.execute("SELECT raw_id, tool_version FROM raw_append_chain_backfill_receipts").fetchall()
    finally:
        conn.close()
    return dict(rows)


def test_dry_run_makes_zero_mutations(tmp_path: Path) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before = _revision_authority_rows(archive_root)
    assert all(authority == "quarantined" for authority, _evidence in before.values())

    report = apply_raw_append_chain_backfill(archive_root, dry_run=True)

    assert report.applied is False
    assert report.scanned_count == 3  # exact, diverged, missing -- has-membership excluded
    assert report.promoted_count == 1
    assert report.diverged_count == 1
    assert report.source_missing_count == 1
    assert report.promoted_raw_ids == ()

    assert _revision_authority_rows(archive_root) == before
    assert _receipt_rows(archive_root) == {}


def test_apply_promotes_only_membershipless_exact_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = _build_fixture_archive(tmp_path)

    validated: list[tuple[Path, object]] = []

    def _fake_validate(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        validated.append((manifest, tier))
        assert connection.execute("SELECT 1").fetchone() == (1,)
        return manifest.with_name("verification-receipt.json")

    monkeypatch.setattr(
        "polylogue.maintenance.raw_append_chain_backfill_apply.validate_migration_backup_manifest",
        _fake_validate,
    )

    linkage_before = _linkage_rows(archive_root)

    manifest = tmp_path / "verified-backup" / "manifest.json"
    report = apply_raw_append_chain_backfill(archive_root, backup_manifest=manifest, dry_run=False)

    assert report.applied is True
    assert report.scanned_count == 3
    assert report.promoted_count == 1
    assert set(report.promoted_raw_ids) == {"raw-exact"}
    assert report.backup_manifest == manifest
    assert validated == [(manifest, ArchiveTier.SOURCE), (manifest, ArchiveTier.SOURCE)]

    rows = _revision_authority_rows(archive_root)
    assert rows["raw-exact"] == ("byte_proven", "live_source_verification_v1")
    # Never touched.
    assert rows["raw-diverged"] == ("quarantined", None)
    assert rows["raw-missing"] == ("quarantined", None)
    # Has a membership row -- out of this actuator's scope even though its
    # bytes match exactly.
    assert rows["raw-has-membership"] == ("quarantined", None)

    receipts = _receipt_rows(archive_root)
    assert receipts == {"raw-exact": TOOL_VERSION}

    # Revision-graph linkage is untouched -- classify_raw_revision_cohort /
    # _promote_contiguous_append_evidence own that.
    linkage_after = _linkage_rows(archive_root)
    assert linkage_after["raw-exact"] == linkage_before["raw-exact"]


def test_apply_refuses_without_backup_manifest(tmp_path: Path) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before = _revision_authority_rows(archive_root)

    with pytest.raises(RawAppendChainBackfillApplyError, match="backup manifest"):
        apply_raw_append_chain_backfill(archive_root, backup_manifest=None, dry_run=False)

    assert _revision_authority_rows(archive_root) == before
    assert _receipt_rows(archive_root) == {}


def test_apply_refuses_when_backup_manifest_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before = _revision_authority_rows(archive_root)

    def _reject(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        raise ValueError("backup manifest does not match live source.db")

    monkeypatch.setattr(
        "polylogue.maintenance.raw_append_chain_backfill_apply.validate_migration_backup_manifest",
        _reject,
    )

    manifest = tmp_path / "stale-backup" / "manifest.json"
    with pytest.raises(ValueError, match="does not match"):
        apply_raw_append_chain_backfill(archive_root, backup_manifest=manifest, dry_run=False)

    assert _revision_authority_rows(archive_root) == before
    assert _receipt_rows(archive_root) == {}

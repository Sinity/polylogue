"""polylogue-lb39z (Phase 1, item 2): write back already-decided membership verdicts.

Builds a fixture archive with quarantined raw_sessions rows covering every
raw_session_memberships.decision value, and proves:

* dry-run (the default) never mutates anything;
* --apply promotes exactly the write-back-eligible decisions (applied,
  superseded_equivalent, superseded_prefix) with membership
  revision_authority='byte_proven' to raw_sessions.revision_authority=
  'byte_proven', and writes an immutable receipt for each;
* ambiguous, deferred, NULL-decision, and quarantined-membership-authority
  rows are left untouched, no exceptions -- this module transfers only
  already-decided facts, never new judgment;
* applying without a backup manifest is refused before anything is touched;
* predecessor_raw_id / baseline_raw_id / acquisition_generation /
  revision_authority_evidence are never touched by this actuator (that
  linkage belongs to classify_raw_revision_cohort and the live-source-
  verification actuator respectively).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.maintenance.raw_membership_writeback_apply import (
    TOOL_VERSION,
    RawMembershipWriteBackApplyError,
    apply_raw_membership_writeback,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _write_quarantined_raw(archive: ArchiveStore, *, raw_id: str, logical_source_key: str) -> None:
    archive.write_raw_payload(
        provider=Provider.CLAUDE_CODE,
        payload=f"{{raw_id}}: {raw_id}".encode(),
        source_path=f"/capture/{raw_id}.jsonl",
        source_index=-1,
        acquired_at_ms=1_700_000_000_000,
        raw_id=raw_id,
        revision=RawRevisionEnvelope(
            logical_source_key=logical_source_key,
            kind=RawRevisionKind.FULL,
            source_revision=f"{raw_id}-revision",
            acquisition_generation=0,
            authority=RawRevisionAuthority.QUARANTINED,
        ),
    )


def _write_membership(
    conn: sqlite3.Connection,
    *,
    raw_id: str,
    logical_source_key: str,
    decision: str | None,
    membership_authority: str,
    provider_session_id: str = "session-1",
) -> None:
    conn.execute(
        """
        INSERT INTO raw_session_memberships (
            raw_id, logical_source_key, provider_session_id, source_revision,
            normalized_content_hash, message_count, revision_authority,
            decision, decided_at_ms
        ) VALUES (?, ?, ?, 'rev-1', ?, 1, ?, ?, ?)
        """,
        (
            raw_id,
            logical_source_key,
            provider_session_id,
            b"\x00" * 32,
            membership_authority,
            decision,
            1_700_000_000_000 if decision is not None else None,
        ),
    )


def _build_fixture_archive(tmp_path: Path) -> Path:
    """One quarantined raw per membership decision (plus a no-membership control)."""
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for raw_id in (
            "raw-applied",
            "raw-superseded-equivalent",
            "raw-superseded-prefix",
            "raw-ambiguous",
            "raw-deferred",
            "raw-no-membership",
            "raw-decided-but-membership-quarantined",
        ):
            _write_quarantined_raw(archive, raw_id=raw_id, logical_source_key=f"claude-code:{raw_id}")
        archive.commit()

    conn = sqlite3.connect(archive_root / "source.db")
    try:
        _write_membership(
            conn,
            raw_id="raw-applied",
            logical_source_key="claude-code:raw-applied",
            decision="applied",
            membership_authority="byte_proven",
        )
        _write_membership(
            conn,
            raw_id="raw-superseded-equivalent",
            logical_source_key="claude-code:raw-superseded-equivalent",
            decision="superseded_equivalent",
            membership_authority="byte_proven",
        )
        _write_membership(
            conn,
            raw_id="raw-superseded-prefix",
            logical_source_key="claude-code:raw-superseded-prefix",
            decision="superseded_prefix",
            membership_authority="byte_proven",
        )
        _write_membership(
            conn,
            raw_id="raw-ambiguous",
            logical_source_key="claude-code:raw-ambiguous",
            decision="ambiguous",
            membership_authority="byte_proven",
        )
        _write_membership(
            conn,
            raw_id="raw-deferred",
            logical_source_key="claude-code:raw-deferred",
            decision="deferred",
            membership_authority="quarantined",
        )
        # raw-no-membership: no raw_session_memberships row at all (item 3's population).
        _write_membership(
            conn,
            raw_id="raw-decided-but-membership-quarantined",
            logical_source_key="claude-code:raw-decided-but-membership-quarantined",
            decision="applied",
            membership_authority="quarantined",  # membership's OWN authority is unproven -- must not write back
        )
        conn.commit()
    finally:
        conn.close()

    return archive_root


def _revision_authority_rows(archive_root: Path) -> dict[str, str]:
    conn = sqlite3.connect(archive_root / "source.db")
    try:
        rows = conn.execute("SELECT raw_id, revision_authority FROM raw_sessions").fetchall()
    finally:
        conn.close()
    return dict(rows)


def _linkage_rows(archive_root: Path) -> dict[str, tuple[object, object, object, object]]:
    conn = sqlite3.connect(archive_root / "source.db")
    try:
        rows = conn.execute(
            "SELECT raw_id, predecessor_raw_id, baseline_raw_id, acquisition_generation, revision_authority_evidence "
            "FROM raw_sessions"
        ).fetchall()
    finally:
        conn.close()
    return {raw_id: (pred, baseline, gen, evidence) for raw_id, pred, baseline, gen, evidence in rows}


def _receipt_rows(archive_root: Path) -> dict[str, tuple[str, str]]:
    conn = sqlite3.connect(archive_root / "source.db")
    try:
        rows = conn.execute(
            "SELECT raw_id, membership_decision, tool_version FROM raw_membership_writeback_receipts"
        ).fetchall()
    finally:
        conn.close()
    return {raw_id: (decision, tool_version) for raw_id, decision, tool_version in rows}


def test_dry_run_makes_zero_mutations(tmp_path: Path) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before = _revision_authority_rows(archive_root)
    assert all(authority == "quarantined" for authority in before.values())

    report = apply_raw_membership_writeback(archive_root, dry_run=True)

    assert report.applied is False
    assert report.scanned_count == 7
    assert report.promoted_count == 3  # applied, superseded_equivalent, superseded_prefix
    assert report.promoted_raw_ids == ()

    assert _revision_authority_rows(archive_root) == before  # zero mutation
    assert _receipt_rows(archive_root) == {}


def test_apply_promotes_only_decided_byte_proven_membership_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = _build_fixture_archive(tmp_path)

    validated: list[tuple[Path, object]] = []

    def _fake_validate(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        validated.append((manifest, tier))
        assert connection.execute("SELECT 1").fetchone() == (1,)
        return manifest.with_name("verification-receipt.json")

    monkeypatch.setattr(
        "polylogue.maintenance.raw_membership_writeback_apply.validate_migration_backup_manifest",
        _fake_validate,
    )

    linkage_before = _linkage_rows(archive_root)

    manifest = tmp_path / "verified-backup" / "manifest.json"
    report = apply_raw_membership_writeback(archive_root, backup_manifest=manifest, dry_run=False)

    assert report.applied is True
    assert report.scanned_count == 7
    assert report.promoted_count == 3
    assert set(report.promoted_raw_ids) == {"raw-applied", "raw-superseded-equivalent", "raw-superseded-prefix"}
    assert report.backup_manifest == manifest
    # Validated twice: lock-free precheck, then authoritative re-validation
    # once the write lock is held (mirrors migrate_archive_tier's pattern).
    assert validated == [(manifest, ArchiveTier.SOURCE), (manifest, ArchiveTier.SOURCE)]

    rows = _revision_authority_rows(archive_root)
    assert rows["raw-applied"] == "byte_proven"
    assert rows["raw-superseded-equivalent"] == "byte_proven"
    assert rows["raw-superseded-prefix"] == "byte_proven"
    # Never touched: real ambiguity, not-yet-decided, no membership evidence
    # at all, and a decision whose OWN membership authority is unproven.
    assert rows["raw-ambiguous"] == "quarantined"
    assert rows["raw-deferred"] == "quarantined"
    assert rows["raw-no-membership"] == "quarantined"
    assert rows["raw-decided-but-membership-quarantined"] == "quarantined"

    receipts = _receipt_rows(archive_root)
    assert receipts["raw-applied"] == ("applied", TOOL_VERSION)
    assert receipts["raw-superseded-equivalent"] == ("superseded_equivalent", TOOL_VERSION)
    assert receipts["raw-superseded-prefix"] == ("superseded_prefix", TOOL_VERSION)
    assert set(receipts) == {"raw-applied", "raw-superseded-equivalent", "raw-superseded-prefix"}

    # This actuator transfers only revision_authority -- never revision-graph
    # linkage (owned by classify_raw_revision_cohort) or the
    # live-source-verification evidence column.
    linkage_after = _linkage_rows(archive_root)
    for raw_id in ("raw-applied", "raw-superseded-equivalent", "raw-superseded-prefix"):
        pred_before, baseline_before, gen_before, evidence_before = linkage_before[raw_id]
        pred_after, baseline_after, gen_after, evidence_after = linkage_after[raw_id]
        assert (pred_after, baseline_after, gen_after) == (pred_before, baseline_before, gen_before)
        assert evidence_after == evidence_before is None


def test_apply_refuses_without_backup_manifest(tmp_path: Path) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before = _revision_authority_rows(archive_root)

    with pytest.raises(RawMembershipWriteBackApplyError, match="backup manifest"):
        apply_raw_membership_writeback(archive_root, backup_manifest=None, dry_run=False)

    assert _revision_authority_rows(archive_root) == before
    assert _receipt_rows(archive_root) == {}


def test_apply_refuses_when_backup_manifest_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before = _revision_authority_rows(archive_root)

    def _reject(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        raise ValueError("backup manifest does not match live source.db")

    monkeypatch.setattr(
        "polylogue.maintenance.raw_membership_writeback_apply.validate_migration_backup_manifest",
        _reject,
    )

    manifest = tmp_path / "stale-backup" / "manifest.json"
    with pytest.raises(ValueError, match="does not match"):
        apply_raw_membership_writeback(archive_root, backup_manifest=manifest, dry_run=False)

    assert _revision_authority_rows(archive_root) == before
    assert _receipt_rows(archive_root) == {}

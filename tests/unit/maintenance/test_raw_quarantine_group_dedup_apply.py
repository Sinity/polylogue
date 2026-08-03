"""polylogue-zm4w8: the actuator for fully-quarantined byte-identical group dedup.

Builds a fixture archive with a genuine same-source_path, byte-identical
quarantined group (never indexed anywhere) plus a genuine singleton
(non-grouped) quarantined raw, and proves:

* dry-run (the default) never mutates anything, and never runs the ingest
  pipeline;
* --apply materializes exactly one representative raw per group through the
  real production ingest pipeline (it becomes a genuine ``sessions`` row in
  index.db), marks the rest ``revision_authority='byte_proven'``, and writes
  an immutable receipt naming the representative raw_id/session_id;
* the singleton (non-grouped) row is left untouched;
* applying without a backup manifest is refused before anything is touched;
* applying with an invalid/stale backup manifest is refused before the
  ingest pipeline ever runs.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from polylogue.core.enums import Provider
from polylogue.maintenance.raw_quarantine_group_dedup_apply import (
    TOOL_VERSION,
    RawQuarantineGroupDedupApplyError,
    _checkpoint_live_tier,
    _mark_group_duplicates,
    apply_raw_quarantine_group_dedup,
)
from polylogue.storage.raw_quarantine_group_dedup import RawQuarantineGroup
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _codex_raw_payload(session_id: str, *, text: str) -> bytes:
    """A minimal, genuinely-parseable single-session Codex JSONL raw."""
    session_meta = json.dumps(
        {"type": "session_meta", "payload": {"id": session_id, "timestamp": "2026-06-01T00:00:00Z"}},
        separators=(",", ":"),
    )
    response_item = json.dumps(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "one",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        },
        separators=(",", ":"),
    )
    return (session_meta + "\n" + response_item + "\n").encode()


def _write_quarantined_raw(archive: ArchiveStore, *, raw_id: str, payload: bytes, source_path: str) -> None:
    archive.write_raw_payload(
        provider=Provider.CODEX,
        payload=payload,
        source_path=source_path,
        source_index=-1,
        acquired_at_ms=1_700_000_000_000,
        raw_id=raw_id,
    )


_GROUP_PAYLOAD = _codex_raw_payload("zm4w8-repeated-session", text="repeated content")
_SINGLETON_PAYLOAD = _codex_raw_payload("zm4w8-singleton-session", text="singleton content")


def _build_fixture_archive(tmp_path: Path) -> Path:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        # Three separate acquisitions of the same source file -- the
        # population this actuator resolves.
        for raw_id in ("raw-a", "raw-b", "raw-c"):
            _write_quarantined_raw(
                archive, raw_id=raw_id, payload=_GROUP_PAYLOAD, source_path=str(tmp_path / "repeated.jsonl")
            )
        # A genuine singleton -- never grouped, must never be touched.
        _write_quarantined_raw(
            archive, raw_id="raw-singleton", payload=_SINGLETON_PAYLOAD, source_path=str(tmp_path / "singleton.jsonl")
        )
        archive.commit()

    return archive_root


def _revision_authority_rows(archive_root: Path) -> dict[str, str]:
    conn = sqlite3.connect(archive_root / "source.db")
    try:
        rows = conn.execute("SELECT raw_id, revision_authority FROM raw_sessions").fetchall()
    finally:
        conn.close()
    return dict(rows)


def _receipt_rows(archive_root: Path) -> dict[str, tuple[str, str, str]]:
    conn = sqlite3.connect(archive_root / "source.db")
    try:
        rows = conn.execute(
            "SELECT raw_id, representative_raw_id, representative_session_id, tool_version "
            "FROM raw_quarantine_group_dedup_receipts"
        ).fetchall()
    finally:
        conn.close()
    return {
        raw_id: (rep_raw_id, rep_session_id, tool_version) for raw_id, rep_raw_id, rep_session_id, tool_version in rows
    }


def _index_sessions_rows(archive_root: Path) -> list[tuple[str, str | None]]:
    conn = sqlite3.connect(archive_root / "index.db")
    try:
        rows = conn.execute("SELECT session_id, raw_id FROM sessions ORDER BY session_id").fetchall()
    finally:
        conn.close()
    return [(session_id, raw_id) for session_id, raw_id in rows]


async def test_dry_run_makes_zero_mutations(tmp_path: Path) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before_source = _revision_authority_rows(archive_root)
    before_index = _index_sessions_rows(archive_root)
    assert all(authority == "quarantined" for authority in before_source.values())

    report = await apply_raw_quarantine_group_dedup(archive_root, dry_run=True)

    assert report.applied is False
    assert report.scanned_count == 4
    assert report.group_count == 1
    assert report.promoted_count == 1
    assert report.marked_duplicate_count == 2
    promotion = report.promotions[0]
    assert promotion.representative_raw_id == "raw-a"
    assert promotion.representative_session_id == ""
    assert promotion.duplicate_raw_ids == ("raw-b", "raw-c")

    assert _revision_authority_rows(archive_root) == before_source
    assert _index_sessions_rows(archive_root) == before_index
    assert _receipt_rows(archive_root) == {}


async def test_apply_refuses_without_backup_manifest(tmp_path: Path) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before = _revision_authority_rows(archive_root)

    with pytest.raises(RawQuarantineGroupDedupApplyError, match="backup manifest"):
        await apply_raw_quarantine_group_dedup(archive_root, backup_manifest=None, dry_run=False)

    assert _revision_authority_rows(archive_root) == before
    assert _receipt_rows(archive_root) == {}


async def test_apply_refuses_when_backup_manifest_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before = _revision_authority_rows(archive_root)

    def _reject(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        raise ValueError("backup manifest does not match live source.db")

    monkeypatch.setattr(
        "polylogue.maintenance.raw_quarantine_group_dedup_apply.validate_migration_backup_manifest",
        _reject,
    )

    manifest = tmp_path / "stale-backup" / "manifest.json"
    with pytest.raises(ValueError, match="does not match"):
        await apply_raw_quarantine_group_dedup(archive_root, backup_manifest=manifest, dry_run=False)

    assert _revision_authority_rows(archive_root) == before
    assert _receipt_rows(archive_root) == {}


async def test_apply_materializes_one_representative_and_marks_duplicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = _build_fixture_archive(tmp_path)

    validated: list[tuple[Path, object]] = []

    def _fake_validate(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        validated.append((manifest, tier))
        assert connection.execute("SELECT 1").fetchone() == (1,)
        return manifest.with_name("verification-receipt.json")

    monkeypatch.setattr(
        "polylogue.maintenance.raw_quarantine_group_dedup_apply.validate_migration_backup_manifest",
        _fake_validate,
    )

    manifest = tmp_path / "verified-backup" / "manifest.json"
    report = await apply_raw_quarantine_group_dedup(archive_root, backup_manifest=manifest, dry_run=False)

    assert report.applied is True
    assert report.group_count == 1
    assert report.promoted_count == 1
    assert report.marked_duplicate_count == 2
    assert report.backup_manifest == manifest
    assert len(validated) >= 2  # precheck + at least one locked-transaction re-validation

    promotion = report.promotions[0]
    assert promotion.representative_raw_id == "raw-a"
    assert promotion.representative_session_id != ""
    assert promotion.duplicate_raw_ids == ("raw-b", "raw-c")

    # The representative raw is now genuinely indexed.
    index_sessions = _index_sessions_rows(archive_root)
    assert (promotion.representative_session_id, "raw-a") in index_sessions
    # No session materialized for the duplicates or the untouched singleton.
    indexed_raw_ids = {raw_id for _session_id, raw_id in index_sessions}
    assert "raw-b" not in indexed_raw_ids
    assert "raw-c" not in indexed_raw_ids
    assert "raw-singleton" not in indexed_raw_ids

    rows = _revision_authority_rows(archive_root)
    assert rows["raw-a"] == "quarantined"  # untouched -- proof is its own indexed session, not this field
    assert rows["raw-b"] == "byte_proven"
    assert rows["raw-c"] == "byte_proven"
    # The genuine singleton is never touched.
    assert rows["raw-singleton"] == "quarantined"

    receipts = _receipt_rows(archive_root)
    assert receipts == {
        "raw-b": ("raw-a", promotion.representative_session_id, TOOL_VERSION),
        "raw-c": ("raw-a", promotion.representative_session_id, TOOL_VERSION),
    }


async def test_apply_leaves_group_untouched_when_representative_produces_no_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A representative whose bytes fail to parse (decode error, no session
    materialized) must leave the WHOLE group untouched: no duplicate marked,
    no receipt, and the representative itself still quarantined -- never
    guessed at. Covers raw_quarantine_group_dedup_apply.py's "zero sessions"
    branch (CodeRabbit PR #3697)."""
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    unparseable_payload = b"this is not valid json at all, just garbage bytes\n"
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for raw_id in ("raw-x", "raw-y"):
            _write_quarantined_raw(
                archive, raw_id=raw_id, payload=unparseable_payload, source_path=str(tmp_path / "garbage.jsonl")
            )
        archive.commit()

    before = _revision_authority_rows(archive_root)

    def _fake_validate(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        return manifest.with_name("verification-receipt.json")

    monkeypatch.setattr(
        "polylogue.maintenance.raw_quarantine_group_dedup_apply.validate_migration_backup_manifest",
        _fake_validate,
    )

    manifest = tmp_path / "verified-backup" / "manifest.json"
    report = await apply_raw_quarantine_group_dedup(archive_root, backup_manifest=manifest, dry_run=False)

    assert report.applied is True
    assert report.group_count == 1  # classified as a group -- classification doesn't require parseability
    assert report.promoted_count == 0
    assert report.marked_duplicate_count == 0
    assert _revision_authority_rows(archive_root) == before
    assert _receipt_rows(archive_root) == {}
    assert _index_sessions_rows(archive_root) == []


async def test_apply_leaves_group_untouched_when_representative_materializes_multiple_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A representative raw that is a genuine multi-session capture file
    (e.g. a grouped Claude Code JSONL split by sessionId) materializes MORE
    than one sessions row from one raw_id -- this actuator's whole premise
    is one raw/one representative session, so it must refuse to guess which
    materialized session is "the" representative and leave the group
    untouched entirely, exactly like the zero-session case."""
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    multi_session_payload = (
        b'{"type":"user","sessionId":"first-session","uuid":"u1",'
        b'"message":{"role":"user","content":"one"}}\n'
        b'{"type":"user","sessionId":"second-session","uuid":"u2",'
        b'"message":{"role":"user","content":"two"}}\n'
    )
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for raw_id in ("raw-multi-a", "raw-multi-b"):
            archive.write_raw_payload(
                provider=Provider.CLAUDE_CODE,
                payload=multi_session_payload,
                source_path=str(tmp_path / "multi.jsonl"),
                source_index=-1,
                acquired_at_ms=1_700_000_000_000,
                raw_id=raw_id,
            )
        archive.commit()

    before = _revision_authority_rows(archive_root)

    def _fake_validate(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        return manifest.with_name("verification-receipt.json")

    monkeypatch.setattr(
        "polylogue.maintenance.raw_quarantine_group_dedup_apply.validate_migration_backup_manifest",
        _fake_validate,
    )

    manifest = tmp_path / "verified-backup" / "manifest.json"
    report = await apply_raw_quarantine_group_dedup(archive_root, backup_manifest=manifest, dry_run=False)

    assert report.applied is True
    assert report.group_count == 1
    assert report.promoted_count == 0
    assert report.marked_duplicate_count == 0
    assert _revision_authority_rows(archive_root) == before
    assert _receipt_rows(archive_root) == {}
    # The representative's own content DID materialize (real, legitimate
    # sessions) -- this actuator just declines to build a receipt around it.
    index_sessions = _index_sessions_rows(archive_root)
    assert len(index_sessions) == 2
    assert all(raw_id == "raw-multi-a" for _session_id, raw_id in index_sessions)


def test_mark_group_duplicates_rolls_back_on_receipt_constraint_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Covers the phase-two rollback branch directly: a receipt INSERT that
    violates a CHECK constraint must roll back the WHOLE transaction --
    including any earlier duplicate's revision_authority UPDATE already
    applied within the same locked transaction -- not partially commit
    (CodeRabbit PR #3697)."""
    archive_root = _build_fixture_archive(tmp_path)
    before = _revision_authority_rows(archive_root)

    def _fake_validate(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        return manifest.with_name("verification-receipt.json")

    monkeypatch.setattr(
        "polylogue.maintenance.raw_quarantine_group_dedup_apply.validate_migration_backup_manifest",
        _fake_validate,
    )

    # blob_hash with the wrong length violates raw_quarantine_group_dedup_receipts'
    # CHECK(length(blob_hash) = 32) on the very first duplicate's INSERT.
    bad_group = RawQuarantineGroup(
        source_path=str(tmp_path / "repeated.jsonl"),
        blob_hash=b"\x00" * 31,
        blob_size=len(_GROUP_PAYLOAD),
        raw_ids=("raw-a", "raw-b", "raw-c"),
    )
    manifest = tmp_path / "verified-backup" / "manifest.json"

    with pytest.raises(sqlite3.IntegrityError):
        _mark_group_duplicates(archive_root / "source.db", manifest, bad_group, "fake-representative-session-id")

    assert _revision_authority_rows(archive_root) == before
    assert _receipt_rows(archive_root) == {}


def test_checkpoint_live_tier_refuses_when_wal_checkpoint_reports_busy(tmp_path: Path) -> None:
    """``PRAGMA wal_checkpoint(TRUNCATE)`` always returns one row
    ``(busy, log, checkpointed)``. ``busy=1`` means the WAL was NOT fully
    truncated (another connection held a blocking lock); a subsequent
    backup-manifest fingerprint check must not silently attest against a
    tier that still has uncheckpointed frames (CodeRabbit PR #3697)."""

    class _BusyCursor:
        def fetchone(self) -> tuple[int, int, int]:
            return (1, 0, 0)  # busy=1

    class _BusyConnection:
        def execute(self, _sql: str) -> _BusyCursor:
            return _BusyCursor()

    with pytest.raises(RawQuarantineGroupDedupApplyError, match="blocked by another connection"):
        _checkpoint_live_tier(cast(sqlite3.Connection, _BusyConnection()))


async def test_apply_with_no_qualifying_groups_is_a_clean_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        _write_quarantined_raw(
            archive, raw_id="raw-only", payload=_SINGLETON_PAYLOAD, source_path=str(tmp_path / "only.jsonl")
        )
        archive.commit()

    def _fake_validate(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        return manifest.with_name("verification-receipt.json")

    monkeypatch.setattr(
        "polylogue.maintenance.raw_quarantine_group_dedup_apply.validate_migration_backup_manifest",
        _fake_validate,
    )

    manifest = tmp_path / "verified-backup" / "manifest.json"
    report = await apply_raw_quarantine_group_dedup(archive_root, backup_manifest=manifest, dry_run=False)

    assert report.applied is True
    assert report.group_count == 0
    assert report.promoted_count == 0
    assert report.marked_duplicate_count == 0
    assert _receipt_rows(archive_root) == {}

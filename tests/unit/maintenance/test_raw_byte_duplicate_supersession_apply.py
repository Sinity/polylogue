"""polylogue-6753s: the actuator for byte-duplicate supersession.

Builds a fixture archive with a genuine byte-duplicate pair (one indexed, one
quarantined-with-no-logical_source_key sharing its blob_hash) plus a genuine
non-duplicate quarantined-no-lsk raw (must NOT be resolved), and proves:

* dry-run (the default) never mutates anything, and never writes to index.db;
* --apply promotes only the true duplicate to revision_authority='byte_proven'
  (revision_authority_evidence left NULL -- this is a distinct, independently
  verifiable mechanism recorded in its own receipt table, not the
  live-source-verification vocabulary), and writes an immutable receipt
  naming the indexed twin's raw_id/session_id;
* the novel (non-duplicate) row and the already-indexed twin's own row are
  left untouched;
* applying without a backup manifest is refused before anything is touched;
* predecessor_raw_id / baseline_raw_id / acquisition_generation /
  revision_authority_evidence are never touched.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.maintenance.raw_byte_duplicate_supersession_apply import (
    TOOL_VERSION,
    RawByteDuplicateSupersessionApplyError,
    apply_raw_byte_duplicate_supersession,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

_DUPLICATE_PAYLOAD = b'{"messages":["dup"]}\n'
_NOVEL_PAYLOAD = b'{"messages":["novel"]}\n'


def _write_quarantined_no_lsk_raw(archive: ArchiveStore, *, raw_id: str, payload: bytes, source_path: str) -> None:
    archive.write_raw_payload(
        provider=Provider.CLAUDE_CODE,
        payload=payload,
        source_path=source_path,
        source_index=-1,
        acquired_at_ms=1_700_000_000_000,
        raw_id=raw_id,
    )


def _index_session(conn: sqlite3.Connection, *, raw_id: str, native_id: str) -> None:
    conn.execute(
        "INSERT INTO sessions (origin, native_id, content_hash, raw_id, created_at_ms, updated_at_ms) "
        "VALUES ('claude-code-session', ?, ?, ?, 0, 0)",
        (native_id, b"\x11" * 32, raw_id),
    )


def _build_fixture_archive(tmp_path: Path) -> Path:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        # The already-indexed twin -- never touched by the actuator.
        _write_quarantined_no_lsk_raw(
            archive,
            raw_id="raw-indexed-twin",
            payload=_DUPLICATE_PAYLOAD,
            source_path=str(tmp_path / "twin.jsonl"),
        )
        # The genuine duplicate -- byte-identical to the indexed twin, no
        # logical_source_key: the population this bead resolves.
        _write_quarantined_no_lsk_raw(
            archive,
            raw_id="raw-duplicate",
            payload=_DUPLICATE_PAYLOAD,
            source_path=str(tmp_path / "dup.jsonl"),
        )
        # A genuinely novel row: quarantined, no logical_source_key, but no
        # indexed byte-identical twin anywhere -- must never be resolved
        # here (polylogue-lkrc/hjpx's reconciler scope, not this bead's).
        _write_quarantined_no_lsk_raw(
            archive,
            raw_id="raw-novel",
            payload=_NOVEL_PAYLOAD,
            source_path=str(tmp_path / "novel.jsonl"),
        )
        archive.commit()

        _index_session(archive._conn, raw_id="raw-indexed-twin", native_id="native-indexed-twin")
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


def _linkage_rows(archive_root: Path) -> dict[str, tuple[object, object, object]]:
    conn = sqlite3.connect(archive_root / "source.db")
    try:
        rows = conn.execute(
            "SELECT raw_id, predecessor_raw_id, baseline_raw_id, acquisition_generation FROM raw_sessions"
        ).fetchall()
    finally:
        conn.close()
    return {raw_id: (pred, baseline, gen) for raw_id, pred, baseline, gen in rows}


def _receipt_rows(archive_root: Path) -> dict[str, tuple[str, str]]:
    conn = sqlite3.connect(archive_root / "source.db")
    try:
        rows = conn.execute(
            "SELECT raw_id, duplicate_of_raw_id, tool_version FROM raw_byte_duplicate_supersession_receipts"
        ).fetchall()
    finally:
        conn.close()
    return {raw_id: (duplicate_of_raw_id, tool_version) for raw_id, duplicate_of_raw_id, tool_version in rows}


def _index_sessions_rows(archive_root: Path) -> list[tuple[str, str | None]]:
    conn = sqlite3.connect(archive_root / "index.db")
    try:
        rows = conn.execute("SELECT session_id, raw_id FROM sessions ORDER BY session_id").fetchall()
    finally:
        conn.close()
    return [(session_id, raw_id) for session_id, raw_id in rows]


def test_dry_run_makes_zero_mutations(tmp_path: Path) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before_source = _revision_authority_rows(archive_root)
    before_index = _index_sessions_rows(archive_root)
    assert all(authority == "quarantined" for authority, _evidence in before_source.values())

    report = apply_raw_byte_duplicate_supersession(archive_root, dry_run=True)

    assert report.applied is False
    # raw-indexed-twin, raw-duplicate, raw-novel are all quarantined with no
    # logical_source_key -- the whole population is in scope. Only
    # raw-duplicate has an OTHER raw (raw-indexed-twin) that is indexed;
    # raw-indexed-twin's only same-hash sibling (raw-duplicate) is not
    # indexed, so it is correctly classified as novel, not as a duplicate of
    # itself.
    assert report.scanned_count == 3
    assert report.promoted_count == 1
    assert report.novel_count == 2
    assert report.promoted_raw_ids == ()

    assert _revision_authority_rows(archive_root) == before_source
    assert _index_sessions_rows(archive_root) == before_index
    assert _receipt_rows(archive_root) == {}


def test_apply_promotes_only_the_true_duplicate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = _build_fixture_archive(tmp_path)

    validated: list[tuple[Path, object]] = []

    def _fake_validate(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        validated.append((manifest, tier))
        assert connection.execute("SELECT 1").fetchone() == (1,)
        return manifest.with_name("verification-receipt.json")

    monkeypatch.setattr(
        "polylogue.maintenance.raw_byte_duplicate_supersession_apply.validate_migration_backup_manifest",
        _fake_validate,
    )

    linkage_before = _linkage_rows(archive_root)
    index_before = _index_sessions_rows(archive_root)

    manifest = tmp_path / "verified-backup" / "manifest.json"
    report = apply_raw_byte_duplicate_supersession(archive_root, backup_manifest=manifest, dry_run=False)

    assert report.applied is True
    assert report.promoted_count == 1
    assert set(report.promoted_raw_ids) == {"raw-duplicate"}
    assert report.novel_count == 2
    assert report.backup_manifest == manifest
    assert validated == [(manifest, ArchiveTier.SOURCE), (manifest, ArchiveTier.SOURCE)]

    rows = _revision_authority_rows(archive_root)
    assert rows["raw-duplicate"] == ("byte_proven", None)
    # The indexed twin's own row is never touched.
    assert rows["raw-indexed-twin"] == ("quarantined", None)
    # Genuinely novel row is never touched.
    assert rows["raw-novel"] == ("quarantined", None)

    receipts = _receipt_rows(archive_root)
    assert receipts == {"raw-duplicate": ("raw-indexed-twin", TOOL_VERSION)}

    # Revision-graph linkage untouched.
    linkage_after = _linkage_rows(archive_root)
    assert linkage_after["raw-duplicate"] == linkage_before["raw-duplicate"]
    assert linkage_after["raw-indexed-twin"] == linkage_before["raw-indexed-twin"]

    # index.db is never written to.
    assert _index_sessions_rows(archive_root) == index_before


def test_apply_refuses_without_backup_manifest(tmp_path: Path) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before = _revision_authority_rows(archive_root)

    with pytest.raises(RawByteDuplicateSupersessionApplyError, match="backup manifest"):
        apply_raw_byte_duplicate_supersession(archive_root, backup_manifest=None, dry_run=False)

    assert _revision_authority_rows(archive_root) == before
    assert _receipt_rows(archive_root) == {}


def test_apply_refuses_when_backup_manifest_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = _build_fixture_archive(tmp_path)
    before = _revision_authority_rows(archive_root)

    def _reject(manifest: Path, tier: object, *, connection: sqlite3.Connection) -> Path:
        raise ValueError("backup manifest does not match live source.db")

    monkeypatch.setattr(
        "polylogue.maintenance.raw_byte_duplicate_supersession_apply.validate_migration_backup_manifest",
        _reject,
    )

    manifest = tmp_path / "stale-backup" / "manifest.json"
    with pytest.raises(ValueError, match="does not match"):
        apply_raw_byte_duplicate_supersession(archive_root, backup_manifest=manifest, dry_run=False)

    assert _revision_authority_rows(archive_root) == before
    assert _receipt_rows(archive_root) == {}

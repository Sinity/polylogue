"""polylogue-lb39z (Phase 1, item 3): membershipless append-chain backfill.

Proves the read-only classifier scopes strictly to the named population:
quarantined, revision_kind='append', with zero raw_session_memberships rows
-- and that it reuses the identical live-source byte-window proof
polylogue-u19l already validated, including the Codex header-strip case.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.live_source_reconciliation import LiveSourceVerdict
from polylogue.storage.raw_append_chain_backfill import plan_append_chain_backfill
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _write_raw(
    archive: ArchiveStore,
    *,
    raw_id: str,
    payload: bytes,
    source_path: str,
    kind: RawRevisionKind,
    logical_source_key: str,
    append_start_offset: int | None = None,
    append_end_offset: int | None = None,
) -> None:
    predecessor_source_revision = f"{raw_id}-predecessor" if kind is RawRevisionKind.APPEND else None
    archive.write_raw_payload(
        provider=Provider.CLAUDE_CODE,
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


def test_scopes_to_quarantined_append_rows_with_no_membership_row(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    exact_live = tmp_path / "exact.jsonl"
    exact_live.write_bytes(b"PREFIX-BYTES" + b'{"delta":1}\n')

    diverged_live = tmp_path / "diverged.jsonl"
    diverged_live.write_bytes(b"PREFIX-BYTES" + b'{"CHANGED":true}\n')

    missing_live_path = str(tmp_path / "gone.jsonl")

    has_membership_live = tmp_path / "has-membership.jsonl"
    has_membership_live.write_bytes(b"PREFIX-BYTES" + b'{"delta":1}\n')

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        # Membershipless, exact match at its own append window -- the target population.
        _write_raw(
            archive,
            raw_id="raw-membershipless-exact",
            payload=b'{"delta":1}\n',
            source_path=str(exact_live),
            kind=RawRevisionKind.APPEND,
            logical_source_key="claude-code:exact",
            append_start_offset=len(b"PREFIX-BYTES"),
            append_end_offset=len(b"PREFIX-BYTES") + len(b'{"delta":1}\n'),
        )
        # Membershipless, but its own window has diverged.
        _write_raw(
            archive,
            raw_id="raw-membershipless-diverged",
            payload=b'{"delta":1}\n',
            source_path=str(diverged_live),
            kind=RawRevisionKind.APPEND,
            logical_source_key="claude-code:diverged",
            append_start_offset=len(b"PREFIX-BYTES"),
            append_end_offset=len(b"PREFIX-BYTES") + len(b'{"delta":1}\n'),
        )
        # Membershipless, source file gone.
        _write_raw(
            archive,
            raw_id="raw-membershipless-missing",
            payload=b'{"delta":1}\n',
            source_path=missing_live_path,
            kind=RawRevisionKind.APPEND,
            logical_source_key="claude-code:missing",
            append_start_offset=0,
            append_end_offset=len(b'{"delta":1}\n'),
        )
        # HAS a membership row already -- must be excluded even though its
        # bytes would otherwise match exactly (this is u19l's population,
        # not item 3's).
        _write_raw(
            archive,
            raw_id="raw-has-membership",
            payload=b'{"delta":1}\n',
            source_path=str(has_membership_live),
            kind=RawRevisionKind.APPEND,
            logical_source_key="claude-code:has-membership",
            append_start_offset=len(b"PREFIX-BYTES"),
            append_end_offset=len(b"PREFIX-BYTES") + len(b'{"delta":1}\n'),
        )
        # A membershipless FULL row -- out of scope (title says "append-chain").
        _write_raw(
            archive,
            raw_id="raw-membershipless-full",
            payload=b'{"a":1}\n',
            source_path=str(tmp_path / "full.jsonl"),
            kind=RawRevisionKind.FULL,
            logical_source_key="claude-code:full",
        )
        (tmp_path / "full.jsonl").write_bytes(b'{"a":1}\n')
        archive.commit()

    conn = sqlite3.connect(archive_root / "source.db")
    try:
        _write_membership(conn, raw_id="raw-has-membership", logical_source_key="claude-code:has-membership")
        conn.commit()
    finally:
        conn.close()

    conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    try:
        plan = plan_append_chain_backfill(conn, blob_store=BlobStore(archive_root / "blob"))
    finally:
        conn.close()

    assert plan.scanned_count == 3  # exact, diverged, missing -- has-membership and full excluded
    assert {c.raw_id for c in plan.exact_match} == {"raw-membershipless-exact"}
    assert {c.raw_id for c in plan.diverged} == {"raw-membershipless-diverged"}
    assert {c.raw_id for c in plan.source_missing} == {"raw-membershipless-missing"}
    for candidate in plan.exact_match:
        assert candidate.comparison.verdict == LiveSourceVerdict.EXACT_MATCH

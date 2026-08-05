"""polylogue-6753s: byte-duplicate supersession classifier.

Proves the read-only classifier scopes strictly to the named population
(quarantined, ``logical_source_key IS NULL``) and only flags a candidate as a
duplicate when its ``blob_hash`` matches some OTHER raw that already has a
materialized ``sessions`` row in index.db -- never the candidate's own row,
never a raw sharing bytes with only unindexed siblings, and never a candidate
that already has a ``logical_source_key`` (out of this bead's scope; that
population is reachable by the ordinary reconciler).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Provider
from polylogue.storage.raw_byte_duplicate_supersession import plan_byte_duplicate_supersession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _write_quarantined_no_lsk_raw(archive: ArchiveStore, *, raw_id: str, payload: bytes, source_path: str) -> None:
    """Write a raw the way real acquisition does before any reconciler ever sees it: no revision envelope at all."""
    archive.write_raw_payload(
        provider=Provider.CLAUDE_CODE,
        payload=payload,
        source_path=source_path,
        source_index=-1,
        acquired_at_ms=1_700_000_000_000,
        raw_id=raw_id,
    )


def _write_raw_with_logical_key(
    archive: ArchiveStore, *, raw_id: str, payload: bytes, source_path: str, logical_source_key: str
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
            kind=RawRevisionKind.FULL,
            source_revision=f"{raw_id}-revision",
            acquisition_generation=0,
            authority=RawRevisionAuthority.QUARANTINED,
        ),
    )


def _index_session(conn: sqlite3.Connection, *, raw_id: str, native_id: str) -> None:
    conn.execute(
        "INSERT INTO sessions (origin, native_id, content_hash, raw_id, created_at_ms, updated_at_ms) "
        "VALUES ('claude-code-session', ?, ?, ?, 0, 0)",
        (native_id, b"\x11" * 32, raw_id),
    )


def test_only_flags_quarantined_no_lsk_raws_byte_identical_to_an_indexed_twin(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    duplicate_payload = b'{"messages":["dup"]}\n'
    novel_payload = b'{"messages":["novel"]}\n'
    unindexed_twin_payload = b'{"messages":["unindexed-twin"]}\n'

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        # The target population: quarantined, no logical_source_key, and an
        # OTHER raw with the same bytes IS indexed below -- must be flagged.
        _write_quarantined_no_lsk_raw(
            archive,
            raw_id="raw-dup",
            payload=duplicate_payload,
            source_path=str(tmp_path / "dup.jsonl"),
        )
        # The already-indexed twin -- byte-identical, but its OWN row must
        # never be touched by this classifier (it's the reference, not the
        # duplicate).
        _write_quarantined_no_lsk_raw(
            archive,
            raw_id="raw-dup-indexed-twin",
            payload=duplicate_payload,
            source_path=str(tmp_path / "dup-twin.jsonl"),
        )
        # A parser-failed raw can share bytes with a materialized twin, but
        # parse failure is evidence to recover, not duplicate authority to
        # supersede away.
        _write_quarantined_no_lsk_raw(
            archive,
            raw_id="raw-parser-failed-dup",
            payload=duplicate_payload,
            source_path=str(tmp_path / "parser-failed-dup.jsonl"),
        )
        archive.mark_raw_parse_failed(
            "raw-parser-failed-dup",
            provider=Provider.CLAUDE_CODE,
            error=ValueError("deliberate parser failure"),
        )
        # Quarantined, no logical_source_key, but genuinely unique bytes --
        # no indexed twin exists anywhere. Must be left alone (novel; in
        # scope for lkrc/hjpx's reconciler, not this bead).
        _write_quarantined_no_lsk_raw(
            archive,
            raw_id="raw-novel",
            payload=novel_payload,
            source_path=str(tmp_path / "novel.jsonl"),
        )
        # Quarantined, no logical_source_key, and shares bytes with ANOTHER
        # raw -- but that other raw is never indexed. Must NOT be flagged:
        # byte-identical to *something*, but not to anything the archive
        # already accepted.
        _write_quarantined_no_lsk_raw(
            archive,
            raw_id="raw-unindexed-dup",
            payload=unindexed_twin_payload,
            source_path=str(tmp_path / "unindexed-dup.jsonl"),
        )
        _write_quarantined_no_lsk_raw(
            archive,
            raw_id="raw-unindexed-dup-sibling",
            payload=unindexed_twin_payload,
            source_path=str(tmp_path / "unindexed-dup-sibling.jsonl"),
        )
        # Out of scope: already HAS a logical_source_key, even though its
        # bytes match the indexed twin exactly -- the ordinary reconciler
        # already sees this row; this bead's classifier must not touch it.
        _write_raw_with_logical_key(
            archive,
            raw_id="raw-has-lsk",
            payload=duplicate_payload,
            source_path=str(tmp_path / "has-lsk.jsonl"),
            logical_source_key="claude-code:has-lsk",
        )
        archive.commit()

        _index_session(archive._conn, raw_id="raw-dup-indexed-twin", native_id="native-dup-indexed-twin")
        archive.commit()

    source_conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    index_conn = sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)
    try:
        plan = plan_byte_duplicate_supersession(source_conn, index_conn)
    finally:
        source_conn.close()
        index_conn.close()

    # raw-dup, raw-novel, raw-unindexed-dup, raw-unindexed-dup-sibling --
    # raw-dup-indexed-twin and raw-has-lsk are excluded from the scan itself
    # (the twin has no logical_source_key either, so it too satisfies the
    # WHERE clause -- but it must not appear as a "duplicate OF itself").
    assert plan.scanned_count == 5
    assert {c.raw_id for c in plan.duplicates} == {"raw-dup"}
    assert "raw-parser-failed-dup" not in {c.raw_id for c in plan.duplicates}
    duplicate = next(c for c in plan.duplicates if c.raw_id == "raw-dup")
    assert duplicate.duplicate_of_raw_id == "raw-dup-indexed-twin"
    assert duplicate.duplicate_of_session_id == "claude-code-session:native-dup-indexed-twin"
    assert duplicate.blob_size == len(duplicate_payload)
    # novel + both unindexed-dup siblings (neither has an indexed twin) are
    # left alone.
    assert plan.novel_count == 4
    assert plan.duplicate_bytes == len(duplicate_payload)


def test_limit_caps_scanned_rows(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for i in range(3):
            _write_quarantined_no_lsk_raw(
                archive,
                raw_id=f"raw-{i}",
                payload=f'{{"n":{i}}}\n'.encode(),
                source_path=str(tmp_path / f"{i}.jsonl"),
            )
        archive.commit()

    source_conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    index_conn = sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)
    try:
        plan = plan_byte_duplicate_supersession(source_conn, index_conn, limit=2)
    finally:
        source_conn.close()
        index_conn.close()

    assert plan.scanned_count == 2

"""polylogue-u19l: live-source-verification classification for quarantined raws.

Proves the three classifications the investigation bead asked for against a
real archive: a quarantined row whose recorded source file still contains its
exact bytes, one that has diverged, and one whose source file is gone --
plus the Codex-specific historical-header-strip case that made naive byte
comparison permanently fail for Codex append captures before polylogue-u19l's
part (B) fix.
"""

from __future__ import annotations

import sqlite3
from json import dumps as json_dumps
from pathlib import Path

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.config import Config
from polylogue.core.enums import Provider
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.live_source_reconciliation import (
    LiveSourceVerdict,
    compare_raw_against_live_source,
    plan_quarantined_live_source_reconciliation,
)
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _config(root: Path) -> Config:
    return Config(archive_root=root, render_root=root / "render", sources=[], db_path=root / "index.db")


def test_compare_raw_against_live_source_exact_match(tmp_path: Path) -> None:
    live = tmp_path / "session.jsonl"
    live.write_bytes(b'{"a":1}\n{"b":2}\n')
    comparison = compare_raw_against_live_source(
        raw_payload=b'{"a":1}\n{"b":2}\n',
        provider=Provider.CLAUDE_CODE,
        revision_kind=RawRevisionKind.FULL,
        source_path=live,
    )
    assert comparison.verdict == LiveSourceVerdict.EXACT_MATCH
    assert comparison.matched_after_codex_header_strip is False


def test_compare_raw_against_live_source_diverged(tmp_path: Path) -> None:
    live = tmp_path / "session.jsonl"
    live.write_bytes(b'{"a":1}\n{"CHANGED":true}\n')
    comparison = compare_raw_against_live_source(
        raw_payload=b'{"a":1}\n{"b":2}\n',
        provider=Provider.CLAUDE_CODE,
        revision_kind=RawRevisionKind.FULL,
        source_path=live,
    )
    assert comparison.verdict == LiveSourceVerdict.DIVERGED


def test_compare_raw_against_live_source_missing(tmp_path: Path) -> None:
    comparison = compare_raw_against_live_source(
        raw_payload=b'{"a":1}\n',
        provider=Provider.CLAUDE_CODE,
        revision_kind=RawRevisionKind.FULL,
        source_path=tmp_path / "does-not-exist.jsonl",
    )
    assert comparison.verdict == LiveSourceVerdict.SOURCE_MISSING


def test_compare_raw_against_live_source_codex_append_offset_window(tmp_path: Path) -> None:
    """A post-u19l Codex append raw: literal bytes, no synthetic header."""
    live = tmp_path / "rollout-x.jsonl"
    prefix = b'{"type":"session_meta","payload":{"id":"abc"}}\n'
    delta = b'{"type":"response_item","payload":{"id":"m1"}}\n'
    live.write_bytes(prefix + delta)
    comparison = compare_raw_against_live_source(
        raw_payload=delta,
        provider=Provider.CODEX,
        revision_kind=RawRevisionKind.APPEND,
        source_path=live,
        append_start_offset=len(prefix),
        append_end_offset=len(prefix) + len(delta),
    )
    assert comparison.verdict == LiveSourceVerdict.EXACT_MATCH
    assert comparison.matched_after_codex_header_strip is False


def test_compare_raw_against_live_source_codex_append_historical_header_strip(tmp_path: Path) -> None:
    """A pre-u19l historical Codex append raw: synthetic header spliced in.

    The archived blob is never a literal slice of the live file (the whole
    root cause this bead diagnosed), but the real tail bytes underneath it
    still match exactly once the synthetic header is stripped.
    """
    live = tmp_path / "rollout-y.jsonl"
    prefix = b'{"type":"session_meta","payload":{"id":"abc"}}\n'
    delta = b'{"type":"response_item","payload":{"id":"m1"}}\n'
    live.write_bytes(prefix + delta)
    synthetic_header = json_dumps({"type": "session_meta", "payload": {"id": "abc"}}, separators=(",", ":")).encode()
    archived_payload = synthetic_header + b"\n" + delta
    comparison = compare_raw_against_live_source(
        raw_payload=archived_payload,
        provider=Provider.CODEX,
        revision_kind=RawRevisionKind.APPEND,
        source_path=live,
        append_start_offset=len(prefix),
        append_end_offset=len(prefix) + len(delta),
    )
    assert comparison.verdict == LiveSourceVerdict.EXACT_MATCH
    assert comparison.matched_after_codex_header_strip is True


def test_compare_raw_against_live_source_does_not_strip_real_session_meta(tmp_path: Path) -> None:
    """A genuinely divergent Codex row must not be laundered by header-strip.

    If the payload's first line happens to be a real, differently-shaped
    session_meta record (not the exact synthetic shape this bead's part (B)
    retired), it must not be stripped -- a real content difference stays
    DIVERGED, not falsely reported as an exact match.
    """
    live = tmp_path / "rollout-z.jsonl"
    live.write_bytes(b'{"type":"response_item","payload":{"id":"m1"}}\n')
    archived_payload = (
        b'{"type":"session_meta","payload":{"id":"abc","timestamp":"2026-01-01T00:00:00Z"}}\n'
        b'{"type":"response_item","payload":{"id":"m1"}}\n'
    )
    comparison = compare_raw_against_live_source(
        raw_payload=archived_payload,
        provider=Provider.CODEX,
        revision_kind=RawRevisionKind.APPEND,
        source_path=live,
        append_start_offset=0,
        append_end_offset=len(b'{"type":"response_item","payload":{"id":"m1"}}\n'),
    )
    assert comparison.verdict == LiveSourceVerdict.DIVERGED


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
            append_start_offset=append_start_offset,
            append_end_offset=append_end_offset,
            authority=RawRevisionAuthority.QUARANTINED,
        ),
    )


def test_plan_quarantined_live_source_reconciliation_classifies_every_row(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    exact_live = tmp_path / "exact.jsonl"
    exact_live.write_bytes(b'{"a":1}\n')
    diverged_live = tmp_path / "diverged.jsonl"
    diverged_live.write_bytes(b'{"a":"CHANGED"}\n')
    missing_live_path = str(tmp_path / "gone.jsonl")

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
        archive.commit()

    blob_store = BlobStore(archive_root / "blob")
    conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    try:
        plan = plan_quarantined_live_source_reconciliation(conn, blob_store=blob_store)
    finally:
        conn.close()

    assert plan.scanned_count == 3
    assert {c.raw_id for c in plan.exact_match} == {"raw-exact"}
    assert {c.raw_id for c in plan.diverged} == {"raw-diverged"}
    assert {c.raw_id for c in plan.source_missing} == {"raw-missing"}
    assert plan.exact_match_bytes == len(b'{"a":1}\n')
    assert plan.codex_header_strip_match_count == 0

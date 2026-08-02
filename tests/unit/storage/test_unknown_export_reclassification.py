"""polylogue-mvq8: re-detection report for browser captures stamped unknown-export.

Proves the report-only classifier against a real archive: an unknown-export
raw row whose blob is actually a browser-capture envelope with a recoverable
provider (the exact >1MiB raw_provider_payload shape that used to defeat the
1MiB prefix probe), one that is genuinely not a browser capture at all (stays
unknown), and one whose blob is missing from the store.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from polylogue.archive.revision_authority import RawRevisionAuthority, RawRevisionEnvelope, RawRevisionKind
from polylogue.core.enums import Origin, Provider
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.unknown_export_reclassification import (
    UnknownExportReclassificationVerdict,
    plan_unknown_export_reclassification,
)


def _write_unknown_export_raw(archive: ArchiveStore, *, raw_id: str, payload: bytes, source_path: str) -> None:
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


def _browser_capture_envelope_past_prefix(*, session_provider: str, provider_session_id: str) -> bytes:
    """A browser-capture envelope whose ``session.provider`` sits past 1MiB.

    Mirrors the real receiver's key-sorted output shape:
    ``raw_provider_payload`` (unbounded) sorts before ``session``
    alphabetically -- this is the exact repro shape polylogue-mvq8 measured.
    """
    huge_padding = "x" * (1024 * 1024 + 64 * 1024)
    payload = {
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "capture_id": f"{session_provider}:{provider_session_id}",
        "provenance": {
            "source_url": f"https://example.com/{provider_session_id}",
            "captured_at": "2026-04-24T00:00:00+00:00",
            "adapter_name": "test-adapter",
        },
        "raw_provider_payload": {"padding": huge_padding},
        "session": {
            "provider": session_provider,
            "provider_session_id": provider_session_id,
            "turns": [{"provider_turn_id": "u1", "role": "user", "text": "hi"}],
        },
    }
    return json.dumps(payload).encode("utf-8")


def test_plan_unknown_export_reclassification_classifies_every_row(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    reclassifiable_payload = _browser_capture_envelope_past_prefix(
        session_provider="chatgpt", provider_session_id="past-prefix"
    )
    not_a_browser_capture_payload = json.dumps({"some": "unrelated document"}).encode("utf-8")

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        _write_unknown_export_raw(
            archive,
            raw_id="raw-reclassifiable",
            payload=reclassifiable_payload,
            source_path="/spool/browser-capture/unknown/reclassifiable.json",
        )
        _write_unknown_export_raw(
            archive,
            raw_id="raw-still-unknown",
            payload=not_a_browser_capture_payload,
            source_path="/spool/browser-capture/unknown/still-unknown.json",
        )
        archive.commit()

    blob_store = BlobStore(archive_root / "blob")
    conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    try:
        plan = plan_unknown_export_reclassification(conn, blob_store=blob_store, source_path_like=None)
    finally:
        conn.close()

    assert plan.scanned_count == 2
    assert {c.raw_id for c in plan.reclassifiable} == {"raw-reclassifiable"}
    assert {c.raw_id for c in plan.still_unknown} == {"raw-still-unknown"}
    assert plan.blob_missing == ()

    (reclassified,) = plan.reclassifiable
    assert reclassified.verdict == UnknownExportReclassificationVerdict.RECLASSIFIABLE
    assert reclassified.recovered_provider is Provider.CHATGPT
    assert reclassified.recovered_origin is Origin.CHATGPT_EXPORT
    assert reclassified.blob_size == len(reclassifiable_payload)

    (still_unknown,) = plan.still_unknown
    assert still_unknown.verdict == UnknownExportReclassificationVerdict.STILL_UNKNOWN
    assert still_unknown.recovered_provider is None
    assert still_unknown.recovered_origin is None


def test_plan_unknown_export_reclassification_scopes_to_source_path_like(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    reclassifiable_payload = _browser_capture_envelope_past_prefix(
        session_provider="claude-ai", provider_session_id="in-scope"
    )

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        _write_unknown_export_raw(
            archive,
            raw_id="raw-in-scope",
            payload=reclassifiable_payload,
            source_path="/spool/browser-capture/unknown/in-scope.json",
        )
        _write_unknown_export_raw(
            archive,
            raw_id="raw-out-of-scope",
            payload=reclassifiable_payload,
            source_path="/inbox/unrelated/out-of-scope.json",
        )
        archive.commit()

    blob_store = BlobStore(archive_root / "blob")
    conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    try:
        plan = plan_unknown_export_reclassification(conn, blob_store=blob_store)
    finally:
        conn.close()

    assert plan.scanned_count == 1
    assert {c.raw_id for c in plan.reclassifiable} == {"raw-in-scope"}

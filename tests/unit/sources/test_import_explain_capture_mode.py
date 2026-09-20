"""Capture-mode resolution reaches the import-explain read surface.

``docs/provider-origin-identity.md`` requires ImportExplain to report capture
mode and public origin as separate evidence. The durable resolution
(``source.raw_capture_observations``) existed and was tested at the storage
tier, but no read surface carried it: ``grep -rn capture_mode polylogue/api
polylogue/cli polylogue/mcp`` returned nothing (polylogue-7xg00).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import get_args

from polylogue.core.enums import Origin, Provider
from polylogue.sources.import_explain import explain_import_archive
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.source_write import write_source_raw_session
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.surfaces.payloads import CaptureModeResolutionStatus

_PAYLOAD = b'{"chunkedPrompt": {"chunks": []}}'


def _source_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.SOURCE)
    return conn


def test_surface_status_vocabulary_matches_the_storage_owner() -> None:
    """Anti-vacuity: the surface Literal is a second spelling of the storage
    tier's status vocabulary (the layering ratchet forbids importing it), so a
    new status added on one side and not the other goes red here."""

    from polylogue.storage.sqlite.archive_tiers import source_write

    assert set(get_args(CaptureModeResolutionStatus)) == set(get_args(source_write.CaptureModeResolutionStatus))


def test_import_explain_reports_every_observed_capture_mode(tmp_path: Path) -> None:
    """Anti-vacuity: read ``raw_sessions.capture_mode`` (the first-known cache)
    instead of the observation table, or drop the field from the payload, and
    this reports ``("gemini",)``/``()`` instead of both observed modes.

    A GEMINI export and a live DRIVE acquisition of byte-identical bytes share
    one content-derived ``raw_id`` and one public origin (AISTUDIO_DRIVE). The
    Provider -> Origin mapping is non-injective, so the surface cannot recover
    the acquisition mechanism from the origin -- it has to carry the durable
    observation set.
    """

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    conn = _source_db(archive_root / "source.db")
    try:
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.AISTUDIO_DRIVE,
            capture_mode=Provider.GEMINI,
            source_path="/tmp/export.json",
            source_index=0,
            payload=_PAYLOAD,
            acquired_at_ms=1_000,
        )
        write_source_raw_session(
            conn,
            origin=Origin.AISTUDIO_DRIVE,
            capture_mode=Provider.DRIVE,
            source_path="/tmp/export.json",
            source_index=0,
            payload=_PAYLOAD,
            acquired_at_ms=2_000,
            raw_id=raw_id,
        )
        conn.commit()
    finally:
        conn.close()

    payload = explain_import_archive(archive_root, raw_ref=f"raw:{raw_id}")

    assert len(payload.entries) == 1
    entry = payload.entries[0]
    assert entry.detected_origin == Origin.AISTUDIO_DRIVE.value
    assert entry.capture_mode_status == "ambiguous"
    assert entry.capture_modes == (Provider.GEMINI.value, Provider.DRIVE.value)
    assert any("capture mode is ambiguous" in caveat for caveat in entry.caveats)


def test_import_explain_reports_a_single_observed_capture_mode(tmp_path: Path) -> None:
    """Anti-vacuity: hardcode ``status`` to ``"ambiguous"`` or always emit the
    caveat and this goes red -- one observation is unambiguous, not a gap."""

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    conn = _source_db(archive_root / "source.db")
    try:
        raw_id = write_source_raw_session(
            conn,
            origin=Origin.CLAUDE_CODE_SESSION,
            capture_mode=Provider.CLAUDE_CODE,
            source_path="/tmp/record.jsonl",
            source_index=0,
            payload=b'{"kind":"session"}',
            acquired_at_ms=1_000,
        )
        conn.commit()
    finally:
        conn.close()

    entry = explain_import_archive(archive_root, raw_ref=f"raw:{raw_id}").entries[0]
    assert entry.capture_mode_status == "unambiguous"
    assert entry.capture_modes == (Provider.CLAUDE_CODE.value,)
    assert not any("capture mode is ambiguous" in caveat for caveat in entry.caveats)

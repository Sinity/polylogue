"""Terminal receipts for retained Codex state exports."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from polylogue.core.enums import Origin, Provider
from polylogue.sources import codex_state_projection
from polylogue.sources.parsers import codex_state

logger = logging.getLogger(__name__)

CODEX_STATE_CENSUS_DETAIL = "retained Codex state evidence applied"


def record_codex_state_snapshot_terminal(
    archive: Any,
    raw_id: str,
    *,
    state_path: Path,
    state_kind: str,
    source_path: str,
    acquired_at_ms: int,
    censused_at_ms: int,
    blob_hash: str | None = None,
) -> None:
    """Finalize one admitted Codex state export as terminal non-session evidence.

    A state export has no byte frontier and never yields a session, so the
    cursor-authority gate can only account for it through a terminal
    source-tier receipt: the ``non_session`` membership census plus a
    finalized parse state. Live ingest and retained-raw replay both end here
    so a raw admitted by either route satisfies the same gate.

    A ``thread_state`` export also recomputes the index-tier thread-state
    projection, which is why both routes pass through one function.
    """
    from polylogue.storage.raw_authority import RAW_AUTHORITY_PARSER_FINGERPRINT

    if state_kind == codex_state_projection.THREAD_STATE_KIND:
        codex_state_projection.apply_retained_state_export(
            archive,
            raw_id,
            export_path=state_path,
            blob_hash=blob_hash or archive.raw_revision_descriptor(raw_id)[1],
            observed_at_ms=acquired_at_ms,
        )
    archive.replace_raw_membership_census(
        raw_id,
        [],
        parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
        censused_at_ms=censused_at_ms,
        detail=CODEX_STATE_CENSUS_DETAIL,
        retire_full_revision_governance=True,
    )
    archive.mark_raw_parse_succeeded(raw_id, provider=Provider.CODEX)


def _unreceipted_codex_state_raw_ids(source_db: Path) -> list[str]:
    origin = Origin.CODEX_SESSION.value
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as conn:
        rows = conn.execute(
            """
            SELECT r.raw_id
            FROM raw_sessions AS r
            WHERE r.origin = ?
              AND r.parsed_at_ms IS NULL
              AND r.parse_error IS NULL
              AND (
                  lower(r.source_path) GLOB '*.sqlite'
                  OR lower(r.source_path) GLOB '*.sqlite3'
                  OR lower(r.source_path) GLOB '*.db'
              )
            ORDER BY r.raw_id
            """,
            (origin,),
        ).fetchall()
        candidates = [str(row[0]) for row in rows]
        if not candidates:
            return []
        placeholders = ", ".join("?" for _ in candidates)
        typed_non_session = {
            str(row[0])
            for row in conn.execute(
                f"SELECT raw_id FROM raw_artifacts WHERE parse_as_session = 0 AND raw_id IN ({placeholders})",
                candidates,
            )
        }
    return [raw_id for raw_id in candidates if raw_id not in typed_non_session]


def _thread_state_projection_is_current(archive_root: Path) -> bool:
    """Return whether the index projection already names the newest export.

    Read-only over both tiers, so an already-converged archive needs no
    writable open to prove there is nothing to do.
    """
    from polylogue.storage.archive_identity import resolve_active_index_path

    try:
        index_db = resolve_active_index_path(archive_root)
    except Exception:
        return True
    if not index_db.is_file():
        return True
    try:
        with closing(sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)) as source_conn:
            latest = codex_state_projection.latest_retained_state_export(source_conn)
        if latest is None:
            return True
        with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as index_conn:
            current = codex_state_projection.projection_provenance(index_conn)
    except sqlite3.Error:
        return True
    return current is not None and current.raw_id == latest[0] and current.blob_hash == latest[1]


def resolve_retained_codex_state_receipts(archive_root: Path) -> int:
    """Finalize admitted Codex state exports that carry no terminal receipt.

    Runs before the raw-materialization source-selection gate: the gate
    counts such a raw as an incomparable cursor row, and every route that
    could finalize it is behind the same gate. The receipt is derived from
    the immutable retained export only. Returns the number of raws finalized.

    The same pass reconciles the index-tier thread-state projection against
    the newest retained export, so a reindex that applied the export before
    the sessions it describes converges without depending on replay order.
    """
    source_db = archive_root / "source.db"
    if not source_db.is_file():
        return 0
    raw_ids = _unreceipted_codex_state_raw_ids(source_db)
    if not raw_ids and _thread_state_projection_is_current(archive_root):
        return 0
    from polylogue.sources.live.archive_open import _open_archive_for_live_write

    resolved = 0
    with _open_archive_for_live_write(archive_root) as archive:
        for raw_id in raw_ids:
            provider, blob_hash, source_path, _kind, _payload_size = archive.raw_revision_descriptor(raw_id)
            if provider is not Provider.CODEX:
                continue
            state_path = archive.blob_path_for_hash(blob_hash)
            if state_path is None:
                continue
            state_kind = codex_state.classify_codex_sqlite_path(state_path, immutable=True)
            if state_kind not in codex_state.IN_SCOPE_KINDS:
                continue
            observed_at_ms = archive.raw_revision_observed_at_ms(raw_id)
            record_codex_state_snapshot_terminal(
                archive,
                raw_id,
                state_path=state_path,
                state_kind=state_kind,
                source_path=source_path,
                acquired_at_ms=observed_at_ms,
                censused_at_ms=observed_at_ms,
                blob_hash=blob_hash,
            )
            resolved += 1
        index_conn = archive.index_connection
        if index_conn is not None:
            codex_state_projection.ensure_thread_state_projection(
                index_conn,
                archive.source_connection,
                blob_path_for_hash=archive.blob_path_for_hash,
            )
    if resolved:
        logger.info("codex state: finalized %d retained export(s) without a terminal receipt", resolved)
    return resolved


__all__ = [
    "CODEX_STATE_CENSUS_DETAIL",
    "record_codex_state_snapshot_terminal",
    "resolve_retained_codex_state_receipts",
]

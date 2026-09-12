"""Terminal receipts for retained Codex state exports."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.parse import quote

from polylogue.core.enums import Origin, Provider
from polylogue.sources import codex_state_projection
from polylogue.sources.parsers import codex_state
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.materials import admit_material, link_material

logger = logging.getLogger(__name__)

CODEX_STATE_CENSUS_DETAIL = "retained Codex state evidence applied"


def _upsert_codex_material(
    archive: Any,
    *,
    raw_id: str,
    source_path: str,
    thread_id: str,
    kind: str,
    item_id: str,
    payload: dict[str, object],
    observed_at_ms: int,
) -> None:
    """Retain one generated Codex record through the shared material route."""
    conn = archive.source_connection
    # A Codex install is the scope of state-database observations.  Names in
    # different installs are not competing revisions: R3 requires each to
    # remain independently readable.  The thread is part of the logical
    # coordinate too; it prevents a provider reusing a goal id from joining
    # two unrelated sessions.
    source_scope = str(Path(source_path).parent)
    source_uri = (
        f"codex://state/{kind}/{quote(thread_id, safe='')}/{quote(item_id, safe='')}"
        f"?scope={quote(source_scope, safe='')}"
    )
    referrer_ref = f"codex-session:{thread_id}"
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    previous = conn.execute(
        "SELECT material_id FROM material_observations WHERE source_uri = ? AND referrer_ref = ? "
        "AND acquisition_state != 'superseded' ORDER BY created_at_ms DESC, material_id DESC LIMIT 1",
        (source_uri, referrer_ref),
    ).fetchone()
    material = admit_material(
        conn,
        blob_store=BlobStore(archive.archive_root / "blob"),
        source_uri=source_uri,
        referrer_ref=referrer_ref,
        observed_at_ms=observed_at_ms,
        payload=encoded,
        media_type="application/json",
        filename=f"{kind}-{item_id}.json",
        privacy_classification="private",
        supersedes_material_id=str(previous[0]) if previous is not None else None,
    )
    if previous is not None and str(previous[0]) != material.material_id:
        conn.execute(
            "UPDATE material_observations SET acquisition_state = 'superseded' WHERE material_id = ?",
            (str(previous[0]),),
        )
    link_material(
        conn,
        material.material_id,
        raw_id,
        relation="acquired_from",
        authority="provider",
        observed_at_ms=observed_at_ms,
        source_diagnostic=f"Codex {kind} retained export {raw_id}",
    )
    link_material(
        conn,
        material.material_id,
        referrer_ref,
        relation="refers_to",
        authority="provider",
        observed_at_ms=observed_at_ms,
        source_diagnostic="Codex provider-generated state associated with its thread",
    )


def materialize_codex_state_content(
    archive: Any,
    raw_id: str,
    *,
    state_path: Path,
    source_path: str,
    state_kind: str,
    acquired_at_ms: int,
) -> None:
    """Project retained goals/memories into the existing material read model."""
    if state_kind == "goals":
        for goal in codex_state.parse_codex_goals_db(state_path, immutable=True):
            _upsert_codex_material(
                archive,
                raw_id=raw_id,
                source_path=source_path,
                thread_id=goal.thread_id,
                kind="goal",
                item_id=goal.goal_id,
                observed_at_ms=acquired_at_ms,
                payload={
                    "thread_id": goal.thread_id,
                    "goal_id": goal.goal_id,
                    "objective": goal.objective,
                    "status": goal.status,
                    "token_budget": goal.token_budget,
                    "tokens_used": goal.tokens_used,
                    "time_used_seconds": goal.time_used_seconds,
                    "created_at_ms": goal.created_at_ms,
                    "updated_at_ms": goal.updated_at_ms,
                    "provider": "codex",
                    "generated": False,
                },
            )
    elif state_kind == "memories":
        for memory in codex_state.parse_codex_memories_db(state_path, immutable=True):
            _upsert_codex_material(
                archive,
                raw_id=raw_id,
                source_path=source_path,
                thread_id=memory.thread_id,
                kind="memory",
                item_id=memory.thread_id,
                observed_at_ms=acquired_at_ms,
                payload={
                    "thread_id": memory.thread_id,
                    "raw_memory": memory.raw_memory,
                    "rollout_summary": memory.rollout_summary,
                    "source_updated_at_ms": memory.source_updated_at_ms,
                    "generated_at_ms": memory.generated_at_ms,
                    "usage_count": memory.usage_count,
                    "has_rollout_slug": memory.has_rollout_slug,
                    "selected_for_phase2": memory.selected_for_phase2,
                    "provider": "codex",
                    "generated": True,
                },
            )


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

    if state_kind in {"goals", "memories"}:
        materialize_codex_state_content(
            archive,
            raw_id,
            state_path=state_path,
            source_path=source_path,
            state_kind=state_kind,
            acquired_at_ms=acquired_at_ms,
        )
    if state_kind == codex_state_projection.THREAD_STATE_KIND:
        codex_state_projection.apply_retained_state_export(
            archive,
            raw_id,
            export_path=state_path,
            blob_hash=blob_hash or archive.raw_revision_descriptor(raw_id)[1],
            observed_at_ms=acquired_at_ms,
            source_path=source_path,
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
            latest = codex_state_projection.latest_retained_state_exports(source_conn)
        if not latest:
            return True
        with closing(sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)) as index_conn:
            current = {
                export.source_scope: codex_state_projection.projection_provenance(
                    index_conn, source_scope=export.source_scope
                )
                for export in latest
            }
    except sqlite3.Error as exc:
        logger.debug("codex state: could not compare the retained export against the projection: %s", exc)
        return True
    for export in latest:
        projection = current[export.source_scope]
        if projection is None:
            return False
        if (
            projection.raw_id != export.raw_id
            or projection.blob_hash != export.blob_hash
            or (projection.observed_at_ms, projection.observation_order)
            != (export.observed_at_ms, export.observation_order)
        ):
            return False
    return True


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
        # ``close`` does not commit, and the projection is the last write in
        # this pass: without this the recomputed rows are discarded.
        archive.commit()
    if resolved:
        logger.info("codex state: finalized %d retained export(s) without a terminal receipt", resolved)
    return resolved


__all__ = [
    "CODEX_STATE_CENSUS_DETAIL",
    "materialize_codex_state_content",
    "record_codex_state_snapshot_terminal",
    "resolve_retained_codex_state_receipts",
]

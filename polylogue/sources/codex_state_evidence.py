"""Durable session-linked evidence derived from retained Codex state."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from json import dumps as json_dumps
from pathlib import Path
from typing import Any

from polylogue.core.enums import Origin, Provider
from polylogue.sources.parsers import codex_state

logger = logging.getLogger(__name__)

CODEX_STATE_CENSUS_DETAIL = "retained Codex state evidence applied"


def write_codex_thread_state_evidence(
    archive: Any,
    snapshot: codex_state.CodexStateSnapshot,
    *,
    source_path: str,
    acquired_at_ms: int,
    observation_order: int | None = None,
) -> None:
    """Attach state-db thread metadata to existing Codex sessions.

    The state database is evidence about sessions, never a session itself.
    Both live ingest and retained-raw replay call this writer so a
    source-only acquisition is completed from the immutable snapshot when
    the derived tier returns.
    """
    from polylogue.storage.sqlite.archive_tiers.source_write import ArchiveHookEvent

    def event_identity(base: str) -> str:
        # A state snapshot is a durable observation, not a mutable hook
        # envelope. Once source v36 made hook-event identity immutable, a
        # stable per-thread id could no longer represent A -> B -> A state
        # observations without either losing history or raising a conflict.
        # Raw-payload receipt order is the acquisition authority used by
        # retained replay, so carry it into the event identity.
        if observation_order is None:
            return base
        return f"{base}:observation-{observation_order:020d}"

    for thread in snapshot.threads:
        payload: dict[str, object] = {
            "thread_id": thread.thread_id,
            "title": thread.title,
            "cwd": thread.cwd,
            "source": thread.source,
            "model": thread.model,
            "agent_nickname": thread.agent_nickname,
            "agent_role": thread.agent_role,
            "archived": thread.archived,
        }
        encoded = json_dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        archive.write_hook_event(
            provider=Provider.CODEX,
            payload=encoded,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            hook_event=ArchiveHookEvent(
                hook_event_id=event_identity(f"codex-thread-title:{thread.thread_id}"),
                origin=Origin.CODEX_SESSION,
                source_path=source_path,
                event_type="codex_thread_title",
                payload=payload,
                observed_at_ms=acquired_at_ms,
                native_id=f"{thread.thread_id}:codex_thread_title",
                session_native_id=thread.thread_id,
            ),
            carrier_relative_path=f"{source_path}::{event_identity(f'codex-thread-title:{thread.thread_id}')}",
        )
    for edge in snapshot.spawn_edges:
        payload = {
            "parent_thread_id": edge.parent_thread_id,
            "child_thread_id": edge.child_thread_id,
            "status": edge.status,
        }
        encoded = json_dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        archive.write_hook_event(
            provider=Provider.CODEX,
            payload=encoded,
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            hook_event=ArchiveHookEvent(
                hook_event_id=event_identity(f"codex-thread-spawn-edge:{edge.parent_thread_id}:{edge.child_thread_id}"),
                origin=Origin.CODEX_SESSION,
                source_path=source_path,
                event_type="codex_thread_spawn_edge",
                payload=payload,
                observed_at_ms=acquired_at_ms,
                native_id=f"{edge.parent_thread_id}:{edge.child_thread_id}:codex_thread_spawn_edge",
                session_native_id=edge.parent_thread_id,
            ),
            carrier_relative_path=(
                f"{source_path}::{event_identity(f'codex-thread-spawn-edge:{edge.parent_thread_id}:{edge.child_thread_id}')}"
            ),
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
) -> None:
    """Finalize one admitted Codex state snapshot as terminal non-session evidence.

    A state snapshot has no byte frontier and never yields a session, so the
    cursor-authority gate can only account for it through a terminal
    source-tier receipt: the ``non_session`` membership census plus a
    finalized parse state. Live ingest and retained-raw replay both end here
    so a raw admitted by either route satisfies the same gate.
    """
    from polylogue.storage.raw_authority import RAW_AUTHORITY_PARSER_FINGERPRINT

    if state_kind == "thread_state":
        write_codex_thread_state_evidence(
            archive,
            codex_state.parse_codex_state_db(state_path, immutable=True),
            source_path=source_path,
            acquired_at_ms=acquired_at_ms,
            observation_order=archive.raw_revision_observation_order(raw_id)[1],
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
        receipted = {
            str(row[0])
            for row in conn.execute(
                f"""
                SELECT raw_id FROM raw_membership_census WHERE raw_id IN ({placeholders})
                UNION
                SELECT raw_id FROM raw_artifacts WHERE parse_as_session = 0 AND raw_id IN ({placeholders})
                """,
                (*candidates, *candidates),
            )
        }
    return [raw_id for raw_id in candidates if raw_id not in receipted]


def resolve_retained_codex_state_receipts(archive_root: Path) -> int:
    """Finalize admitted Codex state snapshots that carry no terminal receipt.

    Runs before the raw-materialization source-selection gate: the gate
    counts such a raw as an incomparable cursor row, and every route that
    could finalize it is behind the same gate. The receipt is derived from
    the immutable retained blob only; nothing here touches the derived tier.
    Returns the number of raws finalized.
    """
    source_db = archive_root / "source.db"
    if not source_db.is_file():
        return 0
    raw_ids = _unreceipted_codex_state_raw_ids(source_db)
    if not raw_ids:
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
            )
            resolved += 1
    if resolved:
        logger.info("codex state: finalized %d retained snapshot(s) without a terminal receipt", resolved)
    return resolved


__all__ = [
    "CODEX_STATE_CENSUS_DETAIL",
    "record_codex_state_snapshot_terminal",
    "resolve_retained_codex_state_receipts",
    "write_codex_thread_state_evidence",
]

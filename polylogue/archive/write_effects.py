"""Consolidated archive write side effects.

This module is the ONLY place where post-write side effects run:
- FTS repair for changed session IDs
- Search cache invalidation

Every archive write path MUST route through this module or the
ArchiveWriteGateway that wraps it.

Effects are declared entries in ``WRITE_EFFECT_REGISTRY`` (polylogue-0aj),
not inlined branches in ``commit_archive_write_effects``. Each
``WriteEffect`` fixes three things that used to be implicit and had to be
re-derived by reading the function body: *when* it runs relative to the
commit boundary (``phase``), *whether* it runs for this write
(``should_run``), and *what happens on failure* (``failure_policy``). Adding
a new post-commit consumer (embedding-scheduling, SSE announce, daemon cache
invalidation — polylogue-yp0's bus subscribers) means adding a registry
entry here, not re-reasoning the whole choke point's ordering by hand.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from polylogue.archive.write_gateway import (
    WriteEffectReceipt,
    WriteOperation,
    WriteResult,
    write_operation_policy_for,
)
from polylogue.logging import ERROR, WARNING, emit

WriteEffectPhase = Literal["in-transaction", "post-commit"]
"""When a ``WriteEffect`` runs relative to the commit boundary.

- ``in-transaction``: runs before ``conn.commit()``, inside the same
  transaction as the row writes (atomicity argument — FTS trigger
  drop/restore must not straddle a commit, see docs/internals.md).
- ``post-commit``: runs after ``conn.commit()``, on the same connection.
"""

WriteEffectFailurePolicy = Literal["abort", "log-and-continue"]
"""What happens when a ``WriteEffect.run`` raises.

- ``abort``: propagate — the caller's transaction/commit is not safe to
  continue past this effect.
- ``log-and-continue``: log the exception and continue to the next effect.
  Reserved for effects whose failure must not poison unrelated effects (the
  historical blob-lease-leak bug class this module used to carry — see the
  removed-lease note at the bottom of this docstring set).
"""


@dataclass(frozen=True, slots=True)
class WriteEffectContext:
    """Per-commit state threaded through the write-effect registry."""

    conn: sqlite3.Connection
    op: WriteOperation
    payload: dict[str, Any]
    changed_session_ids: tuple[str, ...]
    staleness_key: str
    run_archive_effects: bool


def _always_run(_ctx: WriteEffectContext) -> bool:
    return True


@dataclass(frozen=True, slots=True)
class WriteEffect:
    """One declared post-write side effect run by ``commit_archive_write_effects``.

    ``run`` and ``should_run`` receive the shared ``WriteEffectContext`` for
    this commit rather than closing over ad hoc locals, so a new effect's
    inputs are visible from its signature.
    """

    name: str
    phase: WriteEffectPhase
    run: Callable[[WriteEffectContext], None]
    should_run: Callable[[WriteEffectContext], bool] = _always_run
    failure_policy: WriteEffectFailurePolicy = "abort"


def _ensure_fts_triggers_effect(ctx: WriteEffectContext) -> None:
    from polylogue.storage.fts.fts_lifecycle import ensure_fts_triggers_sync

    ensure_fts_triggers_sync(ctx.conn)


def _repair_message_fts_should_run(ctx: WriteEffectContext) -> bool:
    return bool(ctx.changed_session_ids) and bool(ctx.payload.get("repair_message_fts", True))


def _repair_message_fts_effect(ctx: WriteEffectContext) -> None:
    from polylogue.storage.fts.fts_lifecycle import repair_message_fts_index_sync

    repair_message_fts_index_sync(ctx.conn, ctx.changed_session_ids)


def _invalidate_search_cache_should_run(ctx: WriteEffectContext) -> bool:
    return bool(ctx.changed_session_ids)


def _invalidate_search_cache_effect(_ctx: WriteEffectContext) -> None:
    from polylogue.storage.search.cache import invalidate_search_cache

    invalidate_search_cache()


def _invalidate_insights_should_run(ctx: WriteEffectContext) -> bool:
    return bool(ctx.changed_session_ids)


def _invalidate_insights_effect(ctx: WriteEffectContext) -> None:
    """Invalidate derived inputs on the admitted archive transaction."""
    session_ids = ctx.changed_session_ids
    # Keep each statement below SQLite's variable limit while preserving the
    # caller-owned transaction and its single-writer admission.
    for start in range(0, len(session_ids), 500):
        chunk = session_ids[start : start + 500]
        placeholders = ", ".join("?" for _ in chunk)
        ctx.conn.execute(
            f"UPDATE session_profiles SET source_sort_key = NULL, source_updated_at = NULL "
            f"WHERE session_id IN ({placeholders})",
            chunk,
        )


def _announce_ingest_should_run(ctx: WriteEffectContext) -> bool:
    return bool(ctx.changed_session_ids)


def _announce_ingest_effect(ctx: WriteEffectContext) -> None:
    """Announce a committed archive write on the daemon's in-process bus.

    Post-commit, never before: a subscriber that woke on an announcement of an
    uncommitted write would read rows that may still roll back. Delivery is
    best-effort by the bus' own contract and its consumers keep a slow
    reconciliation tick, so a missed announcement costs latency, not work
    (polylogue-14t7).
    """
    from polylogue.daemon.event_bus import IngestCommitted, daemon_event_bus

    daemon_event_bus().publish(
        IngestCommitted(
            cursor=ctx.staleness_key or None,
            session_refs=tuple(ctx.changed_session_ids),
        )
    )


WRITE_EFFECT_REGISTRY: tuple[WriteEffect, ...] = (
    WriteEffect(
        name="ensure_fts_triggers",
        phase="in-transaction",
        run=_ensure_fts_triggers_effect,
    ),
    WriteEffect(
        name="repair_message_fts",
        phase="in-transaction",
        run=_repair_message_fts_effect,
        should_run=_repair_message_fts_should_run,
    ),
    WriteEffect(
        name="invalidate_search_cache",
        phase="post-commit",
        run=_invalidate_search_cache_effect,
        should_run=_invalidate_search_cache_should_run,
    ),
    WriteEffect(
        name="announce_ingest_committed",
        phase="post-commit",
        run=_announce_ingest_effect,
        should_run=_announce_ingest_should_run,
        # An announcement that fails must never fail a committed archive write.
        failure_policy="log-and-continue",
    ),
    WriteEffect(
        name="invalidate_session_insights",
        phase="in-transaction",
        run=_invalidate_insights_effect,
        should_run=_invalidate_insights_should_run,
        failure_policy="log-and-continue",
    ),
)
"""Ordered, declared effects for the archive write choke point.

The registry declares the transaction and post-commit effects. An earlier
revision of the choke point also
acquired/released a blob-GC lease here, keyed by
``_blob_hashes``/``_operation_id`` payload entries. No production caller
ever populated those keys (polylogue-v7e0), so the branch never executed;
it was removed rather than left as unreachable code or ported into this
registry as a fourth do-nothing entry. Publication reservations and GC's
final locked liveness/reservation recheck protect a blob write racing a
concurrent ``blob-gc`` run; the age floor remains defense-in-depth. See
``docs/internals.md`` "GC concurrency model" for the current contract.
"""


def _run_registered_effects(
    registry: Sequence[WriteEffect],
    phase: WriteEffectPhase,
    ctx: WriteEffectContext,
    timings: dict[str, float],
) -> list[WriteEffectReceipt]:
    receipts: list[WriteEffectReceipt] = []
    for effect in registry:
        if effect.phase != phase:
            continue
        if not effect.should_run(ctx):
            receipts.append(WriteEffectReceipt(effect.name, effect.phase, "skipped"))
            continue
        started_at = time.perf_counter()
        try:
            effect.run(ctx)
        except Exception as exc:
            if effect.failure_policy == "log-and-continue":
                emit(
                    "archive.write_effect.failed",
                    level=ERROR,
                    outcome="error",
                    effect=effect.name,
                    phase=effect.phase,
                    operation=ctx.op.value,
                    error_type=type(exc).__name__,
                    error_detail=str(exc),
                )
                receipts.append(WriteEffectReceipt(effect.name, effect.phase, "failed", error=str(exc)))
                continue
            raise
        timings[effect.name] = time.perf_counter() - started_at
        receipts.append(WriteEffectReceipt(effect.name, effect.phase, "applied"))
    return receipts


def commit_archive_write_effects(
    conn: sqlite3.Connection,
    op: WriteOperation,
    payload: dict[str, Any],
) -> WriteResult:
    """Run the canonical post-write side effects for an archive write.

    Walks ``WRITE_EFFECT_REGISTRY`` in declaration order: every
    ``in-transaction`` effect whose ``should_run`` passes, then
    ``conn.commit()``, then every ``post-commit`` effect whose
    ``should_run`` passes.

        Parameters
        ----------
        conn:
            Open SQLite connection. The caller owns the connection lifecycle.
        op:
            Write operation type (ingest, delete, tag_update, etc.).
        payload:
            Operation payload. Expected keys:
            - ``changed_session_ids``: sequence of session IDs whose
              FTS rows should be repaired.
            - ``effect_scope``: ``"archive-index"`` (the default) runs
              registered index effects; ``"user-overlay"`` commits a
              declared user.db writer without index effects.
            - ``repair_message_fts``: bool, default True. Only a writer that
              has already settled the FTS rows of ``changed_session_ids``
              itself sets it False (session deletion clears them before the
              rows they index are gone).
            - ``_connection``: (optional) forwarded from the gateway when an
              external connection is already in use.

        Returns
        -------
        WriteResult with status, rows_affected, and operation_id.
    """
    changed_ids: Sequence[str] = payload.get("changed_session_ids", [])
    sorted_ids: tuple[str, ...] = tuple(sorted(set(changed_ids))) if changed_ids else ()
    effect_scope = payload.get("effect_scope", "archive-index")
    policy = write_operation_policy_for(op, effect_scope)
    if "_db_path" not in payload:
        database_row = conn.execute("PRAGMA database_list").fetchall()
        if database_row and database_row[0][2]:
            payload = {**payload, "_db_path": database_row[0][2]}
    archive_identity = str(payload.get("_db_path", ""))
    staleness_key = f"{archive_identity}:{effect_scope}:{op.value}:{','.join(sorted_ids)}"
    ctx = WriteEffectContext(
        conn=conn,
        op=op,
        payload=payload,
        changed_session_ids=sorted_ids,
        staleness_key=staleness_key,
        run_archive_effects=policy.run_archive_effects,
    )

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    receipts = (
        _run_registered_effects(WRITE_EFFECT_REGISTRY, "in-transaction", ctx, timings)
        if policy.run_archive_effects
        else []
    )
    t_commit = time.perf_counter()
    from polylogue.storage.sqlite.reference_seal import current_index_mutation_scope

    mutation_scope = current_index_mutation_scope()
    if mutation_scope is not None and mutation_scope.conn is conn:
        mutation_scope.commit()
    else:
        conn.commit()
    commit_elapsed_s = time.perf_counter() - t_commit
    if policy.run_archive_effects:
        receipts.extend(_run_registered_effects(WRITE_EFFECT_REGISTRY, "post-commit", ctx, timings))
    total_effect_elapsed_s = time.perf_counter() - t0

    if total_effect_elapsed_s >= 1.0:
        emit(
            "archive.write_effects.slow",
            level=WARNING,
            outcome="degraded",
            reason="effects_exceeded_one_second",
            operation=op.value,
            sessions=len(sorted_ids),
            elapsed_ms=round(commit_elapsed_s * 1000, 3),
            duration_ms=round(total_effect_elapsed_s * 1000, 3),
        )
        for effect_name, elapsed in timings.items():
            emit(
                "archive.write_effect.timing",
                level=WARNING,
                effect=effect_name,
                operation=op.value,
                duration_ms=round(elapsed * 1000, 3),
            )

    return WriteResult(
        operation_id=str(uuid4()),
        operation=op,
        rows_affected=len(sorted_ids),
        status="committed",
        effect_receipts=tuple(receipts),
    )


__all__ = [
    "WRITE_EFFECT_REGISTRY",
    "WriteEffect",
    "WriteEffectContext",
    "WriteEffectReceipt",
    "WriteEffectFailurePolicy",
    "WriteEffectPhase",
    "commit_archive_write_effects",
]

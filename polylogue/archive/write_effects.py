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

import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from polylogue.archive.write_gateway import WriteOperation, WriteResult

logger = logging.getLogger(__name__)

WriteEffectPhase = Literal["in-transaction", "post-commit", "async-deferred"]
"""When a ``WriteEffect`` runs relative to the commit boundary.

- ``in-transaction``: runs before ``conn.commit()``, inside the same
  transaction as the row writes (atomicity argument — FTS trigger
  drop/restore must not straddle a commit, see docs/internals.md).
- ``post-commit``: runs after ``conn.commit()``, on the same connection.
- ``async-deferred``: queued after commit and delivered outside the request
  path. Deferred failures are recorded as retryable and cannot affect the
  committed transaction.
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
    deferred_scheduler: Callable[[WriteEffect, WriteEffectContext], None] | None = None


@dataclass(frozen=True, slots=True)
class WriteEffectReceipt:
    name: str
    phase: WriteEffectPhase
    disposition: Literal["applied", "enqueued", "skipped", "failed"]
    retryable: bool = False
    error: str | None = None


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


class DeferredEffectQueue:
    """Bounded process-local delivery for effects that must not delay writes."""

    def __init__(self, *, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="polylogue-write-effect")
        self._lock = threading.Lock()
        self._pending: set[str] = set()
        self._receipts: dict[str, WriteEffectReceipt] = {}

    def enqueue(self, effect: WriteEffect, ctx: WriteEffectContext) -> None:
        key = f"{effect.name}:{ctx.staleness_key}"
        with self._lock:
            if key in self._pending:
                return
            self._pending.add(key)
            self._receipts[key] = WriteEffectReceipt(effect.name, effect.phase, "enqueued", retryable=True)
        self._executor.submit(self._deliver, key, effect, ctx)

    def _deliver(self, key: str, effect: WriteEffect, ctx: WriteEffectContext) -> None:
        try:
            effect.run(ctx)
        except Exception as exc:
            logger.exception("deferred_write_effect_failed effect=%s key=%s", effect.name, key)
            with self._lock:
                self._pending.discard(key)
                self._receipts[key] = WriteEffectReceipt(
                    effect.name, effect.phase, "failed", retryable=True, error=str(exc)
                )
            return
        with self._lock:
            self._pending.discard(key)
            self._receipts[key] = WriteEffectReceipt(effect.name, effect.phase, "applied")

    def receipt(self, effect_name: str, staleness_key: str) -> WriteEffectReceipt | None:
        with self._lock:
            return self._receipts.get(f"{effect_name}:{staleness_key}")


DEFERRED_EFFECT_QUEUE = DeferredEffectQueue()


def _ensure_fts_triggers_effect(ctx: WriteEffectContext) -> None:
    from polylogue.storage.fts.fts_lifecycle import ensure_fts_triggers_sync

    ensure_fts_triggers_sync(ctx.conn)


def _repair_message_fts_should_run(ctx: WriteEffectContext) -> bool:
    return bool(ctx.changed_session_ids) and bool(ctx.payload.get("repair_message_fts", True))


def _repair_message_fts_effect(ctx: WriteEffectContext) -> None:
    from polylogue.storage.fts.fts_lifecycle import repair_message_fts_index_sync

    repair_message_fts_index_sync(ctx.conn, ctx.changed_session_ids, record_exact_snapshot=False)


def _invalidate_search_cache_should_run(ctx: WriteEffectContext) -> bool:
    return bool(ctx.changed_session_ids)


def _invalidate_search_cache_effect(_ctx: WriteEffectContext) -> None:
    from polylogue.storage.search.cache import invalidate_search_cache

    invalidate_search_cache()


def _invalidate_insights_should_run(ctx: WriteEffectContext) -> bool:
    return bool(ctx.changed_session_ids)


def _invalidate_insights_effect(ctx: WriteEffectContext) -> None:
    """Mark session insight inputs stale in a separate post-commit connection."""
    db_path = ctx.payload.get("_db_path")
    if not db_path:
        raise RuntimeError("deferred insight invalidation requires _db_path")
    from polylogue.storage.sqlite.connection import open_connection

    with open_connection(db_path) as conn:
        placeholders = ", ".join("?" for _ in ctx.changed_session_ids)
        conn.execute(
            f"UPDATE session_profiles SET source_sort_key = NULL, source_updated_at = NULL "
            f"WHERE session_id IN ({placeholders})",
            ctx.changed_session_ids,
        )
        conn.commit()


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
        name="invalidate_session_insights",
        phase="async-deferred",
        run=_invalidate_insights_effect,
        should_run=_invalidate_insights_should_run,
        failure_policy="log-and-continue",
    ),
)
"""Ordered, declared effects for the archive write choke point.

These three entries are a behavior-identical extraction of what
``commit_archive_write_effects`` used to inline directly (slice 1 of
polylogue-0aj). An earlier revision of the choke point also
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
        if phase == "async-deferred":
            scheduler = ctx.deferred_scheduler or DEFERRED_EFFECT_QUEUE.enqueue
            try:
                scheduler(effect, ctx)
            except Exception as exc:
                logger.exception("deferred_write_effect_enqueue_failed effect=%s", effect.name)
                receipts.append(WriteEffectReceipt(effect.name, effect.phase, "failed", retryable=True, error=str(exc)))
            else:
                receipts.append(WriteEffectReceipt(effect.name, effect.phase, "enqueued", retryable=True))
            continue
        started_at = time.perf_counter()
        try:
            effect.run(ctx)
        except Exception:
            if effect.failure_policy == "log-and-continue":
                logger.exception(
                    "write_effect_failed effect=%s phase=%s operation=%s",
                    effect.name,
                    effect.phase,
                    ctx.op.value,
                )
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
            - ``repair_message_fts``: bool, default True — set False to skip
              the message-FTS repair effect even when session IDs changed.
            - ``_connection``: (optional) forwarded from the gateway when an
              external connection is already in use.

        Returns
        -------
        WriteResult with status, rows_affected, and operation_id.
    """
    changed_ids: Sequence[str] = payload.get("changed_session_ids", [])
    sorted_ids: tuple[str, ...] = tuple(sorted(set(changed_ids))) if changed_ids else ()
    if "_db_path" not in payload:
        database_row = conn.execute("PRAGMA database_list").fetchall()
        if database_row and database_row[0][2]:
            payload = {**payload, "_db_path": database_row[0][2]}
    staleness_key = f"{op.value}:{','.join(sorted_ids)}"
    ctx = WriteEffectContext(
        conn=conn,
        op=op,
        payload=payload,
        changed_session_ids=sorted_ids,
        staleness_key=staleness_key,
        deferred_scheduler=payload.get("deferred_scheduler"),
    )

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    receipts = _run_registered_effects(WRITE_EFFECT_REGISTRY, "in-transaction", ctx, timings)
    t_commit = time.perf_counter()
    conn.commit()
    commit_elapsed_s = time.perf_counter() - t_commit
    receipts.extend(_run_registered_effects(WRITE_EFFECT_REGISTRY, "post-commit", ctx, timings))
    receipts.extend(_run_registered_effects(WRITE_EFFECT_REGISTRY, "async-deferred", ctx, timings))
    total_effect_elapsed_s = time.perf_counter() - t0

    if total_effect_elapsed_s >= 1.0:
        effect_breakdown = " ".join(f"{name}_s={elapsed:.3f}" for name, elapsed in timings.items())
        logger.info(
            "slow_archive_write_effects operation=%s sessions=%d %s commit_s=%.3f total_s=%.3f",
            op.value,
            len(sorted_ids),
            effect_breakdown,
            commit_elapsed_s,
            total_effect_elapsed_s,
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
    "DeferredEffectQueue",
    "DEFERRED_EFFECT_QUEUE",
    "WriteEffectFailurePolicy",
    "WriteEffectPhase",
    "commit_archive_write_effects",
]

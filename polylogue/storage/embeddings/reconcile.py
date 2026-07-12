"""Cross-tier reconciliation for orphaned embedding rows.

``embeddings.db`` is a rebuildable tier (see ``docs/architecture.md``), but it
is not rebuilt in lockstep with ``index.db``. A full re-ingest, a targeted
``ops reset --index``, or a provider full-replace parse can leave
``message_embeddings_meta`` / ``message_embeddings`` (the ``vec0`` table) /
``embedding_status`` rows in ``embeddings.db`` pointing at message and session
identities that no longer exist in the rebuilt ``index.db``. These orphan rows
inflate coverage counters, waste storage, and made approximate coverage
report over 100% (polylogue-1dk1: 675,825 meta rows vs 675,725 status-summed
embedded messages, 11,348 orphan message ids, 6 orphan session status rows on
the 2026-07-10 audit).

Reconciliation here is strictly identity-scoped and applies three guards:

* **Identity guard** — a row is only a reconciliation candidate when its
  ``message_id``/``session_id`` is absent from the live index (a ``NOT
  EXISTS`` join against the attached ``index.db``). This is the sole
  deletion trigger.
* **Content-hash guard** — a ``content_hash`` mismatch on a ``message_id``
  that *does* still exist in the index is NOT an orphan. That is
  stale-but-alive vector supersession (a changed message body at a stable
  position), owned by the changed-text re-embed path (polylogue-0k6) via
  ``needs_reindex`` + in-place upsert, not this reconciler. This reconciler
  never treats a content-hash mismatch as a deletion trigger — only the
  identity join does — so an active vector for a message that still exists is
  always preserved even while it is stale and pending re-embed.
* **Quiet-window guard** — a candidate is skipped when its own
  ``embedded_at_ms``/``last_embedded_at_ms`` is newer than
  ``quiet_window_ms``. Index materialization for a full-replace session is
  not guaranteed atomic across every message row, so a message can be
  transiently absent from ``index.db`` mid-write; the quiet window keeps
  reconciliation from racing a session whose embeddings were only just
  written or that is still settling from an in-flight rebuild/replace.

Bounded via ``max_count`` so a daemon convergence pass or CLI break-glass
invocation can run repeatedly (idempotent — an already-clean archive reports
zero orphans and mutates nothing) until ``more_pending`` is ``False``.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_QUIET_WINDOW_MS = 5 * 60 * 1000  # 5 minutes
DEFAULT_SAMPLE_SIZE = 30
DEFAULT_MAX_COUNT = 500


@dataclass(frozen=True, slots=True)
class EmbeddingOrphanSample:
    """One representative orphan row surfaced for operator inspection."""

    kind: str  # "message" | "status"
    message_id: str | None
    session_id: str | None
    content_hash_hex: str | None
    action: str  # "would_remove" | "removed" | "skipped_recent"

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "content_hash_hex": self.content_hash_hex,
            "action": self.action,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingOrphanReconcileReport:
    """Inspect/reconcile report for one bounded reconciliation pass."""

    index_db: str
    embeddings_db: str
    dry_run: bool
    now_ms: int
    quiet_window_ms: int
    scanned_message_meta_rows: int
    scanned_status_rows: int
    orphan_message_rows: int
    orphan_status_rows: int
    skipped_recent_message_rows: int
    skipped_recent_status_rows: int
    removed_message_rows: int
    removed_vector_rows: int
    removed_status_rows: int
    sessions_recounted: int
    more_pending: bool
    samples: tuple[EmbeddingOrphanSample, ...]

    @property
    def ok(self) -> bool:
        """Whether this pass fully drained the orphan backlog it observed."""
        return not self.more_pending

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "index_db": self.index_db,
            "embeddings_db": self.embeddings_db,
            "dry_run": self.dry_run,
            "now_ms": self.now_ms,
            "quiet_window_ms": self.quiet_window_ms,
            "scanned_message_meta_rows": self.scanned_message_meta_rows,
            "scanned_status_rows": self.scanned_status_rows,
            "orphan_message_rows": self.orphan_message_rows,
            "orphan_status_rows": self.orphan_status_rows,
            "skipped_recent_message_rows": self.skipped_recent_message_rows,
            "skipped_recent_status_rows": self.skipped_recent_status_rows,
            "removed_message_rows": self.removed_message_rows,
            "removed_vector_rows": self.removed_vector_rows,
            "removed_status_rows": self.removed_status_rows,
            "sessions_recounted": self.sessions_recounted,
            "more_pending": self.more_pending,
            "samples": [sample.to_dict() for sample in self.samples],
        }


def inspect_embedding_orphans(
    index_db_path: str | Path,
    embeddings_db_path: str | Path | None = None,
    *,
    quiet_window_ms: int = DEFAULT_QUIET_WINDOW_MS,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    now_ms: int | None = None,
) -> EmbeddingOrphanReconcileReport:
    """Read-only orphan census — the CLI/MCP break-glass inspect surface."""

    return reconcile_embedding_orphans(
        index_db_path,
        embeddings_db_path,
        dry_run=True,
        max_count=None,
        sample_size=sample_size,
        quiet_window_ms=quiet_window_ms,
        now_ms=now_ms,
    )


def reconcile_embedding_orphans(
    index_db_path: str | Path,
    embeddings_db_path: str | Path | None = None,
    *,
    dry_run: bool = True,
    max_count: int | None = DEFAULT_MAX_COUNT,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    quiet_window_ms: int = DEFAULT_QUIET_WINDOW_MS,
    now_ms: int | None = None,
) -> EmbeddingOrphanReconcileReport:
    """Reconcile one bounded batch of orphan embedding rows.

    Idempotent and resumable: safe to call repeatedly (a clean archive
    reports zero orphans and performs no writes), and a batch capped by
    ``max_count`` leaves ``more_pending=True`` so a caller (daemon
    convergence tick, CLI retry) can drain the rest across subsequent calls.
    """

    index_path = Path(index_db_path)
    embeddings_path = (
        Path(embeddings_db_path) if embeddings_db_path is not None else index_path.with_name("embeddings.db")
    )
    resolved_now_ms = now_ms if now_ms is not None else int(time.time() * 1000)

    if not embeddings_path.exists():
        return EmbeddingOrphanReconcileReport(
            index_db=str(index_path),
            embeddings_db=str(embeddings_path),
            dry_run=dry_run,
            now_ms=resolved_now_ms,
            quiet_window_ms=quiet_window_ms,
            scanned_message_meta_rows=0,
            scanned_status_rows=0,
            orphan_message_rows=0,
            orphan_status_rows=0,
            skipped_recent_message_rows=0,
            skipped_recent_status_rows=0,
            removed_message_rows=0,
            removed_vector_rows=0,
            removed_status_rows=0,
            sessions_recounted=0,
            more_pending=False,
            samples=(),
        )

    conn = sqlite3.connect(embeddings_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        from polylogue.storage.sqlite.sqlite_vec_extension import try_load_sqlite_vec

        loaded, error = try_load_sqlite_vec(conn)
        if not loaded:
            raise RuntimeError("embedding orphan reconciliation requires sqlite-vec") from error

        conn.execute("ATTACH DATABASE ? AS idx", (str(index_path),))

        scanned_message_meta_rows = _scalar(conn, "SELECT COUNT(*) FROM message_embeddings_meta")
        scanned_status_rows = _scalar(conn, "SELECT COUNT(*) FROM embedding_status")

        message_rows = conn.execute(
            """
            SELECT em.message_id AS message_id,
                   me.session_id AS session_id,
                   em.content_hash AS content_hash,
                   em.embedded_at_ms AS embedded_at_ms
            FROM message_embeddings_meta AS em
            LEFT JOIN message_embeddings AS me ON me.message_id = em.message_id
            WHERE NOT EXISTS (
                SELECT 1 FROM idx.messages AS im WHERE im.message_id = em.message_id
            )
            ORDER BY em.message_id
            """
        ).fetchall()

        message_candidates: list[sqlite3.Row] = []
        skipped_recent_message = 0
        for row in message_rows:
            if _is_recent(row["embedded_at_ms"], resolved_now_ms, quiet_window_ms):
                skipped_recent_message += 1
                continue
            message_candidates.append(row)

        limited_message = message_candidates if max_count is None else message_candidates[: max(0, max_count)]

        status_rows = conn.execute(
            """
            SELECT session_id, last_embedded_at_ms
            FROM embedding_status
            WHERE NOT EXISTS (
                SELECT 1 FROM idx.sessions AS ims WHERE ims.session_id = embedding_status.session_id
            )
            ORDER BY session_id
            """
        ).fetchall()

        status_candidates: list[sqlite3.Row] = []
        skipped_recent_status = 0
        for row in status_rows:
            if _is_recent(row["last_embedded_at_ms"], resolved_now_ms, quiet_window_ms):
                skipped_recent_status += 1
                continue
            status_candidates.append(row)

        status_budget = None if max_count is None else max(0, max_count - len(limited_message))
        limited_status = status_candidates if status_budget is None else status_candidates[:status_budget]

        removed_message_rows = 0
        removed_vector_rows = 0
        removed_status_rows = 0
        sessions_recounted = 0
        action_label = "would_remove" if dry_run else "removed"

        if not dry_run and limited_message:
            with conn:
                for row in limited_message:
                    conn.execute("DELETE FROM message_embeddings_meta WHERE message_id = ?", (row["message_id"],))
                    conn.execute("DELETE FROM message_embeddings WHERE message_id = ?", (row["message_id"],))
            removed_message_rows = len(limited_message)
            removed_vector_rows = len(limited_message)

            affected_sessions = sorted({row["session_id"] for row in limited_message if row["session_id"]})
            if affected_sessions:
                with conn:
                    for session_id in affected_sessions:
                        remaining = _scalar(
                            conn, "SELECT COUNT(*) FROM message_embeddings WHERE session_id = ?", (session_id,)
                        )
                        conn.execute(
                            "UPDATE embedding_status SET message_count_embedded = ? WHERE session_id = ?",
                            (remaining, session_id),
                        )
                sessions_recounted = len(affected_sessions)

        if not dry_run and limited_status:
            with conn:
                for row in limited_status:
                    conn.execute("DELETE FROM embedding_status WHERE session_id = ?", (row["session_id"],))
            removed_status_rows = len(limited_status)

        samples: list[EmbeddingOrphanSample] = [
            EmbeddingOrphanSample(
                kind="message",
                message_id=row["message_id"],
                session_id=row["session_id"],
                content_hash_hex=bytes(row["content_hash"]).hex() if row["content_hash"] is not None else None,
                action=action_label,
            )
            for row in limited_message[:sample_size]
        ]
        remaining_sample_budget = max(0, sample_size - len(samples))
        samples.extend(
            EmbeddingOrphanSample(
                kind="status",
                message_id=None,
                session_id=row["session_id"],
                content_hash_hex=None,
                action=action_label,
            )
            for row in limited_status[:remaining_sample_budget]
        )

        orphan_message_rows = len(message_candidates) + skipped_recent_message
        orphan_status_rows = len(status_candidates) + skipped_recent_status
        more_pending = (orphan_message_rows - removed_message_rows) > 0 or (
            orphan_status_rows - removed_status_rows
        ) > 0

        return EmbeddingOrphanReconcileReport(
            index_db=str(index_path),
            embeddings_db=str(embeddings_path),
            dry_run=dry_run,
            now_ms=resolved_now_ms,
            quiet_window_ms=quiet_window_ms,
            scanned_message_meta_rows=scanned_message_meta_rows,
            scanned_status_rows=scanned_status_rows,
            orphan_message_rows=orphan_message_rows,
            orphan_status_rows=orphan_status_rows,
            skipped_recent_message_rows=skipped_recent_message,
            skipped_recent_status_rows=skipped_recent_status,
            removed_message_rows=removed_message_rows,
            removed_vector_rows=removed_vector_rows,
            removed_status_rows=removed_status_rows,
            sessions_recounted=sessions_recounted,
            more_pending=more_pending,
            samples=tuple(samples),
        )
    finally:
        conn.close()


def _is_recent(timestamp_ms: int | None, now_ms: int, quiet_window_ms: int) -> bool:
    if timestamp_ms is None:
        return False
    return (now_ms - int(timestamp_ms)) < quiet_window_ms


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0) if row is not None else 0


__all__ = [
    "DEFAULT_MAX_COUNT",
    "DEFAULT_QUIET_WINDOW_MS",
    "DEFAULT_SAMPLE_SIZE",
    "EmbeddingOrphanReconcileReport",
    "EmbeddingOrphanSample",
    "inspect_embedding_orphans",
    "reconcile_embedding_orphans",
]

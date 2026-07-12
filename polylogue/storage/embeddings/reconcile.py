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

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from polylogue.storage.sqlite.archive_tiers.index import INDEX_SCHEMA_VERSION

DEFAULT_QUIET_WINDOW_MS = 5 * 60 * 1000  # 5 minutes
DEFAULT_SAMPLE_SIZE = 30
DEFAULT_MAX_COUNT = 500
EmbeddingReconcileMutationAuthority = Literal["daemon-coordinator", "offline-exclusive"]


@dataclass(frozen=True, slots=True)
class _IndexIdentity:
    resolved_path: str
    device: int
    inode: int


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
    scanned_vector_rows: int
    scanned_status_rows: int
    orphan_message_rows: int
    orphan_message_meta_rows: int
    orphan_vector_rows: int
    orphan_status_rows: int
    skipped_recent_message_rows: int
    skipped_recent_status_rows: int
    candidate_message_rows: int
    candidate_message_meta_rows: int
    candidate_vector_rows: int
    candidate_status_rows: int
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
            "scanned_vector_rows": self.scanned_vector_rows,
            "scanned_status_rows": self.scanned_status_rows,
            "orphan_message_rows": self.orphan_message_rows,
            "orphan_message_meta_rows": self.orphan_message_meta_rows,
            "orphan_vector_rows": self.orphan_vector_rows,
            "orphan_status_rows": self.orphan_status_rows,
            "skipped_recent_message_rows": self.skipped_recent_message_rows,
            "skipped_recent_status_rows": self.skipped_recent_status_rows,
            "candidate_message_rows": self.candidate_message_rows,
            "candidate_message_meta_rows": self.candidate_message_meta_rows,
            "candidate_vector_rows": self.candidate_vector_rows,
            "candidate_status_rows": self.candidate_status_rows,
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
    mutation_authority: EmbeddingReconcileMutationAuthority | None = None,
) -> EmbeddingOrphanReconcileReport:
    """Reconcile one bounded batch of orphan embedding rows.

    Idempotent and resumable: safe to call repeatedly (a clean archive
    reports zero orphans and performs no writes), and a batch capped by
    ``max_count`` leaves ``more_pending=True`` so a caller (daemon
    convergence tick, CLI retry) can drain the rest across subsequent calls.

    Apply is deliberately unavailable to arbitrary callers. The daemon must
    enter through its process-wide write coordinator; the manual CLI must own
    the archive's exclusive offline lease. The attached index identity is
    checked before mutation and again before commit so a generation swap
    rolls the whole transaction back instead of deleting against stale truth.
    """

    index_path = Path(index_db_path)
    embeddings_path = (
        Path(embeddings_db_path) if embeddings_db_path is not None else index_path.with_name("embeddings.db")
    )
    resolved_now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    if not dry_run and mutation_authority is None:
        raise RuntimeError(
            "embedding orphan reconciliation apply requires daemon-coordinator or offline-exclusive authority"
        )

    if not embeddings_path.exists():
        return EmbeddingOrphanReconcileReport(
            index_db=str(index_path),
            embeddings_db=str(embeddings_path),
            dry_run=dry_run,
            now_ms=resolved_now_ms,
            quiet_window_ms=quiet_window_ms,
            scanned_message_meta_rows=0,
            scanned_vector_rows=0,
            scanned_status_rows=0,
            orphan_message_rows=0,
            orphan_message_meta_rows=0,
            orphan_vector_rows=0,
            orphan_status_rows=0,
            skipped_recent_message_rows=0,
            skipped_recent_status_rows=0,
            candidate_message_rows=0,
            candidate_message_meta_rows=0,
            candidate_vector_rows=0,
            candidate_status_rows=0,
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

        expected_index_identity = _index_identity(index_path)
        conn.execute("ATTACH DATABASE ? AS idx", (expected_index_identity.resolved_path,))
        if not dry_run:
            actual_index_schema_version = _scalar(conn, "PRAGMA idx.user_version")
            if actual_index_schema_version != INDEX_SCHEMA_VERSION:
                raise RuntimeError(
                    "embedding orphan reconciliation apply requires an authoritative index schema: "
                    f"active index is v{actual_index_schema_version}, packaged index is v{INDEX_SCHEMA_VERSION}"
                )
            _assert_active_index_generation(index_path)
        conn.execute("BEGIN" if dry_run else "BEGIN IMMEDIATE")

        scanned_message_meta_rows = _scalar(conn, "SELECT COUNT(*) FROM message_embeddings_meta")
        scanned_vector_rows = _scalar(conn, "SELECT COUNT(*) FROM message_embeddings")
        scanned_status_rows = _scalar(conn, "SELECT COUNT(*) FROM embedding_status")

        message_rows = _orphan_message_rows(conn)

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
        candidate_message_rows = len(limited_message)
        candidate_message_meta_rows = sum(int(row["has_meta"]) for row in limited_message)
        candidate_vector_rows = sum(int(row["has_vector"]) for row in limited_message)
        candidate_status_rows = len(limited_status)

        removed_message_rows = 0
        removed_vector_rows = 0
        removed_status_rows = 0
        sessions_recounted = 0
        action_label = "would_remove" if dry_run else "removed"

        if not dry_run:
            _assert_index_identity(index_path, expected_index_identity)
            _assert_active_index_generation(index_path)
            affected_sessions: set[str] = set()
            for row in limited_message:
                message_id = str(row["message_id"])
                removed_meta = _delete_if_still_orphan(conn, "message_embeddings_meta", message_id)
                removed_vector = _delete_if_still_orphan(conn, "message_embeddings", message_id)
                removed_message_rows += removed_meta
                removed_vector_rows += removed_vector
                if (removed_meta or removed_vector) and row["session_id"]:
                    affected_sessions.add(str(row["session_id"]))

            for session_id in sorted(affected_sessions):
                if not _index_session_exists(conn, session_id):
                    continue
                remaining = _scalar(conn, "SELECT COUNT(*) FROM message_embeddings WHERE session_id = ?", (session_id,))
                cursor = conn.execute(
                    """
                    UPDATE embedding_status
                    SET message_count_embedded = ?, needs_reindex = 1
                    WHERE session_id = ?
                    """,
                    (remaining, session_id),
                )
                sessions_recounted += max(0, cursor.rowcount)

            for row in limited_status:
                cursor = conn.execute(
                    """
                    DELETE FROM embedding_status
                    WHERE session_id = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM idx.sessions WHERE idx.sessions.session_id = embedding_status.session_id
                      )
                    """,
                    (row["session_id"],),
                )
                removed_status_rows += max(0, cursor.rowcount)

            _assert_index_identity(index_path, expected_index_identity)
            _assert_active_index_generation(index_path)
            conn.commit()

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
        orphan_message_meta_rows = sum(int(row["has_meta"]) for row in message_rows)
        orphan_vector_rows = sum(int(row["has_vector"]) for row in message_rows)
        orphan_status_rows = len(status_candidates) + skipped_recent_status
        if dry_run:
            more_pending = orphan_message_rows > 0 or orphan_status_rows > 0
            conn.rollback()
        else:
            more_pending = bool(_orphan_message_rows(conn)) or bool(
                conn.execute(
                    """
                    SELECT 1 FROM embedding_status
                    WHERE NOT EXISTS (
                        SELECT 1 FROM idx.sessions WHERE idx.sessions.session_id = embedding_status.session_id
                    )
                    LIMIT 1
                    """
                ).fetchone()
            )

        return EmbeddingOrphanReconcileReport(
            index_db=str(index_path),
            embeddings_db=str(embeddings_path),
            dry_run=dry_run,
            now_ms=resolved_now_ms,
            quiet_window_ms=quiet_window_ms,
            scanned_message_meta_rows=scanned_message_meta_rows,
            scanned_vector_rows=scanned_vector_rows,
            scanned_status_rows=scanned_status_rows,
            orphan_message_rows=orphan_message_rows,
            orphan_message_meta_rows=orphan_message_meta_rows,
            orphan_vector_rows=orphan_vector_rows,
            orphan_status_rows=orphan_status_rows,
            skipped_recent_message_rows=skipped_recent_message,
            skipped_recent_status_rows=skipped_recent_status,
            candidate_message_rows=candidate_message_rows,
            candidate_message_meta_rows=candidate_message_meta_rows,
            candidate_vector_rows=candidate_vector_rows,
            candidate_status_rows=candidate_status_rows,
            removed_message_rows=removed_message_rows,
            removed_vector_rows=removed_vector_rows,
            removed_status_rows=removed_status_rows,
            sessions_recounted=sessions_recounted,
            more_pending=more_pending,
            samples=tuple(samples),
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _orphan_message_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        WITH embedding_ids AS (
            SELECT message_id FROM message_embeddings_meta
            UNION
            SELECT message_id FROM message_embeddings
        )
        SELECT ids.message_id,
               COALESCE(
                   vec.session_id,
                   (
                       SELECT status.session_id
                       FROM embedding_status AS status
                       WHERE ids.message_id LIKE status.session_id || ':%'
                       ORDER BY length(status.session_id) DESC
                       LIMIT 1
                   )
               ) AS session_id,
               meta.content_hash,
               meta.embedded_at_ms,
               meta.message_id IS NOT NULL AS has_meta,
               vec.message_id IS NOT NULL AS has_vector
        FROM embedding_ids AS ids
        LEFT JOIN message_embeddings_meta AS meta ON meta.message_id = ids.message_id
        LEFT JOIN message_embeddings AS vec ON vec.message_id = ids.message_id
        WHERE NOT EXISTS (
            SELECT 1 FROM idx.messages AS indexed WHERE indexed.message_id = ids.message_id
        )
        ORDER BY ids.message_id
        """
    ).fetchall()


def _delete_if_still_orphan(conn: sqlite3.Connection, table: str, message_id: str) -> int:
    if table not in {"message_embeddings_meta", "message_embeddings"}:
        raise ValueError(f"unsupported embedding table: {table}")
    cursor = conn.execute(
        f"""
        DELETE FROM {table}
        WHERE message_id = ?
          AND NOT EXISTS (
              SELECT 1 FROM idx.messages WHERE idx.messages.message_id = {table}.message_id
          )
        """,
        (message_id,),
    )
    return max(0, cursor.rowcount)


def _index_session_exists(conn: sqlite3.Connection, session_id: str) -> bool:
    return conn.execute("SELECT 1 FROM idx.sessions WHERE session_id = ? LIMIT 1", (session_id,)).fetchone() is not None


def _index_identity(index_path: Path) -> _IndexIdentity:
    resolved = index_path.resolve(strict=True)
    stat = resolved.stat()
    return _IndexIdentity(str(resolved), stat.st_dev, stat.st_ino)


def _assert_index_identity(index_path: Path, expected: _IndexIdentity) -> None:
    actual = _index_identity(index_path)
    if actual != expected:
        raise RuntimeError(
            "active index generation changed during embedding orphan reconciliation; transaction rolled back"
        )


def _assert_active_index_generation(index_path: Path) -> None:
    """Require the active source-snapshotted generation when generations exist.

    Legacy archives have a direct ``index.db`` and no generation metadata, so
    the schema and inode guards remain their authority proof.  Once a rebuild
    creates generation metadata, however, a same-schema inactive candidate is
    not safe deletion truth: only the generation named by the active pointer
    and marked ``active`` after a source snapshot may drive reconciliation.
    """

    root = index_path.parent
    generations = root / ".index-generations"
    metadata_paths = tuple(generations.glob("*/generation.json")) if generations.is_dir() else ()
    if not metadata_paths:
        return

    pointer = root / ".index-active-pointer"
    if not pointer.is_file():
        raise RuntimeError("embedding orphan reconciliation requires an active index generation pointer")
    try:
        pointed_path = Path(pointer.read_text(encoding="utf-8").strip()).resolve(strict=True)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "embedding orphan reconciliation found an unreadable active index generation pointer"
        ) from exc
    if pointed_path != index_path.resolve(strict=True):
        raise RuntimeError("embedding orphan reconciliation refuses a non-active index generation")

    for metadata_path in metadata_paths:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            generation_path = Path(str(payload["index_path"])).resolve(strict=True)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if generation_path == pointed_path:
            if (
                payload.get("state") == "active"
                and isinstance(payload.get("source_snapshot"), str)
                and payload["source_snapshot"]
            ):
                return
            raise RuntimeError("embedding orphan reconciliation requires an active source-snapshotted index generation")
    raise RuntimeError("embedding orphan reconciliation active index is missing generation readiness evidence")


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
    "EmbeddingReconcileMutationAuthority",
    "EmbeddingOrphanReconcileReport",
    "EmbeddingOrphanSample",
    "inspect_embedding_orphans",
    "reconcile_embedding_orphans",
]

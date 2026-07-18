"""Daemon-safe source-to-index rebuild execution.

The operation owns the write-side rebuild protocol; CLI and HTTP are adapters.
Callers must hold the daemon writer coordinator for an online rebuild.  The
offline guard rejects every other live-daemon caller, preserving break-glass
operation after the daemon has stopped.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from polylogue.config import Config
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.paths import render_root
from polylogue.storage.archive_identity import ArchiveLocation


@dataclass(frozen=True, slots=True)
class RebuildIndexRequest:
    """One bounded source snapshot replay request."""

    archive_root: Path
    only_missing: bool = False
    raw_ids: tuple[str, ...] = ()
    max_blob_mb: float | None = None
    promote: bool = True
    operation_id: str | None = None
    raw_batch_size: int = 500
    pass_byte_budget_mb: float | None = None
    pass_deadline_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class RebuildIndexReceipt:
    """Typed evidence emitted after one source-to-index rebuild pass."""

    archive_root: str
    raw_session_count: int
    selected_raw_count: int
    skipped_by_blob_limit_count: int
    status: str
    materialized: bool
    materialization: dict[str, object]
    generation: dict[str, object]
    readiness: dict[str, object]
    replay: dict[str, object]
    transaction: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "archive_root": self.archive_root,
            "raw_session_count": self.raw_session_count,
            "selected_raw_count": self.selected_raw_count,
            "skipped_by_blob_limit_count": self.skipped_by_blob_limit_count,
            "status": self.status,
            "materialized": self.materialized,
            "materialization": self.materialization,
            "generation": self.generation,
            "readiness": self.readiness,
            "transaction": self.transaction,
            **self.replay,
        }


def validate_rebuild_index_request(request: RebuildIndexRequest) -> None:
    """Reject selection and transaction combinations that cannot be promoted safely."""
    if request.raw_ids and request.only_missing:
        raise ValueError("--raw-id cannot be combined with --only-missing")
    if (request.raw_ids or request.only_missing) and request.promote:
        raise ValueError("partial rebuild selections require --no-promote and can never replace the active index")
    if request.max_blob_mb is not None and request.max_blob_mb <= 0:
        raise ValueError("max blob size must be positive")
    if request.max_blob_mb is not None and not request.raw_ids and not request.only_missing:
        raise ValueError("--max-blob-mb requires --only-missing or --raw-id")
    if request.raw_batch_size <= 0:
        raise ValueError("raw batch size must be positive")
    if request.pass_byte_budget_mb is not None and request.pass_byte_budget_mb <= 0:
        raise ValueError("pass byte budget must be positive")
    if request.pass_deadline_seconds is not None and request.pass_deadline_seconds <= 0:
        raise ValueError("pass deadline must be positive")
    if request.operation_id is not None and (
        request.raw_ids or request.only_missing or request.max_blob_mb is not None
    ):
        raise ValueError("--operation-id only resumes an unfiltered full-source rebuild")
    if request.operation_id is not None and (
        request.pass_byte_budget_mb is not None or request.pass_deadline_seconds is not None
    ):
        raise ValueError("resumed rebuild budgets are durable; omit pass budget options with --operation-id")


def count_source_raw_sessions(root: Path) -> int:
    source_db = root / "source.db"
    if not source_db.exists():
        return 0
    with contextlib.closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0)) as conn:
        row = conn.execute("SELECT COUNT(*) FROM raw_sessions").fetchone()
    return int(row[0]) if row is not None else 0


def missing_index_raw_ids(root: Path) -> list[str]:
    source_db = root / "source.db"
    index_db = ArchiveLocation.resolve(root).active_index_path
    if not source_db.exists() or not index_db.exists():
        return []
    with contextlib.closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0)) as conn:
        conn.execute("ATTACH DATABASE ? AS idx", (str(index_db),))
        rows = conn.execute(
            """
            SELECT r.raw_id FROM raw_sessions r
            WHERE NOT EXISTS (SELECT 1 FROM idx.sessions s WHERE s.raw_id = r.raw_id)
            ORDER BY r.acquired_at_ms, r.raw_id
            """
        ).fetchall()
    return [str(row[0]) for row in rows]


def all_index_rebuild_raw_ids(root: Path) -> list[str]:
    source_db = root / "source.db"
    if not source_db.exists():
        return []
    with contextlib.closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0)) as conn:
        rows = conn.execute("SELECT raw_id FROM raw_sessions ORDER BY acquired_at_ms, raw_id").fetchall()
    return [str(row[0]) for row in rows]


def filter_raw_ids_by_max_blob_size(root: Path, raw_ids: list[str], max_blob_mb: float | None) -> list[str]:
    if max_blob_mb is None or not raw_ids:
        return raw_ids
    source_db = root / "source.db"
    placeholders = ",".join("?" for _ in raw_ids)
    with contextlib.closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True, timeout=10.0)) as conn:
        rows = conn.execute(
            f"SELECT raw_id FROM raw_sessions WHERE raw_id IN ({placeholders}) AND blob_size <= ? "
            "ORDER BY acquired_at_ms, raw_id",
            (*raw_ids, int(max_blob_mb * 1024 * 1024)),
        ).fetchall()
    return [str(row[0]) for row in rows]


def select_rebuild_raw_ids(request: RebuildIndexRequest) -> tuple[int, list[str], int]:
    """Select source rows deterministically before the replay starts."""
    root = request.archive_root
    raw_count = count_source_raw_sessions(root)
    raw_ids = (
        list(dict.fromkeys(request.raw_ids))
        if request.raw_ids
        else missing_index_raw_ids(root)
        if request.only_missing
        else all_index_rebuild_raw_ids(root)
    )
    unfiltered_count = len(raw_ids)
    selected = filter_raw_ids_by_max_blob_size(root, raw_ids, request.max_blob_mb)
    return raw_count, selected, unfiltered_count - len(selected)


async def rebuild_index_from_source(request: RebuildIndexRequest) -> RebuildIndexReceipt:
    """Replay one source snapshot into an owned generation and optionally promote it."""
    from polylogue.cli.commands.status import _archive_readiness_status
    from polylogue.maintenance.replay import rebuild_index_from_source as replay_source
    from polylogue.storage.index_generation import IndexGenerationStore, RebuildLease, source_revision_snapshot
    from polylogue.storage.repair import repair_session_insights

    validate_rebuild_index_request(request)
    root = request.archive_root
    active_config = Config(
        archive_root=root,
        render_root=render_root(),
        sources=[],
        db_path=ArchiveLocation.resolve(root).active_index_path,
    )
    if reason := offline_maintenance_block_reason(active_config, active=True, dry_run=False):
        raise RuntimeError(reason)

    generation_store = IndexGenerationStore(root)
    with RebuildLease(root):
        raw_count = count_source_raw_sessions(root)
        if raw_count == 0:
            return RebuildIndexReceipt(
                archive_root=str(root),
                raw_session_count=0,
                selected_raw_count=0,
                skipped_by_blob_limit_count=0,
                status="empty-source",
                materialized=False,
                materialization={},
                generation={},
                readiness={},
                replay={},
            )
        resumable_full_source = not request.raw_ids and not request.only_missing and request.max_blob_mb is None
        transaction = None
        page = None
        pass_started_at_ms = int(time.time() * 1000)
        if resumable_full_source:
            transaction = (
                generation_store.load_transaction(request.operation_id)
                if request.operation_id is not None
                else generation_store.create_transaction(
                    source_snapshot=source_revision_snapshot(root),
                    pass_byte_budget=(
                        int(request.pass_byte_budget_mb * 1024 * 1024)
                        if request.pass_byte_budget_mb is not None
                        else None
                    ),
                    pass_deadline_ms=(
                        int(request.pass_deadline_seconds * 1000) if request.pass_deadline_seconds is not None else None
                    ),
                )
            )
            if transaction.status in {"promoted", "stale"}:
                raise RuntimeError(
                    f"rebuild operation {transaction.operation_id} is {transaction.status}; start a new operation"
                )
            if source_revision_snapshot(root) != transaction.source_snapshot:
                generation_store.checkpoint_transaction(
                    transaction,
                    status="stale",
                    error="source evidence changed since this rebuild was planned",
                )
                raise RuntimeError(
                    f"rebuild operation {transaction.operation_id} is stale because source evidence changed"
                )
            generation = generation_store.load(transaction.generation_id)
            if generation.owner_id != transaction.generation_owner_id or generation.state != "inactive":
                raise RuntimeError(f"rebuild operation {transaction.operation_id} lost its inactive candidate")
            page = generation_store.next_raw_page(transaction, limit=request.raw_batch_size)
            selected_raw_ids = [raw_id for raw_id, _acquired_at_ms, _blob_size in page.rows]
            selected_raw_count = len(selected_raw_ids)
            skipped_by_blob_limit_count = 0
        else:
            raw_count, selected_raw_ids, skipped_by_blob_limit_count = select_rebuild_raw_ids(request)
            selected_raw_count = len(selected_raw_ids)
            generation = generation_store.create(source_snapshot=source_revision_snapshot(root))
        source_drifted = False
        try:
            generation_root = Path(generation.index_path).parent
            config = Config(
                archive_root=generation_root,
                render_root=render_root(),
                sources=[],
                db_path=Path(generation.index_path),
            )
            replay = await replay_source(
                config,
                raw_ids=selected_raw_ids,
                raw_batch_size=request.raw_batch_size,
                ingest_workers=None,
                materialize=True,
                progress_callback=None,
                owned_inactive_generation=(generation.generation_id, generation.owner_id),
            )
            if transaction is not None and selected_raw_ids:
                if source_revision_snapshot(root) != transaction.source_snapshot:
                    transaction = generation_store.checkpoint_transaction(
                        transaction,
                        status="stale",
                        error="source evidence changed during this bounded rebuild pass",
                    )
                    source_drifted = True
                    raise RuntimeError(
                        f"rebuild operation {transaction.operation_id} is stale because source evidence changed"
                    )
                assert page is not None
                last_raw_id, last_acquired_at_ms, _blob_size = page.rows[-1]
                elapsed_ms = int(time.time() * 1000) - pass_started_at_ms
                deadline_expired = (
                    transaction.pass_deadline_ms is not None and elapsed_ms >= transaction.pass_deadline_ms
                )
                status = "deferred" if page.deferred_reason == "byte-budget" or deadline_expired else "paused"
                transaction = generation_store.checkpoint_transaction(
                    transaction,
                    status=status,
                    last_acquired_at_ms=last_acquired_at_ms,
                    last_raw_id=last_raw_id,
                    processed_raw_count=transaction.processed_raw_count + len(selected_raw_ids),
                    processed_blob_bytes=transaction.processed_blob_bytes + sum(row[2] for row in page.rows),
                )
                if page.has_more or deadline_expired:
                    return RebuildIndexReceipt(
                        archive_root=str(root),
                        raw_session_count=raw_count,
                        selected_raw_count=selected_raw_count,
                        skipped_by_blob_limit_count=0,
                        status=status,
                        materialized=False,
                        materialization={},
                        generation=cast(dict[str, object], asdict(generation)),
                        readiness={},
                        replay=replay,
                        transaction=cast(dict[str, object], asdict(transaction)),
                    )
            insight_result = repair_session_insights(
                config,
                dry_run=False,
                archive_root_override=generation_root,
                owned_inactive_generation=(generation.generation_id, generation.owner_id),
            )
            if not insight_result.success:
                raise RuntimeError(f"session insight materialization failed: {insight_result.detail}")
            if source_revision_snapshot(root) != generation.source_snapshot:
                if transaction is not None:
                    transaction = generation_store.checkpoint_transaction(
                        transaction,
                        status="stale",
                        error="source evidence changed before terminal readiness",
                    )
                    source_drifted = True
                raise RuntimeError(f"source evidence changed while rebuilding {generation.generation_id}")
            readiness = _archive_readiness_status(generation_root)
            if not readiness.get("checked") or int(readiness.get("blocked_surface_count", 1)) != 0:
                blocked = [
                    name
                    for name, info in cast(dict[str, dict[str, object]], readiness.get("surfaces", {})).items()
                    if info.get("ready") is not True
                ]
                detail = (
                    f"reason: {readiness.get('reason')}"
                    if not readiness.get("checked")
                    else "blocked surfaces: " + ", ".join(blocked)
                )
                raise RuntimeError(f"inactive generation {generation.generation_id} is not exact-ready; {detail}")
            if transaction is not None:
                transaction = generation_store.checkpoint_transaction(transaction, status="ready")
            if request.promote:
                generation = generation_store.promote(generation)
                if transaction is not None:
                    transaction = generation_store.checkpoint_transaction(transaction, status="promoted")
        except Exception:
            if transaction is not None and not source_drifted:
                with contextlib.suppress(Exception):
                    generation_store.checkpoint_transaction(
                        transaction,
                        status="failed",
                        error="bounded rebuild pass failed; candidate retained for diagnosis or explicit recovery",
                    )
            else:
                with contextlib.suppress(Exception):
                    generation_store.discard_if_inactive(generation)
            raise
    return RebuildIndexReceipt(
        archive_root=str(root),
        raw_session_count=raw_count,
        selected_raw_count=selected_raw_count,
        skipped_by_blob_limit_count=skipped_by_blob_limit_count,
        status="replayed",
        materialized=True,
        materialization=cast(dict[str, object], insight_result.to_dict()),
        generation=cast(dict[str, object], asdict(generation)),
        readiness=cast(dict[str, object], readiness),
        replay=replay,
        transaction=cast(dict[str, object], asdict(transaction)) if transaction is not None else None,
    )


def rebuild_index_from_source_sync(request: RebuildIndexRequest) -> RebuildIndexReceipt:
    """Synchronous adapter for offline CLI callers."""
    return asyncio.run(rebuild_index_from_source(request))


__all__ = [
    "RebuildIndexReceipt",
    "RebuildIndexRequest",
    "all_index_rebuild_raw_ids",
    "count_source_raw_sessions",
    "filter_raw_ids_by_max_blob_size",
    "missing_index_raw_ids",
    "rebuild_index_from_source",
    "rebuild_index_from_source_sync",
    "select_rebuild_raw_ids",
    "validate_rebuild_index_request",
]

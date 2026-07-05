"""Consolidated archive repair: orphan detection, FTS repair, session insights, WAL."""

from __future__ import annotations

import asyncio
import re
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

from polylogue.archive.raw_materialization import parsed_non_session_artifact_reason
from polylogue.config import Config
from polylogue.core.enums import Origin, Provider
from polylogue.core.json import JSONDocument, json_document
from polylogue.core.sources import origin_from_provider, provider_from_origin
from polylogue.logging import get_logger
from polylogue.maintenance.models import DerivedModelStatus, MaintenanceCategory
from polylogue.maintenance.offline_guard import offline_maintenance_block_reason
from polylogue.maintenance.targets import (
    CLEANUP_TARGETS,
    SAFE_REPAIR_TARGETS,
    MaintenanceTargetSpec,
    build_maintenance_target_catalog,
)
from polylogue.protocols import ProgressCallback
from polylogue.sources.dispatch import is_stream_record_provider
from polylogue.storage.blob_repair import count_orphaned_blobs_sync, repair_orphaned_blobs_data
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.insights.session.repair_assessment import (
    assess_session_insight_repairs,
)
from polylogue.storage.insights.session.runtime import SESSION_INSIGHT_MATERIALIZATION_TYPES
from polylogue.storage.message_type_backfill import (
    BackfillResult,
    count_messages_by_type_sync,
    count_unclassified_message_type_sync,
)

logger = get_logger(__name__)
_MAINTENANCE_TARGET_CATALOG = build_maintenance_target_catalog()
_PROBE_ONLY_EXACT_MESSAGE_ROW_LIMIT = 100_000
RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES = 1024 * 1024 * 1024


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(max(value, 0))
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(amount)} B"
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{int(amount)} B"


@dataclass(frozen=True)
class RawMaterializationCandidates:
    raw_ids: list[str]
    missing_blobs: int
    already_parsed: int
    raw_blob_bytes: dict[str, int] = field(default_factory=dict)
    raw_origins: dict[str, str] = field(default_factory=dict)
    raw_source_paths: dict[str, str] = field(default_factory=dict)
    missing_blob_source_available: int = 0
    missing_blob_source_missing: int = 0

    @property
    def total_blob_bytes(self) -> int:
        return sum(self.raw_blob_bytes.get(raw_id, 0) for raw_id in self.raw_ids)

    @property
    def max_blob_bytes(self) -> int:
        return max((self.raw_blob_bytes.get(raw_id, 0) for raw_id in self.raw_ids), default=0)


def _raw_materialization_origin_from_provider(provider: str | None) -> str | None:
    if provider is None:
        return None
    return origin_from_provider(Provider.from_string(provider)).value


def _raw_materialization_source_available(source_path: str) -> bool:
    if not source_path:
        return False
    path = Path(source_path)
    if path.exists():
        return True
    if ":" in source_path:
        outer, _inner = source_path.split(":", 1)
        return Path(outer).exists()
    return False


def _raw_materialization_candidate_ids(
    config: Config,
    *,
    raw_artifact_id: str | None = None,
    provider: str | None = None,
    source_family: str | None = None,
    source_root: Path | None = None,
) -> RawMaterializationCandidates:
    """Return replayable raw ids plus missing-blob debt count.

    Raw evidence is the durable source of truth, but a raw row whose
    content-addressed blob is absent cannot be reparsed without outside
    evidence. Keep those rows as debt instead of mutating or deleting them.
    Broad repair queues raw rows that are not materialized in the attached
    index tier, including rows whose raw evidence was parsed before an index
    reset or interrupted replay. Parse failures, intentionally skipped rows,
    missing blobs, and source-path/native-id aliases remain excluded or counted
    as debt instead of being blindly retried.
    """
    source_db = config.archive_root / "source.db"
    index_db = config.archive_root / "index.db"
    if not source_db.exists() or not index_db.exists():
        return RawMaterializationCandidates([], 0, 0)
    blob_store = BlobStore(config.archive_root / "blob")
    raw_ids: list[str] = []
    raw_blob_bytes: dict[str, int] = {}
    raw_origins: dict[str, str] = {}
    raw_source_paths: dict[str, str] = {}
    missing_blobs = 0
    missing_blob_source_available = 0
    missing_blob_source_missing = 0
    already_parsed = 0
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("ATTACH DATABASE ? AS index_tier", (str(index_db),))
        params: list[object] = []
        raw_filter = ""
        if raw_artifact_id is not None:
            raw_filter = "AND r.raw_id = ?"
            params.append(raw_artifact_id)
        origin_filter = ""
        provider_origin = _raw_materialization_origin_from_provider(provider)
        if provider_origin is not None:
            origin_filter += " AND r.origin = ?"
            params.append(provider_origin)
        if source_family is not None:
            origin_filter += " AND r.origin = ?"
            params.append(source_family)
        source_root_filter = ""
        if source_root is not None:
            normalized_root = str(source_root).rstrip("/")
            source_root_filter = " AND (r.source_path = ? OR r.source_path LIKE ?)"
            params.extend((normalized_root, f"{normalized_root}/%"))
        rows = conn.execute(
            f"""
            SELECT r.raw_id, r.origin, r.native_id, r.source_path, r.blob_hash, r.blob_size, r.parsed_at_ms,
                   r.parse_error
            FROM raw_sessions AS r
            LEFT JOIN index_tier.sessions AS s_by_raw ON s_by_raw.raw_id = r.raw_id
            LEFT JOIN index_tier.sessions AS s_by_native
              ON r.native_id IS NOT NULL
             AND s_by_native.origin = r.origin
             AND s_by_native.native_id = r.native_id
            LEFT JOIN raw_sessions AS existing_native_raw
              ON existing_native_raw.raw_id = s_by_native.raw_id
            WHERE s_by_raw.raw_id IS NULL
              AND (
                s_by_native.native_id IS NULL
                OR existing_native_raw.raw_id IS NULL
              )
              AND (
                r.parse_error IS NULL
                OR (
                  r.parse_error LIKE 'decode:%No such file or directory:%'
                )
              )
              AND NOT (
                COALESCE(r.validation_status, '') = 'skipped'
                AND r.parsed_at_ms IS NOT NULL
                AND r.parse_error IS NULL
              )
              {raw_filter}
              {origin_filter}
              {source_root_filter}
            ORDER BY r.acquired_at_ms DESC, r.raw_id ASC
            """,
            params,
        ).fetchall()
        for row in rows:
            if row["parse_error"] and not _raw_materialization_retryable_missing_blob_error(row["parse_error"]):
                continue
            if _raw_materialized_by_source_path_native(conn, row):
                continue
            if _raw_materialization_parsed_non_session_artifact(config.archive_root, row):
                continue
            blob_hash = row["blob_hash"].hex() if isinstance(row["blob_hash"], bytes) else str(row["blob_hash"])
            if blob_store.exists(blob_hash):
                raw_id = str(row["raw_id"])
                raw_ids.append(raw_id)
                raw_origins[raw_id] = str(row["origin"] or "")
                raw_source_paths[raw_id] = str(row["source_path"] or "")
                blob_size = row["blob_size"]
                if isinstance(blob_size, int):
                    raw_blob_bytes[raw_id] = blob_size
                if row["parsed_at_ms"] is not None:
                    already_parsed += 1
            else:
                missing_blobs += 1
                if _raw_materialization_source_available(str(row["source_path"] or "")):
                    missing_blob_source_available += 1
                else:
                    missing_blob_source_missing += 1
    return RawMaterializationCandidates(
        raw_ids=raw_ids,
        missing_blobs=missing_blobs,
        already_parsed=already_parsed,
        raw_blob_bytes=raw_blob_bytes,
        raw_origins=raw_origins,
        raw_source_paths=raw_source_paths,
        missing_blob_source_available=missing_blob_source_available,
        missing_blob_source_missing=missing_blob_source_missing,
    )


def _raw_materialization_stream_safe(candidates: RawMaterializationCandidates, raw_id: str) -> bool:
    origin = Origin.from_string(candidates.raw_origins.get(raw_id))
    provider = provider_from_origin(origin)
    return is_stream_record_provider(candidates.raw_source_paths.get(raw_id), provider)


def _raw_materialization_retryable_missing_blob_error(parse_error: object) -> bool:
    if not isinstance(parse_error, str):
        return False
    return parse_error.startswith("decode:") and "No such file or directory" in parse_error


def _raw_materialization_total_bytes(candidates: RawMaterializationCandidates, raw_ids: list[str]) -> int:
    return sum(candidates.raw_blob_bytes.get(raw_id, 0) for raw_id in raw_ids)


def _raw_materialization_max_bytes(candidates: RawMaterializationCandidates, raw_ids: list[str]) -> int:
    return max((candidates.raw_blob_bytes.get(raw_id, 0) for raw_id in raw_ids), default=0)


def _raw_materialization_bucket_summary(
    candidates: RawMaterializationCandidates,
    *,
    values: dict[str, str],
    key_name: str,
    limit: int,
) -> list[dict[str, object]]:
    buckets: dict[str, dict[str, int | str]] = {}
    for raw_id in candidates.raw_ids:
        key = values.get(raw_id) or "unknown"
        size = candidates.raw_blob_bytes.get(raw_id, 0)
        bucket = buckets.setdefault(key, {key_name: key, "raw_count": 0, "total_blob_bytes": 0, "max_blob_bytes": 0})
        bucket["raw_count"] = int(bucket["raw_count"]) + 1
        bucket["total_blob_bytes"] = int(bucket["total_blob_bytes"]) + size
        bucket["max_blob_bytes"] = max(int(bucket["max_blob_bytes"]), size)
    return [
        dict(bucket)
        for bucket in sorted(
            buckets.values(),
            key=lambda item: (-int(item["total_blob_bytes"]), -int(item["raw_count"]), str(item[key_name])),
        )[:limit]
    ]


def _raw_materialization_missing_blob_detail(candidates: RawMaterializationCandidates, *, final: bool) -> str:
    verb = "remain blocked by" if final else "blocked by"
    detail = f"{candidates.missing_blobs:,} raw rows {verb} missing blobs"
    parts: list[str] = []
    if candidates.missing_blob_source_available:
        parts.append(f"{candidates.missing_blob_source_available:,} with source paths still present")
    if candidates.missing_blob_source_missing:
        parts.append(f"{candidates.missing_blob_source_missing:,} with source paths missing")
    if parts:
        detail += f" ({'; '.join(parts)})"
    return detail


def raw_materialization_replay_backlog(config: Config, *, limit: int = 10) -> dict[str, object]:
    """Return a read-only weighted backlog for raw source-to-index replay.

    The report uses the same candidate selector as ``repair_raw_materialization``
    so diagnostics and actual replay agree about which raw rows are actionable.
    It does not parse raw blobs or mutate the archive.
    """

    source_db = config.archive_root / "source.db"
    index_db = config.archive_root / "index.db"
    if not source_db.exists() or not index_db.exists():
        return {
            "available": False,
            "reason": "source_or_index_tier_missing",
            "candidate_count": 0,
            "top_raw_rows": [],
            "origin_summary": [],
            "source_path_summary": [],
        }
    candidates = _raw_materialization_candidate_ids(config)
    raw_ids_by_size = sorted(
        candidates.raw_ids,
        key=lambda raw_id: (-candidates.raw_blob_bytes.get(raw_id, 0), raw_id),
    )
    top_raw_rows = [
        {
            "raw_id": raw_id,
            "origin": candidates.raw_origins.get(raw_id) or "unknown",
            "source_path": candidates.raw_source_paths.get(raw_id) or "",
            "blob_size": candidates.raw_blob_bytes.get(raw_id, 0),
            "oversized": candidates.raw_blob_bytes.get(raw_id, 0) > RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES,
            "stream_safe": _raw_materialization_stream_safe(candidates, raw_id),
        }
        for raw_id in raw_ids_by_size[:limit]
    ]
    oversized_raw_ids = [
        raw_id
        for raw_id in candidates.raw_ids
        if candidates.raw_blob_bytes.get(raw_id, 0) > RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES
    ]
    oversized_stream_safe = [
        raw_id for raw_id in oversized_raw_ids if _raw_materialization_stream_safe(candidates, raw_id)
    ]
    return {
        "available": True,
        "candidate_count": len(candidates.raw_ids),
        "missing_blob_count": candidates.missing_blobs,
        "missing_blob_source_available_count": candidates.missing_blob_source_available,
        "missing_blob_source_missing_count": candidates.missing_blob_source_missing,
        "already_parsed_count": candidates.already_parsed,
        "total_blob_bytes": candidates.total_blob_bytes,
        "max_blob_bytes": candidates.max_blob_bytes,
        "execute_blob_limit_bytes": RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES,
        "oversized_count": len(oversized_raw_ids),
        "oversized_stream_safe_count": len(oversized_stream_safe),
        "top_raw_rows": top_raw_rows,
        "origin_summary": _raw_materialization_bucket_summary(
            candidates,
            values=candidates.raw_origins,
            key_name="origin",
            limit=limit,
        ),
        "source_path_summary": _raw_materialization_bucket_summary(
            candidates,
            values=candidates.raw_source_paths,
            key_name="source_path",
            limit=limit,
        ),
    }


def _raw_materialized_by_source_path_native(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    origin = str(row["origin"] or "")
    if not origin:
        return False
    for native_id in _source_path_native_id_candidates(str(row["source_path"] or "")):
        existing = conn.execute(
            """
            SELECT 1
            FROM index_tier.sessions AS s
            JOIN raw_sessions AS existing_raw ON existing_raw.raw_id = s.raw_id
            WHERE s.origin = ?
              AND s.native_id = ?
            LIMIT 1
            """,
            (origin, native_id),
        ).fetchone()
        if existing is not None:
            return True
    return False


def _raw_materialization_parsed_non_session_artifact(archive_root: Path, row: sqlite3.Row) -> bool:
    keys = set(row.keys())
    parse_error = row["parse_error"] if "parse_error" in keys else None
    parsed_at_ms = row["parsed_at_ms"] if "parsed_at_ms" in keys else None
    blob_hash = row["blob_hash"] if "blob_hash" in keys else None
    if parse_error or parsed_at_ms is None:
        return False
    return (
        parsed_non_session_artifact_reason(
            archive_root=archive_root,
            origin=str(row["origin"] or ""),
            source_path=str(row["source_path"] or ""),
            blob_hash=blob_hash,
        )
        is not None
    )


def _source_path_native_id_candidates(source_path: str) -> tuple[str, ...]:
    if not source_path:
        return ()
    name = Path(source_path).name
    candidates: list[str] = []
    current = name
    for _ in range(4):
        stem = Path(current).stem
        if stem == current:
            break
        current = stem
        if current and current not in candidates:
            candidates.append(current)
        unsplit = re.sub(r"_\d+$", "", current)
        if unsplit and unsplit != current and unsplit not in candidates:
            candidates.append(unsplit)
    return tuple(candidates)


def _open_archive_index_connection() -> sqlite3.Connection:
    from polylogue.paths import active_index_db_path

    conn = sqlite3.connect(active_index_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _resolve_convergence_debt(
    *,
    ops_db: Path,
    stage: str,
    target_type: str,
    target_id: str,
) -> None:
    """Best-effort resolution for ops-tier convergence debt.

    Maintenance targets are explicit convergence actuators. When one proves a
    target ready, stale daemon debt for the same target must stop appearing as
    actionable work.
    """
    if not ops_db.exists():
        return
    try:
        with sqlite3.connect(ops_db) as conn:
            table_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='convergence_debt'"
            ).fetchone()
            if not table_exists:
                return
            conn.execute(
                """
                DELETE FROM convergence_debt
                WHERE stage = ? AND target_type = ? AND target_id = ?
                """,
                (stage, target_type, target_id),
            )
            conn.commit()
    except sqlite3.Error as exc:
        logger.warning(
            "convergence_debt_resolve_failed",
            stage=stage,
            target_type=target_type,
            target_id=target_id,
            error=str(exc),
        )


def _resolve_session_insight_convergence_debt(
    *,
    ops_db: Path,
    session_ids: tuple[str, ...] | None,
) -> None:
    """Clear proven session-insight convergence debt after maintenance repair."""
    if not ops_db.exists():
        return
    try:
        with sqlite3.connect(ops_db) as conn:
            table_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='convergence_debt'"
            ).fetchone()
            if not table_exists:
                return
            if session_ids is None:
                conn.execute(
                    """
                    DELETE FROM convergence_debt
                    WHERE stage = 'insights'
                      AND target_type = 'session_id'
                    """
                )
            else:
                for session_id in session_ids:
                    conn.execute(
                        """
                        DELETE FROM convergence_debt
                        WHERE stage = 'insights'
                          AND target_type = 'session_id'
                          AND target_id = ?
                        """,
                        (session_id,),
                    )
            conn.commit()
    except sqlite3.Error as exc:
        logger.warning(
            "session_insight_convergence_debt_resolve_failed",
            session_ids=session_ids,
            error=str(exc),
        )


def _session_insight_materializer_version() -> int:
    from polylogue.storage.runtime import SESSION_INSIGHT_MATERIALIZER_VERSION

    return SESSION_INSIGHT_MATERIALIZER_VERSION


def _session_insight_requires_archive_wide_rebuild(status: object) -> bool:
    return any(
        int(getattr(status, attr, 0) or 0) > 0
        for attr in (
            "orphan_profile_row_count",
            "orphan_latency_profile_row_count",
            "orphan_work_event_inference_count",
            "orphan_phase_inference_count",
            "stale_day_summary_count",
        )
    )


def _session_insight_aggregate_debt_count(status: object) -> int:
    return sum(
        int(getattr(status, attr, 0) or 0)
        for attr in (
            "missing_thread_materialization_count",
            "stale_thread_count",
            "orphan_thread_count",
            "stale_tag_rollup_count",
            "stale_day_summary_count",
        )
    )


def _targeted_session_insight_rebuild_ids(
    conn: sqlite3.Connection | None,
    status: object,
) -> tuple[str, ...] | None:
    if conn is None or _session_insight_requires_archive_wide_rebuild(status):
        return None

    materialization_selects = "\nUNION\n".join(
        """
        SELECT s.session_id
        FROM sessions AS s
        WHERE NOT EXISTS (
            SELECT 1
            FROM insight_materialization AS m
            WHERE m.insight_type = ?
              AND m.session_id = s.session_id
              AND m.materializer_version = ?
              AND ABS(COALESCE(m.source_sort_key_ms, 0) - COALESCE(s.sort_key_ms, 0)) = 0
        )
        """
        for _insight_type in SESSION_INSIGHT_MATERIALIZATION_TYPES
        if _insight_type != "thread"
    )
    materializer_version = _session_insight_materializer_version()
    rows = conn.execute(
        f"""
        SELECT DISTINCT session_id
        FROM (
            SELECT s.session_id
            FROM sessions AS s
            WHERE NOT EXISTS (
                SELECT 1 FROM session_profiles AS p WHERE p.session_id = s.session_id
            )
            UNION
            SELECT s.session_id
            FROM sessions AS s
            JOIN session_profiles AS p ON p.session_id = s.session_id
            WHERE p.materializer_version != ?
               OR ABS(COALESCE(p.source_sort_key, 0.0) - COALESCE(CAST(s.sort_key_ms AS REAL)/1000.0, 0.0)) > 0.000001
            UNION
            SELECT p.session_id
            FROM session_profiles AS p
            JOIN sessions AS s ON s.session_id = p.session_id
            WHERE NOT EXISTS (
                SELECT 1 FROM session_latency_profiles AS lp WHERE lp.session_id = p.session_id
            )
            UNION
            SELECT lp.session_id
            FROM session_latency_profiles AS lp
            JOIN sessions AS s ON s.session_id = lp.session_id
            WHERE lp.materializer_version != ?
               OR ABS(COALESCE(lp.source_sort_key, 0.0) - COALESCE(CAST(s.sort_key_ms AS REAL)/1000.0, 0.0)) > 0.000001
            UNION
            SELECT p.session_id
            FROM session_profiles AS p
            WHERE p.work_event_count != (
                SELECT COUNT(*) FROM session_work_events AS e WHERE e.session_id = p.session_id
            )
            UNION
            SELECT p.session_id
            FROM session_profiles AS p
            WHERE p.phase_count != (
                SELECT COUNT(*) FROM session_phases AS ph WHERE ph.session_id = p.session_id
            )
            UNION
            {materialization_selects}
        )
        ORDER BY session_id
        """,
        (
            materializer_version,
            materializer_version,
            *(
                value
                for insight_type in SESSION_INSIGHT_MATERIALIZATION_TYPES
                if insight_type != "thread"
                for value in (insight_type, materializer_version)
            ),
        ),
    ).fetchall()
    return tuple(str(row["session_id"] if isinstance(row, sqlite3.Row) else row[0]) for row in rows)


def _archive_index_present(config: Config) -> bool:
    index_db = config.archive_root / "index.db"
    if not index_db.exists():
        return False
    try:
        with sqlite3.connect(f"file:{index_db}?mode=ro", uri=True) as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0] or 0)
    except sqlite3.Error:
        return False
    return version > 0


def offline_maintenance_blockers(
    config: Config,
    *,
    repair: bool,
    cleanup: bool,
    dry_run: bool,
    targets: tuple[str, ...] = (),
) -> list[RepairResult]:
    detail = offline_maintenance_block_reason(config, active=repair or cleanup, dry_run=dry_run)
    if detail is None:
        return []
    selected_targets = targets or tuple(SAFE_REPAIR_TARGETS if repair else ()) + tuple(
        CLEANUP_TARGETS if cleanup else ()
    )
    return [
        _repair_result(target_name, repaired_count=0, success=False, detail=detail) for target_name in selected_targets
    ]


@dataclass
class RepairResult:
    name: str
    category: MaintenanceCategory
    destructive: bool
    repaired_count: int
    success: bool
    detail: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "name": self.name,
                "category": self.category.value,
                "destructive": self.destructive,
                "repaired_count": self.repaired_count,
                "success": self.success,
                "detail": self.detail,
                "metrics": dict(self.metrics),
            }
        )


# ---------------------------------------------------------------------------
# Orphan count queries (formerly archive_debt_counts)
# ---------------------------------------------------------------------------


def count_orphaned_messages_sync(conn: sqlite3.Connection) -> int:
    """Count messages whose parent session row is missing.

    keys each message to ``sessions`` via
    ``messages.session_id REFERENCES sessions(session_id) ON DELETE
    CASCADE``. The cascade makes a message without its session
    structurally impossible — deleting a session deletes its messages in
    the same statement. This query therefore reports the honest native
    invariant: it joins ``messages`` to ``sessions`` and counts the rows
    with no matching session, which is always 0 on a consistent archive
    archive. It is retained as an integrity probe so a corrupted file
    (FK disabled during a hand edit) is still observable.
    """
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM messages m
            LEFT JOIN sessions s ON s.session_id = m.session_id
            WHERE s.session_id IS NULL
            """
        ).fetchone()[0]
    )


def has_orphaned_messages_sync(conn: sqlite3.Connection) -> bool:
    return bool(
        conn.execute(
            """
            SELECT 1
            FROM messages m
            LEFT JOIN sessions s ON s.session_id = m.session_id
            WHERE s.session_id IS NULL
            LIMIT 1
            """
        ).fetchone()
    )


def count_empty_sessions_sync(conn: sqlite3.Connection) -> int:
    """Count sessions that carry no messages.

    The native session/message tree replaces the legacy
    session/message tables: an "empty session" is a ``sessions``
    row with no ``messages`` row referencing it.
    """
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.session_id
            WHERE m.session_id IS NULL
            """
        ).fetchone()[0]
    )


def count_orphaned_attachments_sync(conn: sqlite3.Connection) -> int:
    """Count attachment refs without a parent and attachments without refs.

    Native ``attachment_refs`` keys to ``sessions``/``messages`` with
    ``ON DELETE CASCADE`` / ``SET NULL``; ``attachments`` carry a
    materialized ``ref_count``. A ref without a live parent or an
    attachment with no surviving ref is the archive orphan signature.
    """
    orphaned_refs = int(
        conn.execute(
            """
            SELECT COUNT(*) FROM attachment_refs ar
            WHERE (ar.message_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.message_id = ar.message_id))
               OR NOT EXISTS (SELECT 1 FROM sessions s WHERE s.session_id = ar.session_id)
            """
        ).fetchone()[0]
    )
    unreferenced_attachments = int(
        conn.execute(
            """
            SELECT COUNT(*) FROM attachments a
            WHERE NOT EXISTS (SELECT 1 FROM attachment_refs ar WHERE ar.attachment_id = a.attachment_id)
            """
        ).fetchone()[0]
    )
    return orphaned_refs + unreferenced_attachments


def _table_has_more_than(conn: sqlite3.Connection, table_name: str, row_limit: int) -> bool:
    row = conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1 OFFSET ?", (max(0, row_limit),)).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Derived repair count helpers (formerly archive_debt_repairs)
# ---------------------------------------------------------------------------


def session_insight_repair_count(derived_statuses: dict[str, DerivedModelStatus]) -> int:
    keys = [
        "session_profile_rows",
        "session_work_events",
        "session_work_events_fts",
        "session_phases",
        "threads",
        "threads_fts",
        "session_tag_rollups",
    ]
    maybe_statuses = [derived_statuses.get(k) for k in keys]
    if not all(status is not None for status in maybe_statuses):
        return 0
    statuses = [status for status in maybe_statuses if status is not None]
    total = 0
    for s in statuses:
        total += max(0, int(s.pending_documents or 0))
        total += max(0, int(s.pending_rows or 0))
        total += max(0, int(s.stale_rows or 0))
        total += max(0, int(s.orphan_rows or 0))
    return total


# ---------------------------------------------------------------------------
# Archive debt collection (formerly archive_debt.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchiveDebtStatus:
    """Simple debt/orphan status for a single maintenance target."""

    name: str
    category: MaintenanceCategory
    destructive: bool
    issue_count: int
    detail: str
    maintenance_target: str
    skipped: bool = False

    @property
    def healthy(self) -> bool:
        return self.issue_count == 0 and not self.skipped

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "name": self.name,
                "category": self.category.value,
                "destructive": self.destructive,
                "issue_count": self.issue_count,
                "detail": self.detail,
                "maintenance_target": self.maintenance_target,
                "healthy": self.healthy,
                "skipped": self.skipped,
            }
        )


def _maintenance_target_spec(name: str) -> MaintenanceTargetSpec:
    spec = _MAINTENANCE_TARGET_CATALOG.resolve_name(name)
    if spec is None:
        raise KeyError(f"Unknown maintenance target: {name}")
    return spec


def _repair_result(
    target_name: str,
    *,
    repaired_count: int,
    success: bool,
    detail: str,
    metrics: dict[str, float] | None = None,
) -> RepairResult:
    spec = _maintenance_target_spec(target_name)
    return RepairResult(
        name=spec.name,
        category=spec.category,
        destructive=spec.destructive,
        repaired_count=repaired_count,
        success=success,
        detail=detail,
        metrics=dict(metrics or {}),
    )


def _internal_derived_repair_result(
    name: str,
    *,
    repaired_count: int,
    success: bool,
    detail: str,
    metrics: dict[str, float] | None = None,
) -> RepairResult:
    return RepairResult(
        name=name,
        category=MaintenanceCategory.DERIVED_REPAIR,
        destructive=False,
        repaired_count=repaired_count,
        success=success,
        detail=detail,
        metrics=dict(metrics or {}),
    )


def _archive_debt_status(
    target_name: str,
    *,
    issue_count: int,
    detail: str,
    skipped: bool = False,
) -> ArchiveDebtStatus:
    spec = _maintenance_target_spec(target_name)
    return ArchiveDebtStatus(
        name=spec.name,
        category=spec.category,
        destructive=spec.destructive,
        issue_count=issue_count,
        detail=detail,
        maintenance_target=spec.name,
        skipped=skipped,
    )


def collect_archive_debt_statuses_sync(
    conn: sqlite3.Connection,
    *,
    db_path: Path | str | None = None,
    derived_statuses: dict[str, DerivedModelStatus] | None = None,
    include_expensive: bool = True,
    probe_only: bool = False,
    target_names: tuple[str, ...] = (),
) -> dict[str, ArchiveDebtStatus]:
    from polylogue.storage.derived.derived_status import collect_derived_model_statuses_sync

    selected = set(target_names) if target_names else set(_MAINTENANCE_TARGET_CATALOG.names())
    needs_session_insights = "session_insights" in selected
    statuses = (
        derived_statuses or collect_derived_model_statuses_sync(conn, verify_full=include_expensive)
        if needs_session_insights
        else {}
    )

    skip_large_message_scans = (
        probe_only
        and not include_expensive
        and _table_has_more_than(conn, "messages", _PROBE_ONLY_EXACT_MESSAGE_ROW_LIMIT)
    )
    debt_statuses: dict[str, ArchiveDebtStatus] = {}

    if "orphaned_messages" in selected:
        orphaned_messages = 0 if skip_large_message_scans else count_orphaned_messages_sync(conn)
        debt_statuses["orphaned_messages"] = _archive_debt_status(
            "orphaned_messages",
            issue_count=orphaned_messages,
            detail=(
                "Skipped exact orphaned-message scan in probe mode; use --deep for exact count"
                if skip_large_message_scans
                else "No orphaned messages"
                if orphaned_messages == 0
                else (
                    "Orphaned messages present; use --deep for exact count"
                    if probe_only and not include_expensive
                    else f"{orphaned_messages:,} orphaned messages"
                )
            ),
            skipped=skip_large_message_scans,
        )
    if "empty_sessions" in selected:
        empty_sessions = 0 if skip_large_message_scans else count_empty_sessions_sync(conn)
        debt_statuses["empty_sessions"] = _archive_debt_status(
            "empty_sessions",
            issue_count=empty_sessions,
            detail=(
                "Skipped exact empty-session scan in probe mode; use --deep for exact count"
                if skip_large_message_scans
                else "No empty sessions"
                if empty_sessions == 0
                else f"{empty_sessions:,} empty sessions"
            ),
            skipped=skip_large_message_scans,
        )
    if "orphaned_attachments" in selected:
        orphaned_attachments = count_orphaned_attachments_sync(conn)
        debt_statuses["orphaned_attachments"] = _archive_debt_status(
            "orphaned_attachments",
            issue_count=orphaned_attachments,
            detail="No orphaned attachments"
            if orphaned_attachments == 0
            else f"{orphaned_attachments:,} orphaned attachment rows",
        )
    if "session_insights" in selected:
        session_insights = session_insight_repair_count(statuses)
        debt_statuses["session_insights"] = _archive_debt_status(
            "session_insights",
            issue_count=session_insights,
            detail="Session insight read models ready"
            if session_insights == 0
            else f"{session_insights:,} pending/stale/orphaned session-insight rows",
        )
    if "message_type_backfill" in selected:
        unclassified = 0 if skip_large_message_scans else count_unclassified_message_type_sync(conn)
        debt_statuses["message_type_backfill"] = _archive_debt_status(
            "message_type_backfill",
            issue_count=unclassified,
            detail=(
                "Skipped exact message-type backfill scan in probe mode; use --deep for exact count"
                if skip_large_message_scans
                else "No messages need context/protocol classification"
                if unclassified == 0
                else f"{unclassified:,} messages would be classified as context or protocol"
            ),
            skipped=skip_large_message_scans,
        )
    if include_expensive and "orphaned_blobs" in selected:
        orphaned_blobs = count_orphaned_blobs_sync(conn, db_path=db_path)
        debt_statuses["orphaned_blobs"] = _archive_debt_status(
            "orphaned_blobs",
            issue_count=orphaned_blobs,
            detail="No orphaned blobs" if orphaned_blobs == 0 else f"{orphaned_blobs:,} orphaned blob files on disk",
        )
    if include_expensive and "superseded_raw_snapshots" in selected:
        superseded_raw_snapshots = count_superseded_raw_snapshots_sync(conn)
        debt_statuses["superseded_raw_snapshots"] = _archive_debt_status(
            "superseded_raw_snapshots",
            issue_count=superseded_raw_snapshots,
            detail=(
                "No superseded live raw snapshots"
                if superseded_raw_snapshots == 0
                else f"{superseded_raw_snapshots:,} superseded live raw snapshots"
            ),
        )
    return debt_statuses


def preview_counts_from_archive_debt(
    statuses: dict[str, ArchiveDebtStatus],
) -> dict[str, int]:
    preview_targets = set(_MAINTENANCE_TARGET_CATALOG.preview_target_names())
    return {
        status.maintenance_target: status.issue_count
        for status in statuses.values()
        if status.issue_count > 0 or status.maintenance_target in preview_targets
    }


# ---------------------------------------------------------------------------
# Generic SQL repair helper
# ---------------------------------------------------------------------------


def _run_sql_repair(
    target_name: str,
    *,
    count_sql: str,
    action_sql: str | None,
    dry_run: bool,
    conn: sqlite3.Connection,
) -> RepairResult:
    try:
        count = conn.execute(count_sql).fetchone()[0]
        if dry_run:
            return _repair_result(
                target_name,
                repaired_count=count,
                success=True,
                detail=f"Would: {count} rows affected" if count else "Would: No issues found",
            )
        if action_sql:
            result = conn.execute(action_sql)
            conn.commit()
            return _repair_result(
                target_name,
                repaired_count=result.rowcount,
                success=True,
                detail=f"Repaired {result.rowcount} rows" if result.rowcount else "No repairs needed",
            )
        return _repair_result(
            target_name,
            repaired_count=0,
            success=True,
            detail="No action SQL provided",
        )
    except Exception as exc:
        return _repair_result(
            target_name,
            repaired_count=0,
            success=False,
            detail=f"Repair failed: {exc}",
        )


# ---------------------------------------------------------------------------
# Cleanup repairs (orphans, empty sessions, attachments)
# ---------------------------------------------------------------------------


def repair_orphaned_messages(config: Config, dry_run: bool = False) -> RepairResult:
    """Delete messages whose parent session row is missing.

    On the archive ``messages.session_id`` cascades from
    ``sessions``, so a session-less message can only exist after a
    file-level corruption (FK disabled during a hand edit). The repair
    counts such rows via :func:`count_orphaned_messages_sync` and, when
    any are found, deletes the orphan ``messages`` rows directly; the
    ``blocks`` rows beneath them cascade away through
    ``blocks.message_id REFERENCES messages ON DELETE CASCADE``.
    """
    with _open_archive_index_connection() as conn:
        count = count_orphaned_messages_sync(conn)
        if count == 0:
            return _repair_result(
                "orphaned_messages",
                repaired_count=0,
                success=True,
                detail="No orphaned messages found",
            )
        try:
            if dry_run:
                return _repair_result(
                    "orphaned_messages",
                    repaired_count=count,
                    success=True,
                    detail=f"Would: Delete {count} orphaned messages",
                )
            result = conn.execute(
                """
                DELETE FROM messages
                WHERE NOT EXISTS (
                    SELECT 1 FROM sessions s WHERE s.session_id = messages.session_id
                )
                """
            )
            conn.commit()
            return _repair_result(
                "orphaned_messages",
                repaired_count=result.rowcount,
                success=True,
                detail=f"Deleted {result.rowcount} orphaned messages",
            )
        except Exception as exc:
            return _repair_result(
                "orphaned_messages",
                repaired_count=0,
                success=False,
                detail=f"Failed to delete orphaned messages: {exc}",
            )


def preview_orphaned_messages(*, count: int) -> RepairResult:
    return _repair_result(
        "orphaned_messages",
        repaired_count=count,
        success=True,
        detail=f"Would: Delete {count} orphaned messages" if count else "Would: No orphaned messages found",
    )


def repair_empty_sessions(config: Config, dry_run: bool = False) -> RepairResult:
    with _open_archive_index_connection() as conn:
        return _run_sql_repair(
            "empty_sessions",
            count_sql="SELECT COUNT(*) FROM sessions c WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.session_id = c.session_id)",
            action_sql="DELETE FROM sessions WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.session_id = sessions.session_id)",
            dry_run=dry_run,
            conn=conn,
        )


def preview_empty_sessions(*, count: int) -> RepairResult:
    return _repair_result(
        "empty_sessions",
        repaired_count=count,
        success=True,
        detail=f"Would: {count} rows affected" if count else "Would: No issues found",
    )


# ---------------------------------------------------------------------------
# Blob cleanup
# ---------------------------------------------------------------------------


def repair_orphaned_blobs(config: Config, dry_run: bool = False) -> RepairResult:
    outcome = repair_orphaned_blobs_data(config, dry_run=dry_run)
    return _repair_result(
        "orphaned_blobs",
        repaired_count=outcome.repaired_count,
        success=outcome.success,
        detail=outcome.detail,
    )


def count_superseded_raw_snapshots_sync(conn: sqlite3.Connection) -> int:
    from polylogue.storage.raw_retention import superseded_raw_snapshot_candidates

    return len(superseded_raw_snapshot_candidates(conn, limit=10_000))


def _index_referenced_raw_ids(index_db_path: Path) -> set[str]:
    if not index_db_path.exists():
        return set()
    try:
        with sqlite3.connect(f"file:{index_db_path}?mode=ro", uri=True) as conn:
            rows = conn.execute("SELECT DISTINCT raw_id FROM sessions WHERE raw_id IS NOT NULL").fetchall()
    except sqlite3.Error:
        return set()
    return {str(row[0]) for row in rows if row[0] is not None}


def repair_superseded_raw_snapshots(config: Config, dry_run: bool = False) -> RepairResult:
    import sqlite3

    from polylogue.storage.raw_retention import cleanup_superseded_raw_snapshots
    from polylogue.storage.sqlite.connection import open_connection

    repair_db_path = config.archive_root / "source.db"
    if repair_db_path.exists():
        index_db_path = config.archive_root / "index.db"
        if not index_db_path.exists():
            index_db_path = config.db_path
        protected_raw_ids = _index_referenced_raw_ids(index_db_path)
        with sqlite3.connect(repair_db_path) as conn:
            conn.row_factory = sqlite3.Row
            result = cleanup_superseded_raw_snapshots(
                conn,
                dry_run=dry_run,
                limit=10_000,
                protected_raw_ids=protected_raw_ids,
            )
    else:
        with open_connection(config.db_path) as conn:
            result = cleanup_superseded_raw_snapshots(conn, dry_run=dry_run, limit=10_000)
    if dry_run:
        skipped_detail = (
            f"; skipped {result.skipped_referenced_count:,} index-referenced raw rows"
            if result.skipped_referenced_count
            else ""
        )
        return _repair_result(
            "superseded_raw_snapshots",
            repaired_count=result.candidate_count,
            success=True,
            detail=(
                f"Would: delete {result.candidate_count:,} superseded raw snapshots "
                f"({result.deleted_raw_bytes:,} referenced bytes)"
                f"{skipped_detail}"
            ),
        )
    skipped_detail = (
        f"; skipped {result.skipped_referenced_count:,} index-referenced raw rows"
        if result.skipped_referenced_count
        else ""
    )
    return _repair_result(
        "superseded_raw_snapshots",
        repaired_count=result.deleted_raw_count,
        success=not result.errors,
        detail=(
            f"Deleted {result.deleted_raw_count:,} raw rows and {result.deleted_blob_count:,} blob files "
            f"({result.deleted_blob_bytes:,} bytes)"
            f"{skipped_detail}" + (f"; errors: {'; '.join(result.errors[:3])}" if result.errors else "")
        ),
    )


def preview_orphaned_blobs(*, count: int) -> RepairResult:
    return _repair_result(
        "orphaned_blobs",
        repaired_count=count,
        success=True,
        detail=f"Would: delete {count} orphaned blobs" if count else "Would: No orphaned blobs found",
    )


def preview_superseded_raw_snapshots(*, count: int) -> RepairResult:
    return _repair_result(
        "superseded_raw_snapshots",
        repaired_count=count,
        success=True,
        detail=(
            f"Would: delete {count} superseded live raw snapshots"
            if count
            else "Would: No superseded live raw snapshots found"
        ),
    )


def repair_orphaned_attachments(config: Config, dry_run: bool = False) -> RepairResult:
    try:
        with _open_archive_index_connection() as conn:
            if dry_run:
                return preview_orphaned_attachments(count=count_orphaned_attachments_sync(conn))

            ref_result = conn.execute(
                "DELETE FROM attachment_refs WHERE message_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM messages m WHERE m.message_id = attachment_refs.message_id)"
            )
            refs_deleted = ref_result.rowcount

            conv_ref_result = conn.execute(
                "DELETE FROM attachment_refs WHERE NOT EXISTS (SELECT 1 FROM sessions c WHERE c.session_id = attachment_refs.session_id)"
            )
            conv_refs_deleted = conv_ref_result.rowcount

            att_result = conn.execute(
                "DELETE FROM attachments WHERE NOT EXISTS (SELECT 1 FROM attachment_refs ar WHERE ar.attachment_id = attachments.attachment_id)"
            )
            atts_deleted = att_result.rowcount
            conn.commit()

            total = refs_deleted + conv_refs_deleted + atts_deleted
            return _repair_result(
                "orphaned_attachments",
                repaired_count=total,
                success=True,
                detail=f"Cleaned {refs_deleted} orphaned refs, {conv_refs_deleted} conv refs, {atts_deleted} attachments",
            )
    except Exception as exc:
        return _repair_result(
            "orphaned_attachments",
            repaired_count=0,
            success=False,
            detail=f"Failed to clean orphaned attachments: {exc}",
        )


def preview_orphaned_attachments(*, count: int) -> RepairResult:
    return _repair_result(
        "orphaned_attachments",
        repaired_count=count,
        success=True,
        detail=f"Would: Clean {count} orphaned attachment rows" if count else "Would: No orphaned attachments found",
    )


# ---------------------------------------------------------------------------
# Derived repairs (session insights, actions, FTS, WAL)
# ---------------------------------------------------------------------------


def repair_session_insights(
    config: Config,
    dry_run: bool = False,
    *,
    progress_callback: ProgressCallback | None = None,
    progress_total: int | None = None,
    session_ids: tuple[str, ...] | None = None,
) -> RepairResult:
    """Repair / rebuild session insights.

    When ``session_ids`` is given, the rebuild is narrowed to that
    set instead of touching the full archive — used by the maintenance
    planner to honor :class:`MaintenanceScopeFilter.session_ids`.
    """
    from polylogue.api.archive import _rebuild_archive_session_insights
    from polylogue.paths import active_index_db_path
    from polylogue.storage.insights.session.rebuild import refresh_session_insight_aggregates_sync
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    try:
        archive_root = active_index_db_path().parent
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            status = archive.session_insight_status()
            assessment = assess_session_insight_repairs(status)
            aggregate_debt = _session_insight_aggregate_debt_count(status)
            targeted_session_ids = (
                None
                if session_ids is not None or assessment.row_debt == 0
                else _targeted_session_insight_rebuild_ids(getattr(archive, "_conn", None), status)
            )

            if dry_run:
                if session_ids is not None:
                    pending = min(assessment.row_debt, len(session_ids))
                    detail = (
                        "Would: session insights already ready"
                        if pending == 0
                        else f"Would: rebuild session insights for {pending:,} scoped session(s)"
                    )
                elif targeted_session_ids is not None:
                    pending = len(targeted_session_ids) + aggregate_debt
                    detail = (
                        "Would: session insights already ready"
                        if pending == 0
                        else (
                            "Would: rebuild session insights for "
                            f"{len(targeted_session_ids):,} candidate session(s)"
                            f" and refresh {aggregate_debt:,} aggregate/thread-materialization debt row(s)"
                            f" to repair {assessment.row_debt:,} total debt row(s)"
                        )
                    )
                elif assessment.row_debt == 0:
                    pending = 0
                    detail = "Would: session insights already ready"
                else:
                    pending = status.total_sessions
                    detail = (
                        "Would: rebuild archive-wide session insights "
                        f"for {pending:,} session(s) to repair {assessment.row_debt:,} debt row(s)"
                    )
                return _repair_result(
                    "session_insights",
                    repaired_count=pending,
                    success=True,
                    detail=detail,
                )

            if session_ids is None and assessment.row_debt == 0:
                return _repair_result(
                    "session_insights",
                    repaired_count=0,
                    success=True,
                    detail="Session insights already ready",
                )

            rebuild_session_ids = session_ids if session_ids is not None else targeted_session_ids
            rebuilt = _rebuild_archive_session_insights(
                archive,
                session_ids=rebuild_session_ids,
                progress_callback=progress_callback,
            )
            rebuilt_count = rebuilt.total()
            refreshed = archive.session_insight_status()
            if session_ids is None and _session_insight_aggregate_debt_count(refreshed) > 0:
                aggregate_counts = refresh_session_insight_aggregates_sync(
                    archive._conn,
                    progress_callback=progress_callback,
                )
                rebuilt_count += aggregate_counts.total()
                refreshed = archive.session_insight_status()
            # A narrowed rebuild only attests its own slice; do not
            # demand global readiness for a scope-filtered call.
            success = True if session_ids is not None else assess_session_insight_repairs(refreshed).row_debt == 0
            if success:
                _resolve_session_insight_convergence_debt(
                    ops_db=config.archive_root / "ops.db",
                    session_ids=session_ids,
                )
            return _repair_result(
                "session_insights",
                repaired_count=rebuilt_count,
                success=success,
                detail="Session insights ready" if success else "Session insights still incomplete",
            )
    except Exception as exc:
        return _repair_result(
            "session_insights",
            repaired_count=0,
            success=False,
            detail=f"Failed to repair session insights: {exc}",
        )


def preview_session_insights(*, count: int) -> RepairResult:
    return _repair_result(
        "session_insights",
        repaired_count=count,
        success=True,
        detail="Would: session insights already ready"
        if count == 0
        else f"Would: rebuild session-insight rows/fts for {count:,} pending items",
    )


def repair_raw_materialization(
    config: Config,
    dry_run: bool = False,
    *,
    raw_artifact_id: str | None = None,
    provider: str | None = None,
    source_family: str | None = None,
    source_root: Path | None = None,
    raw_artifact_limit: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> RepairResult:
    """Materialize replayable raw evidence into the index tier."""
    candidates = _raw_materialization_candidate_ids(
        config,
        raw_artifact_id=raw_artifact_id,
        provider=provider,
        source_family=source_family,
        source_root=source_root,
    )
    candidate_raw_ids = candidates.raw_ids
    raw_ids = list(candidate_raw_ids)
    if raw_artifact_limit is not None:
        raw_ids = sorted(raw_ids, key=lambda raw_id: (candidates.raw_blob_bytes.get(raw_id, 0), raw_id))
        raw_ids = raw_ids[:raw_artifact_limit]
    missing_blobs = candidates.missing_blobs
    selected_total_bytes = _raw_materialization_total_bytes(candidates, raw_ids)
    selected_max_bytes = _raw_materialization_max_bytes(candidates, raw_ids)
    metrics = {
        "raw_materialization_candidate_count": float(len(candidate_raw_ids)),
        "raw_materialization_selected_count": float(len(raw_ids)),
        "raw_materialization_missing_blob_count": float(missing_blobs),
        "raw_materialization_missing_blob_source_available_count": float(candidates.missing_blob_source_available),
        "raw_materialization_missing_blob_source_missing_count": float(candidates.missing_blob_source_missing),
        "raw_materialization_already_parsed_count": float(candidates.already_parsed),
        "raw_materialization_total_blob_bytes": float(candidates.total_blob_bytes),
        "raw_materialization_max_blob_bytes": float(candidates.max_blob_bytes),
        "raw_materialization_selected_total_blob_bytes": float(selected_total_bytes),
        "raw_materialization_selected_max_blob_bytes": float(selected_max_bytes),
    }
    if raw_artifact_limit is not None:
        metrics["raw_materialization_limit"] = float(raw_artifact_limit)
    oversized_candidate_raw_ids = [
        raw_id
        for raw_id in raw_ids
        if candidates.raw_blob_bytes.get(raw_id, 0) > RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES
    ]
    oversized_stream_safe_raw_ids = [
        raw_id for raw_id in oversized_candidate_raw_ids if _raw_materialization_stream_safe(candidates, raw_id)
    ]
    oversized_stream_safe_raw_id_set = set(oversized_stream_safe_raw_ids)
    oversized_raw_ids = [
        raw_id for raw_id in oversized_candidate_raw_ids if raw_id not in oversized_stream_safe_raw_id_set
    ]
    oversized_raw_id_set = set(oversized_raw_ids)
    executable_raw_ids = [raw_id for raw_id in raw_ids if raw_id not in oversized_raw_id_set]
    if oversized_raw_ids:
        metrics["raw_materialization_oversized_count"] = float(len(oversized_raw_ids))
        metrics["raw_materialization_execute_blob_limit_bytes"] = float(RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES)
    if oversized_stream_safe_raw_ids:
        metrics["raw_materialization_stream_oversized_count"] = float(len(oversized_stream_safe_raw_ids))
    if dry_run:
        if not raw_ids:
            detail = "Raw materialization ready"
            if missing_blobs:
                detail += f"; {_raw_materialization_missing_blob_detail(candidates, final=False)}"
            return _internal_derived_repair_result(
                "raw_materialization",
                repaired_count=0,
                success=missing_blobs == 0,
                detail=detail,
                metrics=metrics,
            )
        byte_detail = ""
        if raw_ids:
            byte_detail = (
                f"; selected raw payload bytes total={_format_bytes(selected_total_bytes)}, "
                f"largest={_format_bytes(selected_max_bytes)}"
            )
        oversized_detail = ""
        if oversized_raw_ids:
            oversized_detail = (
                f"; {len(oversized_raw_ids):,} raw rows exceed actual replay limit "
                f"{_format_bytes(RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES)}"
            )
        if oversized_stream_safe_raw_ids:
            oversized_detail += (
                f"; {len(oversized_stream_safe_raw_ids):,} oversized stream-record raw rows can use streaming replay"
            )
        if candidates.already_parsed:
            detail = (
                f"Would: replay {len(raw_ids):,} of {len(candidate_raw_ids):,} raw rows into index.db "
                f"({candidates.already_parsed:,} already parsed but not materialized)"
            )
        else:
            detail = (
                f"Would: replay {len(raw_ids):,} of {len(candidate_raw_ids):,} "
                "acquired-but-unparsed raw rows into index.db"
            )
        detail += byte_detail
        detail += oversized_detail
        if missing_blobs:
            detail += f"; {_raw_materialization_missing_blob_detail(candidates, final=False)}"
        return _internal_derived_repair_result(
            "raw_materialization",
            repaired_count=len(raw_ids),
            success=True,
            detail=detail,
            metrics=metrics,
        )
    if not raw_ids:
        detail = "Raw materialization ready"
        if missing_blobs:
            detail += f"; {_raw_materialization_missing_blob_detail(candidates, final=True)}"
        return _internal_derived_repair_result(
            "raw_materialization",
            repaired_count=0,
            success=missing_blobs == 0,
            detail=detail,
            metrics=metrics,
        )
    if not executable_raw_ids:
        return _internal_derived_repair_result(
            "raw_materialization",
            repaired_count=0,
            success=False,
            detail=(
                f"Raw materialization blocked: {len(oversized_raw_ids):,} raw rows exceed actual replay limit "
                f"{_format_bytes(RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES)}; "
                f"largest={_format_bytes(candidates.max_blob_bytes)}. Use a streaming provider-specific "
                "repair path before replaying these rows."
            ),
            metrics=metrics,
        )

    async def _run() -> tuple[int, int]:
        from polylogue.pipeline.services.parsing import ParsingService
        from polylogue.storage.repository import SessionRepository
        from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

        backend = SQLiteBackend(db_path=config.archive_root / "index.db")
        repository = SessionRepository(backend=backend, archive_root=config.archive_root)
        processed_total = 0
        failure_total = 0
        try:
            service = ParsingService(repository=repository, archive_root=config.archive_root, config=config)
            total = len(executable_raw_ids)
            if progress_callback is not None:
                if total == 1:
                    raw_id = executable_raw_ids[0]
                    raw_size = candidates.raw_blob_bytes.get(raw_id, 0)
                    progress_callback(
                        0,
                        desc=(f"raw_materialization: parsing raw 1/1 {raw_id} size={_format_bytes(raw_size)}"),
                    )
                else:
                    progress_callback(
                        0,
                        desc=(
                            f"raw_materialization: parsing {total:,} selected raw rows "
                            f"of {len(candidate_raw_ids):,} candidates"
                        ),
                    )
            result = await service.parse_from_raw(
                raw_ids=executable_raw_ids,
                progress_callback=progress_callback,
                # Raw materialization is a convergence repair from durable
                # source evidence. If the index still has an older row for the
                # same native session whose source raw row disappeared, the
                # normal duplicate-protection path would preserve stale
                # unbacked content and leave readiness permanently blocked.
                force_write=True,
                repair_message_fts=False,
            )
            processed_total += len(result.processed_ids)
            failure_total += result.parse_failures
            if progress_callback is not None:
                progress_callback(
                    total,
                    desc=f"raw_materialization: parsed {total:,} raw rows changed={processed_total}",
                )
            return processed_total, failure_total
        finally:
            await repository.close()

    try:
        processed, failures = asyncio.run(_run())
    except Exception as exc:
        return _internal_derived_repair_result(
            "raw_materialization",
            repaired_count=0,
            success=False,
            detail=f"Failed to materialize raw evidence: {exc}",
            metrics=metrics,
        )
    metrics["raw_materialization_parse_failure_count"] = float(failures)
    metrics["raw_materialization_session_change_count"] = float(processed)
    if candidates.already_parsed:
        detail = (
            f"Replayed {len(executable_raw_ids):,} of {len(candidate_raw_ids):,} raw rows "
            f"({candidates.already_parsed:,} already parsed but not materialized); "
            f"{processed:,} sessions changed; message FTS left to ingest triggers or daemon convergence"
        )
    else:
        detail = (
            f"Replayed {len(executable_raw_ids):,} of {len(candidate_raw_ids):,} acquired-but-unparsed raw rows; "
            f"{processed:,} sessions changed; message FTS left to ingest triggers or daemon convergence"
        )
    if missing_blobs:
        detail += f"; {_raw_materialization_missing_blob_detail(candidates, final=True)}"
    if oversized_raw_ids:
        detail += (
            f"; {len(oversized_raw_ids):,} raw rows remain blocked by replay limit "
            f"{_format_bytes(RAW_MATERIALIZATION_EXECUTE_BLOB_LIMIT_BYTES)}"
        )
    if oversized_stream_safe_raw_ids:
        detail += f"; {len(oversized_stream_safe_raw_ids):,} oversized stream-record raw rows used streaming replay"
    if failures:
        detail += f"; {failures:,} raw rows failed during parse/write"
    return _internal_derived_repair_result(
        "raw_materialization",
        repaired_count=processed,
        success=missing_blobs == 0 and failures == 0 and not oversized_raw_ids,
        detail=detail,
        metrics=metrics,
    )


def _to_repair_result(result: BackfillResult) -> RepairResult:
    """Adapt a ``BackfillResult`` to the shared ``RepairResult`` shape."""
    return RepairResult(
        name=result.name,
        category=result.category,
        destructive=result.destructive,
        repaired_count=result.repaired_count,
        success=result.success,
        detail=result.detail,
    )


def preview_message_type_backfill(*, count: int) -> RepairResult:
    """Preview handler for the #839 message_type backfill.

    Thin shim over ``message_type_backfill.preview_backfill`` so the
    repair orchestrator's preview dispatch keeps working.
    """
    from polylogue.storage.message_type_backfill import preview_backfill

    return _to_repair_result(preview_backfill(count=count))


def repair_message_type_backfill(config: Config, dry_run: bool = False) -> RepairResult:
    """Backfill ``message_type`` for pre-#839 rows.

    Delegates to ``storage.message_type_backfill.run_backfill``; the
    implementation lives there to keep this module under its file-size
    budget (see ``docs/plans/file-size-budgets.yaml``).
    """
    from polylogue.storage.message_type_backfill import run_backfill

    return _to_repair_result(run_backfill(config, dry_run=dry_run))


_PREVIEW_HANDLERS: dict[str, Callable[..., RepairResult]] = {
    "session_insights": preview_session_insights,
    "message_type_backfill": preview_message_type_backfill,
    "orphaned_messages": preview_orphaned_messages,
    "empty_sessions": preview_empty_sessions,
    "orphaned_attachments": preview_orphaned_attachments,
    "orphaned_blobs": preview_orphaned_blobs,
    "superseded_raw_snapshots": preview_superseded_raw_snapshots,
}


_REPAIR_HANDLERS: dict[str, Callable[..., RepairResult]] = {
    "session_insights": repair_session_insights,
    "message_type_backfill": repair_message_type_backfill,
    "orphaned_messages": repair_orphaned_messages,
    "empty_sessions": repair_empty_sessions,
    "orphaned_attachments": repair_orphaned_attachments,
    "orphaned_blobs": repair_orphaned_blobs,
    "superseded_raw_snapshots": repair_superseded_raw_snapshots,
}


# ---------------------------------------------------------------------------
# Orchestration (run_safe_repairs, run_archive_cleanup, run_selected_maintenance)
# ---------------------------------------------------------------------------


def run_safe_repairs(
    config: Config,
    dry_run: bool = False,
    *,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
    session_insight_progress_callback: ProgressCallback | None = None,
    session_insight_progress_total: int | None = None,
) -> list[RepairResult]:
    preview_counts = preview_counts or {}
    selected = set(targets) if targets else set(SAFE_REPAIR_TARGETS)
    results: list[RepairResult] = []
    for target_name in SAFE_REPAIR_TARGETS:
        if target_name not in selected:
            continue
        if dry_run and target_name in preview_counts:
            preview = _PREVIEW_HANDLERS.get(target_name)
            if preview is not None:
                results.append(preview(count=preview_counts[target_name]))
                continue
        repair = _REPAIR_HANDLERS[target_name]
        if target_name == "session_insights":
            results.append(
                repair(
                    config,
                    dry_run=dry_run,
                    progress_callback=session_insight_progress_callback,
                    progress_total=session_insight_progress_total,
                )
            )
            continue
        results.append(repair(config, dry_run=dry_run))
    return results


def run_archive_cleanup(
    config: Config,
    dry_run: bool = False,
    *,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
) -> list[RepairResult]:
    preview_counts = preview_counts or {}
    selected = set(targets) if targets else set(CLEANUP_TARGETS)
    results: list[RepairResult] = []
    for target_name in CLEANUP_TARGETS:
        if target_name not in selected:
            continue
        if dry_run and target_name in preview_counts:
            results.append(_PREVIEW_HANDLERS[target_name](count=preview_counts[target_name]))
            continue
        results.append(_REPAIR_HANDLERS[target_name](config, dry_run=dry_run))
    return results


def run_selected_maintenance(
    config: Config,
    *,
    repair: bool,
    cleanup: bool,
    dry_run: bool = False,
    preview_counts: dict[str, int] | None = None,
    targets: tuple[str, ...] = (),
    session_insight_progress_callback: ProgressCallback | None = None,
    session_insight_progress_total: int | None = None,
) -> list[RepairResult]:
    blockers = offline_maintenance_blockers(
        config,
        repair=repair,
        cleanup=cleanup,
        dry_run=dry_run,
        targets=targets,
    )
    if blockers:
        return blockers
    results: list[RepairResult] = []
    repair_targets = tuple(name for name in targets if name in SAFE_REPAIR_TARGETS)
    cleanup_targets = tuple(name for name in targets if name in CLEANUP_TARGETS)
    if repair:
        results.extend(
            run_safe_repairs(
                config,
                dry_run=dry_run,
                preview_counts=preview_counts,
                targets=repair_targets,
                session_insight_progress_callback=session_insight_progress_callback,
                session_insight_progress_total=session_insight_progress_total,
            )
        )
    if cleanup:
        results.extend(
            run_archive_cleanup(config, dry_run=dry_run, preview_counts=preview_counts, targets=cleanup_targets)
        )
    return results


__all__ = [
    "ArchiveDebtStatus",
    "RepairResult",
    "collect_archive_debt_statuses_sync",
    "count_empty_sessions_sync",
    "count_orphaned_attachments_sync",
    "count_orphaned_blobs_sync",
    "count_superseded_raw_snapshots_sync",
    "count_orphaned_messages_sync",
    "count_messages_by_type_sync",
    "count_unclassified_message_type_sync",
    "preview_counts_from_archive_debt",
    "preview_empty_sessions",
    "preview_orphaned_attachments",
    "preview_orphaned_blobs",
    "preview_superseded_raw_snapshots",
    "preview_orphaned_messages",
    "preview_message_type_backfill",
    "preview_session_insights",
    "raw_materialization_replay_backlog",
    "repair_empty_sessions",
    "repair_message_type_backfill",
    "repair_orphaned_attachments",
    "repair_orphaned_blobs",
    "repair_raw_materialization",
    "repair_superseded_raw_snapshots",
    "repair_orphaned_messages",
    "repair_session_insights",
    "run_archive_cleanup",
    "run_safe_repairs",
    "run_selected_maintenance",
    "session_insight_repair_count",
]

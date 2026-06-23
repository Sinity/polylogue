"""Live batch progress recording helpers."""

from __future__ import annotations

import sqlite3
from json import dumps as json_dumps
from pathlib import Path
from typing import Any

from polylogue.core.metrics import (
    read_cgroup_memory_current_mb,
    read_cgroup_memory_peak_mb,
    read_cgroup_memory_stat_mb,
    read_cgroup_memory_swap_current_mb,
    read_cgroup_path,
    read_current_rss_mb,
    read_peak_rss_children_mb,
    read_peak_rss_self_mb,
)


def record_attempt_progress(
    cursor: Any,
    attempt_id: str,
    *,
    phase: str,
    status: str = "running",
    queued_file_count: int | None = None,
    needed_file_count: int | None = None,
    skipped_file_count: int | None = None,
    succeeded_file_count: int,
    failed_file_count: int,
    input_bytes: int | None = None,
    source_payload_read_bytes: int,
    cursor_fingerprint_read_bytes: int,
    archive_write_bytes_delta: int | None = None,
    parse_time_s: float,
    convergence_time_s: float = 0.0,
    total_time_s: float | None = None,
    current_source: str | None = None,
    current_path: Path | None = None,
    error: str | None = None,
    stage_payload: dict[str, object] | None = None,
    stage_timings_s: dict[str, float] | None = None,
    stale_cursor_write_count: int | None = None,
) -> None:
    rss_current_mb = read_current_rss_mb()
    rss_peak_self_mb = read_peak_rss_self_mb()
    rss_peak_children_mb = read_peak_rss_children_mb()
    cgroup_path = read_cgroup_path()
    cgroup_current_mb = read_cgroup_memory_current_mb()
    cgroup_peak_mb = read_cgroup_memory_peak_mb()
    cgroup_swap_mb = read_cgroup_memory_swap_current_mb()
    cgroup_stat_mb = read_cgroup_memory_stat_mb()
    progress_recorded = cursor.update_ingest_attempt(
        attempt_id,
        phase=phase,
        status=status,
        succeeded_file_count=succeeded_file_count,
        failed_file_count=failed_file_count,
        source_payload_read_bytes=source_payload_read_bytes,
        cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
        parse_time_s=round(parse_time_s, 6),
        convergence_time_s=round(convergence_time_s, 6),
        current_source=current_source,
        current_path=current_path,
        error=error,
        stage_payload=stage_payload,
        rss_current_mb=rss_current_mb,
        rss_peak_self_mb=rss_peak_self_mb,
        rss_peak_children_mb=rss_peak_children_mb,
        cgroup_path=cgroup_path,
        cgroup_memory_current_mb=cgroup_current_mb,
        cgroup_memory_peak_mb=cgroup_peak_mb,
        cgroup_memory_swap_current_mb=cgroup_swap_mb,
        cgroup_memory_anon_mb=cgroup_stat_mb.get("anon"),
        cgroup_memory_file_mb=cgroup_stat_mb.get("file"),
        cgroup_memory_inactive_file_mb=cgroup_stat_mb.get("inactive_file"),
        stale_cursor_write_count=stale_cursor_write_count,
    )
    if progress_recorded is False:
        return
    record_event = getattr(cursor, "record_ingest_stage_event", None)
    if not callable(record_event):
        return
    record_event(
        attempt_id,
        phase=phase,
        status=status,
        queued_file_count=queued_file_count,
        needed_file_count=needed_file_count,
        skipped_file_count=skipped_file_count,
        succeeded_file_count=succeeded_file_count,
        failed_file_count=failed_file_count,
        input_bytes=input_bytes,
        source_payload_read_bytes=source_payload_read_bytes,
        cursor_fingerprint_read_bytes=cursor_fingerprint_read_bytes,
        archive_write_bytes_delta=archive_write_bytes_delta,
        parse_time_s=round(parse_time_s, 6),
        convergence_time_s=round(convergence_time_s, 6),
        total_time_s=total_time_s,
        current_source=current_source,
        current_path=current_path,
        error=error,
        rss_current_mb=rss_current_mb,
        rss_peak_self_mb=rss_peak_self_mb,
        rss_peak_children_mb=rss_peak_children_mb,
        cgroup_path=cgroup_path,
        cgroup_memory_current_mb=cgroup_current_mb,
        cgroup_memory_peak_mb=cgroup_peak_mb,
        cgroup_memory_swap_current_mb=cgroup_swap_mb,
        cgroup_memory_anon_mb=cgroup_stat_mb.get("anon"),
        cgroup_memory_file_mb=cgroup_stat_mb.get("file"),
        cgroup_memory_inactive_file_mb=cgroup_stat_mb.get("inactive_file"),
        stage_payload=stage_payload,
        stage_timings_json=None
        if not stage_timings_s
        else json_dumps(stage_timings_s, sort_keys=True, separators=(",", ":")),
    )


def session_ids_for_source_path(path: Path, *, archive_root: Path | None = None) -> tuple[str, ...]:
    if archive_root is None:
        return ()
    return _schema_archive_session_ids_for_source_path(archive_root, path)


def _schema_archive_session_ids_for_source_path(archive_root: Path, path: Path) -> tuple[str, ...]:
    index_db = archive_root / "index.db"
    source_db = archive_root / "source.db"
    if not index_db.exists() or not source_db.exists():
        return ()
    try:
        conn = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
        try:
            conn.execute("ATTACH DATABASE ? AS source_tier", (f"file:{source_db}?mode=ro",))
            rows = conn.execute(
                """
                SELECT s.session_id
                FROM sessions AS s
                JOIN source_tier.raw_sessions AS r ON r.raw_id = s.raw_id
                WHERE r.source_path = ?
                ORDER BY s.sort_key_ms DESC, s.created_at_ms DESC, s.session_id
                """,
                (str(path),),
            ).fetchall()
            conn.execute("DETACH DATABASE source_tier")
        finally:
            conn.close()
    except sqlite3.Error:
        return ()
    return tuple(str(row[0]) for row in rows if row[0])


__all__ = ["session_ids_for_source_path", "record_attempt_progress"]

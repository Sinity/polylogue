"""Live batch progress recording helpers."""

from __future__ import annotations

from json import dumps as json_dumps
from pathlib import Path
from typing import Any

from polylogue.core.metrics import (
    read_cgroup_memory_current_mb,
    read_cgroup_memory_peak_mb,
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
    cursor.update_ingest_attempt(
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
        rss_current_mb=rss_current_mb,
        rss_peak_self_mb=rss_peak_self_mb,
        rss_peak_children_mb=rss_peak_children_mb,
        cgroup_path=cgroup_path,
        cgroup_memory_current_mb=cgroup_current_mb,
        cgroup_memory_peak_mb=cgroup_peak_mb,
        cgroup_memory_swap_current_mb=cgroup_swap_mb,
        stale_cursor_write_count=stale_cursor_write_count,
    )
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
        stage_timings_json=None
        if not stage_timings_s
        else json_dumps(stage_timings_s, sort_keys=True, separators=(",", ":")),
    )


def conversation_ids_for_source_path(cursor: Any, path: Path) -> tuple[str, ...]:
    try:
        with cursor._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT c.conversation_id
                FROM conversations AS c
                JOIN raw_conversations AS r ON r.raw_id = c.raw_id
                WHERE r.source_path = ?
                ORDER BY c.conversation_id
                """,
                (str(path),),
            ).fetchall()
    except Exception:
        return ()
    return tuple(str(row[0]) for row in rows if row[0])


__all__ = ["conversation_ids_for_source_path", "record_attempt_progress"]

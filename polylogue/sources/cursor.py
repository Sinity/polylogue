"""Cursor state management and path selection for source iteration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from polylogue.logging import get_logger
from polylogue.sources.assembly import SidecarData
from polylogue.storage.cursor_state import CursorFailurePayload, CursorStatePayload
from polylogue.types import Provider

logger = get_logger(__name__)

# Slice B: stat fields used for cursor-based fast-path skipping.
# Maps cursor column name (shorter, DB-friendly) to stat_result attribute.
_CURSOR_STAT_MAP: dict[str, str] = {
    "st_dev": "st_dev",
    "st_ino": "st_ino",
    "st_size": "st_size",
    "mtime_ns": "st_mtime_ns",
}


def _stat_matches_cursor(st: os.stat_result, cursor_fields: dict[str, object]) -> bool:
    """Return True when *all* cursor-stored stat fields match the live file stat.

    The cursor dict may be sparse; only fields present in both sides are
    compared. At minimum *st_dev*, *st_ino*, *st_size*, and *mtime_ns*
    must match for the file to be considered unchanged.
    """
    for cursor_field, stat_attr in _CURSOR_STAT_MAP.items():
        cursor_val = cursor_fields.get(cursor_field)
        if cursor_val is None:
            return False
        stat_val = getattr(st, stat_attr, None)
        if stat_val is None or stat_val != cursor_val:
            return False
    return True


def _get_file_mtime(path: Path) -> str | None:
    """Get ISO-format mtime for a path, or None on OSError."""
    try:
        st = path.stat()
        return datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return None


def _record_cursor_failure(
    cursor_state: CursorStatePayload | None,
    path: str,
    error: str,
) -> None:
    """Record a file processing failure in cursor_state."""
    if cursor_state is not None:
        failed_files = cursor_state.setdefault("failed_files", [])
        failed_files.append(CursorFailurePayload(path=path, error=error))
        cursor_state["failed_count"] = int(cursor_state.get("failed_count", 0) or 0) + 1


def _initialize_cursor_state(
    cursor_state: CursorStatePayload | None,
    paths: list[Path],
) -> None:
    """Populate cursor bookkeeping for a source iteration pass."""
    if cursor_state is None:
        return

    cursor_state["file_count"] = len(paths)
    cursor_state.setdefault("failed_files", [])
    cursor_state.setdefault("failed_count", 0)
    if not paths:
        return

    try:
        latest = max(paths, key=lambda p: p.stat().st_mtime)
        cursor_state["latest_mtime"] = latest.stat().st_mtime
        cursor_state["latest_path"] = str(latest)
    except OSError:
        pass


def _select_paths_for_processing(
    paths: list[Path],
    *,
    include_file_mtime: bool,
    known_mtimes: dict[str, str] | None = None,
    known_cursors: dict[str, dict[str, object]] | None = None,
) -> tuple[list[tuple[Path, str | None]], int]:
    """Filter unchanged files and return `(path, file_mtime)` tuples.

    Slice B: When ``known_cursors`` is provided and all stat fields (dev,
    ino, size, mtime_ns) match the live file, the file is skipped without
    falling through to the mtime comparison or any full-file I/O.
    """
    selected: list[tuple[Path, str | None]] = []
    skipped_mtime = 0
    zip_known_mtimes: dict[str, str] = {}

    if known_mtimes:
        for key, mtime in known_mtimes.items():
            zip_path, sep, _entry = key.partition(":")
            if sep and zip_path not in zip_known_mtimes:
                zip_known_mtimes[zip_path] = mtime

    for path in paths:
        # Slice B: fast-path against known cursor stat fields.
        if known_cursors is not None:
            try:
                st = path.stat()
                cursor_fields = known_cursors.get(str(path))
                if cursor_fields is not None and _stat_matches_cursor(st, cursor_fields):
                    skipped_mtime += 1
                    logger.debug(
                        "cursor match: %s (dev=%s ino=%s size=%s mtime=%s)",
                        path,
                        st.st_dev,
                        st.st_ino,
                        st.st_size,
                        st.st_mtime_ns,
                    )
                    continue
            except OSError:
                pass
            # Fall through to mtime-based comparison if cursor check is
            # inconclusive (no cursor data or stat mismatch).

        file_mtime = _get_file_mtime(path) if include_file_mtime else None
        if known_mtimes and file_mtime:
            path_str = str(path)
            # Direct match (non-ZIP files stored by exact path)
            if known_mtimes.get(path_str) == file_mtime:
                skipped_mtime += 1
                continue
            # ZIP match: entries are stored as "path.zip:entry.json" — check
            # if any entry for this ZIP has matching mtime
            if path_str.endswith(".zip") and zip_known_mtimes.get(path_str) == file_mtime:
                skipped_mtime += 1
                continue
        selected.append((path, file_mtime))

    return selected, skipped_mtime


def _log_source_iteration_summary(
    *,
    source_name: str,
    total_paths: int,
    skipped_mtime: int,
    failed_count: int,
    failure_kind: str,
) -> None:
    """Emit common skip/failure summaries for source iterators."""
    if skipped_mtime > 0:
        logger.info(
            "Skipped %d of %d files from source %r (unchanged mtime)",
            skipped_mtime,
            total_paths,
            source_name,
        )

    if failed_count > 0:
        logger.warning(
            "Skipped %d of %d files from source %r due to %s errors. Run with --verbose for details.",
            failed_count,
            total_paths,
            source_name,
            failure_kind,
        )


@dataclass
class _ParseContext:
    """All context needed to parse a stream and yield conversations."""

    provider_hint: Provider
    should_group: bool
    source_path_str: str  # For RawConversationData.source_path
    fallback_id: str  # path.stem, used as fallback conversation ID
    file_mtime: str | None
    capture_raw: bool
    sidecar_data: SidecarData


__all__ = [
    "_get_file_mtime",
    "_record_cursor_failure",
    "_initialize_cursor_state",
    "_select_paths_for_processing",
    "_log_source_iteration_summary",
    "_ParseContext",
]

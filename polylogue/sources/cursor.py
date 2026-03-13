"""Cursor state management and path selection for source iteration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from polylogue.logging import get_logger
from polylogue.types import Provider

from .parsers.claude import SessionIndexEntry

logger = get_logger(__name__)


def _get_file_mtime(path: Path) -> str | None:
    """Get ISO-format mtime for a path, or None on OSError."""
    try:
        st = path.stat()
        return datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return None


def _record_cursor_failure(
    cursor_state: dict[str, Any] | None,
    path: str,
    error: str,
) -> None:
    """Record a file processing failure in cursor_state."""
    if cursor_state is not None:
        cursor_state["failed_files"].append({"path": path, "error": error})
        cursor_state["failed_count"] = cursor_state.get("failed_count", 0) + 1


def _initialize_cursor_state(
    cursor_state: dict[str, Any] | None,
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
) -> tuple[list[tuple[Path, str | None]], int]:
    """Filter unchanged files and return `(path, file_mtime)` tuples."""
    selected: list[tuple[Path, str | None]] = []
    skipped_mtime = 0

    for path in paths:
        file_mtime = _get_file_mtime(path) if include_file_mtime else None
        if known_mtimes and file_mtime and known_mtimes.get(str(path)) == file_mtime:
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
            "Skipped %d of %d files from source %r due to %s errors. "
            "Run with --verbose for details.",
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
    session_index: dict[str, SessionIndexEntry]
    detect_path: Path  # For detect_provider calls on individual payloads


__all__ = [
    "_get_file_mtime",
    "_record_cursor_failure",
    "_initialize_cursor_state",
    "_select_paths_for_processing",
    "_log_source_iteration_summary",
    "_ParseContext",
]

"""Cursor state management and path selection for source iteration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from polylogue.core.enums import Origin, Provider
from polylogue.logging import get_logger
from polylogue.sources.assembly import SidecarData
from polylogue.sources.origin_specs import database_member_for_filename
from polylogue.sources.sqlite_snapshot import is_sqlite_path
from polylogue.storage.cursor_state import CursorFailurePayload, CursorStatePayload

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
    source_name: str | None = None,
) -> tuple[list[tuple[Path, str | None]], int]:
    """Filter unchanged files and return `(path, file_mtime)` tuples.

    Byte files may skip only when their complete captured stat tuple matches.
    ZIP containers and mutable SQLite inputs require acquisition: timestamps
    cannot certify ZIP membership or a database's committed WAL revision.
    ``known_mtimes`` remains a caller input but never proves local byte currency.
    """
    selected: list[tuple[Path, str | None]] = []
    skipped_mtime = 0
    for path in paths:
        cursor_fields = known_cursors.get(str(path)) if known_cursors is not None else None
        is_hermes = source_name == Provider.HERMES.value or (
            cursor_fields is not None and cursor_fields.get("origin") == Origin.HERMES_SESSION.value
        )
        if is_hermes:
            # Neither timestamp nor equal byte statistics can prove that a
            # mutable alias still declares the same Hermes profile.
            from polylogue.sources.parsers.hermes_identity import observe_profile_namespace

            try:
                observed = path.stat()
                profile = observe_profile_namespace(path, observed)
                profile_matches = cursor_fields is not None and cursor_fields.get("captured_profile_key") == profile.key
            except OSError:
                profile_matches = False
            if not profile_matches:
                selected.append((path, _get_file_mtime(path) if include_file_mtime else None))
                continue
        # Declared database members carry a logical revision, not the main
        # file's physical stat identity. Other SQLite-shaped names also stay
        # on acquisition's existing bound shape/admission route.
        logical_input = database_member_for_filename(path.name) is not None or is_sqlite_path(path)
        if known_cursors is not None and path.suffix.lower() != ".zip" and not logical_input:
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
        file_mtime = _get_file_mtime(path) if include_file_mtime else None
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
    """All context needed to parse a stream and yield sessions."""

    provider_hint: Provider
    should_group: bool
    source_path_str: str  # For RawSessionData.source_path
    fallback_id: str  # path.stem, used as fallback session ID
    file_mtime: str | None
    capture_raw: bool
    sidecar_data: SidecarData
    # The origin this file's location binds, or ``None`` where the location
    # classifies (the import inbox, and export archives opened from it).
    bound_provider: Provider | None = None


__all__ = [
    "_get_file_mtime",
    "_record_cursor_failure",
    "_initialize_cursor_state",
    "_select_paths_for_processing",
    "_log_source_iteration_summary",
    "_ParseContext",
]

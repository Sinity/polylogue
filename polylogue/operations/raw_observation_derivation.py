"""Product seam for raw recovery through the derivation kernel."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from polylogue.daemon.derivation import (
    Budget,
    DerivationFrame,
    DerivationRegistry,
    DerivationReport,
    PassCursor,
    converge,
)
from polylogue.storage.archive_identity import ArchiveLocation
from polylogue.storage.derived.raw import RAW_OBSERVATION_DOMAIN as _RAW_OBSERVATION_DOMAIN
from polylogue.storage.derived.raw import RawObservationDerivation, RawObservationScope

RAW_OBSERVATION_DOMAIN = _RAW_OBSERVATION_DOMAIN
_RAW_OBSERVATION_RECIPE_VERSION = RawObservationDerivation.recipe_version


def make_raw_observation_derivation(
    archive_root: Path, *, max_payload_bytes: int, stream_safe_only: bool = False
) -> RawObservationDerivation:
    """Construct the storage-owned raw adapter from the operations boundary."""
    return RawObservationDerivation(
        archive_root, max_payload_bytes=max_payload_bytes, stream_safe_only=stream_safe_only
    )


def raw_observation_output_session_ids(archive_root: Path, raw_id: str) -> tuple[str, ...]:
    """Read every active session output in the seed raw's replay component."""
    from polylogue.operations.operation_context import open_operation_read

    with open_operation_read(archive_root) as pinned:
        component_raw_ids, _logical_keys = pinned.archive.expand_raw_membership_selection([raw_id])
        if not component_raw_ids:
            return ()
        index = pinned.archive.index_connection
        if index is None:
            return ()
        rows = index.execute(
            f"""SELECT s.session_id
            FROM sessions AS s
            JOIN raw_revision_heads AS head
              ON head.session_id = s.session_id
             AND head.accepted_raw_id = s.raw_id
            WHERE head.accepted_raw_id IN ({",".join("?" for _ in component_raw_ids)})
            ORDER BY s.session_id""",
            component_raw_ids,
        ).fetchall()
        return tuple(str(row[0]) for row in rows)


def raw_observation_frame(
    archive_root: Path,
    *,
    source_roots: Sequence[Path] = (),
    raw_ids: Sequence[str] = (),
) -> DerivationFrame:
    return DerivationFrame(
        archive_root=str(archive_root),
        source_revision=str(ArchiveLocation.resolve(archive_root).active_index_path.resolve()),
        recipe_versions={RAW_OBSERVATION_DOMAIN: _RAW_OBSERVATION_RECIPE_VERSION},
        scope=RawObservationScope(source_roots=tuple(source_roots), raw_ids=tuple(raw_ids)),
    )


def raw_observation_pending_roots(
    archive_root: Path,
    paths: Sequence[Path],
    *,
    continuations: dict[tuple[Path, ...], tuple[str, str | None]] | None = None,
    limit: int = 128,
) -> set[Path]:
    """Inspect one page; an unfinished traversal remains pending.

    The caller may retain a disposable continuation after an all-valid page.
    A page containing pending work is revisited until publication resolves it.
    No partial all-valid prefix can certify the entire selected source scope.
    """
    adapter = make_raw_observation_derivation(archive_root, max_payload_bytes=64 * 1024 * 1024)
    pending: set[Path] = set()
    ordered = tuple(dict.fromkeys(paths))
    if not ordered:
        return pending
    frame = raw_observation_frame(archive_root, source_roots=ordered)
    binding = frame.source_revision + ":" + frame.recipe_version(RAW_OBSERVATION_DOMAIN)
    previous_binding, cursor = (continuations or {}).get(ordered, (binding, None))
    if previous_binding != binding:
        cursor = None
    keys, next_cursor = adapter.required_page(frame, cursor=cursor, limit=limit)
    stale = tuple(key for key, status in adapter.inspect(frame, keys).items() if status != "valid") if keys else ()
    for source_path in adapter.source_paths(stale).values():
        pending.update(
            path
            for path in ordered
            if source_path == str(path).rstrip("/") or source_path.startswith(str(path).rstrip("/") + "/")
        )
    if continuations is not None:
        continuations[ordered] = (binding, cursor if stale else next_cursor)
    if next_cursor is not None:
        pending.update(ordered)
    return pending


def raw_observation_backlog_snapshot(archive_root: Path, *, limit: int) -> dict[str, object]:
    """Describe one bounded canonical raw-observation page for status surfaces.

    Status is observational and must not reintroduce a second all-raw
    materialization census.  The returned rows are therefore a page from the
    same derivation traversal fair intake owns, with its pending verdicts from
    the adapter's authoritative inspection.
    """
    if limit < 1:
        raise ValueError("limit must be positive")

    def unavailable(reason: str) -> dict[str, object]:
        return {
            "available": False,
            "reason": reason,
            "scan": "bounded_raw_observation_page",
            "candidate_count": 0,
            "total_blob_bytes": 0,
            "max_blob_bytes": 0,
            "top_raw_rows": [],
            "origin_summary": [],
            "source_path_summary": [],
            "page_limit": limit,
            "page_complete": True,
        }

    adapter = make_raw_observation_derivation(archive_root, max_payload_bytes=64 * 1024 * 1024)
    frame = raw_observation_frame(archive_root)
    try:
        raw_ids, next_cursor = adapter.required_page(frame, cursor=None, limit=limit)
        states = adapter.inspect(frame, raw_ids)
    except FileNotFoundError as exc:
        return unavailable(str(exc))
    pending_ids = tuple(raw_id for raw_id in raw_ids if states.get(raw_id) != "valid")
    if not pending_ids:
        return {
            "available": True,
            "scan": "bounded_raw_observation_page",
            "candidate_count": 0,
            "total_blob_bytes": 0,
            "max_blob_bytes": 0,
            "top_raw_rows": [],
            "origin_summary": [],
            "source_path_summary": [],
            "page_limit": limit,
            "page_complete": next_cursor is None,
        }
    with adapter._read() as conn:
        selected = conn.execute(
            f"SELECT raw_id, origin, source_path, blob_size FROM raw_sessions "
            f"WHERE raw_id IN ({','.join('?' for _ in pending_ids)})",
            pending_ids,
        ).fetchall()
    rows: list[dict[str, Any]] = sorted(
        (
            {
                "raw_id": str(row["raw_id"]),
                "origin": str(row["origin"]),
                "source_path": str(row["source_path"] or ""),
                "blob_size": int(row["blob_size"] or 0),
                "oversized": False,
                "stream_safe": True,
            }
            for row in selected
        ),
        key=lambda row: (-int(row["blob_size"]), str(row["raw_id"])),
    )
    origin_summary: dict[str, dict[str, int | str]] = {}
    source_path_summary: dict[str, dict[str, int | str]] = {}
    for row in rows:
        origin = str(row["origin"])
        path = str(row["source_path"])
        for summary, key, name in ((origin_summary, origin, "origin"), (source_path_summary, path, "source_path")):
            entry = summary.setdefault(key, {name: key, "raw_count": 0, "total_blob_bytes": 0, "max_blob_bytes": 0})
            entry["raw_count"] = int(entry["raw_count"]) + 1
            entry["total_blob_bytes"] = int(entry["total_blob_bytes"]) + int(row["blob_size"])
            entry["max_blob_bytes"] = max(int(entry["max_blob_bytes"]), int(row["blob_size"]))
    return {
        "available": True,
        "scan": "bounded_raw_observation_page",
        "candidate_count": len(rows),
        "total_blob_bytes": sum(int(row["blob_size"]) for row in rows),
        "max_blob_bytes": max((int(row["blob_size"]) for row in rows), default=0),
        "top_raw_rows": rows,
        "origin_summary": sorted(origin_summary.values(), key=lambda row: str(row["origin"])),
        "source_path_summary": sorted(source_path_summary.values(), key=lambda row: str(row["source_path"])),
        "page_limit": limit,
        "page_complete": next_cursor is None,
    }


def converge_raw_observations(
    archive_root: Path,
    *,
    source_roots: Sequence[Path],
    limit: int,
    max_payload_bytes: int,
    cursor: PassCursor | None = None,
) -> DerivationReport:
    adapter = make_raw_observation_derivation(archive_root, max_payload_bytes=max_payload_bytes)
    return converge(
        DerivationRegistry((adapter,)),
        raw_observation_frame(archive_root, source_roots=source_roots),
        # Each discovered key needs inspection before compute and again to
        # certify publication. Discovery alone must not exhaust that budget.
        budget=Budget(page=min(128, limit), discovery=limit, inspection=2 * limit, compute=limit, publication=limit),
        cursor=cursor,
    )

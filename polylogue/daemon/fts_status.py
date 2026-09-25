"""FTS readiness projection for daemon status payloads."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from polylogue.core.payload_coercion import row_int as _row_int
from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.logging import WARNING, emit
from polylogue.operations.status_protocol import ComponentSnapshot, StatusComponentRegistry, StatusComponentSpec
from polylogue.storage.fts.fts_lifecycle import FtsInvariantSnapshot, FtsSurfaceInvariant, fts_invariant_snapshot_sync
from polylogue.storage.sqlite.connection_profile import open_readonly_connection

_FTS_READINESS_DEADLINE_S = 1.5
_FTS_READINESS_TTL_S = 1.0
_FTS_READINESS_REGISTRIES_LOCK = threading.Lock()
_FTS_READINESS_REGISTRIES: dict[str, StatusComponentRegistry] = {}


def _fts_readiness_fingerprint(dbf: Path) -> str:
    """Identify the SQLite generation and committed/WAL state being inspected."""
    candidates = [dbf]
    try:
        archive_index = _archive_index_path_for(dbf)
    except (OSError, sqlite3.Error):
        archive_index = None
    if archive_index is not None and archive_index != dbf:
        candidates.append(archive_index)
    parts: list[str] = []
    for path in candidates:
        for candidate in (path, Path(f"{path}-wal")):
            try:
                stat = candidate.stat()
            except OSError:
                parts.append(f"{candidate.resolve(strict=False)}:missing")
            else:
                parts.append(f"{candidate.resolve(strict=False)}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(parts)


def _fts_readiness_registry(dbf: Path) -> StatusComponentRegistry:
    key = str(dbf.resolve(strict=False))
    with _FTS_READINESS_REGISTRIES_LOCK:
        registry = _FTS_READINESS_REGISTRIES.get(key)
        if registry is None:
            registry = StatusComponentRegistry(
                [
                    StatusComponentSpec(
                        name="fts_readiness",
                        scope="lexical",
                        collector=lambda: _collect_fts_readiness_component(dbf, exact=False),
                        deadline_s=_FTS_READINESS_DEADLINE_S,
                        cost_class="moderate",
                        fingerprint=lambda: _fts_readiness_fingerprint(dbf),
                        ttl_s=_FTS_READINESS_TTL_S,
                    )
                ]
            )
            _FTS_READINESS_REGISTRIES[key] = registry
        return registry


class FTSReadiness(BaseModel):
    indexed_surface: str = "messages_fts"
    inspection_state: Literal["fresh", "stale", "refreshing", "timed_out", "unavailable", "degraded"] = "fresh"
    messages_ready: bool = False
    invariant_ready: bool = False
    # Counts have no honest zero default. A missing/corrupt source is not an
    # empty FTS relation, and the status route must carry that distinction.
    message_indexed_count: int | None = None
    message_indexable_count: int | None = None
    coverage_pct: float | None = None
    coverage_exact: bool = False
    surfaces: dict[str, dict[str, int | bool | str | None]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _withhold_unreadable_coverage(cls, value: object) -> object:
        """Do not serialize the reader's error sentinel as measured coverage."""
        if not isinstance(value, dict):
            return value
        if (
            value.get("messages_ready") is False
            and value.get("surfaces") == {}
            and value.get("message_indexed_count") is None
            and value.get("message_indexable_count") is None
        ):
            return {**value, "coverage_pct": None, "coverage_exact": False}
        return value


def _surface_payload(surface: FtsSurfaceInvariant) -> dict[str, int | bool | str | None]:
    return {
        "source_exists": surface.source_exists,
        "exists": surface.exists,
        "source_rows": surface.source_rows,
        "indexed_rows": surface.indexed_rows,
        "triggers_present": surface.triggers_present,
        "missing_rows": surface.missing_rows,
        "excess_rows": surface.excess_rows,
        "duplicate_rows": surface.duplicate_rows,
        "identity_mismatch_rows": surface.identity_mismatch_rows,
        "ready": surface.ready,
        "exact": True,
    }


def _payload_int(surface: dict[str, int | bool | str | None], key: str) -> int:
    return _row_int(surface.get(key))


def _archive_index_path_for(dbf: Path) -> Path | None:
    from polylogue.storage.archive_identity import ArchiveLocation

    index_db = ArchiveLocation.resolve(dbf.parent).active_index_path
    return index_db if index_db.exists() else None


def _collect_fts_readiness_component(dbf: Path, *, exact: bool) -> dict[str, object]:
    """Pair a measured payload with the source generation observed on entry."""
    fingerprint = _fts_readiness_fingerprint(dbf)
    return {
        "payload": _collect_fts_readiness_info(dbf, exact=exact),
        "source_fingerprint": fingerprint,
    }


def _archive_blocks_surface(conn: sqlite3.Connection) -> dict[str, int | bool | str | None]:
    """Read authoritative FTS membership; freshness rows never certify it.

    polylogue-crwl6 AC6: the standing readiness binding is consulted first.
    It is not a freshness record -- it is this domain's own statement of what a
    completed global inspection found, retired by the database itself on any
    write to the block columns the surface reduces.  When no binding stands the
    authoritative inspection runs exactly as before; nothing here reports a
    verdict that was not measured against the current relations.
    """
    # Imported here, not at module scope: polylogue.operations.fts_derivation
    # imports polylogue.daemon.derivation, whose package __init__ reaches back
    # into this module. A module-level import makes importing the operations
    # module directly an ImportError.
    from polylogue.operations.fts_derivation import archive_fts_surface, bound_archive_fts_surface

    bound = bound_archive_fts_surface(conn)
    if bound is not None:
        return bound
    return archive_fts_surface(conn)


def _archive_readiness_payload(conn: sqlite3.Connection, *, exact: bool) -> dict[str, object] | None:
    if not _table_exists(conn, "blocks") and not _table_exists(conn, "messages_fts"):
        return None
    del exact
    blocks = _archive_blocks_surface(conn)
    effective_exact = True
    block_source_rows = _payload_int(blocks, "source_rows")
    block_indexed_rows = _payload_int(blocks, "indexed_rows")
    invariant_ready = bool(blocks["ready"])
    return {
        "indexed_surface": "messages_fts",
        "messages_ready": invariant_ready,
        "invariant_ready": invariant_ready,
        "message_indexed_count": block_indexed_rows,
        "message_indexable_count": block_source_rows,
        # source_rows == 0 means there is nothing to index -- a genuine,
        # vacuously-true "fully covered" state. invariant_ready alone is
        # NOT evidence of coverage: it only proves triggers/tables exist,
        # so reporting 100.0 whenever it happens to be true (regardless of
        # source_rows) fabricates a measured percentage for an unmeasured
        # branch (polylogue-oitx). Report None ("not applicable / unknown")
        # instead; consumers already render that explicitly (polylogue-roax,
        # cli/commands/status.py's null-coverage handling).
        "coverage_pct": (round((block_indexed_rows / block_source_rows) * 100, 1) if block_source_rows > 0 else None),
        "coverage_exact": effective_exact,
        "surfaces": {"messages_fts": blocks},
    }


def _archive_readiness_info(index_db: Path, *, exact: bool) -> dict[str, object] | None:
    if not index_db.exists():
        return None
    try:
        conn = open_readonly_connection(index_db)
        try:
            return _archive_readiness_payload(conn, exact=exact)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        emit(
            "daemon.fts.readiness_query_failed",
            level=WARNING,
            outcome="degraded",
            reason="archive_readiness_unreadable",
            path=index_db,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        # The readiness query failed: nothing here was measured.  Every
        # sibling key already reports the not-ready value (polylogue-bu47u:
        # a readiness flag must never return True out of an exception
        # handler). ``coverage_pct`` is unknown, not zero -- consumers render
        # ``None`` explicitly as "coverage unknown".
        return {
            "indexed_surface": "messages_fts",
            "inspection_state": "unavailable",
            "messages_ready": False,
            "invariant_ready": False,
            "coverage_pct": None,
            "coverage_exact": False,
            "surfaces": {},
        }


def _exact_readiness_payload(snapshot: FtsInvariantSnapshot) -> dict[str, object]:
    surfaces = {surface.name: _surface_payload(surface) for surface in snapshot.surfaces}
    messages = snapshot.messages
    # source_rows == 0 is a zero-denominator case -- report unmeasured
    # (None) rather than a fabricated 100.0 (polylogue-oitx).
    coverage_pct = round((messages.indexed_rows / messages.source_rows) * 100, 1) if messages.source_rows > 0 else None
    return {
        "messages_ready": messages.ready,
        "invariant_ready": snapshot.ready,
        "message_indexed_count": messages.indexed_rows,
        "message_indexable_count": messages.source_rows,
        "coverage_pct": coverage_pct,
        "coverage_exact": True,
        "surfaces": surfaces,
    }


def _unreadable_fts_readiness(reason: str) -> dict[str, object]:
    """The readiness payload for an index this probe could not read.

    Every count is null and coverage is unknown. A ``coverage_pct`` of ``0.0``
    here reported a measured empty index for a database nothing opened, and
    reached ``/api/status`` and the plaintext CLI unaltered through
    ``status_snapshot``'s minimal path, which publishes this dict directly
    rather than through ``FTSReadiness`` (polylogue-20d.17.4 AC3).
    """
    return {
        "indexed_surface": "messages_fts",
        "messages_ready": False,
        "invariant_ready": False,
        "message_indexed_count": None,
        "message_indexable_count": None,
        "coverage_pct": None,
        "coverage_exact": False,
        "inspection_state": "unavailable",
        "unavailable_reason": reason,
        "surfaces": {},
    }


def _collect_fts_readiness_info(dbf: Path, *, exact: bool = False) -> dict[str, object]:
    """Perform one authoritative FTS readiness collection.

    Readiness is always computed from the current output relation and its
    canonical inputs. ``exact`` is retained as a call-compatible parameter;
    no freshness/debt observation can make the result cheaper or certify it.
    This is the registry's collector, so it may outlive the caller's deadline;
    the registry owns that timeout and keeps this one aggregate attempt alive.
    """
    if not dbf.exists():
        archive_index = _archive_index_path_for(dbf)
        if archive_index is not None:
            archive_info = _archive_readiness_info(archive_index, exact=exact)
            if archive_info is not None:
                return archive_info
        return {
            "messages_ready": False,
            "coverage_pct": None,
            "coverage_exact": False,
        }
    try:
        # Readiness reports a skewed or unstamped tier as not-ready data; it must
        # not raise the status surface out of service.
        conn = open_readonly_connection(dbf, validate_schema=False)
        try:
            conn.execute("BEGIN")
            archive_info = _archive_readiness_payload(conn, exact=True)
            if archive_info is not None:
                return archive_info
            return _exact_readiness_payload(fts_invariant_snapshot_sync(conn))
        finally:
            conn.rollback()
            conn.close()
    except sqlite3.Error as exc:
        emit(
            "daemon.fts.readiness_query_failed",
            level=WARNING,
            outcome="degraded",
            reason="readiness_unreadable",
            path=dbf,
            error_type=type(exc).__name__,
            error_detail=str(exc),
        )
        unreadable_reason = f"{type(exc).__name__}: {exc}"
    # The handler records why the read failed; the payload is named once and
    # fabricates nothing, so this site no longer improvises its own
    # unavailable-as-a-value policy.
    return _unreadable_fts_readiness(unreadable_reason)


def _not_measured_readiness(state: str, *, error: str | None = None) -> dict[str, object]:
    """Return a non-positive payload when the current aggregate did not finish."""
    return {
        "indexed_surface": "messages_fts",
        "inspection_state": state,
        "messages_ready": False,
        "invariant_ready": False,
        "message_indexed_count": None,
        "message_indexable_count": None,
        "coverage_pct": None,
        "coverage_exact": False,
        "unavailable_reason": error or f"FTS readiness inspection {state}",
        "surfaces": {},
    }


def fts_readiness_info(dbf: Path, *, exact: bool = False) -> dict[str, object]:
    """Return FTS readiness through one persistent, deadline-bounded collector.

    The unbound route is a fixed set of aggregate COUNT queries. A slow query
    therefore has no row-level progress to checkpoint: callers stop waiting
    at the declared deadline, and a later call observes this registry's same
    in-flight attempt until it produces the actual verdict.
    """
    del exact
    registry = _fts_readiness_registry(dbf)
    snapshot: ComponentSnapshot = registry.collect(names=["fts_readiness"])["fts_readiness"]
    if snapshot.state in {"timed_out", "refreshing", "unavailable", "degraded"}:
        return _not_measured_readiness(snapshot.state, error=snapshot.error)
    current_fingerprint = _fts_readiness_fingerprint(dbf)
    if snapshot.state == "stale" and snapshot.fingerprint != current_fingerprint:
        return _not_measured_readiness("refreshing", error="FTS readiness source changed during refresh")
    if not isinstance(snapshot.value, dict):
        return _not_measured_readiness("unavailable", error="FTS readiness collector returned no payload")
    payload = snapshot.value.get("payload")
    source_fingerprint = snapshot.value.get("source_fingerprint")
    if not isinstance(payload, dict) or not isinstance(source_fingerprint, str):
        return _not_measured_readiness("unavailable", error="FTS readiness collector returned an invalid envelope")
    if source_fingerprint != current_fingerprint:
        registry.request_refresh("fts_readiness")
        return _not_measured_readiness("refreshing", error="FTS readiness source changed during inspection")
    if payload.get("inspection_state") == "unavailable":
        return payload
    return {**payload, "inspection_state": snapshot.state}

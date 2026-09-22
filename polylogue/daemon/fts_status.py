"""FTS readiness projection for daemon status payloads."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from polylogue.core.payload_coercion import row_int as _row_int
from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.logging import WARNING, emit
from polylogue.storage.fts.fts_lifecycle import FtsInvariantSnapshot, FtsSurfaceInvariant, fts_invariant_snapshot_sync
from polylogue.storage.sqlite.connection_profile import open_readonly_connection


class FTSReadiness(BaseModel):
    indexed_surface: str = "messages_fts"
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
        "unavailable_reason": reason,
        "surfaces": {},
    }


def fts_readiness_info(dbf: Path, *, exact: bool = False) -> dict[str, object]:
    """Return FTS readiness for health/status probes.

    Readiness is always computed from the current output relation and its
    canonical inputs.  ``exact`` is retained as a call-compatible parameter;
    no freshness/debt observation can make the result cheaper or certify it.
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

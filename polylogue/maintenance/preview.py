"""Preview surface: staleness inventory by model and scope.

Read-only enumeration of stale or missing derived artifacts in the archive,
grouped by model and typed ``InvalidationReason``. Used by the maintenance
planner (``polylogue ops maintenance preview``) and by operators who want to see
"what will be rebuilt and why" before triggering any mutation.

The inventory is sourced from
:func:`polylogue.storage.derived.derived_status.collect_derived_model_statuses_sync`
plus the archive-debt orphan counts in
:mod:`polylogue.storage.repair`; the preview never mutates the database (a
write-watching SQLite hook used in tests confirms zero writes during a
preview).

Models inventoried:

* ``messages_fts``,
  ``session_profile_rows``, ``session_work_event_inference``,
  ``session_work_event_inference_fts``, phase interval rows,
  ``threads``, ``threads_fts``, ``session_tag_rollups`` — derived read
  models reported by ``collect_derived_model_statuses_sync``.
* ``transcript_embeddings``, ``retrieval_evidence``,
  ``retrieval_inference``, ``retrieval_enrichment`` — retrieval-layer
  read models.
* ``orphaned_messages``, ``empty_sessions``, ``orphaned_attachments`` —
  archive-cleanup scopes (orphan rows only).
* ``message_type_backfill`` — message-type classification backlog.

The inventory is produced at the model granularity reported by the
derived-status collector; mapping back to ``MaintenanceTargetSpec`` is the
planner's job and lives in ``planner.preview_backfill``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from polylogue.core.json import JSONDocument, json_document
from polylogue.maintenance.models import DerivedModelStatus
from polylogue.storage.message_type_backfill import count_unclassified_message_type_sync
from polylogue.storage.repair import (
    count_empty_sessions_sync,
    count_orphaned_attachments_sync,
    count_orphaned_blobs_sync,
    count_orphaned_messages_sync,
)


class InvalidationReason(str, Enum):
    """Typed reason a derived row is considered stale or missing.

    The reasons are not mutually exclusive — a single model may emit
    multiple :class:`StalenessItem` rows with different reasons.
    """

    MISSING = "missing"
    """Source rows exist but no derived row has been materialized yet."""

    STALE = "stale"
    """Derived row exists but is older than its source (mtime/content drift)."""

    ORPHAN = "orphan"
    """Derived row references a parent row that no longer exists."""

    MISSING_PROVENANCE = "missing_provenance"
    """Derived row exists but cannot be linked back to an upstream input."""

    VERSION_MISMATCH = "version_mismatch"
    """Derived row was produced by an older ``materializer_version``."""

    ORPHAN_ARCHIVE_ROW = "orphan_archive_row"
    """Archive row (messages, attachments, content blocks) with no parent."""


_DERIVED_MODEL_SCOPE = "derived"
_RETRIEVAL_MODEL_SCOPE = "retrieval"
_ARCHIVE_CLEANUP_SCOPE = "archive_cleanup"
_BACKFILL_SCOPE = "backfill"

ALL_SCOPES: tuple[str, ...] = (
    _DERIVED_MODEL_SCOPE,
    _RETRIEVAL_MODEL_SCOPE,
    _ARCHIVE_CLEANUP_SCOPE,
    _BACKFILL_SCOPE,
)


@dataclass(frozen=True, slots=True)
class StalenessItem:
    """One (model, reason) staleness count, optionally with sample ids."""

    model: str
    scope: str
    reason: InvalidationReason
    count: int
    source_total: int
    materialized_total: int
    detail: str
    sample_ids: tuple[str, ...] = ()
    truncated: bool = False

    @property
    def fraction(self) -> float:
        """Fraction of source rows considered stale for this reason.

        Returns 0.0 if ``source_total`` is zero. Capped at 1.0.
        """

        if self.source_total <= 0:
            return 0.0
        ratio = self.count / self.source_total
        return ratio if ratio < 1.0 else 1.0

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "model": self.model,
                "scope": self.scope,
                "reason": self.reason.value,
                "count": self.count,
                "source_total": self.source_total,
                "materialized_total": self.materialized_total,
                "fraction": round(self.fraction, 6),
                "detail": self.detail,
                "sample_ids": list(self.sample_ids),
                "truncated": self.truncated,
            }
        )


@dataclass(frozen=True, slots=True)
class StalenessInventory:
    """Complete staleness inventory for one preview call."""

    captured_at: str
    db_path: str
    scopes: tuple[str, ...]
    items: tuple[StalenessItem, ...] = field(default_factory=tuple)

    def by_model(self) -> dict[str, tuple[StalenessItem, ...]]:
        result: dict[str, list[StalenessItem]] = {}
        for item in self.items:
            result.setdefault(item.model, []).append(item)
        return {model: tuple(rows) for model, rows in result.items()}

    def total_stale(self) -> int:
        return sum(item.count for item in self.items)

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "captured_at": self.captured_at,
                "db_path": self.db_path,
                "scopes": list(self.scopes),
                "total_stale": self.total_stale(),
                "items": [item.to_dict() for item in self.items],
            }
        )


# ---------------------------------------------------------------------------
# Inventory builders
# ---------------------------------------------------------------------------


def _model_items(
    status: DerivedModelStatus,
    *,
    scope: str,
) -> list[StalenessItem]:
    """Project a :class:`DerivedModelStatus` into reason-tagged rows.

    Always emits at least one row per (model, possible reason) so empty
    models report ``count=0`` explicitly rather than absence.
    """

    source_total = status.source_documents or status.source_rows or status.materialized_documents
    materialized_total = status.materialized_rows or status.materialized_documents

    items: list[StalenessItem] = []

    pending = max(0, int(status.pending_rows or 0) + int(status.pending_documents or 0))
    items.append(
        StalenessItem(
            model=status.name,
            scope=scope,
            reason=InvalidationReason.MISSING,
            count=pending,
            source_total=source_total,
            materialized_total=materialized_total,
            detail=(f"{pending:,} unmaterialized rows/documents" if pending > 0 else "All source rows materialized"),
        )
    )

    stale = max(0, int(status.stale_rows or 0))
    items.append(
        StalenessItem(
            model=status.name,
            scope=scope,
            reason=InvalidationReason.STALE,
            count=stale,
            source_total=source_total,
            materialized_total=materialized_total,
            detail=(f"{stale:,} stale rows" if stale > 0 else "No stale rows"),
        )
    )

    orphan = max(0, int(status.orphan_rows or 0))
    items.append(
        StalenessItem(
            model=status.name,
            scope=scope,
            reason=InvalidationReason.ORPHAN,
            count=orphan,
            source_total=source_total,
            materialized_total=materialized_total,
            detail=(f"{orphan:,} orphan rows" if orphan > 0 else "No orphan rows"),
        )
    )

    missing_provenance = max(0, int(status.missing_provenance_rows or 0))
    if missing_provenance > 0:
        items.append(
            StalenessItem(
                model=status.name,
                scope=scope,
                reason=InvalidationReason.MISSING_PROVENANCE,
                count=missing_provenance,
                source_total=source_total,
                materialized_total=materialized_total,
                detail=f"{missing_provenance:,} rows missing provenance",
            )
        )

    if status.matches_version is False and materialized_total > 0:
        items.append(
            StalenessItem(
                model=status.name,
                scope=scope,
                reason=InvalidationReason.VERSION_MISMATCH,
                count=materialized_total,
                source_total=source_total,
                materialized_total=materialized_total,
                detail=(
                    f"materializer_version={status.materializer_version!r}; "
                    f"all {materialized_total:,} rows would be rebuilt"
                ),
            )
        )

    return items


_DERIVED_MODEL_NAMES: frozenset[str] = frozenset(
    {
        "messages_fts",
        "session_profile_rows",
        "session_work_event_inference",
        "session_work_event_inference_fts",
        "session_phase_inference",
        "threads",
        "threads_fts",
        "session_tag_rollups",
    }
)

_RETRIEVAL_MODEL_NAMES: frozenset[str] = frozenset(
    {
        "transcript_embeddings",
        "retrieval_evidence",
        "retrieval_inference",
        "retrieval_enrichment",
    }
)


def _archive_cleanup_items(
    conn: sqlite3.Connection,
    *,
    db_path: Path | str | None,
    include_expensive: bool,
) -> list[StalenessItem]:
    """Orphan-row counts for archive-cleanup scopes (read-only)."""

    if not include_expensive:
        return [
            StalenessItem(
                model=model,
                scope=_ARCHIVE_CLEANUP_SCOPE,
                reason=InvalidationReason.ORPHAN_ARCHIVE_ROW,
                count=0,
                source_total=0,
                materialized_total=0,
                detail="Exact archive-cleanup count skipped by shallow preview",
                truncated=True,
            )
            for model in (
                "orphaned_messages",
                "empty_sessions",
                "orphaned_attachments",
                "orphaned_blobs",
            )
        ]

    orphan_messages = count_orphaned_messages_sync(conn)
    empty_sessions = count_empty_sessions_sync(conn)
    orphan_attachments = count_orphaned_attachments_sync(conn)

    rows: list[tuple[str, int, str]] = [
        (
            "orphaned_messages",
            orphan_messages,
            "messages referencing missing sessions",
        ),
        (
            "empty_sessions",
            empty_sessions,
            "sessions with no messages",
        ),
        (
            "orphaned_attachments",
            orphan_attachments,
            "attachment refs without parent rows",
        ),
    ]

    if include_expensive:
        orphan_blobs = count_orphaned_blobs_sync(conn, db_path=db_path)
        rows.append(
            (
                "orphaned_blobs",
                orphan_blobs,
                "content-addressed blob files with no source blob reference",
            )
        )

    items: list[StalenessItem] = []
    for model, count, label in rows:
        items.append(
            StalenessItem(
                model=model,
                scope=_ARCHIVE_CLEANUP_SCOPE,
                reason=InvalidationReason.ORPHAN_ARCHIVE_ROW,
                count=count,
                source_total=count,
                materialized_total=count,
                detail=(f"{count:,} {label}" if count > 0 else f"No {label}"),
            )
        )
    return items


def _backfill_items(conn: sqlite3.Connection) -> list[StalenessItem]:
    """Pending backfills (message_type classification)."""

    unclassified = count_unclassified_message_type_sync(conn)
    total_messages = int(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0])
    return [
        StalenessItem(
            model="message_type_backfill",
            scope=_BACKFILL_SCOPE,
            reason=InvalidationReason.MISSING,
            count=unclassified,
            source_total=total_messages,
            materialized_total=max(0, total_messages - unclassified),
            detail=(
                f"{unclassified:,} of {total_messages:,} messages need context/protocol classification"
                if unclassified > 0
                else "All messages have message_type classified"
            ),
        )
    ]


def _coerce_scopes(scopes: Iterable[str] | None) -> tuple[str, ...]:
    if not scopes:
        return ALL_SCOPES
    requested = tuple(dict.fromkeys(scopes))
    unknown = [s for s in requested if s not in ALL_SCOPES]
    if unknown:
        raise ValueError(f"Unknown preview scopes: {unknown}; valid: {list(ALL_SCOPES)}")
    return requested


def staleness_inventory(
    db_path: Path | str | sqlite3.Connection | None = None,
    *,
    scopes: Iterable[str] | None = None,
    verify_full: bool = True,
    sample_limit: int = 0,
) -> StalenessInventory:
    """Enumerate stale/missing derived artifacts by model and scope.

    Read-only — performs no mutations. Returns one row per
    (model, :class:`InvalidationReason`) pair. Models with nothing stale
    still produce explicit ``count=0`` rows for ``MISSING``, ``STALE``,
    and ``ORPHAN`` so consumers can render a complete inventory without
    distinguishing "absent" from "zero".

    Parameters
    ----------
    db_path:
        Path, open connection, or ``None`` to use the configured archive.
    scopes:
        Subset of :data:`ALL_SCOPES` to inventory. Defaults to all.
    verify_full:
        Whether to force full freshness verification in the derived-status
        collector. Defaults to ``True`` for accurate counts.
    sample_limit:
        Reserved for future per-reason id sampling. Currently unused; the
        inventory does not surface ids in this PR. (Future: surface up to
        N ids per stale (model, reason) when the planner needs them.)
    """

    from contextlib import closing, nullcontext

    from polylogue.paths import active_index_db_path
    from polylogue.storage.derived.derived_status import collect_derived_model_statuses_sync
    from polylogue.storage.sqlite.connection_profile import open_readonly_connection

    selected_scopes = _coerce_scopes(scopes)
    _ = sample_limit  # reserved; see docstring.

    captured_at = datetime.now(timezone.utc).isoformat()

    # The archive is the split-file store; reads target
    # ``index.db`` (sessions/messages/blocks tree) directly. A pre-opened
    # connection is used as-is; otherwise the caller's path — or the active
    # ``index.db`` — is opened read-only.
    from contextlib import AbstractContextManager

    connection_manager: AbstractContextManager[sqlite3.Connection]
    if isinstance(db_path, sqlite3.Connection):
        connection_manager = nullcontext(db_path)
    else:
        resolved = Path(db_path) if db_path is not None else active_index_db_path()
        if not resolved.exists():
            # No archive `index.db` yet (fresh archive before first ingest).
            # There is nothing to inventory; report an empty result rather
            # than failing to open a non-existent file.
            return StalenessInventory(
                captured_at=captured_at,
                db_path=str(resolved),
                scopes=selected_scopes,
                items=(),
            )
        connection_manager = closing(open_readonly_connection(resolved))

    with connection_manager as conn:
        if _DERIVED_MODEL_SCOPE in selected_scopes or _RETRIEVAL_MODEL_SCOPE in selected_scopes:
            statuses = collect_derived_model_statuses_sync(conn, verify_full=verify_full)
        else:
            statuses = {}
        resolved_path = _resolve_db_path(conn)

        items: list[StalenessItem] = []

        if _DERIVED_MODEL_SCOPE in selected_scopes:
            for name, status in statuses.items():
                if name in _DERIVED_MODEL_NAMES:
                    items.extend(_model_items(status, scope=_DERIVED_MODEL_SCOPE))

        if _RETRIEVAL_MODEL_SCOPE in selected_scopes:
            for name, status in statuses.items():
                if name in _RETRIEVAL_MODEL_NAMES:
                    items.extend(_model_items(status, scope=_RETRIEVAL_MODEL_SCOPE))

        if _ARCHIVE_CLEANUP_SCOPE in selected_scopes:
            items.extend(_archive_cleanup_items(conn, db_path=resolved_path, include_expensive=verify_full))

        if _BACKFILL_SCOPE in selected_scopes:
            items.extend(_backfill_items(conn))

    return StalenessInventory(
        captured_at=captured_at,
        db_path=resolved_path,
        scopes=selected_scopes,
        items=tuple(items),
    )


def _resolve_db_path(conn: sqlite3.Connection) -> str:
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
    except sqlite3.Error:
        return ":memory:"
    if row is None:
        return ":memory:"
    # PRAGMA database_list rows: (seq, name, file)
    file_part = row[2] if len(row) > 2 else ""
    return str(file_part) if file_part else ":memory:"


__all__ = [
    "ALL_SCOPES",
    "InvalidationReason",
    "StalenessInventory",
    "StalenessItem",
    "staleness_inventory",
]

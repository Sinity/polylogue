"""Read-only accounting for parent sessions asserted by indexed topology.

The topology table is not an authority for acquisition: it only tells us what
the parser asserted.  This census joins those assertions to the durable source
ledger and to the candidate index using the complete public identity
``(origin, native_id)``.  A native id alone is deliberately never sufficient;
the same id is valid in more than one origin.

The census also walks every retained raw for the two historical parent origins.
An unmaterialized raw must have a durable, source-grounded disposition.  This
is the anti-vacuity boundary for the old "parsed but never indexed" symptom:
the check cannot become green merely because a derived candidate list forgot a
row.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.json import JSONDocument, json_document
from polylogue.storage.introspection import table_exists

PARENT_ORIGINS: tuple[str, ...] = ("claude-code-session", "codex-session")

_MATERIALIZED = "materialized"
_MATERIALIZED_UNRESOLVED = "materialized_unresolved"
_AVAILABLE_UNMATERIALIZED = "source_available_unmaterialized"
_UNAVAILABLE = "source_unavailable"
_NOT_ACQUIRED = "not_acquired"


@dataclass(frozen=True, slots=True)
class ParentReferenceEvidence:
    """One unique asserted parent identity and its source/index evidence."""

    origin: str
    native_id: str
    reference_count: int
    source_raw_ids: tuple[str, ...]
    source_bytes: int
    source_paths: tuple[str, ...]
    indexed_session_ids: tuple[str, ...]
    resolved_reference_count: int
    disposition: str
    reason: str

    @property
    def source_available(self) -> bool:
        return bool(self.source_raw_ids) and self.disposition != _UNAVAILABLE

    def to_json(self) -> JSONDocument:
        return json_document(
            {
                "origin": self.origin,
                "native_id": self.native_id,
                "reference_count": self.reference_count,
                "source_raw_ids": list(self.source_raw_ids),
                "source_bytes": self.source_bytes,
                "source_paths": list(self.source_paths),
                "indexed_session_ids": list(self.indexed_session_ids),
                "resolved_reference_count": self.resolved_reference_count,
                "disposition": self.disposition,
                "reason": self.reason,
            }
        )


@dataclass(frozen=True, slots=True)
class RawParentDisposition:
    """Disposition for one retained parent-origin raw without an index row."""

    raw_id: str
    origin: str
    native_id: str | None
    source_path: str
    blob_size: int
    disposition: str
    reason: str

    def to_json(self) -> JSONDocument:
        return json_document(
            {
                "raw_id": self.raw_id,
                "origin": self.origin,
                "native_id": self.native_id,
                "source_path": self.source_path,
                "blob_size": self.blob_size,
                "disposition": self.disposition,
                "reason": self.reason,
            }
        )


@dataclass(frozen=True, slots=True)
class ParentSessionAccountingReport:
    """Conservation and identity evidence for a candidate parent cohort."""

    available: bool
    reason: str | None
    reference_total: int
    unique_parent_total: int
    resolved_reference_total: int
    source_available_total: int
    source_unavailable_total: int
    not_acquired_total: int
    materialized_parent_total: int
    unresolved_reference_total: int
    available_unmaterialized_total: int
    raw_total: int
    frontier_total: int
    frontier_bytes: int
    raw_materialized_total: int
    raw_explained_total: int
    raw_unexplained_total: int
    raw_disposition_counts: dict[str, int]
    references: tuple[ParentReferenceEvidence, ...] = ()
    raw_dispositions: tuple[RawParentDisposition, ...] = ()
    untyped_denominator: int = 0
    untyped_total: int = 0
    origins: tuple[str, ...] = PARENT_ORIGINS

    @property
    def blocking_count(self) -> int:
        # ``untyped_total`` is the separately reported raw denominator's
        # unexplained subset, so do not count that alias twice.
        return self.unresolved_reference_total + self.available_unmaterialized_total + self.raw_unexplained_total

    @property
    def warning_count(self) -> int:
        return self.source_unavailable_total + self.not_acquired_total

    def to_json(self) -> JSONDocument:
        return json_document(
            {
                "available": self.available,
                "reason": self.reason,
                "reference_total": self.reference_total,
                "unique_parent_total": self.unique_parent_total,
                "resolved_reference_total": self.resolved_reference_total,
                "source_available_total": self.source_available_total,
                "source_unavailable_total": self.source_unavailable_total,
                "not_acquired_total": self.not_acquired_total,
                "materialized_parent_total": self.materialized_parent_total,
                "unresolved_reference_total": self.unresolved_reference_total,
                "available_unmaterialized_total": self.available_unmaterialized_total,
                "raw_total": self.raw_total,
                "frontier_total": self.frontier_total,
                "frontier_bytes": self.frontier_bytes,
                "raw_materialized_total": self.raw_materialized_total,
                "raw_explained_total": self.raw_explained_total,
                "raw_unexplained_total": self.raw_unexplained_total,
                "raw_disposition_counts": dict(sorted(self.raw_disposition_counts.items())),
                "untyped_denominator": self.untyped_denominator,
                "untyped_total": self.untyped_total,
                "origins": list(self.origins),
                "references": [entry.to_json() for entry in self.references],
                "raw_dispositions": [entry.to_json() for entry in self.raw_dispositions],
            }
        )

    def summary(self) -> str:
        if not self.available:
            return f"parent-session accounting unavailable: {self.reason or 'unknown reason'}"
        parts = [
            f"{self.reference_total:,} parent reference(s), {self.unique_parent_total:,} unique parent(s)",
            f"materialized={self.materialized_parent_total:,}",
            f"unresolved-references={self.unresolved_reference_total:,}",
            f"source-available={self.source_available_total:,}",
            f"available-unmaterialized={self.available_unmaterialized_total:,}",
            f"raw={self.raw_total:,}, raw-unexplained={self.raw_unexplained_total:,}",
        ]
        if self.untyped_denominator:
            parts.append(f"untyped-cohort={self.untyped_total:,}/{self.untyped_denominator:,}")
        return "; ".join(parts)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not table_exists(conn, table):
        return set()
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _source_path_available(archive_root: Path | None, source_path: str) -> bool:
    if not source_path:
        return False
    path = Path(source_path)
    if archive_root is not None and not path.is_absolute():
        path = archive_root / path
    return path.exists()


def _raw_disposition(row: sqlite3.Row, *, source_available: bool) -> tuple[str, str]:
    """Apply the durable refusal ladder to one unmaterialized raw."""
    parse_error = row["parse_error"]
    validation_status = row["validation_status"]
    parsed_at_ms = row["parsed_at_ms"]
    revision_authority = row["revision_authority"]
    artifact_kind = row["artifact_kind"]
    parse_as_session = row["parse_as_session"]
    if parse_error:
        return ("parse_failure", "raw_sessions.parse_error records a parser refusal")
    if validation_status == "failed":
        return ("validation_rejected", "raw_sessions.validation_status=failed records schema refusal")
    if parse_as_session == 0 and artifact_kind and artifact_kind != "unknown":
        return ("non_session_artifact", f"raw_artifacts declares {artifact_kind}")
    if parsed_at_ms is None:
        return ("pending", "raw is acquired but has not reached the parse boundary")
    if revision_authority == "quarantined":
        return ("revision_authority_quarantined", "source reconciliation has not granted write authority")
    if not source_available:
        return ("source_unavailable", "raw payload and source path are unavailable")
    return ("untyped_unmaterialized", "parsed raw has no durable refusal and no indexed session")


def audit_parent_session_accounting(
    source: sqlite3.Connection,
    index: sqlite3.Connection,
    *,
    archive_root: Path | None = None,
    origins: tuple[str, ...] = PARENT_ORIGINS,
    untyped_denominator: int = 0,
    untyped_total: int = 0,
) -> ParentSessionAccountingReport:
    """Run the parent identity/conservation census on read-only connections.

    ``source`` and ``index`` may be independently opened read-only handles;
    this is intentional so the routine cannot accidentally become a writer or
    make the rebuildable index authoritative for source evidence.
    """
    if not table_exists(index, "session_links"):
        return ParentSessionAccountingReport(
            available=False,
            reason="index.db.session_links is absent",
            reference_total=0,
            unique_parent_total=0,
            resolved_reference_total=0,
            source_available_total=0,
            source_unavailable_total=0,
            not_acquired_total=0,
            materialized_parent_total=0,
            unresolved_reference_total=0,
            available_unmaterialized_total=0,
            raw_total=0,
            frontier_total=0,
            frontier_bytes=0,
            raw_materialized_total=0,
            raw_explained_total=0,
            raw_unexplained_total=0,
            raw_disposition_counts={},
            untyped_denominator=untyped_denominator,
            untyped_total=untyped_total,
            origins=origins,
        )
    source_columns = _table_columns(source, "raw_sessions")
    if not source_columns:
        return ParentSessionAccountingReport(
            available=False,
            reason="source.db.raw_sessions is absent",
            reference_total=0,
            unique_parent_total=0,
            resolved_reference_total=0,
            source_available_total=0,
            source_unavailable_total=0,
            not_acquired_total=0,
            materialized_parent_total=0,
            unresolved_reference_total=0,
            available_unmaterialized_total=0,
            raw_total=0,
            frontier_total=0,
            frontier_bytes=0,
            raw_materialized_total=0,
            raw_explained_total=0,
            raw_unexplained_total=0,
            raw_disposition_counts={},
            untyped_denominator=untyped_denominator,
            untyped_total=untyped_total,
            origins=origins,
        )

    index.row_factory = sqlite3.Row
    source.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in origins)
    links = index.execute(
        f"""
        SELECT dst_origin, dst_native_id, COUNT(*) AS reference_count,
               SUM(resolved_dst_session_id IS NOT NULL) AS resolved_count
        FROM session_links
        WHERE dst_origin IN ({placeholders})
        GROUP BY dst_origin, dst_native_id
        ORDER BY dst_origin, dst_native_id
        """,
        origins,
    ).fetchall()

    refs: list[ParentReferenceEvidence] = []
    for link in links:
        origin = str(link["dst_origin"])
        native_id = str(link["dst_native_id"])
        raws = source.execute(
            """
            SELECT r.raw_id, r.source_path, r.blob_size, r.blob_hash,
                   r.parsed_at_ms, r.parse_error, r.validation_status,
                   r.revision_authority,
                   a.artifact_kind, a.parse_as_session
            FROM raw_sessions AS r
            LEFT JOIN raw_artifacts AS a ON a.raw_id = r.raw_id
            WHERE r.origin = ? AND (r.native_id = ? OR r.logical_source_key = ?)
            ORDER BY r.raw_id
            """,
            (origin, native_id, f"{origin}:{native_id}"),
        ).fetchall()
        raw_ids = tuple(str(row["raw_id"]) for row in raws)
        source_paths = tuple(sorted({str(row["source_path"]) for row in raws}))
        source_bytes = sum(int(row["blob_size"] or 0) for row in raws)
        source_present = bool(raws)
        source_available = False
        for row in raws:
            blob_present = bool(row["blob_hash"])
            if table_exists(source, "blob_refs") and blob_present:
                blob_present = (
                    source.execute(
                        "SELECT 1 FROM blob_refs WHERE blob_hash = ? LIMIT 1", (row["blob_hash"],)
                    ).fetchone()
                    is not None
                )
            source_available = (
                source_available or blob_present or _source_path_available(archive_root, str(row["source_path"]))
            )
        sessions = index.execute(
            "SELECT session_id FROM sessions WHERE origin = ? AND native_id = ? ORDER BY session_id",
            (origin, native_id),
        ).fetchall()
        session_ids = tuple(str(row[0]) for row in sessions)
        resolved_count = int(link["resolved_count"] or 0)
        reference_count = int(link["reference_count"])
        if session_ids and resolved_count == reference_count:
            disposition, reason = _MATERIALIZED, "exact origin/native identity is present in index.db.sessions"
        elif session_ids:
            unresolved = reference_count - resolved_count
            disposition, reason = (
                _MATERIALIZED_UNRESOLVED,
                f"candidate session exists but {unresolved} parent reference(s) remain unresolved",
            )
        elif not source_present:
            disposition, reason = _NOT_ACQUIRED, "no retained raw has this exact source identity"
        elif not source_available:
            disposition, reason = _UNAVAILABLE, "retained identity exists but neither bytes nor source path survives"
        else:
            disposition, reason = (
                _AVAILABLE_UNMATERIALIZED,
                "retained source identity is available but no candidate session has its exact identity",
            )
        refs.append(
            ParentReferenceEvidence(
                origin=origin,
                native_id=native_id,
                reference_count=reference_count,
                source_raw_ids=raw_ids,
                source_bytes=source_bytes,
                source_paths=source_paths,
                indexed_session_ids=session_ids,
                resolved_reference_count=resolved_count,
                disposition=disposition,
                reason=reason,
            )
        )

    raw_rows = source.execute(
        f"""
        SELECT r.raw_id, r.origin, r.native_id, r.source_path, r.blob_size,
               r.blob_hash, r.parsed_at_ms, r.parse_error, r.validation_status,
               r.revision_authority,
               a.artifact_kind, a.parse_as_session
        FROM raw_sessions AS r
        LEFT JOIN raw_artifacts AS a ON a.raw_id = r.raw_id
        WHERE r.origin IN ({placeholders})
        ORDER BY r.origin, r.raw_id
        """,
        origins,
    ).fetchall()
    dispositions: list[RawParentDisposition] = []
    raw_counts: dict[str, int] = {}
    raw_materialized = 0
    for row in raw_rows:
        materialized = index.execute(
            """
            SELECT 1 FROM sessions
            WHERE raw_id = ? OR (origin = ? AND native_id = ?)
            LIMIT 1
            """,
            (row["raw_id"], row["origin"], row["native_id"]),
        ).fetchone()
        if materialized is not None:
            raw_materialized += 1
            continue
        blob_present = bool(row["blob_hash"])
        if table_exists(source, "blob_refs") and blob_present:
            blob_present = (
                source.execute("SELECT 1 FROM blob_refs WHERE blob_hash = ? LIMIT 1", (row["blob_hash"],)).fetchone()
                is not None
            )
        disposition, reason = _raw_disposition(
            row,
            source_available=blob_present or _source_path_available(archive_root, str(row["source_path"])),
        )
        raw_counts[disposition] = raw_counts.get(disposition, 0) + 1
        dispositions.append(
            RawParentDisposition(
                raw_id=str(row["raw_id"]),
                origin=str(row["origin"]),
                native_id=str(row["native_id"]) if row["native_id"] is not None else None,
                source_path=str(row["source_path"]),
                blob_size=int(row["blob_size"] or 0),
                disposition=disposition,
                reason=reason,
            )
        )

    return ParentSessionAccountingReport(
        available=True,
        reason=None,
        reference_total=sum(int(row["reference_count"]) for row in links),
        unique_parent_total=len(refs),
        resolved_reference_total=sum(entry.resolved_reference_count for entry in refs),
        source_available_total=sum(
            entry.disposition in {_AVAILABLE_UNMATERIALIZED, _MATERIALIZED, _MATERIALIZED_UNRESOLVED} for entry in refs
        ),
        source_unavailable_total=sum(entry.disposition == _UNAVAILABLE for entry in refs),
        not_acquired_total=sum(entry.disposition == _NOT_ACQUIRED for entry in refs),
        materialized_parent_total=sum(entry.disposition == _MATERIALIZED for entry in refs),
        unresolved_reference_total=sum(
            max(0, entry.reference_count - entry.resolved_reference_count)
            if entry.disposition == _MATERIALIZED_UNRESOLVED
            else 0
            for entry in refs
        ),
        available_unmaterialized_total=sum(entry.disposition == _AVAILABLE_UNMATERIALIZED for entry in refs),
        raw_total=len(raw_rows),
        frontier_total=len(raw_rows),
        frontier_bytes=sum(int(row["blob_size"] or 0) for row in raw_rows),
        raw_materialized_total=raw_materialized,
        raw_explained_total=sum(raw_counts.values()),
        raw_unexplained_total=raw_counts.get("untyped_unmaterialized", 0),
        raw_disposition_counts=raw_counts,
        references=tuple(refs),
        raw_dispositions=tuple(dispositions),
        # Keep the untyped-source denominator separate from parent references:
        # it is the complete retained parent-origin raw population, not a
        # historical constant (the old nine-row observation is only a witness).
        untyped_denominator=len(raw_rows),
        untyped_total=(untyped_total if untyped_total else raw_counts.get("untyped_unmaterialized", 0)),
        origins=origins,
    )


__all__ = [
    "PARENT_ORIGINS",
    "ParentReferenceEvidence",
    "RawParentDisposition",
    "ParentSessionAccountingReport",
    "audit_parent_session_accounting",
]

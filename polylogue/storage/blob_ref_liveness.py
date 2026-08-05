"""Set-based liveness classification for source-tier ``blob_refs`` rows.

The source tier stores attachment blob refs against the parent ``raw_id``.
That is a source acquisition reference, not an index-tier ``attachment_refs``
edge. The canonical mapping is shared with blob GC so reconciliation and GC
prove liveness with the same referent joins.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

from polylogue.storage.blob_gc import BLOB_REF_LIVENESS_JOIN
from polylogue.storage.hook_payload_ref_reconciliation import _create_match_stage
from polylogue.storage.introspection import table_exists


class BlobRefLivenessError(RuntimeError):
    """Raised when the source-tier reference shape cannot be proven safe."""


@dataclass(frozen=True, slots=True)
class BlobRefLivenessCandidate:
    """One blob-ref row whose actual source-tier referent is absent."""

    blob_hash: str
    ref_type: str
    ref_id: str
    source_path: str | None
    size_bytes: int
    acquired_at_ms: int
    referent_table: str
    referent_column: str

    def to_dict(self) -> dict[str, object]:
        return {
            "blob_hash": self.blob_hash,
            "ref_type": self.ref_type,
            "ref_id": self.ref_id,
            "source_path": self.source_path,
            "size_bytes": self.size_bytes,
            "acquired_at_ms": self.acquired_at_ms,
            "referent_table": self.referent_table,
            "referent_column": self.referent_column,
        }


@dataclass(frozen=True, slots=True)
class BlobRefLivenessClassification:
    """Complete read-only classification of source-tier blob references."""

    scanned_count: int
    ref_type_counts: dict[str, int]
    orphaned_by_ref_type: dict[str, int]
    ref_type_joins: tuple[tuple[str, str, str], ...]
    unknown_ref_types: tuple[str, ...]
    unavailable_ref_types: tuple[str, ...]
    rekeyable_hook_payload_count: int
    candidates: tuple[BlobRefLivenessCandidate, ...]
    candidate_count: int | None = None

    @property
    def orphaned_count(self) -> int:
        return len(self.candidates) if self.candidate_count is None else self.candidate_count

    @property
    def safe_to_apply(self) -> bool:
        return not self.unknown_ref_types and not self.unavailable_ref_types and not self.rekeyable_hook_payload_count

    def to_dict(self, *, include_candidates: bool = False, sample_limit: int = 30) -> dict[str, object]:
        payload: dict[str, object] = {
            "scanned_count": self.scanned_count,
            "ref_type_counts": dict(self.ref_type_counts),
            "orphaned_count": self.orphaned_count,
            "orphaned_by_ref_type": dict(self.orphaned_by_ref_type),
            "ref_type_joins": [
                {"ref_type": ref_type, "referent_table": table, "referent_column": column}
                for ref_type, table, column in self.ref_type_joins
            ],
            "unknown_ref_types": list(self.unknown_ref_types),
            "unavailable_ref_types": list(self.unavailable_ref_types),
            "rekeyable_hook_payload_count": self.rekeyable_hook_payload_count,
            "safe_to_apply": self.safe_to_apply,
        }
        if include_candidates:
            payload["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        else:
            payload["samples"] = [candidate.to_dict() for candidate in self.candidates[: max(0, sample_limit)]]
        return payload


@dataclass(frozen=True, slots=True)
class BlobRefLivenessStagedPlan:
    """SQLite-backed candidate plan whose rows never need Python duplication."""

    classification: BlobRefLivenessClassification
    candidate_table: str

    def candidates(self, conn: sqlite3.Connection) -> Iterator[BlobRefLivenessCandidate]:
        return (
            BlobRefLivenessCandidate(
                blob_hash=bytes(row[0]).hex(),
                ref_type=str(row[1]),
                ref_id=str(row[2]),
                source_path=str(row[3]) if row[3] is not None else None,
                size_bytes=int(row[4]),
                acquired_at_ms=int(row[5]),
                referent_table=str(row[6]),
                referent_column=str(row[7]),
            )
            for row in conn.execute(
                f"""
                SELECT blob_hash, ref_type, ref_id, source_path, size_bytes,
                       acquired_at_ms, referent_table, referent_column
                FROM {self.candidate_table}
                ORDER BY ref_type, ref_id, blob_hash
                """
            )
        )


def _blob_ref_columns(conn: sqlite3.Connection) -> set[str]:
    return {str(row[1]) for row in conn.execute("PRAGMA table_info(blob_refs)")}


def _referent_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _candidate_from_row(row: sqlite3.Row | tuple[object, ...]) -> BlobRefLivenessCandidate:
    return BlobRefLivenessCandidate(
        blob_hash=cast(bytes, row[0]).hex(),
        ref_type=str(row[1]),
        ref_id=str(row[2]),
        source_path=str(row[3]) if row[3] is not None else None,
        size_bytes=int(cast(int, row[4])),
        acquired_at_ms=int(cast(int, row[5])),
        referent_table=str(row[6]),
        referent_column=str(row[7]),
    )


def stage_blob_ref_liveness(conn: sqlite3.Connection, *, sample_limit: int = 30) -> BlobRefLivenessStagedPlan:
    """Classify every ``blob_refs`` row using its mapped referent LEFT JOIN.

    Candidate rows are staged in SQLite. This is the apply path's bounded
    representation: receipts and deletes can stream from the temp table rather
    than materializing the same population in several Python lists.
    """

    if not table_exists(conn, "blob_refs"):
        return BlobRefLivenessStagedPlan(
            BlobRefLivenessClassification(0, {}, {}, (), (), (), 0, (), 0),
            "blob_ref_liveness_candidates",
        )
    columns = _blob_ref_columns(conn)
    required = {"blob_hash", "ref_id", "ref_type", "source_path", "size_bytes", "acquired_at_ms"}
    missing = sorted(required - columns)
    if missing:
        raise BlobRefLivenessError(f"blob_refs is missing required columns: {', '.join(missing)}")

    ref_type_counts = {
        str(row[0]): int(row[1])
        for row in conn.execute("SELECT ref_type, COUNT(*) FROM blob_refs GROUP BY ref_type ORDER BY ref_type")
    }
    known = {ref_type: (table, column) for ref_type, table, column in BLOB_REF_LIVENESS_JOIN}
    unknown = tuple(sorted(set(ref_type_counts) - set(known)))
    unavailable: list[str] = []
    branches: list[str] = []
    params: list[str] = []
    ref_type_joins: list[tuple[str, str, str]] = []
    for ref_type, (table, column) in known.items():
        if not ref_type_counts.get(ref_type):
            continue
        if not table_exists(conn, table) or column not in _referent_columns(conn, table):
            unavailable.append(ref_type)
            continue
        branches.append(
            f"""
            SELECT b.blob_hash, b.ref_type, b.ref_id, b.source_path, b.size_bytes,
                   b.acquired_at_ms, '{table}' AS referent_table, '{column}' AS referent_column
            FROM blob_refs AS b
            LEFT JOIN {table} AS r ON r.{column} = b.ref_id
            WHERE b.ref_type = ? AND r.{column} IS NULL
            """
        )
        params.append(ref_type)
        ref_type_joins.append((ref_type, table, column))

    rekeyable_hook_payload_count = 0
    if table_exists(conn, "raw_hook_events"):
        _scanned_hook_refs, rekeyable_hook_payload_count, _matched_bytes, ambiguous_count = _create_match_stage(conn)
        rekeyable_hook_payload_count += ambiguous_count
    else:
        conn.execute("DROP TABLE IF EXISTS temp.hook_payload_ref_reconciliation_matches")
        conn.execute(
            """
            CREATE TEMP TABLE hook_payload_ref_reconciliation_matches (
                blob_hash BLOB NOT NULL,
                orphaned_ref_id TEXT NOT NULL,
                source_path TEXT,
                size_bytes INTEGER NOT NULL,
                acquired_at_ms INTEGER NOT NULL,
                hook_event_id TEXT NOT NULL,
                PRIMARY KEY (blob_hash, orphaned_ref_id)
            ) STRICT
            """
        )
        conn.execute(
            """
            CREATE TEMP TABLE hook_payload_ref_reconciliation_ambiguous (
                blob_hash BLOB NOT NULL,
                orphaned_ref_id TEXT NOT NULL,
                PRIMARY KEY (blob_hash, orphaned_ref_id)
            ) STRICT
            """
        )

    candidate_table = "blob_ref_liveness_candidates"
    conn.execute(f"DROP TABLE IF EXISTS temp.{candidate_table}")
    conn.execute(
        f"""
        CREATE TEMP TABLE {candidate_table} (
            blob_hash BLOB NOT NULL,
            ref_type TEXT NOT NULL,
            ref_id TEXT NOT NULL,
            source_path TEXT,
            size_bytes INTEGER NOT NULL,
            acquired_at_ms INTEGER NOT NULL,
            referent_table TEXT NOT NULL,
            referent_column TEXT NOT NULL,
            PRIMARY KEY (blob_hash, ref_type, ref_id)
        ) STRICT
        """
    )
    if branches:
        query = " UNION ALL ".join(branches)
        conn.execute(
            f"""
            INSERT INTO {candidate_table} (
                blob_hash, ref_type, ref_id, source_path, size_bytes, acquired_at_ms,
                referent_table, referent_column
            )
            SELECT candidate.blob_hash, candidate.ref_type, candidate.ref_id,
                   candidate.source_path, candidate.size_bytes, candidate.acquired_at_ms,
                   candidate.referent_table, candidate.referent_column
            FROM ({query}) AS candidate
            WHERE NOT EXISTS (
                SELECT 1
                FROM temp.hook_payload_ref_reconciliation_matches AS hook_match
                WHERE candidate.ref_type = 'raw_payload'
                  AND hook_match.blob_hash = candidate.blob_hash
                  AND hook_match.orphaned_ref_id = candidate.ref_id
            )
              AND NOT EXISTS (
                SELECT 1
                FROM temp.hook_payload_ref_reconciliation_ambiguous AS ambiguous_hook
                WHERE candidate.ref_type = 'raw_payload'
                  AND ambiguous_hook.blob_hash = candidate.blob_hash
                  AND ambiguous_hook.orphaned_ref_id = candidate.ref_id
            )
            """,
            tuple(params),
        )
    candidate_count = int(conn.execute(f"SELECT COUNT(*) FROM {candidate_table}").fetchone()[0])
    orphaned_by_ref_type = {
        str(row[0]): int(row[1])
        for row in conn.execute(f"SELECT ref_type, COUNT(*) FROM {candidate_table} GROUP BY ref_type ORDER BY ref_type")
    }
    samples = tuple(
        _candidate_from_row(row)
        for row in conn.execute(
            f"""
            SELECT blob_hash, ref_type, ref_id, source_path, size_bytes,
                   acquired_at_ms, referent_table, referent_column
            FROM {candidate_table}
            ORDER BY ref_type, ref_id, blob_hash
            LIMIT ?
            """,
            (max(0, sample_limit),),
        )
    )
    classification = BlobRefLivenessClassification(
        scanned_count=sum(ref_type_counts.values()),
        ref_type_counts=ref_type_counts,
        orphaned_by_ref_type=orphaned_by_ref_type,
        ref_type_joins=tuple(ref_type_joins),
        unknown_ref_types=unknown,
        unavailable_ref_types=tuple(sorted(unavailable)),
        rekeyable_hook_payload_count=rekeyable_hook_payload_count,
        candidates=samples,
        candidate_count=candidate_count,
    )
    return BlobRefLivenessStagedPlan(classification, candidate_table)


def classify_blob_ref_liveness(conn: sqlite3.Connection) -> BlobRefLivenessClassification:
    """Return the complete candidate projection for read-only callers."""

    staged = stage_blob_ref_liveness(conn, sample_limit=0)
    candidates = tuple(staged.candidates(conn))
    return BlobRefLivenessClassification(
        scanned_count=staged.classification.scanned_count,
        ref_type_counts=staged.classification.ref_type_counts,
        orphaned_by_ref_type=staged.classification.orphaned_by_ref_type,
        ref_type_joins=staged.classification.ref_type_joins,
        unknown_ref_types=staged.classification.unknown_ref_types,
        unavailable_ref_types=staged.classification.unavailable_ref_types,
        rekeyable_hook_payload_count=staged.classification.rekeyable_hook_payload_count,
        candidates=candidates,
        candidate_count=len(candidates),
    )


__all__ = [
    "BlobRefLivenessCandidate",
    "BlobRefLivenessClassification",
    "BlobRefLivenessError",
    "classify_blob_ref_liveness",
    "stage_blob_ref_liveness",
    "BlobRefLivenessStagedPlan",
]

"""Set-based liveness classification for source-tier ``blob_refs`` rows.

The source tier stores attachment blob refs against the parent ``raw_id``.
That is a source acquisition reference, not an index-tier ``attachment_refs``
edge. The canonical mapping is shared with blob GC so reconciliation and GC
prove liveness with the same referent joins.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from polylogue.storage.blob_gc import BLOB_REF_LIVENESS_JOIN
from polylogue.storage.hook_payload_ref_reconciliation import plan_hook_payload_ref_reconciliation
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

    @property
    def safe_to_apply(self) -> bool:
        return not self.unknown_ref_types and not self.unavailable_ref_types and not self.rekeyable_hook_payload_count

    def to_dict(self, *, include_candidates: bool = False, sample_limit: int = 30) -> dict[str, object]:
        payload: dict[str, object] = {
            "scanned_count": self.scanned_count,
            "ref_type_counts": dict(self.ref_type_counts),
            "orphaned_count": len(self.candidates),
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


def _blob_ref_columns(conn: sqlite3.Connection) -> set[str]:
    return {str(row[1]) for row in conn.execute("PRAGMA table_info(blob_refs)")}


def _referent_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def classify_blob_ref_liveness(conn: sqlite3.Connection) -> BlobRefLivenessClassification:
    """Classify every ``blob_refs`` row using its mapped referent LEFT JOIN.

    The SQL is set-based: one ``LEFT JOIN`` branch per known ref type, joined
    on the actual referent table and key. Unknown ref types, missing referent
    tables, and missing referent columns are reported and make an apply unsafe
    rather than silently treating an unproven row as dead.
    """

    if not table_exists(conn, "blob_refs"):
        return BlobRefLivenessClassification(0, {}, {}, (), (), (), 0, ())
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

    rekeyable_hook_payloads = {
        (candidate.blob_hash.hex(), candidate.orphaned_ref_id)
        for candidate in plan_hook_payload_ref_reconciliation(conn).matched
    }
    candidates: list[BlobRefLivenessCandidate] = []
    if branches:
        query = " UNION ALL ".join(branches) + " ORDER BY 2, 3, 1"
        for row in conn.execute(query, tuple(params)):
            blob_hash = bytes(row[0]).hex()
            ref_type = str(row[1])
            ref_id = str(row[2])
            # Pre-v22 hook refs were stored as raw_payload with a synthetic
            # raw id. They are durable live payloads awaiting a provable
            # re-key to hook_payload, never deletion candidates.
            if (blob_hash, ref_id) in rekeyable_hook_payloads:
                continue
            candidates.append(
                BlobRefLivenessCandidate(
                    blob_hash=blob_hash,
                    ref_type=ref_type,
                    ref_id=ref_id,
                    source_path=str(row[3]) if row[3] is not None else None,
                    size_bytes=int(row[4]),
                    acquired_at_ms=int(row[5]),
                    referent_table=str(row[6]),
                    referent_column=str(row[7]),
                )
            )
    orphaned_by_ref_type: dict[str, int] = {}
    for candidate in candidates:
        orphaned_by_ref_type[candidate.ref_type] = orphaned_by_ref_type.get(candidate.ref_type, 0) + 1
    return BlobRefLivenessClassification(
        scanned_count=sum(ref_type_counts.values()),
        ref_type_counts=ref_type_counts,
        orphaned_by_ref_type=orphaned_by_ref_type,
        ref_type_joins=tuple(ref_type_joins),
        unknown_ref_types=unknown,
        unavailable_ref_types=tuple(sorted(unavailable)),
        rekeyable_hook_payload_count=len(rekeyable_hook_payloads),
        candidates=tuple(candidates),
    )


__all__ = [
    "BlobRefLivenessCandidate",
    "BlobRefLivenessClassification",
    "BlobRefLivenessError",
    "classify_blob_ref_liveness",
]

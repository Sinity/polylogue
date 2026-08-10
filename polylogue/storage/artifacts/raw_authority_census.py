"""Read-only census of quarantined raw authority and artifact backlog.

The census is deliberately a projection, not another authority state machine.
It uses :func:`inspect_raw_artifact` for every row whose retained bytes exist,
then combines that observation with the existing indexed-session evidence. The
result is a deterministic, mutually-exclusive partition that an operator can
review before the artifact-only persistence actuator is run.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from polylogue.archive.artifact_taxonomy import ArtifactKind
from polylogue.core.enums import ArtifactSupportStatus
from polylogue.storage.artifacts.inspection import inspect_raw_artifact
from polylogue.storage.blob_store import BlobStore, get_blob_store
from polylogue.storage.raw_byte_duplicate_supersession import plan_byte_duplicate_supersession
from polylogue.storage.runtime import ArtifactObservationRecord
from polylogue.storage.sqlite.queries.mappers_archive import _row_to_raw_session

TOOL_VERSION = "raw-authority-artifact-census-v1"
MAX_APPLY_ROWS = 5_000


class RawAuthorityBucket(StrEnum):
    """The complete and mutually-exclusive quarantine census partition."""

    ARTIFACT = "artifact"
    TERMINAL_BYTE_DUPLICATE = "terminal_byte_duplicate_superseded"
    NOVEL_MATERIALIZATION = "novel_materialization_candidate"
    MISSING_BYTES = "missing_bytes"
    UNRESOLVED_AUTHORITY = "unresolved_authority"


@dataclass(frozen=True, slots=True)
class RawAuthorityCensusEntry:
    raw_id: str
    bucket: RawAuthorityBucket
    blob_size: int
    reason: str
    artifact_kind: str | None = None
    support_status: ArtifactSupportStatus | None = None
    duplicate_of_raw_id: str | None = None
    duplicate_of_session_id: str | None = None
    observation: ArtifactObservationRecord | None = None

    def to_receipt_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "raw_id": self.raw_id,
            "bucket": self.bucket.value,
            "blob_size": self.blob_size,
            "reason": self.reason,
        }
        if self.artifact_kind is not None:
            payload["artifact_kind"] = self.artifact_kind
        if self.support_status is not None:
            payload["support_status"] = self.support_status.value
        if self.duplicate_of_raw_id is not None:
            payload["duplicate_of_raw_id"] = self.duplicate_of_raw_id
        if self.duplicate_of_session_id is not None:
            payload["duplicate_of_session_id"] = self.duplicate_of_session_id
        return payload


@dataclass(frozen=True, slots=True)
class RawAuthorityArtifactCensus:
    """Deterministic census result, ordered by raw identifier."""

    total_quarantined_count: int
    entries: tuple[RawAuthorityCensusEntry, ...]
    after_raw_id: str | None = None
    has_more: bool | None = None

    @property
    def scanned_count(self) -> int:
        return len(self.entries)

    @property
    def truncated(self) -> bool:
        return self.has_more if self.has_more is not None else self.scanned_count < self.total_quarantined_count

    @property
    def next_after_raw_id(self) -> str | None:
        """Return the exclusive cursor for the next bounded page, if any."""
        if not self.truncated or not self.entries:
            return None
        return self.entries[-1].raw_id

    def entries_for(self, bucket: RawAuthorityBucket) -> tuple[RawAuthorityCensusEntry, ...]:
        return tuple(entry for entry in self.entries if entry.bucket is bucket)

    def counts(self) -> dict[str, int]:
        return {bucket.value: len(self.entries_for(bucket)) for bucket in RawAuthorityBucket}

    def bytes_by_bucket(self) -> dict[str, int]:
        return {
            bucket.value: sum(entry.blob_size for entry in self.entries_for(bucket)) for bucket in RawAuthorityBucket
        }

    def artifact_observations(self) -> tuple[ArtifactObservationRecord, ...]:
        return tuple(
            entry.observation
            for entry in self.entries_for(RawAuthorityBucket.ARTIFACT)
            if entry.observation is not None
        )

    def receipt_payload(
        self,
        *,
        mode: str,
        observations_written: int = 0,
        observed_at_ms: int | None = None,
        evidence: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Return a canonical, identifier-bearing operator receipt.

        Source paths and payload content are intentionally absent. Raw ids,
        artifact kinds, and authority witnesses are sufficient to reconcile the
        result without copying private source locations into an operator file.
        """
        if mode not in {"dry_run", "apply"}:
            raise ValueError(f"unsupported census receipt mode: {mode!r}")
        raw_ids_by_bucket = {
            bucket.value: [entry.raw_id for entry in self.entries_for(bucket)] for bucket in RawAuthorityBucket
        }
        payload: dict[str, object] = {
            "tool_version": TOOL_VERSION,
            "mode": mode,
            "observed_at_ms": observed_at_ms,
            "total_quarantined_count": self.total_quarantined_count,
            "scanned_count": self.scanned_count,
            "truncated": self.truncated,
            "counts": self.counts(),
            "bytes_by_bucket": self.bytes_by_bucket(),
            "raw_ids_by_bucket": raw_ids_by_bucket,
            "entries": [entry.to_receipt_dict() for entry in self.entries],
            "observations_written": observations_written,
            "page": {
                "after_raw_id": self.after_raw_id,
                "next_after_raw_id": self.next_after_raw_id,
            },
            "scope": {
                "logical_database_operations": (["upsert raw_artifacts observations"] if mode == "apply" else []),
                "physical_database_operations": (
                    ["PRAGMA wal_checkpoint(TRUNCATE) on source.db after initial backup validation"]
                    if mode == "apply"
                    else []
                ),
                "unchanged": ["raw_sessions rows", "revision authority", "index.db rows", "blob store"],
            },
        }
        if evidence is not None:
            payload["evidence"] = evidence
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        payload["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
        return payload


def _blob_hash_hex(value: object) -> str | None:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        return None
    raw = bytes(value)
    if len(raw) != 32:
        return None
    return raw.hex()


def _indexed_raw_ids(index_conn: sqlite3.Connection, raw_ids: list[str]) -> set[str]:
    """Return selected raws with a materialized session, without classifying authority."""
    indexed_session_by_raw_id: dict[str, str] = {}
    for start in range(0, len(raw_ids), 500):
        raw_id_chunk = raw_ids[start : start + 500]
        if not raw_id_chunk:
            continue
        marks = ",".join("?" for _ in raw_id_chunk)
        rows = index_conn.execute(
            f"SELECT raw_id, session_id FROM sessions WHERE raw_id IN ({marks}) ORDER BY raw_id, session_id",
            raw_id_chunk,
        ).fetchall()
        for row in rows:
            indexed_session_by_raw_id.setdefault(str(row["raw_id"]), str(row["session_id"]))
    return set(indexed_session_by_raw_id)


def scan_quarantined_raw_authority(
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection,
    *,
    blob_store: BlobStore | None = None,
    limit: int | None = None,
    after_raw_id: str | None = None,
    raw_ids: Sequence[str] | None = None,
    total_quarantined_count: int | None = None,
    has_more: bool | None = None,
) -> RawAuthorityArtifactCensus:
    """Classify quarantined raw rows without mutating either database.

    Classification precedence is fixed: missing bytes, indexed byte-identical
    twin, known non-session artifact, parseable novel raw, unresolved
    authority. Every selected row lands in exactly one bucket.
    """
    if limit is not None and limit < 0:
        raise ValueError("census limit must be non-negative")
    blob_store = blob_store or get_blob_store()
    previous_source_factory = source_conn.row_factory
    previous_index_factory = index_conn.row_factory
    source_conn.row_factory = sqlite3.Row
    index_conn.row_factory = sqlite3.Row
    try:
        if raw_ids is not None and limit is not None:
            raise ValueError("raw_ids cannot be combined with limit")
        selected = tuple(dict.fromkeys(raw_ids or ()))
        if raw_ids is None:
            total = int(
                source_conn.execute(
                    """
                    SELECT COUNT(*) FROM raw_sessions
                    WHERE revision_authority = 'quarantined'
                      AND parse_error IS NULL
                      AND (? IS NULL OR raw_id > ?)
                    """,
                    (after_raw_id, after_raw_id),
                ).fetchone()[0]
            )
        else:
            total = total_quarantined_count if total_quarantined_count is not None else len(selected)
            if not selected:
                return RawAuthorityArtifactCensus(total_quarantined_count=total, entries=(), has_more=has_more)
        query = """
            SELECT rowid AS raw_rowid, *
            FROM raw_sessions
            WHERE 1 = 1
        """
        params: list[object] = []
        if raw_ids is not None:
            placeholders = ",".join("?" for _ in selected)
            query += f" AND raw_id IN ({placeholders})"
            params.extend(selected)
        else:
            query += " AND revision_authority = 'quarantined' AND parse_error IS NULL"
        if raw_ids is None and after_raw_id is not None:
            query += " AND raw_id > ?"
            params.append(after_raw_id)
        query += " ORDER BY raw_id"
        if raw_ids is None and limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        rows = source_conn.execute(query, params).fetchall()
        if raw_ids is not None:
            found = {str(row["raw_id"]) for row in rows}
            if found != set(selected):
                raise ValueError("checkpoint candidate raw is missing from source.db")
            if any(row["revision_authority"] != "quarantined" or row["parse_error"] is not None for row in rows):
                raise ValueError("checkpoint candidate no longer belongs to the accepted parser universe")
        duplicate_plan = plan_byte_duplicate_supersession(
            source_conn,
            index_conn,
            limit=limit if raw_ids is None else None,
            after_raw_id=after_raw_id if raw_ids is None else None,
            raw_ids=selected if raw_ids is not None else None,
        )
        duplicate_by_raw_id = {candidate.raw_id: candidate for candidate in duplicate_plan.duplicates}
        indexed_raw_ids = _indexed_raw_ids(index_conn, [str(row["raw_id"]) for row in rows])
        entries: list[RawAuthorityCensusEntry] = []
        for row in rows:
            raw_id = str(row["raw_id"])
            blob_size = int(row["blob_size"] or 0)
            blob_hash_hex = _blob_hash_hex(row["blob_hash"])
            if blob_hash_hex is None or not blob_store.exists(blob_hash_hex):
                entries.append(
                    RawAuthorityCensusEntry(
                        raw_id=raw_id,
                        bucket=RawAuthorityBucket.MISSING_BYTES,
                        blob_size=blob_size,
                        reason="retained blob is absent from the content-addressed store",
                    )
                )
                continue

            if duplicate := duplicate_by_raw_id.get(raw_id):
                observation = inspect_raw_artifact(_row_to_raw_session(row), blob_store=blob_store)
                entries.append(
                    RawAuthorityCensusEntry(
                        raw_id=raw_id,
                        bucket=RawAuthorityBucket.TERMINAL_BYTE_DUPLICATE,
                        blob_size=blob_size,
                        reason="byte-identical raw already has an indexed twin",
                        artifact_kind=observation.artifact_kind,
                        support_status=observation.support_status,
                        duplicate_of_raw_id=duplicate.duplicate_of_raw_id,
                        duplicate_of_session_id=duplicate.duplicate_of_session_id,
                        observation=observation,
                    )
                )
                continue

            observation = inspect_raw_artifact(_row_to_raw_session(row), blob_store=blob_store)
            if not observation.parse_as_session and observation.artifact_kind != ArtifactKind.UNKNOWN.value:
                entries.append(
                    RawAuthorityCensusEntry(
                        raw_id=raw_id,
                        bucket=RawAuthorityBucket.ARTIFACT,
                        blob_size=blob_size,
                        reason=observation.classification_reason,
                        artifact_kind=observation.artifact_kind,
                        support_status=observation.support_status,
                        observation=observation,
                    )
                )
                continue

            if (
                observation.parse_as_session
                and observation.support_status is ArtifactSupportStatus.SUPPORTED_PARSEABLE
                and row["logical_source_key"] is None
                and raw_id not in indexed_raw_ids
            ):
                entries.append(
                    RawAuthorityCensusEntry(
                        raw_id=raw_id,
                        bucket=RawAuthorityBucket.NOVEL_MATERIALIZATION,
                        blob_size=blob_size,
                        reason="parseable session payload has no indexed twin or logical authority key",
                        artifact_kind=observation.artifact_kind,
                        support_status=observation.support_status,
                        observation=observation,
                    )
                )
                continue

            entries.append(
                RawAuthorityCensusEntry(
                    raw_id=raw_id,
                    bucket=RawAuthorityBucket.UNRESOLVED_AUTHORITY,
                    blob_size=blob_size,
                    reason=("raw payload is not a proven artifact, terminal duplicate, or novel parseable candidate"),
                    artifact_kind=observation.artifact_kind,
                    support_status=observation.support_status,
                    observation=observation,
                )
            )
        return RawAuthorityArtifactCensus(
            total_quarantined_count=total,
            entries=tuple(entries),
            after_raw_id=after_raw_id,
            has_more=has_more,
        )
    finally:
        source_conn.row_factory = previous_source_factory
        index_conn.row_factory = previous_index_factory


def write_artifact_observations(
    conn: sqlite3.Connection,
    observations: tuple[ArtifactObservationRecord, ...],
) -> int:
    """Upsert only non-session artifact observations, without committing."""
    from polylogue.storage.artifacts.persistence import upsert_artifact_observations

    for observation in observations:
        if observation.parse_as_session:
            raise ValueError("artifact census actuator received a session observation")
    return upsert_artifact_observations(conn, observations)


__all__ = [
    "MAX_APPLY_ROWS",
    "RawAuthorityArtifactCensus",
    "RawAuthorityBucket",
    "RawAuthorityCensusEntry",
    "TOOL_VERSION",
    "scan_quarantined_raw_authority",
    "write_artifact_observations",
]

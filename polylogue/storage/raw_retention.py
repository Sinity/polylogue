"""Retention cleanup for superseded live raw payload snapshots."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import closing, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from polylogue.core.raw_failure_evidence import RAW_FAILURE_EVIDENCE_KINDS, RawFailureEvidenceKind
from polylogue.logging import get_logger
from polylogue.storage.archive_identity import ArchiveLocationError, resolve_active_index_path
from polylogue.storage.blob_store import BlobStore, get_blob_store
from polylogue.storage.introspection import column_exists as _column_exists
from polylogue.storage.introspection import table_exists as _table_exists

logger = get_logger(__name__)

_TERMINAL_RAW_FAILURE_EVIDENCE_KINDS = frozenset(
    kind.value for kind in RawFailureEvidenceKind if kind.lifecycle == "terminal"
)

_V1_RAW_CANDIDATE_SQL = """
WITH ranked AS (
    SELECT
        raw_id,
        source_path,
        source_index,
        blob_hash,
        blob_size,
        acquired_at_ms,
        ROW_NUMBER() OVER (
            PARTITION BY source_path, source_index
            ORDER BY acquired_at_ms DESC, raw_id DESC
        ) AS recency
    FROM raw_sessions
    WHERE source_index IN (-1, 0)
      AND (? IS NULL OR source_path = ?)
      AND (? IS NULL OR acquired_at_ms >= ?)
)
SELECT raw_id, source_path, source_index, blob_hash, blob_size
FROM ranked AS r
WHERE r.recency > CASE WHEN r.source_index = 0 THEN ? ELSE ? END
ORDER BY r.blob_size DESC, r.acquired_at_ms ASC, r.raw_id ASC
LIMIT ?
"""

_RAW_REVISION_CHAIN_COLUMN_NAMES = (
    "raw_id",
    "source_index",
    "logical_source_key",
    "revision_kind",
    "source_revision",
    "predecessor_source_revision",
    "predecessor_raw_id",
    "baseline_raw_id",
    "append_start_offset",
    "append_end_offset",
    "acquisition_generation",
    "revision_authority",
    "blob_size",
)
_RAW_REVISION_CHAIN_COLUMNS = ", ".join(_RAW_REVISION_CHAIN_COLUMN_NAMES)


class RawRetentionSafetyError(RuntimeError):
    """Raised when active raw evidence cannot be proven safe for retention."""


class _RawRevisionAuthorityUnavailableError(RawRetentionSafetyError):
    """Raised when source-tier authority cannot be read, not when it is invalid."""


@dataclass(frozen=True)
class RawSnapshotCleanupCandidate:
    raw_id: str
    source_path: str
    source_index: int
    blob_size: int
    blob_hash: str | None = None

    @property
    def blob_store_hash(self) -> str:
        return self.blob_hash or self.raw_id


@dataclass(frozen=True)
class RawSnapshotCleanupResult:
    candidate_count: int
    deleted_raw_count: int
    deleted_blob_count: int
    deleted_raw_bytes: int
    deleted_blob_bytes: int
    skipped_missing_source_count: int
    skipped_referenced_count: int = 0
    deleted_orphaned_authority_plan_count: int = 0
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class RawRetentionAuthority:
    """Index-proven source rows to preserve and rows authorized for deletion."""

    protected_raw_ids: frozenset[str]
    eligible_raw_ids: frozenset[str]


@dataclass(frozen=True)
class _IndexRawRevisionHead:
    logical_source_key: str
    accepted_raw_id: str
    accepted_source_revision: str
    accepted_frontier_kind: str
    accepted_frontier: int
    acquisition_generation: int
    append_end_offset: int | None


@dataclass(frozen=True)
class _EligibleRawReceipt:
    raw_id: str
    logical_source_key: str
    source_revision: str
    baseline_raw_id: str | None
    predecessor_raw_id: str | None


def _blob_hash_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.hex() if len(value) == 32 else None
    text = str(value)
    return text if text else None


def _timestamp_ms(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def _active_index_raw_authority(
    index_db_path: Path,
) -> tuple[frozenset[str], tuple[_IndexRawRevisionHead, ...], tuple[_EligibleRawReceipt, ...]]:
    """Read current raw references and explicit deletion receipts read-only.

    ``head.accepted_frontier_kind = 'byte'`` below is a deliberate, permanent
    restriction, not an unexamined narrowing (polylogue-hgsq). A byte
    frontier's supersession claim is independently re-derivable from
    source-tier facts alone -- ``_validate_byte_head`` /
    ``_validate_active_revision_chain`` recompute append-offset containment
    straight from ``raw_sessions`` byte accounting, so retention never has to
    trust the parser/classifier that first produced the receipt. A semantic
    frontier's claim (``classify_membership_revisions`` /
    ``_strictly_dominates`` in ``archive/session_revision_membership.py``,
    invoked from the multi-session membership-replay path in
    ``storage/sqlite/archive_tiers/archive.py``) is a one-time judgment over
    *parsed* message/event/attachment hash sequences; only a scalar hash
    count (``accepted_frontier``), not the hash sequences themselves,
    survives into ``raw_revision_heads``, so there is no way to re-verify
    that domination later without re-parsing and re-trusting the classifier.
    Verified empirically against the live archive (2026-07-29): of 10,607
    raws superseded under a semantic head, 10,606 carry
    ``raw_sessions.revision_authority = 'quarantined'`` in the source tier
    (one is 'byte_proven' for an unrelated reason) -- so even removing this
    predicate would change nothing, because every one of them would still be
    rejected downstream by ``_validate_eligible_receipt``'s unconditional
    ``revision_kind in {'full','append'}`` + ``revision_authority ==
    'byte_proven'`` requirement on the raw itself. Building a semantic
    release path is a distinct, schema-bearing undertaking (persisting a
    durable, source-tier-anchored fingerprint of the domination proof, and a
    matching independent re-verification step) that touches
    ``raw_authority.py`` / ``revision_application.py`` /
    ``sources/revision_backfill.py`` and very likely a derived-tier schema
    bump -- out of scope here and not a quick relaxation of this line.
    """
    if not index_db_path.is_file():
        raise RawRetentionSafetyError(f"index tier is unavailable: {index_db_path}")
    try:
        uri = f"{index_db_path.resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.execute("PRAGMA query_only = ON")
            session_rows = conn.execute("SELECT DISTINCT raw_id FROM sessions WHERE raw_id IS NOT NULL").fetchall()
            head_rows = conn.execute(
                """SELECT logical_source_key, accepted_raw_id, accepted_source_revision,
                          accepted_frontier_kind, accepted_frontier,
                          acquisition_generation, append_end_offset
                   FROM raw_revision_heads"""
            ).fetchall()
            eligible_rows = conn.execute(
                """SELECT DISTINCT application.raw_id,
                          application.logical_source_key,
                          application.source_revision,
                          application.baseline_raw_id,
                          application.predecessor_raw_id
                   FROM raw_revision_applications AS application
                   JOIN raw_revision_heads AS head
                     ON head.logical_source_key = application.logical_source_key
                    AND head.session_id = application.session_id
                    AND head.accepted_raw_id = application.accepted_raw_id
                    AND head.accepted_source_revision = application.accepted_source_revision
                    AND head.accepted_content_hash = application.accepted_content_hash
                    AND head.acquisition_generation = application.acquisition_generation
                    AND head.append_end_offset IS application.append_end_offset
                    AND head.decided_at_ms = application.decided_at_ms
                   WHERE application.decision = 'superseded'
                     AND head.accepted_frontier_kind = 'byte'"""
            ).fetchall()
    except (OSError, sqlite3.Error) as exc:
        raise RawRetentionSafetyError(f"index tier raw authority is unreadable: {exc}") from exc
    session_raw_ids = frozenset(str(row[0]) for row in session_rows if row[0] is not None and str(row[0]))
    heads = tuple(
        _IndexRawRevisionHead(
            logical_source_key=str(row[0]),
            accepted_raw_id=str(row[1]),
            accepted_source_revision=str(row[2]),
            accepted_frontier_kind=str(row[3]),
            accepted_frontier=int(row[4]),
            acquisition_generation=int(row[5]),
            append_end_offset=int(row[6]) if row[6] is not None else None,
        )
        for row in head_rows
    )
    eligible_receipts = tuple(
        _EligibleRawReceipt(
            raw_id=str(row[0]),
            logical_source_key=str(row[1]),
            source_revision=str(row[2]),
            baseline_raw_id=str(row[3]) if row[3] is not None else None,
            predecessor_raw_id=str(row[4]) if row[4] is not None else None,
        )
        for row in eligible_rows
    )
    return session_raw_ids, heads, eligible_receipts


def _raw_revision_rows(
    conn: sqlite3.Connection,
    raw_ids: set[str],
    *,
    allow_missing: bool = False,
) -> dict[str, sqlite3.Row]:
    rows_by_id: dict[str, sqlite3.Row] = {}
    pending = set(raw_ids)
    while pending:
        batch = tuple(sorted(pending)[:500])
        pending.difference_update(batch)
        placeholders = ", ".join("?" for _ in batch)
        try:
            rows = conn.execute(
                f"SELECT {_RAW_REVISION_CHAIN_COLUMNS} FROM raw_sessions WHERE raw_id IN ({placeholders})",
                batch,
            ).fetchall()
        except sqlite3.Error as exc:
            raise _RawRevisionAuthorityUnavailableError(f"source raw revision authority is unreadable: {exc}") from exc
        found = {str(row["raw_id"]): row for row in rows}
        missing = set(batch).difference(found)
        if missing and not allow_missing:
            rendered = ", ".join(sorted(missing)[:3])
            raise RawRetentionSafetyError(f"active index raw is missing from source tier: {rendered}")
        rows_by_id.update(found)
        for row in rows:
            if str(row["revision_kind"]) != "append":
                continue
            predecessor = row["predecessor_raw_id"]
            if predecessor is not None and str(predecessor) not in rows_by_id:
                pending.add(str(predecessor))
    return rows_by_id


def _validate_active_revision_chain(
    rows_by_id: dict[str, sqlite3.Row],
    seed_raw_id: str,
) -> frozenset[str]:
    protected: set[str] = set()
    current_raw_id = seed_raw_id
    chain_baseline_raw_id: str | None = None
    while True:
        if current_raw_id in protected:
            raise RawRetentionSafetyError(f"active raw revision chain contains a cycle at {current_raw_id}")
        protected.add(current_raw_id)
        row = rows_by_id.get(current_raw_id)
        if row is None:
            raise RawRetentionSafetyError(f"active index raw is missing from source tier: {current_raw_id}")
        kind = str(row["revision_kind"])
        if kind == "full":
            if str(row["revision_authority"]) != "byte_proven":
                raise RawRetentionSafetyError(f"active full raw lacks byte-proven authority: {current_raw_id}")
            if chain_baseline_raw_id is not None and current_raw_id != chain_baseline_raw_id:
                raise RawRetentionSafetyError(f"active append chain terminates at the wrong baseline: {current_raw_id}")
            # A full payload is a self-contained reset even when historical
            # classification records an older full predecessor.
            return frozenset(protected)
        if kind == "unknown":
            if int(row["source_index"]) == -1:
                raise RawRetentionSafetyError(f"active append raw lacks revision authority: {current_raw_id}")
            return frozenset(protected)
        if kind != "append":
            raise RawRetentionSafetyError(f"active raw has unsupported revision kind {kind!r}: {current_raw_id}")
        if str(row["revision_authority"]) != "byte_proven":
            raise RawRetentionSafetyError(f"active append raw lacks byte-proven authority: {current_raw_id}")
        logical_source_key = row["logical_source_key"]
        predecessor_raw_id = row["predecessor_raw_id"]
        predecessor_revision = row["predecessor_source_revision"]
        declared_baseline_value = row["baseline_raw_id"]
        if not all((logical_source_key, predecessor_raw_id, predecessor_revision, declared_baseline_value)):
            raise RawRetentionSafetyError(f"active append raw has an incomplete predecessor envelope: {current_raw_id}")
        declared_baseline = str(row["baseline_raw_id"])
        if chain_baseline_raw_id is None:
            chain_baseline_raw_id = declared_baseline
        elif declared_baseline != chain_baseline_raw_id:
            raise RawRetentionSafetyError(f"active append chain changes baseline identity: {current_raw_id}")
        parent_id = str(predecessor_raw_id)
        parent = rows_by_id.get(parent_id)
        if parent is None:
            raise RawRetentionSafetyError(f"active append predecessor is missing from source tier: {parent_id}")
        if parent["logical_source_key"] != logical_source_key:
            raise RawRetentionSafetyError(f"active append predecessor crosses logical sources: {current_raw_id}")
        if parent["source_revision"] != predecessor_revision:
            raise RawRetentionSafetyError(f"active append predecessor revision does not match: {current_raw_id}")
        parent_kind = str(parent["revision_kind"])
        if str(parent["revision_authority"]) != "byte_proven":
            raise RawRetentionSafetyError(f"active append predecessor lacks byte-proven authority: {parent_id}")
        if parent_kind == "append" and str(parent["baseline_raw_id"] or "") != chain_baseline_raw_id:
            raise RawRetentionSafetyError(f"active append predecessor changes baseline identity: {current_raw_id}")
        expected_offset = parent["blob_size"] if parent_kind == "full" else parent["append_end_offset"]
        if parent_kind not in {"full", "append"} or expected_offset != row["append_start_offset"]:
            raise RawRetentionSafetyError(f"active append predecessor is not byte-contiguous: {current_raw_id}")
        parent_generation = parent["acquisition_generation"]
        child_generation = row["acquisition_generation"]
        if parent_generation is None or child_generation is None or int(child_generation) != int(parent_generation) + 1:
            raise RawRetentionSafetyError(f"active append predecessor generation does not match: {current_raw_id}")
        current_raw_id = parent_id


def _validate_byte_head(row: sqlite3.Row, head: _IndexRawRevisionHead) -> None:
    if str(row["revision_kind"]) not in {"full", "append"}:
        raise RawRetentionSafetyError(
            f"byte head references a raw without typed revision authority: {head.accepted_raw_id}"
        )
    if row["logical_source_key"] != head.logical_source_key:
        raise RawRetentionSafetyError(f"accepted raw logical source disagrees with index head: {head.accepted_raw_id}")
    if row["source_revision"] != head.accepted_source_revision:
        raise RawRetentionSafetyError(f"accepted raw revision disagrees with index head: {head.accepted_raw_id}")
    if row["acquisition_generation"] != head.acquisition_generation:
        raise RawRetentionSafetyError(f"accepted raw generation disagrees with index head: {head.accepted_raw_id}")
    is_append = str(row["revision_kind"]) == "append"
    raw_frontier = row["append_end_offset"] if is_append else row["blob_size"]
    if raw_frontier != head.accepted_frontier:
        raise RawRetentionSafetyError(f"accepted raw frontier disagrees with index head: {head.accepted_raw_id}")
    if (is_append and raw_frontier != head.append_end_offset) or (not is_append and head.append_end_offset is not None):
        raise RawRetentionSafetyError(f"accepted raw append end disagrees with index head: {head.accepted_raw_id}")


def _validate_eligible_receipt(row: sqlite3.Row, receipt: _EligibleRawReceipt) -> None:
    if str(row["revision_kind"]) not in {"full", "append"}:
        raise RawRetentionSafetyError(f"superseded receipt references an untyped raw: {receipt.raw_id}")
    if str(row["revision_authority"]) != "byte_proven":
        raise RawRetentionSafetyError(f"superseded receipt references unproven raw evidence: {receipt.raw_id}")
    if row["logical_source_key"] != receipt.logical_source_key:
        raise RawRetentionSafetyError(f"superseded receipt crosses logical sources: {receipt.raw_id}")
    if row["source_revision"] != receipt.source_revision:
        raise RawRetentionSafetyError(f"superseded receipt revision disagrees with source evidence: {receipt.raw_id}")
    if row["baseline_raw_id"] != receipt.baseline_raw_id:
        raise RawRetentionSafetyError(f"superseded receipt baseline disagrees with source evidence: {receipt.raw_id}")
    if row["predecessor_raw_id"] != receipt.predecessor_raw_id:
        raise RawRetentionSafetyError(
            f"superseded receipt predecessor disagrees with source evidence: {receipt.raw_id}"
        )


def active_raw_retention_authority(
    conn: sqlite3.Connection,
    *,
    index_db_path: Path,
    terminal_source_paths: Iterable[Path] | None = None,
) -> RawRetentionAuthority:
    """Return current protection plus explicitly authorized deletion rows.

    The index selects active leaves. Source-tier predecessor evidence proves
    which append fragments are required to reconstruct each leaf. Only an
    immutable ``superseded`` receipt tied to the current head authorizes raw
    deletion. Callers must serialize this read with source deletion under the
    daemon's single-writer contract, or stop the daemon for manual cleanup.
    ``terminal_source_paths`` scopes terminal-artifact protection only for a
    deletion operation constrained to those same physical paths; callers that
    may delete archive-wide must leave it unset.
    """
    original_row_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        session_raw_ids, heads, eligible_receipts = _active_index_raw_authority(index_db_path)
        seeds = set(session_raw_ids)
        seeds.update(head.accepted_raw_id for head in heads)
        if not seeds:
            all_raw_ids = frozenset(str(row[0]) for row in conn.execute("SELECT raw_id FROM raw_sessions").fetchall())
            terminal_artifact_raw_ids = _terminal_artifact_raw_ids(conn)
            if all_raw_ids and all_raw_ids.issubset(terminal_artifact_raw_ids):
                return RawRetentionAuthority(protected_raw_ids=all_raw_ids, eligible_raw_ids=frozenset())
            if all_raw_ids:
                raise RawRetentionSafetyError("source tier contains raw evidence but index has no raw authority")
            return RawRetentionAuthority(protected_raw_ids=frozenset(), eligible_raw_ids=frozenset())
        terminal_artifact_raw_ids = _terminal_artifact_raw_ids(conn, source_paths=terminal_source_paths)
        authority_raw_ids = seeds.union(receipt.raw_id for receipt in eligible_receipts)
        rows_by_id = _raw_revision_rows(conn, authority_raw_ids)
        protected: set[str] = set()
        byte_head_raw_ids = {head.accepted_raw_id for head in heads if head.accepted_frontier_kind == "byte"}
        semantic_only_raw_ids = {
            head.accepted_raw_id for head in heads if head.accepted_frontier_kind != "byte"
        }.difference(byte_head_raw_ids)
        # A semantic membership head is accepted authority for retention, but
        # it is deliberately not a byte-predecessor proof. Keep it protected
        # without reinterpreting it as one.
        protected.update(semantic_only_raw_ids)
        protected.update(terminal_artifact_raw_ids)
        for seed_raw_id in sorted(session_raw_ids):
            if seed_raw_id in semantic_only_raw_ids:
                continue
            protected.update(_validate_active_revision_chain(rows_by_id, seed_raw_id))
        for head in heads:
            if head.accepted_raw_id in semantic_only_raw_ids:
                continue
            row = rows_by_id[head.accepted_raw_id]
            if head.accepted_frontier_kind == "byte":
                _validate_byte_head(row, head)
            protected.update(_validate_active_revision_chain(rows_by_id, head.accepted_raw_id))
        eligible: set[str] = set()
        for receipt in eligible_receipts:
            _validate_eligible_receipt(rows_by_id[receipt.raw_id], receipt)
            eligible.add(receipt.raw_id)
        protected_ids = frozenset(protected)
        return RawRetentionAuthority(
            protected_raw_ids=protected_ids,
            eligible_raw_ids=frozenset(eligible.difference(protected_ids)),
        )
    finally:
        conn.row_factory = original_row_factory


def protected_active_raw_revision_ids(
    conn: sqlite3.Connection,
    *,
    index_db_path: Path,
) -> frozenset[str]:
    """Compatibility projection for callers that only inspect protection."""
    return active_raw_retention_authority(conn, index_db_path=index_db_path).protected_raw_ids


# ---------------------------------------------------------------------------
# Stale supersession receipt reissue (polylogue-ktwa)
# ---------------------------------------------------------------------------
#
# ``_active_index_raw_authority`` above is deliberately strict: it only admits
# a ``superseded`` raw for release when its receipt still joins the CURRENT
# ``raw_revision_heads`` row on all eight identity columns. That join is
# correct -- a receipt proven against a head state that no longer holds must
# not authorize deletion. But nothing re-proves supersession when the head
# later advances, so a raw superseded against head generation N keeps a
# receipt naming N forever and becomes permanently unreleasable once the head
# moves to N+1, even though it is now *more* superseded than when the receipt
# was written.
#
# This section re-evaluates already-``superseded`` receipts against the live
# head and, when supersession still holds, writes a brand-new receipt (a new
# ``decision_id`` -- existing rows are never mutated) whose accepted_* fields
# exactly mirror the current head, so it immediately satisfies the
# eight-column join above.
#
# "Still holds" is proven by the *same* rule already coded into
# ``_validate_active_revision_chain``'s full-kind branch: "A full payload is
# a self-contained reset even when historical classification records an
# older full predecessor." A byte-proven ``full`` raw re-captures a logical
# source's entire content from scratch, so once it is the accepted head, it
# unconditionally supersedes every *other* typed, byte-proven raw sharing the
# same ``logical_source_key`` -- independent of that raw's own lineage. This
# is deliberately the *only* proof this pass accepts:
#
# * An ongoing append chain never needs reissue in the first place -- an
#   append raw's own predecessors are recorded ``applied_append`` /
#   ``selected_baseline`` once and stay that way; they are never later
#   relabelled ``superseded``, and ``active_raw_retention_authority``
#   protects them (via ``_validate_active_revision_chain``) for as long as
#   they remain reachable from the accepted head, because their bytes are
#   still required to reconstruct it. So when the current head is itself an
#   ``append`` (not a fresh ``full`` reset), this pass fails closed and
#   reissues nothing for that logical source -- there is no independent,
#   generally safe rule (short of the ancestor-chain walk retention already
#   performs, which would only ever re-derive "protected", never "eligible")
#   that proves a *different* raw is superseded by an in-progress append
#   chain.
# * A validated live-data query against the archive that motivated this bead
#   confirmed the shape empirically: every stale byte-headed receipt where
#   the head's ``accepted_raw_id`` changed pointed at a head whose own
#   ``revision_kind`` is ``full``; every stale receipt where the head's
#   ``accepted_raw_id`` stayed the same only ever differed on a *semantic*
#   frontier head (already excluded below), never a byte one.


@dataclass(frozen=True)
class _CurrentRawRevisionHead:
    """Full ``raw_revision_heads`` row -- superset of ``_IndexRawRevisionHead``.

    The retention join intentionally reads only the columns it needs.
    Reissue additionally needs ``session_id``, ``accepted_content_hash``, and
    ``decided_at_ms`` to construct a fresh receipt that exactly reproduces the
    current head's identity, so it reads the table separately rather than
    widening the safety-critical retention projection.
    """

    logical_source_key: str
    session_id: str
    accepted_raw_id: str
    accepted_source_revision: str
    accepted_content_hash: bytes
    accepted_frontier_kind: str
    accepted_frontier: int
    acquisition_generation: int
    append_end_offset: int | None
    decided_at_ms: int


@dataclass(frozen=True)
class _StaleSupersededApplication:
    raw_id: str
    session_id: str
    logical_source_key: str
    source_revision: str
    accepted_raw_id: str | None
    accepted_source_revision: str | None
    accepted_content_hash: bytes | None
    acquisition_generation: int
    append_end_offset: int | None
    decided_at_ms: int


def _read_current_raw_revision_heads(index_db_path: Path) -> dict[str, _CurrentRawRevisionHead]:
    if not index_db_path.is_file():
        raise RawRetentionSafetyError(f"index tier is unavailable: {index_db_path}")
    try:
        uri = f"{index_db_path.resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.execute("PRAGMA query_only = ON")
            rows = conn.execute(
                """SELECT logical_source_key, session_id, accepted_raw_id, accepted_source_revision,
                          accepted_content_hash, accepted_frontier_kind, accepted_frontier,
                          acquisition_generation, append_end_offset, decided_at_ms
                   FROM raw_revision_heads"""
            ).fetchall()
    except (OSError, sqlite3.Error) as exc:
        raise RawRetentionSafetyError(f"index tier raw authority is unreadable: {exc}") from exc
    heads: dict[str, _CurrentRawRevisionHead] = {}
    for row in rows:
        accepted_content_hash = row[4]
        if not isinstance(accepted_content_hash, bytes):
            raise RawRetentionSafetyError(f"raw revision head has a non-blob accepted_content_hash: {row[0]!r}")
        heads[str(row[0])] = _CurrentRawRevisionHead(
            logical_source_key=str(row[0]),
            session_id=str(row[1]),
            accepted_raw_id=str(row[2]),
            accepted_source_revision=str(row[3]),
            accepted_content_hash=accepted_content_hash,
            accepted_frontier_kind=str(row[5]),
            accepted_frontier=int(row[6]),
            acquisition_generation=int(row[7]),
            append_end_offset=int(row[8]) if row[8] is not None else None,
            decided_at_ms=int(row[9]),
        )
    return heads


def _read_superseded_applications(index_db_path: Path) -> tuple[_StaleSupersededApplication, ...]:
    if not index_db_path.is_file():
        raise RawRetentionSafetyError(f"index tier is unavailable: {index_db_path}")
    try:
        uri = f"{index_db_path.resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.execute("PRAGMA query_only = ON")
            rows = conn.execute(
                """SELECT raw_id, session_id, logical_source_key, source_revision,
                          accepted_raw_id, accepted_source_revision, accepted_content_hash,
                          acquisition_generation, append_end_offset, decided_at_ms
                   FROM raw_revision_applications
                   WHERE decision = 'superseded'"""
            ).fetchall()
    except (OSError, sqlite3.Error) as exc:
        raise RawRetentionSafetyError(f"index tier raw authority is unreadable: {exc}") from exc
    applications: list[_StaleSupersededApplication] = []
    for row in rows:
        accepted_content_hash = row[6]
        if accepted_content_hash is not None and not isinstance(accepted_content_hash, bytes):
            raise RawRetentionSafetyError(f"superseded application has a non-blob accepted_content_hash: {row[0]!r}")
        applications.append(
            _StaleSupersededApplication(
                raw_id=str(row[0]),
                session_id=str(row[1]),
                logical_source_key=str(row[2]),
                source_revision=str(row[3]),
                accepted_raw_id=str(row[4]) if row[4] is not None else None,
                accepted_source_revision=str(row[5]) if row[5] is not None else None,
                accepted_content_hash=accepted_content_hash,
                acquisition_generation=int(row[7]),
                append_end_offset=int(row[8]) if row[8] is not None else None,
                decided_at_ms=int(row[9]),
            )
        )
    return tuple(applications)


def _application_matches_head(app: _StaleSupersededApplication, head: _CurrentRawRevisionHead) -> bool:
    """Mirror the exact eight-column join ``_active_index_raw_authority`` uses."""
    return (
        app.session_id == head.session_id
        and app.accepted_raw_id == head.accepted_raw_id
        and app.accepted_source_revision == head.accepted_source_revision
        and app.accepted_content_hash == head.accepted_content_hash
        and app.acquisition_generation == head.acquisition_generation
        and app.append_end_offset == head.append_end_offset
        and app.decided_at_ms == head.decided_at_ms
    )


@dataclass(frozen=True)
class StaleSupersessionCandidate:
    """One superseded raw whose stale receipt is provably reissuable.

    ``source_revision``/``baseline_raw_id``/``predecessor_raw_id`` are the
    superseded raw's own source-tier identity (unchanged by reissue);
    ``current_head`` is the live full-reset head the fresh receipt binds to.
    """

    raw_id: str
    session_id: str
    logical_source_key: str
    source_revision: str
    baseline_raw_id: str | None
    predecessor_raw_id: str | None
    current_head: _CurrentRawRevisionHead


@dataclass(frozen=True)
class StaleSupersessionIneligible:
    raw_id: str
    session_id: str
    logical_source_key: str
    reason: str


@dataclass(frozen=True)
class StaleSupersessionReissuePlan:
    """Read-only projection: what a reissue pass would do against the current head.

    ``already_current_count`` -- superseded raws whose existing receipt
    already matches the current head (nothing to do; already eligible for
    release under ``active_raw_retention_authority``).
    ``stale_count`` -- distinct superseded raws whose receipt does not match
    the current head and were evaluated for reissue.
    ``eligible`` -- the *complete* (never sampled) set a reissue would write
    a fresh receipt for -- callers writing receipts must use every item here,
    not just ``ineligible_samples``.
    ``ineligible_reason_counts`` -- exact per-reason counts for the stale
    raws reissue refuses; ``ineligible_samples`` carries a capped sample for
    operator inspection.
    """

    already_current_count: int
    stale_count: int
    eligible: tuple[StaleSupersessionCandidate, ...]
    ineligible_reason_counts: Mapping[str, int]
    ineligible_samples: tuple[StaleSupersessionIneligible, ...]


def plan_stale_supersession_reissue(
    source_conn: sqlite3.Connection,
    *,
    index_db_path: Path,
    sample_limit: int = 20,
) -> StaleSupersessionReissuePlan:
    """Read-only: find superseded receipts stale against the current head.

    A receipt is *stale* when its (session_id, accepted_raw_id,
    accepted_source_revision, accepted_content_hash, acquisition_generation,
    append_end_offset, decided_at_ms) tuple no longer matches the live
    ``raw_revision_heads`` row for its logical source -- the exact eight
    columns ``_active_index_raw_authority`` joins on (see
    ``_application_matches_head``).

    A stale raw is eligible for a fresh receipt only when *all* hold:

    * the current head's ``accepted_frontier_kind`` is ``'byte'`` (fail
      closed on semantic heads, matching the retention join -- this is a
      permanent restriction, not a placeholder to relax later; see the
      polylogue-hgsq rationale in ``_active_index_raw_authority`` above);
    * the current head's own raw is a byte-proven ``full`` reset -- the one
      case ``_validate_active_revision_chain`` documents as unconditionally
      superseding every other raw sharing the logical source, regardless of
      lineage (see the module comment above for why an in-progress append
      chain is deliberately never treated as proof of a *different* raw's
      supersession);
    * the candidate raw itself is a typed, byte-proven raw for that same
      logical source, distinct from the head's own raw.

    Every other outcome (missing head, mismatched session, non-full head,
    untyped/unproven raw, disagreeing source evidence) is recorded as
    ineligible with an exact reason and never produces a receipt.
    """
    original_row_factory = source_conn.row_factory
    source_conn.row_factory = sqlite3.Row
    try:
        heads = _read_current_raw_revision_heads(index_db_path)
        applications = _read_superseded_applications(index_db_path)

        groups: dict[tuple[str, str, str], list[_StaleSupersededApplication]] = {}
        for app in applications:
            groups.setdefault((app.raw_id, app.session_id, app.logical_source_key), []).append(app)

        already_current = 0
        stale: dict[tuple[str, str, str], _StaleSupersededApplication] = {}
        ineligible_counts: dict[str, int] = {}
        ineligible_samples: list[StaleSupersessionIneligible] = []

        def _mark_ineligible(raw_id: str, session_id: str, logical_source_key: str, reason: str) -> None:
            ineligible_counts[reason] = ineligible_counts.get(reason, 0) + 1
            if len(ineligible_samples) < sample_limit:
                ineligible_samples.append(
                    StaleSupersessionIneligible(
                        raw_id=raw_id, session_id=session_id, logical_source_key=logical_source_key, reason=reason
                    )
                )

        for (raw_id, session_id, logical_source_key), rows in groups.items():
            head = heads.get(logical_source_key)
            if head is None:
                _mark_ineligible(raw_id, session_id, logical_source_key, "no current head for logical source")
                continue
            if any(_application_matches_head(row, head) for row in rows):
                already_current += 1
                continue
            if head.accepted_frontier_kind != "byte":
                _mark_ineligible(raw_id, session_id, logical_source_key, "current head frontier_kind is not byte")
                continue
            if session_id != head.session_id:
                _mark_ineligible(raw_id, session_id, logical_source_key, "current head belongs to a different session")
                continue
            representative = max(rows, key=lambda row: row.decided_at_ms)
            stale[(raw_id, session_id, logical_source_key)] = representative

        # One lookup per distinct head raw, cached across every stale raw that
        # shares a logical source (typically many-to-one).
        head_row_cache: dict[str, sqlite3.Row | None] = {}
        eligible: list[StaleSupersessionCandidate] = []
        for (raw_id, session_id, logical_source_key), rep in stale.items():
            head = heads[logical_source_key]
            if logical_source_key not in head_row_cache:
                fetched = _raw_revision_rows(source_conn, {head.accepted_raw_id}, allow_missing=True)
                head_row_cache[logical_source_key] = fetched.get(head.accepted_raw_id)
            head_row = head_row_cache[logical_source_key]
            if head_row is None:
                _mark_ineligible(raw_id, session_id, logical_source_key, "current head raw is missing from source tier")
                continue
            if str(head_row["revision_kind"]) != "full":
                _mark_ineligible(
                    raw_id,
                    session_id,
                    logical_source_key,
                    "current head is not a byte-proven full reset (in-progress append chain)",
                )
                continue
            if str(head_row["revision_authority"]) != "byte_proven":
                _mark_ineligible(raw_id, session_id, logical_source_key, "current head lacks byte-proven authority")
                continue
            if raw_id == head.accepted_raw_id:
                _mark_ineligible(raw_id, session_id, logical_source_key, "raw is the current accepted head")
                continue
            candidate_rows = _raw_revision_rows(source_conn, {raw_id}, allow_missing=True)
            row = candidate_rows.get(raw_id)
            if row is None:
                _mark_ineligible(raw_id, session_id, logical_source_key, "raw is missing from source tier")
                continue
            if str(row["revision_kind"]) not in {"full", "append"}:
                _mark_ineligible(raw_id, session_id, logical_source_key, "raw lacks typed revision authority")
                continue
            if str(row["revision_authority"]) != "byte_proven":
                _mark_ineligible(raw_id, session_id, logical_source_key, "raw lacks byte-proven authority")
                continue
            if row["logical_source_key"] != logical_source_key:
                _mark_ineligible(raw_id, session_id, logical_source_key, "raw crosses logical sources")
                continue
            if str(row["source_revision"]) != rep.source_revision:
                _mark_ineligible(
                    raw_id,
                    session_id,
                    logical_source_key,
                    "source revision disagrees between application receipt and source tier",
                )
                continue
            eligible.append(
                StaleSupersessionCandidate(
                    raw_id=raw_id,
                    session_id=session_id,
                    logical_source_key=logical_source_key,
                    source_revision=rep.source_revision,
                    baseline_raw_id=str(row["baseline_raw_id"]) if row["baseline_raw_id"] is not None else None,
                    predecessor_raw_id=(
                        str(row["predecessor_raw_id"]) if row["predecessor_raw_id"] is not None else None
                    ),
                    current_head=head,
                )
            )
        return StaleSupersessionReissuePlan(
            already_current_count=already_current,
            stale_count=len(stale),
            eligible=tuple(eligible),
            ineligible_reason_counts=dict(ineligible_counts),
            ineligible_samples=tuple(ineligible_samples),
        )
    finally:
        source_conn.row_factory = original_row_factory


@dataclass(frozen=True)
class StaleSupersessionReissueResult:
    already_current_count: int
    stale_count: int
    eligible_count: int
    reissued_count: int
    ineligible_reason_counts: Mapping[str, int]
    ineligible_samples: tuple[StaleSupersessionIneligible, ...]
    errors: tuple[str, ...] = ()


def reissue_stale_supersession_receipts(
    source_conn: sqlite3.Connection,
    index_conn: sqlite3.Connection,
    *,
    index_db_path: Path,
    dry_run: bool = True,
    sample_limit: int = 20,
) -> StaleSupersessionReissueResult:
    """Re-issue a fresh supersession receipt for raws proven stale against the
    current head.

    Deletion-authority scope: this writes ONLY new ``raw_revision_applications``
    rows with ``decision = 'superseded'``. It never deletes a blob, never
    touches ``raw_revision_heads``, and never mutates an existing receipt row
    -- ``record_revision_application_sync`` is append-only / immutable by
    construction (a genuine identity conflict raises rather than overwriting).
    ``dry_run=True`` (the default) performs no writes at all; it only runs
    :func:`plan_stale_supersession_reissue` and reports what would happen.

    ``source_conn`` is the source-tier connection (read-only is sufficient:
    only ``raw_sessions`` chain rows are read). ``index_conn`` must be a
    writable index-tier connection when ``dry_run=False``, since
    ``raw_revision_applications`` is an index-tier table; it is not opened by
    this function so the caller controls transaction/commit scope.
    """
    plan = plan_stale_supersession_reissue(source_conn, index_db_path=index_db_path, sample_limit=sample_limit)
    reissued = 0
    errors: list[str] = []
    if not dry_run and plan.eligible:
        from polylogue.archive.revision_replay import ApplicationDecision
        from polylogue.storage.sqlite.archive_tiers.revision_application import (
            RevisionApplicationReceipt,
            record_revision_application_sync,
        )

        for item in plan.eligible:
            head = item.current_head
            receipt = RevisionApplicationReceipt(
                raw_id=item.raw_id,
                session_id=item.session_id,
                logical_source_key=item.logical_source_key,
                source_revision=item.source_revision,
                acquisition_generation=head.acquisition_generation,
                decision=ApplicationDecision.SUPERSEDED,
                accepted_raw_id=head.accepted_raw_id,
                accepted_source_revision=head.accepted_source_revision,
                accepted_content_hash=head.accepted_content_hash,
                accepted_frontier_kind=head.accepted_frontier_kind,
                accepted_frontier=head.accepted_frontier,
                baseline_raw_id=item.baseline_raw_id,
                predecessor_raw_id=item.predecessor_raw_id,
                append_end_offset=head.append_end_offset,
                detail=f"stale_supersession_reissue:current_head={head.accepted_raw_id}",
            )
            try:
                record_revision_application_sync(index_conn, receipt, decided_at_ms=head.decided_at_ms)
            except (RuntimeError, ValueError) as exc:
                errors.append(f"{item.raw_id[:16]}: {exc}")
            else:
                reissued += 1
        index_conn.commit()
    return StaleSupersessionReissueResult(
        already_current_count=plan.already_current_count,
        stale_count=plan.stale_count,
        eligible_count=len(plan.eligible),
        reissued_count=reissued,
        ineligible_reason_counts=plan.ineligible_reason_counts,
        ineligible_samples=plan.ineligible_samples,
        errors=tuple(errors),
    )


# ---------------------------------------------------------------------------
# Raw-frontier integrity readiness (polylogue-yla8.7)
# ---------------------------------------------------------------------------
#
# ``active_raw_retention_authority`` above answers "what may retention delete
# right now?" and fails *closed* (raises) the moment it finds one broken
# chain, because that is the correct behaviour for a cleanup gate. Ordinary
# process health and raw-materialization candidate counts never call it, so a
# broken accepted append head or a cursor that has committed past the
# material actually accepted into the index can sit invisible until an
# operator runs manual SQL (as happened before yla8.6). This section reuses
# the exact same source binding and chain validators
# (``_validate_byte_head`` plus ``_validate_active_revision_chain``) to build
# a *reporting* projection instead: it never raises on a per-seed violation,
# it counts and samples it, so readiness and retention safety cannot drift.

RawFrontierIntegrityStatus = Literal["healthy", "unknown", "violated"]
CursorAuthorityGapState = Literal[
    "deferred",
    "source_raws_without_accepted_head",
    "cursor_path_absent_from_source",
    "accepted_head_missing_source",
]
"""Typed check outcome. ``"unknown"`` means the check's authority tier could
not be read — it is never collapsed into a false ``"healthy"`` zero."""


@dataclass(frozen=True)
class BrokenAppendHeadSample:
    """One active index raw seed whose revision chain failed validation.

    Accepted heads are the usual seed, so the wire field remains
    ``accepted_raw_id``. ``sessions.raw_id`` is an equally load-bearing
    retention seed and is reported here when no head references the same raw.
    """

    logical_source_key: str
    accepted_raw_id: str
    reason: str


@dataclass(frozen=True)
class CursorAheadSample:
    """One cursor with at least one accepted byte head behind its frontier."""

    source_path: str
    logical_source_key: str
    cursor_byte_offset: int
    accepted_frontier: int
    affected_head_count: int


@dataclass(frozen=True)
class CursorAuthorityGapSample:
    """One cursor/head that cannot be joined to comparable byte authority."""

    state: CursorAuthorityGapState
    source_path: str | None
    logical_source_key: str | None
    cursor_byte_offset: int | None
    reason: str


@dataclass(frozen=True)
class _OpsCursorAuthority:
    source_path: str
    byte_offset: int
    deferred_end_offset: int | None

    @property
    def is_deferred(self) -> bool:
        return self.deferred_end_offset is not None and self.deferred_end_offset > self.byte_offset


@dataclass(frozen=True)
class RawFrontierIntegritySnapshot:
    """Two of the three yla8.7 raw-frontier integrity facts.

    The third fact — an index ``sessions.raw_id`` absent from
    ``source.raw_sessions`` — is intentionally **not** recomputed here: it is
    already exact-projected by
    :func:`polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot`
    as ``lost_source_evidence_count`` / ``lost_source_evidence_samples`` and
    already gates ``raw_materialization_ready()`` / claim-guard ``converged``.
    Callers compose that existing signal alongside this snapshot instead of
    re-querying it (no duplicated SQL/semantics).
    """

    broken_head_status: RawFrontierIntegrityStatus
    broken_head_count: int
    broken_head_checked_count: int
    broken_head_samples: tuple[BrokenAppendHeadSample, ...]
    broken_head_reason: str

    cursor_ahead_status: RawFrontierIntegrityStatus
    cursor_ahead_count: int
    cursor_ahead_checked_count: int
    cursor_head_comparison_count: int
    cursor_ahead_comparison_count: int
    cursor_ahead_samples: tuple[CursorAheadSample, ...]
    cursor_authority_gap_count: int
    cursor_authority_gap_samples: tuple[CursorAuthorityGapSample, ...]
    cursor_ahead_reason: str

    @property
    def overall_status(self) -> RawFrontierIntegrityStatus:
        return combine_raw_frontier_integrity_statuses(self.broken_head_status, self.cursor_ahead_status)


@dataclass(frozen=True)
class RawFrontierIntegrityProjection:
    """Canonical three-signal status projection shared by every surface."""

    available: bool
    overall_status: RawFrontierIntegrityStatus
    broken_head_status: RawFrontierIntegrityStatus
    broken_head_count: int
    broken_head_checked_count: int
    broken_head_samples: tuple[BrokenAppendHeadSample, ...]
    broken_head_reason: str
    missing_source_raw_status: RawFrontierIntegrityStatus
    missing_source_raw_count: int
    missing_source_raw_samples: tuple[Mapping[str, object], ...]
    missing_source_raw_reason: str
    cursor_ahead_status: RawFrontierIntegrityStatus
    cursor_ahead_count: int
    cursor_ahead_checked_count: int
    cursor_head_comparison_count: int
    cursor_ahead_comparison_count: int
    cursor_ahead_samples: tuple[CursorAheadSample, ...]
    cursor_authority_gap_count: int
    cursor_authority_gap_samples: tuple[CursorAuthorityGapSample, ...]
    cursor_ahead_reason: str

    @property
    def summary(self) -> str:
        """Canonical operator summary shared by every readiness surface."""

        return raw_frontier_integrity_summary(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "overall_status": self.overall_status,
            "broken_head_status": self.broken_head_status,
            "broken_head_count": self.broken_head_count,
            "broken_head_checked_count": self.broken_head_checked_count,
            "broken_head_samples": [
                {
                    "logical_source_key": sample.logical_source_key,
                    "accepted_raw_id": sample.accepted_raw_id,
                    "reason": sample.reason,
                }
                for sample in self.broken_head_samples
            ],
            "broken_head_reason": self.broken_head_reason,
            "missing_source_raw_status": self.missing_source_raw_status,
            "missing_source_raw_count": self.missing_source_raw_count,
            "missing_source_raw_samples": [dict(sample) for sample in self.missing_source_raw_samples],
            "missing_source_raw_reason": self.missing_source_raw_reason,
            "cursor_ahead_status": self.cursor_ahead_status,
            "cursor_ahead_count": self.cursor_ahead_count,
            "cursor_ahead_checked_count": self.cursor_ahead_checked_count,
            "cursor_head_comparison_count": self.cursor_head_comparison_count,
            "cursor_ahead_comparison_count": self.cursor_ahead_comparison_count,
            "cursor_ahead_samples": [
                {
                    "source_path": sample.source_path,
                    "logical_source_key": sample.logical_source_key,
                    "cursor_byte_offset": sample.cursor_byte_offset,
                    "accepted_frontier": sample.accepted_frontier,
                    "affected_head_count": sample.affected_head_count,
                }
                for sample in self.cursor_ahead_samples
            ],
            "cursor_authority_gap_count": self.cursor_authority_gap_count,
            "cursor_authority_gap_samples": [
                {
                    "state": sample.state,
                    "source_path": sample.source_path,
                    "logical_source_key": sample.logical_source_key,
                    "cursor_byte_offset": sample.cursor_byte_offset,
                    "reason": sample.reason,
                }
                for sample in self.cursor_authority_gap_samples
            ],
            "cursor_ahead_reason": self.cursor_ahead_reason,
        }


def combine_raw_frontier_integrity_statuses(
    *statuses: RawFrontierIntegrityStatus,
) -> RawFrontierIntegrityStatus:
    """Combine independent checks without hiding a proven violation.

    A known violation dominates an unavailable sibling check. Unknown still
    dominates healthy, so incomplete authority never renders green.
    """

    if "violated" in statuses:
        return "violated"
    if "unknown" in statuses:
        return "unknown"
    return "healthy"


def raw_frontier_integrity_summary(
    integrity: Mapping[str, object] | RawFrontierIntegrityProjection,
) -> str:
    """Return one canonical claim/readiness summary for the projection.

    Both daemon and direct status adapt their typed payloads to this helper so
    reason ordering and mixed violated/unknown reporting cannot drift between
    surfaces.
    """

    overall_status: str
    reasons: tuple[str, ...]
    if isinstance(integrity, RawFrontierIntegrityProjection):
        overall_status = integrity.overall_status
        reasons = (
            integrity.broken_head_reason,
            integrity.missing_source_raw_reason,
            integrity.cursor_ahead_reason,
        )
    else:
        overall_status = str(integrity.get("overall_status") or "unknown")
        reasons = tuple(
            str(integrity.get(key) or "")
            for key in ("broken_head_reason", "missing_source_raw_reason", "cursor_ahead_reason")
        )
    if overall_status == "healthy":
        return "ready"
    rendered = tuple(reason for reason in reasons if reason)
    return "; ".join(rendered) if rendered else "raw frontier integrity unavailable"


def missing_source_raw_integrity_status(
    raw_materialization_readiness: Mapping[str, object],
    *,
    sample_limit: int | None = None,
) -> tuple[RawFrontierIntegrityStatus, int, tuple[Mapping[str, object], ...], str]:
    """Project the existing lost-source-evidence signal into this vocabulary."""

    available = bool(raw_materialization_readiness.get("available"))
    count_value = raw_materialization_readiness.get("lost_source_evidence_count")
    count = int(count_value) if isinstance(count_value, (bool, int, float, str)) else 0
    raw_samples = raw_materialization_readiness.get("lost_source_evidence_samples")
    all_samples = (
        tuple(sample for sample in raw_samples if isinstance(sample, Mapping)) if isinstance(raw_samples, list) else ()
    )
    samples = all_samples if sample_limit is None else all_samples[: max(0, sample_limit)]
    if not available:
        return "unknown", count, samples, "raw materialization readiness unavailable"
    if count:
        return (
            "violated",
            count,
            samples,
            f"{count} indexed session(s) reference raw evidence missing from source tier",
        )
    return "healthy", 0, samples, ""


def raw_frontier_integrity_projection(
    archive_root: Path,
    raw_materialization_readiness: Mapping[str, object],
    *,
    sample_limit: int = 10,
) -> RawFrontierIntegrityProjection:
    """Build the canonical split-tier projection consumed by all surfaces."""

    missing_status, missing_count, missing_samples, missing_reason = missing_source_raw_integrity_status(
        raw_materialization_readiness,
        sample_limit=sample_limit,
    )
    try:
        index_db_path = resolve_active_index_path(archive_root)
    except ArchiveLocationError as exc:
        return unknown_raw_frontier_integrity_projection(f"active index pointer unavailable: {exc}")
    source_db_path = archive_root / "source.db"
    ops_db_path = archive_root / "ops.db"
    snapshot = _unavailable_frontier_integrity_snapshot(f"source tier is unavailable: {source_db_path}")

    if source_db_path.is_file():
        try:
            from polylogue.storage.sqlite.connection_profile import open_readonly_connection

            conn = open_readonly_connection(source_db_path)
        except (OSError, sqlite3.Error) as exc:
            logger.warning("raw frontier integrity: source tier is unreadable: %s", exc)
            snapshot = _unavailable_frontier_integrity_snapshot(f"source tier is unreadable: {exc}")
        else:
            try:
                snapshot = raw_frontier_integrity_snapshot(
                    conn,
                    index_db_path=index_db_path,
                    ops_db_path=ops_db_path,
                    sample_limit=sample_limit,
                )
            finally:
                conn.close()

    statuses = (snapshot.broken_head_status, missing_status, snapshot.cursor_ahead_status)
    overall_status = combine_raw_frontier_integrity_statuses(*statuses)
    return RawFrontierIntegrityProjection(
        available="unknown" not in statuses,
        overall_status=overall_status,
        broken_head_status=snapshot.broken_head_status,
        broken_head_count=snapshot.broken_head_count,
        broken_head_checked_count=snapshot.broken_head_checked_count,
        broken_head_samples=snapshot.broken_head_samples,
        broken_head_reason=snapshot.broken_head_reason,
        missing_source_raw_status=missing_status,
        missing_source_raw_count=missing_count,
        missing_source_raw_samples=missing_samples,
        missing_source_raw_reason=missing_reason,
        cursor_ahead_status=snapshot.cursor_ahead_status,
        cursor_ahead_count=snapshot.cursor_ahead_count,
        cursor_ahead_checked_count=snapshot.cursor_ahead_checked_count,
        cursor_head_comparison_count=snapshot.cursor_head_comparison_count,
        cursor_ahead_comparison_count=snapshot.cursor_ahead_comparison_count,
        cursor_ahead_samples=snapshot.cursor_ahead_samples,
        cursor_authority_gap_count=snapshot.cursor_authority_gap_count,
        cursor_authority_gap_samples=snapshot.cursor_authority_gap_samples,
        cursor_ahead_reason=snapshot.cursor_ahead_reason,
    )


def unknown_raw_frontier_integrity_projection(reason: str) -> RawFrontierIntegrityProjection:
    """Return the canonical explicit-unknown projection for an unavailable read.

    Cache and presentation adapters use this instead of inventing partial
    legacy payloads. Zero-valued counts are not healthy claims because every
    check is explicitly ``unknown`` and ``available`` is false.
    """

    snapshot = _unavailable_frontier_integrity_snapshot(reason)
    return RawFrontierIntegrityProjection(
        available=False,
        overall_status="unknown",
        broken_head_status=snapshot.broken_head_status,
        broken_head_count=snapshot.broken_head_count,
        broken_head_checked_count=snapshot.broken_head_checked_count,
        broken_head_samples=snapshot.broken_head_samples,
        broken_head_reason=snapshot.broken_head_reason,
        missing_source_raw_status="unknown",
        missing_source_raw_count=0,
        missing_source_raw_samples=(),
        missing_source_raw_reason=reason,
        cursor_ahead_status=snapshot.cursor_ahead_status,
        cursor_ahead_count=snapshot.cursor_ahead_count,
        cursor_ahead_checked_count=snapshot.cursor_ahead_checked_count,
        cursor_head_comparison_count=snapshot.cursor_head_comparison_count,
        cursor_ahead_comparison_count=snapshot.cursor_ahead_comparison_count,
        cursor_ahead_samples=snapshot.cursor_ahead_samples,
        cursor_authority_gap_count=snapshot.cursor_authority_gap_count,
        cursor_authority_gap_samples=snapshot.cursor_authority_gap_samples,
        cursor_ahead_reason=snapshot.cursor_ahead_reason,
    )


def raw_frontier_integrity_snapshot(
    conn: sqlite3.Connection,
    *,
    index_db_path: Path,
    ops_db_path: Path,
    sample_limit: int = 10,
) -> RawFrontierIntegritySnapshot:
    """Read-only substrate integrity projection over source/index/ops.

    Reports two authority gaps that ordinary process health and
    raw-materialization candidate counts cannot see on their own
    (polylogue-yla8.7):

    * ``broken_head`` — a distinct active raw seed from either
      ``index.sessions.raw_id`` or ``index.raw_revision_heads`` whose
      transitive predecessor chain is missing or invalid. Uses the exact same
      :func:`_validate_active_revision_chain` that
      :func:`active_raw_retention_authority` uses to protect retention, so
      cleanup safety and readiness visibility cannot drift.
    * ``cursor_ahead`` — an ``ops.ingest_cursor`` committed byte frontier past
      the byte frontier actually accepted into the index for that logical
      source — the exact symptom yla8.6 found only via manual SQL.

    Each check independently degrades to ``"unknown"`` (never a false
    ``"healthy"`` zero) when its authority tier cannot be read.

    Exact totals deliberately inspect every current index seed and committed,
    non-excluded cursor; only samples are capped. On the 2026-07-12 live
    archive this covered 17,619 distinct seeds in 1130.902 ms cold and
    266.659/276.331 ms warm. A cardinality cap would make ordinary status
    cheaper by hiding unchecked authority, so this projection records
    empirical boundedness rather than inventing a false-green permanent cap.
    """
    original_row_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        try:
            session_raw_ids, heads, _eligible = _active_index_raw_authority(index_db_path)
        except RawRetentionSafetyError as exc:
            return _unavailable_frontier_integrity_snapshot(str(exc))

        source_reason = _source_tier_unavailable_reason(conn)
        if source_reason is not None:
            return _unavailable_frontier_integrity_snapshot(source_reason)

        broken_status, broken_count, broken_checked, broken_samples, broken_reason = _check_broken_active_chains(
            conn,
            session_raw_ids,
            heads,
            sample_limit=sample_limit,
        )
        (
            cursor_status,
            cursor_count,
            cursor_checked,
            cursor_comparisons,
            cursor_ahead_comparisons,
            cursor_samples,
            cursor_gap_count,
            cursor_gap_samples,
            cursor_reason,
        ) = _check_cursor_ahead_of_accepted(conn, ops_db_path, heads, sample_limit=sample_limit)
        return RawFrontierIntegritySnapshot(
            broken_head_status=broken_status,
            broken_head_count=broken_count,
            broken_head_checked_count=broken_checked,
            broken_head_samples=broken_samples,
            broken_head_reason=broken_reason,
            cursor_ahead_status=cursor_status,
            cursor_ahead_count=cursor_count,
            cursor_ahead_checked_count=cursor_checked,
            cursor_head_comparison_count=cursor_comparisons,
            cursor_ahead_comparison_count=cursor_ahead_comparisons,
            cursor_ahead_samples=cursor_samples,
            cursor_authority_gap_count=cursor_gap_count,
            cursor_authority_gap_samples=cursor_gap_samples,
            cursor_ahead_reason=cursor_reason,
        )
    finally:
        conn.row_factory = original_row_factory


def _unavailable_frontier_integrity_snapshot(reason: str) -> RawFrontierIntegritySnapshot:
    return RawFrontierIntegritySnapshot(
        broken_head_status="unknown",
        broken_head_count=0,
        broken_head_checked_count=0,
        broken_head_samples=(),
        broken_head_reason=reason,
        cursor_ahead_status="unknown",
        cursor_ahead_count=0,
        cursor_ahead_checked_count=0,
        cursor_head_comparison_count=0,
        cursor_ahead_comparison_count=0,
        cursor_ahead_samples=(),
        cursor_authority_gap_count=0,
        cursor_authority_gap_samples=(),
        cursor_ahead_reason=reason,
    )


def _source_tier_unavailable_reason(conn: sqlite3.Connection) -> str | None:
    try:
        columns = {str(row[1]) for row in conn.execute("PRAGMA table_xinfo(raw_sessions)").fetchall()}
        missing = set(_RAW_REVISION_CHAIN_COLUMN_NAMES).difference(columns)
        if missing:
            rendered = ", ".join(sorted(missing))
            return f"source raw revision authority is unreadable: schema missing column(s): {rendered}"
        conn.execute(f"SELECT {_RAW_REVISION_CHAIN_COLUMNS} FROM raw_sessions LIMIT 0")
    except sqlite3.Error as exc:
        logger.warning("raw frontier integrity: source revision authority is unreadable: %s", exc)
        return f"source raw revision authority is unreadable: {exc}"
    return None


def _check_broken_active_chains(
    conn: sqlite3.Connection,
    session_raw_ids: frozenset[str],
    heads: tuple[_IndexRawRevisionHead, ...],
    *,
    sample_limit: int,
) -> tuple[RawFrontierIntegrityStatus, int, int, tuple[BrokenAppendHeadSample, ...], str]:
    """Validate every distinct retention seed against one complete authority read."""

    samples: list[BrokenAppendHeadSample] = []
    broken_count = 0
    heads_by_raw_id: dict[str, list[_IndexRawRevisionHead]] = {}
    for head in heads:
        heads_by_raw_id.setdefault(head.accepted_raw_id, []).append(head)
    seed_raw_ids = set(session_raw_ids).union(heads_by_raw_id)
    # Membership-governed snapshots carry a semantic head, not a byte
    # predecessor chain. They remain active source authority and must be
    # retained, but applying byte-chain validation to them turns a normal
    # membership snapshot into a false broken-head violation. A raw selected
    # by both regimes remains byte-validated.
    byte_head_raw_ids = {head.accepted_raw_id for head in heads if head.accepted_frontier_kind == "byte"}
    semantic_only_raw_ids = {
        head.accepted_raw_id for head in heads if head.accepted_frontier_kind != "byte"
    }.difference(byte_head_raw_ids)
    try:
        rows_by_id = _raw_revision_rows(conn, seed_raw_ids, allow_missing=True)
    except _RawRevisionAuthorityUnavailableError as exc:
        logger.warning("raw frontier integrity: %s", exc)
        return "unknown", 0, 0, (), str(exc)

    checked_count = 0
    for seed_raw_id in sorted(seed_raw_ids):
        seed_heads = heads_by_raw_id.get(seed_raw_id, [])
        row = rows_by_id.get(seed_raw_id)
        if row is None and not seed_heads:
            # Directly missing sessions.raw_id rows are counted once by the
            # canonical lost-source-evidence projection. There is no chain to
            # traverse here; do not double-count the same absence.
            continue
        checked_count += 1
        try:
            if row is None:
                raise RawRetentionSafetyError(f"active index raw is missing from source tier: {seed_raw_id}")
            if seed_raw_id in semantic_only_raw_ids:
                continue
            for head in seed_heads:
                if head.accepted_frontier_kind == "byte":
                    _validate_byte_head(row, head)
            _validate_active_revision_chain(rows_by_id, seed_raw_id)
        except RawRetentionSafetyError as exc:
            broken_count += 1
            if len(samples) < sample_limit:
                if seed_heads:
                    logical_source_key = seed_heads[0].logical_source_key
                elif row is not None:
                    logical_source_key = str(row["logical_source_key"] or "session.raw_id")
                else:
                    logical_source_key = "session.raw_id"
                samples.append(
                    BrokenAppendHeadSample(
                        logical_source_key=logical_source_key,
                        accepted_raw_id=seed_raw_id,
                        reason=str(exc),
                    )
                )
    status: RawFrontierIntegrityStatus = "violated" if broken_count else "healthy"
    reason = (
        ""
        if broken_count == 0
        else f"{broken_count} active index raw seed(s) have a broken predecessor chain or invalid source binding"
    )
    return status, broken_count, checked_count, tuple(samples), reason


def _check_cursor_ahead_of_accepted(
    conn: sqlite3.Connection,
    ops_db_path: Path,
    heads: tuple[_IndexRawRevisionHead, ...],
    *,
    sample_limit: int,
) -> tuple[
    RawFrontierIntegrityStatus,
    int,
    int,
    int,
    int,
    tuple[CursorAheadSample, ...],
    int,
    tuple[CursorAuthorityGapSample, ...],
    str,
]:
    try:
        cursor_map = _ops_cursor_byte_offsets(ops_db_path)
    except RawRetentionSafetyError as exc:
        return "unknown", 0, 0, 0, 0, (), 0, (), str(exc)

    try:
        source_path_by_raw_id = _source_paths_for_raw_ids(conn, {head.accepted_raw_id for head in heads})
    except sqlite3.Error as exc:
        logger.warning("raw frontier integrity: source raw path lookup failed: %s", exc)
        return "unknown", 0, 0, 0, 0, (), 0, (), f"source raw path lookup failed: {exc}"

    byte_heads_by_path: dict[str, list[_IndexRawRevisionHead]] = {}
    all_head_paths: set[str] = set()
    gaps: list[CursorAuthorityGapSample] = []
    gap_count = 0
    for head in heads:
        source_path = source_path_by_raw_id.get(head.accepted_raw_id)
        if source_path is None:
            if head.accepted_frontier_kind == "byte":
                gap_count += 1
                if len(gaps) < sample_limit:
                    gaps.append(
                        CursorAuthorityGapSample(
                            state="accepted_head_missing_source",
                            source_path=None,
                            logical_source_key=head.logical_source_key,
                            cursor_byte_offset=None,
                            reason=f"accepted byte head raw is absent from source tier: {head.accepted_raw_id}",
                        )
                    )
            continue
        all_head_paths.add(source_path)
        if head.accepted_frontier_kind == "byte":
            byte_heads_by_path.setdefault(source_path, []).append(head)

    samples: list[CursorAheadSample] = []
    ahead_count = 0
    checked = 0
    comparison_count = 0
    ahead_comparison_count = 0
    try:
        source_paths = _source_paths_for_paths(conn, set(cursor_map))
    except sqlite3.Error as exc:
        logger.warning("raw frontier integrity: cursor source path lookup failed: %s", exc)
        return "unknown", 0, 0, 0, 0, (), 0, (), f"cursor source path lookup failed: {exc}"
    try:
        terminal_artifact_paths = _terminal_artifact_paths(conn, set(cursor_map))
    except sqlite3.Error as exc:
        logger.warning("raw frontier integrity: terminal artifact authority lookup failed: %s", exc)
        return "unknown", 0, 0, 0, 0, (), 0, (), f"terminal artifact authority is unreadable: {exc}"
    for path, cursor in cursor_map.items():
        cursor_offset = cursor.byte_offset
        if cursor.is_deferred:
            gap_count += 1
            if len(gaps) < sample_limit:
                gaps.append(
                    CursorAuthorityGapSample(
                        state="deferred",
                        source_path=path,
                        logical_source_key=None,
                        cursor_byte_offset=cursor_offset,
                        reason=(
                            "ingest cursor has durably captured material awaiting authority resolution "
                            f"through byte {cursor.deferred_end_offset}"
                        ),
                    )
                )
            continue
        comparable_heads = byte_heads_by_path.get(path)
        if not comparable_heads:
            # A path governed exclusively by membership authority has no
            # comparable byte frontier and is intentionally out of scope.
            if path in all_head_paths or path in terminal_artifact_paths:
                continue
            gap_count += 1
            if len(gaps) < sample_limit:
                gaps.append(
                    CursorAuthorityGapSample(
                        state=(
                            "source_raws_without_accepted_head"
                            if path in source_paths
                            else "cursor_path_absent_from_source"
                        ),
                        source_path=path,
                        logical_source_key=None,
                        cursor_byte_offset=cursor_offset,
                        reason=(
                            "source tier has raw evidence but index has no accepted byte head"
                            if path in source_paths
                            else "ingest cursor path is absent from source tier"
                        ),
                    )
                )
            continue
        checked += 1
        comparison_count += len(comparable_heads)
        ahead_heads = [head for head in comparable_heads if cursor_offset > head.accepted_frontier]
        if not ahead_heads:
            continue
        ahead_count += 1
        ahead_comparison_count += len(ahead_heads)
        if len(samples) < sample_limit:
            representative = min(ahead_heads, key=lambda head: (head.accepted_frontier, head.logical_source_key))
            samples.append(
                CursorAheadSample(
                    source_path=path,
                    logical_source_key=representative.logical_source_key,
                    cursor_byte_offset=cursor_offset,
                    accepted_frontier=representative.accepted_frontier,
                    affected_head_count=len(ahead_heads),
                )
            )

    status: RawFrontierIntegrityStatus = "violated" if ahead_count else "unknown" if gap_count else "healthy"
    reasons: list[str] = []
    if ahead_count:
        reasons.append(
            f"{ahead_count} ingest cursor row(s) committed past accepted raw material "
            f"across {ahead_comparison_count} cursor/head comparison(s)"
        )
    if gap_count:
        reasons.append(f"{gap_count} cursor/head authority row(s) could not be compared")
    return (
        status,
        ahead_count,
        checked,
        comparison_count,
        ahead_comparison_count,
        tuple(samples),
        gap_count,
        tuple(gaps),
        "; ".join(reasons),
    )


def _source_paths_for_raw_ids(conn: sqlite3.Connection, raw_ids: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    pending = set(raw_ids)
    while pending:
        batch = tuple(sorted(pending)[:500])
        pending.difference_update(batch)
        placeholders = ", ".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT raw_id, source_path FROM raw_sessions WHERE raw_id IN ({placeholders})", batch
        ).fetchall()
        for row in rows:
            result[str(row[0])] = str(row[1])
    return result


def _source_paths_for_paths(conn: sqlite3.Connection, source_paths: set[str]) -> set[str]:
    result: set[str] = set()
    pending = set(source_paths)
    while pending:
        batch = tuple(sorted(pending)[:500])
        pending.difference_update(batch)
        placeholders = ", ".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT DISTINCT source_path FROM raw_sessions WHERE source_path IN ({placeholders})", batch
        ).fetchall()
        result.update(str(row[0]) for row in rows if row[0] is not None)
    return result


def _terminal_artifact_paths(conn: sqlite3.Connection, source_paths: set[str]) -> set[str]:
    """Return paths whose every current source coordinate is terminal evidence.

    A full-route cursor can legitimately advance over a workflow/fact artifact
    that has no session head. ``raw_artifacts.parse_as_session = 0`` is the
    source-tier terminal authority for that case. Ordinary artifact upserts
    retain the source coordinate's latest receipt while ``raw_sessions``
    retains its historical acquisition evidence, so authority attaches to each
    coordinate's newest raw observation rather than requiring a duplicate
    receipt on every historical raw. A failure-kind carrier remains authority
    only while that raw's current parse or validation state is failed; a later
    successful reparse makes the retained carrier historical evidence. Every
    ``(origin, source_index)`` member of a physical path must be terminal before
    the cursor path is exempt.
    """

    result: set[str] = set()
    raw_failure_kinds = tuple(sorted(RAW_FAILURE_EVIDENCE_KINDS))
    terminal_raw_failure_kinds = tuple(sorted(_TERMINAL_RAW_FAILURE_EVIDENCE_KINDS))
    raw_failure_placeholders = ", ".join("?" for _ in raw_failure_kinds)
    terminal_raw_failure_placeholders = ", ".join("?" for _ in terminal_raw_failure_kinds)
    path_batch_size = max(1, 500 - len(raw_failure_kinds) - len(terminal_raw_failure_kinds))
    pending = set(source_paths)
    while pending:
        batch = tuple(sorted(pending)[:path_batch_size])
        pending.difference_update(batch)
        placeholders = ", ".join("?" for _ in batch)
        rows = conn.execute(
            f"""
            WITH newest_per_coordinate AS (
                SELECT raw_id, source_path, origin, source_index, parse_error,
                       validation_status, validated_at_ms, parsed_at_ms
                FROM (
                    SELECT
                        raw_id,
                        source_path,
                        origin,
                        source_index,
                        parse_error,
                        validation_status,
                        validated_at_ms,
                        parsed_at_ms,
                        ROW_NUMBER() OVER (
                            PARTITION BY source_path, origin, source_index
                            ORDER BY acquired_at_ms DESC, rowid DESC
                        ) AS coordinate_rank
                    FROM raw_sessions
                    WHERE source_path IN ({placeholders})
                )
                WHERE coordinate_rank = 1
            ),
            terminal_artifacts AS (
                SELECT artifact.raw_id
                FROM raw_artifacts AS artifact
                JOIN newest_per_coordinate AS evidence_raw ON evidence_raw.raw_id = artifact.raw_id
                WHERE artifact.parse_as_session = 0
                  AND (
                      (
                          artifact.artifact_kind NOT IN ({raw_failure_placeholders})
                          AND (
                              evidence_raw.parsed_at_ms IS NULL
                              OR artifact.last_observed_at_ms >= evidence_raw.parsed_at_ms
                          )
                      )
                      OR (
                          artifact.artifact_kind IN ({terminal_raw_failure_placeholders})
                          AND (
                              evidence_raw.parse_error IS NOT NULL
                              OR (
                                  evidence_raw.validation_status = 'failed'
                                  AND (
                                      evidence_raw.parsed_at_ms IS NULL
                                      OR evidence_raw.validated_at_ms IS NULL
                                      -- A legacy tie has no proven winner;
                                      -- retain it rather than deleting raw
                                      -- authority based on an arbitrary side.
                                      OR evidence_raw.validated_at_ms >= evidence_raw.parsed_at_ms
                                  )
                              )
                          )
                      )
                  )
            )
            SELECT DISTINCT terminal_raw.source_path
            FROM terminal_artifacts AS artifact
            JOIN newest_per_coordinate AS terminal_raw ON terminal_raw.raw_id = artifact.raw_id
            WHERE NOT EXISTS (
                SELECT 1
                FROM newest_per_coordinate AS coordinate
                WHERE coordinate.source_path = terminal_raw.source_path
                  AND NOT EXISTS (
                      SELECT 1
                      FROM terminal_artifacts AS current_artifact
                      WHERE current_artifact.raw_id = coordinate.raw_id
                  )
            )
            """,
            (*batch, *raw_failure_kinds, *terminal_raw_failure_kinds),
        ).fetchall()
        result.update(str(row[0]) for row in rows)
    return result


def _terminal_artifact_raw_ids(
    conn: sqlite3.Connection,
    *,
    source_paths: Iterable[Path] | None = None,
) -> frozenset[str]:
    """Return all retained raw evidence for paths with terminal current observations."""

    selected_paths = (
        {str(row[0]) for row in conn.execute("SELECT DISTINCT source_path FROM raw_sessions").fetchall()}
        if source_paths is None
        else {str(path) for path in source_paths}
    )
    terminal_paths = _terminal_artifact_paths(conn, selected_paths)
    if not terminal_paths:
        return frozenset()
    raw_ids: set[str] = set()
    pending = set(terminal_paths)
    while pending:
        batch = tuple(sorted(pending)[:500])
        pending.difference_update(batch)
        placeholders = ", ".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT raw_id FROM raw_sessions WHERE source_path IN ({placeholders})",
            batch,
        ).fetchall()
        raw_ids.update(str(row[0]) for row in rows)
    return frozenset(raw_ids)


def _ops_cursor_byte_offsets(ops_db_path: Path) -> dict[str, _OpsCursorAuthority]:
    if not ops_db_path.is_file():
        raise RawRetentionSafetyError(f"ops tier is unavailable: {ops_db_path}")
    try:
        uri = f"{ops_db_path.resolve().as_uri()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.execute("PRAGMA query_only = ON")
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'ingest_cursor'"
            ).fetchone()
            if has_table is None:
                raise RawRetentionSafetyError(f"ops tier has no ingest_cursor table: {ops_db_path}")
            # Excluded cursors are quarantined inputs, not active committed
            # ingest authority. Their exclusion/failure state is surfaced by
            # the ordinary cursor workload status instead of frontier parity.
            rows = conn.execute(
                """
                SELECT source_path, byte_offset, deferred_end_offset
                FROM ingest_cursor
                WHERE COALESCE(excluded, 0) = 0 AND byte_offset IS NOT NULL
                """,
            ).fetchall()
    except (OSError, sqlite3.Error) as exc:
        raise RawRetentionSafetyError(f"ops tier raw cursor authority is unreadable: {exc}") from exc
    return {
        str(row[0]): _OpsCursorAuthority(
            source_path=str(row[0]),
            byte_offset=int(row[1]),
            deferred_end_offset=int(row[2]) if row[2] is not None else None,
        )
        for row in rows
        if row[1] is not None
    }


# ---------------------------------------------------------------------------
# Orphaned raw-authority plan cleanup (polylogue-i3zo)
# ---------------------------------------------------------------------------
#
# raw_authority_plans references raws by JSON string (input_raw_ids_json),
# not by FK, so deleting a raw_sessions row here does NOT cascade-clean a
# frontier plan naming it. A plan whose every input raw is gone and which no
# census still references is unprocessable and must be pruned in the same
# transaction as the raw delete, or the row simply accumulates forever
# (bounded growth was the original motivation; see polylogue-i3zo).
#
# CENSUS-SAFETY (revised after a real review finding on the first version of
# this fix, chatgpt-codex-connector on PR #3530): the first draft deleted
# `raw_authority_census_plans` / `raw_authority_census_post_plans` /
# `raw_authority_blockers` rows for any orphaned plan, unconditionally. That
# is wrong: `raw_authority_censuses.plan_count` / `post_plan_count` are
# immutable per-census totals read verbatim by `read_raw_authority_census`
# (`polylogue/storage/raw_authority.py`) to report a census's total AND to
# decide whether pagination has more pages (`next_offset < max(total,
# post_total)`). Deleting a census's plan-membership rows without adjusting
# those counts makes the total lie, and once a census's rows are ALL
# removed, `next_offset` (computed from `len(plans)`, now always 0) stops
# advancing -- `next_query_handle` is reissued forever. Deleting an
# unresolved `raw_authority_blockers` row is worse: `prune_raw_authority_census_history`
# (same module) already treats an unresolved blocker as "a live durable
# obligation, not history" and refuses to touch it for exactly this reason.
#
# The fix: never delete `raw_authority_census_plans` / `_post_plans` /
# `raw_authority_blockers` from this function at all -- only ever delete a
# `raw_authority_plans` row once it has ZERO remaining references in any of
# those three tables. This composes with, rather than fights, the existing
# retention path: every `raw_authority_plans` row is inserted in the same
# transaction as its owning census's `raw_authority_census_plans` row
# (`record_raw_authority_census` in raw_authority.py), so a plan can only
# reach zero census references once `prune_raw_authority_census_history`'s
# own plan-retention window (`RAW_AUTHORITY_CENSUS_PLAN_RETENTION`, 8
# generations) has already decided its census-plan detail is safe to drop --
# which that function only does once no unresolved blocker remains. This
# function therefore only ever touches a `raw_authority_plans` row that
# established machinery has already judged expendable; it cannot advance
# that judgment early, and cannot itself corrupt any `plan_count` /
# `post_plan_count` because it never deletes the rows those counts describe.
#
# Consequence, stated honestly: a plan still referenced by an active
# ('planned', unresolved) census -- a genuine live obligation -- is left
# untouched by this pass even if every raw it names is already gone. That
# is deliberate: the original 2026-07-22 live incident this bead cites was
# caused by a one-time repair script (`hook_deinflation.py`, since deleted)
# deleting raws out from under an in-flight census apply directly, bypassing
# retention authority entirely -- a defect in that now-gone one-time script,
# not something `cleanup_superseded_raw_snapshots` reproduces (it only
# deletes raws proven superseded via `active_raw_retention_authority` /
# the file-set recency heuristic, an orthogonal concern from census
# membership). A dangling reference from a still-live census plan to an
# already-superseded raw is a pre-existing reconciler-robustness question
# (tracked separately, polylogue-7f8yc) and is not this function's job to
# solve by deleting durable census evidence. The pagination/plan_count
# staleness this review also flagged already exists today via
# prune_raw_authority_census_history's own plan-retention window, entirely
# independent of this function (tracked separately, polylogue-4uzoo).
#
# Set-based (single pass, LEFT JOIN on the raw_id PK): expand each plan's
# input raws, GROUP BY plan, keep plans whose inputs are ALL missing. The
# equivalent correlated-subquery form measured ~26 billion ops (>1h) on the
# live archive; this form completes in ~0.3s. A plan with empty inputs
# produces no json_each rows and is correctly excluded (it is not
# orphaned-by-missing-raw). The candidate set from this query is then
# anti-joined against the three census/blocker tables (temp `plan_id`
# indexes make that anti-join index-driven too) before anything is deleted.
_PURELY_ORPHANED_AUTHORITY_PLANS_SQL = """
    SELECT p.plan_id
    FROM raw_authority_plans p, json_each(p.input_raw_ids_json) j
    LEFT JOIN raw_sessions r ON r.raw_id = j.value
    GROUP BY p.plan_id
    HAVING SUM(CASE WHEN r.raw_id IS NULL THEN 1 ELSE 0 END) > 0
       AND SUM(CASE WHEN r.raw_id IS NOT NULL THEN 1 ELSE 0 END) = 0
"""


def _delete_orphaned_raw_authority_plans(conn: sqlite3.Connection) -> int:
    """Prune ``raw_authority_plans`` rows that are both raw-orphaned (every
    input raw is gone) AND already census-free (zero remaining rows in
    ``raw_authority_census_plans`` / ``_post_plans`` / ``raw_authority_blockers``).

    Never deletes from the census/blocker tables themselves -- see the
    module comment above for why that would corrupt the immutable
    ``raw_authority_censuses.plan_count`` / ``post_plan_count`` ledger and
    silently drop live blocker obligations. Must run in the same transaction
    as (and after) the raw delete that may have orphaned these plans, so
    "orphaned" is always computed against the post-delete raw set. No-ops
    (returns 0) when ``raw_authority_plans`` doesn't exist, so this is safe
    to call unconditionally.
    """
    if not _table_exists(conn, "raw_authority_plans"):
        return 0
    conn.execute("CREATE TEMP TABLE _orphan_authority_plan_candidates (plan_id TEXT PRIMARY KEY)")
    try:
        conn.execute(f"INSERT INTO _orphan_authority_plan_candidates (plan_id) {_PURELY_ORPHANED_AUTHORITY_PLANS_SQL}")
        candidate_count = int(conn.execute("SELECT COUNT(*) FROM _orphan_authority_plan_candidates").fetchone()[0])
        if candidate_count == 0:
            return 0
        # Exclude any candidate still referenced by a census or an
        # (un)resolved blocker -- never delete a plan those durable rows
        # describe. Temp full plan_id indexes make this index-driven instead
        # of a full scan of the (potentially hundreds-of-thousands-row)
        # census/blocker tables per candidate.
        reference_indexes = {
            "_tmp_rawretention_authplan_cp": "raw_authority_census_plans",
            "_tmp_rawretention_authplan_cpp": "raw_authority_census_post_plans",
            "_tmp_rawretention_authplan_blk": "raw_authority_blockers",
        }
        for index_name, table in reference_indexes.items():
            conn.execute(f"CREATE INDEX {index_name} ON {table}(plan_id)")
        try:
            conn.execute(
                """
                DELETE FROM _orphan_authority_plan_candidates
                WHERE plan_id IN (SELECT plan_id FROM raw_authority_census_plans)
                   OR plan_id IN (SELECT plan_id FROM raw_authority_census_post_plans)
                   OR plan_id IN (SELECT plan_id FROM raw_authority_blockers)
                """
            )
        finally:
            for index_name in reference_indexes:
                conn.execute(f"DROP INDEX IF EXISTS {index_name}")
        count = int(conn.execute("SELECT COUNT(*) FROM _orphan_authority_plan_candidates").fetchone()[0])
        if count == 0:
            return 0
        conn.execute(
            "DELETE FROM raw_authority_plans WHERE plan_id IN (SELECT plan_id FROM _orphan_authority_plan_candidates)"
        )
        return count
    finally:
        conn.execute("DROP TABLE IF EXISTS _orphan_authority_plan_candidates")


def _superseded_archive_raw_session_candidates(
    conn: sqlite3.Connection,
    *,
    source_path: Path | None,
    keep_full_snapshots: int,
    keep_append_snapshots: int,
    min_acquired_at: str | None,
    limit: int,
) -> list[RawSnapshotCleanupCandidate]:
    source_path_str = str(source_path) if source_path is not None else None
    min_acquired_at_ms = _timestamp_ms(min_acquired_at)
    rows = conn.execute(
        _V1_RAW_CANDIDATE_SQL,
        (
            source_path_str,
            source_path_str,
            min_acquired_at_ms,
            min_acquired_at_ms,
            max(1, keep_full_snapshots),
            max(1, keep_append_snapshots),
            limit,
        ),
    ).fetchall()
    candidates: list[RawSnapshotCleanupCandidate] = []
    for row in rows:
        row_source_path = str(row[1])
        if not Path(row_source_path).exists():
            continue
        blob_hash = _blob_hash_text(row[3])
        if blob_hash is None:
            continue
        candidates.append(
            RawSnapshotCleanupCandidate(
                raw_id=str(row[0]),
                source_path=row_source_path,
                source_index=int(row[2] or 0),
                blob_size=int(row[4] or 0),
                blob_hash=blob_hash,
            )
        )
    return candidates


def superseded_raw_snapshot_candidates(
    conn: sqlite3.Connection,
    *,
    source_path: Path | None = None,
    keep_full_snapshots: int = 1,
    keep_append_snapshots: int = 1,
    min_acquired_at: str | None = None,
    limit: int = 1_000,
) -> list[RawSnapshotCleanupCandidate]:
    """Return redundant live raw snapshots that are safe to compact.

    The source file must still exist on disk before a row is returned. If
    the source disappeared, the raw blob may be the only remaining copy and
    is deliberately preserved.
    """
    if limit <= 0:
        return []

    if not _table_exists(conn, "raw_sessions"):
        return []
    return _superseded_archive_raw_session_candidates(
        conn,
        source_path=source_path,
        keep_full_snapshots=keep_full_snapshots,
        keep_append_snapshots=keep_append_snapshots,
        min_acquired_at=min_acquired_at,
        limit=limit,
    )


def cleanup_superseded_raw_snapshots(
    conn: sqlite3.Connection,
    *,
    source_path: Path | None = None,
    keep_full_snapshots: int = 1,
    keep_append_snapshots: int = 1,
    min_acquired_at: str | None = None,
    limit: int = 1_000,
    dry_run: bool = True,
    blob_store: BlobStore | None = None,
    protected_raw_ids: set[str] | frozenset[str] | None = None,
    eligible_raw_ids: set[str] | frozenset[str] | None = None,
) -> RawSnapshotCleanupResult:
    all_candidates = superseded_raw_snapshot_candidates(
        conn,
        source_path=source_path,
        keep_full_snapshots=keep_full_snapshots,
        keep_append_snapshots=keep_append_snapshots,
        min_acquired_at=min_acquired_at,
        limit=limit,
    )
    protected = protected_raw_ids or frozenset()
    eligible = eligible_raw_ids
    candidates = [
        candidate
        for candidate in all_candidates
        if candidate.raw_id not in protected and (eligible is None or candidate.raw_id in eligible)
    ]
    skipped_referenced_count = sum(candidate.raw_id in protected for candidate in all_candidates)
    if not candidates:
        return RawSnapshotCleanupResult(
            candidate_count=0,
            deleted_raw_count=0,
            deleted_blob_count=0,
            deleted_raw_bytes=0,
            deleted_blob_bytes=0,
            skipped_missing_source_count=0,
            skipped_referenced_count=skipped_referenced_count,
        )

    raw_ids = [candidate.raw_id for candidate in candidates]
    raw_bytes = sum(candidate.blob_size for candidate in candidates)
    if dry_run:
        return RawSnapshotCleanupResult(
            candidate_count=len(candidates),
            deleted_raw_count=0,
            deleted_blob_count=0,
            deleted_raw_bytes=raw_bytes,
            deleted_blob_bytes=0,
            skipped_missing_source_count=0,
            skipped_referenced_count=skipped_referenced_count,
        )

    placeholders = ", ".join("?" for _ in raw_ids)
    if _column_exists(conn, "blob_refs", "ref_id"):
        conn.execute(f"DELETE FROM blob_refs WHERE ref_id IN ({placeholders})", raw_ids)
    elif _column_exists(conn, "blob_refs", "raw_id"):
        conn.execute(f"DELETE FROM blob_refs WHERE raw_id IN ({placeholders})", raw_ids)
    conn.execute(f"DELETE FROM raw_sessions WHERE raw_id IN ({placeholders})", raw_ids)
    # Prune raw-authority plans/blockers/census rows now dangling on the
    # deleted raws (input_raw_ids_json is a JSON string, not an FK, so no
    # cascade cleaned them -- polylogue-i3zo). Same transaction, after the raw
    # delete, so orphaned-ness is computed against the post-delete raw set.
    deleted_orphaned_authority_plan_count = _delete_orphaned_raw_authority_plans(conn)
    conn.commit()

    store = blob_store if blob_store is not None else get_blob_store()
    deleted_blob_count = 0
    deleted_blob_bytes = 0
    errors: list[str] = []
    for candidate in candidates:
        if _column_exists(conn, "blob_refs", "blob_hash"):
            try:
                blob_hash = bytes.fromhex(candidate.blob_store_hash)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if conn.execute("SELECT 1 FROM blob_refs WHERE blob_hash = ? LIMIT 1", (blob_hash,)).fetchone():
                continue
        else:
            # Without a reference catalog, row cleanup cannot prove that this
            # process owns the final CAS reference. Leave bytes for typed GC.
            continue
        try:
            path = store.blob_path(candidate.blob_store_hash)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not path.exists():
            continue
        with suppress(OSError):
            deleted_blob_bytes += path.stat().st_size
        try:
            path.unlink()
            deleted_blob_count += 1
        except OSError as exc:
            errors.append(f"{candidate.raw_id[:16]}: {exc}")

    return RawSnapshotCleanupResult(
        candidate_count=len(candidates),
        deleted_raw_count=len(candidates),
        deleted_blob_count=deleted_blob_count,
        deleted_raw_bytes=raw_bytes,
        deleted_blob_bytes=deleted_blob_bytes,
        skipped_missing_source_count=0,
        skipped_referenced_count=skipped_referenced_count,
        deleted_orphaned_authority_plan_count=deleted_orphaned_authority_plan_count,
        errors=tuple(errors),
    )


def compact_paths_superseded_raw_snapshots(
    conn: sqlite3.Connection,
    source_paths: Iterable[Path],
    *,
    limit_per_path: int = 25,
    min_acquired_at: str | None = None,
    dry_run: bool = False,
    protected_raw_ids: set[str] | frozenset[str] | None = None,
    eligible_raw_ids: set[str] | frozenset[str] | None = None,
) -> RawSnapshotCleanupResult:
    totals = RawSnapshotCleanupResult(
        candidate_count=0,
        deleted_raw_count=0,
        deleted_blob_count=0,
        deleted_raw_bytes=0,
        deleted_blob_bytes=0,
        skipped_missing_source_count=0,
    )
    errors: list[str] = []
    for path in source_paths:
        result = cleanup_superseded_raw_snapshots(
            conn,
            source_path=path,
            keep_full_snapshots=1_000_000,
            min_acquired_at=min_acquired_at,
            limit=limit_per_path,
            dry_run=dry_run,
            protected_raw_ids=protected_raw_ids,
            eligible_raw_ids=eligible_raw_ids,
        )
        errors.extend(result.errors)
        totals = RawSnapshotCleanupResult(
            candidate_count=totals.candidate_count + result.candidate_count,
            deleted_raw_count=totals.deleted_raw_count + result.deleted_raw_count,
            deleted_blob_count=totals.deleted_blob_count + result.deleted_blob_count,
            deleted_raw_bytes=totals.deleted_raw_bytes + result.deleted_raw_bytes,
            deleted_blob_bytes=totals.deleted_blob_bytes + result.deleted_blob_bytes,
            skipped_missing_source_count=totals.skipped_missing_source_count + result.skipped_missing_source_count,
            skipped_referenced_count=totals.skipped_referenced_count + result.skipped_referenced_count,
            deleted_orphaned_authority_plan_count=(
                totals.deleted_orphaned_authority_plan_count + result.deleted_orphaned_authority_plan_count
            ),
            errors=tuple(errors),
        )
    return totals


__all__ = [
    "BrokenAppendHeadSample",
    "CursorAheadSample",
    "CursorAuthorityGapSample",
    "CursorAuthorityGapState",
    "RawFrontierIntegrityProjection",
    "RawFrontierIntegritySnapshot",
    "RawFrontierIntegrityStatus",
    "RawSnapshotCleanupCandidate",
    "RawSnapshotCleanupResult",
    "RawRetentionAuthority",
    "RawRetentionSafetyError",
    "StaleSupersessionCandidate",
    "StaleSupersessionIneligible",
    "StaleSupersessionReissuePlan",
    "StaleSupersessionReissueResult",
    "active_raw_retention_authority",
    "cleanup_superseded_raw_snapshots",
    "combine_raw_frontier_integrity_statuses",
    "compact_paths_superseded_raw_snapshots",
    "missing_source_raw_integrity_status",
    "plan_stale_supersession_reissue",
    "protected_active_raw_revision_ids",
    "raw_frontier_integrity_projection",
    "raw_frontier_integrity_summary",
    "raw_frontier_integrity_snapshot",
    "reissue_stale_supersession_receipts",
    "superseded_raw_snapshot_candidates",
    "unknown_raw_frontier_integrity_projection",
]

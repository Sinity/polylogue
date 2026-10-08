"""Retention cleanup for superseded live raw payload snapshots."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, get_args

from polylogue.archive.revision_authority import raw_receipt_order_sql
from polylogue.core.errors import SchemaSkew
from polylogue.core.raw_failure_evidence import RAW_FAILURE_EVIDENCE_KINDS, RawFailureEvidenceKind
from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.core.timestamps import to_epoch_ms
from polylogue.logging import get_logger
from polylogue.storage.archive_identity import resolve_active_index_path
from polylogue.storage.blob_store import BlobStore, get_blob_store

logger = get_logger(__name__)

_TERMINAL_RAW_FAILURE_EVIDENCE_KINDS = frozenset(
    kind.value for kind in RawFailureEvidenceKind if kind.lifecycle == "terminal"
)

_V1_RAW_CANDIDATE_SQL = f"""
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
            ORDER BY {raw_receipt_order_sql("raw_sessions")} DESC, raw_id DESC
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
    errors: tuple[str, ...] = ()
    #: Source paths this pass did not finish compacting. A pass is bounded per
    #: path (``limit_per_path``); when a path fills that bound the remaining
    #: superseded snapshots are still there afterwards. Naming them is what
    #: lets the caller retain them as retryable backlog instead of reporting a
    #: finished answer it did not compute. ``errors`` is deliberately not a
    #: residual: those come from the blob unlink, whose subjects are already
    #: unreferenced and belong to the ordinary blob-GC owner.
    residual_source_paths: tuple[str, ...] = ()


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


def _active_index_raw_authority(
    index_db_path: Path,
    *,
    raw_ids: frozenset[str] | None = None,
    logical_source_keys: frozenset[str] | None = None,
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
        from polylogue.storage.sqlite.connection_profile import open_readonly_connection

        with closing(open_readonly_connection(index_db_path.resolve(), timeout_class="background-read")) as conn:
            if raw_ids is None:
                session_rows = conn.execute("SELECT DISTINCT raw_id FROM sessions WHERE raw_id IS NOT NULL").fetchall()
            else:
                session_rows = _index_rows_for_raw_ids(
                    conn,
                    "SELECT DISTINCT raw_id FROM sessions WHERE raw_id IN ({placeholders})",
                    raw_ids,
                )
            if logical_source_keys is None:
                head_rows = conn.execute(_INDEX_RETENTION_HEAD_SQL.format(where_clause="")).fetchall()
                eligible_rows = conn.execute(_INDEX_RETENTION_ELIGIBLE_SQL.format(where_clause="")).fetchall()
            else:
                head_rows = _index_rows_for_logical_source_keys(conn, _INDEX_RETENTION_HEAD_SQL, logical_source_keys)
                eligible_rows = _index_rows_for_logical_source_keys(
                    conn, _INDEX_RETENTION_ELIGIBLE_SQL, logical_source_keys
                )
    except (OSError, sqlite3.Error, SchemaSkew) as exc:
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


_INDEX_RETENTION_HEAD_SQL = """SELECT logical_source_key, accepted_raw_id, accepted_source_revision,
                                      accepted_frontier_kind, accepted_frontier,
                                      acquisition_generation, append_end_offset
                               FROM raw_revision_heads AS head
                               WHERE 1 = 1{where_clause}"""

_INDEX_RETENTION_ELIGIBLE_SQL = """SELECT DISTINCT application.raw_id,
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
                                     AND head.accepted_frontier_kind = 'byte'{where_clause}"""

_RETENTION_SCOPE_BATCH_SIZE = 500


def _value_batches(values: frozenset[str]) -> Iterable[tuple[str, ...]]:
    ordered = tuple(sorted(values))
    for start in range(0, len(ordered), _RETENTION_SCOPE_BATCH_SIZE):
        yield ordered[start : start + _RETENTION_SCOPE_BATCH_SIZE]


def _index_rows_for_raw_ids(
    conn: sqlite3.Connection,
    statement: str,
    raw_ids: frozenset[str],
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for batch in _value_batches(raw_ids):
        if not batch:
            continue
        placeholders = ", ".join("?" for _ in batch)
        rows.extend(conn.execute(statement.format(placeholders=placeholders), batch).fetchall())
    return rows


def _index_rows_for_logical_source_keys(
    conn: sqlite3.Connection,
    statement: str,
    logical_source_keys: frozenset[str],
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for batch in _value_batches(logical_source_keys):
        if not batch:
            continue
        placeholders = ", ".join("?" for _ in batch)
        rows.extend(
            conn.execute(
                statement.format(where_clause=f" AND head.logical_source_key IN ({placeholders})"), batch
            ).fetchall()
        )
    return rows


def _active_index_raw_authority_from_connection(
    conn: sqlite3.Connection,
) -> tuple[frozenset[str], tuple[_IndexRawRevisionHead, ...], tuple[_EligibleRawReceipt, ...]]:
    """Read the retention authority from an already-observed index tier.

    The failure contract is the path reader's, not the connection's: a damaged
    index page, a busy pinned reader or a tier missing the revision tables must
    reach the caller as ``RawRetentionSafetyError`` so the frontier resolves to
    ``unknown``. Letting a bare ``sqlite3.Error`` escape turns a
    degraded-but-readable archive into a raised status operation instead.
    """

    try:
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
    def read_rows(batch: tuple[str, ...]) -> Sequence[sqlite3.Row]:
        placeholders = ", ".join("?" for _ in batch)
        with closing(
            conn.execute(
                f"SELECT {_RAW_REVISION_CHAIN_COLUMNS} FROM raw_sessions WHERE raw_id IN ({placeholders})",
                batch,
            )
        ) as rows:
            return rows.fetchall()

    return _raw_revision_rows_from_inputs(read_rows, raw_ids, allow_missing=allow_missing)


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
            authority = str(row["revision_authority"])
            # A baseline full raw -- the first-ever observation of a logical
            # source key -- is admitted as FULL/ASSERTED by design: the
            # BASELINE outcome in archive_tiers/raw_admission.py has no prior
            # head to compare bytes against, so there is nothing for a byte
            # proof to be over. Demanding byte_proven of every full head
            # therefore reported a violation for every source seen exactly
            # once, which is the entire content of a freshly ingested archive
            # -- `ops status` called it "raw frontier integrity violated" and
            # handed out a repair hint. Require the byte proof only where a
            # predecessor actually exists for it to be a proof about.
            has_predecessor = row["predecessor_raw_id"] is not None
            if authority != "byte_proven" and not (authority == "asserted" and not has_predecessor):
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
    authority_source_paths: Iterable[Path] | None = None,
) -> RawRetentionAuthority:
    """Return current protection plus explicitly authorized deletion rows.

    The index selects active leaves. Source-tier predecessor evidence proves
    which append fragments are required to reconstruct each leaf. Only an
    immutable ``superseded`` receipt tied to the current head authorizes raw
    deletion. Callers must serialize this read with source deletion under the
    daemon's single-writer contract, or stop the daemon for manual cleanup.
    ``terminal_source_paths`` scopes terminal-artifact protection only for a
    deletion operation constrained to those same physical paths. Supplying
    ``authority_source_paths`` also narrows the index authority reads to the
    raw identities and logical sources carried by that deletion page. This is
    safe only for a caller that will delete rows from those same paths.
    """
    original_row_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        scoped_raw_ids: frozenset[str] | None = None
        scoped_logical_source_keys: frozenset[str] | None = None
        if authority_source_paths is not None:
            scoped_paths = tuple(sorted({str(path) for path in authority_source_paths}))
            if scoped_paths:
                scoped_raw_ids, scoped_logical_source_keys = _raw_retention_scope(conn, scoped_paths)
            else:
                scoped_raw_ids = frozenset()
                scoped_logical_source_keys = frozenset()
        session_raw_ids, heads, eligible_receipts = _active_index_raw_authority(
            index_db_path,
            raw_ids=scoped_raw_ids,
            logical_source_keys=scoped_logical_source_keys,
        )
        seeds = set(session_raw_ids)
        seeds.update(head.accepted_raw_id for head in heads)
        if not seeds:
            if terminal_source_paths is None:
                raw_rows = conn.execute("SELECT raw_id FROM raw_sessions").fetchall()
            else:
                selected_paths = tuple(sorted({str(path) for path in terminal_source_paths}))
                if selected_paths:
                    placeholders = ", ".join("?" for _ in selected_paths)
                    raw_rows = conn.execute(
                        f"SELECT raw_id FROM raw_sessions WHERE source_path IN ({placeholders})",
                        selected_paths,
                    ).fetchall()
                else:
                    raw_rows = []
            all_raw_ids = frozenset(str(row[0]) for row in raw_rows)
            terminal_artifact_raw_ids = _terminal_artifact_raw_ids(conn, source_paths=terminal_source_paths)
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
    except sqlite3.Error as exc:
        raise RawRetentionSafetyError(f"raw retention authority is unreadable: {exc}") from exc
    finally:
        conn.row_factory = original_row_factory


def _raw_retention_scope(
    conn: sqlite3.Connection,
    source_paths: tuple[str, ...],
) -> tuple[frozenset[str], frozenset[str]]:
    """Return the raw and logical identities a path-bounded cleanup can touch.

    A membership-governed raw keeps its acquisition envelope's (pending)
    logical key, while its index heads are keyed by the session identities
    its census recorded. Both keys are in scope; otherwise the heads of a
    censused raw are invisible and its session row reads as an unproven
    byte chain.
    """

    raw_ids: set[str] = set()
    logical_source_keys: set[str] = set()
    for start in range(0, len(source_paths), _RETENTION_SCOPE_BATCH_SIZE):
        paths = source_paths[start : start + _RETENTION_SCOPE_BATCH_SIZE]
        placeholders = ", ".join("?" for _ in paths)
        rows = conn.execute(
            f"SELECT raw_id, logical_source_key FROM raw_sessions WHERE source_path IN ({placeholders}) "
            "UNION ALL SELECT membership.raw_id, membership.logical_source_key "
            "FROM raw_session_memberships AS membership JOIN raw_sessions AS raw ON raw.raw_id = membership.raw_id "
            f"WHERE raw.source_path IN ({placeholders})",
            (*paths, *paths),
        ).fetchall()
        for raw_id, logical_source_key in rows:
            raw_ids.add(str(raw_id))
            if logical_source_key is not None and str(logical_source_key):
                logical_source_keys.add(str(logical_source_key))
    return frozenset(raw_ids), frozenset(logical_source_keys)


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


def broken_head_reason(broken_count: int) -> str:
    """The operator reason for active raw seeds whose predecessor chain failed."""
    if broken_count == 0:
        return ""
    return f"{broken_count} active index raw seed(s) have a broken predecessor chain or invalid source binding"


def cursor_ahead_reason(ahead_count: int, ahead_comparison_count: int, gap_count: int, deferred_count: int) -> str:
    """The operator reason for cursor rows ahead of, or incomparable with, accepted raw."""
    reasons: list[str] = []
    if ahead_count:
        reasons.append(
            f"{ahead_count} ingest cursor row(s) committed past accepted raw material "
            f"across {ahead_comparison_count} cursor/head comparison(s)"
        )
    if gap_count:
        reasons.append(f"{gap_count} cursor/head authority row(s) could not be compared")
    if deferred_count:
        reasons.append(f"{deferred_count} ingest cursor(s) deferred awaiting the quiet window")
    return "; ".join(reasons)


_FINDINGS_COUNTS = (
    "head_checks",
    "blocking_heads",
    "broken_heads",
    "cursor_checks",
    "cursor_comparisons",
    "cursor_ahead",
    "cursor_ahead_comparisons",
    "cursor_gaps",
    "cursor_deferred",
    "missing_session_raws",
)
_FINDINGS_SAMPLES = ("broken_head_samples", "cursor_ahead_samples", "cursor_gap_samples")
_GAP_STATES = frozenset(get_args(CursorAuthorityGapState))


def _require_sample(raw: object, fields: dict[str, tuple[type, ...]]) -> dict[str, object]:
    if not isinstance(raw, dict) or set(raw) != set(fields):
        raise ValueError("frontier inspection sample has an undeclared shape")
    for name, types in fields.items():
        value = raw[name]
        if type(value) not in types or (type(value) is int and value < 0):
            raise ValueError(f"frontier inspection sample field {name} has an undeclared value")
    return raw


def _broken_head_from_document(raw: object) -> BrokenAppendHeadSample:
    sample = _require_sample(raw, {"logical_source_key": (str,), "accepted_raw_id": (str,), "reason": (str,)})
    return BrokenAppendHeadSample(
        logical_source_key=str(sample["logical_source_key"]),
        accepted_raw_id=str(sample["accepted_raw_id"]),
        reason=str(sample["reason"]),
    )


def _cursor_ahead_from_document(raw: object) -> CursorAheadSample:
    sample = _require_sample(
        raw,
        {
            "source_path": (str,),
            "logical_source_key": (str,),
            "cursor_byte_offset": (int,),
            "accepted_frontier": (int,),
            "affected_head_count": (int,),
            "canonical_source_path": (str, type(None)),
        },
    )
    return CursorAheadSample(
        source_path=str(sample["source_path"]),
        logical_source_key=str(sample["logical_source_key"]),
        cursor_byte_offset=cast(int, sample["cursor_byte_offset"]),
        accepted_frontier=cast(int, sample["accepted_frontier"]),
        affected_head_count=cast(int, sample["affected_head_count"]),
        canonical_source_path=cast(str | None, sample["canonical_source_path"]),
    )


def _cursor_gap_from_document(raw: object) -> CursorAuthorityGapSample:
    sample = _require_sample(
        raw,
        {
            "state": (str,),
            "source_path": (str, type(None)),
            "logical_source_key": (str, type(None)),
            "cursor_byte_offset": (int, type(None)),
            "reason": (str,),
        },
    )
    if sample["state"] not in _GAP_STATES:
        raise ValueError("frontier inspection gap sample names an undeclared state")
    return CursorAuthorityGapSample(
        state=cast(CursorAuthorityGapState, sample["state"]),
        source_path=cast(str | None, sample["source_path"]),
        logical_source_key=cast(str | None, sample["logical_source_key"]),
        cursor_byte_offset=cast(int | None, sample["cursor_byte_offset"]),
        reason=str(sample["reason"]),
    )


@dataclass(frozen=True)
class FrontierInspectionFindings:
    """Per-category results of one completed frontier inspection pass.

    The inspection records this document in its Ops mark, so status renders
    the certificate's categories instead of repeating the corpus inspection.
    ``mode`` says what the pass covered: a ``delta`` pass starts from a
    healthy certificate, so its violation counts and samples are complete
    while its checked counts cover only the changed keys.
    """

    mode: str
    head_checks: int
    blocking_heads: int
    broken_heads: int
    broken_head_samples: tuple[BrokenAppendHeadSample, ...]
    cursor_checks: int
    cursor_comparisons: int
    cursor_ahead: int
    cursor_ahead_comparisons: int
    cursor_ahead_samples: tuple[CursorAheadSample, ...]
    cursor_gaps: int
    cursor_gap_samples: tuple[CursorAuthorityGapSample, ...]
    cursor_deferred: int
    missing_session_raws: int

    def to_document(self) -> str:
        import json
        from dataclasses import asdict

        payload: dict[str, object] = {name: getattr(self, name) for name in _FINDINGS_COUNTS}
        payload["mode"] = self.mode
        for name in _FINDINGS_SAMPLES:
            payload[name] = [asdict(sample) for sample in getattr(self, name)]
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_document(cls, document: str) -> FrontierInspectionFindings:
        """Decode exactly the recorded shape; anything else is a ``ValueError``."""
        import json

        payload = json.loads(document)
        if not isinstance(payload, dict) or set(payload) != {*_FINDINGS_COUNTS, "mode", *_FINDINGS_SAMPLES}:
            raise ValueError("frontier inspection findings have an undeclared shape")
        counts: dict[str, int] = {}
        for name in _FINDINGS_COUNTS:
            value = payload[name]
            if type(value) is not int or value < 0:
                raise ValueError(f"frontier inspection finding {name} is not a nonnegative count")
            counts[name] = value
        mode = payload["mode"]
        if mode not in {"full", "delta"}:
            raise ValueError("frontier inspection findings name an undeclared mode")
        for name in _FINDINGS_SAMPLES:
            if not isinstance(payload[name], list):
                raise ValueError(f"frontier inspection {name} are not a list")
        broken = tuple(_broken_head_from_document(raw) for raw in payload["broken_head_samples"])
        ahead = tuple(_cursor_ahead_from_document(raw) for raw in payload["cursor_ahead_samples"])
        gaps = tuple(_cursor_gap_from_document(raw) for raw in payload["cursor_gap_samples"])
        for samples, count in ((broken, "broken_heads"), (ahead, "cursor_ahead"), (gaps, "cursor_gaps")):
            if len(samples) > counts[count]:
                raise ValueError(f"frontier inspection records more samples than {count}")
        return cls(
            mode=mode,
            broken_head_samples=broken,
            cursor_ahead_samples=ahead,
            cursor_gap_samples=gaps,
            **counts,
        )


@dataclass(frozen=True)
class CursorAheadSample:
    """One cursor with at least one accepted byte head behind its frontier."""

    source_path: str
    logical_source_key: str
    cursor_byte_offset: int
    accepted_frontier: int
    affected_head_count: int
    canonical_source_path: str | None = None


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
    canonical_source_path: str | None = None

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
    #: Cursors whose durably captured tail awaits the quiet window. A typed,
    #: safe state: never a gap, never a reason to block other paths.
    cursor_authority_deferred_count: int
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
    cursor_authority_deferred_count: int
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
            "cursor_authority_deferred_count": self.cursor_authority_deferred_count,
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


def _raw_frontier_integrity_from_coverage(
    coverage: Mapping[str, object],
    materialization: Mapping[str, object],
    *,
    sample_limit: int = 10,
) -> RawFrontierIntegrityProjection:
    """Reduce the completed certificate and missing-Source signal once."""
    missing_status, missing_count, missing_samples, missing_reason = missing_source_raw_integrity_status(
        materialization, sample_limit=sample_limit
    )
    current = bool(coverage.get("current"))
    healthy = current and bool(coverage.get("healthy"))
    blocked = current and coverage.get("state") == "blocked"
    findings = coverage.get("findings")
    if current and isinstance(findings, FrontierInspectionFindings):
        return _raw_frontier_integrity_from_findings(
            findings,
            blocked=blocked,
            missing=(missing_status, missing_count, missing_samples, missing_reason),
            available=bool(coverage.get("available")),
        )
    reason = (
        ""
        if healthy
        else str(
            coverage.get("detail")
            or ("accepted frontier inspection is blocked" if blocked else "accepted frontier inspection is not current")
        )
    )
    status: RawFrontierIntegrityStatus = "healthy" if healthy else "unknown"
    return RawFrontierIntegrityProjection(
        available=bool(coverage.get("available")) and current,
        overall_status="violated"
        if blocked or missing_status == "violated"
        else combine_raw_frontier_integrity_statuses(status, missing_status, status),
        broken_head_status=status,
        broken_head_count=0,
        broken_head_checked_count=0,
        broken_head_samples=(),
        broken_head_reason=reason,
        missing_source_raw_status=missing_status,
        missing_source_raw_count=missing_count,
        missing_source_raw_samples=missing_samples,
        missing_source_raw_reason=missing_reason,
        cursor_ahead_status=status,
        cursor_ahead_count=0,
        cursor_ahead_checked_count=0,
        cursor_head_comparison_count=0,
        cursor_ahead_comparison_count=0,
        cursor_ahead_samples=(),
        cursor_authority_gap_count=0,
        cursor_authority_gap_samples=(),
        cursor_authority_deferred_count=0,
        cursor_ahead_reason=reason,
    )


def _raw_frontier_integrity_from_findings(
    findings: FrontierInspectionFindings,
    *,
    blocked: bool,
    missing: tuple[RawFrontierIntegrityStatus, int, tuple[Mapping[str, object], ...], str],
    available: bool,
) -> RawFrontierIntegrityProjection:
    """Render each category of a current certificate from its recorded findings."""
    missing_status, missing_count, missing_samples, missing_reason = missing
    broken_status: RawFrontierIntegrityStatus = "violated" if findings.broken_heads else "healthy"
    cursor_status: RawFrontierIntegrityStatus = (
        "violated" if findings.cursor_ahead else "unknown" if findings.cursor_gaps else "healthy"
    )
    # A blocked certificate is a violation even when its cause (a blocking
    # obligation, a lost session raw) has no category of its own here.
    overall: RawFrontierIntegrityStatus = (
        "violated" if blocked else combine_raw_frontier_integrity_statuses(broken_status, missing_status, cursor_status)
    )
    return RawFrontierIntegrityProjection(
        available=available,
        overall_status=overall,
        broken_head_status=broken_status,
        broken_head_count=findings.broken_heads,
        broken_head_checked_count=findings.head_checks,
        broken_head_samples=findings.broken_head_samples,
        broken_head_reason=broken_head_reason(findings.broken_heads),
        missing_source_raw_status=missing_status,
        missing_source_raw_count=missing_count,
        missing_source_raw_samples=missing_samples,
        missing_source_raw_reason=missing_reason,
        cursor_ahead_status=cursor_status,
        cursor_ahead_count=findings.cursor_ahead,
        cursor_ahead_checked_count=findings.cursor_checks,
        cursor_head_comparison_count=findings.cursor_comparisons,
        cursor_ahead_comparison_count=findings.cursor_ahead_comparisons,
        cursor_ahead_samples=findings.cursor_ahead_samples,
        cursor_authority_gap_count=findings.cursor_gaps,
        cursor_authority_gap_samples=findings.cursor_gap_samples,
        cursor_authority_deferred_count=findings.cursor_deferred,
        cursor_ahead_reason=cursor_ahead_reason(
            findings.cursor_ahead, findings.cursor_ahead_comparisons, findings.cursor_gaps, findings.cursor_deferred
        ),
    )


def raw_frontier_integrity_projection(
    archive_root: Path,
    raw_materialization_readiness: Mapping[str, object],
    *,
    sample_limit: int = 10,
) -> RawFrontierIntegrityProjection:
    """Read completed coverage without repeating the original corpus inspection."""
    from polylogue.core.errors import SchemaRefusalError
    from polylogue.core.evidence import Measured, Unavailable
    from polylogue.storage.frontier_inspection import read_frontier_coverage_for_archive
    from polylogue.storage.tier_access import capture_sqlite_read

    coverage: dict[str, object]
    from polylogue.storage.archive_identity import ArchiveLocationError

    try:
        read = capture_sqlite_read(lambda: read_frontier_coverage_for_archive(archive_root))
    except (OSError, ValueError, ArchiveLocationError, SchemaRefusalError) as failure:
        # An incoherent active Index pointer or a refused tier (missing,
        # unreadable or schema-skewed) leaves coverage unavailable with its
        # typed detail; it never aborts the status read.
        coverage = {"available": False, "current": False, "healthy": False, "detail": str(failure)}
    else:
        if isinstance(read, Measured):
            coverage = read.value
        else:
            detail = read.detail if isinstance(read, Unavailable) else None
            coverage = {
                "available": False,
                "current": False,
                "healthy": False,
                "detail": detail or "sqlite_read_failed",
            }
    return _raw_frontier_integrity_from_coverage(coverage, raw_materialization_readiness, sample_limit=sample_limit)


@dataclass(frozen=True)
class RawFrontierBlockedPaths:
    """Which source paths the frontier proof refuses, and what it cannot attribute.

    ``source_paths`` holds every path a *violation* names: an ingest cursor
    committed past accepted raw material, or an accepted head whose source
    chain is broken. Processing such a path cannot repair it, so it is
    refused while the rest of a batch proceeds.

    ``gap_source_paths`` holds every path an authority *gap* names: a cursor
    whose head could not be compared (raw evidence with no accepted head, a
    cursor path absent from the source tier). A gap is an unknown, and
    ingesting or materializing the path is the only thing that resolves it,
    so callers admit these paths.

    ``unattributed_reason`` is set when something blocks that no path
    explains (an unreadable tier, a raw with no source path, missing source
    raws); that still blocks everything.
    """

    source_paths: frozenset[str]
    unattributed_reason: str | None
    gap_source_paths: frozenset[str] = frozenset()


def raw_frontier_blocked_selected_paths(archive_root: Path, selected_paths: Sequence[Path]) -> RawFrontierBlockedPaths:
    """Check chain and cursor authority for the selected connected source paths.

    Global index-to-source existence is proved separately before this read.
    A path with no retained raw is still checked against its ops cursor.
    """
    try:
        return _raw_frontier_blocked_selected_paths(archive_root, selected_paths)
    except (OSError, RawRetentionSafetyError, ValueError) as exc:
        return RawFrontierBlockedPaths(frozenset(), f"selected source frontier is unreadable: {exc}")


def _raw_frontier_blocked_selected_paths(archive_root: Path, selected_paths: Sequence[Path]) -> RawFrontierBlockedPaths:
    from polylogue.storage.sqlite.archive_tiers.revision_governance import expand_raw_membership_selection_sync
    from polylogue.storage.sqlite.connection_profile import attach_readonly_database, open_readonly_connection

    try:
        index_path = resolve_active_index_path(archive_root)
        source_path = archive_root / "source.db"
        ops_path = archive_root / "ops.db"
        if not source_path.is_file() or not index_path.is_file() or not ops_path.is_file():
            return RawFrontierBlockedPaths(frozenset(), "required frontier authority tier is unavailable")
        spellings = {str(path) for path in selected_paths}
        spellings.update(str(path.resolve()) for path in selected_paths)
        with closing(open_readonly_connection(source_path)) as conn:
            conn.row_factory = sqlite3.Row
            attach_readonly_database(conn, index_path, alias="index_tier")
            conn.execute("BEGIN")
            source_reason = _source_tier_unavailable_reason(conn)
            if source_reason is not None:
                return RawFrontierBlockedPaths(frozenset(), source_reason)
            if conn.execute("SELECT 1 FROM raw_sessions WHERE canonical_source_path IS NULL LIMIT 1").fetchone():
                return RawFrontierBlockedPaths(frozenset(), "raw source canonical path authority is unavailable")
            raw_ids: set[str] = set()
            for batch in _value_batches(frozenset(spellings)):
                marks = ",".join("?" for _ in batch)
                raw_ids.update(
                    str(row[0])
                    for row in conn.execute(
                        f"SELECT raw_id FROM raw_sessions WHERE source_path IN ({marks}) "
                        f"OR canonical_source_path IN ({marks})",
                        (*batch, *batch),
                    )
                )
            component, logical_keys = expand_raw_membership_selection_sync(conn, sorted(raw_ids))
            component_set = set(component)
            paths_by_raw = _source_paths_for_raw_ids(conn, component_set)
            canonical_by_raw = _canonical_paths_for_raw_ids(conn, component_set)
            component_paths = set(paths_by_raw.values()) | set(canonical_by_raw.values())

            def paths_for_logical_key(key: str) -> set[str]:
                # Every stored spelling: a refusal must match whichever path
                # the watcher selects, including the canonical real path.
                return {
                    str(value)
                    for row in conn.execute(
                        "SELECT source_path, canonical_source_path FROM raw_sessions WHERE logical_source_key = ? "
                        "UNION SELECT r.source_path, r.canonical_source_path FROM raw_session_memberships m "
                        "JOIN raw_sessions r ON r.raw_id = m.raw_id WHERE m.logical_source_key = ?",
                        (key, key),
                    )
                    for value in row
                    if value is not None
                }

            heads: list[_IndexRawRevisionHead] = []
            sessions: set[str] = set()
            for batch in _value_batches(frozenset(component)):
                marks = ",".join("?" for _ in batch)
                sessions.update(
                    str(row[0])
                    for row in conn.execute(
                        f"SELECT DISTINCT raw_id FROM index_tier.sessions WHERE raw_id IN ({marks})", batch
                    )
                )
                heads.extend(
                    _IndexRawRevisionHead(*tuple(row))
                    for row in conn.execute(
                        f"SELECT logical_source_key, accepted_raw_id, accepted_source_revision, "
                        f"accepted_frontier_kind, accepted_frontier, acquisition_generation, append_end_offset "
                        f"FROM index_tier.raw_revision_heads WHERE accepted_raw_id IN ({marks})",
                        batch,
                    )
                )
            for batch in _value_batches(frozenset(logical_keys)):
                marks = ",".join("?" for _ in batch)
                heads.extend(
                    _IndexRawRevisionHead(*tuple(row))
                    for row in conn.execute(
                        f"SELECT logical_source_key, accepted_raw_id, accepted_source_revision, "
                        f"accepted_frontier_kind, accepted_frontier, acquisition_generation, append_end_offset "
                        f"FROM index_tier.raw_revision_heads WHERE logical_source_key IN ({marks})",
                        batch,
                    )
                )
            unique_heads = tuple({(head.logical_source_key, head.accepted_raw_id): head for head in heads}.values())
            paths_by_raw.update(_source_paths_for_raw_ids(conn, {head.accepted_raw_id for head in unique_heads}))
            broken, count, _checked, samples, reason = _check_broken_active_chains(
                conn, frozenset(sessions), unique_heads, sample_limit=len(component) + len(unique_heads) + 1
            )
            if broken == "unknown":
                return RawFrontierBlockedPaths(frozenset(), reason)
            refused: set[str] = set()
            byte_heads_by_raw: dict[str, list[_IndexRawRevisionHead]] = {}
            for head in unique_heads:
                if head.accepted_frontier_kind == "byte":
                    byte_heads_by_raw.setdefault(head.accepted_raw_id, []).append(head)
            for sample in samples:
                path = paths_by_raw.get(sample.accepted_raw_id)
                if path is None:
                    return RawFrontierBlockedPaths(
                        frozenset(), f"broken head has no source path: {sample.accepted_raw_id}"
                    )
                refused.add(path)
                # A raw can have several byte and semantic heads. Attribute
                # each bad head binding to its key; a broken predecessor chain
                # affects every byte head. Semantic-only siblings do not
                # inherit a byte violation just because they share the raw.
                byte_heads = byte_heads_by_raw.get(sample.accepted_raw_id, [])
                relevant_keys: set[str] = set()
                if byte_heads:
                    revision_rows = _raw_revision_rows(conn, {sample.accepted_raw_id}, allow_missing=True)
                    raw_row = revision_rows.get(sample.accepted_raw_id)
                    if raw_row is None:
                        return RawFrontierBlockedPaths(frozenset(), "selected byte-head raw disappeared")
                    for head in byte_heads:
                        try:
                            _validate_byte_head(raw_row, head)
                        except RawRetentionSafetyError:
                            relevant_keys.add(head.logical_source_key)
                    try:
                        _validate_active_revision_chain(revision_rows, sample.accepted_raw_id)
                    except RawRetentionSafetyError:
                        relevant_keys.update(head.logical_source_key for head in byte_heads)
                elif sample.logical_source_key != "session.raw_id":
                    relevant_keys.add(sample.logical_source_key)
                for key in relevant_keys:
                    refused.update(paths_for_logical_key(key))
            with closing(open_readonly_connection(ops_path, validate_schema=False)) as ops:
                if ops.execute(
                    "SELECT 1 FROM ingest_cursor WHERE canonical_source_path IS NULL "
                    "AND byte_offset IS NOT NULL AND excluded = 0 LIMIT 1"
                ).fetchone():
                    return RawFrontierBlockedPaths(frozenset(), "ops cursor canonical path authority is unavailable")
                cursor = _check_cursor_ahead_of_accepted(
                    conn,
                    ops_path,
                    unique_heads,
                    sample_limit=len(component_paths) + len(spellings) + len(unique_heads) + 1,
                    ops_conn=ops,
                    source_paths=frozenset(component_paths | spellings),
                    compare_canonical_paths=True,
                )
            (
                status,
                _ahead_count,
                _cursor_checked,
                _comparisons,
                _ahead_comparisons,
                ahead,
                _gap_count,
                gaps,
                _deferred,
                cursor_reason,
            ) = cursor
            if status == "unknown" and not gaps:
                return RawFrontierBlockedPaths(frozenset(), cursor_reason)
            for ahead_sample in ahead:
                refused.add(ahead_sample.source_path)
                ahead_canonical = ahead_sample.canonical_source_path or ahead_sample.source_path
                refused.add(ahead_canonical)
                ahead_keys = {
                    head.logical_source_key
                    for head in unique_heads
                    if head.accepted_frontier_kind == "byte"
                    and ahead_sample.cursor_byte_offset > head.accepted_frontier
                    and canonical_by_raw.get(head.accepted_raw_id) == ahead_canonical
                }
                for key in ahead_keys or ({ahead_sample.logical_source_key} if ahead_sample.logical_source_key else ()):
                    refused.update(paths_for_logical_key(key))
            if any(gap.source_path is None for gap in gaps):
                return RawFrontierBlockedPaths(frozenset(), "selected accepted head has no source path")
            return RawFrontierBlockedPaths(
                frozenset(refused),
                None,
                gap_source_paths=frozenset(gap.source_path for gap in gaps if gap.source_path is not None).difference(
                    refused
                ),
            )
    except SchemaSkew as exc:
        return RawFrontierBlockedPaths(frozenset(), f"source tier schema is not admitted: {exc}")
    except sqlite3.Error as exc:
        raise RawRetentionSafetyError(f"selected source frontier query failed: {exc}") from exc


def raw_frontier_blocked_raw_ids(archive_root: Path, raw_ids: Sequence[str]) -> RawFrontierBlockedPaths:
    """Authorize selected observations against only their source frontier.

    Archive-wide missing-source diagnostics remain the responsibility of the
    explicit integrity projection. This admission checks the selected connected
    authority component, including its byte heads, predecessor chains and cursors.
    """
    from polylogue.core.evidence import Measured, Unavailable
    from polylogue.storage.sqlite.archive_tiers.revision_governance import expand_raw_membership_selection_sync
    from polylogue.storage.sqlite.connection_profile import attach_readonly_database, open_readonly_connection
    from polylogue.storage.tier_access import capture_sqlite_read

    if not raw_ids:
        return RawFrontierBlockedPaths(frozenset(), None)

    def read_selected() -> RawFrontierBlockedPaths:
        with closing(open_readonly_connection(archive_root / "source.db")) as conn:
            conn.row_factory = sqlite3.Row
            attach_readonly_database(conn, resolve_active_index_path(archive_root), alias="index_tier")
            conn.execute("BEGIN")
            component, logical_keys = expand_raw_membership_selection_sync(conn, list(raw_ids))
            paths_by_raw = _source_paths_for_raw_ids(conn, set(component))
            if any(raw_id not in paths_by_raw for raw_id in raw_ids):
                return RawFrontierBlockedPaths(frozenset(), "selected raw has no durable source path")
            paths = frozenset(paths_by_raw.values())
            # One physical path can carry several independent authority
            # components. A broken chain or cursor-ahead head of another
            # component on a selected path refuses that path too, exactly as
            # the path-level projection would, so every head and session on
            # the selected paths is checked, not only the selected component's.
            path_raws = _raw_ids_and_keys_for_source_paths(conn, set(paths))
            checked_raw_ids = set(component) | set(path_raws)
            checked_keys = set(logical_keys) | {key for key in path_raws.values() if key is not None}
            heads = _heads_for_raws_or_keys(conn, checked_raw_ids, checked_keys)
            sessions = _session_raw_ids_among(conn, checked_raw_ids)
            broken, count, _checked, _samples, reason = _check_broken_active_chains(
                conn, sessions, heads, sample_limit=len(component) + len(heads)
            )
            if broken == "unknown":
                return RawFrontierBlockedPaths(frozenset(), reason)
            if count:
                return RawFrontierBlockedPaths(paths, None)
            ops_path = archive_root / "ops.db"
            # A reset disposable tier has no cursor claims to compare. The
            # selected durable chain above remains mandatory authority.
            if not ops_path.exists():
                return RawFrontierBlockedPaths(frozenset(), None)
            with closing(open_readonly_connection(ops_path, validate_schema=False)) as ops:
                cursor = _check_cursor_ahead_of_accepted(
                    conn, None, heads, sample_limit=len(paths) + len(heads), ops_conn=ops, source_paths=paths
                )
            status, count, _checked, _comparisons, _ahead, samples, gaps, gap_samples, _deferred, reason = cursor
            if status == "unknown" and not gaps:
                return RawFrontierBlockedPaths(frozenset(), reason)
            # A violated logical source is refused through every path that
            # carries it (rotated/resumed files, ZIP members), not only the
            # path its cursor sample named; otherwise a sibling raw publishes
            # while its logical frontier is still violated.
            ahead_keys = {sample.logical_source_key for sample in samples if sample.logical_source_key}
            refused = frozenset(
                {sample.source_path for sample in samples} | _source_paths_for_logical_keys(conn, ahead_keys)
            )
            return RawFrontierBlockedPaths(
                refused,
                None,
                gap_source_paths=frozenset(
                    sample.source_path for sample in gap_samples if sample.source_path is not None
                ).difference(refused),
            )

    try:
        evidence = capture_sqlite_read(read_selected)
    except (OSError, RawRetentionSafetyError, SchemaSkew) as exc:
        return RawFrontierBlockedPaths(frozenset(), f"selected source frontier is unreadable: {exc}")
    if isinstance(evidence, Measured):
        return evidence.value
    if isinstance(evidence, Unavailable):
        return RawFrontierBlockedPaths(
            frozenset(), f"selected source frontier is unreadable: {evidence.detail or evidence.reason}"
        )
    raise AssertionError("scoped frontier read produced an unsupported evidence state")


def raw_frontier_blocked_source_paths(
    archive_root: Path,
    raw_materialization_readiness: Mapping[str, object],
) -> RawFrontierBlockedPaths:
    """Attribute the frontier proof's refusals to exact source paths."""

    if not (archive_root / "source.db").is_file():
        # No source tier means nothing has been acquired yet, so there is
        # nothing to select and nothing to refuse. An unreadable tier still
        # refuses below: absence and damage are different states.
        return RawFrontierBlockedPaths(frozenset(), None)
    projection = raw_frontier_integrity_projection(
        archive_root,
        raw_materialization_readiness,
        sample_limit=1_000_000,
    )
    if projection.available and projection.overall_status == "healthy":
        return RawFrontierBlockedPaths(frozenset(), None)
    paths: set[str] = set()
    gap_paths: set[str] = set()
    unattributed: list[str] = []
    if projection.missing_source_raw_count:
        unattributed.append(projection.missing_source_raw_reason)
    if projection.broken_head_status == "unknown" and not projection.broken_head_count:
        unattributed.append(projection.broken_head_reason)
    if projection.cursor_ahead_status == "unknown" and not (
        projection.cursor_ahead_count or projection.cursor_authority_gap_count
    ):
        unattributed.append(projection.cursor_ahead_reason)
    blocked_logical_keys: set[str] = set()
    for ahead in projection.cursor_ahead_samples:
        paths.add(ahead.source_path)
        if ahead.logical_source_key:
            blocked_logical_keys.add(ahead.logical_source_key)
    for sample in projection.broken_head_samples:
        if sample.logical_source_key:
            blocked_logical_keys.add(sample.logical_source_key)
    for gap in projection.cursor_authority_gap_samples:
        if gap.source_path is None:
            unattributed.append(gap.reason)
        else:
            gap_paths.add(gap.source_path)
    if projection.broken_head_samples or blocked_logical_keys:
        raw_ids = {sample.accepted_raw_id for sample in projection.broken_head_samples}
        source_db_path = archive_root / "source.db"
        try:
            from polylogue.storage.sqlite.connection_profile import open_readonly_connection

            conn = open_readonly_connection(source_db_path)
        except (OSError, sqlite3.Error, SchemaSkew) as exc:
            unattributed.append(f"source tier is unreadable: {exc}")
        else:
            try:
                by_raw = _source_paths_for_raw_ids(conn, raw_ids)
                # Refuse by logical source key, not by the one physical path
                # a sample named: every sibling path of a violated logical
                # source is refused too. The reported refusal count stays
                # per-path so a caller still sees which files were held back.
                sibling_paths = (
                    _source_paths_for_logical_keys(conn, blocked_logical_keys) if blocked_logical_keys else set()
                )
            except sqlite3.Error as exc:
                unattributed.append(f"source raw path lookup failed: {exc}")
                by_raw = {}
                sibling_paths = set()
            finally:
                conn.close()
            for sample in projection.broken_head_samples:
                path = by_raw.get(sample.accepted_raw_id)
                if path is None:
                    unattributed.append(f"broken head {sample.accepted_raw_id} has no source path")
                else:
                    paths.add(path)
            paths.update(sibling_paths)
    return RawFrontierBlockedPaths(
        frozenset(paths),
        "; ".join(unattributed) or None,
        gap_source_paths=frozenset(gap_paths.difference(paths)),
    )


def unknown_raw_frontier_integrity_projection(
    reason: str,
    *,
    missing_source_raw_status: RawFrontierIntegrityStatus = "unknown",
    missing_source_raw_count: int = 0,
    missing_source_raw_samples: tuple[Mapping[str, object], ...] = (),
    missing_source_raw_reason: str | None = None,
) -> RawFrontierIntegrityProjection:
    """Return the canonical explicit-unknown projection for an unavailable read.

    Cache and presentation adapters use this instead of inventing partial
    legacy payloads. Zero-valued counts are not healthy claims because every
    check is explicitly ``unknown`` and ``available`` is false.
    """

    snapshot = _unavailable_frontier_integrity_snapshot(reason)
    statuses = (snapshot.broken_head_status, missing_source_raw_status, snapshot.cursor_ahead_status)
    return RawFrontierIntegrityProjection(
        available=False,
        overall_status=combine_raw_frontier_integrity_statuses(*statuses),
        broken_head_status=snapshot.broken_head_status,
        broken_head_count=snapshot.broken_head_count,
        broken_head_checked_count=snapshot.broken_head_checked_count,
        broken_head_samples=snapshot.broken_head_samples,
        broken_head_reason=snapshot.broken_head_reason,
        missing_source_raw_status=missing_source_raw_status,
        missing_source_raw_count=missing_source_raw_count,
        missing_source_raw_samples=missing_source_raw_samples,
        missing_source_raw_reason=reason if missing_source_raw_reason is None else missing_source_raw_reason,
        cursor_ahead_status=snapshot.cursor_ahead_status,
        cursor_ahead_count=snapshot.cursor_ahead_count,
        cursor_ahead_checked_count=snapshot.cursor_ahead_checked_count,
        cursor_head_comparison_count=snapshot.cursor_head_comparison_count,
        cursor_ahead_comparison_count=snapshot.cursor_ahead_comparison_count,
        cursor_ahead_samples=snapshot.cursor_ahead_samples,
        cursor_authority_gap_count=snapshot.cursor_authority_gap_count,
        cursor_authority_gap_samples=snapshot.cursor_authority_gap_samples,
        cursor_authority_deferred_count=snapshot.cursor_authority_deferred_count,
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
            cursor_deferred_count,
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
            cursor_authority_deferred_count=cursor_deferred_count,
            cursor_ahead_reason=cursor_reason,
        )
    finally:
        conn.row_factory = original_row_factory


def raw_frontier_integrity_snapshot_from_connections(
    source_conn: sqlite3.Connection,
    *,
    index_conn: sqlite3.Connection,
    ops_conn: sqlite3.Connection | None,
    ops_db_path: Path | None = None,
    ops_schema: str = "ops_tier",
    sample_limit: int = 10,
) -> RawFrontierIntegritySnapshot:
    """Run the frontier proof over already-pinned tier readers.

    Operation reads have a fixed publication snapshot.  Opening the index or
    ops path here would silently compare source evidence with a later
    generation, so this variant deliberately accepts the observed handles
    (with ``ops_conn=None`` representing an unavailable optional authority)
    instead.  ``ops_db_path`` is retained only for the diagnostic reason when
    that optional handle is unavailable.  It is otherwise the same proof as
    :func:`raw_frontier_integrity_snapshot`.
    """

    original_row_factory = source_conn.row_factory
    source_conn.row_factory = sqlite3.Row
    try:
        try:
            session_raw_ids, heads, _eligible = _active_index_raw_authority_from_connection(index_conn)
        except RawRetentionSafetyError as exc:
            return _unavailable_frontier_integrity_snapshot(str(exc))

        source_reason = _source_tier_unavailable_reason(source_conn)
        if source_reason is not None:
            return _unavailable_frontier_integrity_snapshot(source_reason)

        broken_status, broken_count, broken_checked, broken_samples, broken_reason = _check_broken_active_chains(
            source_conn,
            session_raw_ids,
            heads,
            sample_limit=sample_limit,
        )
        if ops_conn is None:
            # The pinned read has no ops generation. Opening ``ops_db_path``
            # now would compare this snapshot with a later, unpinned tier.
            cursor_result: tuple[
                RawFrontierIntegrityStatus,
                int,
                int,
                int,
                int,
                tuple[CursorAheadSample, ...],
                int,
                tuple[CursorAuthorityGapSample, ...],
                int,
                str,
            ] = ("unknown", 0, 0, 0, 0, (), 0, (), 0, f"ops tier is unavailable in the pinned read: {ops_db_path}")
        else:
            cursor_result = _check_cursor_ahead_of_accepted(
                source_conn,
                None,
                heads,
                sample_limit=sample_limit,
                ops_conn=ops_conn,
                ops_schema=ops_schema,
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
            cursor_deferred_count,
            cursor_reason,
        ) = cursor_result
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
            cursor_authority_deferred_count=cursor_deferred_count,
            cursor_ahead_reason=cursor_reason,
        )
    finally:
        source_conn.row_factory = original_row_factory


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
        cursor_authority_deferred_count=0,
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
    return _check_broken_active_chain_inputs(
        lambda raw_ids: _raw_revision_rows(conn, raw_ids, allow_missing=True),
        session_raw_ids,
        heads,
        sample_limit=sample_limit,
    )


def _check_cursor_ahead_of_accepted(
    conn: sqlite3.Connection,
    ops_db_path: Path | None,
    heads: tuple[_IndexRawRevisionHead, ...],
    *,
    sample_limit: int,
    ops_conn: sqlite3.Connection | None = None,
    ops_schema: str = "main",
    source_paths: frozenset[str] | None = None,
    compare_canonical_paths: bool = False,
) -> tuple[
    RawFrontierIntegrityStatus,
    int,
    int,
    int,
    int,
    tuple[CursorAheadSample, ...],
    int,
    tuple[CursorAuthorityGapSample, ...],
    int,
    str,
]:
    try:
        cursor_map = (
            _ops_cursor_byte_offsets_from_connection(ops_conn, schema=ops_schema, source_paths=source_paths)
            if ops_conn is not None
            else _ops_cursor_byte_offsets(_required_ops_path(ops_db_path))
        )
    except RawRetentionSafetyError as exc:
        return "unknown", 0, 0, 0, 0, (), 0, (), 0, str(exc)

    try:
        head_raw_ids = {head.accepted_raw_id for head in heads}
        source_path_by_raw_id = _source_paths_for_raw_ids(conn, head_raw_ids)
        canonical_by_raw_id = _canonical_paths_for_raw_ids(conn, head_raw_ids) if compare_canonical_paths else {}
    except sqlite3.Error as exc:
        logger.warning("raw frontier integrity: source raw path lookup failed: %s", exc)
        return "unknown", 0, 0, 0, 0, (), 0, (), 0, f"source raw path lookup failed: {exc}"

    try:
        retained_source_paths = _source_paths_for_paths(conn, set(cursor_map))
    except sqlite3.Error as exc:
        logger.warning("raw frontier integrity: cursor source path lookup failed: %s", exc)
        return "unknown", 0, 0, 0, 0, (), 0, (), 0, f"cursor source path lookup failed: {exc}"
    try:
        terminal_artifact_paths = _terminal_artifact_paths(conn, set(cursor_map))
    except sqlite3.Error as exc:
        logger.warning("raw frontier integrity: terminal artifact authority lookup failed: %s", exc)
        return "unknown", 0, 0, 0, 0, (), 0, (), 0, f"terminal artifact authority is unreadable: {exc}"
    return _compare_cursor_frontier_inputs(
        heads,
        cursor_map=cursor_map,
        source_path_by_raw_id=source_path_by_raw_id,
        canonical_by_raw_id=canonical_by_raw_id,
        retained_source_paths=retained_source_paths,
        terminal_artifact_paths=terminal_artifact_paths,
        compare_canonical_paths=compare_canonical_paths,
        sample_limit=sample_limit,
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


def _canonical_paths_for_raw_ids(conn: sqlite3.Connection, raw_ids: set[str]) -> dict[str, str]:
    """The canonical path each raw stored at acquisition (absent when unrecorded)."""
    result: dict[str, str] = {}
    pending = set(raw_ids)
    while pending:
        batch = tuple(sorted(pending)[:500])
        pending.difference_update(batch)
        placeholders = ", ".join("?" for _ in batch)
        for row in conn.execute(
            f"SELECT raw_id, canonical_source_path FROM raw_sessions "
            f"WHERE raw_id IN ({placeholders}) AND canonical_source_path IS NOT NULL",
            batch,
        ):
            result[str(row[0])] = str(row[1])
    return result


def _raw_ids_and_keys_for_source_paths(conn: sqlite3.Connection, source_paths: set[str]) -> dict[str, str | None]:
    """Every raw recorded at one of ``source_paths``, with its logical source key."""
    result: dict[str, str | None] = {}
    pending = set(source_paths)
    while pending:
        batch = tuple(sorted(pending)[:500])
        pending.difference_update(batch)
        placeholders = ", ".join("?" for _ in batch)
        for raw_id, logical_key in conn.execute(
            f"SELECT raw_id, logical_source_key FROM raw_sessions WHERE source_path IN ({placeholders})", batch
        ):
            result[str(raw_id)] = None if logical_key is None else str(logical_key)
    return result


def _heads_for_raws_or_keys(
    conn: sqlite3.Connection, raw_ids: set[str], logical_keys: set[str]
) -> tuple[_IndexRawRevisionHead, ...]:
    """Index heads accepting one of ``raw_ids`` or governing one of ``logical_keys``, read in pages."""
    found: dict[str, _IndexRawRevisionHead] = {}
    columns = (
        "logical_source_key, accepted_raw_id, accepted_source_revision, "
        "accepted_frontier_kind, accepted_frontier, acquisition_generation, append_end_offset"
    )
    for column, values in (("accepted_raw_id", raw_ids), ("logical_source_key", logical_keys)):
        pending = set(values)
        while pending:
            batch = tuple(sorted(pending)[:500])
            pending.difference_update(batch)
            placeholders = ", ".join("?" for _ in batch)
            for row in conn.execute(
                f"SELECT {columns} FROM index_tier.raw_revision_heads WHERE {column} IN ({placeholders})", batch
            ):
                head = _IndexRawRevisionHead(*tuple(row))
                found[head.logical_source_key] = head
    return tuple(found[key] for key in sorted(found))


def _session_raw_ids_among(conn: sqlite3.Connection, raw_ids: set[str]) -> frozenset[str]:
    """The subset of ``raw_ids`` that an index session is materialized from."""
    result: set[str] = set()
    pending = set(raw_ids)
    while pending:
        batch = tuple(sorted(pending)[:500])
        pending.difference_update(batch)
        placeholders = ", ".join("?" for _ in batch)
        result.update(
            str(row[0])
            for row in conn.execute(
                f"SELECT DISTINCT raw_id FROM index_tier.sessions WHERE raw_id IN ({placeholders})", batch
            )
        )
    return frozenset(result)


def _source_paths_for_logical_keys(conn: sqlite3.Connection, logical_keys: set[str]) -> set[str]:
    """Every durable source path that carries one of ``logical_keys``.

    A logical source is reachable through more than one physical path
    (rotated/resumed session files, ZIP-expanded members). Refusing only the
    path a violation sample happened to name would admit a sibling path whose
    logical frontier is still violated, so the gate would not be an isolation
    boundary.
    """
    result: set[str] = set()
    pending = set(logical_keys)
    while pending:
        batch = tuple(sorted(pending)[:500])
        pending.difference_update(batch)
        placeholders = ", ".join("?" for _ in batch)
        rows = conn.execute(
            f"SELECT DISTINCT source_path FROM raw_sessions WHERE logical_source_key IN ({placeholders})", batch
        ).fetchall()
        for row in rows:
            if row[0] is not None:
                result.add(str(row[0]))
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
    def read_rows(sql: str, parameters: tuple[object, ...], paths: tuple[str, ...]) -> Sequence[sqlite3.Row]:
        with closing(conn.execute(sql, parameters)) as rows:
            return rows.fetchall()

    return _terminal_artifact_paths_from_inputs(read_rows, source_paths)


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
        from polylogue.storage.sqlite.connection_profile import open_readonly_connection

        with closing(open_readonly_connection(ops_db_path.resolve(), timeout_class="background-read")) as conn:
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
    except (OSError, sqlite3.Error, SchemaSkew) as exc:
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


def _required_ops_path(path: Path | None) -> Path:
    if path is None:
        raise RawRetentionSafetyError("ops tier is unavailable")
    return path


def _ops_cursor_byte_offsets_from_connection(
    conn: sqlite3.Connection,
    *,
    schema: str = "main",
    source_paths: frozenset[str] | None = None,
) -> dict[str, _OpsCursorAuthority]:
    """Read cursor authority from an operation's pinned ops handle.

    Shares :func:`_ops_cursor_byte_offsets`'s failure contract: a
    degraded-but-readable ops tier converts to ``RawRetentionSafetyError``
    rather than escaping as a bare driver error. The typed refusal for a
    missing ``ingest_cursor`` table was copied across when this twin landed,
    but the ``(OSError, sqlite3.Error)`` conversion was not -- so the sole
    caller ``_check_cursor_ahead_of_accepted`` reads as protected (it catches
    ``RawRetentionSafetyError``) while a locked or corrupt ops handle raises
    straight through it.
    """

    if schema not in {"main", "ops_tier"}:
        raise ValueError(f"unsupported ops cursor reader schema: {schema!r}")
    try:
        return _ops_cursor_byte_offsets_from_present_connection(conn, schema=schema, source_paths=source_paths)
    except (OSError, sqlite3.Error) as exc:
        raise RawRetentionSafetyError(f"ops tier raw cursor authority is unreadable: {exc}") from exc


def _ops_cursor_byte_offsets_from_present_connection(
    conn: sqlite3.Connection,
    *,
    schema: str,
    source_paths: frozenset[str] | None,
) -> dict[str, _OpsCursorAuthority]:
    """Unguarded read body; see the wrapper for the failure contract."""

    has_table = conn.execute(
        f"SELECT 1 FROM {schema}.sqlite_schema WHERE type = 'table' AND name = 'ingest_cursor'"
    ).fetchone()
    if has_table is None:
        raise RawRetentionSafetyError("ops tier has no ingest_cursor table")
    path_filter = (
        ""
        if source_paths is None
        else f"AND (source_path IN ({','.join('?' for _ in source_paths)}) "
        f"OR canonical_source_path IN ({','.join('?' for _ in source_paths)}))"
    )
    rows = conn.execute(
        f"""
        SELECT source_path, byte_offset, deferred_end_offset, canonical_source_path
        FROM {schema}.ingest_cursor
        WHERE COALESCE(excluded, 0) = 0 AND byte_offset IS NOT NULL
        {path_filter}
        """,
        (*source_paths, *source_paths) if source_paths is not None else (),
    ).fetchall()
    return {
        str(row[0]): _OpsCursorAuthority(
            source_path=str(row[0]),
            byte_offset=int(row[1]),
            deferred_end_offset=int(row[2]) if row[2] is not None else None,
            canonical_source_path=str(row[3]) if row[3] is not None else None,
        )
        for row in rows
        if row[1] is not None
    }


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
    min_acquired_at_ms = to_epoch_ms(min_acquired_at, numeric_unit="milliseconds")
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
    index_conn: sqlite3.Connection | None = None,
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
    conn.execute(f"DELETE FROM blob_refs WHERE ref_id IN ({placeholders})", raw_ids)
    conn.execute(f"DELETE FROM raw_sessions WHERE raw_id IN ({placeholders})", raw_ids)
    conn.commit()

    store = blob_store if blob_store is not None else get_blob_store()
    deleted_blob_count = 0
    deleted_blob_bytes = 0
    errors: list[str] = []

    def main_database_path(connection: sqlite3.Connection | None) -> Path | None:
        if connection is None:
            return None
        for _seq, name, file_name in connection.execute("PRAGMA database_list"):
            if str(name) != "main":
                continue
            path = str(file_name or "")
            if not path or path == ":memory:" or path.startswith("file::memory:"):
                return None
            return Path(path)
        return None

    source_path = main_database_path(conn)
    index_path = main_database_path(index_conn)
    candidate_hashes = {candidate.blob_store_hash for candidate in candidates if len(candidate.blob_store_hash) == 64}
    if source_path is None or index_path is None:
        errors.append("source or index tier is unavailable")
    else:
        from polylogue.storage.blob_gc import unlink_unreferenced_blob_hashes_under_exclusion

        deleted_blob_count, deleted_blob_bytes, unlink_errors = unlink_unreferenced_blob_hashes_under_exclusion(
            source_path, index_path, store.root, candidate_hashes
        )
        errors.extend(unlink_errors)

    return RawSnapshotCleanupResult(
        candidate_count=len(candidates),
        deleted_raw_count=len(candidates),
        deleted_blob_count=deleted_blob_count,
        deleted_raw_bytes=raw_bytes,
        deleted_blob_bytes=deleted_blob_bytes,
        skipped_missing_source_count=0,
        skipped_referenced_count=skipped_referenced_count,
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
    index_conn: sqlite3.Connection | None = None,
) -> RawSnapshotCleanupResult:
    """Compact each path's superseded snapshots under one shared authority.

    The per-path bound (``limit_per_path``) keeps one pass proportional to the
    batch rather than to the archive. A bounded answer must still name what it
    did not reach, so a path that fills its bound is returned in
    ``residual_source_paths`` and the caller retains it as retryable backlog
    instead of dropping it.
    """
    totals = RawSnapshotCleanupResult(
        candidate_count=0,
        deleted_raw_count=0,
        deleted_blob_count=0,
        deleted_raw_bytes=0,
        deleted_blob_bytes=0,
        skipped_missing_source_count=0,
    )
    errors: list[str] = []
    residual: list[str] = []
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
            index_conn=index_conn,
        )
        errors.extend(result.errors)
        if result.candidate_count >= limit_per_path:
            residual.append(str(path))
        totals = RawSnapshotCleanupResult(
            candidate_count=totals.candidate_count + result.candidate_count,
            deleted_raw_count=totals.deleted_raw_count + result.deleted_raw_count,
            deleted_blob_count=totals.deleted_blob_count + result.deleted_blob_count,
            deleted_raw_bytes=totals.deleted_raw_bytes + result.deleted_raw_bytes,
            deleted_blob_bytes=totals.deleted_blob_bytes + result.deleted_blob_bytes,
            skipped_missing_source_count=totals.skipped_missing_source_count + result.skipped_missing_source_count,
            skipped_referenced_count=totals.skipped_referenced_count + result.skipped_referenced_count,
            errors=tuple(errors),
            residual_source_paths=tuple(residual),
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
    "active_raw_retention_authority",
    "cleanup_superseded_raw_snapshots",
    "combine_raw_frontier_integrity_statuses",
    "compact_paths_superseded_raw_snapshots",
    "missing_source_raw_integrity_status",
    "protected_active_raw_revision_ids",
    "raw_frontier_integrity_projection",
    "raw_frontier_integrity_summary",
    "raw_frontier_integrity_snapshot",
    "superseded_raw_snapshot_candidates",
    "unknown_raw_frontier_integrity_projection",
]


@dataclass(frozen=True)
class _CursorFrontierComparison:
    checked: bool
    comparison_count: int
    ahead_count: int
    representative: _IndexRawRevisionHead | None
    deferred: bool
    gap: bool


def _check_broken_active_chain_inputs(
    read_rows: Callable[[set[str]], dict[str, sqlite3.Row]],
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
        rows_by_id = read_rows(seed_raw_ids)
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
    return status, broken_count, checked_count, tuple(samples), broken_head_reason(broken_count)


def _classify_cursor_frontier_input(
    cursor: _OpsCursorAuthority,
    byte_heads: Iterable[_IndexRawRevisionHead],
    *,
    has_any_head: bool,
    terminal_artifact: bool,
) -> _CursorFrontierComparison:
    """The canonical cursor decision over every comparable original head."""
    count = 0
    ahead = 0
    representative = None
    for head in byte_heads:
        count += 1
        if cursor.byte_offset > head.accepted_frontier:
            ahead += 1
            if representative is None or (head.accepted_frontier, head.logical_source_key) < (
                representative.accepted_frontier,
                representative.logical_source_key,
            ):
                representative = head
    if cursor.is_deferred and not ahead:
        return _CursorFrontierComparison(False, 0, 0, None, True, False)
    if not count:
        return _CursorFrontierComparison(False, 0, 0, None, False, not (has_any_head or terminal_artifact))
    return _CursorFrontierComparison(True, count, ahead, representative, False, False)


def cursor_gap_sample(path: str, cursor_offset: int, *, retained: bool) -> CursorAuthorityGapSample:
    """The sample for a cursor whose path cannot be joined to an accepted byte head."""
    return CursorAuthorityGapSample(
        state="source_raws_without_accepted_head" if retained else "cursor_path_absent_from_source",
        source_path=path,
        logical_source_key=None,
        cursor_byte_offset=cursor_offset,
        reason=(
            "source tier has raw evidence but index has no accepted byte head"
            if retained
            else "ingest cursor path is absent from source tier"
        ),
    )


def cursor_ahead_sample(
    path: str, cursor: _OpsCursorAuthority, comparison: _CursorFrontierComparison
) -> CursorAheadSample:
    """The sample for a cursor committed past its representative accepted byte head."""
    representative = comparison.representative
    if representative is None:
        raise ValueError("a cursor ahead of accepted raw names no representative head")
    return CursorAheadSample(
        source_path=path,
        logical_source_key=representative.logical_source_key,
        cursor_byte_offset=cursor.byte_offset,
        accepted_frontier=representative.accepted_frontier,
        affected_head_count=comparison.ahead_count,
        canonical_source_path=cursor.canonical_source_path,
    )


def _compare_cursor_frontier_inputs(
    heads: tuple[_IndexRawRevisionHead, ...],
    *,
    cursor_map: dict[str, _OpsCursorAuthority],
    source_path_by_raw_id: dict[str, str],
    canonical_by_raw_id: dict[str, str],
    retained_source_paths: set[str],
    terminal_artifact_paths: set[str],
    compare_canonical_paths: bool,
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
    int,
    str,
]:
    """The canonical cursor comparison over explicitly retained inputs."""
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
        # Compare the canonical path stored at acquisition. Re-resolving an
        # obsolete spelling against today's filesystem names the wrong file
        # once its symlink is removed or retargeted.
        comparison_path = canonical_by_raw_id.get(head.accepted_raw_id, source_path)
        all_head_paths.add(comparison_path)
        if head.accepted_frontier_kind == "byte":
            byte_heads_by_path.setdefault(comparison_path, []).append(head)

    samples: list[CursorAheadSample] = []
    ahead_count = 0
    checked = 0
    comparison_count = 0
    ahead_comparison_count = 0
    deferred_count = 0
    for path, cursor in cursor_map.items():
        comparison_path = (cursor.canonical_source_path or path) if compare_canonical_paths else path
        cursor_offset = cursor.byte_offset
        comparison = _classify_cursor_frontier_input(
            cursor,
            byte_heads_by_path.get(comparison_path, ()),
            has_any_head=comparison_path in all_head_paths,
            terminal_artifact=path in terminal_artifact_paths,
        )
        if comparison.deferred:
            deferred_count += 1
            continue
        if not comparison.checked:
            if not comparison.gap:
                continue
            gap_count += 1
            if len(gaps) < sample_limit:
                gaps.append(cursor_gap_sample(path, cursor_offset, retained=path in retained_source_paths))
            continue
        checked += 1
        comparison_count += comparison.comparison_count
        if not comparison.ahead_count:
            continue
        ahead_count += 1
        ahead_comparison_count += comparison.ahead_count
        if len(samples) < sample_limit:
            samples.append(cursor_ahead_sample(path, cursor, comparison))

    status: RawFrontierIntegrityStatus = "violated" if ahead_count else "unknown" if gap_count else "healthy"
    return (
        status,
        ahead_count,
        checked,
        comparison_count,
        ahead_comparison_count,
        tuple(samples),
        gap_count,
        tuple(gaps),
        deferred_count,
        cursor_ahead_reason(ahead_count, ahead_comparison_count, gap_count, deferred_count),
    )


def _raw_revision_rows_from_inputs(
    read_rows: Callable[[tuple[str, ...]], Sequence[sqlite3.Row]],
    raw_ids: set[str],
    *,
    allow_missing: bool = False,
) -> dict[str, sqlite3.Row]:
    rows_by_id: dict[str, sqlite3.Row] = {}
    pending = set(raw_ids)
    while pending:
        batch = tuple(sorted(pending)[:500])
        pending.difference_update(batch)
        try:
            rows = read_rows(batch)
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


def _terminal_artifact_paths_from_inputs(
    read_rows: Callable[[str, tuple[object, ...], tuple[str, ...]], Sequence[sqlite3.Row]],
    source_paths: set[str],
) -> set[str]:
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
    # The path batch binds once for observation receipts and once for raw rows.
    path_batch_size = max(1, (500 - len(raw_failure_kinds) - len(terminal_raw_failure_kinds)) // 2)
    pending = set(source_paths)
    while pending:
        batch = tuple(sorted(pending)[:path_batch_size])
        pending.difference_update(batch)
        placeholders = ", ".join("?" for _ in batch)
        rows = read_rows(
            f"""
            WITH latest_raw_observation AS (
                SELECT raw_id, acquired_at_ms, observation_rowid
                FROM (
                    SELECT
                        ref_id AS raw_id,
                        acquired_at_ms,
                        rowid AS observation_rowid,
                        ROW_NUMBER() OVER (
                            PARTITION BY ref_id
                            ORDER BY rowid DESC
                        ) AS observation_rank
                    FROM blob_refs
                    WHERE ref_type = 'raw_payload'
                      AND source_path IN ({placeholders})
                )
                WHERE observation_rank = 1
            ),
            newest_per_coordinate AS (
                SELECT raw_id, source_path, origin, source_index, parse_error,
                       validation_status, validated_at_ms, parsed_at_ms
                FROM (
                    SELECT
                        raw.raw_id,
                        raw.source_path,
                        raw.origin,
                        raw.source_index,
                        raw.parse_error,
                        raw.validation_status,
                        raw.validated_at_ms,
                        raw.parsed_at_ms,
                        ROW_NUMBER() OVER (
                            PARTITION BY raw.source_path, raw.origin, raw.source_index
                            ORDER BY COALESCE(observation.observation_rowid, raw.rowid) DESC
                        ) AS coordinate_rank
                    FROM raw_sessions AS raw
                    LEFT JOIN latest_raw_observation AS observation ON observation.raw_id = raw.raw_id
                    WHERE raw.source_path IN ({placeholders})
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
            ),
            terminal_evidence AS (
                SELECT raw_id FROM terminal_artifacts
                UNION
                SELECT evidence_raw.raw_id
                FROM newest_per_coordinate AS evidence_raw
                JOIN raw_membership_census AS census ON census.raw_id = evidence_raw.raw_id
                WHERE census.status = 'non_session'
            )
            SELECT DISTINCT terminal_raw.source_path
            FROM terminal_evidence AS artifact
            JOIN newest_per_coordinate AS terminal_raw ON terminal_raw.raw_id = artifact.raw_id
            WHERE NOT EXISTS (
                SELECT 1
                FROM newest_per_coordinate AS coordinate
                WHERE coordinate.source_path = terminal_raw.source_path
                  AND NOT EXISTS (
                      SELECT 1
                      FROM terminal_evidence AS current_artifact
                      WHERE current_artifact.raw_id = coordinate.raw_id
                  )
            )
            """,
            (*batch, *batch, *raw_failure_kinds, *terminal_raw_failure_kinds),
            batch,
        )
        result.update(str(row[0]) for row in rows)
    return result

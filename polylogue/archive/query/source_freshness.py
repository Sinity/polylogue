"""Bounded, exact-source freshness evidence from filesystem bytes to search.

The projection in this module is deliberately read-only and source-keyed.  It
never derives raw revision authority and never decides whether a revision may
be replayed: those semantics remain owned by ``polylogue-lkrc`` and
``polylogue-yla8`` respectively.  This reader only reports their persisted
ledger fields alongside cursor, parse, index, FTS, and convergence evidence.

Every archive relation is queried with an exact source/raw/session key, a row
limit, and a SQLite VM progress budget.  A data-table query whose plan would
scan the whole relation is rejected instead of being allowed to turn an
operator receipt into an archive-wide census.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Final, Literal, cast
from urllib.parse import quote

from polylogue.core.dates import utc_now
from polylogue.core.evidence_value import (
    CoverageExclusion,
    EvidenceValue,
    FactFamilySpec,
    FrameCoverage,
    FreshnessProvenance,
    TemporalProvenance,
    ValueState,
    sum_evidence_values,
)
from polylogue.core.refs import ObjectRef

_RAW_AUTHORITY_OWNER: Final = "polylogue-lkrc"
_REPLAY_PREVENTION_OWNER: Final = "polylogue-yla8"

_SOURCE_CURSOR_BYTE_LAG_DEFINITION_REF: Final = ObjectRef(
    kind="insight",
    object_id="source-cursor-byte-lag:v1",
)
SOURCE_CURSOR_BYTE_LAG_FAMILY: Final = FactFamilySpec(
    family="archive.source_cursor_byte_lag",
    owner="polylogue.archive.query.source_freshness",
    source_adapter="project_named_source_freshness",
    public_field="byte_lag",
    renderer_label="cursor byte lag",
    value_schema="integer",
    unit="bytes",
    grain="source_path",
    denominator="declared exact source paths",
    definition_ref=_SOURCE_CURSOR_BYTE_LAG_DEFINITION_REF,
    required_axes=frozenset(
        {
            "value_state",
            "measurement_authority",
            "evidence_refs",
            "definition_ref",
            "temporal",
            "enumeration",
            "coverage",
            "freshness",
        }
    ),
    allowed_states=frozenset({"known", "unknown", "unavailable"}),
    allowed_authorities=frozenset({"structural"}),
    authority_precedence=("structural",),
    requires_last_good_when_degraded=True,
)


class NamedSourceStage(str, Enum):
    """Monotone named-source path used by miss diagnostics."""

    UNSEEN = "unseen"
    ACQUIRED_UNPARSED = "acquired-unparsed"
    PARSED_UNINDEXED = "parsed-unindexed"
    INDEXED_UNCONVERGED = "indexed-unconverged"
    SEARCHABLE = "searchable"


class NamedSourceOperationalState(str, Enum):
    """Operational state, intentionally separate from pipeline stage."""

    UNSEEN = "unseen"
    ACTIVE = "active"
    IDLE = "idle"
    DEGRADED = "degraded"


class NamedSourceOperationalReason(str, Enum):
    """Exact reason for the operational state classification."""

    SOURCE_STAT_ERROR = "source-stat-error"
    SOURCE_MISSING = "source-missing"
    CURSOR_EXCLUDED = "cursor-excluded"
    CURSOR_RETRYING = "cursor-retrying"
    CURSOR_AHEAD = "cursor-ahead"
    BROKEN_HEAD = "broken-head"
    PENDING_BYTES = "pending-bytes"
    CAUGHT_UP = "caught-up"
    CURSOR_MISSING = "cursor-missing"
    NO_EVIDENCE = "no-evidence"


ParseState = Literal["unseen", "pending", "failed", "parsed"]
CursorState = Literal["unseen", "excluded", "retrying", "ahead", "behind", "idle"]


@dataclass(frozen=True, slots=True)
class ProjectionLimits:
    """Hard bounds for one exact-source receipt."""

    max_raw_revisions: int = 16
    max_sessions: int = 64
    max_messages: int = 256
    max_blocks: int = 512
    max_attempt_rows: int = 8
    max_application_rows: int = 8
    max_debt_rows: int = 32
    sqlite_vm_steps: int = 250_000
    sqlite_value_bytes: int = 64 * 1024
    busy_timeout_ms: int = 150
    attempt_tail_bytes: int = 64 * 1024
    cursor_export_bytes: int = 2 * 1024 * 1024

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field.name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class SourceStatEvidence:
    exists: bool
    size_bytes: int | None = None
    mtime_ns: int | None = None
    device: int | None = None
    inode: int | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CursorEvidence:
    present: bool
    state: CursorState
    source: str | None = None
    observed_size_bytes: int | None = None
    byte_offset: int | None = None
    pending_bytes: int | None = None
    unobserved_growth_bytes: int | None = None
    cursor_ahead_bytes: int | None = None
    observed_size_ahead_bytes: int | None = None
    failure_count: int = 0
    excluded: bool = False
    next_retry_at: str | None = None
    updated_at_ms: int | None = None
    age_ms: int | None = None


@dataclass(frozen=True, slots=True)
class RetryEvidence:
    reason: str | None = None
    reason_source: str | None = None
    attempt_id: str | None = None
    attempt_status: str | None = None
    attempt_phase: str | None = None
    observed_at_ms: int | None = None


@dataclass(frozen=True, slots=True)
class RawRevisionEvidence:
    raw_id: str
    origin: str | None
    native_id: str | None
    source_index: int | None
    blob_hash: str | None
    validation_status: str | None
    parse_error: str | None
    parsed_at_ms: int | None
    observed_at_ms: int | None
    revision_authority: str | None
    accepted_by_acquisition: bool
    authority_owner: str = _RAW_AUTHORITY_OWNER


@dataclass(frozen=True, slots=True)
class ParseEvidence:
    state: ParseState
    raw_id: str | None = None
    parsed_at_ms: int | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RevisionApplicationEvidence:
    raw_id: str
    decision: str | None
    detail: str | None
    observed_at_ms: int | None
    owner: str = _REPLAY_PREVENTION_OWNER


@dataclass(frozen=True, slots=True)
class IndexEvidence:
    available: bool
    accepted_raw_indexed: bool = False
    broken_head: bool = False
    accepted_session_ids: tuple[str, ...] = ()
    source_session_ids: tuple[str, ...] = ()
    session_count_lower_bound: int = 0
    sessions_truncated: bool = False
    source_raw_scope_truncated: bool = False
    high_water_ms: int | None = None
    high_water_column: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class FtsEvidence:
    available: bool
    converged: bool = False
    recorded_state: str | None = None
    checked_at: str | None = None
    triggers_present: bool | None = None
    source_searchable_blocks: int = 0
    indexed_searchable_blocks: int = 0
    blocks_truncated: bool = False
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class InsightEvidence:
    available: bool
    converged: bool = False
    state: str = "unavailable"
    debt_count_lower_bound: int = 0
    debt_stages: tuple[str, ...] = ()
    debt_errors: tuple[str, ...] = ()
    debt_truncated: bool = False
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class QueryPlanEvidence:
    database: str
    label: str
    details: tuple[str, ...]
    safe: bool


@dataclass(frozen=True, slots=True)
class QueryReceipt:
    read_only: bool
    exact_source: bool
    archive_wide_aggregates: bool
    query_count: int
    sqlite_vm_steps_per_query: int
    sqlite_value_bytes: int
    max_rows_per_relation: int
    query_plans: tuple[QueryPlanEvidence, ...]
    unsafe_scan_rejections: tuple[str, ...]
    attempt_tail_bytes_read: int
    attempt_tail_lines_examined: int
    cursor_export_bytes_read: int


@dataclass(frozen=True, slots=True)
class OwnershipBoundary:
    raw_authority_owner: str = _RAW_AUTHORITY_OWNER
    replay_prevention_owner: str = _REPLAY_PREVENTION_OWNER
    projection_role: str = "read-only-observer"


@dataclass(frozen=True, slots=True)
class NamedSourceFreshness:
    """Complete typed projection for exactly one source path."""

    source_path: str
    observed_at: str
    stage: NamedSourceStage
    operational_state: NamedSourceOperationalState
    operational_reason: NamedSourceOperationalReason
    source_stat: SourceStatEvidence
    cursor: CursorEvidence
    byte_lag: EvidenceValue[int]
    retry: RetryEvidence
    accepted_raw_revision: RawRevisionEvidence | None
    raw_revisions: tuple[RawRevisionEvidence, ...]
    raw_revisions_truncated: bool
    parse: ParseEvidence
    revision_applications: tuple[RevisionApplicationEvidence, ...]
    revision_applications_truncated: bool
    index: IndexEvidence
    fts: FtsEvidence
    insights: InsightEvidence
    ownership: OwnershipBoundary
    receipt: QueryReceipt
    errors: tuple[str, ...]
    projection_sha256: str

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _jsonable(self))


@dataclass(slots=True)
class _ReceiptBuilder:
    query_count: int = 0
    plans: list[QueryPlanEvidence] | None = None
    scan_rejections: list[str] | None = None
    tail_bytes_read: int = 0
    tail_lines_examined: int = 0
    cursor_export_bytes_read: int = 0

    def __post_init__(self) -> None:
        if self.plans is None:
            self.plans = []
        if self.scan_rejections is None:
            self.scan_rejections = []


class _ReadonlyDatabase(AbstractContextManager["_ReadonlyDatabase"]):
    """Small read-only SQLite wrapper with a per-query VM budget."""

    def __init__(
        self,
        path: Path,
        *,
        label: str,
        limits: ProjectionLimits,
        receipt: _ReceiptBuilder,
    ) -> None:
        self.path = path
        self.label = label
        self.limits = limits
        self.receipt = receipt
        self.conn: sqlite3.Connection | None = None
        self._progress_ticks = 0

    def __enter__(self) -> _ReadonlyDatabase:
        uri = f"file:{quote(str(self.path.resolve()))}?mode=ro"
        conn = sqlite3.connect(
            uri,
            uri=True,
            timeout=max(self.limits.busy_timeout_ms, 1) / 1000.0,
        )
        conn.row_factory = sqlite3.Row
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, self.limits.sqlite_value_bytes)
        conn.execute("PRAGMA query_only = ON")
        conn.execute(f"PRAGMA busy_timeout = {max(self.limits.busy_timeout_ms, 1)}")
        self.conn = conn
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.conn is not None:
            self.conn.set_progress_handler(None, 0)
            self.conn.close()
            self.conn = None

    def _connection(self) -> sqlite3.Connection:
        if self.conn is None:
            raise RuntimeError("read-only database is not open")
        return self.conn

    def _start_query_budget(self) -> None:
        self._progress_ticks = 0
        interval = min(1_000, self.limits.sqlite_vm_steps)
        tick_limit = max(self.limits.sqlite_vm_steps // interval, 1)

        def stop_after_budget() -> int:
            self._progress_ticks += 1
            return int(self._progress_ticks >= tick_limit)

        self._connection().set_progress_handler(stop_after_budget, interval)

    def _finish_query_budget(self) -> None:
        self._connection().set_progress_handler(None, 0)

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> list[sqlite3.Row]:
        self.receipt.query_count += 1
        self._start_query_budget()
        try:
            return self._connection().execute(sql, params).fetchall()
        finally:
            self._finish_query_budget()

    def one(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Row | None:
        rows = self.execute(sql, params)
        return rows[0] if rows else None

    def table_exists(self, table: str) -> bool:
        row = self.one(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ? LIMIT 1",
            (table,),
        )
        return row is not None

    def columns(self, table: str) -> frozenset[str]:
        if not _safe_identifier(table):
            return frozenset()
        rows = self.execute(f"PRAGMA table_xinfo({_quote_identifier(table)})")
        return frozenset(str(row[1]) for row in rows)

    def exact_rows(
        self,
        *,
        label: str,
        sql: str,
        params: tuple[object, ...],
        protected_tables: tuple[str, ...],
    ) -> list[sqlite3.Row] | None:
        """Execute only when EXPLAIN shows no full scan of protected tables."""
        self.receipt.query_count += 1
        self._start_query_budget()
        try:
            plan_rows = self._connection().execute(f"EXPLAIN QUERY PLAN {sql}", params).fetchall()
        finally:
            self._finish_query_budget()
        details = tuple(str(row[3]) for row in plan_rows)
        unsafe = _unsafe_plan_details(details, protected_tables)
        safe = not unsafe
        assert self.receipt.plans is not None
        self.receipt.plans.append(QueryPlanEvidence(database=self.label, label=label, details=details, safe=safe))
        if unsafe:
            rejection = f"{self.label}:{label}: " + "; ".join(unsafe)
            assert self.receipt.scan_rejections is not None
            self.receipt.scan_rejections.append(rejection)
            return None
        return self.execute(sql, params)


def project_named_source_freshness(
    archive_root: Path,
    source_path: Path,
    *,
    now: datetime | None = None,
    cursor_export: Path | None = None,
    attempt_log: Path | None = None,
    limits: ProjectionLimits | None = None,
) -> NamedSourceFreshness:
    """Project one exact source from filesystem stat through search evidence.

    ``archive_root`` is read only. ``cursor_export`` and ``attempt_log`` are
    optional operator-provided fallbacks for deployments where the live ops
    database is unavailable.  No directory walk is performed.
    """

    resolved_limits = limits or ProjectionLimits()
    observed = _as_utc(now or utc_now())
    source_key = os.fspath(source_path)
    receipt = _ReceiptBuilder()
    errors: list[str] = []

    source_stat = _stat_source(source_path)
    if source_stat.error is not None:
        errors.append(f"source stat failed: {source_stat.error}")
    cursor = _load_cursor(
        archive_root,
        source_key,
        source_stat=source_stat,
        observed=observed,
        cursor_export=cursor_export,
        limits=resolved_limits,
        receipt=receipt,
        errors=errors,
    )
    byte_lag = _source_cursor_byte_lag(source_key, source_stat, cursor, observed=observed)
    retry = _load_retry_evidence(
        archive_root,
        source_key,
        cursor=cursor,
        attempt_log=attempt_log,
        limits=resolved_limits,
        receipt=receipt,
        errors=errors,
    )
    raw_revisions, raw_truncated, accepted_raw = _load_raw_revisions(
        archive_root,
        source_key,
        limits=resolved_limits,
        receipt=receipt,
        errors=errors,
    )
    parse = _parse_evidence(accepted_raw)
    applications, applications_truncated = _load_revision_applications(
        archive_root,
        accepted_raw,
        limits=resolved_limits,
        receipt=receipt,
        errors=errors,
    )
    index = _load_index_evidence(
        archive_root,
        accepted_raw,
        raw_revisions,
        raw_scope_truncated=raw_truncated,
        limits=resolved_limits,
        receipt=receipt,
        errors=errors,
    )
    fts = _load_fts_evidence(
        archive_root,
        index,
        limits=resolved_limits,
        receipt=receipt,
        errors=errors,
    )
    insights = _load_insight_evidence(
        archive_root,
        source_key,
        index,
        limits=resolved_limits,
        receipt=receipt,
        errors=errors,
    )

    stage = _classify_stage(accepted_raw, parse, index, fts, insights)
    operational_state, operational_reason = _classify_operational_state(source_stat, cursor, accepted_raw, index)
    max_rows = max(
        resolved_limits.max_raw_revisions,
        resolved_limits.max_sessions,
        resolved_limits.max_messages,
        resolved_limits.max_blocks,
        resolved_limits.max_attempt_rows,
        resolved_limits.max_application_rows,
        resolved_limits.max_debt_rows,
    )
    query_receipt = QueryReceipt(
        read_only=True,
        exact_source=True,
        archive_wide_aggregates=False,
        query_count=receipt.query_count,
        sqlite_vm_steps_per_query=resolved_limits.sqlite_vm_steps,
        sqlite_value_bytes=resolved_limits.sqlite_value_bytes,
        # Every bounded data query may fetch one sentinel row to prove truncation.
        max_rows_per_relation=max_rows + 1,
        query_plans=tuple(receipt.plans or ()),
        unsafe_scan_rejections=tuple(receipt.scan_rejections or ()),
        attempt_tail_bytes_read=receipt.tail_bytes_read,
        attempt_tail_lines_examined=receipt.tail_lines_examined,
        cursor_export_bytes_read=receipt.cursor_export_bytes_read,
    )

    base_payload = {
        "source_path": source_key,
        "observed_at": observed.isoformat(),
        "stage": stage.value,
        "operational_state": operational_state.value,
        "operational_reason": operational_reason.value,
        "source_stat": _jsonable(source_stat),
        "cursor": _jsonable(cursor),
        "byte_lag": _jsonable(byte_lag),
        "retry": _jsonable(retry),
        "accepted_raw_revision": _jsonable(accepted_raw),
        "raw_revisions": _jsonable(raw_revisions),
        "raw_revisions_truncated": raw_truncated,
        "parse": _jsonable(parse),
        "revision_applications": _jsonable(applications),
        "revision_applications_truncated": applications_truncated,
        "index": _jsonable(index),
        "fts": _jsonable(fts),
        "insights": _jsonable(insights),
        "ownership": _jsonable(OwnershipBoundary()),
        "receipt": _jsonable(query_receipt),
        "errors": list(errors),
    }
    digest = hashlib.sha256(
        json.dumps(
            base_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return NamedSourceFreshness(
        source_path=source_key,
        observed_at=observed.isoformat(),
        stage=stage,
        operational_state=operational_state,
        operational_reason=operational_reason,
        source_stat=source_stat,
        cursor=cursor,
        byte_lag=byte_lag,
        retry=retry,
        accepted_raw_revision=accepted_raw,
        raw_revisions=raw_revisions,
        raw_revisions_truncated=raw_truncated,
        parse=parse,
        revision_applications=applications,
        revision_applications_truncated=applications_truncated,
        index=index,
        fts=fts,
        insights=insights,
        ownership=OwnershipBoundary(),
        receipt=query_receipt,
        errors=tuple(errors),
        projection_sha256=digest,
    )


def aggregate_named_source_byte_lag(
    freshnesses: Iterable[NamedSourceFreshness],
    *,
    expected_source_paths: Sequence[str | Path],
    now: datetime | None = None,
) -> EvidenceValue[int]:
    """Conserve byte-lag totals across a declared exact-source frame.

    Repeated projections for the same source deduplicate by ``file`` identity;
    contradictory duplicates, missing sources, or unknown lag keep the total
    unknown instead of contributing a numeric zero.
    """

    expected_refs = tuple(
        sorted(
            {ObjectRef(kind="file", object_id=os.fspath(path)) for path in expected_source_paths},
            key=lambda ref: ref.format(),
        )
    )
    normalized = "\n".join(ref.format() for ref in expected_refs)
    aggregate_ref = ObjectRef(
        kind="insight",
        object_id="source-cursor-byte-lag-total:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    )
    observed_at = _as_utc(now or utc_now()).isoformat()
    return sum_evidence_values(
        (freshness.byte_lag for freshness in freshnesses),
        spec=SOURCE_CURSOR_BYTE_LAG_FAMILY,
        fact_ref=aggregate_ref,
        observed_at=observed_at,
        intended_frame="named source byte lag for exact source paths",
        expected_fact_refs=expected_refs,
    )


def _source_cursor_byte_lag(
    source_key: str,
    source_stat: SourceStatEvidence,
    cursor: CursorEvidence,
    *,
    observed: datetime,
) -> EvidenceValue[int]:
    fact_ref = ObjectRef(kind="file", object_id=source_key)
    cursor_ref = ObjectRef(kind="run", object_id=f"ingest-cursor:{source_key}")
    evidence_refs = (fact_ref, cursor_ref) if cursor.present else (fact_ref,)
    exclusions = (CoverageExclusion(subject_ref=fact_ref, reason="ingest-cursor-excluded"),) if cursor.excluded else ()

    value_state: ValueState
    if source_stat.error is not None:
        value_state = "unavailable"
        value = None
        freshness = FreshnessProvenance(
            state="unavailable",
            evaluated_at=observed.isoformat(),
            cause="source-stat-error",
        )
    elif cursor.pending_bytes is None:
        value_state = "unknown"
        value = None
        if cursor.present:
            freshness = FreshnessProvenance(
                state="degraded",
                evaluated_at=observed.isoformat(),
                cause="source-size-or-cursor-offset-unavailable",
                last_good_at=_timestamp_iso(cursor.updated_at_ms),
                last_good_evidence_refs=(cursor_ref,),
            )
        else:
            freshness = FreshnessProvenance(
                state="unavailable",
                evaluated_at=observed.isoformat(),
                cause="cursor-missing",
            )
    else:
        value_state = "known"
        value = cursor.pending_bytes
        degradation_cause: str | None = None
        if cursor.excluded:
            degradation_cause = "cursor-excluded"
        elif cursor.failure_count > 0:
            degradation_cause = "cursor-retrying"
        elif (cursor.cursor_ahead_bytes or 0) > 0 or (cursor.observed_size_ahead_bytes or 0) > 0:
            degradation_cause = "cursor-ahead"
        elif not source_stat.exists:
            degradation_cause = "source-missing"
        if degradation_cause is None:
            freshness = FreshnessProvenance(
                state="fresh",
                evaluated_at=observed.isoformat(),
            )
        else:
            freshness = FreshnessProvenance(
                state="degraded",
                evaluated_at=observed.isoformat(),
                cause=degradation_cause,
                last_good_at=_timestamp_iso(cursor.updated_at_ms),
                last_good_evidence_refs=(cursor_ref,),
            )

    evidence = EvidenceValue(
        family=SOURCE_CURSOR_BYTE_LAG_FAMILY.family,
        fact_ref=fact_ref,
        value_state=value_state,
        value=value,
        measurement_authority=("structural",),
        weakest_measurement_authority="structural",
        evidence_refs=evidence_refs,
        definition_ref=SOURCE_CURSOR_BYTE_LAG_FAMILY.definition_ref,
        temporal=TemporalProvenance.from_source(
            observed_at=observed.isoformat(),
            time_source="materialization_ts",
        ),
        enumeration="census",
        coverage=FrameCoverage(
            intended_frame="one exact source path",
            grain=SOURCE_CURSOR_BYTE_LAG_FAMILY.grain,
            denominator=SOURCE_CURSOR_BYTE_LAG_FAMILY.denominator,
            intended_count=1,
            observed_count=1,
            supported_count=1 if value_state == "known" else 0,
            complete=value_state == "known" and not cursor.excluded,
            intended_refs=(fact_ref,),
            observed_refs=(fact_ref,),
            exclusions=exclusions,
        ),
        freshness=freshness,
    )
    SOURCE_CURSOR_BYTE_LAG_FAMILY.require(cast(EvidenceValue[object], evidence))
    return evidence


def _timestamp_iso(value_ms: int | None) -> str | None:
    if value_ms is None:
        return None
    return datetime.fromtimestamp(value_ms / 1000.0, tz=UTC).isoformat()


def _stat_source(source_path: Path) -> SourceStatEvidence:
    try:
        stat = source_path.stat()
    except FileNotFoundError:
        return SourceStatEvidence(exists=False)
    except OSError as exc:
        return SourceStatEvidence(exists=False, error=f"{type(exc).__name__}: {exc}")
    return SourceStatEvidence(
        exists=True,
        size_bytes=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        device=int(stat.st_dev),
        inode=int(stat.st_ino),
    )


def _load_cursor(
    archive_root: Path,
    source_key: str,
    *,
    source_stat: SourceStatEvidence,
    observed: datetime,
    cursor_export: Path | None,
    limits: ProjectionLimits,
    receipt: _ReceiptBuilder,
    errors: list[str],
) -> CursorEvidence:
    for db_path, db_label, table in (
        (archive_root / "ops.db", "ops.db", "ingest_cursor"),
        (archive_root / "index.db", "index.db", "live_cursor"),
    ):
        if not db_path.exists():
            continue
        try:
            with _ReadonlyDatabase(db_path, label=db_label, limits=limits, receipt=receipt) as db:
                if not db.table_exists(table):
                    continue
                columns = db.columns(table)
                row = _exact_cursor_row(db, table, columns, source_key)
                if row is not None:
                    return _cursor_from_mapping(
                        dict(row),
                        source=db_label,
                        source_stat=source_stat,
                        observed=observed,
                    )
        except sqlite3.Error as exc:
            errors.append(f"{db_label} cursor read failed: {exc}")

    if cursor_export is not None:
        exported = _cursor_from_export(
            cursor_export,
            source_key,
            source_stat=source_stat,
            observed=observed,
            limits=limits,
            receipt=receipt,
            errors=errors,
        )
        if exported is not None:
            return exported
    return CursorEvidence(present=False, state="unseen")


def _exact_cursor_row(
    db: _ReadonlyDatabase,
    table: str,
    columns: frozenset[str],
    source_key: str,
) -> sqlite3.Row | None:
    if "source_path" not in columns:
        return None
    names = (
        "source_path",
        "stat_size" if "stat_size" in columns else "byte_size",
        "byte_offset",
        "failure_count",
        "excluded",
        "next_retry_at",
        "updated_at_ms" if "updated_at_ms" in columns else "updated_at",
    )
    aliases = (
        "source_path",
        "observed_size",
        "byte_offset",
        "failure_count",
        "excluded",
        "next_retry_at",
        "updated_at",
    )
    selected = [
        (f"{_quote_identifier(name)} AS {_quote_identifier(alias)}" if name in columns else f"NULL AS {alias}")
        for name, alias in zip(names, aliases, strict=True)
    ]
    sql = f"SELECT {', '.join(selected)} FROM {_quote_identifier(table)} WHERE source_path = ? LIMIT 1"
    rows = db.exact_rows(
        label=f"{table}-by-source-path",
        sql=sql,
        params=(source_key,),
        protected_tables=(table,),
    )
    return rows[0] if rows else None


def _cursor_from_mapping(
    row: dict[str, object],
    *,
    source: str,
    source_stat: SourceStatEvidence,
    observed: datetime,
) -> CursorEvidence:
    observed_size = _optional_int(row.get("observed_size"))
    byte_offset = _optional_int(row.get("byte_offset"))
    failure_count = _optional_int(row.get("failure_count")) or 0
    excluded_value = row.get("excluded")
    excluded = excluded_value if isinstance(excluded_value, bool) else bool(_optional_int(excluded_value) or False)
    updated_at_ms = _timestamp_ms(row.get("updated_at"))
    actual_size = source_stat.size_bytes
    pending_bytes = None if actual_size is None or byte_offset is None else max(actual_size - byte_offset, 0)
    unobserved_growth = None if actual_size is None or observed_size is None else max(actual_size - observed_size, 0)
    cursor_ahead = None if actual_size is None or byte_offset is None else max(byte_offset - actual_size, 0)
    observed_size_ahead = None if actual_size is None or observed_size is None else max(observed_size - actual_size, 0)
    if excluded:
        state: CursorState = "excluded"
    elif failure_count > 0:
        state = "retrying"
    elif (cursor_ahead or 0) > 0 or (observed_size_ahead or 0) > 0:
        state = "ahead"
    elif pending_bytes is not None and pending_bytes > 0:
        state = "behind"
    else:
        state = "idle"
    return CursorEvidence(
        present=True,
        state=state,
        source=source,
        observed_size_bytes=observed_size,
        byte_offset=byte_offset,
        pending_bytes=pending_bytes,
        unobserved_growth_bytes=unobserved_growth,
        cursor_ahead_bytes=cursor_ahead,
        observed_size_ahead_bytes=observed_size_ahead,
        failure_count=failure_count,
        excluded=excluded,
        next_retry_at=_optional_str(row.get("next_retry_at")),
        updated_at_ms=updated_at_ms,
        age_ms=None if updated_at_ms is None else max(int(observed.timestamp() * 1000) - updated_at_ms, 0),
    )


def _cursor_from_export(
    path: Path,
    source_key: str,
    *,
    source_stat: SourceStatEvidence,
    observed: datetime,
    limits: ProjectionLimits,
    receipt: _ReceiptBuilder,
    errors: list[str],
) -> CursorEvidence | None:
    try:
        with path.open("rb") as stream:
            raw = stream.read(limits.cursor_export_bytes + 1)
        receipt.cursor_export_bytes_read += len(raw)
        if len(raw) > limits.cursor_export_bytes:
            errors.append(f"cursor export rejected: content exceeds {limits.cursor_export_bytes}-byte bound")
            return None
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cursor export read failed: {exc}")
        return None
    candidates: list[dict[str, object]] = []
    if isinstance(payload, dict):
        direct = payload.get(source_key)
        if isinstance(direct, dict):
            candidates.append(cast(dict[str, object], direct))
        records = payload.get("records")
        if isinstance(records, list):
            candidates.extend(
                cast(
                    list[dict[str, object]],
                    [item for item in records if isinstance(item, dict)],
                )
            )
        if payload.get("source_path") == source_key:
            candidates.append(cast(dict[str, object], payload))
    elif isinstance(payload, list):
        candidates.extend(
            cast(
                list[dict[str, object]],
                [item for item in payload if isinstance(item, dict)],
            )
        )
    for candidate in candidates:
        if candidate.get("source_path") not in (None, source_key):
            continue
        normalized = dict(candidate)
        if "observed_size" not in normalized:
            normalized["observed_size"] = normalized.get("stat_size", normalized.get("byte_size"))
        if "updated_at" not in normalized:
            normalized["updated_at"] = normalized.get("updated_at_ms")
        return _cursor_from_mapping(
            normalized,
            source="cursor-export",
            source_stat=source_stat,
            observed=observed,
        )
    return None


def _load_retry_evidence(
    archive_root: Path,
    source_key: str,
    *,
    cursor: CursorEvidence,
    attempt_log: Path | None,
    limits: ProjectionLimits,
    receipt: _ReceiptBuilder,
    errors: list[str],
) -> RetryEvidence:
    if cursor.present and not cursor.excluded and cursor.failure_count == 0 and cursor.next_retry_at is None:
        # Attempt history is append-only. A healthy current cursor must not be
        # decorated with a stale failure from a previous retry cycle.
        return RetryEvidence()
    ops_db = archive_root / "ops.db"
    if ops_db.exists():
        try:
            with _ReadonlyDatabase(ops_db, label="ops.db", limits=limits, receipt=receipt) as db:
                if db.table_exists("ingest_attempts"):
                    columns = db.columns("ingest_attempts")
                    row = _exact_attempt_row(db, columns, source_key, limits.max_attempt_rows)
                    if row is not None:
                        return _retry_from_attempt(dict(row))
        except sqlite3.Error as exc:
            errors.append(f"ops.db attempt read failed: {exc}")

    if attempt_log is not None:
        attempt = _attempt_from_bounded_tail(
            attempt_log,
            source_key,
            limits=limits,
            receipt=receipt,
            errors=errors,
        )
        if attempt is not None:
            return attempt
    if cursor.excluded:
        return RetryEvidence(
            reason=f"excluded after {cursor.failure_count} recorded failures",
            reason_source="cursor-state",
            observed_at_ms=cursor.updated_at_ms,
        )
    if cursor.failure_count > 0:
        return RetryEvidence(
            reason=f"retry pending after {cursor.failure_count} recorded failures",
            reason_source="cursor-state",
            observed_at_ms=cursor.updated_at_ms,
        )
    return RetryEvidence()


def _exact_attempt_row(
    db: _ReadonlyDatabase,
    columns: frozenset[str],
    source_key: str,
    limit: int,
) -> sqlite3.Row | None:
    if "source_path" not in columns:
        return None
    selected = []
    for name in (
        "attempt_id",
        "status",
        "phase",
        "error_message",
        "started_at_ms",
        "heartbeat_at_ms",
        "finished_at_ms",
    ):
        selected.append(f"{name} AS {name}" if name in columns else f"NULL AS {name}")
    timestamps = [name for name in ("heartbeat_at_ms", "finished_at_ms", "started_at_ms") if name in columns]
    order = f"COALESCE({', '.join(timestamps)}, 0) DESC" if timestamps else "rowid DESC"
    sql = (
        f"SELECT {', '.join(selected)} FROM ingest_attempts "
        f"WHERE source_path = ? ORDER BY {order}, rowid DESC LIMIT {max(limit, 1)}"
    )
    rows = db.exact_rows(
        label="ingest-attempts-by-source-path",
        sql=sql,
        params=(source_key,),
        protected_tables=("ingest_attempts",),
    )
    if not rows:
        return None
    for row in rows:
        if _optional_str(row["error_message"]):
            return row
    return rows[0]


def _retry_from_attempt(row: dict[str, object]) -> RetryEvidence:
    observed_at = next(
        (
            _timestamp_ms(row.get(name))
            for name in ("heartbeat_at_ms", "finished_at_ms", "started_at_ms")
            if _timestamp_ms(row.get(name)) is not None
        ),
        None,
    )
    return RetryEvidence(
        reason=_optional_str(row.get("error_message")),
        reason_source="ops.ingest_attempts",
        attempt_id=_optional_str(row.get("attempt_id")),
        attempt_status=_optional_str(row.get("status")),
        attempt_phase=_optional_str(row.get("phase")),
        observed_at_ms=observed_at,
    )


def _attempt_from_bounded_tail(
    path: Path,
    source_key: str,
    *,
    limits: ProjectionLimits,
    receipt: _ReceiptBuilder,
    errors: list[str],
) -> RetryEvidence | None:
    try:
        size = path.stat().st_size
        start = max(size - limits.attempt_tail_bytes, 0)
        with path.open("rb") as handle:
            handle.seek(start)
            data = handle.read(limits.attempt_tail_bytes)
    except OSError as exc:
        errors.append(f"attempt log tail read failed: {exc}")
        return None
    receipt.tail_bytes_read += len(data)
    if start > 0:
        first_newline = data.find(b"\n")
        data = b"" if first_newline < 0 else data[first_newline + 1 :]
    lines = data.splitlines()
    receipt.tail_lines_examined += len(lines)
    fallback: RetryEvidence | None = None
    for raw_line in reversed(lines):
        try:
            item = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(item, dict) or not _attempt_mentions_source(item, source_key):
            continue
        normalized = cast(dict[str, object], item)
        evidence = RetryEvidence(
            reason=_optional_str(normalized.get("error_message") or normalized.get("error")),
            reason_source="attempt-log-tail",
            attempt_id=_optional_str(normalized.get("attempt_id")),
            attempt_status=_optional_str(normalized.get("status")),
            attempt_phase=_optional_str(normalized.get("phase")),
            observed_at_ms=_timestamp_ms(
                normalized.get("heartbeat_at_ms")
                or normalized.get("finished_at_ms")
                or normalized.get("started_at_ms")
                or normalized.get("updated_at")
            ),
        )
        if evidence.reason:
            return evidence
        if fallback is None:
            fallback = evidence
    return fallback


def _attempt_mentions_source(item: dict[str, object], source_key: str) -> bool:
    for key in ("source_path", "current_path"):
        if item.get(key) == source_key:
            return True
    paths = item.get("source_paths")
    if isinstance(paths, list) and source_key in paths:
        return True
    encoded_paths = item.get("source_paths_json")
    if isinstance(encoded_paths, str):
        try:
            decoded = json.loads(encoded_paths)
        except json.JSONDecodeError:
            decoded = None
        return isinstance(decoded, list) and source_key in decoded
    return False


def _load_raw_revisions(
    archive_root: Path,
    source_key: str,
    *,
    limits: ProjectionLimits,
    receipt: _ReceiptBuilder,
    errors: list[str],
) -> tuple[tuple[RawRevisionEvidence, ...], bool, RawRevisionEvidence | None]:
    source_db = archive_root / "source.db"
    if not source_db.exists():
        return (), False, None
    try:
        with _ReadonlyDatabase(source_db, label="source.db", limits=limits, receipt=receipt) as db:
            if not db.table_exists("raw_sessions"):
                return (), False, None
            columns = db.columns("raw_sessions")
            if not {"raw_id", "source_path"} <= columns:
                errors.append("source.db raw_sessions lacks raw_id/source_path")
                return (), False, None
            selected_names = (
                "raw_id",
                "origin",
                "native_id",
                "source_index",
                "blob_hash",
                "validation_status",
                "parse_error",
                "parsed_at_ms",
                "revision_authority",
            )
            selected = [name if name in columns else f"NULL AS {name}" for name in selected_names]
            observed_candidates = [
                name
                for name in ("acquired_at_ms", "observed_at_ms", "created_at_ms", "parsed_at_ms")
                if name in columns
            ]
            observed_expr = f"COALESCE({', '.join(observed_candidates)}, 0)" if observed_candidates else "rowid"
            selected.append(f"{observed_expr} AS observed_at_ms")
            sql = (
                f"SELECT {', '.join(selected)} FROM raw_sessions "
                f"WHERE source_path = ? ORDER BY {observed_expr} DESC, rowid DESC "
                f"LIMIT {max(limits.max_raw_revisions, 1) + 1}"
            )
            rows = db.exact_rows(
                label="raw-revisions-by-source-path",
                sql=sql,
                params=(source_key,),
                protected_tables=("raw_sessions",),
            )
            if rows is None:
                errors.append("source.db exact raw lookup rejected unsafe query plan")
                return (), False, None

            # The recent-revision window is deliberately bounded, but the
            # accepted row must not disappear merely because many newer rows
            # were marked skipped. Query it separately with the same exact
            # source key rather than scanning beyond the window in Python.
            accepted_predicate = (
                "COALESCE(validation_status, '') != 'skipped'" if "validation_status" in columns else "1 = 1"
            )
            accepted_sql = (
                f"SELECT {', '.join(selected)} FROM raw_sessions "
                f"WHERE source_path = ? AND {accepted_predicate} "
                f"ORDER BY {observed_expr} DESC, rowid DESC LIMIT 1"
            )
            accepted_rows = db.exact_rows(
                label="accepted-raw-revision-by-source-path",
                sql=accepted_sql,
                params=(source_key,),
                protected_tables=("raw_sessions",),
            )
            if accepted_rows is None:
                errors.append("source.db accepted raw lookup rejected unsafe query plan")
                accepted_rows = []
    except sqlite3.Error as exc:
        errors.append(f"source.db raw revision read failed: {exc}")
        return (), False, None
    truncated = len(rows) > limits.max_raw_revisions
    revisions = tuple(_raw_revision_from_row(row) for row in rows[: limits.max_raw_revisions])
    accepted = _raw_revision_from_row(accepted_rows[0]) if accepted_rows else None
    return revisions, truncated, accepted


def _raw_revision_from_row(row: sqlite3.Row) -> RawRevisionEvidence:
    validation = _optional_str(row["validation_status"])
    # This is acquisition eligibility only. It mirrors the archive's existing
    # non-skipped raw census and does not reinterpret lkrc authority or yla8
    # replay decisions.
    accepted = validation is None or validation.lower() != "skipped"
    return RawRevisionEvidence(
        raw_id=str(row["raw_id"]),
        origin=_optional_str(row["origin"]),
        native_id=_optional_str(row["native_id"]),
        source_index=_optional_int(row["source_index"]),
        blob_hash=_blob_hash(row["blob_hash"]),
        validation_status=validation,
        parse_error=_optional_str(row["parse_error"]),
        parsed_at_ms=_timestamp_ms(row["parsed_at_ms"]),
        observed_at_ms=_timestamp_ms(row["observed_at_ms"]),
        revision_authority=_optional_str(row["revision_authority"]),
        accepted_by_acquisition=accepted,
    )


def _parse_evidence(raw: RawRevisionEvidence | None) -> ParseEvidence:
    if raw is None:
        return ParseEvidence(state="unseen")
    if raw.parse_error:
        return ParseEvidence(state="failed", raw_id=raw.raw_id, error=raw.parse_error)
    if raw.parsed_at_ms is not None:
        return ParseEvidence(state="parsed", raw_id=raw.raw_id, parsed_at_ms=raw.parsed_at_ms)
    return ParseEvidence(state="pending", raw_id=raw.raw_id)


def _load_revision_applications(
    archive_root: Path,
    raw: RawRevisionEvidence | None,
    *,
    limits: ProjectionLimits,
    receipt: _ReceiptBuilder,
    errors: list[str],
) -> tuple[tuple[RevisionApplicationEvidence, ...], bool]:
    if raw is None:
        return (), False
    index_db = archive_root / "index.db"
    if not index_db.exists():
        return (), False
    try:
        with _ReadonlyDatabase(index_db, label="index.db", limits=limits, receipt=receipt) as db:
            if not db.table_exists("raw_revision_applications"):
                return (), False
            columns = db.columns("raw_revision_applications")
            if "raw_id" not in columns:
                return (), False
            selected = [name if name in columns else f"NULL AS {name}" for name in ("raw_id", "decision", "detail")]
            observed_candidates = [
                name
                for name in ("applied_at_ms", "observed_at_ms", "created_at_ms", "updated_at_ms")
                if name in columns
            ]
            observed_expr = f"COALESCE({', '.join(observed_candidates)}, 0)" if observed_candidates else "rowid"
            selected.append(f"{observed_expr} AS observed_at_ms")
            sql = (
                f"SELECT {', '.join(selected)} FROM raw_revision_applications "
                f"WHERE raw_id = ? ORDER BY {observed_expr} DESC, rowid DESC "
                f"LIMIT {max(limits.max_application_rows, 1) + 1}"
            )
            rows = db.exact_rows(
                label="revision-applications-by-raw-id",
                sql=sql,
                params=(raw.raw_id,),
                protected_tables=("raw_revision_applications",),
            )
            if rows is None:
                errors.append("index.db revision application lookup rejected unsafe query plan")
                return (), False
    except sqlite3.Error as exc:
        errors.append(f"index.db revision application read failed: {exc}")
        return (), False
    truncated = len(rows) > limits.max_application_rows
    applications = tuple(
        RevisionApplicationEvidence(
            raw_id=str(row["raw_id"]),
            decision=_optional_str(row["decision"]),
            detail=_optional_str(row["detail"]),
            observed_at_ms=_timestamp_ms(row["observed_at_ms"]),
        )
        for row in rows[: limits.max_application_rows]
    )
    return applications, truncated


def _load_index_evidence(
    archive_root: Path,
    accepted_raw: RawRevisionEvidence | None,
    raw_revisions: tuple[RawRevisionEvidence, ...],
    *,
    raw_scope_truncated: bool,
    limits: ProjectionLimits,
    receipt: _ReceiptBuilder,
    errors: list[str],
) -> IndexEvidence:
    index_db = archive_root / "index.db"
    if not index_db.exists():
        return IndexEvidence(available=False, reason="index.db missing")
    if accepted_raw is None:
        return IndexEvidence(available=True, reason="no accepted raw revision")
    try:
        with _ReadonlyDatabase(index_db, label="index.db", limits=limits, receipt=receipt) as db:
            if not db.table_exists("sessions"):
                return IndexEvidence(available=False, reason="sessions table missing")
            columns = db.columns("sessions")
            if not {"session_id", "raw_id"} <= columns:
                return IndexEvidence(available=False, reason="sessions lacks session_id/raw_id")
            high_water_column = next(
                (name for name in ("sort_key_ms", "updated_at_ms", "created_at_ms") if name in columns),
                None,
            )
            high_water_expr = high_water_column or "0"
            accepted_sql = (
                f"SELECT session_id, {high_water_expr} AS high_water_ms FROM sessions "
                "WHERE raw_id = ? ORDER BY high_water_ms DESC, session_id "
                f"LIMIT {max(limits.max_sessions, 1) + 1}"
            )
            accepted_rows = db.exact_rows(
                label="accepted-sessions-by-raw-id",
                sql=accepted_sql,
                params=(accepted_raw.raw_id,),
                protected_tables=("sessions",),
            )
            if accepted_rows is None:
                errors.append("index.db accepted session lookup rejected unsafe query plan")
                return IndexEvidence(available=False, reason="unsafe accepted session query plan")
            raw_ids = tuple(dict.fromkeys((accepted_raw.raw_id, *(revision.raw_id for revision in raw_revisions))))
            source_rows = accepted_rows
            if raw_ids:
                placeholders = ",".join("?" for _ in raw_ids)
                source_sql = (
                    f"SELECT session_id, {high_water_expr} AS high_water_ms FROM sessions "
                    f"WHERE raw_id IN ({placeholders}) ORDER BY high_water_ms DESC, session_id "
                    f"LIMIT {max(limits.max_sessions, 1) + 1}"
                )
                selected = db.exact_rows(
                    label="source-sessions-by-bounded-raw-ids",
                    sql=source_sql,
                    params=cast(tuple[object, ...], raw_ids),
                    protected_tables=("sessions",),
                )
                if selected is not None:
                    source_rows = selected
                else:
                    errors.append("index.db source high-water lookup rejected unsafe query plan")
    except sqlite3.Error as exc:
        errors.append(f"index.db session read failed: {exc}")
        return IndexEvidence(available=False, reason=str(exc))

    accepted_truncated = len(accepted_rows) > limits.max_sessions
    source_truncated = len(source_rows) > limits.max_sessions
    accepted_ids = tuple(str(row["session_id"]) for row in accepted_rows[: limits.max_sessions])
    source_ids = tuple(str(row["session_id"]) for row in source_rows[: limits.max_sessions])
    high_water = None
    if source_rows and high_water_column is not None:
        high_water = _timestamp_ms(source_rows[0]["high_water_ms"])
    broken_head = bool(source_ids) and not bool(accepted_ids)
    reason = None
    if broken_head:
        reason = "accepted raw revision is absent from the index while an older source revision remains indexed"
    return IndexEvidence(
        available=True,
        accepted_raw_indexed=bool(accepted_ids),
        broken_head=broken_head,
        accepted_session_ids=accepted_ids,
        source_session_ids=source_ids,
        session_count_lower_bound=min(len(source_rows), limits.max_sessions),
        sessions_truncated=accepted_truncated or source_truncated,
        source_raw_scope_truncated=raw_scope_truncated,
        high_water_ms=high_water,
        high_water_column=high_water_column,
        reason=reason,
    )


def _load_fts_evidence(
    archive_root: Path,
    index: IndexEvidence,
    *,
    limits: ProjectionLimits,
    receipt: _ReceiptBuilder,
    errors: list[str],
) -> FtsEvidence:
    if not index.available or not index.accepted_raw_indexed:
        return FtsEvidence(available=False, reason="accepted raw revision is not indexed")
    if index.sessions_truncated:
        return FtsEvidence(available=True, reason="accepted session scope exceeded bound")
    index_db = archive_root / "index.db"
    try:
        with _ReadonlyDatabase(index_db, label="index.db", limits=limits, receipt=receipt) as db:
            required = ("messages", "blocks", "messages_fts")
            if not all(db.table_exists(table) for table in required):
                return FtsEvidence(
                    available=False,
                    reason="messages, blocks, or messages_fts missing",
                )
            message_columns = db.columns("messages")
            block_columns = db.columns("blocks")
            if not {"message_id", "session_id"} <= message_columns:
                return FtsEvidence(available=False, reason="messages lacks message_id/session_id")
            if not {"message_id", "search_text"} <= block_columns:
                return FtsEvidence(available=False, reason="blocks lacks message_id/search_text")
            session_ids = index.accepted_session_ids
            message_ids, messages_truncated = _bounded_ids(
                db,
                table="messages",
                id_column="message_id",
                key_column="session_id",
                keys=session_ids,
                limit=limits.max_messages,
                label="messages-by-accepted-session-ids",
                errors=errors,
            )
            if messages_truncated:
                return FtsEvidence(available=True, reason="message scope exceeded bound")
            if not message_ids:
                (
                    recorded_state,
                    checked_at,
                    recorded_ready,
                    triggers_present,
                ) = _recorded_fts_state(db)
                return FtsEvidence(
                    available=True,
                    converged=recorded_ready and triggers_present,
                    recorded_state=recorded_state,
                    checked_at=checked_at,
                    triggers_present=triggers_present,
                    reason=_fts_reason(recorded_ready, triggers_present, exact_converged=True),
                )
            block_rows = _bounded_searchable_blocks(
                db,
                message_ids,
                limit=limits.max_blocks,
                errors=errors,
            )
            if block_rows is None:
                return FtsEvidence(available=False, reason="unsafe exact block query plan")
            blocks_truncated = len(block_rows) > limits.max_blocks
            block_rowids = tuple(int(row["rowid"]) for row in block_rows[: limits.max_blocks])
            recorded_state, checked_at, recorded_ready, triggers_present = _recorded_fts_state(db)
            if blocks_truncated:
                return FtsEvidence(
                    available=True,
                    recorded_state=recorded_state,
                    checked_at=checked_at,
                    triggers_present=triggers_present,
                    source_searchable_blocks=len(block_rowids),
                    blocks_truncated=True,
                    reason="searchable block scope exceeded bound",
                )
            if not block_rowids:
                return FtsEvidence(
                    available=True,
                    converged=recorded_ready and triggers_present,
                    recorded_state=recorded_state,
                    checked_at=checked_at,
                    triggers_present=triggers_present,
                    reason=_fts_reason(recorded_ready, triggers_present, exact_converged=True),
                )
            placeholders = ",".join("?" for _ in block_rowids)
            fts_sql = (
                f"SELECT rowid FROM messages_fts WHERE rowid IN ({placeholders}) "
                f"ORDER BY rowid LIMIT {len(block_rowids) + 1}"
            )
            fts_rows = db.exact_rows(
                label="fts-rowids-by-exact-block-rowids",
                sql=fts_sql,
                params=cast(tuple[object, ...], block_rowids),
                protected_tables=("messages_fts",),
            )
            if fts_rows is None:
                return FtsEvidence(available=False, reason="FTS row query rejected")
            indexed_rowids = {int(row[0]) for row in fts_rows}
            exact_converged = indexed_rowids == set(block_rowids)
            converged = recorded_ready and triggers_present and exact_converged
            reason = _fts_reason(recorded_ready, triggers_present, exact_converged)
            return FtsEvidence(
                available=True,
                converged=converged,
                recorded_state=recorded_state,
                checked_at=checked_at,
                triggers_present=triggers_present,
                source_searchable_blocks=len(block_rowids),
                indexed_searchable_blocks=len(indexed_rowids),
                reason=reason,
            )
    except sqlite3.Error as exc:
        errors.append(f"index.db FTS read failed: {exc}")
        return FtsEvidence(available=False, reason=str(exc))


def _bounded_ids(
    db: _ReadonlyDatabase,
    *,
    table: str,
    id_column: str,
    key_column: str,
    keys: tuple[str, ...],
    limit: int,
    label: str,
    errors: list[str],
) -> tuple[tuple[str, ...], bool]:
    if not keys:
        return (), False
    placeholders = ",".join("?" for _ in keys)
    sql = (
        f"SELECT {_quote_identifier(id_column)} FROM {_quote_identifier(table)} "
        f"WHERE {_quote_identifier(key_column)} IN ({placeholders}) "
        f"ORDER BY {_quote_identifier(id_column)} LIMIT {max(limit, 1) + 1}"
    )
    rows = db.exact_rows(
        label=label,
        sql=sql,
        params=cast(tuple[object, ...], keys),
        protected_tables=(table,),
    )
    if rows is None:
        errors.append(f"{db.label} {label} rejected unsafe query plan")
        return (), False
    return tuple(str(row[0]) for row in rows[:limit]), len(rows) > limit


def _bounded_searchable_blocks(
    db: _ReadonlyDatabase,
    message_ids: tuple[str, ...],
    *,
    limit: int,
    errors: list[str],
) -> list[sqlite3.Row] | None:
    if not message_ids:
        return []
    placeholders = ",".join("?" for _ in message_ids)
    sql = (
        "SELECT rowid FROM blocks "
        f"WHERE message_id IN ({placeholders}) AND COALESCE(search_text, '') != '' "
        f"ORDER BY rowid LIMIT {max(limit, 1) + 1}"
    )
    rows = db.exact_rows(
        label="searchable-blocks-by-message-ids",
        sql=sql,
        params=cast(tuple[object, ...], message_ids),
        protected_tables=("blocks",),
    )
    if rows is None:
        errors.append(f"{db.label} searchable block lookup rejected unsafe query plan")
    return rows


def _recorded_fts_state(
    db: _ReadonlyDatabase,
) -> tuple[str | None, str | None, bool, bool]:
    trigger_names = ("messages_fts_ai", "messages_fts_ad", "messages_fts_au")
    placeholders = ",".join("?" for _ in trigger_names)
    trigger_row = db.one(
        f"SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' AND name IN ({placeholders})",
        trigger_names,
    )
    triggers_present = trigger_row is not None and int(trigger_row[0] or 0) == len(trigger_names)
    if not db.table_exists("fts_freshness_state"):
        return None, None, False, triggers_present
    columns = db.columns("fts_freshness_state")
    selected = [
        name if name in columns else f"NULL AS {name}"
        for name in (
            "state",
            "checked_at",
            "source_rows",
            "indexed_rows",
            "missing_rows",
            "excess_rows",
            "duplicate_rows",
        )
    ]
    rows = db.exact_rows(
        label="fts-freshness-state-by-surface",
        sql=(f"SELECT {', '.join(selected)} FROM fts_freshness_state WHERE surface = ? LIMIT 1"),
        params=("messages_fts",),
        protected_tables=("fts_freshness_state",),
    )
    if not rows:
        return None, None, False, triggers_present
    row = rows[0]
    state = _optional_str(row["state"])
    source_rows = _optional_int(row["source_rows"]) or 0
    indexed_rows = _optional_int(row["indexed_rows"]) or 0
    ready = (
        state == "ready"
        and not (source_rows == 0 and indexed_rows == 0)
        and source_rows == indexed_rows
        and (_optional_int(row["missing_rows"]) or 0) == 0
        and (_optional_int(row["excess_rows"]) or 0) == 0
        and (_optional_int(row["duplicate_rows"]) or 0) == 0
    )
    return state, _optional_str(row["checked_at"]), ready, triggers_present


def _fts_reason(
    recorded_ready: bool,
    triggers_present: bool,
    exact_converged: bool,
) -> str | None:
    if not recorded_ready:
        return "FTS freshness ledger is not ready"
    if not triggers_present:
        return "FTS maintenance triggers are missing"
    if not exact_converged:
        return "one or more exact-source blocks are absent from messages_fts"
    return None


def _load_insight_evidence(
    archive_root: Path,
    source_key: str,
    index: IndexEvidence,
    *,
    limits: ProjectionLimits,
    receipt: _ReceiptBuilder,
    errors: list[str],
) -> InsightEvidence:
    ops_db = archive_root / "ops.db"
    if not ops_db.exists():
        return InsightEvidence(available=False, reason="ops.db missing")
    target_ids = tuple(dict.fromkeys((source_key, *index.accepted_session_ids)))
    rows: list[sqlite3.Row] = []
    query_truncated = False
    try:
        with _ReadonlyDatabase(ops_db, label="ops.db", limits=limits, receipt=receipt) as db:
            if not db.table_exists("convergence_debt"):
                return InsightEvidence(available=False, reason="convergence_debt table missing")
            columns = db.columns("convergence_debt")
            target_column = "target_id" if "target_id" in columns else "subject_id"
            stage_column = "stage" if "stage" in columns else None
            if target_column not in columns or stage_column is None:
                return InsightEvidence(
                    available=False,
                    reason="convergence_debt lacks target/stage columns",
                )
            selected = [
                f"{stage_column} AS stage",
                "status AS status" if "status" in columns else "NULL AS status",
                (
                    "last_error AS error"
                    if "last_error" in columns
                    else "error AS error"
                    if "error" in columns
                    else "NULL AS error"
                ),
            ]
            if "target_type" in columns and target_column == "target_id":
                # The owner ledger keys debt by (stage, target_type, target_id).
                # Keep source-path and session-id namespaces distinct so an
                # accidental identifier collision cannot manufacture debt.
                target_groups = (
                    ("source_path", (source_key,)),
                    ("session_id", index.accepted_session_ids),
                )
                for target_type, ids in target_groups:
                    if not ids:
                        continue
                    placeholders = ",".join("?" for _ in ids)
                    sql = (
                        f"SELECT {', '.join(selected)} FROM convergence_debt "
                        f"WHERE target_type = ? AND {target_column} IN ({placeholders}) "
                        f"AND {stage_column} = ? "
                        f"ORDER BY rowid DESC LIMIT {max(limits.max_debt_rows, 1) + 1}"
                    )
                    params = cast(tuple[object, ...], (target_type, *ids, "insights"))
                    selected_rows = db.exact_rows(
                        label=f"insight-debt-by-{target_type}",
                        sql=sql,
                        params=params,
                        protected_tables=("convergence_debt",),
                    )
                    if selected_rows is None:
                        errors.append("ops.db insight debt lookup rejected unsafe query plan")
                        return InsightEvidence(
                            available=False,
                            reason="unsafe convergence debt query plan",
                        )
                    query_truncated = query_truncated or len(selected_rows) > limits.max_debt_rows
                    rows.extend(selected_rows[: limits.max_debt_rows])
            else:
                # Legacy ledgers without target_type cannot disambiguate the
                # namespaces; preserve bounded compatibility and report only
                # exact target IDs.
                placeholders = ",".join("?" for _ in target_ids)
                sql = (
                    f"SELECT {', '.join(selected)} FROM convergence_debt "
                    f"WHERE {target_column} IN ({placeholders}) AND {stage_column} = ? "
                    f"ORDER BY rowid DESC LIMIT {max(limits.max_debt_rows, 1) + 1}"
                )
                params = cast(tuple[object, ...], (*target_ids, "insights"))
                selected_rows = db.exact_rows(
                    label="insight-debt-by-source-or-session-id",
                    sql=sql,
                    params=params,
                    protected_tables=("convergence_debt",),
                )
                if selected_rows is None:
                    errors.append("ops.db insight debt lookup rejected unsafe query plan")
                    return InsightEvidence(
                        available=False,
                        reason="unsafe convergence debt query plan",
                    )
                query_truncated = len(selected_rows) > limits.max_debt_rows
                rows.extend(selected_rows[: limits.max_debt_rows])
    except sqlite3.Error as exc:
        errors.append(f"ops.db insight debt read failed: {exc}")
        return InsightEvidence(available=False, reason=str(exc))
    truncated = query_truncated or len(rows) > limits.max_debt_rows
    visible = rows[: limits.max_debt_rows]
    if visible:
        return InsightEvidence(
            available=True,
            converged=False,
            state="debt-recorded",
            debt_count_lower_bound=len(visible),
            debt_stages=tuple(sorted({_optional_str(row["stage"]) or "insights" for row in visible})),
            debt_errors=tuple(dict.fromkeys(error for row in visible if (error := _optional_str(row["error"])))),
            debt_truncated=truncated,
            reason="insight convergence debt is recorded",
        )
    return InsightEvidence(
        available=True,
        converged=True,
        state="no-recorded-debt",
        reason=("positive completion is inferred only from the owner ledger having no exact-source debt"),
    )


def _classify_stage(
    raw: RawRevisionEvidence | None,
    parse: ParseEvidence,
    index: IndexEvidence,
    fts: FtsEvidence,
    insights: InsightEvidence,
) -> NamedSourceStage:
    if raw is None:
        return NamedSourceStage.UNSEEN
    if index.accepted_raw_indexed:
        if fts.converged and insights.converged:
            return NamedSourceStage.SEARCHABLE
        return NamedSourceStage.INDEXED_UNCONVERGED
    if parse.state != "parsed":
        return NamedSourceStage.ACQUIRED_UNPARSED
    return NamedSourceStage.PARSED_UNINDEXED


def _classify_operational_state(
    source_stat: SourceStatEvidence,
    cursor: CursorEvidence,
    raw: RawRevisionEvidence | None,
    index: IndexEvidence,
) -> tuple[NamedSourceOperationalState, NamedSourceOperationalReason]:
    # Exclusion/failure is intentionally evaluated before any caught-up/idle
    # test.  This is the ordering defect from the live incident.
    if cursor.excluded:
        return (
            NamedSourceOperationalState.DEGRADED,
            NamedSourceOperationalReason.CURSOR_EXCLUDED,
        )
    if cursor.failure_count > 0:
        return (
            NamedSourceOperationalState.DEGRADED,
            NamedSourceOperationalReason.CURSOR_RETRYING,
        )
    if source_stat.error is not None:
        return (
            NamedSourceOperationalState.DEGRADED,
            NamedSourceOperationalReason.SOURCE_STAT_ERROR,
        )
    if not source_stat.exists and (cursor.present or raw is not None):
        return (
            NamedSourceOperationalState.DEGRADED,
            NamedSourceOperationalReason.SOURCE_MISSING,
        )
    if (cursor.cursor_ahead_bytes or 0) > 0 or (cursor.observed_size_ahead_bytes or 0) > 0:
        return (
            NamedSourceOperationalState.DEGRADED,
            NamedSourceOperationalReason.CURSOR_AHEAD,
        )
    if index.broken_head:
        return (
            NamedSourceOperationalState.DEGRADED,
            NamedSourceOperationalReason.BROKEN_HEAD,
        )
    if cursor.present:
        if cursor.pending_bytes is not None and cursor.pending_bytes > 0:
            return (
                NamedSourceOperationalState.ACTIVE,
                NamedSourceOperationalReason.PENDING_BYTES,
            )
        return (
            NamedSourceOperationalState.IDLE,
            NamedSourceOperationalReason.CAUGHT_UP,
        )
    if source_stat.exists or raw is not None:
        return (
            NamedSourceOperationalState.ACTIVE,
            NamedSourceOperationalReason.CURSOR_MISSING,
        )
    return (
        NamedSourceOperationalState.UNSEEN,
        NamedSourceOperationalReason.NO_EVIDENCE,
    )


def _unsafe_plan_details(
    details: tuple[str, ...],
    protected_tables: tuple[str, ...],
) -> tuple[str, ...]:
    unsafe: list[str] = []
    for detail in details:
        normalized = " ".join(detail.upper().replace('"', "").split())
        for table in protected_tables:
            marker = table.upper()
            scans_table = f"SCAN {marker}" in normalized
            virtual_table_lookup = "VIRTUAL TABLE INDEX" in normalized
            if scans_table and not virtual_table_lookup:
                unsafe.append(detail)
    return tuple(unsafe)


def _quote_identifier(value: str) -> str:
    if not _safe_identifier(value):
        raise ValueError(f"unsafe SQLite identifier: {value!r}")
    return f'"{value}"'


def _safe_identifier(value: str) -> bool:
    return bool(value) and value.replace("_", "a").isalnum() and not value[0].isdigit()


def _timestamp_ms(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
            except ValueError:
                return None
            return int(_as_utc(parsed).timestamp() * 1000)
    return None


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _blob_hash(value: object) -> str | None:
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, str) and value:
        return value
    return None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, EvidenceValue):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set | frozenset):
        return [_jsonable(item) for item in value]
    return str(value)


__all__ = [
    "CursorEvidence",
    "FtsEvidence",
    "IndexEvidence",
    "InsightEvidence",
    "NamedSourceFreshness",
    "NamedSourceOperationalReason",
    "NamedSourceOperationalState",
    "NamedSourceStage",
    "OwnershipBoundary",
    "ParseEvidence",
    "ProjectionLimits",
    "QueryPlanEvidence",
    "QueryReceipt",
    "RawRevisionEvidence",
    "RetryEvidence",
    "RevisionApplicationEvidence",
    "SOURCE_CURSOR_BYTE_LAG_FAMILY",
    "SourceStatEvidence",
    "aggregate_named_source_byte_lag",
    "project_named_source_freshness",
]

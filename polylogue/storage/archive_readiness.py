"""Shared archive readiness helpers."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

from polylogue.archive.raw_materialization import source_path_native_id_candidates
from polylogue.archive.revision_authority import (
    RawRevisionAuthority,
    parser_census_identity_measurement,
)
from polylogue.core.payload_coercion import row_int as _row_int
from polylogue.core.raw_failure_evidence import RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS
from polylogue.core.sqlite_introspection import table_exists as _table_exists
from polylogue.core.sqlite_introspection import view_exists
from polylogue.logging import get_logger
from polylogue.storage.derived.session.status import session_insight_status_sync
from polylogue.storage.raw_authority import iter_parser_census_logical_keys, raw_authority_parser_fingerprint
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.connection_profile import attach_readonly_database, open_readonly_connection

logger = get_logger(__name__)

ArchiveTierVersionStatus = Literal["ok", "missing", "mismatch", "invalid"]


_DURABLE_TIERS: tuple[ArchiveTier, ...] = (ArchiveTier.SOURCE, ArchiveTier.USER, ArchiveTier.AUDIT)


def lost_durable_tiers(archive_root: Path) -> tuple[ArchiveTier, ...]:
    """Return the durable tiers an established archive lost outside Polylogue.

    Bootstrap creates every durable tier together under one pending intent,
    and opening the archive refuses a durable tier missing beside a surviving
    one rather than recreating it. A fresh root (no durable tier) and a
    pending bootstrap intent are recoverable by opening the archive, so
    neither is lost.
    """
    ledger = archive_root / ".maintenance-state" / "durable-change-trains"
    if (ledger / ".bootstrap.pending").is_file():
        return ()
    missing = tuple(tier for tier in _DURABLE_TIERS if not (archive_root / f"{tier.value}.db").exists())
    if len(missing) == len(_DURABLE_TIERS):
        return ()
    return missing


@dataclass(frozen=True, slots=True)
class ArchiveTierProbe:
    """One archive tier file's existence/size/schema-version facts.

    The sole, shared read-only probe of a tier file's ``PRAGMA user_version``
    against ``ARCHIVE_VERSION_BY_TIER`` (polylogue-703 -- ONE status
    assembly). ``polylogue.daemon.status`` (daemon HTTP/TUI/web status) and
    ``polylogue.cli.commands.status`` (CLI direct-fallback status, used when
    the daemon is unreachable) both build their per-tier status payload from
    this probe instead of each reimplementing the PRAGMA read independently
    -- the prior independent implementations were the mechanism behind a
    real production disagreement between bare-CLI and daemon-backed status
    (2026-07-03). Do not add a third independent tier-version probe; import
    this one.
    """

    tier: ArchiveTier
    path: str
    exists: bool
    size_bytes: int
    wal_size_bytes: int
    user_version: int | None
    expected_user_version: int
    version_status: ArchiveTierVersionStatus


def probe_archive_tier(tier: ArchiveTier, path: Path) -> ArchiveTierProbe:
    """Read-only probe of one archive tier file's existence/size/version."""
    expected_user_version = ARCHIVE_VERSION_BY_TIER[tier]
    if not path.exists():
        return ArchiveTierProbe(
            tier=tier,
            path=str(path),
            exists=False,
            size_bytes=0,
            wal_size_bytes=0,
            user_version=None,
            expected_user_version=expected_user_version,
            version_status="missing",
        )
    wal_path = Path(f"{path}-wal")
    user_version: int | None = None
    version_status: ArchiveTierVersionStatus = "invalid"
    try:
        # Readiness must inspect the version that the connection factory would
        # otherwise reject. This probe reports schema skew as data for status
        # and recovery callers; it does not authorize ordinary tier reads.
        conn = open_readonly_connection(path, validate_schema=False)
        try:
            user_version = _row_int(conn.execute("PRAGMA user_version").fetchone()[0])
            version_status = "ok" if user_version == expected_user_version else "mismatch"
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("archive tier version probe failed for %s (%s): %s", path, tier.value, exc, exc_info=True)
    return ArchiveTierProbe(
        tier=tier,
        path=str(path),
        exists=True,
        size_bytes=path.stat().st_size,
        wal_size_bytes=wal_path.stat().st_size if wal_path.exists() else 0,
        user_version=user_version,
        expected_user_version=expected_user_version,
        version_status=version_status,
    )


CLAUDE_WORKFLOW_STAGE_NAME = "claude_workflow"
"""daemon_stage_events ``stage`` value written by the claude_workflow
convergence stage (daemon/convergence_stages.py); imported from there so the
writer and this reader cannot drift apart."""


def claude_workflow_materialization_status(ops_db: Path) -> dict[str, object] | None:
    """Return the most recently recorded Claude Workflow materialization summary.

    Reads the latest ``daemon_stage_events`` row written by
    ``daemon.convergence_stages``'s claude_workflow stage each time it
    materializes evidence graphs. Returns ``None`` when the stage has never
    run against this archive (ops.db missing, table missing, or no rows).
    """
    if not ops_db.exists():
        return None
    try:
        with closing(open_readonly_connection(ops_db, timeout_class="background-read", validate_schema=False)) as conn:
            conn.row_factory = sqlite3.Row
            has_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'daemon_stage_events'"
            ).fetchone()
            if has_table is None:
                return None
            row = conn.execute(
                """
                SELECT status, observed_at_ms, payload_json
                FROM daemon_stage_events
                WHERE stage = ?
                ORDER BY observed_at_ms DESC, rowid DESC
                LIMIT 1
                """,
                (CLAUDE_WORKFLOW_STAGE_NAME,),
            ).fetchone()
    except sqlite3.Error as exc:
        logger.warning("claude workflow materialization status query failed for %s: %s", ops_db, exc, exc_info=True)
        return None
    if row is None:
        return None
    try:
        payload = json.loads(row["payload_json"] or "{}")
    except (TypeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload["status"] = str(row["status"])
    payload["observed_at_ms"] = int(row["observed_at_ms"])
    return payload


class RawMaterializationAssessmentState(str, Enum):
    """Whether the raw-materialization denominator supports a verdict."""

    UNMEASURED = "unmeasured"
    POPULATED_CONVERGED = "populated_converged"
    POPULATED_UNCONVERGED = "populated_unconverged"


@dataclass(frozen=True, slots=True)
class RawMaterializationAssessment:
    """Immutable verdict over one raw-materialization readiness payload.

    The old boolean projection made an empty or unavailable denominator
    indistinguishable from a populated archive whose raw rows had converged.
    Keep its count evidence with the verdict so status surfaces do not each
    classify the payload differently. ``raw_artifact_count`` is the verdict's
    required denominator; ``materialized_raw_artifact_count`` is a progress
    projection and may be absent on compatibility snapshots. The authoritative
    convergence evidence remains the complete blocker set plus parser census.
    """

    state: RawMaterializationAssessmentState
    reason: str
    raw_artifact_count: int | None
    materialized_raw_artifact_count: int | None
    blocking_counts: tuple[tuple[str, int], ...] = ()
    detail: str | None = None

    @property
    def ready(self) -> bool:
        """Strict compatibility projection for boolean consumers."""

        return self.state is RawMaterializationAssessmentState.POPULATED_CONVERGED

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "reason": self.reason,
            "raw_artifact_count": self.raw_artifact_count,
            "materialized_raw_artifact_count": self.materialized_raw_artifact_count,
            "blocking_counts": dict(self.blocking_counts),
            "detail": self.detail,
        }


_RAW_MATERIALIZATION_BLOCKING_KEYS: tuple[str, ...] = (
    "critical",
    "warning",
    "actionable",
    "blocked",
    "affected_actionable",
    "affected_blocked",
    "affected_open",
    "lost_source_evidence_count",
    "unchecked",
    "affected_unchecked",
    "raw_authority_blocker_count",
    "raw_authority_parser_census_incomplete_count",
)


def _readiness_mapping(readiness: Mapping[str, Any] | object | None) -> Mapping[str, Any] | None:
    if readiness is None:
        return None
    if isinstance(readiness, Mapping):
        return readiness
    model_dump = getattr(readiness, "model_dump", None)
    dumped = model_dump() if callable(model_dump) else None
    return dumped if isinstance(dumped, Mapping) else None


def _exact_nonnegative_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def assess_raw_materialization(readiness: Mapping[str, Any] | object | None) -> RawMaterializationAssessment:
    """Classify raw materialization from its one authoritative snapshot.

    Zero raw artifacts is not a vacuous convergence proof: no raw artifact was
    measured. Unavailable snapshots, invalid parser censuses, and a missing raw
    denominator likewise withhold a verdict. Only a populated snapshot with
    every existing blocking counter at zero is converged.
    """

    payload = _readiness_mapping(readiness)
    if payload is None:
        return RawMaterializationAssessment(
            RawMaterializationAssessmentState.UNMEASURED,
            "readiness_unavailable",
            None,
            None,
            detail="raw-materialization readiness was not inspected",
        )
    if payload.get("available") is not True:
        detail = payload.get("error") or payload.get("reason")
        return RawMaterializationAssessment(
            RawMaterializationAssessmentState.UNMEASURED,
            "readiness_unavailable",
            None,
            None,
            detail=str(detail) if detail else None,
        )
    raw_artifact_count = _exact_nonnegative_int(payload.get("raw_artifact_count"))
    materialized_raw_artifact_count = _exact_nonnegative_int(payload.get("materialized_raw_artifact_count"))
    blocking_counts: list[tuple[str, int]] = []
    for key in _RAW_MATERIALIZATION_BLOCKING_KEYS:
        value = payload.get(key, 0)
        count = _exact_nonnegative_int(value)
        if count is None:
            return RawMaterializationAssessment(
                RawMaterializationAssessmentState.UNMEASURED,
                "blocking_count_invalid",
                raw_artifact_count,
                materialized_raw_artifact_count,
                detail=f"{key} is not a non-negative integer",
            )
        blocking_counts.append((key, count))
    frozen_blocking_counts = tuple(blocking_counts)
    # An already-observed blocker remains a refutation when the denominator, a
    # separate debt classifier or census is unavailable. Unknown auxiliary
    # evidence must never erase a measured reason this archive is not converged.
    if any(count > 0 for _, count in frozen_blocking_counts):
        return RawMaterializationAssessment(
            RawMaterializationAssessmentState.POPULATED_UNCONVERGED,
            "blocking_materialization_debt",
            raw_artifact_count,
            materialized_raw_artifact_count,
            frozen_blocking_counts,
            detail=str(payload.get("debt_classifier_error")) if payload.get("debt_classifier_error") else None,
        )
    if raw_artifact_count is None:
        return RawMaterializationAssessment(
            RawMaterializationAssessmentState.UNMEASURED,
            "raw_artifact_count_unavailable",
            None,
            materialized_raw_artifact_count,
            frozen_blocking_counts,
        )
    if raw_artifact_count == 0:
        return RawMaterializationAssessment(
            RawMaterializationAssessmentState.UNMEASURED,
            "zero_denominator",
            0,
            materialized_raw_artifact_count,
            frozen_blocking_counts,
        )
    if payload.get("debt_classifier_error"):
        return RawMaterializationAssessment(
            RawMaterializationAssessmentState.UNMEASURED,
            "debt_classifier_unavailable",
            raw_artifact_count,
            materialized_raw_artifact_count,
            frozen_blocking_counts,
            detail=str(payload["debt_classifier_error"]),
        )
    parser_census = payload.get("raw_authority_parser_census")
    if not isinstance(parser_census, Mapping):
        return RawMaterializationAssessment(
            RawMaterializationAssessmentState.UNMEASURED,
            "parser_census_unavailable",
            None,
            None,
        )
    if parser_census.get("available") is not True:
        return RawMaterializationAssessment(
            RawMaterializationAssessmentState.UNMEASURED,
            "parser_census_invalid",
            None,
            None,
            detail=str(parser_census.get("error") or parser_census.get("reason") or "parser census unavailable"),
        )

    return RawMaterializationAssessment(
        RawMaterializationAssessmentState.POPULATED_CONVERGED,
        "all_blocking_counts_zero",
        raw_artifact_count,
        materialized_raw_artifact_count,
        frozen_blocking_counts,
    )


def raw_materialization_ready(readiness: Mapping[str, Any] | object | None) -> bool:
    """Return whether raw acquisition and index materialization are converged.

    Classified alias/non-session join gaps are acceptable: they mean the raw
    row has been explained. Actionable/open/blocking debt is not acceptable for
    product archive readiness.
    """
    return assess_raw_materialization(readiness).ready


def _pinned_parser_census_projection(
    conn: sqlite3.Connection,
    *,
    source_schema: str,
    index_conn: sqlite3.Connection,
) -> dict[str, object]:
    """Apply the durable parser-census classifier to a pinned source alias."""

    required = (
        "raw_authority_parser_census",
        "raw_artifacts",
        "raw_membership_census",
        "raw_session_memberships",
    )
    if any(not _table_exists(index_conn, table, schema=source_schema) for table in required):
        return {
            "available": False,
            "complete_count": 0,
            "non_session_count": 0,
            "incomplete_count": 0,
            "incomplete_blob_bytes": 0,
            "missing_receipt_count": 0,
            "non_complete_receipt_count": 0,
            "incomplete_origin_summary": [],
        }

    terminal_placeholders = ", ".join("(?, ?)" for _ in RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS)
    rows = conn.execute(
        f"""
        SELECT r.raw_id, r.origin, COALESCE(r.blob_size, 0), p.raw_id, p.parser_fingerprint,
               p.status, p.logical_keys_json, r.logical_source_key, r.revision_kind,
               (r.parse_error IS NOT NULL OR r.validation_status = 'failed'),
               EXISTS(
                   SELECT 1 FROM {source_schema}.raw_artifacts a
                   WHERE a.raw_id = r.raw_id AND a.parse_as_session = 0
                     AND a.schema_eligible = 0 AND a.decode_error IS NULL
                     AND a.malformed_jsonl_lines = 0
                     AND (a.artifact_kind, a.support_status) NOT IN ({terminal_placeholders})
               ) AND NOT EXISTS(
                   SELECT 1 FROM {source_schema}.raw_artifacts sibling
                   WHERE sibling.raw_id = r.raw_id
                     AND (sibling.parse_as_session <> 0 OR sibling.schema_eligible <> 0
                          OR sibling.decode_error IS NOT NULL OR sibling.malformed_jsonl_lines <> 0
                          OR (sibling.artifact_kind, sibling.support_status) IN ({terminal_placeholders}))
               ),
               EXISTS(
                   SELECT 1 FROM {source_schema}.raw_membership_census mc
                   WHERE mc.raw_id = r.raw_id AND mc.parser_fingerprint = ? AND mc.status = 'non_session'
                     AND mc.member_count = 0
               ),
               EXISTS(
                   SELECT 1 FROM {source_schema}.raw_membership_census mc
                   WHERE mc.raw_id = r.raw_id AND r.source_index < 0
                     AND mc.parser_fingerprint = ? AND mc.status = 'failed'
                     AND mc.revision_authority = ?
               ),
               COALESCE(r.validation_status, '') = 'skipped',
               EXISTS(SELECT 1 FROM main.sessions s WHERE s.raw_id = r.raw_id)
        FROM {source_schema}.raw_sessions r
        LEFT JOIN {source_schema}.raw_authority_parser_census p ON p.raw_id = r.raw_id
        ORDER BY r.raw_id
        """,
        (
            *(value for pair in RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS for value in pair),
            *(value for pair in RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS for value in pair),
            raw_authority_parser_fingerprint(),
            raw_authority_parser_fingerprint(),
            RawRevisionAuthority.BYTE_PROVEN.value,
        ),
    )
    complete_count = non_session_count = incomplete_count = incomplete_blob_bytes = missing_receipt_count = (
        non_complete_receipt_count
    ) = 0
    incomplete_origins: Counter[str] = Counter()
    incomplete_origin_bytes: Counter[str] = Counter()
    current_row: tuple[object, ...] | None = None

    def assess() -> None:
        nonlocal \
            complete_count, \
            non_session_count, \
            incomplete_count, \
            incomplete_blob_bytes, \
            missing_receipt_count, \
            non_complete_receipt_count
        assert current_row is not None
        (
            _raw_id,
            origin,
            blob_size,
            receipt_raw_id,
            fingerprint,
            status,
            logical_keys_json,
            typed_key,
            revision_kind,
            validation_failed,
            typed_non_session,
            parser_confirmed_non_session,
            byte_governed_fragment,
            is_skipped,
            is_materialized,
        ) = current_row
        complete = False
        if (
            receipt_raw_id is not None
            and str(fingerprint) == raw_authority_parser_fingerprint()
            and str(status) == "complete"
        ):
            with (
                closing(
                    conn.execute(
                        f"SELECT logical_source_key FROM {source_schema}.raw_session_memberships WHERE raw_id=? ORDER BY logical_source_key",
                        (_raw_id,),
                    )
                ) as memberships,
                parser_census_identity_measurement(
                    raw_logical_key=typed_key,
                    revision_kind=revision_kind,
                    membership_logical_keys=(row[0] for row in memberships),
                    observed_logical_keys=iter_parser_census_logical_keys(logical_keys_json),
                    observed_are_receipt=True,
                ) as measured,
            ):
                complete = measured.complete(
                    typed_non_session=bool(typed_non_session),
                    parser_confirmed_non_session=bool(parser_confirmed_non_session),
                    byte_governed_fragment=bool(byte_governed_fragment),
                )
        if complete:
            complete_count += 1
            if (
                bool(typed_non_session)
                and bool(parser_confirmed_non_session)
                and not bool(validation_failed)
                and not bool(is_skipped)
                and not bool(is_materialized)
            ):
                non_session_count += 1
            return
        incomplete_count += 1
        size = int(cast(int | None, blob_size) or 0)
        incomplete_blob_bytes += size
        origin_key = str(origin)
        incomplete_origins[origin_key] += 1
        incomplete_origin_bytes[origin_key] += size
        if receipt_raw_id is None:
            missing_receipt_count += 1
        else:
            non_complete_receipt_count += 1

    try:
        for row in rows:
            current_row = tuple(row)
            assess()
    finally:
        rows.close()
    return {
        "available": True,
        "complete_count": complete_count,
        "non_session_count": non_session_count,
        "incomplete_count": incomplete_count,
        "incomplete_blob_bytes": incomplete_blob_bytes,
        "missing_receipt_count": missing_receipt_count,
        "non_complete_receipt_count": non_complete_receipt_count,
        "incomplete_origin_summary": [
            {"origin": origin, "count": count, "blob_bytes": incomplete_origin_bytes[origin]}
            for origin, count in sorted(
                incomplete_origins.items(), key=lambda item: (-incomplete_origin_bytes[item[0]], -item[1], item[0])
            )[:16]
        ],
    }


def _pinned_authority_frontier_projection(
    conn: sqlite3.Connection,
    *,
    source_schema: str,
    index_conn: sqlite3.Connection,
) -> dict[str, object]:
    """Read the durable frontier obligations through a pinned source alias.

    The per-pass census ledger this used to read is retired (polylogue-6kur
    ruling 2026-09-15): an inspection pass records nothing, so the durable
    answer to "is the accepted frontier authorized?" is the unresolved-blocker
    set, which every pass publishes and tombstones.
    """
    if not _table_exists(index_conn, "raw_authority_blockers", schema=source_schema):
        return {
            "raw_authority_frontier_remediation_refs": [],
            "raw_authority_blocker_count": 0,
        }
    blocker_count = int(
        conn.execute(
            f"SELECT COUNT(*) FROM {source_schema}.raw_authority_blockers WHERE resolved_at_ms IS NULL"
        ).fetchone()[0]
    )
    remediation_refs = [
        {"blocker_id": str(blocker_id), "plan_id": str(plan_id), "observed_pass_id": _text_or_none(observed_pass_id)}
        for blocker_id, plan_id, observed_pass_id in conn.execute(
            f"""
            SELECT b.blocker_id,
                   json_extract(b.expected_json, '$.plan_id'),
                   b.observed_pass_id
            FROM {source_schema}.raw_authority_blockers b
            WHERE b.resolved_at_ms IS NULL
              AND json_extract(b.expected_json, '$.authority_witness.schema') =
                  'polylogue.raw-authority-frontier-plan.v1'
            ORDER BY b.created_at_ms, b.blocker_id
            LIMIT 16
            """
        )
    ]
    return {
        "raw_authority_frontier_remediation_refs": remediation_refs,
        "raw_authority_blocker_count": blocker_count,
    }


def _text_or_none(value: object) -> str | None:
    """Normalize a nullable durable text column for a read payload."""
    return None if value is None else str(value)


def raw_materialization_readiness_from_pinned_index(
    index_conn: sqlite3.Connection,
    *,
    archive_root: Path,
    source_schema: str = "source_tier",
    classify_gaps: bool = False,
) -> dict[str, object]:
    """Project raw/index materialization from an already-pinned reader.

    Shares the path twin's failure contract: a degraded-but-readable tier
    (SQLITE_BUSY on the pinned reader, a malformed page, a column the pinned
    snapshot predates) resolves to ``{"available": False, "error": ...}``
    rather than raising. ``_raw_materialization_status`` calls this
    unconditionally, so a raise there fails the whole status operation.
    """

    try:
        return _raw_materialization_readiness_from_pinned_index(
            index_conn,
            archive_root=archive_root,
            source_schema=source_schema,
            classify_gaps=classify_gaps,
        )
    except (OSError, sqlite3.Error) as exc:
        return {"available": False, "error": str(exc)}


def _raw_materialization_readiness_from_pinned_index(
    index_conn: sqlite3.Connection,
    *,
    archive_root: Path,
    source_schema: str = "source_tier",
    classify_gaps: bool = False,
) -> dict[str, object]:
    """Project raw/index materialization from an already-pinned reader.

    ``index_conn`` must already have ``source_schema`` attached and snapshot
    forced by its caller.  The existing gap classifier is reused verbatim;
    this adapter changes only connection ownership, never policy.

    The default is the bounded periodic-status contract, matching what
    :func:`archive_readiness_status` passes to the path twin. Exact
    classification opens a blob and reads JSONL per unmaterialized raw, so
    defaulting it on made every status poll walk the whole raw corpus while
    ``sessions`` is still empty or partial -- that is, throughout a rebuild.
    Diagnostic callers ask for it explicitly.
    """

    if source_schema not in {"source", "source_tier"}:
        raise ValueError(f"unsupported raw materialization source schema: {source_schema!r}")
    aliases = {str(row[1]) for row in index_conn.execute("PRAGMA database_list").fetchall()}
    if source_schema not in aliases or not _table_exists(index_conn, "raw_sessions", schema=source_schema):
        return {"available": False, "error": "source.db or index.db missing"}
    if not _table_exists(index_conn, "sessions", schema="main"):
        return {"available": False, "error": "source.db or index.db missing"}
    conn = index_conn

    parser_census = _pinned_parser_census_projection(
        conn,
        source_schema=source_schema,
        index_conn=index_conn,
    )
    parser_non_session_count = int(cast(int, parser_census["non_session_count"])) if parser_census["available"] else 0
    authority_projection = _pinned_authority_frontier_projection(
        conn,
        source_schema=source_schema,
        index_conn=index_conn,
    )
    row = conn.execute(
        f"""
        WITH raw_rows AS (
            SELECT r.raw_id, r.origin, r.validation_status, r.parse_error, r.parsed_at_ms,
                   EXISTS (SELECT 1 FROM main.sessions s WHERE s.raw_id = r.raw_id) AS is_materialized
            FROM {source_schema}.raw_sessions r
            WHERE COALESCE(r.validation_status, '') != 'skipped'
        ), materialization AS (
            SELECT COUNT(*) AS raw_artifact_count,
                   COALESCE(SUM(CASE WHEN is_materialized THEN 1 ELSE 0 END), 0) AS materialized_raw_artifact_count
            FROM raw_rows
        ), gaps AS (
            SELECT raw_id, origin, validation_status, parse_error, parsed_at_ms FROM raw_rows WHERE NOT is_materialized
        )
        SELECT materialization.raw_artifact_count, materialization.materialized_raw_artifact_count,
               (SELECT COUNT(*) FROM main.sessions) AS archive_session_count,
               materialization.raw_artifact_count - materialization.materialized_raw_artifact_count AS join_gap_count,
               COUNT(gaps.raw_id) AS total,
               COALESCE(SUM(CASE WHEN gaps.parse_error IS NOT NULL THEN 1 ELSE 0 END), 0) AS parse_failed,
               COALESCE(SUM(CASE WHEN gaps.parsed_at_ms IS NOT NULL AND gaps.parse_error IS NULL THEN 1 ELSE 0 END), 0) AS parsed_without_index_session
        -- LEFT JOIN, not CROSS JOIN: a fully converged archive has no gap
        -- rows, and a cross join with an empty side leaves every
        -- non-aggregated materialization column NULL, which the callers below
        -- coerce to 0 -- reporting a converged archive as an empty one.
        FROM materialization LEFT JOIN gaps ON 1 = 1
        """
    ).fetchone()
    # ``source_family_counts`` is a per-origin breakdown of the same gap
    # population ``total`` counts exactly, so it is exhaustive by construction:
    # the group key is ``raw_sessions.origin``, whose vocabulary is validated
    # at the write boundary, and the aggregate is one row per distinct origin.
    # A ``LIMIT`` here would silently drop whole origins out of a census whose
    # total kept counting them -- the two numbers would stop adding up with no
    # gap named anywhere.
    family_rows = conn.execute(
        f"""
        SELECT r.origin, COUNT(*) FROM {source_schema}.raw_sessions r
        LEFT JOIN main.sessions s ON s.raw_id = r.raw_id
        WHERE s.raw_id IS NULL AND COALESCE(r.validation_status, '') != 'skipped'
        GROUP BY r.origin ORDER BY COUNT(*) DESC, r.origin
        """
    ).fetchall()
    skipped = int(
        conn.execute(
            f"SELECT COUNT(*) FROM {source_schema}.raw_sessions WHERE validation_status = 'skipped'"
        ).fetchone()[0]
        or 0
    )
    classified_counts: Counter[str] = Counter()
    parse_failed_origins: set[str] = set()
    if classify_gaps:
        gap_rows = conn.execute(
            f"""WITH raw_rows AS (
                SELECT {_raw_gap_select_columns()},
                       EXISTS (SELECT 1 FROM main.sessions s WHERE s.raw_id = r.raw_id) AS is_materialized
                FROM {source_schema}.raw_sessions r WHERE COALESCE(r.validation_status, '') != 'skipped'
            ) SELECT * FROM raw_rows WHERE NOT is_materialized"""
        ).fetchall()
        classified_counts, parse_failed_origins = _classify_raw_gap_rows(
            conn,
            archive_root,
            gap_rows,
            has_revision_applications=bool(_table_exists(index_conn, "raw_revision_applications", schema="main")),
            has_membership_census=bool(_table_exists(index_conn, "raw_membership_census", schema=source_schema)),
            has_session_memberships=bool(_table_exists(index_conn, "raw_session_memberships", schema=source_schema)),
            source_schema=source_schema,
        )
    adoption_deferred_count = 0
    if _table_exists(index_conn, "raw_revision_applications", schema="main"):
        adoption_deferred_count = int(
            conn.execute(
                f"""
                SELECT COUNT(DISTINCT r.raw_id)
                FROM {source_schema}.raw_sessions r
                JOIN main.raw_revision_applications a ON a.raw_id = r.raw_id
                WHERE a.decision = 'deferred'
                  AND a.detail = 'ordinary_replay:incomparable_existing_index_state'
                  AND NOT EXISTS (SELECT 1 FROM main.sessions s WHERE s.raw_id = r.raw_id)
                """
            ).fetchone()[0]
            or 0
        )
    total = int(row[4] or 0)
    parse_failed = int(row[5] or 0)
    classified = sum(count for category, count in classified_counts.items() if category not in _RAW_GAP_OWED_CATEGORIES)
    cheap_non_session_count = 0 if classify_gaps else parser_non_session_count
    classified += cheap_non_session_count
    affected_actionable = classified_counts.get("parse-failed", 0) + classified_counts.get(
        RAW_ALIAS_BLOB_MISSING_CATEGORY, 0
    )
    unchecked = max(total - classified - affected_actionable - adoption_deferred_count, 0)
    category_counts: dict[str, int] = {
        "raw_id_join_gap": unchecked,
        "skipped": skipped,
        "parse_failed": affected_actionable,
        "raw_parse_failed": parse_failed,
        "parsed_without_index_session": int(row[6] or 0),
    }
    if cheap_non_session_count:
        category_counts["parsed-non-session-artifact"] = cheap_non_session_count
    if adoption_deferred_count:
        category_counts["adoption_deferred"] = adoption_deferred_count
    category_counts.update(
        {category: count for category, count in classified_counts.items() if category != "parse-failed"}
    )
    return {
        "available": True,
        "classification": "cheap_projection"
        if (classify_gaps and (classified or adoption_deferred_count)) or cheap_non_session_count
        else "not_run",
        "precision": "raw_id_join_gap",
        "raw_artifact_count": int(row[0] or 0),
        "materialized_raw_artifact_count": int(row[1] or 0),
        "archive_session_count": int(row[2] or 0),
        "join_gap_count": int(row[3] or total),
        "total": total,
        "critical": len(parse_failed_origins),
        "warning": 0,
        "actionable": len(parse_failed_origins),
        "blocked": adoption_deferred_count,
        "classified": classified,
        "unchecked": unchecked,
        "affected_total": total,
        "affected_actionable": affected_actionable,
        "affected_blocked": adoption_deferred_count,
        "affected_open": 0,
        "affected_classified": classified,
        "affected_unchecked": unchecked,
        "lost_source_evidence_count": _missing_source_raw_session_count(conn, source_schema=source_schema),
        "lost_source_evidence_samples": _missing_source_raw_session_samples(conn, source_schema=source_schema),
        "category_counts": category_counts,
        "source_family_counts": {str(item[0]): int(item[1] or 0) for item in family_rows},
        **authority_projection,
        "raw_authority_parser_census": parser_census,
        "raw_authority_parser_census_incomplete_count": _safe_int(parser_census["incomplete_count"]),
        "raw_authority_parser_census_incomplete_blob_bytes": _safe_int(parser_census["incomplete_blob_bytes"]),
    }


def raw_materialization_readiness_snapshot(
    active_archive: Path,
    *,
    classify_gaps: bool = True,
) -> dict[str, object]:
    """Return compact raw→index materialization readiness for an archive root.

    Exact classification may inspect every raw-id gap and is therefore reserved
    for explicit diagnostic reads. ``classify_gaps=False`` keeps the aggregate
    counters and durable authority state but marks all unclassified gaps as
    unchecked, which is the bounded periodic-status contract.
    """
    from polylogue.storage.archive_identity import ArchiveLocation, ArchiveLocationError

    try:
        location = ArchiveLocation.resolve(active_archive)
    except ArchiveLocationError as exc:
        return {"available": False, "error": str(exc)}
    active_archive = location.configured_root
    source_db = location.active_tier("source").resolved_path
    index_db = location.active_tier("index").resolved_path
    if not source_db.exists() or not index_db.exists():
        return {"available": False, "error": "source.db or index.db missing"}
    try:
        # Source selection decides which raws the frontier admits, so it reads
        # through the schema-enforcing open: a bare mode=ro connect would
        # classify an index this runtime cannot interpret.
        from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
        from polylogue.storage.sqlite.connection_profile import open_readonly_connection

        with closing(open_readonly_connection(index_db, tier=ArchiveTier.INDEX)) as conn:
            conn.row_factory = sqlite3.Row
            attach_readonly_database(conn, source_db, alias="source")

            row = conn.execute(
                """
                WITH raw_rows AS (
                    SELECT
                        r.raw_id,
                        r.origin,
                        r.validation_status,
                        r.parse_error,
                        r.parsed_at_ms,
                        EXISTS (
                            SELECT 1
                            FROM main.sessions s
                            WHERE s.raw_id = r.raw_id
                        ) AS is_materialized
                    FROM source.raw_sessions r
                    WHERE COALESCE(r.validation_status, '') != 'skipped'
                ),
                materialization AS (
                    SELECT
                        COUNT(*) AS raw_artifact_count,
                        COALESCE(SUM(CASE WHEN is_materialized THEN 1 ELSE 0 END), 0)
                            AS materialized_raw_artifact_count
                    FROM raw_rows
                ),
                session_count AS (
                    SELECT COUNT(*) AS archive_session_count FROM main.sessions
                ),
                gaps AS (
                    SELECT raw_id, origin, validation_status, parse_error, parsed_at_ms
                    FROM raw_rows
                    WHERE NOT is_materialized
                )
                SELECT
                    materialization.raw_artifact_count,
                    materialization.materialized_raw_artifact_count,
                    session_count.archive_session_count,
                    (materialization.raw_artifact_count - materialization.materialized_raw_artifact_count)
                        AS join_gap_count,
                    COUNT(gaps.raw_id) AS total,
                    COALESCE(SUM(CASE WHEN validation_status = 'skipped' THEN 1 ELSE 0 END), 0) AS skipped,
                    COALESCE(SUM(CASE WHEN parse_error IS NOT NULL THEN 1 ELSE 0 END), 0) AS parse_failed,
                    COALESCE(SUM(CASE WHEN parsed_at_ms IS NOT NULL AND parse_error IS NULL THEN 1 ELSE 0 END), 0)
                        AS parsed_without_index_session
                -- The one-row totals drive the join, and the gap rows hang off
                -- them. Driving from ``gaps`` instead leaves every
                -- non-aggregated total NULL once the archive has converged and
                -- ``gaps`` is empty, so a fully materialized archive reported
                -- zero raw artifacts and zero sessions.
                FROM materialization
                CROSS JOIN session_count
                LEFT JOIN gaps ON 1 = 1
                """
            ).fetchone()
            family_rows = conn.execute(
                """
                SELECT r.origin, COUNT(*) AS count
                FROM source.raw_sessions r
                LEFT JOIN main.sessions s ON s.raw_id = r.raw_id
                WHERE s.raw_id IS NULL
                  AND COALESCE(r.validation_status, '') != 'skipped'
                GROUP BY r.origin
                ORDER BY count DESC, r.origin
                LIMIT 16
                """
            ).fetchall()
            classified_counts: Counter[str] = Counter()
            parse_failed_origins: set[str] = set()
            if classify_gaps:
                raw_select_columns = _raw_gap_select_columns()
                gap_rows = conn.execute(
                    f"""
                    WITH raw_rows AS (
                        SELECT
                            {raw_select_columns},
                            EXISTS (
                                SELECT 1
                                FROM main.sessions s
                                WHERE s.raw_id = r.raw_id
                            ) AS is_materialized
                        FROM source.raw_sessions r
                        WHERE COALESCE(r.validation_status, '') != 'skipped'
                    )
                    SELECT *
                    FROM raw_rows
                    WHERE NOT is_materialized
                    """,
                ).fetchall()
                classified_counts, parse_failed_origins = _classify_raw_gap_rows(
                    conn,
                    active_archive,
                    gap_rows,
                    has_revision_applications=bool(_table_exists(conn, "raw_revision_applications", schema="main")),
                    has_membership_census=bool(_table_exists(conn, "raw_membership_census", schema="source")),
                    has_session_memberships=bool(_table_exists(conn, "raw_session_memberships", schema="source")),
                )
            adoption_deferred_count = 0
            if _table_exists(conn, "raw_revision_applications", schema="main"):
                adoption_deferred_count = int(
                    conn.execute(
                        """
                        SELECT COUNT(DISTINCT r.raw_id)
                        FROM source.raw_sessions AS r
                        JOIN main.raw_revision_applications AS a ON a.raw_id = r.raw_id
                        WHERE a.decision = 'deferred'
                          AND a.detail = 'ordinary_replay:incomparable_existing_index_state'
                          AND NOT EXISTS (
                              SELECT 1 FROM main.sessions AS s WHERE s.raw_id = r.raw_id
                          )
                        """
                    ).fetchone()[0]
                    or 0
                )
            lost_source_evidence_count = _missing_source_raw_session_count(conn)
            lost_source_evidence_samples = _missing_source_raw_session_samples(conn)
            authority_frontier_remediation_refs: list[dict[str, object]] = []
            parser_census_available = False
            parser_census_complete_count = 0
            parser_census_non_session_count = 0
            parser_census_incomplete_count = 0
            parser_census_incomplete_blob_bytes = 0
            parser_census_missing_receipt_count = 0
            parser_census_non_complete_receipt_count = 0
            parser_census_origin_summary: list[dict[str, object]] = []
            parser_summary = _pinned_parser_census_projection(
                conn,
                source_schema="source",
                index_conn=conn,
            )
            parser_census_available = bool(parser_summary["available"])
            parser_census_complete_count = int(cast(int, parser_summary["complete_count"]))
            parser_census_non_session_count = int(cast(int, parser_summary["non_session_count"]))
            parser_census_incomplete_count = int(cast(int, parser_summary["incomplete_count"]))
            parser_census_incomplete_blob_bytes = int(cast(int, parser_summary["incomplete_blob_bytes"]))
            parser_census_missing_receipt_count = int(cast(int, parser_summary["missing_receipt_count"]))
            parser_census_non_complete_receipt_count = int(cast(int, parser_summary["non_complete_receipt_count"]))
            parser_census_origin_summary = cast(list[dict[str, object]], parser_summary["incomplete_origin_summary"])
            authority_blocker_count = 0
            if _table_exists(conn, "raw_authority_blockers", schema="source"):
                authority_blocker_count = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM source.raw_authority_blockers WHERE resolved_at_ms IS NULL"
                    ).fetchone()[0]
                )
                authority_frontier_remediation_refs = [
                    {
                        "blocker_id": str(blocker_id),
                        "plan_id": str(plan_id),
                        "observed_pass_id": _text_or_none(observed_pass_id),
                    }
                    for blocker_id, plan_id, observed_pass_id in conn.execute(
                        """
                        SELECT b.blocker_id,
                               json_extract(b.expected_json, '$.plan_id'),
                               b.observed_pass_id
                        FROM source.raw_authority_blockers AS b
                        WHERE b.resolved_at_ms IS NULL
                          AND json_extract(b.expected_json, '$.authority_witness.schema') =
                              'polylogue.raw-authority-frontier-plan.v1'
                        ORDER BY b.created_at_ms, b.blocker_id
                        LIMIT 16
                        """
                    )
                ]
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
        }
    total = int(row["total"] or 0)
    raw_artifact_count = int(row["raw_artifact_count"] or 0)
    materialized_raw_artifact_count = int(row["materialized_raw_artifact_count"] or 0)
    archive_session_count = int(row["archive_session_count"] or 0)
    join_gap_count = int(row["join_gap_count"] or total)
    skipped = int(row["skipped"] or 0)
    raw_parse_failed = int(row["parse_failed"] or 0)
    parsed_without_index_session = int(row["parsed_without_index_session"] or 0)
    parse_failed = classified_counts.get("parse-failed", 0)
    classified = sum(count for category, count in classified_counts.items() if category not in _RAW_GAP_OWED_CATEGORIES)
    cheap_non_session_count = 0 if classify_gaps else parser_census_non_session_count
    classified += cheap_non_session_count
    alias_blob_missing = classified_counts.get(RAW_ALIAS_BLOB_MISSING_CATEGORY, 0)
    actionable = len(parse_failed_origins)
    critical = actionable
    affected_actionable = parse_failed + alias_blob_missing
    unchecked = max(total - classified - affected_actionable - adoption_deferred_count, 0)
    classification = "cheap_projection" if classify_gaps and (classified or adoption_deferred_count) else "not_run"
    if cheap_non_session_count:
        classification = "cheap_projection"
    raw_id_join_gap_count = unchecked
    category_counts: dict[str, int] = {
        "raw_id_join_gap": raw_id_join_gap_count,
        "skipped": skipped,
        "parse_failed": parse_failed,
        "raw_parse_failed": raw_parse_failed,
        "parsed_without_index_session": parsed_without_index_session,
    }
    if cheap_non_session_count:
        category_counts["parsed-non-session-artifact"] = cheap_non_session_count
    if adoption_deferred_count:
        category_counts["adoption_deferred"] = adoption_deferred_count
    category_counts.update(
        {category: count for category, count in classified_counts.items() if category != "parse-failed"}
    )
    return {
        "available": True,
        "classification": classification,
        "precision": "raw_id_join_gap",
        "raw_artifact_count": raw_artifact_count,
        "materialized_raw_artifact_count": materialized_raw_artifact_count,
        "archive_session_count": archive_session_count,
        "join_gap_count": join_gap_count,
        "total": total,
        "critical": critical,
        "warning": 0,
        "actionable": actionable,
        "blocked": adoption_deferred_count,
        "classified": classified,
        "unchecked": unchecked,
        "affected_total": total,
        "affected_actionable": affected_actionable,
        "affected_blocked": adoption_deferred_count,
        "affected_open": 0,
        "affected_classified": classified,
        "affected_unchecked": unchecked,
        "lost_source_evidence_count": lost_source_evidence_count,
        "lost_source_evidence_samples": lost_source_evidence_samples,
        "category_counts": category_counts,
        "source_family_counts": {str(item["origin"]): int(item["count"] or 0) for item in family_rows},
        "raw_authority_frontier_remediation_refs": authority_frontier_remediation_refs,
        "raw_authority_blocker_count": authority_blocker_count,
        "raw_authority_parser_census": {
            "available": parser_census_available,
            "complete_count": parser_census_complete_count,
            "incomplete_count": parser_census_incomplete_count,
            "incomplete_blob_bytes": parser_census_incomplete_blob_bytes,
            "missing_receipt_count": parser_census_missing_receipt_count,
            "non_complete_receipt_count": parser_census_non_complete_receipt_count,
            "incomplete_origin_summary": parser_census_origin_summary,
        },
        "raw_authority_parser_census_incomplete_count": parser_census_incomplete_count,
        "raw_authority_parser_census_incomplete_blob_bytes": parser_census_incomplete_blob_bytes,
    }


def missing_source_raw_session_evidence(active_archive: Path, *, limit: int = 10) -> dict[str, object]:
    """Return indexed sessions whose source raw evidence is no longer present.

    This is the reverse of raw materialization debt. Raw materialization asks
    whether source rows have reached the index. This helper asks whether an
    indexed session still has the source row named by ``sessions.raw_id``. A
    missing row is lost source evidence until the exact raw artifact is
    recovered; it must not be repaired by relinking to a same-native but
    different source row.
    """

    source_db = active_archive / "source.db"
    index_db = active_archive / "index.db"
    if not source_db.exists() or not index_db.exists():
        return {
            "available": False,
            "reason": "source.db or index.db missing",
            "missing_raw_session_count": 0,
            "missing_raw_session_samples": [],
            "lost_source_evidence_count": 0,
            "lost_source_evidence_samples": [],
        }
    try:
        with closing(
            open_readonly_connection(index_db, timeout_class="background-read", validate_schema=False)
        ) as conn:
            conn.row_factory = sqlite3.Row
            attach_readonly_database(conn, source_db, alias="source")
            if not _table_exists(conn, "sessions", schema="main") or not _table_exists(
                conn, "raw_sessions", schema="source"
            ):
                return {
                    "available": False,
                    "reason": "sessions or raw_sessions table missing",
                    "missing_raw_session_count": 0,
                    "missing_raw_session_samples": [],
                    "lost_source_evidence_count": 0,
                    "lost_source_evidence_samples": [],
                }
            count = _missing_source_raw_session_count(conn)
            samples = _missing_source_raw_session_samples(conn, limit=limit)
    except sqlite3.Error as exc:
        return {
            "available": False,
            "reason": str(exc),
            "missing_raw_session_count": 0,
            "missing_raw_session_samples": [],
            "lost_source_evidence_count": 0,
            "lost_source_evidence_samples": [],
        }
    return {
        "available": True,
        "reason": None,
        "missing_raw_session_count": count,
        "missing_raw_session_samples": samples,
        "lost_source_evidence_count": count,
        "lost_source_evidence_samples": samples,
    }


def _missing_source_raw_session_count(
    conn: sqlite3.Connection,
    *,
    source_schema: str = "source",
) -> int:
    if not _table_exists(conn, "sessions"):
        return 0
    return _readiness_scalar_int(
        conn,
        f"""
        SELECT COUNT(*)
        FROM sessions AS s
        WHERE s.raw_id IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM {source_schema}.raw_sessions AS r WHERE r.raw_id = s.raw_id
          )
        """,
    )


def _missing_source_raw_session_samples(
    conn: sqlite3.Connection,
    *,
    limit: int = 10,
    source_schema: str = "source",
) -> list[dict[str, object]]:
    if not _table_exists(conn, "sessions"):
        return []
    rows = conn.execute(
        f"""
        SELECT s.session_id,
               s.origin AS origin,
               s.native_id AS native_id,
               s.raw_id,
               s.message_count AS message_count,
               s.updated_at_ms AS updated_at_ms
        FROM sessions AS s
        WHERE s.raw_id IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM {source_schema}.raw_sessions AS r WHERE r.raw_id = s.raw_id
          )
        ORDER BY s.updated_at_ms DESC, s.session_id
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "session_id": str(row[0]),
            "origin": str(row[1]),
            "native_id": str(row[2]),
            "missing_raw_id": str(row[3]),
            "message_count": int(row[4] or 0),
            "updated_at_ms": None if row[5] is None else int(row[5]),
            "evidence_status": "lost_source_evidence",
            "loss_reason": "index_raw_id_missing_from_source_tier",
            "recovery_requirement": "restore_exact_raw_artifact_or_keep_blocked",
        }
        for row in rows
    ]


def _readiness_scalar_int(conn: sqlite3.Connection, sql: str) -> int:
    row = conn.execute(sql).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _raw_gap_select_columns() -> str:
    names = (
        "raw_id",
        "origin",
        "native_id",
        "source_path",
        "blob_hash",
        "source_index",
        "revision_authority",
        "validation_status",
        "parse_error",
        "parsed_at_ms",
        "logical_source_key",
        "revision_kind",
    )
    return ",\n                        ".join(f"r.{name}" for name in names)


#: A raw row whose logical session is present under an alias, but whose own
#: content-addressed blob is absent. The alias reconciles identity, not bytes:
#: the indexed session is a different (usually older) snapshot, so this row is
#: still owed work and must never be reported as a satisfied materialization.
RAW_ALIAS_BLOB_MISSING_CATEGORY = "materialized-alias-blob-missing"

#: Gap categories that are owed work rather than benign classifications. They
#: are excluded from ``classified`` and counted as actionable.
_RAW_GAP_OWED_CATEGORIES = frozenset({"parse-failed", RAW_ALIAS_BLOB_MISSING_CATEGORY})


def _raw_gap_blob_present(archive_root: Path, row: sqlite3.Row) -> bool:
    """Report whether this row's own content-addressed blob exists on disk.

    An unreadable or absent hash is reported as "not present": an unmeasured
    blob is never allowed to stand in for a proven one.
    """
    blob_hash = row["blob_hash"]
    if blob_hash is None:
        return False
    hex_hash = blob_hash.hex() if isinstance(blob_hash, bytes) else str(blob_hash)
    if len(hex_hash) < 3:
        return False
    return (archive_root / "blob" / hex_hash[:2] / hex_hash[2:]).exists()


def _classify_raw_gap_rows(
    conn: sqlite3.Connection,
    archive_root: Path,
    rows: list[sqlite3.Row],
    *,
    has_revision_applications: bool,
    has_membership_census: bool,
    has_session_memberships: bool,
    source_schema: str = "source",
) -> tuple[Counter[str], set[str]]:
    if not rows:
        return Counter(), set()
    counts: Counter[str] = Counter()
    parse_failed_origins: set[str] = set()
    for row in rows:
        category = _raw_gap_category(
            conn,
            archive_root,
            row,
            has_revision_applications=has_revision_applications,
            has_membership_census=has_membership_census,
            has_session_memberships=has_session_memberships,
            source_schema=source_schema,
        )
        if category is not None:
            counts[category] += 1
            if category == "parse-failed":
                parse_failed_origins.add(str(row["origin"] or "unknown"))
    return counts, parse_failed_origins


def _raw_gap_category(
    conn: sqlite3.Connection,
    archive_root: Path,
    row: sqlite3.Row,
    *,
    has_revision_applications: bool,
    has_membership_census: bool,
    has_session_memberships: bool,
    source_schema: str = "source",
) -> str | None:
    can_reconcile_alias = not row["parse_error"] or _retryable_decode_missing_blob_error(row["parse_error"])
    if can_reconcile_alias and _raw_gap_materialized_by_alias(
        conn,
        row,
        source_schema=source_schema,
    ):
        # The alias proves a same-identity session is indexed; it does not
        # prove this artifact's bytes survive. Without its own blob the row is
        # a distinct, unrecoverable snapshot and stays reported as owed work.
        if _raw_gap_blob_present(archive_root, row):
            return "materialized-alias"
        return RAW_ALIAS_BLOB_MISSING_CATEGORY
    if can_reconcile_alias and _raw_gap_matches_missing_index_raw_link(
        conn,
        row,
        source_schema=source_schema,
    ):
        return "lost-source-evidence-alias"
    # Parser failures and explicit validation refusals remain visible even if
    # their bytes or path would otherwise look like a non-session artifact.
    if row["parse_error"] or row["validation_status"] == "failed":
        return "parse-failed"
    if current_typed_non_session_raw(
        conn,
        row,
        source_schema=source_schema,
        has_membership_census=has_membership_census,
        has_session_memberships=has_session_memberships,
    ):
        return "parsed-non-session-artifact"
    authority_category = _raw_gap_authority_category(
        conn,
        row,
        has_revision_applications=has_revision_applications,
        has_membership_census=has_membership_census,
        has_session_memberships=has_session_memberships,
        source_schema=source_schema,
    )
    if authority_category is not None:
        return authority_category
    return None


def current_typed_non_session_raw(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    source_schema: str,
    has_membership_census: bool,
    has_session_memberships: bool,
) -> bool:
    """Use current typed artifact and parser receipts to settle a non-session gap.

    The parser receipt and exact identity measurement are the same authority
    used by the archive's parser-census projection. A filename or a second
    content classifier cannot turn an unmeasured raw into a completed input.
    """
    if (
        not has_membership_census
        or not has_session_memberships
        or not _table_exists(conn, "raw_authority_parser_census", schema=source_schema)
        or not _table_exists(conn, "raw_artifacts", schema=source_schema)
    ):
        return False
    raw_id = str(row["raw_id"])
    fingerprint = raw_authority_parser_fingerprint()
    terminal_pairs = RAW_FAILURE_TERMINAL_EVIDENCE_SUPPORT_STATUS_PAIRS
    terminal_placeholders = ", ".join("(?, ?)" for _ in terminal_pairs)
    receipt = conn.execute(
        f"""SELECT p.parser_fingerprint, p.status, p.logical_keys_json,
                   c.parser_fingerprint, c.status, c.member_count
            FROM {source_schema}.raw_authority_parser_census AS p
            JOIN {source_schema}.raw_membership_census AS c ON c.raw_id=p.raw_id
            WHERE p.raw_id=?
              AND EXISTS (
                  SELECT 1 FROM {source_schema}.raw_artifacts AS a
                  WHERE a.raw_id=p.raw_id AND a.parse_as_session=0
                    AND a.schema_eligible=0 AND a.decode_error IS NULL
                    AND a.malformed_jsonl_lines=0
                    AND (a.artifact_kind, a.support_status) NOT IN ({terminal_placeholders})
              )
              AND NOT EXISTS (
                  SELECT 1 FROM {source_schema}.raw_artifacts AS failure
                  WHERE failure.raw_id=p.raw_id
                    AND (failure.artifact_kind, failure.support_status) IN ({terminal_placeholders})
              )
              AND NOT EXISTS (
                  SELECT 1 FROM {source_schema}.raw_artifacts AS sibling
                  WHERE sibling.raw_id=p.raw_id
                    AND (sibling.parse_as_session<>0 OR sibling.schema_eligible<>0
                         OR sibling.decode_error IS NOT NULL OR sibling.malformed_jsonl_lines<>0)
              )""",
        (
            raw_id,
            *(value for pair in terminal_pairs for value in pair),
            *(value for pair in terminal_pairs for value in pair),
        ),
    ).fetchone()
    if receipt is None:
        return False
    (
        parser_fingerprint,
        parser_status,
        logical_keys_json,
        membership_fingerprint,
        membership_status,
        member_count,
    ) = receipt
    if (
        str(parser_fingerprint) != fingerprint
        or str(parser_status) != "complete"
        or str(membership_fingerprint) != fingerprint
        or str(membership_status) != "non_session"
        or int(member_count) != 0
    ):
        return False

    def membership_keys() -> Any:
        with closing(
            conn.execute(
                f"SELECT logical_source_key FROM {source_schema}.raw_session_memberships "
                "WHERE raw_id=? ORDER BY logical_source_key",
                (raw_id,),
            )
        ) as memberships:
            for membership in memberships:
                yield membership[0]

    with parser_census_identity_measurement(
        raw_logical_key=row["logical_source_key"],
        revision_kind=row["revision_kind"],
        membership_logical_keys=membership_keys(),
        observed_logical_keys=iter_parser_census_logical_keys(logical_keys_json),
        observed_are_receipt=True,
    ) as measured:
        return measured.complete(
            typed_non_session=True,
            parser_confirmed_non_session=True,
            byte_governed_fragment=False,
        )


def _raw_gap_authority_category(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    has_revision_applications: bool,
    has_membership_census: bool,
    has_session_memberships: bool,
    source_schema: str = "source",
) -> str | None:
    raw_id = str(row["raw_id"])
    if has_revision_applications:
        terminal = conn.execute(
            """
            SELECT 1 FROM main.raw_revision_applications
            WHERE raw_id = ?
              AND decision IN ('selected_baseline', 'applied_append', 'superseded', 'ambiguous')
            LIMIT 1
            """,
            (raw_id,),
        ).fetchone()
        if terminal is not None:
            return "revision-application-terminal"

    if has_membership_census and has_session_memberships:
        membership = conn.execute(
            f"""
            SELECT 1
            FROM {source_schema}.raw_membership_census AS c
            WHERE c.raw_id = ?
              AND c.status = 'complete'
              AND c.member_count > 0
              AND c.member_count = (
                SELECT COUNT(*) FROM {source_schema}.raw_session_memberships AS counted
                WHERE counted.raw_id = c.raw_id
              )
              AND NOT EXISTS (
                SELECT 1 FROM {source_schema}.raw_session_memberships AS m
                WHERE m.raw_id = c.raw_id
                  AND (m.decision IS NULL OR m.decision = 'deferred')
              )
            LIMIT 1
            """,
            (raw_id,),
        ).fetchone()
        if membership is not None:
            return "membership-authority-classified"

    if row["source_index"] == -1:
        if row["revision_authority"] == "byte_proven":
            return "append-authority-proven"
        if has_membership_census:
            quarantined = conn.execute(
                f"""
                SELECT 1 FROM {source_schema}.raw_membership_census
                WHERE raw_id = ? AND status = 'failed' AND revision_authority = ?
                LIMIT 1
                """,
                (raw_id, RawRevisionAuthority.BYTE_PROVEN.value),
            ).fetchone()
            if quarantined is not None:
                return "append-authority-quarantined"
    return None


def _retryable_decode_missing_blob_error(parse_error: object) -> bool:
    if not isinstance(parse_error, str):
        return False
    return parse_error.startswith("decode:") and "No such file or directory" in parse_error


def _raw_gap_materialized_by_alias(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    source_schema: str = "source",
) -> bool:
    origin = str(row["origin"] or "")
    if not origin:
        return False
    native_ids: list[str] = []

    def add(value: object) -> None:
        if isinstance(value, str) and value and value not in native_ids:
            native_ids.append(value)

    add(row["native_id"])
    for candidate in source_path_native_id_candidates(str(row["source_path"] or "")):
        add(candidate)
    if not native_ids:
        return False
    for native_id in native_ids:
        existing = conn.execute(
            f"""
            SELECT 1
            FROM main.sessions AS s
            JOIN {source_schema}.raw_sessions AS existing_raw ON existing_raw.raw_id = s.raw_id
            WHERE s.origin = ?
              AND s.native_id = ?
            LIMIT 1
            """,
            (origin, native_id),
        ).fetchone()
        if existing is not None:
            return True
    return False


def _raw_gap_matches_missing_index_raw_link(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    source_schema: str = "source",
) -> bool:
    origin = str(row["origin"] or "")
    if not origin:
        return False
    native_ids: list[str] = []

    def add(value: object) -> None:
        if isinstance(value, str) and value and value not in native_ids:
            native_ids.append(value)

    add(row["native_id"])
    for candidate in source_path_native_id_candidates(str(row["source_path"] or "")):
        add(candidate)
    if not native_ids:
        return False
    for native_id in native_ids:
        existing = conn.execute(
            f"""
            SELECT 1
            FROM main.sessions AS s
            WHERE s.origin = ?
              AND s.native_id = ?
              AND s.raw_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM {source_schema}.raw_sessions AS existing_raw WHERE existing_raw.raw_id = s.raw_id
              )
            LIMIT 1
            """,
            (origin, native_id),
        ).fetchone()
        if existing is not None:
            return True
    return False


# ---------------------------------------------------------------------------
# Archive readiness surfaces
#
# The substrate home for the exact-readiness computation; the CLI
# (``status.py``) delegates to ``archive_readiness_status`` below. The tiny
# SQLite-introspection one-liners (``_fast_count``/``_safe_int``/
# ``_table_exists``/etc.) are duplicated from ``status.py``'s own private
# copies, which serve unrelated status surfaces.
# ---------------------------------------------------------------------------


def _fast_count(conn: sqlite3.Connection, sql: str, params: tuple[object, ...] = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    return int(row[0] or 0) if row is not None else 0


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _action_readiness_counts(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return exact, non-vacuous evidence for the derived ``actions`` view."""
    tool_use_block_count = (
        _fast_count(conn, "SELECT COUNT(*) FROM blocks WHERE block_type = 'tool_use'")
        if _table_exists(conn, "blocks")
        else 0
    )
    actions_view_present = view_exists(conn, "actions")
    action_count = 0
    actions_view_error: str | None = None
    if actions_view_present:
        try:
            action_count = _fast_count(conn, "SELECT COUNT(*) FROM actions")
        except sqlite3.Error as exc:
            actions_view_error = str(exc)
    return {
        "action_count": action_count,
        "tool_use_block_count": tool_use_block_count,
        "actions_view_present": actions_view_present,
        "actions_view_error": actions_view_error,
    }


def _archive_readiness_counts(
    conn: sqlite3.Connection,
    *,
    source_conn: sqlite3.Connection | None,
    source_check_available: bool,
) -> dict[str, Any]:
    sessions_table_present = _table_exists(conn, "sessions")
    messages_table_present = _table_exists(conn, "messages")
    session_count = _fast_count(conn, "SELECT COUNT(*) FROM sessions") if sessions_table_present else 0
    raw_link_count = (
        _fast_count(conn, "SELECT COUNT(*) FROM sessions WHERE raw_id IS NOT NULL")
        if _table_exists(conn, "sessions")
        else 0
    )
    missing_raw_session_count = 0
    missing_raw_session_samples: list[dict[str, Any]] = []
    if source_check_available and source_conn is not None and _table_exists(conn, "sessions"):
        if not _table_exists(conn, "raw_sessions", schema="source_tier"):
            raise sqlite3.OperationalError("exact readiness requires the pinned source_tier attachment")
        missing_raw_session_count = _missing_source_raw_session_count(conn, source_schema="source_tier")
        missing_raw_session_samples = _missing_source_raw_session_samples(conn, source_schema="source_tier")
    insight_status = session_insight_status_sync(conn, verify_freshness=True)
    return {
        # Relation presence is the evidence that separates a measured empty
        # archive from one whose session/message relations could not be read
        # at all; without it the sessions surface reported ready=True either
        # way (polylogue-bu47u).
        "sessions_table_present": sessions_table_present,
        "messages_table_present": messages_table_present,
        "session_count": session_count,
        "raw_link_count": raw_link_count,
        "missing_raw_session_count": missing_raw_session_count,
        "missing_raw_session_samples": missing_raw_session_samples,
        "lost_source_evidence_count": missing_raw_session_count,
        "lost_source_evidence_samples": missing_raw_session_samples,
        "message_count": _fast_count(conn, "SELECT COUNT(*) FROM messages") if _table_exists(conn, "messages") else 0,
        "text_block_count": _fast_count(conn, "SELECT COUNT(*) FROM blocks WHERE search_text != ''")
        if _table_exists(conn, "blocks")
        else 0,
        "messages_fts_count": _fast_count(conn, "SELECT COUNT(*) FROM messages_fts")
        if _table_exists(conn, "messages_fts")
        else 0,
        "profile_row_count": insight_status.profile_row_count,
        "missing_profile_row_count": insight_status.missing_profile_row_count,
        "stale_profile_row_count": insight_status.stale_profile_row_count,
        "orphan_profile_row_count": insight_status.orphan_profile_row_count,
        "thread_count": insight_status.thread_count,
        "root_thread_count": insight_status.root_threads,
        "stale_thread_count": insight_status.stale_thread_count,
        "orphan_thread_count": insight_status.orphan_thread_count,
        **_action_readiness_counts(conn),
        "missing_latency_profile_row_count": insight_status.missing_latency_profile_row_count,
    }


def _archive_status_surfaces(counts: dict[str, Any], *, source_check_available: bool) -> dict[str, dict[str, Any]]:
    def surface(*, ready: bool | None, blockers: list[str], evidence: dict[str, Any]) -> dict[str, Any]:
        return {"ready": ready, "blockers": blockers, "evidence": evidence}

    def count(key: str, default: int = 0) -> int:
        return int(counts.get(key, default))

    def present_blockers(*keys: str) -> list[str]:
        return [key for key in keys if count(key) != 0]

    def mismatch_blocker(actual_key: str, expected_key: str, blocker: str) -> list[str]:
        expected = count(expected_key, count(actual_key))
        return [blocker] if count(actual_key) != expected else []

    parser_census = counts.get("raw_authority_parser_census")
    parser_census_incomplete_count = count("raw_authority_parser_census_incomplete_count")
    assessment = assess_raw_materialization(
        {
            **counts,
            "available": source_check_available and counts.get("available", True) is True,
            "lost_source_evidence_count": count("missing_raw_session_count"),
        }
    )
    raw_blockers: list[str] = []
    raw_ready: bool | None
    if not source_check_available:
        raw_ready = None
        raw_blockers.append("source_tier_unavailable")
    else:
        raw_ready = None if assessment.state is RawMaterializationAssessmentState.UNMEASURED else assessment.ready
        for key, value in assessment.blocking_counts:
            if value:
                raw_blockers.append(
                    {
                        "raw_authority_parser_census_incomplete_count": "parser_census_incomplete",
                        "lost_source_evidence_count": "missing_source_raw_sessions",
                    }.get(key, key)
                )
        if not raw_blockers and not assessment.ready:
            raw_blockers.append(assessment.reason)

    # The sessions surface published a literal ``ready=True`` whatever the
    # counts said, so a missing sessions/messages relation -- which yields a
    # fabricated zero here -- still certified the surface (polylogue-bu47u).
    # Compute it from blockers like every sibling surface.
    session_blockers: list[str] = []
    if not bool(counts.get("sessions_table_present", True)):
        session_blockers.append("sessions_relation_missing")
    if not bool(counts.get("messages_table_present", True)):
        session_blockers.append("messages_relation_missing")

    search_blockers = ["messages_fts_row_mismatch"] if count("text_block_count") != count("messages_fts_count") else []
    profile_blockers: list[str] = []
    if count("missing_profile_row_count"):
        profile_blockers.append("missing_profile_rows")
    profile_blockers.extend(present_blockers("stale_profile_row_count", "orphan_profile_row_count"))

    thread_blockers = present_blockers(
        "missing_thread_row_count",
        "stale_thread_count",
        "orphan_thread_count",
    )
    thread_blockers.extend(mismatch_blocker("thread_count", "root_thread_count", "thread_root_mismatch"))
    latency_blockers = present_blockers("missing_latency_profile_row_count", "stale_latency_profile_row_count")
    latency_ready = not latency_blockers
    tool_usage_blockers: list[str] = []
    if not bool(counts.get("actions_view_present", False)):
        tool_usage_blockers.append("actions_view_missing")
    elif counts.get("actions_view_error"):
        tool_usage_blockers.append("actions_view_unreadable")
    elif count("tool_use_block_count") != count("action_count"):
        tool_usage_blockers.append("actions_tool_use_count_mismatch")

    return {
        "archive_sessions": surface(
            ready=not session_blockers,
            blockers=session_blockers,
            evidence={
                "session_count": count("session_count"),
                "message_count": count("message_count"),
                "sessions_table_present": bool(counts.get("sessions_table_present", True)),
                "messages_table_present": bool(counts.get("messages_table_present", True)),
            },
        ),
        "raw_artifacts": surface(
            ready=raw_ready,
            blockers=raw_blockers,
            evidence={
                "source_check_available": source_check_available,
                "raw_link_count": count("raw_link_count"),
                "missing_raw_session_count": count("missing_raw_session_count"),
                "missing_raw_session_samples": list(counts.get("missing_raw_session_samples") or []),
                "lost_source_evidence_count": count("lost_source_evidence_count"),
                "lost_source_evidence_samples": list(counts.get("lost_source_evidence_samples") or []),
                "parser_census": dict(parser_census) if isinstance(parser_census, Mapping) else None,
                "parser_census_incomplete_count": parser_census_incomplete_count,
                "materialization": assessment.to_dict(),
            },
        ),
        "search": surface(
            ready=not search_blockers,
            blockers=search_blockers,
            evidence={
                "text_block_count": count("text_block_count"),
                "messages_fts_count": count("messages_fts_count"),
            },
        ),
        "session_profiles": surface(
            ready=not profile_blockers,
            blockers=profile_blockers,
            evidence={
                "profile_row_count": count("profile_row_count"),
                "missing_profile_row_count": count("missing_profile_row_count"),
                "missing_row_count": count("missing_profile_row_count"),
                "stale_profile_row_count": count("stale_profile_row_count"),
                "orphan_profile_row_count": count("orphan_profile_row_count"),
            },
        ),
        "threads": surface(
            ready=not thread_blockers,
            blockers=thread_blockers,
            evidence={
                "thread_count": count("thread_count"),
                "root_thread_count": count("root_thread_count", count("thread_count")),
                "missing_row_count": count("missing_thread_row_count"),
                "stale_thread_count": count("stale_thread_count"),
                "orphan_thread_count": count("orphan_thread_count"),
            },
        ),
        "tool_usage": surface(
            ready=not tool_usage_blockers,
            blockers=tool_usage_blockers,
            evidence={
                "action_count": count("action_count"),
                "tool_use_block_count": count("tool_use_block_count"),
                "actions_view_present": bool(counts.get("actions_view_present", False)),
                "actions_view_error": counts.get("actions_view_error"),
            },
        ),
        "latency_profiles": surface(
            ready=latency_ready,
            blockers=latency_blockers,
            evidence={"missing_row_count": count("missing_latency_profile_row_count")},
        ),
    }


def archive_readiness_status(root: Path) -> dict[str, Any]:
    """Return the exact-readiness surface report for one archive root.

    Serves the CLI's ``status`` reporting.
    """
    from polylogue.storage.archive_identity import ArchiveLocation, ArchiveLocationError

    try:
        location = ArchiveLocation.resolve(root)
    except ArchiveLocationError as exc:
        return {"checked": False, "reason": str(exc), "surfaces": {}}

    index_db = location.active_index.resolved_path
    source_db = location.configured_tier("source").resolved_path
    if not index_db.exists():
        return {"checked": False, "reason": "missing_index_tier", "surfaces": {}}
    try:
        conn = open_readonly_connection(index_db, timeout_class="background-read", validate_schema=False)
        source_conn: sqlite3.Connection | None = None
        try:
            source_check_available = source_db.exists()
            try:
                if source_check_available:
                    source_conn = open_readonly_connection(
                        source_db, timeout_class="background-read", validate_schema=False
                    )
                    source_check_available = _table_exists(source_conn, "raw_sessions")
                raw_projection: Mapping[str, object] | None = None
                if source_check_available:
                    attach_readonly_database(conn, source_db, alias="source_tier")
                    raw_projection = raw_materialization_readiness_from_pinned_index(
                        conn,
                        archive_root=location.configured_root,
                        source_schema="source_tier",
                        classify_gaps=False,
                    )
                return archive_readiness_status_from_connections(
                    conn,
                    source_conn if source_check_available else None,
                    raw_materialization_readiness=raw_projection,
                )
            finally:
                if source_conn is not None:
                    source_conn.close()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return {"checked": False, "reason": str(exc), "surfaces": {}}


def archive_readiness_status_from_connections(
    index_conn: sqlite3.Connection,
    source_conn: sqlite3.Connection | None,
    *,
    raw_materialization_readiness: Mapping[str, object] | None,
) -> dict[str, Any]:
    """Build archive readiness from an operation's pinned tier readers.

    The Index reader retains its source_tier attachment for the same pinned
    Source snapshot. Like the path twin, a degraded-but-readable tier resolves to ``{"checked": False, "reason": ...}`` rather than raising, so
    a status poll reports the degradation instead of failing.
    """

    try:
        return _archive_readiness_status_from_connections(
            index_conn, source_conn, raw_materialization_readiness=raw_materialization_readiness
        )
    except (OSError, sqlite3.Error) as exc:
        return {"checked": False, "reason": str(exc), "surfaces": {}}


def _archive_readiness_status_from_connections(
    index_conn: sqlite3.Connection,
    source_conn: sqlite3.Connection | None,
    *,
    raw_materialization_readiness: Mapping[str, object] | None,
) -> dict[str, Any]:
    """Build archive readiness from an operation's pinned tier readers.

    Unlike :func:`archive_readiness_status`, this function neither resolves a
    root nor opens a database.  The supplied materialization projection is
    intentionally an argument: its parser-census evidence must come from the
    same pinned index/source snapshot as the surface counts.
    """

    if not _table_exists(index_conn, "sessions"):
        return {"checked": False, "reason": "missing_sessions_table", "surfaces": {}}
    source_check_available = source_conn is not None and _table_exists(source_conn, "raw_sessions")
    counts = _archive_readiness_counts(
        index_conn,
        source_conn=source_conn,
        source_check_available=source_check_available,
    )
    if isinstance(raw_materialization_readiness, Mapping):
        counts.update(raw_materialization_readiness)

    surfaces = _archive_status_surfaces(counts, source_check_available=source_check_available)
    ready_count = sum(1 for info in surfaces.values() if info["ready"] is True)
    blocked_count = sum(1 for info in surfaces.values() if info["ready"] is not True)
    return {
        "checked": True,
        "reason": None,
        "source_check_available": source_check_available,
        "ready_surface_count": ready_count,
        "blocked_surface_count": blocked_count,
        "total_surface_count": len(surfaces),
        "counts": counts,
        "surfaces": surfaces,
    }


__all__ = [
    "RawMaterializationAssessment",
    "RawMaterializationAssessmentState",
    "assess_raw_materialization",
    "lost_durable_tiers",
    "archive_readiness_status",
    "archive_readiness_status_from_connections",
    "missing_source_raw_session_evidence",
    "raw_materialization_readiness_from_pinned_index",
    "raw_materialization_readiness_snapshot",
    "raw_materialization_ready",
]

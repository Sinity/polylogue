"""Durable immutable plans and conservation receipts for raw reconciliation.

The source tier is the authority for this ledger.  ``index.db`` may be rebuilt
and ``ops.db`` may be deleted; neither event is allowed to erase replay
fairness, stale-plan blockers, or the proof that a complete census was
accounted for.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from polylogue.core.json import JSONDocument, json_document
from polylogue.logging import get_logger

RAW_AUTHORITY_PARSER_FINGERPRINT = "revision-membership-v1"
logger = get_logger(__name__)


class RawReplayPlanStatus(StrEnum):
    EXECUTED = "executed"
    RETRYABLE = "retryable"
    DEFERRED = "deferred"
    TERMINAL = "terminal"
    REJECTED_STALE = "rejected_stale"
    CARRIED_FORWARD = "carried_forward"


@dataclass(frozen=True, slots=True)
class RawReplayPlan:
    plan_id: str
    input_digest: str
    input_raw_ids: tuple[str, ...]
    logical_keys: tuple[str, ...]
    authority_witness: JSONDocument
    source_preconditions: JSONDocument
    index_preconditions: JSONDocument

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "plan_id": self.plan_id,
                "input_digest": self.input_digest,
                "input_raw_ids": list(self.input_raw_ids),
                "logical_keys": list(self.logical_keys),
                "authority_witness": self.authority_witness,
                "source_preconditions": self.source_preconditions,
                "index_preconditions": self.index_preconditions,
            }
        )


@dataclass(frozen=True, slots=True)
class RawReplayPlanOutcome:
    plan_id: str
    input_raw_ids: tuple[str, ...]
    status: RawReplayPlanStatus
    reason: str
    next_action: str
    application_receipt: JSONDocument | None = None

    def to_dict(self) -> JSONDocument:
        payload: dict[str, object] = {
            "plan_id": self.plan_id,
            "input_raw_ids": list(self.input_raw_ids),
            "status": self.status.value,
            "reason": self.reason,
            "next_action": self.next_action,
        }
        if self.application_receipt is not None:
            payload["application_receipt"] = self.application_receipt
        return json_document(payload)


@dataclass(frozen=True, slots=True)
class RawAuthorityCensusReceipt:
    census_id: str
    sequence_no: int
    inventory_digest: str
    residual_digest: str
    plan_count: int
    executable_plan_count: int
    residual_plan_count: int
    predecessor_census_id: str | None
    fixed_point: bool

    @property
    def query_handle(self) -> str:
        return f"raw-authority-census:{self.census_id}"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_value(value: object) -> object:
    if isinstance(value, bytes):
        return value.hex()
    return value


def _rows(conn: sqlite3.Connection, sql: str, params: Sequence[object] = ()) -> list[dict[str, object]]:
    cursor = conn.execute(sql, tuple(params))
    names = tuple(column[0] for column in cursor.description or ())
    return [{name: _json_value(value) for name, value in zip(names, row, strict=True)} for row in cursor]


def build_raw_replay_plan(conn: sqlite3.Connection, input_raw_ids: Sequence[str]) -> RawReplayPlan:
    """Snapshot one complete component from an attached source/index pair."""
    raw_ids = tuple(sorted(dict.fromkeys(input_raw_ids)))
    if not raw_ids:
        raise ValueError("raw replay plan requires at least one input raw id")
    marks = ",".join("?" for _ in raw_ids)
    source_rows = _rows(
        conn,
        f"""
        SELECT raw_id, origin, native_id, source_path, source_index,
               hex(blob_hash) AS blob_hash, blob_size, logical_source_key,
               revision_kind, source_revision, predecessor_source_revision,
               predecessor_raw_id, baseline_raw_id, append_start_offset,
               append_end_offset, acquisition_generation, revision_authority
        FROM raw_sessions WHERE raw_id IN ({marks}) ORDER BY raw_id
        """,
        raw_ids,
    )
    if tuple(str(row["raw_id"]) for row in source_rows) != raw_ids:
        raise RuntimeError("raw replay plan input disappeared during census")
    membership_rows = _rows(
        conn,
        f"""
        SELECT raw_id, logical_source_key, provider_session_id, source_revision,
               hex(normalized_content_hash) AS normalized_content_hash,
               message_count, predecessor_raw_id, acquisition_generation,
               revision_authority, decision
        FROM raw_session_memberships
        WHERE raw_id IN ({marks})
        ORDER BY raw_id, logical_source_key
        """,
        raw_ids,
    )
    census_rows = _rows(
        conn,
        f"""
        SELECT raw_id, parser_fingerprint, status, member_count, detail
        FROM raw_membership_census
        WHERE raw_id IN ({marks}) ORDER BY raw_id
        """,
        raw_ids,
    )
    logical_keys = tuple(
        sorted(
            {
                str(value)
                for row in (*source_rows, *membership_rows)
                if (value := row.get("logical_source_key")) is not None
            }
        )
    )
    if logical_keys:
        key_marks = ",".join("?" for _ in logical_keys)
        head_rows = _rows(
            conn,
            f"""
            SELECT logical_source_key, session_id, accepted_raw_id,
                   accepted_source_revision, hex(accepted_content_hash) AS accepted_content_hash,
                   accepted_frontier_kind, accepted_frontier,
                   acquisition_generation, append_end_offset
            FROM index_tier.raw_revision_heads
            WHERE logical_source_key IN ({key_marks}) ORDER BY logical_source_key
            """,
            logical_keys,
        )
    else:
        head_rows = []
    session_rows = _rows(
        conn,
        f"""
        SELECT session_id, raw_id, hex(content_hash) AS content_hash, message_count
        FROM index_tier.sessions
        WHERE raw_id IN ({marks}) ORDER BY session_id
        """,
        raw_ids,
    )
    authority_witness = json_document(
        {
            "membership_census": census_rows,
            "memberships": membership_rows,
            "revision_heads": head_rows,
        }
    )
    source_preconditions = json_document({"raw_sessions": source_rows})
    index_preconditions = json_document({"sessions": session_rows, "revision_heads": head_rows})
    identity = {
        "schema": "polylogue.raw-replay-plan.v2",
        "input_raw_ids": list(raw_ids),
        "logical_keys": list(logical_keys),
        "authority_witness": authority_witness,
        "source_preconditions": source_preconditions,
        "index_preconditions": index_preconditions,
    }
    input_digest = _digest(identity)
    return RawReplayPlan(
        plan_id=f"raw-replay:{input_digest}",
        input_digest=input_digest,
        input_raw_ids=raw_ids,
        logical_keys=logical_keys,
        authority_witness=authority_witness,
        source_preconditions=source_preconditions,
        index_preconditions=index_preconditions,
    )


def build_raw_replay_plans(archive_root: Path, components: Sequence[tuple[str, ...]]) -> tuple[RawReplayPlan, ...]:
    if not components:
        return ()
    with closing(sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)) as conn:
        conn.execute("ATTACH DATABASE ? AS index_tier", (str(archive_root / "index.db"),))
        return tuple(build_raw_replay_plan(conn, component) for component in components)


def raw_replay_plan_last_attempts(archive_root: Path) -> dict[str, int]:
    """Return durable attempt order; deleting ops.db cannot reset fairness."""
    source_db = archive_root / "source.db"
    if not source_db.is_file():
        return {}
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_authority_census_plans'"
        ).fetchone()
        if exists is None:
            return {}
        return {
            str(row[0]): int(row[1])
            for row in conn.execute(
                """
                SELECT plan_id, MAX(recorded_at_ms)
                FROM raw_authority_census_plans
                WHERE selected = 1 GROUP BY plan_id
                """
            )
        }


def unresolved_raw_authority_blockers(archive_root: Path) -> int:
    source_db = archive_root / "source.db"
    if not source_db.is_file():
        return 0
    with closing(sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_authority_blockers'"
        ).fetchone()
        if exists is None:
            return 0
        return int(
            conn.execute("SELECT COUNT(*) FROM raw_authority_blockers WHERE resolved_at_ms IS NULL").fetchone()[0]
        )


def record_raw_authority_census(
    archive_root: Path,
    plans: Sequence[RawReplayPlan],
    *,
    selected_plan_ids: set[str],
    scope: Mapping[str, object],
    residual: Mapping[str, object],
) -> RawAuthorityCensusReceipt:
    """Atomically publish a complete plan census with carried-forward outcomes."""
    now = int(time.time() * 1000)
    inventory_digest = _digest([plan.plan_id for plan in plans])
    residual_digest = _digest(residual)
    scope_json = _canonical_json(scope)
    with closing(sqlite3.connect(archive_root / "source.db")) as conn, conn:
        previous = conn.execute(
            """
            SELECT census_id, sequence_no, inventory_digest, residual_digest,
                   executable_plan_count, scope_json
            FROM raw_authority_censuses ORDER BY sequence_no DESC LIMIT 1
            """
        ).fetchone()
        sequence_no = int(previous[1]) + 1 if previous is not None else 1
        predecessor = str(previous[0]) if previous is not None else None
        executable_count = len(selected_plan_ids)
        fixed_point = bool(
            previous is not None
            and int(previous[4]) == 0
            and executable_count == 0
            and str(previous[2]) == inventory_digest
            and str(previous[3]) == residual_digest
            and str(previous[5]) == scope_json
        )
        census_id = f"census:{sequence_no}:{inventory_digest[:16]}:{residual_digest[:16]}"
        for plan in plans:
            values = (
                plan.plan_id,
                plan.input_digest,
                _canonical_json(list(plan.input_raw_ids)),
                _canonical_json(list(plan.logical_keys)),
                _canonical_json(plan.authority_witness),
                _canonical_json(plan.source_preconditions),
                _canonical_json(plan.index_preconditions),
                now,
            )
            conn.execute(
                """
                INSERT INTO raw_authority_plans (
                    plan_id, input_digest, input_raw_ids_json, logical_keys_json,
                    authority_witness_json, source_preconditions_json,
                    index_preconditions_json, created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO NOTHING
                """,
                values,
            )
            stored = conn.execute(
                """
                SELECT input_digest, input_raw_ids_json, logical_keys_json,
                       authority_witness_json, source_preconditions_json,
                       index_preconditions_json
                FROM raw_authority_plans WHERE plan_id = ?
                """,
                (plan.plan_id,),
            ).fetchone()
            if stored != values[1:7]:
                raise RuntimeError(f"immutable raw replay plan collision: {plan.plan_id}")
        conn.execute(
            """
            INSERT INTO raw_authority_censuses (
                census_id, sequence_no, scope_json, parser_fingerprint,
                inventory_digest, residual_digest, plan_count,
                executable_plan_count, residual_plan_count,
                predecessor_census_id, fixed_point, created_at_ms, completed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                census_id,
                sequence_no,
                scope_json,
                RAW_AUTHORITY_PARSER_FINGERPRINT,
                inventory_digest,
                residual_digest,
                len(plans),
                executable_count,
                len(plans) - executable_count,
                predecessor,
                int(fixed_point),
                now,
                now,
            ),
        )
        for ordinal, plan in enumerate(plans):
            selected = plan.plan_id in selected_plan_ids
            conn.execute(
                """
                INSERT INTO raw_authority_census_plans (
                    census_id, plan_id, ordinal, selected, outcome_status,
                    reason, next_action, application_receipt_json, recorded_at_ms
                ) VALUES (?, ?, ?, ?, 'carried_forward', ?, ?, '{}', ?)
                """,
                (
                    census_id,
                    plan.plan_id,
                    ordinal,
                    int(selected),
                    "selected plan awaits a typed application outcome"
                    if selected
                    else "bounded scheduler carried this complete plan forward unchanged",
                    "execute this plan in the current pass" if selected else "retain for a later bounded pass",
                    now,
                ),
            )
    return RawAuthorityCensusReceipt(
        census_id=census_id,
        sequence_no=sequence_no,
        inventory_digest=inventory_digest,
        residual_digest=residual_digest,
        plan_count=len(plans),
        executable_plan_count=executable_count,
        residual_plan_count=len(plans) - executable_count,
        predecessor_census_id=predecessor,
        fixed_point=fixed_point,
    )


def validate_raw_replay_plan(archive_root: Path, plan: RawReplayPlan) -> tuple[bool, JSONDocument]:
    try:
        observed = build_raw_replay_plans(archive_root, (plan.input_raw_ids,))[0]
    except Exception as exc:
        logger.warning("raw replay plan validation could not rebuild %s", plan.plan_id, exc_info=True)
        return False, json_document({"error": f"{type(exc).__name__}: {exc}"})
    return observed == plan, observed.to_dict()


def raw_replay_application_receipt(archive_root: Path, plan: RawReplayPlan) -> JSONDocument:
    marks = ",".join("?" for _ in plan.input_raw_ids)
    with closing(sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)) as conn:
        conn.execute("ATTACH DATABASE ? AS index_tier", (str(archive_root / "index.db"),))
        source = _rows(
            conn,
            f"""
            SELECT raw_id, parsed_at_ms, parse_error
            FROM raw_sessions WHERE raw_id IN ({marks}) ORDER BY raw_id
            """,
            plan.input_raw_ids,
        )
        memberships = _rows(
            conn,
            f"""
            SELECT raw_id, logical_source_key, decision, decided_at_ms
            FROM raw_session_memberships
            WHERE raw_id IN ({marks}) ORDER BY raw_id, logical_source_key
            """,
            plan.input_raw_ids,
        )
        applications = _rows(
            conn,
            f"""
            SELECT decision_id, raw_id, session_id, logical_source_key, decision,
                   accepted_raw_id, hex(accepted_content_hash) AS accepted_content_hash,
                   decided_at_ms
            FROM index_tier.raw_revision_applications
            WHERE raw_id IN ({marks}) ORDER BY raw_id, decision_id
            """,
            plan.input_raw_ids,
        )
    return json_document(
        {
            "schema": "polylogue.raw-replay-application-receipt.v1",
            "source_rows": source,
            "membership_rows": memberships,
            "application_rows": applications,
        }
    )


def record_raw_replay_outcome(
    archive_root: Path,
    census_id: str,
    outcome: RawReplayPlanOutcome,
) -> None:
    now = int(time.time() * 1000)
    with closing(sqlite3.connect(archive_root / "source.db")) as conn, conn:
        updated = conn.execute(
            """
            UPDATE raw_authority_census_plans
            SET outcome_status = ?, reason = ?, next_action = ?,
                application_receipt_json = ?, recorded_at_ms = ?
            WHERE census_id = ? AND plan_id = ? AND selected = 1
            """,
            (
                outcome.status.value,
                outcome.reason,
                outcome.next_action,
                _canonical_json(outcome.application_receipt or {}),
                now,
                census_id,
                outcome.plan_id,
            ),
        ).rowcount
        if updated != 1:
            raise RuntimeError(f"outcome does not conserve one selected plan: {outcome.plan_id}")


def reject_stale_raw_replay_plan(
    archive_root: Path,
    census_id: str,
    plan: RawReplayPlan,
    observed: JSONDocument,
) -> RawReplayPlanOutcome:
    """Persist the fail-closed blocker before returning observational output."""
    now = int(time.time() * 1000)
    blocker_id = f"raw-authority-blocker:{_digest([plan.plan_id, observed])}"
    outcome = RawReplayPlanOutcome(
        plan.plan_id,
        plan.input_raw_ids,
        RawReplayPlanStatus.REJECTED_STALE,
        "immutable source/index preconditions changed after the census",
        "resolve the durable raw-authority blocker before automatic convergence resumes",
        json_document({"expected": plan.to_dict(), "observed": observed, "blocker_id": blocker_id}),
    )
    with closing(sqlite3.connect(archive_root / "source.db")) as conn, conn:
        conn.execute(
            """
            INSERT INTO raw_authority_blockers (
                blocker_id, plan_id, census_id, reason, expected_json,
                observed_json, created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(blocker_id) DO NOTHING
            """,
            (
                blocker_id,
                plan.plan_id,
                census_id,
                outcome.reason,
                _canonical_json(plan.to_dict()),
                _canonical_json(observed),
                now,
            ),
        )
        updated = conn.execute(
            """
            UPDATE raw_authority_census_plans
            SET outcome_status = 'rejected_stale', reason = ?, next_action = ?,
                application_receipt_json = ?, recorded_at_ms = ?
            WHERE census_id = ? AND plan_id = ? AND selected = 1
            """,
            (
                outcome.reason,
                outcome.next_action,
                _canonical_json(outcome.application_receipt or {}),
                now,
                census_id,
                plan.plan_id,
            ),
        ).rowcount
        if updated != 1:
            raise RuntimeError(f"stale rejection does not conserve one selected plan: {plan.plan_id}")
    return outcome


__all__ = [
    "RAW_AUTHORITY_PARSER_FINGERPRINT",
    "RawAuthorityCensusReceipt",
    "RawReplayPlan",
    "RawReplayPlanOutcome",
    "RawReplayPlanStatus",
    "build_raw_replay_plan",
    "build_raw_replay_plans",
    "raw_replay_application_receipt",
    "raw_replay_plan_last_attempts",
    "record_raw_authority_census",
    "record_raw_replay_outcome",
    "reject_stale_raw_replay_plan",
    "unresolved_raw_authority_blockers",
    "validate_raw_replay_plan",
]

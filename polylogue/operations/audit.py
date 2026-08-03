"""Durable audit.db repository for mutation authority and lifecycle evidence."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from polylogue.operations.mutation_transaction import (
    MutationAuthorization,
    MutationPlan,
    MutationPreview,
    MutationPrincipal,
    MutationReceipt,
)
from polylogue.storage.sqlite.archive_tiers.audit import AUDIT_DDL, AUDIT_SCHEMA_VERSION

AuditTargetState = Literal[
    "pending",
    "running",
    "applied",
    "already_satisfied",
    "rejected",
    "failed",
    "unknown",
    "acknowledged",
    "cancelled",
]


def token_sha256(token: str) -> str:
    """Return the only representation of a bearer token accepted for storage."""

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuditRepository:
    """Small synchronous repository whose methods make audit transactions explicit."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(AUDIT_DDL)
        conn.execute(f"PRAGMA user_version = {AUDIT_SCHEMA_VERSION}")
        conn.commit()
        try:
            yield conn
        finally:
            conn.close()

    def ensure_archive_authority(self, *, now_ms: int, archive_instance_id: str | None = None) -> str:
        """Create or return the immutable archive lineage id."""

        with self._connection() as conn:
            row = conn.execute("SELECT archive_instance_id FROM archive_authority LIMIT 1").fetchone()
            if row is not None:
                existing = str(row[0])
                if archive_instance_id is not None and archive_instance_id != existing:
                    raise ValueError("audit archive instance identity changed")
                return existing
            instance_id = archive_instance_id or f"archive:{secrets.token_hex(16)}"
            conn.execute(
                "INSERT INTO archive_authority(archive_instance_id, created_at_ms, authority_format) VALUES (?, ?, 1)",
                (instance_id, now_ms),
            )
            conn.commit()
            return instance_id

    def create_preview(self, plan: MutationPlan, principal: MutationPrincipal) -> str:
        """Persist a bounded preview and its normalized target/capability rows."""

        preview_id = f"preview:{secrets.token_urlsafe(18)}"
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO operation_previews (
                    preview_id, operation_name, operation_version, archive_instance_id,
                    archive_identity_digest, plan_hash, parameter_digest, target_digest,
                    target_count, destructive_class, required_confirmation,
                    required_capability_count, principal_actor_ref, principal_surface,
                    role_label, state, created_at_ms, expires_at_ms, plan_format, plan_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?,
                          'polylogue.mutation-plan/v1', ?)
                """,
                (
                    preview_id,
                    plan.operation,
                    plan.operation_version,
                    plan.archive_instance_id,
                    plan.archive_identity_digest,
                    plan.plan_hash,
                    plan.parameter_digest,
                    plan.target_digest or plan.plan_hash,
                    plan.target_count,
                    plan.destructive_class,
                    plan.required_confirmation,
                    len(plan.required_capabilities),
                    principal.actor_ref,
                    principal.surface,
                    principal.role_label,
                    plan.prepared_at_ms,
                    plan.expires_at_ms,
                    json.dumps(
                        {
                            "operation": plan.operation,
                            "operation_version": plan.operation_version,
                            "archive_instance_id": plan.archive_instance_id,
                            "archive_identity_digest": plan.archive_identity_digest,
                            "parameter_digest": plan.parameter_digest,
                            "target_digest": plan.target_digest,
                            "target_refs": list(plan.target_refs),
                            "affected_tiers": list(plan.affected_tiers),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )
            for ordinal, target in enumerate(plan.targets):
                conn.execute(
                    """
                    INSERT INTO operation_preview_targets(
                        preview_id, ordinal, target_kind, target_ref, identity_digest,
                        effect_identity, durability, recovery_policy
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        preview_id,
                        ordinal,
                        target.kind,
                        target.ref,
                        target.identity_digest,
                        target.effect_identity,
                        target.durability,
                        target.recovery,
                    ),
                )
            for capability in plan.required_capabilities:
                conn.execute(
                    "INSERT INTO operation_preview_capabilities(preview_id, capability) VALUES (?, ?)",
                    (preview_id, capability),
                )
            conn.commit()
        return preview_id

    def issue_authorization(
        self,
        preview: MutationPreview,
        principal: MutationPrincipal,
        authorization: MutationAuthorization,
    ) -> str:
        """Persist token digest and exact proved capabilities, never token material."""

        if authorization.token is None:
            raise ValueError("bound authorization requires a token")
        authorization_id = f"authorization:{secrets.token_urlsafe(18)}"
        issued_at_ms = authorization.expires_at_ms - 60_000 if authorization.expires_at_ms is not None else 0
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            preview_row = conn.execute(
                "SELECT plan_hash, expires_at_ms, state, principal_actor_ref FROM operation_previews WHERE preview_id = ?",
                (preview.preview_ref,),
            ).fetchone()
            if preview_row is None:
                raise ValueError(f"unknown preview {preview.preview_ref!r}")
            if str(preview_row[0]) != preview.plan.plan_hash:
                raise ValueError("preview plan hash does not match its durable row")
            if str(preview_row[2]) != "prepared":
                raise ValueError("preview is not authorizable")
            if principal.actor_ref != str(preview_row[3]):
                raise ValueError("authorization principal differs from preview principal")
            conn.execute(
                """
                INSERT INTO operation_authorizations(
                    authorization_id, preview_id, actor_ref, surface, role_label,
                    confirmation_strength, token_sha256, state, issued_at_ms,
                    expires_at_ms, consumed_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, NULL)
                """,
                (
                    authorization_id,
                    preview.preview_ref,
                    principal.actor_ref,
                    principal.surface,
                    principal.role_label,
                    authorization.confirmation_strength,
                    token_sha256(authorization.token),
                    issued_at_ms,
                    authorization.expires_at_ms or issued_at_ms,
                ),
            )
            for capability in authorization.capabilities:
                conn.execute(
                    "INSERT INTO operation_authorization_capabilities(authorization_id, capability) VALUES (?, ?)",
                    (authorization_id, capability),
                )
            conn.commit()
        return authorization_id

    def consume_authorization_and_start(self, preview: MutationPreview, authorization: MutationAuthorization) -> str:
        """Consume a token and create run, targets, and initial attempt atomically."""

        if authorization.token is None:
            raise ValueError("authorization token is missing")
        operation_id = f"operation:{secrets.token_urlsafe(18)}"
        attempt_id = f"attempt:{secrets.token_urlsafe(18)}"
        now_ms = int(time.time() * 1000)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT a.authorization_id, a.preview_id, a.actor_ref, a.surface,
                       a.state, a.expires_at_ms, p.plan_hash
                FROM operation_authorizations AS a
                JOIN operation_previews AS p ON p.preview_id = a.preview_id
                WHERE a.token_sha256 = ?
                """,
                (token_sha256(authorization.token),),
            ).fetchone()
            if row is None or str(row[1]) != preview.preview_ref:
                raise ValueError("authorization token does not match preview")
            if str(row[4]) != "active":
                raise RuntimeError("authorization token is already consumed or revoked")
            if int(row[5]) <= now_ms:
                conn.execute(
                    "UPDATE operation_authorizations SET state = 'expired' WHERE authorization_id = ?",
                    (str(row[0]),),
                )
                conn.commit()
                raise RuntimeError("authorization token is expired")
            if str(row[2]) != authorization.actor or str(row[3]) != (authorization.surface or ""):
                raise ValueError("authorization principal mismatch")
            if str(row[6]) != preview.plan.plan_hash or authorization.plan_hash != preview.plan.plan_hash:
                raise ValueError("authorization plan mismatch")
            conn.execute(
                "UPDATE operation_authorizations SET state = 'consumed', consumed_at_ms = ? WHERE authorization_id = ?",
                (now_ms, str(row[0])),
            )
            conn.execute(
                "UPDATE operation_previews SET state = 'consumed' WHERE preview_id = ?",
                (preview.preview_ref,),
            )
            conn.execute(
                """
                INSERT INTO operation_runs(
                    operation_id, preview_id, initial_authorization_id,
                    operation_name, operation_version, archive_instance_id,
                    archive_identity_digest, plan_hash, parameter_digest,
                    target_digest, target_count, actor_ref, surface, role_label,
                    status, requested_at_ms, started_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (
                    operation_id,
                    preview.preview_ref,
                    str(row[0]),
                    preview.plan.operation,
                    preview.plan.operation_version,
                    preview.plan.archive_instance_id,
                    preview.plan.archive_identity_digest,
                    preview.plan.plan_hash,
                    preview.plan.parameter_digest,
                    preview.plan.target_digest or preview.plan.plan_hash,
                    preview.plan.target_count,
                    authorization.actor,
                    authorization.surface,
                    authorization.role,
                    now_ms,
                    now_ms,
                    now_ms,
                ),
            )
            for capability in authorization.capabilities:
                conn.execute(
                    "INSERT INTO operation_run_capabilities(operation_id, capability) VALUES (?, ?)",
                    (operation_id, capability),
                )
            for ordinal, target in enumerate(preview.plan.targets):
                conn.execute(
                    """
                    INSERT INTO operation_targets(
                        operation_id, ordinal, target_kind, target_ref, identity_digest,
                        effect_identity, state, attempt_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        operation_id,
                        ordinal,
                        target.kind,
                        target.ref,
                        target.identity_digest,
                        target.effect_identity,
                        "running" if ordinal == 0 else "pending",
                    ),
                )
            conn.execute(
                """
                INSERT INTO operation_attempts(
                    attempt_id, operation_id, target_ordinal, authorization_id,
                    state, started_at_ms
                ) VALUES (?, ?, ?, ?, 'running', ?)
                """,
                (attempt_id, operation_id, 0 if preview.plan.targets else None, str(row[0]), now_ms),
            )
            self._append_event(
                conn,
                operation_id=operation_id,
                event_type="authorization_consumed",
                to_state="running",
                actor_ref=authorization.actor,
                occurred_at_ms=now_ms,
                detail={"target_count": preview.plan.target_count},
            )
            conn.commit()
        return operation_id

    def finalize_attempt(
        self,
        operation_id: str,
        *,
        status: str,
        receipt: MutationReceipt | None = None,
        error_summary: str | None = None,
        unknown_reason: str | None = None,
    ) -> None:
        """Finalize one running attempt and parent run in one audit transaction."""

        now_ms = int(time.time() * 1000)
        target_state: AuditTargetState = (
            "unknown" if status == "unknown" else "failed" if status == "failed" else "applied"
        )
        attempt_state = "unknown" if target_state == "unknown" else target_state
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            run = conn.execute(
                "SELECT actor_ref FROM operation_runs WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if run is None:
                raise ValueError(f"unknown operation {operation_id!r}")
            target = conn.execute(
                "SELECT ordinal FROM operation_targets WHERE operation_id = ? AND state = 'running' ORDER BY ordinal LIMIT 1",
                (operation_id,),
            ).fetchone()
            ordinal = int(target[0]) if target is not None else None
            conn.execute(
                """
                UPDATE operation_attempts
                SET state = ?, finished_at_ms = ?, error_summary = ?, unknown_reason = ?
                WHERE operation_id = ? AND state = 'running'
                """,
                (attempt_state, now_ms, error_summary, unknown_reason, operation_id),
            )
            if ordinal is not None:
                conn.execute(
                    """
                    UPDATE operation_targets
                    SET state = ?, completed_at_ms = ?, error_summary = ?, unknown_reason = ?,
                        domain_receipt_ref = ?, domain_receipt_kind = ?
                    WHERE operation_id = ? AND ordinal = ?
                    """,
                    (
                        target_state,
                        now_ms,
                        error_summary,
                        unknown_reason,
                        None if receipt is None else receipt.receipt_ref,
                        None if receipt is None else "mutation-receipt",
                        operation_id,
                        ordinal,
                    ),
                )
            states = [
                str(row[0])
                for row in conn.execute("SELECT state FROM operation_targets WHERE operation_id = ?", (operation_id,))
            ]
            if "unknown" in states:
                run_status, terminal_reason = "interrupted", "unknown_effect"
            elif "failed" in states:
                run_status, terminal_reason = "failed", "domain_failure"
            elif states and all(state in {"applied", "already_satisfied"} for state in states):
                run_status, terminal_reason = "completed", None
            else:
                run_status, terminal_reason = "running", None
            conn.execute(
                """
                UPDATE operation_runs
                SET status = ?, terminal_reason = ?, updated_at_ms = ?,
                    completed_at_ms = CASE WHEN ? IN ('completed', 'failed', 'interrupted') THEN ? ELSE completed_at_ms END,
                    failed_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state = 'failed'),
                    unknown_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state = 'unknown'),
                    affected_count = (SELECT COUNT(*) FROM operation_targets WHERE operation_id = ? AND state IN ('applied', 'already_satisfied')),
                    error_summary = ?, unknown_reason = ?
                WHERE operation_id = ?
                """,
                (
                    run_status,
                    terminal_reason,
                    now_ms,
                    run_status,
                    now_ms,
                    operation_id,
                    operation_id,
                    operation_id,
                    error_summary,
                    unknown_reason,
                    operation_id,
                ),
            )
            self._append_event(
                conn,
                operation_id=operation_id,
                target_ordinal=ordinal,
                event_type="attempt_finalized" if run_status != "interrupted" else "attempt_unknown",
                from_state="running",
                to_state=run_status,
                actor_ref=str(run[0]),
                occurred_at_ms=now_ms,
                detail={"status": status, "reason": (unknown_reason or error_summary or "")[:512]},
            )
            conn.commit()

    def reconcile_attempt(
        self,
        operation_id: str,
        *,
        outcome: Literal["applied", "absent", "unknown"],
        domain_receipt_ref: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Persist an explicit applied/absent/unknown reconciliation decision."""

        now_ms = int(time.time() * 1000)
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            target = conn.execute(
                "SELECT ordinal FROM operation_targets WHERE operation_id = ? AND state = 'unknown' ORDER BY ordinal LIMIT 1",
                (operation_id,),
            ).fetchone()
            ordinal = int(target[0]) if target is not None else None
            if ordinal is None:
                raise ValueError(f"operation {operation_id!r} has no unknown target to reconcile")
            target_state = "applied" if outcome == "applied" else "pending" if outcome == "absent" else "unknown"
            run_state = (
                "completed" if target_state == "applied" else "running" if target_state == "pending" else "interrupted"
            )
            conn.execute(
                "UPDATE operation_attempts SET state = 'reconciled', finished_at_ms = ?, unknown_reason = ? WHERE operation_id = ? AND state = 'unknown'",
                (now_ms, reason, operation_id),
            )
            conn.execute(
                "UPDATE operation_targets SET state = ?, domain_receipt_ref = ?, domain_receipt_kind = ?, completed_at_ms = ? WHERE operation_id = ? AND ordinal = ?",
                (
                    target_state,
                    domain_receipt_ref,
                    "domain" if domain_receipt_ref else None,
                    now_ms if target_state == "applied" else None,
                    operation_id,
                    ordinal,
                ),
            )
            conn.execute(
                "UPDATE operation_runs SET status = ?, terminal_reason = ?, updated_at_ms = ?, completed_at_ms = CASE WHEN ? = 'completed' THEN ? ELSE completed_at_ms END WHERE operation_id = ?",
                (
                    run_state,
                    None if run_state == "completed" else "reconciliation_unknown",
                    now_ms,
                    run_state,
                    now_ms,
                    operation_id,
                ),
            )
            self._append_event(
                conn,
                operation_id=operation_id,
                target_ordinal=ordinal,
                event_type=f"reconciliation_{outcome}",
                from_state="unknown",
                to_state=target_state,
                occurred_at_ms=now_ms,
                detail={"reason": (reason or "")[:512]},
            )
            conn.commit()

    def get_operation(self, operation_id: str) -> dict[str, object] | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM operation_runs WHERE operation_id = ?", (operation_id,)).fetchone()
            return dict(row) if row is not None else None

    def list_events(self, operation_id: str) -> tuple[dict[str, object], ...]:
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM operation_events WHERE operation_id = ? ORDER BY sequence", (operation_id,)
            ).fetchall()
            return tuple(dict(row) for row in rows)

    @staticmethod
    def _append_event(
        conn: sqlite3.Connection,
        *,
        operation_id: str,
        event_type: str,
        occurred_at_ms: int,
        detail: Mapping[str, object],
        target_ordinal: int | None = None,
        attempt_id: str | None = None,
        from_state: str | None = None,
        to_state: str | None = None,
        actor_ref: str | None = None,
    ) -> None:
        sequence = int(
            conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM operation_events WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()[0]
        )
        conn.execute(
            """
            INSERT INTO operation_events(
                operation_id, sequence, target_ordinal, attempt_id, event_type,
                from_state, to_state, actor_ref, occurred_at_ms, detail_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                operation_id,
                sequence,
                target_ordinal,
                attempt_id,
                event_type,
                from_state,
                to_state,
                actor_ref,
                occurred_at_ms,
                json.dumps(dict(detail), sort_keys=True, separators=(",", ":")),
            ),
        )


__all__ = ["AuditRepository", "AuditTargetState", "token_sha256"]

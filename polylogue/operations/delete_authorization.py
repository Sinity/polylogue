"""Daemon-owned, preview-bound authorization for CLI session deletion."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from polylogue.operations.audit import AuditRepository, token_sha256
from polylogue.operations.bindings import OperationBinding
from polylogue.operations.mutation_actuators import SessionDeleteActuator, SessionDeleteArgs
from polylogue.operations.mutation_transaction import (
    AuthorizationMismatchError,
    ConfirmationStrength,
    MutationAuthorization,
    MutationPlan,
    MutationPreview,
    MutationPrincipal,
    MutationReceipt,
    MutationTarget,
    OperationExecutor,
    PlanStaleError,
    TokenConsumedError,
    TokenExpiredError,
)
from polylogue.operations.specs import build_runtime_operation_catalog
from polylogue.storage.archive_identity import ArchiveIdentity
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

_DELETE_CAPABILITY = "archive.delete_session"
# A preview persists one durable target row and one exact effect identity per
# session, then revalidates every target at consumption. Ten 1,000-target
# audit pages keeps that single-writer transaction bounded without regressing
# the established multi-hundred-session CLI delete workflow. Larger selections
# must be split into independent preview/authorize/delete operations.
DELETE_PREVIEW_MAX_SESSION_IDS = 10_000
_DELETE_PREVIEW_RESOLUTION_PAGE_SIZE = 256


class DeleteAuthorizationError(ValueError):
    """A daemon-held delete authorization cannot be prepared or consumed."""


@dataclass(frozen=True, slots=True)
class DeletePreviewPayload:
    preview_ref: str
    session_ids: tuple[str, ...]
    expires_at_ms: int

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "prepared",
            "operation": "delete",
            "preview_ref": self.preview_ref,
            "session_ids": list(self.session_ids),
            "session_count": len(self.session_ids),
            "expires_at_ms": self.expires_at_ms,
        }


def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _parameter_digest(session_ids: tuple[str, ...]) -> str:
    payload = json.dumps(list(session_ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _binding() -> OperationBinding[SessionDeleteArgs, object]:
    spec = build_runtime_operation_catalog().by_name()["mutate-delete-session"]
    return OperationBinding(
        spec=spec,
        actuator=SessionDeleteActuator(),
        declared_capabilities=(_DELETE_CAPABILITY,),
    )


def _canonical_session_ids(archive: ArchiveStore, requested: tuple[str, ...]) -> tuple[str, ...]:
    if len(requested) > DELETE_PREVIEW_MAX_SESSION_IDS:
        raise DeleteAuthorizationError("selection_exceeds_preview_work_budget")
    if len(set(requested)) != len(requested):
        raise DeleteAuthorizationError("selection_is_not_canonical")

    exact_matches = archive.resolve_exact_session_ids(
        requested,
        page_size=_DELETE_PREVIEW_RESOLUTION_PAGE_SIZE,
    )
    canonical: list[str] = []
    canonical_set: set[str] = set()
    for session_id in requested:
        resolved = exact_matches.get(session_id)
        if resolved is None:
            raise DeleteAuthorizationError("selection_is_stale")
        if resolved in canonical_set:
            raise DeleteAuthorizationError("selection_is_not_canonical")
        canonical_set.add(resolved)
        canonical.append(resolved)
    return tuple(canonical)


def _audit_path(archive_root: Path) -> Path:
    return archive_root / "audit.db"


def _audit_repository(archive_root: Path) -> AuditRepository:
    """Create audit authority with the local actuator identity for recovery."""

    return AuditRepository(
        _audit_path(archive_root),
        attempt_owner_id=AuditRepository.current_process_attempt_owner(),
    )


def prepare_cli_delete(
    archive_root: Path,
    requested_session_ids: tuple[str, ...],
    principal: MutationPrincipal,
) -> DeletePreviewPayload:
    """Persist the authenticated caller's exact canonical delete preview."""

    if not requested_session_ids:
        raise DeleteAuthorizationError("selection_is_empty")
    audit = _audit_repository(archive_root)
    executor = OperationExecutor(audit=audit)
    binding = _binding()
    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        session_ids = _canonical_session_ids(archive, requested_session_ids)
        args = SessionDeleteArgs(archive=archive, session_ids=session_ids)
        preview = executor.prepare_bound(
            binding,
            args,
            principal,
            archive_instance_id=audit.ensure_archive_authority(now_ms=_now_ms()),
            archive_identity_digest=ArchiveIdentity.resolve(archive_root).authority_identity_digest,
            parameter_digest=_parameter_digest(session_ids),
        )
    return DeletePreviewPayload(
        preview_ref=preview.preview_ref,
        session_ids=session_ids,
        expires_at_ms=preview.plan.expires_at_ms,
    )


def authorize_cli_delete(
    archive_root: Path,
    preview_ref: str,
    principal: MutationPrincipal,
) -> str:
    """Issue a daemon-held, single-use authorization for one prepared preview."""

    audit = _audit_repository(archive_root)
    preview = _load_preview(audit, preview_ref, principal, require_prepared=True)
    authorization = OperationExecutor(audit=audit).authorize_bound(
        _binding(),
        preview,
        principal,
        confirmation_strength="bound_token",
    )
    if authorization.token is None:
        raise DeleteAuthorizationError("authorization_not_issued")
    return authorization.token


def consume_cli_delete(
    archive_root: Path,
    token: str,
    principal: MutationPrincipal,
) -> MutationReceipt:
    """Atomically consume a daemon-issued delete authorization before mutation."""

    audit = _audit_repository(archive_root)
    preview, authorization = _load_active_authorization(audit, token, principal)
    if audit.ensure_archive_authority(now_ms=_now_ms()) != preview.plan.archive_instance_id:
        raise DeleteAuthorizationError("archive_instance_changed")
    if ArchiveIdentity.resolve(archive_root).authority_identity_digest != preview.plan.archive_identity_digest:
        raise DeleteAuthorizationError("archive_identity_changed")
    session_ids = _session_ids_from_preview(preview)
    try:
        with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
            return OperationExecutor(audit=audit).execute_bound(
                _binding(),
                preview,
                authorization,
                SessionDeleteArgs(archive=archive, session_ids=session_ids),
            )
    except PlanStaleError as exc:
        raise DeleteAuthorizationError("selection_changed_after_authorization") from exc
    except (AuthorizationMismatchError, TokenConsumedError, TokenExpiredError) as exc:
        raise DeleteAuthorizationError("authorization_not_active") from exc


def cancel_cli_delete(
    archive_root: Path,
    preview_ref: str,
    principal: MutationPrincipal,
) -> None:
    """Cancel an authenticated caller's unconfirmed durable delete preview."""

    audit = _audit_repository(archive_root)
    # An explicit decline is itself a terminal decision.  It must remain
    # recordable after the preview's authorization window closes; otherwise
    # the durable row is stranded in ``prepared`` even though the daemon has
    # acknowledged the operator's refusal to mutate.
    preview = _load_preview(audit, preview_ref, principal, require_prepared=True, require_unexpired=False)
    audit.cancel_preview(preview)


def _load_preview(
    audit: AuditRepository,
    preview_ref: str,
    principal: MutationPrincipal,
    *,
    require_prepared: bool,
    require_unexpired: bool = True,
) -> MutationPreview:
    with sqlite3.connect(audit.path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT * FROM operation_previews
            WHERE preview_id = ? AND principal_actor_ref = ? AND principal_surface = ?
            """,
            (preview_ref, principal.actor_ref, principal.surface),
        ).fetchone()
        if row is None:
            raise DeleteAuthorizationError("preview_not_owned")
        if require_prepared and str(row["state"]) != "prepared":
            raise DeleteAuthorizationError("preview_not_active")
        if require_unexpired and int(row["expires_at_ms"]) <= _now_ms():
            raise DeleteAuthorizationError("preview_expired")
        targets = _load_targets(conn, preview_ref)
        capabilities = _load_capabilities(conn, "operation_preview_capabilities", "preview_id", preview_ref)
    return MutationPreview(preview_ref=preview_ref, plan=_plan_from_row(row, targets, capabilities))


def _load_active_authorization(
    audit: AuditRepository,
    token: str,
    principal: MutationPrincipal,
) -> tuple[MutationPreview, MutationAuthorization]:
    with sqlite3.connect(audit.path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT a.*, p.*
            FROM operation_authorizations AS a
            JOIN operation_previews AS p ON p.preview_id = a.preview_id
            WHERE a.token_sha256 = ? AND a.actor_ref = ? AND a.surface = ?
            """,
            (token_sha256(token), principal.actor_ref, principal.surface),
        ).fetchone()
        if row is None:
            raise DeleteAuthorizationError("authorization_not_owned")
        if str(row["state"]) != "active":
            raise DeleteAuthorizationError("authorization_not_active")
        if int(row["expires_at_ms"]) <= _now_ms():
            raise DeleteAuthorizationError("authorization_expired")
        preview_ref = str(row["preview_id"])
        targets = _load_targets(conn, preview_ref)
        preview_capabilities = _load_capabilities(conn, "operation_preview_capabilities", "preview_id", preview_ref)
        capabilities = _load_capabilities(
            conn,
            "operation_authorization_capabilities",
            "authorization_id",
            str(row["authorization_id"]),
        )
    preview = MutationPreview(preview_ref=preview_ref, plan=_plan_from_row(row, targets, preview_capabilities))
    authorization = MutationAuthorization(
        plan_hash=preview.plan.plan_hash,
        actor=principal.actor_ref,
        role=str(row["role_label"] or ""),
        capability=capabilities[0] if capabilities else "",
        confirmation_strength=cast(ConfirmationStrength, str(row["confirmation_strength"])),
        authorized_at=datetime.fromtimestamp(int(row["issued_at_ms"]) / 1000, UTC).isoformat(),
        preview_ref=preview_ref,
        authorization_id=str(row["authorization_id"]),
        token=token,
        expires_at_ms=int(row["expires_at_ms"]),
        capabilities=capabilities,
        surface=principal.surface,
    )
    return preview, authorization


def _load_targets(conn: sqlite3.Connection, preview_ref: str) -> tuple[MutationTarget, ...]:
    rows = conn.execute(
        """
        SELECT target_kind, target_ref, identity_digest, effect_identity, durability, recovery_policy
        FROM operation_preview_targets WHERE preview_id = ? ORDER BY ordinal
        """,
        (preview_ref,),
    ).fetchall()
    return tuple(
        MutationTarget(
            kind=str(row["target_kind"]),
            ref=str(row["target_ref"]),
            policy_key="session-delete",
            identity_digest=str(row["identity_digest"]),
            effect_identity=str(row["effect_identity"]),
            durability=cast(Literal["durable", "derived", "disposable", "external"], str(row["durability"])),
            recovery=cast(
                Literal[
                    "rebuild",
                    "restore_verified_backup",
                    "reauthenticate",
                    "retry_convergent",
                    "reconcile_required",
                    "none",
                ],
                str(row["recovery_policy"]),
            ),
        )
        for row in rows
    )


def _load_capabilities(conn: sqlite3.Connection, table: str, key: str, value: str) -> tuple[str, ...]:
    if table not in {"operation_preview_capabilities", "operation_authorization_capabilities"}:
        raise AssertionError(table)
    rows = conn.execute(f"SELECT capability FROM {table} WHERE {key} = ? ORDER BY capability", (value,)).fetchall()
    return tuple(str(row["capability"]) for row in rows)


def _plan_from_row(
    row: sqlite3.Row,
    targets: tuple[MutationTarget, ...],
    capabilities: tuple[str, ...],
) -> MutationPlan:
    try:
        document = json.loads(str(row["plan_json"]))
    except (json.JSONDecodeError, TypeError) as exc:
        raise DeleteAuthorizationError("preview_plan_invalid") from exc
    if not isinstance(document, dict):
        raise DeleteAuthorizationError("preview_plan_invalid")
    affected_tiers_value = document.get("affected_tiers")
    context_value = document.get("context", {})
    if (
        not isinstance(affected_tiers_value, list)
        or not all(isinstance(tier, str) for tier in affected_tiers_value)
        or not isinstance(context_value, dict)
        or not all(isinstance(key, str) for key in context_value)
    ):
        raise DeleteAuthorizationError("preview_plan_invalid")
    prepared_at_ms = int(row["created_at_ms"])
    return MutationPlan(
        operation=str(row["operation_name"]),
        destructive_class=cast(
            Literal["additive", "reversible", "maintenance", "reset", "delete", "excise"], row["destructive_class"]
        ),
        target_refs=tuple(target.ref for target in targets),
        affected_tiers=tuple(affected_tiers_value),
        reversible=False,
        prepared_at=datetime.fromtimestamp(prepared_at_ms / 1000, UTC).isoformat(),
        plan_hash=str(row["plan_hash"]),
        context=context_value,
        operation_version=int(row["operation_version"]),
        archive_instance_id=str(row["archive_instance_id"]),
        archive_identity_digest=str(row["archive_identity_digest"]),
        required_capabilities=capabilities,
        required_confirmation=cast(ConfirmationStrength, str(row["required_confirmation"])),
        targets=targets,
        parameter_digest=str(row["parameter_digest"]),
        target_digest=str(row["target_digest"]),
        prepared_at_ms=prepared_at_ms,
        expires_at_ms=int(row["expires_at_ms"]),
    )


def _session_ids_from_preview(preview: MutationPreview) -> tuple[str, ...]:
    session_ids: list[str] = []
    for target in preview.plan.targets:
        if target.kind != "session" or not target.ref.startswith("session:"):
            raise DeleteAuthorizationError("preview_targets_invalid")
        session_ids.append(target.ref.removeprefix("session:"))
    if not session_ids:
        raise DeleteAuthorizationError("selection_is_empty")
    return tuple(session_ids)


__all__ = [
    "DeleteAuthorizationError",
    "DeletePreviewPayload",
    "authorize_cli_delete",
    "consume_cli_delete",
    "prepare_cli_delete",
]

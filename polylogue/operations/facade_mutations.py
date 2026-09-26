"""Mutation products called by the embedded archive facade.

The facade resolves read-backed targets and adapts typed results. This module
owns actuator selection and the bound prepare/authorize/execute route.
"""

from __future__ import annotations

from typing import Any, cast

from polylogue.config import Config
from polylogue.operations import mutation_actuators as actuators
from polylogue.operations.archive_mutation import execute_archive_mutation
from polylogue.operations.mutation_transaction import MutationPlan, MutationReceipt


class FacadeProductRefusalError(ValueError):
    """A product precondition failed before a known durable effect."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        super().__init__(detail)


_PRODUCTS: dict[str, tuple[type[Any], type[Any]]] = {
    "add_tag": (actuators.TagAddActuator, actuators.TagAddArgs),
    "remove_tag": (actuators.TagRemoveActuator, actuators.TagRemoveArgs),
    "set_metadata": (actuators.MetadataSetActuator, actuators.MetadataSetArgs),
    "delete_metadata": (actuators.MetadataDeleteActuator, actuators.MetadataDeleteArgs),
    "bulk_tag_sessions": (actuators.BulkTagActuator, actuators.BulkTagArgs),
    "add_mark": (actuators.MarkAddActuator, actuators.MarkArgs),
    "remove_mark": (actuators.MarkRemoveActuator, actuators.MarkArgs),
    "save_annotation": (actuators.AnnotationSaveActuator, actuators.AnnotationSaveArgs),
    "delete_annotation": (actuators.AnnotationDeleteActuator, actuators.AnnotationDeleteArgs),
    "save_view": (actuators.SavedViewSaveActuator, actuators.SavedViewSaveArgs),
    "delete_view": (actuators.SavedViewDeleteActuator, actuators.SavedViewDeleteArgs),
    "create_recall_pack": (actuators.RecallPackSaveActuator, actuators.RecallPackSaveArgs),
    "delete_recall_pack": (actuators.RecallPackDeleteActuator, actuators.RecallPackDeleteArgs),
    "save_workspace": (actuators.WorkspaceSaveActuator, actuators.WorkspaceSaveArgs),
    "delete_workspace": (actuators.WorkspaceDeleteActuator, actuators.WorkspaceDeleteArgs),
    "record_correction": (actuators.CorrectionRecordActuator, actuators.CorrectionRecordArgs),
    "delete_correction": (actuators.CorrectionDeleteActuator, actuators.CorrectionDeleteArgs),
    "clear_corrections": (actuators.CorrectionsClearActuator, actuators.CorrectionsClearArgs),
    "capture_assertion_candidate": (
        actuators.CaptureAssertionCandidateActuator,
        actuators.CaptureAssertionCandidateArgs,
    ),
    "post_blackboard_note": (actuators.BlackboardPostActuator, actuators.BlackboardPostArgs),
}


def _normalize_product_fields(product: str, fields: dict[str, Any]) -> dict[str, Any]:
    if product == "capture_assertion_candidate":
        import hashlib
        import uuid

        from polylogue.core.refs import normalize_object_ref_text

        key = fields.get("idempotency_key")
        if key is None:
            fields["assertion_id"] = f"assertion-terminal-note:{uuid.uuid4()}"
        else:
            identity = hashlib.sha256(
                f"{normalize_object_ref_text(fields['author_ref'])}\0{key.strip()}".encode(
                    "utf-8", errors="surrogatepass"
                )
            ).hexdigest()
            fields["assertion_id"] = f"assertion-terminal-note:{identity}"
    elif product == "post_blackboard_note":
        import uuid

        from polylogue.archive.blackboard import BLACKBOARD_KINDS, build_blackboard_body

        kind = fields.pop("kind")
        if kind not in BLACKBOARD_KINDS:
            raise ValueError(f"kind must be one of {list(BLACKBOARD_KINDS)}, got {kind!r}")
        scope_session = fields.pop("scope_session")
        fields["related_sessions"] = tuple(fields["related_sessions"])
        fields["evidence_refs"] = tuple(fields["evidence_refs"])
        fields["body"] = build_blackboard_body(
            kind=kind,
            title=fields.pop("title"),
            content=fields.pop("content"),
            scope_repo=fields.pop("scope_repo"),
            scope_issue=fields.pop("scope_issue"),
            scope_path=fields.pop("scope_path"),
            related_sessions=fields.pop("related_sessions"),
        )
        fields["note_id"] = str(uuid.uuid4())
        fields["target_type"] = "session" if scope_session else None
        fields["target_id"] = scope_session
    elif product == "bulk_tag_sessions":
        session_ids = fields["session_ids"]
        tags = fields["tags"]
        if not session_ids:
            raise ValueError("bulk_tag_sessions requires at least one session_id")
        if not tags:
            raise ValueError("bulk_tag_sessions requires at least one tag")
        if len(session_ids) > 100:
            raise ValueError("bulk_tag_sessions supports at most 100 session_ids")
        if len(tags) > 20:
            raise ValueError("bulk_tag_sessions supports at most 20 tags")
        fields["session_ids"] = tuple(session_ids)
        fields["tags"] = tuple(tags)
    elif product in {"record_correction", "delete_correction"}:
        from polylogue.analysis.feedback import parse_correction_kind

        parse_correction_kind(fields["kind"])
    return fields


def execute_facade_product(
    config: Config,
    product: str,
    **fields: Any,
) -> tuple[MutationReceipt, MutationPlan]:
    """Execute one named product with its operation-owned actuator contract."""
    fields = _normalize_product_fields(product, fields)
    actuator_type, args_type = _PRODUCTS[product]
    actuator = actuator_type()
    session_id = fields.get("session_id")
    return execute_archive_mutation(
        config,
        actuator,
        lambda archive: args_type(archive=archive, **fields),
        capability=f"archive.{product}",
        session_id=session_id if isinstance(session_id, str) else None,
    )


def delete_session_product(config: Config, session_id: str, *, actor: str) -> tuple[str, MutationReceipt] | None:
    """Delete a resolved session with a bound-token authorization."""
    from polylogue.config import active_archive_root
    from polylogue.operations.archive_mutation import MutationBlockedError, require_archive_write_authority
    from polylogue.operations.bindings import runtime_operation_binding
    from polylogue.operations.mutation_transaction import MutationPrincipal, OperationExecutor
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    require_archive_write_authority(config, "api.delete_session")
    root = active_archive_root(config)
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        try:
            resolved = archive.resolve_session_id(session_id)
        except KeyError:
            return None
        actuator = actuators.SessionDeleteActuator()
        executor = OperationExecutor.for_archive_root(root)
        args = actuators.SessionDeleteArgs(archive=archive, session_ids=(resolved,))
        binding = runtime_operation_binding(actuator)
        principal = MutationPrincipal(actor, frozenset({"archive.delete_session"}), "api", "write")
        preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=root)
        authorization = executor.authorize_bound(binding, preview, principal, confirmation_strength="bound_token")
        receipt = executor.execute_bound(binding, preview, authorization, args)
    if receipt.status == "blocked":
        raise MutationBlockedError(receipt.operation, receipt.detail, receipt.target_refs)
    return resolved, receipt


def record_work_event_product(
    config: Config,
    session_id: str,
    *,
    event_id: str,
    event_type: str,
    summary: str,
    payload: dict[str, object] | None,
    timestamp: str | None,
) -> dict[str, object]:
    """Append a typed work event under archive write authority."""
    from polylogue.config import active_archive_root
    from polylogue.operations.archive_mutation import require_archive_write_authority
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    require_archive_write_authority(config, "api.record_work_event")
    with ArchiveStore.open_existing(active_archive_root(config), read_only=False) as archive:
        return archive.append_work_event(
            session_id=session_id,
            event_type=event_type,
            payload=dict(payload or {}),
            event_id=event_id,
            summary=summary,
            timestamp=timestamp,
        )


async def import_annotation_batch_product(
    config: Config,
    request: Any,
    resolve_ref: Any,
    *,
    registry: Any = None,
) -> Any:
    """Apply one annotation import with a bounded facade ref resolver."""
    from polylogue.annotations.importer import import_annotation_batch
    from polylogue.config import active_archive_root
    from polylogue.operations.archive_mutation import require_archive_write_authority

    require_archive_write_authority(config, "api.import_annotation_batch")

    class _ImportHandle:
        archive_root = active_archive_root(config)

        async def resolve_ref(self, ref: str) -> Any:
            return await resolve_ref(ref)

    if registry is None:
        return await import_annotation_batch(cast(Any, _ImportHandle()), request)
    return await import_annotation_batch(cast(Any, _ImportHandle()), request, registry=registry)


def _to_wire(value: Any) -> Any:
    from dataclasses import asdict, is_dataclass
    from datetime import datetime
    from enum import Enum
    from pathlib import Path

    from pydantic import BaseModel

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _to_wire(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_wire(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _daemon_product(request: Any, context: Any, audit: Any, product: str) -> dict[str, object]:
    """Execute an exact facade product under the daemon's writer and audit."""
    from polylogue.operations.bindings import runtime_operation_binding
    from polylogue.operations.mutation_transaction import OperationExecutor
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    assert context.runtime is not None
    fields = _normalize_product_fields(product, dict(request.payload))
    actuator_type, args_type = _PRODUCTS[product]
    actuator = actuator_type()
    binding = runtime_operation_binding(actuator)
    executor = OperationExecutor(audit=audit, archive_root=context.archive_root)
    with ArchiveStore.open_existing(context.archive_root, read_only=False) as archive:
        args = args_type(archive=archive, **fields)
        try:
            preview = executor.prepare_bound_for_archive(
                binding, args, context.principal, archive_root=context.archive_root
            )
            authorization = executor.authorize_bound(
                binding, preview, context.principal, confirmation_strength="bound_token"
            )
        except KeyError as exc:
            if isinstance(fields.get("session_id"), str):
                raise FacadeProductRefusalError("session_not_found", str(fields["session_id"])) from exc
            raise
        try:
            receipt = executor.execute_bound(binding, preview, authorization, args)
        except KeyError as exc:
            raise FacadeProductRefusalError("mutation_target_vanished", str(exc)) from exc
    if receipt.status == "blocked":
        raise FacadeProductRefusalError("mutation_blocked", receipt.detail or receipt.operation)
    result = {
        "status": receipt.status,
        "affected_count": receipt.affected_count,
        "domain_receipt": _to_wire(dict(receipt.domain_receipt)),
        "plan_context": _to_wire(dict(preview.plan.context)),
    }
    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": "committed" if receipt.affected_count else "no-effect",
        "affected_count": receipt.affected_count,
        "receipt_ref": receipt.receipt_ref,
        "result": result,
    }


def facade_set_metadata(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    del snapshot
    return _daemon_product(request, context, audit, "set_metadata")


def facade_delete_metadata(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    del snapshot
    return _daemon_product(request, context, audit, "delete_metadata")


def facade_bulk_tag_sessions(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    del snapshot
    return _daemon_product(request, context, audit, "bulk_tag_sessions")


def facade_create_recall_pack(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    del snapshot
    return _daemon_product(request, context, audit, "create_recall_pack")


def facade_delete_recall_pack(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    del snapshot
    return _daemon_product(request, context, audit, "delete_recall_pack")


def facade_save_workspace(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    del snapshot
    return _daemon_product(request, context, audit, "save_workspace")


def facade_delete_workspace(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    del snapshot
    return _daemon_product(request, context, audit, "delete_workspace")


def facade_record_correction(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    del snapshot
    return _daemon_product(request, context, audit, "record_correction")


def facade_delete_correction(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    del snapshot
    return _daemon_product(request, context, audit, "delete_correction")


def facade_clear_corrections(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    del snapshot
    return _daemon_product(request, context, audit, "clear_corrections")


def facade_post_blackboard_note(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    del snapshot
    return _daemon_product(request, context, audit, "post_blackboard_note")


def facade_delete_session(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    """Resolve one session and consume an exact bound delete authorization."""
    from polylogue.operations.bindings import runtime_operation_binding
    from polylogue.operations.mutation_transaction import MutationPrincipal, OperationExecutor
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    del snapshot
    assert context.runtime is not None
    token = str(request.payload["session_id"])
    actor = str(request.payload.get("actor") or context.principal.actor_ref)
    with ArchiveStore.open_existing(context.archive_root, read_only=False) as archive:
        try:
            resolved = archive.resolve_session_id(token)
        except KeyError:
            value: dict[str, object] = {"outcome": "not_found", "session_id": token, "detail": "session_not_found"}
            return {
                "operation": request.operation,
                "outcome": "completed",
                "sequence": 1,
                "effect": "no-effect",
                "affected_count": 0,
                "result": {"value": value},
            }
        actuator = actuators.SessionDeleteActuator()
        args = actuators.SessionDeleteArgs(archive=archive, session_ids=(resolved,))
        binding = runtime_operation_binding(actuator)
        principal = MutationPrincipal(
            actor, context.principal.capabilities, context.principal.surface, context.principal.role_label
        )
        executor = OperationExecutor(audit=audit, archive_root=context.archive_root)
        preview = executor.prepare_bound_for_archive(binding, args, principal, archive_root=context.archive_root)
        authorization = executor.authorize_bound(binding, preview, principal, confirmation_strength="bound_token")
        receipt = executor.execute_bound(binding, preview, authorization, args)
    if receipt.status == "blocked":
        raise FacadeProductRefusalError("mutation_blocked", receipt.detail or receipt.operation)
    deleted = receipt.affected_count > 0
    value = {
        "outcome": "deleted" if deleted else "not_found",
        "session_id": resolved,
        "detail": None if deleted else "session_not_found",
    }
    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": "committed" if deleted else "no-effect",
        "affected_count": receipt.affected_count,
        "receipt_ref": receipt.receipt_ref,
        "result": {"value": value},
    }


def facade_add_tag(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    del snapshot
    return _daemon_product(request, context, audit, "add_tag")


def facade_remove_tag(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    del snapshot
    return _daemon_product(request, context, audit, "remove_tag")


def facade_save_view(request: Any, context: Any, audit: Any, snapshot: Any) -> dict[str, object]:
    del snapshot
    return _daemon_product(request, context, audit, "save_view")

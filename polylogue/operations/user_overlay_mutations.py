"""Audited daemon mutations for HTTP user overlays."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.core.user_state_targets import TARGET_MESSAGE, TARGET_SESSION, validate_mark_type, validate_target_kind
from polylogue.operations.bindings import runtime_operation_binding
from polylogue.operations.daemon_mutations import _resolve_session_target
from polylogue.operations.mutation_actuators import (
    AnnotationDeleteActuator,
    AnnotationDeleteArgs,
    AnnotationSaveActuator,
    AnnotationSaveArgs,
    MarkAddActuator,
    MarkArgs,
    MarkRemoveActuator,
    RecallPackDeleteActuator,
    RecallPackDeleteArgs,
    RecallPackSaveActuator,
    RecallPackSaveArgs,
    SavedViewDeleteActuator,
    SavedViewDeleteArgs,
    SavedViewSaveActuator,
    SavedViewSaveArgs,
    WorkspaceDeleteActuator,
    WorkspaceDeleteArgs,
    WorkspaceSaveActuator,
    WorkspaceSaveArgs,
)
from polylogue.operations.mutation_transaction import OperationExecutor
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.surfaces.payloads import MutationResultPayload

if TYPE_CHECKING:
    from polylogue.operations.audit import AuditRepository
    from polylogue.operations.daemon_protocol import DaemonOperationRequest
    from polylogue.operations.operation_context import OperationContext, PinnedOperationRead


def _text(payload: dict[str, object], key: str, default: str = "") -> str:
    value = payload.get(key)
    return str(value) if value is not None else default


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _target(archive: ArchiveStore, root: Any, payload: dict[str, object]) -> tuple[str, str, str, str | None]:
    session_id = _resolve_session_target(archive, root, _text(payload, "session_id"))
    target_type = _text(payload, "target_type", TARGET_SESSION)
    target_id = _text(payload, "target_id") or None
    message_id = _text(payload, "message_id") or None
    validate_target_kind(target_type)
    if target_type == TARGET_SESSION:
        if target_id and _resolve_session_target(archive, root, target_id) != session_id:
            raise ValueError("session target_id must match session_id")
        return target_type, session_id, session_id, None
    if target_type == TARGET_MESSAGE:
        if target_id and message_id and target_id != message_id:
            raise ValueError("message target_id must match message_id")
        message_id = message_id or target_id
        if not message_id:
            raise ValueError("message target requires message_id or target_id")
        if (
            archive._conn.execute(
                "SELECT 1 FROM messages WHERE session_id = ? AND message_id = ?",
                (session_id, message_id),
            ).fetchone()
            is None
        ):
            raise ValueError(f"message {message_id!r} is not in session {session_id!r}")
        return target_type, message_id, session_id, message_id
    from polylogue.api.user_state_resolver import resolve_insight_target

    resolved = asyncio.run(
        resolve_insight_target(
            root,
            target_type=target_type,
            target_id=target_id,
            session_id=session_id,
            message_id=message_id,
        )
    )
    return str(resolved["target_type"]), str(resolved["target_id"]), session_id, resolved.get("message_id")


def _execute(
    request: DaemonOperationRequest,
    context: OperationContext,
    audit: AuditRepository,
    build: Any,
    *,
    public_operation: str,
    resource_type: str | None = None,
    resource_id: str | None = None,
    target: tuple[str, str, str, str | None] | None = None,
    mark_type: str | None = None,
) -> dict[str, object]:
    assert context.runtime is not None
    executor = OperationExecutor(audit=audit, archive_root=context.archive_root)
    with ArchiveStore.open_existing(context.archive_root, read_only=False) as archive:
        actuator, args, resolved_target = build(archive)
        if resolved_target is not None and target is None:
            target = resolved_target
        binding = runtime_operation_binding(actuator)
        preview = executor.prepare_bound_for_archive(
            binding, args, context.principal, archive_root=context.archive_root
        )
        authorization = executor.authorize_bound(
            binding, preview, context.principal, confirmation_strength="bound_token"
        )
        receipt = executor.execute_bound(binding, preview, authorization, args)
    if receipt.status in {"blocked", "unknown"}:
        raise ValueError(receipt.detail or f"{actuator.operation} did not apply")
    created = bool(receipt.domain_receipt.get("created"))
    deleting = public_operation.endswith(".delete") or public_operation == "mark.delete"
    if deleting:
        status = "deleted" if receipt.affected_count else "not_found"
        detail = (
            None
            if receipt.affected_count
            else ("mark_not_present" if public_operation == "mark.delete" else receipt.detail)
        )
    elif public_operation == "mark.add":
        status = "ok" if receipt.affected_count else "unchanged"
        detail = None if receipt.affected_count else "already_present"
    else:
        status = "ok"
        detail = None if created else "updated"
    public_count = 1 if not deleting and public_operation != "mark.add" else receipt.affected_count
    fields: dict[str, object] = {
        "status": status,
        "detail": detail,
        "operation": public_operation,
        "affected_count": public_count,
    }
    if resource_type is not None:
        fields.update(resource_type=resource_type, resource_id=resource_id)
    if target is not None:
        target_type, target_id, session_id, message_id = target
        fields.update(
            target_type=target_type,
            target_id=target_id,
            session_id=session_id,
            message_id=message_id,
            mark_type=mark_type,
        )
    result = MutationResultPayload.model_validate(fields).model_dump(mode="json", exclude_none=True)
    return {
        "operation": request.operation,
        "outcome": "completed",
        "sequence": 1,
        "effect": "committed" if receipt.affected_count else "no-effect",
        "affected_count": receipt.affected_count,
        "receipt_ref": receipt.receipt_ref,
        "result": result,
    }


def user_mark_add(
    request: DaemonOperationRequest, context: OperationContext, audit: AuditRepository, snapshot: PinnedOperationRead
) -> dict[str, object]:
    del snapshot
    payload = request.payload
    mark_type = validate_mark_type(_text(payload, "mark_type"))

    def build(archive: ArchiveStore) -> tuple[Any, Any, Any]:
        target = _target(archive, context.archive_root, payload)
        return MarkAddActuator(), MarkArgs(archive, target[0], target[1], mark_type, target[2]), target

    public_target = (
        _text(payload, "target_type", TARGET_SESSION),
        _text(payload, "target_id") or _text(payload, "message_id") or _text(payload, "session_id"),
        _text(payload, "session_id"),
        _text(payload, "message_id") or None,
    )
    return _execute(
        request, context, audit, build, public_operation="mark.add", mark_type=mark_type, target=public_target
    )


def user_mark_remove(
    request: DaemonOperationRequest, context: OperationContext, audit: AuditRepository, snapshot: PinnedOperationRead
) -> dict[str, object]:
    del snapshot
    payload = request.payload
    mark_type = validate_mark_type(_text(payload, "mark_type"))

    def build(archive: ArchiveStore) -> tuple[Any, Any, Any]:
        target = _target(archive, context.archive_root, payload)
        return MarkRemoveActuator(), MarkArgs(archive, target[0], target[1], mark_type, target[2]), target

    public_target = (
        _text(payload, "target_type", TARGET_SESSION),
        _text(payload, "target_id") or _text(payload, "message_id") or _text(payload, "session_id"),
        _text(payload, "session_id"),
        _text(payload, "message_id") or None,
    )
    return _execute(
        request, context, audit, build, public_operation="mark.delete", mark_type=mark_type, target=public_target
    )


def user_annotation_save(
    request: DaemonOperationRequest, context: OperationContext, audit: AuditRepository, snapshot: PinnedOperationRead
) -> dict[str, object]:
    del snapshot
    payload = request.payload
    annotation_id = _text(payload, "annotation_id")
    note_text = _text(payload, "note_text")
    if not annotation_id.strip() or not note_text.strip():
        raise ValueError("annotation_id and note_text are required")

    def build(archive: ArchiveStore) -> tuple[Any, Any, Any]:
        target = _target(archive, context.archive_root, payload)
        return (
            AnnotationSaveActuator(),
            AnnotationSaveArgs(archive, annotation_id, target[0], target[1], note_text, target[2]),
            None,
        )

    return _execute(
        request,
        context,
        audit,
        build,
        public_operation="annotation.save",
        resource_type="annotation",
        resource_id=annotation_id,
        target=(
            _text(payload, "target_type", TARGET_SESSION),
            _text(payload, "target_id") or _text(payload, "message_id") or _text(payload, "session_id"),
            _text(payload, "session_id"),
            _text(payload, "message_id") or None,
        ),
    )


def user_annotation_delete(
    request: DaemonOperationRequest, context: OperationContext, audit: AuditRepository, snapshot: PinnedOperationRead
) -> dict[str, object]:
    del snapshot
    identifier = _text(request.payload, "id")
    return _execute(
        request,
        context,
        audit,
        lambda archive: (AnnotationDeleteActuator(), AnnotationDeleteArgs(archive, identifier), None),
        public_operation="annotation.delete",
        resource_type="annotation",
        resource_id=identifier,
    )


def user_saved_view_save(
    request: DaemonOperationRequest, context: OperationContext, audit: AuditRepository, snapshot: PinnedOperationRead
) -> dict[str, object]:
    del snapshot
    payload = request.payload
    query = payload["query"]
    if not isinstance(query, dict):
        raise ValueError("query must be an object")
    SessionQuerySpec.from_params(query, strict=True)
    view_id = _text(payload, "view_id")
    name = _text(payload, "name").strip()
    watch = bool(payload.get("watch", False))
    if not name:
        raise ValueError("name is required")
    if watch:
        from polylogue.archive.query.watch_definition import validate_watch_definition

        validate_watch_definition(query)
    return _execute(
        request,
        context,
        audit,
        lambda archive: (SavedViewSaveActuator(), SavedViewSaveArgs(archive, view_id, name, _json(query), watch), None),
        public_operation="saved_view.save",
        resource_type="saved_view",
        resource_id=view_id,
    )


def user_saved_view_delete(
    request: DaemonOperationRequest, context: OperationContext, audit: AuditRepository, snapshot: PinnedOperationRead
) -> dict[str, object]:
    del snapshot
    identifier = _text(request.payload, "id")
    return _execute(
        request,
        context,
        audit,
        lambda archive: (SavedViewDeleteActuator(), SavedViewDeleteArgs(archive, identifier), None),
        public_operation="saved_view.delete",
        resource_type="saved_view",
        resource_id=identifier,
    )


def user_recall_pack_save(
    request: DaemonOperationRequest, context: OperationContext, audit: AuditRepository, snapshot: PinnedOperationRead
) -> dict[str, object]:
    del snapshot
    payload = request.payload
    pack_id, label = _text(payload, "pack_id").strip(), _text(payload, "label").strip()
    body = payload["payload"]
    if not pack_id or not label or not isinstance(body, dict) or not isinstance(body.get("items"), list):
        raise ValueError("invalid recall pack")
    from polylogue.operations.user_overlay_targets import normalize_recall_pack

    def build(archive: ArchiveStore) -> tuple[Any, Any, Any]:
        session_ids, normalized = normalize_recall_pack(archive, label, body)
        return (
            RecallPackSaveActuator(),
            RecallPackSaveArgs(archive, pack_id, label, _json(session_ids), _json(normalized)),
            None,
        )

    return _execute(
        request,
        context,
        audit,
        build,
        public_operation="recall_pack.save",
        resource_type="recall_pack",
        resource_id=pack_id,
    )


def user_recall_pack_delete(
    request: DaemonOperationRequest, context: OperationContext, audit: AuditRepository, snapshot: PinnedOperationRead
) -> dict[str, object]:
    del snapshot
    identifier = _text(request.payload, "id")
    return _execute(
        request,
        context,
        audit,
        lambda archive: (RecallPackDeleteActuator(), RecallPackDeleteArgs(archive, identifier), None),
        public_operation="recall_pack.delete",
        resource_type="recall_pack",
        resource_id=identifier,
    )


def user_workspace_save(
    request: DaemonOperationRequest, context: OperationContext, audit: AuditRepository, snapshot: PinnedOperationRead
) -> dict[str, object]:
    del snapshot
    payload = request.payload
    workspace_id, name = _text(payload, "workspace_id").strip(), _text(payload, "name").strip()
    mode = _text(payload, "mode", "tabs").strip()
    if not workspace_id or not name or mode not in {"tabs", "stack", "compare", "timeline"}:
        raise ValueError("invalid workspace")
    from polylogue.operations.user_overlay_targets import normalize_overlay_item

    def build(archive: ArchiveStore) -> tuple[Any, Any, Any]:
        open_targets = payload.get("open_targets", [])
        active = payload.get("active_target", {})
        if not isinstance(open_targets, list) or not all(isinstance(item, dict) for item in open_targets):
            raise ValueError("open_targets must be a list of objects")
        if not isinstance(active, dict):
            raise ValueError("active_target must be an object")
        targets = [normalize_overlay_item(archive, item) for item in open_targets]
        active_normalized = normalize_overlay_item(archive, active) if active else {}
        return (
            WorkspaceSaveActuator(),
            WorkspaceSaveArgs(
                archive,
                workspace_id,
                name,
                mode,
                _json(targets),
                _json(payload.get("layout", {})),
                _json(active_normalized),
            ),
            None,
        )

    return _execute(
        request,
        context,
        audit,
        build,
        public_operation="workspace.save",
        resource_type="workspace",
        resource_id=workspace_id,
    )


def user_workspace_delete(
    request: DaemonOperationRequest, context: OperationContext, audit: AuditRepository, snapshot: PinnedOperationRead
) -> dict[str, object]:
    del snapshot
    identifier = _text(request.payload, "id")
    return _execute(
        request,
        context,
        audit,
        lambda archive: (WorkspaceDeleteActuator(), WorkspaceDeleteArgs(archive, identifier), None),
        public_operation="workspace.delete",
        resource_type="workspace",
        resource_id=identifier,
    )

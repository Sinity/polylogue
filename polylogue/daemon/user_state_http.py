"""Daemon HTTP handlers for durable reader user state."""

from __future__ import annotations

import hashlib
import json
from http import HTTPStatus
from typing import Any, cast

from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.core.user_state_targets import TARGET_SESSION, is_mark_type_supported, validate_target_kind
from polylogue.surfaces.payloads import MutationResultPayload


def _read_json_body(handler: Any) -> dict[str, object] | None:
    content_length = int(handler.headers.get("Content-Length", 0))
    body_raw = handler.rfile.read(content_length) if content_length > 0 else b"{}"
    body_text = body_raw.decode("utf-8")
    try:
        body = json.loads(body_text)
    except json.JSONDecodeError:
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return None
    if not isinstance(body, dict):
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return None
    return body


def _saved_view_payload(row: dict[str, str]) -> dict[str, object]:
    query_json = row["query_json"]
    try:
        query = json.loads(query_json)
    except json.JSONDecodeError:
        query = None
    return {
        "view_id": row["view_id"],
        "name": row["name"],
        "query": query,
        "query_json": query_json,
        "created_at": row["created_at"],
    }


def _recall_pack_payload(row: dict[str, str]) -> dict[str, object]:
    try:
        session_ids = json.loads(row["session_ids_json"])
    except json.JSONDecodeError:
        session_ids = []
    try:
        payload = json.loads(row["payload_json"])
    except json.JSONDecodeError:
        payload = {}
    return {
        "pack_id": row["pack_id"],
        "label": row["label"],
        "session_ids": session_ids,
        "payload": payload,
        "created_at": row["created_at"],
    }


def _workspace_payload(row: dict[str, str]) -> dict[str, object]:
    try:
        open_targets = json.loads(row["open_targets_json"])
    except json.JSONDecodeError:
        open_targets = []
    try:
        layout = json.loads(row["layout_json"])
    except json.JSONDecodeError:
        layout = {}
    try:
        active_target = json.loads(row["active_target_json"])
    except json.JSONDecodeError:
        active_target = {}
    return {
        "workspace_id": row["workspace_id"],
        "name": row["name"],
        "mode": row["mode"],
        "open_targets": open_targets if isinstance(open_targets, list) else [],
        "layout": layout if isinstance(layout, dict) else {},
        "active_target": active_target if isinstance(active_target, dict) else {},
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _default_saved_view_id(name: str, query_json: str) -> str:
    digest = hashlib.sha256(f"{name}\0{query_json}".encode()).hexdigest()[:16]
    return f"view-{digest}"


def _default_annotation_id(target_type: str, target_id: str, note_text: str) -> str:
    digest = hashlib.sha256(f"{target_type}\0{target_id}\0{note_text}".encode()).hexdigest()[:16]
    return f"annotation-{digest}"


def _mutation_status(created_or_deleted: bool, *, missing_detail: str | None = None) -> tuple[str, str | None, int]:
    if created_or_deleted:
        return "ok", None, 1
    if missing_detail is None:
        return "unchanged", "already_present", 0
    return "not_found", missing_detail, 0


def _save_mutation_status(created: bool) -> tuple[str, str | None, int]:
    if created:
        return "ok", None, 1
    return "ok", "updated", 1


def _send_mutation_result(handler: Any, result: MutationResultPayload, *, created: bool = False) -> None:
    handler._send_json(
        HTTPStatus.CREATED if created else HTTPStatus.OK,
        result.model_dump(mode="json", exclude_none=True),
    )


def dispatch_get(handler: Any, path: list[str], params: dict[str, list[str]]) -> bool:
    if path == ["marks"]:
        handler._handle_user_state(handle_list_marks, params)
        return True
    if path == ["annotations"]:
        handler._handle_user_state(handle_list_annotations, params)
        return True
    if len(path) == 2 and path[0] == "annotations" and path[1]:
        handler._handle_user_state(handle_get_annotation, path[1])
        return True
    if path == ["saved-views"]:
        handler._handle_user_state(handle_list_saved_views)
        return True
    if len(path) == 2 and path[0] == "saved-views" and path[1]:
        handler._handle_user_state(handle_get_saved_view, path[1])
        return True
    if path == ["recall-packs"]:
        handler._handle_user_state(handle_list_recall_packs)
        return True
    if len(path) == 2 and path[0] == "recall-packs" and path[1]:
        handler._handle_user_state(handle_get_recall_pack, path[1])
        return True
    if path == ["workspaces"]:
        handler._handle_user_state(handle_list_workspaces)
        return True
    if len(path) == 2 and path[0] == "workspaces" and path[1]:
        handler._handle_user_state(handle_get_workspace, path[1])
        return True
    return False


def dispatch_post(handler: Any, path: list[str]) -> bool:
    routes: dict[tuple[str, ...], Any] = {
        ("marks",): handle_create_mark,
        ("annotations",): handle_save_annotation,
        ("saved-views",): handle_save_view,
        ("recall-packs",): handle_save_recall_pack,
        ("workspaces",): handle_save_workspace,
    }
    route = routes.get(tuple(path))
    if route is None:
        return False
    handler._handle_user_state(route)
    return True


def dispatch_delete(handler: Any, path: list[str], params: dict[str, list[str]]) -> bool:
    if path == ["marks"]:
        handler._handle_user_state(handle_delete_mark, params)
        return True
    if len(path) == 2 and path[0] == "annotations" and path[1]:
        handler._handle_user_state(handle_delete_annotation, path[1])
        return True
    if len(path) == 2 and path[0] == "saved-views" and path[1]:
        handler._handle_user_state(handle_delete_saved_view, path[1])
        return True
    if len(path) == 2 and path[0] == "recall-packs" and path[1]:
        handler._handle_user_state(handle_delete_recall_pack, path[1])
        return True
    if len(path) == 2 and path[0] == "workspaces" and path[1]:
        handler._handle_user_state(handle_delete_workspace, path[1])
        return True
    return False


def handle_list_marks(handler: Any, params: dict[str, list[str]]) -> None:
    mark_type = handler._get_param(params, "mark_type")
    session_id = handler._get_param(params, "session_id")
    target_type = handler._get_param(params, "target_type")
    target_id = handler._get_param(params, "target_id")
    message_id = handler._get_param(params, "message_id")

    async def _list(poly: Any) -> list[dict[str, str]]:
        return cast(
            list[dict[str, str]],
            await poly.list_marks(
                mark_type=mark_type,
                session_id=session_id,
                target_type=target_type,
                target_id=target_id,
                message_id=message_id,
            ),
        )

    marks = cast(list[dict[str, str]], handler._sync_run(_list))
    handler._send_json(HTTPStatus.OK, {"items": marks, "total": len(marks)})


def handle_create_mark(handler: Any) -> None:
    body = _read_json_body(handler)
    if body is None:
        return
    session_id = str(body.get("session_id") or "")
    mark_type = str(body.get("mark_type") or "")
    target_type = str(body.get("target_type") or TARGET_SESSION)
    target_id = str(body.get("target_id") or "") or None
    message_id = str(body.get("message_id") or "") or None
    if not session_id or not is_mark_type_supported(mark_type):
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return
    try:
        validate_target_kind(target_type)
    except ValueError:
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_target_type")
        return

    async def _create(poly: Any) -> MutationResultPayload:
        created = await poly.add_mark(
            session_id,
            mark_type,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
        return MutationResultPayload(
            status="ok" if created else "unchanged",
            detail=None if created else "already_present",
            operation="mark.add",
            affected_count=1 if created else 0,
            target_type=target_type,
            target_id=target_id or message_id or session_id,
            session_id=session_id,
            message_id=message_id,
            mark_type=mark_type,
        )

    result = cast(MutationResultPayload, handler._sync_run(_create))
    _send_mutation_result(handler, result, created=result.status == "ok")


def handle_delete_mark(handler: Any, params: dict[str, list[str]]) -> None:
    session_id = handler._get_param(params, "session_id")
    mark_type = handler._get_param(params, "mark_type")
    target_type = handler._get_param(params, "target_type", TARGET_SESSION)
    target_id = handler._get_param(params, "target_id")
    message_id = handler._get_param(params, "message_id")
    if not session_id or not mark_type:
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return

    async def _delete(poly: Any) -> MutationResultPayload:
        deleted = await poly.remove_mark(
            session_id,
            mark_type,
            target_type=target_type or TARGET_SESSION,
            target_id=target_id,
            message_id=message_id,
        )
        return MutationResultPayload(
            status="deleted" if deleted else "not_found",
            detail=None if deleted else "mark_not_present",
            operation="mark.delete",
            affected_count=1 if deleted else 0,
            target_type=target_type or TARGET_SESSION,
            target_id=target_id or message_id or session_id,
            session_id=session_id,
            message_id=message_id,
            mark_type=mark_type,
        )

    result = cast(MutationResultPayload, handler._sync_run(_delete))
    _send_mutation_result(handler, result)


def handle_list_annotations(handler: Any, params: dict[str, list[str]]) -> None:
    session_id = handler._get_param(params, "session_id")
    target_type = handler._get_param(params, "target_type")
    target_id = handler._get_param(params, "target_id")
    message_id = handler._get_param(params, "message_id")

    async def _list(poly: Any) -> list[dict[str, str]]:
        return cast(
            list[dict[str, str]],
            await poly.list_annotations(
                session_id=session_id,
                target_type=target_type,
                target_id=target_id,
                message_id=message_id,
            ),
        )

    annotations = cast(list[dict[str, str]], handler._sync_run(_list))
    handler._send_json(HTTPStatus.OK, {"items": annotations, "total": len(annotations)})


def handle_get_annotation(handler: Any, annotation_id: str) -> None:
    async def _get(poly: Any) -> dict[str, str] | None:
        return cast(dict[str, str] | None, await poly.get_annotation(annotation_id))

    row = cast(dict[str, str] | None, handler._sync_run(_get))
    if row is None:
        handler._send_error(HTTPStatus.NOT_FOUND, "not_found")
        return
    handler._send_json(HTTPStatus.OK, row)


def handle_save_annotation(handler: Any) -> None:
    body = _read_json_body(handler)
    if body is None:
        return
    session_id = str(body.get("session_id") or "")
    target_type = str(body.get("target_type") or TARGET_SESSION)
    target_id = str(body.get("target_id") or "") or None
    message_id = str(body.get("message_id") or "") or None
    note_text = str(body.get("note_text") or "")
    if not session_id or not note_text.strip():
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return
    try:
        validate_target_kind(target_type)
    except ValueError:
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_target_type")
        return
    resolved_target_id = target_id or message_id or session_id
    annotation_id = str(body.get("annotation_id") or _default_annotation_id(target_type, resolved_target_id, note_text))

    async def _save(poly: Any) -> MutationResultPayload:
        existing = await poly.get_annotation(annotation_id)
        await poly.save_annotation(
            annotation_id,
            session_id,
            note_text,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
        created = existing is None
        status, detail, affected_count = _save_mutation_status(created)
        return MutationResultPayload(
            status=status,
            detail=detail,
            operation="annotation.save",
            affected_count=affected_count,
            resource_type="annotation",
            resource_id=annotation_id,
            target_type=target_type,
            target_id=resolved_target_id,
            session_id=session_id,
            message_id=message_id,
        )

    result = cast(MutationResultPayload, handler._sync_run(_save))
    _send_mutation_result(handler, result, created=result.detail is None)


def handle_delete_annotation(handler: Any, annotation_id: str) -> None:
    async def _delete(poly: Any) -> MutationResultPayload:
        deleted = await poly.delete_annotation(annotation_id)
        status, detail, affected_count = _mutation_status(deleted, missing_detail="annotation_not_found")
        return MutationResultPayload(
            status="deleted" if status == "ok" else status,
            detail=detail,
            operation="annotation.delete",
            affected_count=affected_count,
            resource_type="annotation",
            resource_id=annotation_id,
        )

    result = cast(MutationResultPayload, handler._sync_run(_delete))
    _send_mutation_result(handler, result)


def handle_list_saved_views(handler: Any) -> None:
    async def _list(poly: Any) -> list[dict[str, str]]:
        return cast(list[dict[str, str]], await poly.list_views())

    rows = cast(list[dict[str, str]], handler._sync_run(_list))
    items = [_saved_view_payload(row) for row in rows]
    handler._send_json(HTTPStatus.OK, {"items": items, "total": len(items)})


def handle_get_saved_view(handler: Any, view_id: str) -> None:
    async def _get(poly: Any) -> dict[str, str] | None:
        return cast(dict[str, str] | None, await poly.get_view(view_id))

    row = cast(dict[str, str] | None, handler._sync_run(_get))
    if row is None:
        handler._send_error(HTTPStatus.NOT_FOUND, "not_found")
        return
    handler._send_json(HTTPStatus.OK, _saved_view_payload(row))


def handle_save_view(handler: Any) -> None:
    body = _read_json_body(handler)
    if body is None:
        return
    name = str(body.get("name") or "").strip()
    query = body.get("query")
    if not name or not isinstance(query, dict):
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return

    SessionQuerySpec.from_params(query, strict=True)
    query_json = json.dumps(query, sort_keys=True, separators=(",", ":"))
    view_id = str(body.get("view_id") or _default_saved_view_id(name, query_json))

    async def _save(poly: Any) -> MutationResultPayload:
        existing = await poly.get_view(view_id)
        await poly.save_view(view_id, name, query_json)
        created = existing is None
        status, detail, affected_count = _save_mutation_status(created)
        return MutationResultPayload(
            status=status,
            detail=detail,
            operation="saved_view.save",
            affected_count=affected_count,
            resource_type="saved_view",
            resource_id=view_id,
        )

    result = cast(MutationResultPayload, handler._sync_run(_save))
    _send_mutation_result(handler, result, created=result.detail is None)


def handle_delete_saved_view(handler: Any, view_id: str) -> None:
    async def _delete(poly: Any) -> MutationResultPayload:
        deleted = await poly.delete_view(view_id)
        status, detail, affected_count = _mutation_status(deleted, missing_detail="saved_view_not_found")
        return MutationResultPayload(
            status="deleted" if status == "ok" else status,
            detail=detail,
            operation="saved_view.delete",
            affected_count=affected_count,
            resource_type="saved_view",
            resource_id=view_id,
        )

    result = cast(MutationResultPayload, handler._sync_run(_delete))
    _send_mutation_result(handler, result)


def handle_list_recall_packs(handler: Any) -> None:
    async def _list(poly: Any) -> list[dict[str, str]]:
        return cast(list[dict[str, str]], await poly.list_recall_packs())

    rows = cast(list[dict[str, str]], handler._sync_run(_list))
    items = [_recall_pack_payload(row) for row in rows]
    handler._send_json(HTTPStatus.OK, {"items": items, "total": len(items)})


def handle_get_recall_pack(handler: Any, pack_id: str) -> None:
    async def _get(poly: Any) -> dict[str, str] | None:
        return cast(dict[str, str] | None, await poly.get_recall_pack(pack_id))

    row = cast(dict[str, str] | None, handler._sync_run(_get))
    if row is None:
        handler._send_error(HTTPStatus.NOT_FOUND, "not_found")
        return
    handler._send_json(HTTPStatus.OK, _recall_pack_payload(row))


def handle_save_recall_pack(handler: Any) -> None:
    body = _read_json_body(handler)
    if body is None:
        return
    pack_id = str(body.get("pack_id") or "").strip()
    label = str(body.get("label") or "").strip()
    payload = body.get("payload", {})
    if not pack_id or not label or not isinstance(payload, dict):
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return
    items = payload.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    async def _save(poly: Any) -> MutationResultPayload:
        existing = await poly.get_recall_pack(pack_id)
        await poly.create_recall_pack(pack_id, label, payload_json)
        created = existing is None
        status, detail, affected_count = _save_mutation_status(created)
        return MutationResultPayload(
            status=status,
            detail=detail,
            operation="recall_pack.save",
            affected_count=affected_count,
            resource_type="recall_pack",
            resource_id=pack_id,
        )

    result = cast(MutationResultPayload, handler._sync_run(_save))
    _send_mutation_result(handler, result, created=result.detail is None)


def handle_delete_recall_pack(handler: Any, pack_id: str) -> None:
    async def _delete(poly: Any) -> MutationResultPayload:
        deleted = await poly.delete_recall_pack(pack_id)
        status, detail, affected_count = _mutation_status(deleted, missing_detail="recall_pack_not_found")
        return MutationResultPayload(
            status="deleted" if status == "ok" else status,
            detail=detail,
            operation="recall_pack.delete",
            affected_count=affected_count,
            resource_type="recall_pack",
            resource_id=pack_id,
        )

    result = cast(MutationResultPayload, handler._sync_run(_delete))
    _send_mutation_result(handler, result)


def handle_list_workspaces(handler: Any) -> None:
    async def _list(poly: Any) -> list[dict[str, str]]:
        return cast(list[dict[str, str]], await poly.list_workspaces())

    rows = cast(list[dict[str, str]], handler._sync_run(_list))
    items = [_workspace_payload(row) for row in rows]
    handler._send_json(HTTPStatus.OK, {"items": items, "total": len(items)})


def handle_get_workspace(handler: Any, workspace_id: str) -> None:
    async def _get(poly: Any) -> dict[str, str] | None:
        return cast(dict[str, str] | None, await poly.get_workspace(workspace_id))

    row = cast(dict[str, str] | None, handler._sync_run(_get))
    if row is None:
        handler._send_error(HTTPStatus.NOT_FOUND, "not_found")
        return
    handler._send_json(HTTPStatus.OK, _workspace_payload(row))


def handle_save_workspace(handler: Any) -> None:
    body = _read_json_body(handler)
    if body is None:
        return
    workspace_id = str(body.get("workspace_id") or "").strip()
    name = str(body.get("name") or "").strip()
    mode = str(body.get("mode") or "tabs").strip()
    open_targets = body.get("open_targets", [])
    layout = body.get("layout", {})
    active_target = body.get("active_target", {})
    if (
        not workspace_id
        or not name
        or mode not in {"tabs", "stack", "compare", "timeline"}
        or not isinstance(open_targets, list)
        or not all(isinstance(item, dict) for item in open_targets)
        or not isinstance(layout, dict)
        or not isinstance(active_target, dict)
    ):
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return

    async def _save(poly: Any) -> MutationResultPayload:
        existing = await poly.get_workspace(workspace_id)
        await poly.save_workspace(
            workspace_id,
            name,
            mode,
            json.dumps(open_targets, sort_keys=True, separators=(",", ":")),
            json.dumps(layout, sort_keys=True, separators=(",", ":")),
            json.dumps(active_target, sort_keys=True, separators=(",", ":")),
        )
        created = existing is None
        status, detail, affected_count = _save_mutation_status(created)
        return MutationResultPayload(
            status=status,
            detail=detail,
            operation="workspace.save",
            affected_count=affected_count,
            resource_type="workspace",
            resource_id=workspace_id,
        )

    result = cast(MutationResultPayload, handler._sync_run(_save))
    _send_mutation_result(handler, result, created=result.detail is None)


def handle_delete_workspace(handler: Any, workspace_id: str) -> None:
    async def _delete(poly: Any) -> MutationResultPayload:
        deleted = await poly.delete_workspace(workspace_id)
        status, detail, affected_count = _mutation_status(deleted, missing_detail="workspace_not_found")
        return MutationResultPayload(
            status="deleted" if status == "ok" else status,
            detail=detail,
            operation="workspace.delete",
            affected_count=affected_count,
            resource_type="workspace",
            resource_id=workspace_id,
        )

    result = cast(MutationResultPayload, handler._sync_run(_delete))
    _send_mutation_result(handler, result)

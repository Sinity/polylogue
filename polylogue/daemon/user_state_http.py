"""Daemon HTTP handlers for durable reader user state."""

from __future__ import annotations

import hashlib
import json
from http import HTTPStatus
from typing import Any, cast

from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.core.user_state_targets import TARGET_KIND_NAMES


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
    target_type = str(body.get("target_type") or "session")
    target_id = str(body.get("target_id") or "") or None
    message_id = str(body.get("message_id") or "") or None
    if not session_id or mark_type not in {"star", "pin", "archive"}:
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return
    if target_type not in TARGET_KIND_NAMES:
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_target_type")
        return

    async def _create(poly: Any) -> dict[str, object]:
        created = await poly.add_mark(
            session_id,
            mark_type,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
        return {
            "target_type": target_type,
            "target_id": target_id or message_id or session_id,
            "session_id": session_id,
            "message_id": message_id,
            "mark_type": mark_type,
            "created": created,
        }

    result = cast(dict[str, object], handler._sync_run(_create))
    handler._send_json(HTTPStatus.CREATED if result["created"] else HTTPStatus.OK, result)


def handle_delete_mark(handler: Any, params: dict[str, list[str]]) -> None:
    session_id = handler._get_param(params, "session_id")
    mark_type = handler._get_param(params, "mark_type")
    target_type = handler._get_param(params, "target_type", "session")
    target_id = handler._get_param(params, "target_id")
    message_id = handler._get_param(params, "message_id")
    if not session_id or not mark_type:
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return

    async def _delete(poly: Any) -> dict[str, object]:
        deleted = await poly.remove_mark(
            session_id,
            mark_type,
            target_type=target_type or "session",
            target_id=target_id,
            message_id=message_id,
        )
        return {
            "target_type": target_type or "session",
            "target_id": target_id or message_id or session_id,
            "session_id": session_id,
            "message_id": message_id,
            "mark_type": mark_type,
            "deleted": deleted,
        }

    handler._send_json(HTTPStatus.OK, handler._sync_run(_delete))


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
    target_type = str(body.get("target_type") or "session")
    target_id = str(body.get("target_id") or "") or None
    message_id = str(body.get("message_id") or "") or None
    note_text = str(body.get("note_text") or "")
    if not session_id or not note_text.strip():
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return
    if target_type not in TARGET_KIND_NAMES:
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_target_type")
        return
    resolved_target_id = target_id or message_id or session_id
    annotation_id = str(body.get("annotation_id") or _default_annotation_id(target_type, resolved_target_id, note_text))

    async def _save(poly: Any) -> dict[str, object]:
        created = await poly.save_annotation(
            annotation_id,
            session_id,
            note_text,
            target_type=target_type,
            target_id=target_id,
            message_id=message_id,
        )
        saved = await poly.get_annotation(annotation_id)
        if saved is None:
            return {"annotation_id": annotation_id, "created": created}
        result: dict[str, object] = dict(saved)
        result["created"] = created
        return result

    result = cast(dict[str, object], handler._sync_run(_save))
    handler._send_json(HTTPStatus.CREATED if result["created"] else HTTPStatus.OK, result)


def handle_delete_annotation(handler: Any, annotation_id: str) -> None:
    async def _delete(poly: Any) -> dict[str, object]:
        deleted = await poly.delete_annotation(annotation_id)
        return {"annotation_id": annotation_id, "deleted": deleted}

    handler._send_json(HTTPStatus.OK, handler._sync_run(_delete))


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

    async def _save(poly: Any) -> dict[str, object]:
        created = await poly.save_view(view_id, name, query_json)
        saved = await poly.get_view(view_id)
        if saved is None:
            return {"view_id": view_id, "created": created}
        result = _saved_view_payload(saved)
        result["created"] = created
        return result

    result = cast(dict[str, object], handler._sync_run(_save))
    handler._send_json(HTTPStatus.CREATED if result["created"] else HTTPStatus.OK, result)


def handle_delete_saved_view(handler: Any, view_id: str) -> None:
    async def _delete(poly: Any) -> dict[str, object]:
        deleted = await poly.delete_view(view_id)
        return {"view_id": view_id, "deleted": deleted}

    handler._send_json(HTTPStatus.OK, handler._sync_run(_delete))


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

    async def _save(poly: Any) -> dict[str, object]:
        created = await poly.create_recall_pack(pack_id, label, payload_json)
        saved = await poly.get_recall_pack(pack_id)
        if saved is None:
            return {"pack_id": pack_id, "created": created}
        result = _recall_pack_payload(saved)
        result["created"] = created
        return result

    result = cast(dict[str, object], handler._sync_run(_save))
    handler._send_json(HTTPStatus.CREATED if result["created"] else HTTPStatus.OK, result)


def handle_delete_recall_pack(handler: Any, pack_id: str) -> None:
    async def _delete(poly: Any) -> dict[str, object]:
        deleted = await poly.delete_recall_pack(pack_id)
        return {"pack_id": pack_id, "deleted": deleted}

    handler._send_json(HTTPStatus.OK, handler._sync_run(_delete))


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

    async def _save(poly: Any) -> dict[str, object]:
        created = await poly.save_workspace(
            workspace_id,
            name,
            mode,
            json.dumps(open_targets, sort_keys=True, separators=(",", ":")),
            json.dumps(layout, sort_keys=True, separators=(",", ":")),
            json.dumps(active_target, sort_keys=True, separators=(",", ":")),
        )
        saved = await poly.get_workspace(workspace_id)
        if saved is None:
            return {"workspace_id": workspace_id, "created": created}
        result = _workspace_payload(saved)
        result["created"] = created
        return result

    result = cast(dict[str, object], handler._sync_run(_save))
    handler._send_json(HTTPStatus.CREATED if result["created"] else HTTPStatus.OK, result)


def handle_delete_workspace(handler: Any, workspace_id: str) -> None:
    async def _delete(poly: Any) -> dict[str, object]:
        deleted = await poly.delete_workspace(workspace_id)
        return {"workspace_id": workspace_id, "deleted": deleted}

    handler._send_json(HTTPStatus.OK, handler._sync_run(_delete))

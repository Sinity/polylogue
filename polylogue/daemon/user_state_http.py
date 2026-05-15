"""Daemon HTTP handlers for durable reader user state."""

from __future__ import annotations

import hashlib
import json
from http import HTTPStatus
from typing import Any, cast

from polylogue.archive.query.spec import ConversationQuerySpec


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
        conversation_ids = json.loads(row["conversation_ids_json"])
    except json.JSONDecodeError:
        conversation_ids = []
    try:
        payload = json.loads(row["payload_json"])
    except json.JSONDecodeError:
        payload = {}
    return {
        "pack_id": row["pack_id"],
        "label": row["label"],
        "conversation_ids": conversation_ids,
        "payload": payload,
        "created_at": row["created_at"],
    }


def _default_saved_view_id(name: str, query_json: str) -> str:
    digest = hashlib.sha256(f"{name}\0{query_json}".encode()).hexdigest()[:16]
    return f"view-{digest}"


def handle_list_marks(handler: Any, params: dict[str, list[str]]) -> None:
    mark_type = handler._get_param(params, "mark_type")
    conversation_id = handler._get_param(params, "conversation_id")

    async def _list(poly: Any) -> list[dict[str, str]]:
        return cast(list[dict[str, str]], await poly.list_marks(mark_type=mark_type, conversation_id=conversation_id))

    marks = cast(list[dict[str, str]], handler._sync_run(_list))
    handler._send_json(HTTPStatus.OK, {"items": marks, "total": len(marks)})


def handle_create_mark(handler: Any) -> None:
    body = _read_json_body(handler)
    if body is None:
        return
    conversation_id = str(body.get("conversation_id") or "")
    mark_type = str(body.get("mark_type") or "")
    if not conversation_id or mark_type not in {"star", "pin", "archive"}:
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return

    async def _create(poly: Any) -> dict[str, object]:
        created = await poly.add_mark(conversation_id, mark_type)
        return {"conversation_id": conversation_id, "mark_type": mark_type, "created": created}

    result = cast(dict[str, object], handler._sync_run(_create))
    handler._send_json(HTTPStatus.CREATED if result["created"] else HTTPStatus.OK, result)


def handle_delete_mark(handler: Any, params: dict[str, list[str]]) -> None:
    conversation_id = handler._get_param(params, "conversation_id")
    mark_type = handler._get_param(params, "mark_type")
    if not conversation_id or not mark_type:
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return

    async def _delete(poly: Any) -> dict[str, object]:
        deleted = await poly.remove_mark(conversation_id, mark_type)
        return {"conversation_id": conversation_id, "mark_type": mark_type, "deleted": deleted}

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

    ConversationQuerySpec.from_params(query, strict=True)
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
    conversation_ids = body.get("conversation_ids")
    payload = body.get("payload", {})
    if not pack_id or not label or not isinstance(conversation_ids, list) or not isinstance(payload, dict):
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return
    conversation_ids_json = json.dumps([str(item) for item in conversation_ids], sort_keys=True)
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    async def _save(poly: Any) -> dict[str, object]:
        created = await poly.create_recall_pack(pack_id, label, conversation_ids_json, payload_json)
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

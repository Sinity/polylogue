"""Executable HTTP declarations and adapters for durable user overlays."""

from __future__ import annotations

import hashlib
import json
from http import HTTPStatus
from typing import Any, cast
from uuid import uuid4

from polylogue.daemon.route_types import AuthPolicy, RouteMethod, RouteSpec
from polylogue.declarations import (
    CompatibilityKey,
    CompletenessEdge,
    DeclarationSpec,
    ExampleSpec,
    HandlerBinding,
    OutputSpec,
)


def _route(
    method: RouteMethod,
    path: str,
    operation: str,
    response: str,
    *,
    symbol: str,
    request: str,
    parameterized: bool = False,
) -> RouteSpec:
    producer = f"polylogue.daemon.http.DaemonAPIHandler.{symbol}"
    binding = f"{method} {path}"
    auth: AuthPolicy = "credential_if_configured" if method == "GET" else "credential_and_same_origin"
    return RouteSpec(
        kernel=DeclarationSpec(
            declaration_id="daemon.user." + operation.replace(".", "_"),
            family_id=f"daemon.user-overlay.{operation}",
            public_name=f"{method.lower()}:{path}",
            owner_path="polylogue/daemon/http.py",
            compatibility=CompatibilityKey(
                "daemon-route",
                "stable",
                "daemon-read" if method == "GET" else "daemon-write",
                response,
                "read-only" if method == "GET" else "durable-user",
            ),
            producer=producer,
            role_gate=auth,
            schema_ref=response,
            discovery_text=f"Execute {operation} through the daemon user-overlay operation.",
            repair_command="devtools render openapi",
            handlers=(HandlerBinding("daemon-http", "polylogue/daemon/http.py", symbol, binding),),
            outputs=(OutputSpec("response", "json", response, path),),
            examples=(ExampleSpec("route", f"Call {binding}", ()),),
            completeness_edges=(
                CompletenessEdge(producer, "daemon-http", "route", "polylogue/daemon/http.py"),
                CompletenessEdge(producer, "openapi-schema", "generated-document", "docs/openapi/search.yaml"),
            ),
        ),
        method=method,
        path=path,
        request_contract=request,
        response_contract=response,
        auth_policy=auth,
        domain_operation=operation,
        passes_path=parameterized or path.startswith("/api/user/"),
        auth_scope="read" if method == "GET" else "user_state",
        write_gate=False,
        kind="user_overlay",
        stability="stable",
    )


_LIST = "OverlayListPayload"
_ITEM = "OverlayItemPayload"
_MUTATION = "MutationResultPayload"
ROUTES: tuple[RouteSpec, ...] = (
    _route(
        "GET",
        "/api/assertions",
        "user.assertions.list",
        "AssertionClaimListPayload",
        symbol="_handle_assertions",
        request="AssertionClaimQuery",
    ),
    _route(
        "GET", "/api/user/marks", "user.marks.list", _LIST, symbol="_handle_user_overlay_get", request="MarkListQuery"
    ),
    _route(
        "GET",
        "/api/user/annotations",
        "user.annotations.list",
        _LIST,
        symbol="_handle_user_overlay_get",
        request="AnnotationListQuery",
    ),
    _route(
        "GET",
        "/api/user/annotations/:id",
        "user.annotations.get",
        _ITEM,
        symbol="_handle_user_overlay_get",
        request="OverlayIdQuery",
        parameterized=True,
    ),
    _route(
        "GET",
        "/api/user/saved-views",
        "user.saved_views.list",
        _LIST,
        symbol="_handle_user_overlay_get",
        request="EmptyQuery",
    ),
    _route(
        "GET",
        "/api/user/saved-views/:id",
        "user.saved_views.get",
        _ITEM,
        symbol="_handle_user_overlay_get",
        request="OverlayIdQuery",
        parameterized=True,
    ),
    _route(
        "GET",
        "/api/user/recall-packs",
        "user.recall_packs.list",
        _LIST,
        symbol="_handle_user_overlay_get",
        request="EmptyQuery",
    ),
    _route(
        "GET",
        "/api/user/recall-packs/:id",
        "user.recall_packs.get",
        _ITEM,
        symbol="_handle_user_overlay_get",
        request="OverlayIdQuery",
        parameterized=True,
    ),
    _route(
        "GET",
        "/api/user/workspaces",
        "user.workspaces.list",
        _LIST,
        symbol="_handle_user_overlay_get",
        request="EmptyQuery",
    ),
    _route(
        "GET",
        "/api/user/workspaces/:id",
        "user.workspaces.get",
        _ITEM,
        symbol="_handle_user_overlay_get",
        request="OverlayIdQuery",
        parameterized=True,
    ),
    _route(
        "POST",
        "/api/user/marks",
        "user.mark.add",
        _MUTATION,
        symbol="_handle_user_overlay_post",
        request="MarkMutationBody",
    ),
    _route(
        "POST",
        "/api/user/annotations",
        "user.annotation.save",
        _MUTATION,
        symbol="_handle_user_overlay_post",
        request="AnnotationMutationBody",
    ),
    _route(
        "POST",
        "/api/user/saved-views",
        "user.saved_view.save",
        _MUTATION,
        symbol="_handle_user_overlay_post",
        request="SavedViewMutationBody",
    ),
    _route(
        "POST",
        "/api/user/recall-packs",
        "user.recall_pack.save",
        _MUTATION,
        symbol="_handle_user_overlay_post",
        request="RecallPackMutationBody",
    ),
    _route(
        "POST",
        "/api/user/workspaces",
        "user.workspace.save",
        _MUTATION,
        symbol="_handle_user_overlay_post",
        request="WorkspaceMutationBody",
    ),
    _route(
        "DELETE",
        "/api/user/marks",
        "user.mark.remove",
        _MUTATION,
        symbol="_handle_user_overlay_delete",
        request="MarkMutationQuery",
    ),
    _route(
        "DELETE",
        "/api/user/annotations/:id",
        "user.annotation.delete",
        _MUTATION,
        symbol="_handle_user_overlay_delete",
        request="OverlayIdQuery",
        parameterized=True,
    ),
    _route(
        "DELETE",
        "/api/user/saved-views/:id",
        "user.saved_view.delete",
        _MUTATION,
        symbol="_handle_user_overlay_delete",
        request="OverlayIdQuery",
        parameterized=True,
    ),
    _route(
        "DELETE",
        "/api/user/recall-packs/:id",
        "user.recall_pack.delete",
        _MUTATION,
        symbol="_handle_user_overlay_delete",
        request="OverlayIdQuery",
        parameterized=True,
    ),
    _route(
        "DELETE",
        "/api/user/workspaces/:id",
        "user.workspace.delete",
        _MUTATION,
        symbol="_handle_user_overlay_delete",
        request="OverlayIdQuery",
        parameterized=True,
    ),
)


def _value(params: dict[str, list[str]], name: str) -> str | None:
    values = params.get(name)
    return values[0] if values else None


def _operation(handler: Any, name: str, payload: dict[str, object]) -> dict[str, object] | None:
    from polylogue.operations.daemon_protocol import DaemonOperationRequest

    try:
        request = DaemonOperationRequest.from_dict(
            DaemonOperationRequest(
                operation=name,
                request_id=uuid4().hex,
                archive_root=str(handler.server.archive_root),
                payload=payload,
            ).to_dict()
        )
    except (TypeError, ValueError):
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return None
    return cast("dict[str, object] | None", handler._execute_daemon_operation(request))


def _result(handler: Any, envelope: dict[str, object] | None) -> dict[str, object] | None:
    if envelope is None:
        return None
    if envelope.get("outcome") != "completed":
        handler._send_daemon_operation(envelope)
        return None
    result = envelope.get("result")
    if not isinstance(result, dict):
        handler._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "invalid_operation_result")
        return None
    nested = result.get("result")
    if isinstance(nested, dict) and isinstance(nested.get("status"), str):
        return nested
    return result


def _route_name(method: str, path: list[str]) -> str | None:
    for route in ROUTES:
        if route.method != method:
            continue
        segments = route.path.strip("/").split("/")
        if len(segments) == len(path) and all(a.startswith(":") or a == b for a, b in zip(segments, path, strict=True)):
            return route.domain_operation
    return None


def handle_get(handler: Any, path: list[str], params: dict[str, list[str]]) -> None:
    name = _route_name("GET", path)
    if name is None:
        handler._send_error(HTTPStatus.NOT_FOUND, "not_found")
        return
    payload: dict[str, object] = {}
    if name == "user.marks.list":
        payload = {
            key: value
            for key in ("mark_type", "session_id", "target_type", "target_id", "message_id")
            if (value := _value(params, key)) is not None
        }
    elif name == "user.annotations.list":
        payload = {
            key: value
            for key in ("session_id", "target_type", "target_id", "message_id")
            if (value := _value(params, key)) is not None
        }
    elif name.endswith(".get"):
        payload = {"id": path[-1]}
    result = _result(handler, _operation(handler, name, payload))
    if result is None:
        return
    if name.endswith(".get"):
        if not result.get("found"):
            handler._send_error(HTTPStatus.NOT_FOUND, "not_found")
            return
        item = result["item"]
        if not isinstance(item, dict):
            handler._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "invalid_operation_result")
            return
        result = item
    handler._send_json(HTTPStatus.OK, result)


def handle_assertions(handler: Any, params: dict[str, list[str]]) -> None:
    from polylogue.archive.query.spec import clamp_query_limit

    raw_kinds = [
        item.strip()
        for key in ("kind", "kinds")
        for value in params.get(key, [])
        for item in value.split(",")
        if item.strip()
    ]
    raw_statuses = [
        item.strip().lower()
        for key in ("status", "statuses")
        for value in params.get(key, [])
        for item in value.split(",")
        if item.strip()
    ]
    statuses: list[str] | None = list(dict.fromkeys(raw_statuses)) if raw_statuses else ["active", "candidate"]
    if statuses is not None and any(token in {"all", "*"} for token in statuses):
        statuses = None
    payload: dict[str, object] = {
        "kinds": list(dict.fromkeys(raw_kinds)) or None,
        "statuses": statuses,
        "limit": clamp_query_limit(handler._get_int(params, "limit", 20), default=20),
    }
    for key in ("target_ref", "scope_ref"):
        value = handler._get_param(params, key)
        if value is not None:
            payload[key] = value
    if "context_inject" in params:
        payload["context_inject"] = handler._get_bool(params, "context_inject")
    result = _result(handler, _operation(handler, "user.assertions.list", payload))
    if result is not None:
        handler._send_json(HTTPStatus.OK, result)


def _read_body(handler: Any) -> dict[str, object] | None:
    from polylogue.operations.daemon_protocol import MAX_DECLARED_OPERATION_BODY_BYTES

    if handler.headers.get("Transfer-Encoding") is not None:
        handler._send_error(HTTPStatus.BAD_REQUEST, "unsupported_transfer_encoding")
        return None
    try:
        raw_length = handler.headers.get("Content-Length", "0")
        if not isinstance(raw_length, str) or not raw_length.isascii() or not raw_length.isdecimal():
            raise ValueError("invalid Content-Length")
        length = int(raw_length)
        if length > MAX_DECLARED_OPERATION_BODY_BYTES:
            handler._send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request_too_large")
            return None
        body = json.loads(handler.rfile.read(length) if length else b"{}")
    except (ValueError, UnicodeDecodeError):
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return None
    if not isinstance(body, dict):
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return None
    return body


def _default_id(name: str, *parts: str) -> str:
    return name + "-" + hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def handle_post(handler: Any, path: list[str], params: dict[str, list[str]]) -> None:
    del params
    name = _route_name("POST", path)
    if name is None:
        handler._send_error(HTTPStatus.NOT_FOUND, "not_found")
        return
    payload = _read_body(handler)
    if payload is None:
        return
    if name == "user.mark.add":
        from polylogue.core.user_state_targets import is_mark_type_supported, validate_target_kind

        if not payload.get("session_id") or not is_mark_type_supported(str(payload.get("mark_type") or "")):
            handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return
        try:
            validate_target_kind(str(payload.get("target_type") or "session"))
        except ValueError:
            handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_target_type")
            return
    elif name == "user.annotation.save":
        from polylogue.core.user_state_targets import validate_target_kind

        if not payload.get("session_id") or not str(payload.get("note_text") or "").strip():
            handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return
        try:
            validate_target_kind(str(payload.get("target_type") or "session"))
        except ValueError:
            handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_target_type")
            return
    elif name == "user.saved_view.save":
        if (
            not str(payload.get("name") or "").strip()
            or not isinstance(payload.get("query"), dict)
            or not isinstance(payload.get("watch", False), bool)
        ):
            handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return
    elif name == "user.recall_pack.save":
        body = payload.get("payload", {})
        if (
            not str(payload.get("pack_id") or "").strip()
            or not str(payload.get("label") or "").strip()
            or not isinstance(body, dict)
            or not isinstance(body.get("items"), list)
            or any(not isinstance(item, dict) for item in body["items"])
        ):
            handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return
    elif name == "user.workspace.save":
        targets = payload.get("open_targets", [])
        if (
            not str(payload.get("workspace_id") or "").strip()
            or not str(payload.get("name") or "").strip()
            or str(payload.get("mode") or "tabs") not in {"tabs", "stack", "compare", "timeline"}
            or not isinstance(targets, list)
            or any(not isinstance(item, dict) for item in targets)
            or not isinstance(payload.get("layout", {}), dict)
            or not isinstance(payload.get("active_target", {}), dict)
        ):
            handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
            return
    if name == "user.annotation.save" and not payload.get("annotation_id"):
        payload["annotation_id"] = _default_id(
            "annotation",
            str(payload.get("target_type") or "session"),
            str(payload.get("target_id") or payload.get("message_id") or payload.get("session_id") or ""),
            str(payload.get("note_text") or ""),
        )
    if name == "user.saved_view.save" and not payload.get("view_id"):
        query = payload.get("query")
        if isinstance(query, dict):
            payload["view_id"] = _default_id(
                "view", str(payload.get("name") or ""), json.dumps(query, sort_keys=True, separators=(",", ":"))
            )
    result = _result(handler, _operation(handler, name, payload))
    if result is not None:
        handler._send_json(HTTPStatus.CREATED if result.get("detail") is None else HTTPStatus.OK, result)


def handle_delete(handler: Any, path: list[str], params: dict[str, list[str]]) -> None:
    name = _route_name("DELETE", path)
    if name is None:
        handler._send_error(HTTPStatus.NOT_FOUND, "not_found")
        return
    payload: dict[str, object] = (
        {"id": path[-1]}
        if name != "user.mark.remove"
        else {
            key: value
            for key in ("session_id", "mark_type", "target_type", "target_id", "message_id")
            if (value := _value(params, key)) is not None
        }
    )
    if name == "user.mark.remove" and (not payload.get("session_id") or not payload.get("mark_type")):
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")
        return
    result = _result(handler, _operation(handler, name, payload))
    if result is not None:
        handler._send_json(HTTPStatus.OK, result)

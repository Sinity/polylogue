"""Reader workspace route payload builders for the local daemon."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, cast

from polylogue.surfaces.payloads import TargetRefPayload

if TYPE_CHECKING:
    from polylogue.api import Polylogue

SessionPayloadLoader = Callable[["Polylogue", str, "MessageWindow"], Awaitable[object]]


WORKSPACE_SHELL_MODES = {"tabs", "stack", "compare", "timeline"}
COMPARE_ALIGN_MODES = {"prompt"}

#: The message window a workspace view is served with when the request
#: declares none. Stack and compare render a narrow reading window over each
#: referenced session, so serving every message of every session shipped a
#: payload that grew with total session length instead of with what the view
#: asked for (polylogue-o0zju). This is a declared default for an explicit
#: parameter, not a ceiling: ``limit`` is whatever the caller asks for,
#: ``limit=0`` requests the whole transcript, and both the window that was
#: served and each session's TRUE message total are reported back, so a
#: bounded response can never read as a complete one.
WORKSPACE_MESSAGE_WINDOW = 50


@dataclass(frozen=True, slots=True)
class MessageWindow:
    """The message window one workspace request asked each session for."""

    limit: int | None = WORKSPACE_MESSAGE_WINDOW
    offset: int = 0

    def payload(self) -> dict[str, object]:
        """Report the window on the envelope, so the bound is observable."""

        return {"limit": self.limit, "offset": self.offset}


def parse_id_list(params: dict[str, list[str]]) -> list[str]:
    ids: list[str] = []
    for raw in params.get("ids", []):
        ids.extend(part.strip() for part in raw.split(",") if part.strip())
    return ids


def parse_message_window(handler: object, params: dict[str, list[str]]) -> MessageWindow:
    """Read the declared ``(limit, offset)`` message window off a request.

    ``limit=0`` is the explicit "whole transcript" request, so a caller that
    genuinely wants every message can still say so; it is not reachable by
    accident, because the absent-parameter case is the declared window.
    """

    raw_limit = handler._get_int(params, "limit", WORKSPACE_MESSAGE_WINDOW)  # type: ignore[attr-defined]
    offset = max(0, handler._get_int(params, "offset", 0))  # type: ignore[attr-defined]
    limit = None if raw_limit <= 0 else raw_limit
    return MessageWindow(limit=limit, offset=offset)


def target_ref_from_session_payload(payload: Mapping[str, object]) -> dict[str, object]:
    target_ref = payload.get("target_ref")
    if isinstance(target_ref, dict):
        return dict(target_ref)
    conv_id = str(payload.get("id") or "")
    return TargetRefPayload.session(conv_id).model_dump(mode="json", exclude_none=True)


def missing_session_target(conv_id: str) -> dict[str, object]:
    return {
        "target_type": "session",
        "target_id": conv_id,
        "session_id": conv_id,
        "status": "missing",
        "disabled_reason": "session_not_found",
    }


def stack_payload(
    ids: list[str], focus: str | None, items: list[dict[str, object]], window: MessageWindow
) -> dict[str, object]:
    """Assemble the stack envelope from already-loaded session payloads.

    Shared by the archive-backed and database-backed stack routes so the
    served window is reported identically on both; an envelope that carried
    the window on only one of them would let the other read as complete.
    """

    return {
        "mode": "stack",
        "ids": ids,
        "focus": focus,
        "items": items,
        "total": len(items),
        "resolved_count": sum(1 for item in items if item["status"] == "resolved"),
        "degraded_count": sum(1 for item in items if item["status"] != "resolved"),
        **window.payload(),
    }


def stack_item(payload: dict[str, object]) -> dict[str, object]:
    """Wrap one resolved session payload as a stack target."""

    return {
        "target_type": "session",
        "target_id": str(payload["id"]),
        "session_id": str(payload["id"]),
        "status": "resolved",
        "identity_key": f"session:{payload['id']}",
        "target_ref": target_ref_from_session_payload(payload),
        "session": payload,
    }


async def build_stack_payload(
    poly: Polylogue,
    ids: list[str],
    focus: str | None,
    load_session: SessionPayloadLoader,
    window: MessageWindow | None = None,
) -> dict[str, object]:
    resolved_window = window if window is not None else MessageWindow()
    items: list[dict[str, object]] = []
    for conv_id in ids:
        payload = await load_session(poly, conv_id, resolved_window)
        if not isinstance(payload, dict):
            items.append(missing_session_target(conv_id))
            continue
        items.append(stack_item(payload))
    return stack_payload(ids, focus, items, resolved_window)


async def build_compare_payload(
    poly: Polylogue,
    left: str,
    right: str,
    align: str,
    load_session: SessionPayloadLoader,
    window: MessageWindow | None = None,
) -> dict[str, object]:
    # Imported lazily to avoid a circular import: ``compare`` re-uses
    # ``missing_session_target`` and ``COMPARE_ALIGN_MODES`` from this
    # module, so importing it at module scope would trip the initial import.
    from polylogue.daemon.compare import build_compare_envelope

    resolved_window = window if window is not None else MessageWindow()
    left_payload = await load_session(poly, left, resolved_window)
    right_payload = await load_session(poly, right, resolved_window)
    envelope = build_compare_envelope(left_payload, right_payload, left, right, align, window=resolved_window)
    return cast("dict[str, object]", envelope)


def handle_stack(handler: object, params: dict[str, list[str]]) -> None:
    ids = parse_id_list(params)
    focus = handler._get_param(params, "focus")  # type: ignore[attr-defined]
    if not ids:
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")  # type: ignore[attr-defined]
        return
    window = parse_message_window(handler, params)

    async def _get(poly: Polylogue) -> object:
        return await build_stack_payload(poly, ids, focus, handler._do_get_session_window, window)  # type: ignore[attr-defined]

    handler._send_json(HTTPStatus.OK, handler._sync_run(_get))  # type: ignore[attr-defined]


def handle_compare(handler: object, params: dict[str, list[str]]) -> None:
    left = handler._get_param(params, "left")  # type: ignore[attr-defined]
    right = handler._get_param(params, "right")  # type: ignore[attr-defined]
    align = handler._get_param(params, "align", "prompt")  # type: ignore[attr-defined]
    if not left or not right or align not in COMPARE_ALIGN_MODES:
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")  # type: ignore[attr-defined]
        return
    window = parse_message_window(handler, params)

    async def _get(poly: Polylogue) -> object:
        load_session = handler._do_get_session_window  # type: ignore[attr-defined]
        return await build_compare_payload(poly, left, right, align or "prompt", load_session, window)

    handler._send_json(HTTPStatus.OK, handler._sync_run(_get))  # type: ignore[attr-defined]


__all__ = [
    "COMPARE_ALIGN_MODES",
    "WORKSPACE_MESSAGE_WINDOW",
    "WORKSPACE_SHELL_MODES",
    "MessageWindow",
    "build_compare_payload",
    "build_stack_payload",
    "parse_id_list",
    "parse_message_window",
    "stack_item",
    "stack_payload",
]

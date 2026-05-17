"""Reader workspace route payload builders for the local daemon."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from http import HTTPStatus
from typing import TYPE_CHECKING, cast

from polylogue.surfaces.payloads import TargetRefPayload

if TYPE_CHECKING:
    from polylogue.api import Polylogue

ConversationPayloadLoader = Callable[["Polylogue", str], Awaitable[object]]


WORKSPACE_SHELL_MODES = {"tabs", "stack", "compare", "timeline"}
COMPARE_ALIGN_MODES = {"prompt"}


def parse_id_list(params: dict[str, list[str]]) -> list[str]:
    ids: list[str] = []
    for raw in params.get("ids", []):
        ids.extend(part.strip() for part in raw.split(",") if part.strip())
    return ids


def target_ref_from_conversation_payload(payload: Mapping[str, object]) -> dict[str, object]:
    target_ref = payload.get("target_ref")
    if isinstance(target_ref, dict):
        return dict(target_ref)
    conv_id = str(payload.get("id") or "")
    return TargetRefPayload.conversation(conv_id).model_dump(mode="json", exclude_none=True)


def missing_conversation_target(conv_id: str) -> dict[str, object]:
    return {
        "target_type": "conversation",
        "target_id": conv_id,
        "conversation_id": conv_id,
        "status": "missing",
        "disabled_reason": "conversation_not_found",
    }


async def build_stack_payload(
    poly: Polylogue,
    ids: list[str],
    focus: str | None,
    load_conversation: ConversationPayloadLoader,
) -> dict[str, object]:
    items: list[dict[str, object]] = []
    for conv_id in ids:
        payload = await load_conversation(poly, conv_id)
        if not isinstance(payload, dict):
            items.append(missing_conversation_target(conv_id))
            continue
        items.append(
            {
                "target_type": "conversation",
                "target_id": str(payload["id"]),
                "conversation_id": str(payload["id"]),
                "status": "resolved",
                "identity_key": f"conversation:{payload['id']}",
                "target_ref": target_ref_from_conversation_payload(payload),
                "conversation": payload,
            }
        )
    return {
        "mode": "stack",
        "ids": ids,
        "focus": focus,
        "items": items,
        "total": len(items),
        "resolved_count": sum(1 for item in items if item["status"] == "resolved"),
        "degraded_count": sum(1 for item in items if item["status"] != "resolved"),
    }


async def build_compare_payload(
    poly: Polylogue,
    left: str,
    right: str,
    align: str,
    load_conversation: ConversationPayloadLoader,
) -> dict[str, object]:
    # Imported lazily to avoid a circular import: ``compare`` re-uses
    # ``missing_conversation_target`` and ``COMPARE_ALIGN_MODES`` from this
    # module, so importing it at module scope would trip the initial import.
    from polylogue.daemon.compare import build_compare_envelope

    left_payload = await load_conversation(poly, left)
    right_payload = await load_conversation(poly, right)
    envelope = build_compare_envelope(left_payload, right_payload, left, right, align)
    return cast("dict[str, object]", envelope)


def dispatch_get(handler: object, path: list[str], params: dict[str, list[str]]) -> bool:
    if path == ["api", "stack"]:
        handle_stack(handler, params)
        return True
    if path == ["api", "compare"]:
        handle_compare(handler, params)
        return True
    return False


def handle_stack(handler: object, params: dict[str, list[str]]) -> None:
    ids = parse_id_list(params)
    focus = handler._get_param(params, "focus")  # type: ignore[attr-defined]
    if not ids:
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")  # type: ignore[attr-defined]
        return

    async def _get(poly: Polylogue) -> object:
        return await build_stack_payload(poly, ids, focus, handler._do_get_conversation)  # type: ignore[attr-defined]

    handler._send_json(HTTPStatus.OK, handler._sync_run(_get))  # type: ignore[attr-defined]


def handle_compare(handler: object, params: dict[str, list[str]]) -> None:
    left = handler._get_param(params, "left")  # type: ignore[attr-defined]
    right = handler._get_param(params, "right")  # type: ignore[attr-defined]
    align = handler._get_param(params, "align", "prompt")  # type: ignore[attr-defined]
    if not left or not right or align not in COMPARE_ALIGN_MODES:
        handler._send_error(HTTPStatus.BAD_REQUEST, "invalid_request")  # type: ignore[attr-defined]
        return

    async def _get(poly: Polylogue) -> object:
        return await build_compare_payload(poly, left, right, align or "prompt", handler._do_get_conversation)  # type: ignore[attr-defined]

    handler._send_json(HTTPStatus.OK, handler._sync_run(_get))  # type: ignore[attr-defined]


__all__ = [
    "COMPARE_ALIGN_MODES",
    "WORKSPACE_SHELL_MODES",
    "build_compare_payload",
    "build_stack_payload",
    "dispatch_get",
    "parse_id_list",
]

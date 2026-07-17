"""Shared helpers and runtime-service access for the MCP server.

Testmon fan-out note (polylogue-9e5.11): this file is a testmon dependency
"hub" -- its recorded fingerprint touches essentially every test in the
suite, so a change here gets no narrowing benefit from testmon (expect a
full-suite-equivalent selection regardless of edit size). Review changes
with that blast radius in mind; see docs/test-economics.md.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, TypeVar, overload

from pydantic import BaseModel

from polylogue.archive.query.spec import (
    QUERY_ACTION_TYPES,
    QUERY_RETRIEVAL_LANES,
    QUERY_SEQUENCE_ACTION_TYPES,
    QuerySpecError,
    clamp_query_limit,
)
from polylogue.core.enums import MessageType, Origin, enum_values
from polylogue.core.errors import (
    EmbeddingRetrievalNotReadyError,
    PolylogueError,
    SchemaVersionMismatchError,
)
from polylogue.logging import get_logger
from polylogue.mcp.declarations.models import MCPRole, mcp_role_allows
from polylogue.mcp.payloads import MCPErrorPayload, MCPFencedCodeBlock
from polylogue.services import RuntimeServices, build_runtime_services
from polylogue.surfaces.payloads import serialize_surface_payload

if TYPE_CHECKING:
    from polylogue.api import Polylogue
    from polylogue.config import Config

logger = get_logger(__name__)
_runtime_services: RuntimeServices | None = None
TResult = TypeVar("TResult")
MCP_RESPONSE_BUDGET_BYTES = 25_000
MCP_RESPONSE_ENVELOPE_HEADROOM_BYTES = 4_096
_QUERY_ERROR_VALID_VALUES: dict[str, tuple[str, ...]] = {
    "sort": ("date", "tokens", "messages", "words", "longest", "random"),
    "origin": enum_values(Origin),
    "exclude_origin": enum_values(Origin),
    "message_type": enum_values(MessageType),
    "retrieval_lane": QUERY_RETRIEVAL_LANES,
    "action": QUERY_ACTION_TYPES,
    "exclude_action": QUERY_ACTION_TYPES,
    "action_sequence": QUERY_SEQUENCE_ACTION_TYPES,
}
_SESSION_ID_ARGUMENT_NAMES: dict[str, str] = {"get_session_summary": "id"}


@dataclass(frozen=True)
class _ResponseContext:
    """Replay information attached by a tool while it serializes a response."""

    tool: str
    arguments: Mapping[str, object]


_response_context_var: ContextVar[_ResponseContext | None] = ContextVar("mcp_response_context", default=None)


class JSONPayloadSerializer(Protocol):
    def __call__(self, payload: BaseModel, *, exclude_none: bool = False) -> str: ...


class ErrorJSONSerializer(Protocol):
    def __call__(self, message: str, **extra: str) -> str: ...


class FencedCodeExtractor(Protocol):
    def __call__(self, text: str, language: str = "") -> list[MCPFencedCodeBlock]: ...


class ResponseContextHook(Protocol):
    def __call__(self, tool: str, arguments: Mapping[str, object]) -> AbstractContextManager[None]: ...


class SafeCallHook(Protocol):
    def __call__(
        self,
        fn_name: str,
        fn: Callable[[], str],
        *,
        session_id: str | None = None,
        session_ids: tuple[str, ...] = (),
    ) -> str: ...


class AsyncSafeCallHook(Protocol):
    def __call__(
        self,
        fn_name: str,
        fn: Callable[[], Awaitable[str]],
        *,
        session_id: str | None = None,
        session_ids: tuple[str, ...] = (),
    ) -> Awaitable[str]: ...


@dataclass(frozen=True)
class ServerCallbacks:
    json_payload: JSONPayloadSerializer
    clamp_limit: Callable[[int | object], int]
    safe_call: SafeCallHook
    async_safe_call: AsyncSafeCallHook
    error_json: ErrorJSONSerializer
    get_config: Callable[[], Config]
    get_polylogue: Callable[[], Polylogue]
    extract_fenced_code: FencedCodeExtractor
    response_context: ResponseContextHook
    role: MCPRole


def role_allows(role: MCPRole, required: MCPRole) -> bool:
    """Return whether an MCP server role allows the required capability."""

    return mcp_role_allows(role, required)


def _extract_fenced_code(text: str, language: str = "") -> list[MCPFencedCodeBlock]:
    """Extract fenced code blocks from markdown text."""
    if "```" not in text:
        return []
    blocks = text.split("```")
    results: list[MCPFencedCodeBlock] = []
    for i in range(1, len(blocks), 2):
        block = blocks[i]
        lines = block.split("\n", 1)
        block_lang = lines[0].strip() if lines else ""
        code = lines[1] if len(lines) > 1 else block
        if not language or block_lang == language:
            results.append({"language": block_lang, "code": code[:300]})
    return results


@contextmanager
def _response_context(tool: str, arguments: Mapping[str, object]) -> Iterator[None]:
    """Associate one serialized response with its safe, narrower replay call."""
    token = _response_context_var.set(_ResponseContext(tool=tool, arguments=arguments))
    try:
        yield
    finally:
        _response_context_var.reset(token)


def _compact_metadata(
    value: object,
    *,
    string_limit: int = 512,
    item_limit: int = 12,
    depth: int = 0,
    max_depth: int = 3,
) -> object:
    """Keep the budget fallback informative without reproducing its large body."""
    if isinstance(value, str):
        return value if len(value) <= string_limit else value[:string_limit] + "…"
    if isinstance(value, list):
        return {"returned": len(value)}
    if isinstance(value, dict):
        if depth >= max_depth:
            return {"fields": len(value), "truncated": True}
        return {
            str(key): _compact_metadata(
                item,
                string_limit=string_limit,
                item_limit=item_limit,
                depth=depth + 1,
                max_depth=max_depth,
            )
            for key, item in list(value.items())[:item_limit]
        }
    return value


def _fallback_response_arguments(fn_name: str, session_id: str | None) -> dict[str, object]:
    """Return replayable identifiers carried by the safe-call contract."""
    if session_id is None:
        return {}
    return {_SESSION_ID_ARGUMENT_NAMES.get(fn_name, "session_id"): session_id}


def _narrow_continuation(
    context: _ResponseContext | None,
    *,
    consumed: int | None = None,
) -> dict[str, object] | None:
    """Build a direct MCP call descriptor that re-reads the same evidence safely."""
    if context is None:
        return None
    arguments = dict(context.arguments)
    if context.tool == "archive_get_session":
        session_id = arguments.get("session_id")
        if isinstance(session_id, str):
            return {
                "tool": "get_messages",
                "arguments": {
                    "session_id": session_id,
                    "limit": 3,
                    "max_chars_per_message": 4096,
                    "excerpt": True,
                },
                "reason": "The full session exceeded the MCP response budget; read a bounded message page instead.",
            }
    limit = arguments.get("limit")
    if isinstance(limit, int) and not isinstance(limit, bool):
        arguments["limit"] = max(1, min(limit, 3))
    if consumed is not None:
        offset = arguments.get("offset")
        if isinstance(offset, int) and not isinstance(offset, bool):
            arguments["offset"] = offset + consumed
    if context.tool == "get_messages":
        requested_max_chars = arguments.get("max_chars_per_message")
        max_chars = requested_max_chars if isinstance(requested_max_chars, int) else 4096
        arguments["max_chars_per_message"] = min(max_chars, 4096)
        arguments["excerpt"] = True
    if context.tool == "list_corrections":
        requested_max_chars = arguments.get("max_chars_per_correction")
        max_chars = requested_max_chars if isinstance(requested_max_chars, int) else 4096
        arguments["max_chars_per_correction"] = min(max_chars, 4096)
    if "max_chars_per_item" in arguments:
        requested_max_chars = arguments.get("max_chars_per_item")
        max_chars = requested_max_chars if isinstance(requested_max_chars, int) else 512
        arguments["max_chars_per_item"] = min(max_chars, 512)
    return {
        "tool": context.tool,
        "arguments": arguments,
        "reason": "The original response exceeded the MCP response budget; retry this narrower call to read the same evidence.",
    }


def _serialize_payload(payload: BaseModel, *, exclude_none: bool) -> str:
    """Serialize a payload without applying the response budget recursively."""
    to_json = getattr(payload, "to_json", None)
    if callable(to_json):
        result = to_json(exclude_none=exclude_none)
        if not isinstance(result, str):
            raise TypeError(f"{type(payload).__name__}.to_json() returned {type(result).__name__}, expected str")
        return result
    return serialize_surface_payload(payload, exclude_none=exclude_none)


def _bounded_item_page(payload: BaseModel, *, exclude_none: bool) -> tuple[BaseModel, int] | None:
    """Find the largest useful prefix that fits the transport budget."""
    raw_items = getattr(payload, "items", None)
    if not isinstance(raw_items, (tuple, list)) or not raw_items:
        return None
    items = tuple(raw_items)
    low, high = 1, len(items)
    best: tuple[BaseModel, int] | None = None
    while low <= high:
        count = (low + high) // 2
        updates: dict[str, object] = {"items": items[:count]}
        if hasattr(payload, "next_offset"):
            offset = getattr(payload, "offset", 0)
            updates["next_offset"] = offset + count
        candidate = payload.model_copy(update=updates)
        size = len(_serialize_payload(candidate, exclude_none=exclude_none).encode("utf-8"))
        if size <= MCP_RESPONSE_BUDGET_BYTES - MCP_RESPONSE_ENVELOPE_HEADROOM_BYTES:
            best = (candidate, count)
            low = count + 1
        else:
            high = count - 1
    return best


def _budget_envelope(payload: BaseModel, *, original_bytes: int, exclude_none: bool) -> str:
    """Return a useful bounded page, never a metadata-only retry trap."""
    body = payload.model_dump(mode="json", exclude_none=exclude_none)
    context = _response_context_var.get()
    page = _bounded_item_page(payload, exclude_none=exclude_none)
    continuation = _narrow_continuation(context, consumed=page[1] if page is not None else None)
    envelope = {
        "ok": True,
        "status": "response_budget_exceeded",
        "budget_exceeded": True,
        "tool": context.tool if context is not None else None,
        "budget_bytes": MCP_RESPONSE_BUDGET_BYTES,
        "original_bytes": original_bytes,
        "payload_type": type(payload).__name__,
        "metadata": _compact_metadata(body),
        "page": page[0].model_dump(mode="json", exclude_none=exclude_none) if page is not None else None,
        "returned_items": page[1] if page is not None else 0,
        "continuation": continuation,
        "next_action": (
            "Use continuation.tool with continuation.arguments to advance from the returned page."
            if continuation is not None
            else "Request a narrower page or projection for this response."
        ),
    }
    return json.dumps(envelope, indent=2, ensure_ascii=False, default=str)


def _json_payload(payload: BaseModel, *, exclude_none: bool = False) -> str:
    """Serialize typed MCP output, replacing oversized bodies with a safe envelope."""
    result = _serialize_payload(payload, exclude_none=exclude_none)
    original_bytes = len(result.encode("utf-8"))
    return (
        result
        if original_bytes <= MCP_RESPONSE_BUDGET_BYTES
        else _budget_envelope(payload, original_bytes=original_bytes, exclude_none=exclude_none)
    )


def _clamp_limit(limit: int | object) -> int:
    """Clamp limit to the shared ``[1, MAX_QUERY_LIMIT]`` ceiling, default 10.

    Delegates to the canonical :func:`clamp_query_limit` so the MCP, CLI
    query-first, and daemon HTTP page-size ceilings stay identical (#1749).
    """
    return clamp_query_limit(limit, default=10)


def _exception_to_error_json(fn_name: str, exc: BaseException) -> str:
    """Translate an exception raised by an MCP tool body into a typed error JSON.

    The returned string is a serialized :class:`MCPErrorPayload`. Callers must
    return this from the tool body so FastMCP delivers it as a normal tool
    response (with ``is_error=True``) instead of letting the exception escape
    into the stdio loop — an escaping exception kills the entire MCP server
    process and takes every other registered tool offline (#1621).

    Categorization:

    * :class:`SchemaVersionMismatchError` → ``code="schema_version_mismatch"`` with
      ``current_version``/``expected_version`` populated so MCP clients can
      render the same actionable operator message ``readiness_check`` does
      (#1611).
    * Any :class:`PolylogueError` subclass → ``code="polylogue_error"`` with
      ``detail`` set to the exception class name.
    * Any other :class:`Exception` → ``code="internal_error"`` with ``detail``
      set to the exception class name only. The raw exception message is
      deliberately not included so the surface cannot leak credentials, file
      paths, or other internal state.
    """
    from polylogue.archive.query.expression import ExpressionCompileError

    if isinstance(exc, QuerySpecError | ExpressionCompileError):
        field = exc.field
        valid_values = _QUERY_ERROR_VALID_VALUES.get(field, ()) if field is not None else ()
        valid_hint = f" Valid values: {', '.join(valid_values)}." if valid_values else ""
        value = exc.value if isinstance(exc, QuerySpecError) else str(exc)
        payload = MCPErrorPayload(
            message=f"invalid {field}: {value}.{valid_hint}",
            code="invalid_query",
            error="invalid_query",
            detail=type(exc).__name__,
            field=field,
            tool=fn_name,
            valid_values=valid_values,
        )
    elif isinstance(exc, SchemaVersionMismatchError):
        payload = MCPErrorPayload(
            message=str(exc),
            code="schema_version_mismatch",
            error="schema_version_mismatch",
            detail=type(exc).__name__,
            tool=fn_name,
            current_version=exc.current_version,
            expected_version=exc.expected_version,
        )
    elif isinstance(exc, EmbeddingRetrievalNotReadyError):
        # The message is by construction free of secrets — closed
        # vocabulary status enum + fixed command names — so forward it
        # verbatim instead of redacting to the class name (#1503 AC4).
        payload = MCPErrorPayload(
            message=str(exc),
            code="embedding_retrieval_not_ready",
            error="embedding_retrieval_not_ready",
            detail=type(exc).__name__,
            tool=fn_name,
            readiness_status=exc.readiness_status,
        )
    elif isinstance(exc, PolylogueError):
        payload = MCPErrorPayload(
            message=f"{fn_name}: {type(exc).__name__}",
            code="polylogue_error",
            error="polylogue_error",
            detail=type(exc).__name__,
            tool=fn_name,
        )
    else:
        payload = MCPErrorPayload(
            message=f"{fn_name}: internal error ({type(exc).__name__})",
            code="internal_error",
            error="internal_error",
            detail=type(exc).__name__,
            tool=fn_name,
        )
    return _json_payload(payload, exclude_none=True)


def _now_ms() -> int:
    """Return the current wall-clock time in epoch milliseconds."""
    return int(datetime.now(UTC).timestamp() * 1000)


def _record_mcp_call_log(
    fn_name: str,
    session_id: str | None,
    session_ids: tuple[str, ...],
    started_at_ms: int,
    finished_at_ms: int,
    *,
    success: bool,
    error_detail: str | None,
) -> None:
    """Durably spool one MCP call-log event for daemon delivery.

    Persists tool name, session id (when the caller knows one), timing, and
    success/failure to an archive-local filesystem outbox before returning.
    The daemon's write coordinator remains the sole ``ops.db`` writer. Delivery
    failures leave durable retry debt and never alter the tool response.
    """
    try:
        from polylogue.config import load_polylogue_config
        from polylogue.mcp.call_log import enqueue_mcp_call_log

        enqueue_mcp_call_log(
            load_polylogue_config(),
            tool_name=fn_name,
            session_id=session_id,
            session_ids=session_ids,
            started_at_ms=started_at_ms,
            finished_at_ms=finished_at_ms,
            success=success,
            error_detail=error_detail,
        )
    except Exception:
        logger.warning("MCP call-log durable spool failed for %s", fn_name, exc_info=True)


def _mcp_error_detail(result: object) -> str | None:
    """Return the canonical error code when a tool returned MCPErrorPayload."""
    if not isinstance(result, str):
        return None
    try:
        payload = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("ok") is not False or payload.get("status") != "error":
        return None
    error = payload.get("error")
    message = payload.get("message")
    if not isinstance(error, str) or not isinstance(message, str):
        return None
    code = payload.get("code")
    return str(code) if isinstance(code, (int, str)) else error


@overload
def _safe_call(
    fn_name: str,
    fn: Callable[[], str],
    *,
    session_id: str | None = None,
    session_ids: tuple[str, ...] = (),
) -> str: ...


@overload
def _safe_call(
    fn_name: str,
    fn: Callable[[], None],
    *,
    session_id: str | None = None,
    session_ids: tuple[str, ...] = (),
) -> str | None: ...


@overload
def _safe_call(
    fn_name: str,
    fn: Callable[[], TResult],
    *,
    session_id: str | None = None,
    session_ids: tuple[str, ...] = (),
) -> TResult | str: ...


def _safe_call(
    fn_name: str,
    fn: Callable[[], TResult],
    *,
    session_id: str | None = None,
    session_ids: tuple[str, ...] = (),
) -> TResult | str:
    """Call ``fn()`` and return its result, or a typed error JSON on failure.

    Errors are logged server-side and translated through
    :func:`_exception_to_error_json` before being returned. Returning rather
    than raising guarantees that one bad tool call cannot crash the MCP stdio
    loop and disable every other registered tool (#1621). The translated
    payload exposes only the exception class name — never the raw message,
    arguments, or traceback — so error surfaces remain free of secrets.

    Every call is first durably spooled, then idempotently delivered to the
    ``ops.db`` MCP call-log table, keyed by ``fn_name`` and the optional
    ``session_id`` the caller passes when the tool is session-scoped.
    """
    started_at_ms = _now_ms()
    try:
        with _response_context(fn_name, _fallback_response_arguments(fn_name, session_id)):
            result = fn()
    except Exception as exc:
        logger.exception("MCP tool %s failed", fn_name)
        _record_mcp_call_log(
            fn_name,
            session_id,
            session_ids,
            started_at_ms,
            _now_ms(),
            success=False,
            error_detail=type(exc).__name__,
        )
        return _exception_to_error_json(fn_name, exc)
    error_detail = _mcp_error_detail(result)
    _record_mcp_call_log(
        fn_name,
        session_id,
        session_ids,
        started_at_ms,
        _now_ms(),
        success=error_detail is None,
        error_detail=error_detail,
    )
    return result


@overload
async def _async_safe_call(
    fn_name: str,
    fn: Callable[[], Awaitable[str]],
    *,
    session_id: str | None = None,
    session_ids: tuple[str, ...] = (),
) -> str: ...


@overload
async def _async_safe_call(
    fn_name: str,
    fn: Callable[[], Awaitable[None]],
    *,
    session_id: str | None = None,
    session_ids: tuple[str, ...] = (),
) -> str | None: ...


@overload
async def _async_safe_call(
    fn_name: str,
    fn: Callable[[], Awaitable[TResult]],
    *,
    session_id: str | None = None,
    session_ids: tuple[str, ...] = (),
) -> TResult | str: ...


async def _async_safe_call(
    fn_name: str,
    fn: Callable[[], Awaitable[TResult]],
    *,
    session_id: str | None = None,
    session_ids: tuple[str, ...] = (),
) -> TResult | str:
    """Async counterpart of :func:`_safe_call`. Same isolation and call-log contract."""
    started_at_ms = _now_ms()
    try:
        with _response_context(fn_name, _fallback_response_arguments(fn_name, session_id)):
            result = await fn()
    except Exception as exc:
        logger.exception("MCP tool %s failed", fn_name)
        _record_mcp_call_log(
            fn_name,
            session_id,
            session_ids,
            started_at_ms,
            _now_ms(),
            success=False,
            error_detail=type(exc).__name__,
        )
        return _exception_to_error_json(fn_name, exc)
    error_detail = _mcp_error_detail(result)
    _record_mcp_call_log(
        fn_name,
        session_id,
        session_ids,
        started_at_ms,
        _now_ms(),
        success=error_detail is None,
        error_detail=error_detail,
    )
    return result


def _error_json(message: str, **extra: object) -> str:
    """Return a JSON-encoded error dict."""
    code = extra.pop("code", None)
    error = str(code) if code is not None else "error"
    return _json_payload(MCPErrorPayload(message=message, code=code, error=error, **extra), exclude_none=True)  # type: ignore[arg-type]


def _set_runtime_services(services: RuntimeServices | None) -> None:
    """Set the runtime service scope for the server process."""
    global _runtime_services
    _runtime_services = services


def _get_runtime_services() -> RuntimeServices:
    """Return the configured runtime service scope for MCP handlers."""
    global _runtime_services
    if _runtime_services is None:
        _runtime_services = build_runtime_services()
    return _runtime_services


def _get_config() -> Config:
    """Return the MCP config from the configured runtime services."""
    return _get_runtime_services().get_config()


def _get_polylogue() -> Polylogue:
    """Return a ``Polylogue`` facade bound to the configured archive/database."""
    from polylogue.api import Polylogue

    cfg = _get_config()
    return Polylogue(archive_root=cfg.archive_root, db_path=cfg.db_path)


__all__ = [
    "MCPRole",
    "ServerCallbacks",
    "_async_safe_call",
    "_clamp_limit",
    "_error_json",
    "_extract_fenced_code",
    "_get_config",
    "_get_polylogue",
    "_get_runtime_services",
    "_json_payload",
    "_response_context",
    "_safe_call",
    "_set_runtime_services",
    "role_allows",
]

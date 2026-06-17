"""Shared helpers and runtime-service access for the MCP server."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, overload

from pydantic import BaseModel

from polylogue.archive.query.spec import clamp_query_limit
from polylogue.errors import (
    EmbeddingRetrievalNotReadyError,
    PolylogueError,
    SchemaVersionMismatchError,
)
from polylogue.logging import get_logger
from polylogue.mcp.payloads import MCPErrorPayload, MCPFencedCodeBlock
from polylogue.services import RuntimeServices, build_runtime_services
from polylogue.surfaces.payloads import serialize_surface_payload

if TYPE_CHECKING:
    from polylogue.api import Polylogue
    from polylogue.config import Config

logger = get_logger(__name__)
_runtime_services: RuntimeServices | None = None
TResult = TypeVar("TResult")
MCPRole = Literal["read", "write", "admin"]


class JSONPayloadSerializer(Protocol):
    def __call__(self, payload: BaseModel, *, exclude_none: bool = False) -> str: ...


class ErrorJSONSerializer(Protocol):
    def __call__(self, message: str, **extra: str) -> str: ...


class FencedCodeExtractor(Protocol):
    def __call__(self, text: str, language: str = "") -> list[MCPFencedCodeBlock]: ...


@dataclass(frozen=True)
class ServerCallbacks:
    json_payload: JSONPayloadSerializer
    clamp_limit: Callable[[int | object], int]
    safe_call: Callable[[str, Callable[[], str]], str]
    async_safe_call: Callable[[str, Callable[[], Awaitable[str]]], Awaitable[str]]
    error_json: ErrorJSONSerializer
    get_config: Callable[[], Config]
    get_polylogue: Callable[[], Polylogue]
    extract_fenced_code: FencedCodeExtractor
    role: MCPRole


def role_allows(role: MCPRole, required: MCPRole) -> bool:
    """Return whether an MCP server role allows the required capability."""
    order: dict[MCPRole, int] = {"read": 0, "write": 1, "admin": 2}
    return order[role] >= order[required]


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


def _json_payload(payload: BaseModel, *, exclude_none: bool = False) -> str:
    """Serialize a typed MCP payload with canonical JSON formatting."""
    to_json = getattr(payload, "to_json", None)
    if callable(to_json):
        result = to_json(exclude_none=exclude_none)
        if isinstance(result, str):
            return result
        raise TypeError(f"{type(payload).__name__}.to_json() returned {type(result).__name__}, expected str")
    return serialize_surface_payload(payload, exclude_none=exclude_none)


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
    if isinstance(exc, SchemaVersionMismatchError):
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


@overload
def _safe_call(fn_name: str, fn: Callable[[], str]) -> str: ...


@overload
def _safe_call(fn_name: str, fn: Callable[[], None]) -> str | None: ...


@overload
def _safe_call(fn_name: str, fn: Callable[[], TResult]) -> TResult | str: ...


def _safe_call(fn_name: str, fn: Callable[[], TResult]) -> TResult | str:
    """Call ``fn()`` and return its result, or a typed error JSON on failure.

    Errors are logged server-side and translated through
    :func:`_exception_to_error_json` before being returned. Returning rather
    than raising guarantees that one bad tool call cannot crash the MCP stdio
    loop and disable every other registered tool (#1621). The translated
    payload exposes only the exception class name — never the raw message,
    arguments, or traceback — so error surfaces remain free of secrets.
    """
    try:
        return fn()
    except Exception as exc:
        logger.exception("MCP tool %s failed", fn_name)
        return _exception_to_error_json(fn_name, exc)


@overload
async def _async_safe_call(fn_name: str, fn: Callable[[], Awaitable[str]]) -> str: ...


@overload
async def _async_safe_call(fn_name: str, fn: Callable[[], Awaitable[None]]) -> str | None: ...


@overload
async def _async_safe_call(fn_name: str, fn: Callable[[], Awaitable[TResult]]) -> TResult | str: ...


async def _async_safe_call(fn_name: str, fn: Callable[[], Awaitable[TResult]]) -> TResult | str:
    """Async counterpart of :func:`_safe_call`. Same isolation contract."""
    try:
        return await fn()
    except Exception as exc:
        logger.exception("MCP tool %s failed", fn_name)
        return _exception_to_error_json(fn_name, exc)


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
    "_safe_call",
    "_set_runtime_services",
    "role_allows",
]

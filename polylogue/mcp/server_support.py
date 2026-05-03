"""Shared helpers and runtime-service access for the MCP server."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, overload

from pydantic import BaseModel

from polylogue.logging import get_logger
from polylogue.mcp.payloads import MCPErrorPayload, MCPFencedCodeBlock
from polylogue.operations import ArchiveOperations
from polylogue.protocols import ConversationQueryRuntimeStore, TagStore
from polylogue.services import RuntimeServices, build_runtime_services
from polylogue.surfaces.payloads import serialize_surface_payload

if TYPE_CHECKING:
    from polylogue.config import Config
    from polylogue.storage.sqlite.async_sqlite import SQLiteBackend

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
    get_query_store: Callable[[], ConversationQueryRuntimeStore]
    get_tag_store: Callable[[], TagStore]
    get_backend: Callable[[], SQLiteBackend]
    get_config: Callable[[], Config]
    get_archive_ops: Callable[[], ArchiveOperations]
    extract_fenced_code: FencedCodeExtractor
    role: MCPRole


def role_allows(role: MCPRole, required: MCPRole) -> bool:
    """Return whether an MCP server role allows the required capability."""
    order: dict[MCPRole, int] = {"read": 0, "write": 1, "admin": 2}
    return order[role] >= order[required]


def _exception_error_payload(fn_name: str, exc: Exception) -> MCPErrorPayload:
    """Return a sanitized error payload for unexpected MCP failures."""
    return MCPErrorPayload(
        error="internal MCP tool error",
        code="internal_error",
        detail=type(exc).__name__,
        tool=fn_name,
    )


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
    """Ensure limit is a positive integer capped at 1000, defaulting to 10 on bad input."""
    try:
        if isinstance(limit, bool):
            raise TypeError
        if isinstance(limit, int):
            return max(1, min(limit, 1000))
        if isinstance(limit, float):
            return max(1, min(int(limit), 1000))
        if isinstance(limit, str | bytes | bytearray):
            return max(1, min(int(limit), 1000))
        return max(1, min(int(str(limit)), 1000))
    except (TypeError, ValueError):
        return 10


@overload
def _safe_call(fn_name: str, fn: Callable[[], str]) -> str: ...


@overload
def _safe_call(fn_name: str, fn: Callable[[], None]) -> str: ...


@overload
def _safe_call(fn_name: str, fn: Callable[[], TResult]) -> TResult | str: ...


def _safe_call(fn_name: str, fn: Callable[[], TResult]) -> TResult | str:
    """Call fn() and return its result, or a JSON error dict on exception."""
    try:
        return fn()
    except Exception as exc:
        logger.exception("MCP tool %s failed", fn_name)
        return _json_payload(_exception_error_payload(fn_name, exc), exclude_none=True)


@overload
async def _async_safe_call(fn_name: str, fn: Callable[[], Awaitable[str]]) -> str: ...


@overload
async def _async_safe_call(fn_name: str, fn: Callable[[], Awaitable[None]]) -> str: ...


@overload
async def _async_safe_call(fn_name: str, fn: Callable[[], Awaitable[TResult]]) -> TResult | str: ...


async def _async_safe_call(fn_name: str, fn: Callable[[], Awaitable[TResult]]) -> TResult | str:
    """Async version of _safe_call for async tool handlers."""
    try:
        return await fn()
    except Exception as exc:
        logger.exception("MCP tool %s failed", fn_name)
        return _json_payload(_exception_error_payload(fn_name, exc), exclude_none=True)


def _error_json(message: str, **extra: str) -> str:
    """Return a JSON-encoded error dict."""
    code = extra.pop("code", None)
    return _json_payload(MCPErrorPayload(error=message, code=code, **extra), exclude_none=True)


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


def _get_query_store() -> ConversationQueryRuntimeStore:
    """Return the MCP query/runtime store from the configured runtime services."""
    return _get_runtime_services().get_repository()


def _get_tag_store() -> TagStore:
    """Return the MCP tag/metadata store from the configured runtime services."""
    return _get_runtime_services().get_repository()


def _get_backend() -> SQLiteBackend:
    """Return the configured backend for maintenance surfaces."""
    return _get_runtime_services().get_backend()


def _get_config() -> Config:
    """Return the MCP config from the configured runtime services."""
    return _get_runtime_services().get_config()


__all__ = [
    "MCPRole",
    "ServerCallbacks",
    "_async_safe_call",
    "_clamp_limit",
    "_error_json",
    "_extract_fenced_code",
    "_get_backend",
    "_get_config",
    "_get_query_store",
    "_get_runtime_services",
    "_get_tag_store",
    "_json_payload",
    "_safe_call",
    "_set_runtime_services",
    "role_allows",
]

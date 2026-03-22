"""Shared helpers and runtime-service access for the MCP server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from polylogue.logging import get_logger
from polylogue.mcp.payloads import MCPErrorPayload
from polylogue.services import RuntimeServices, build_runtime_services

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)
_runtime_services: RuntimeServices | None = None


@dataclass(frozen=True)
class ServerCallbacks:
    json_payload: Callable[..., str]
    clamp_limit: Callable[[int | Any], int]
    safe_call: Callable[[str, Any], str]
    async_safe_call: Callable[[str, Any], Awaitable[str]]
    error_json: Callable[..., str]
    get_repo: Callable[[], Any]
    get_config: Callable[[], Any]
    get_archive_ops: Callable[[], Any]
    extract_fenced_code: Callable[[str, str], list[dict[str, str]]]


def _extract_fenced_code(text: str, language: str = "") -> list[dict[str, str]]:
    """Extract fenced code blocks from markdown text."""
    if "```" not in text:
        return []
    blocks = text.split("```")
    results = []
    for i in range(1, len(blocks), 2):
        block = blocks[i]
        lines = block.split("\n", 1)
        block_lang = lines[0].strip() if lines else ""
        code = lines[1] if len(lines) > 1 else block
        if not language or block_lang == language:
            results.append({"language": block_lang, "code": code[:300]})
    return results


def _json_payload(payload: Any, *, exclude_none: bool = False) -> str:
    """Serialize a typed MCP payload with canonical JSON formatting."""
    return payload.to_json(exclude_none=exclude_none)


def _clamp_limit(limit: int | Any) -> int:
    """Ensure limit is a positive integer, defaulting to 10 on bad input."""
    try:
        return max(1, int(limit))
    except (TypeError, ValueError):
        return 10


def _safe_call(fn_name: str, fn: Any) -> str:
    """Call fn() and return its result, or a JSON error dict on exception."""
    try:
        return fn()
    except Exception as exc:
        logger.exception("MCP tool %s failed", fn_name)
        return _json_payload(MCPErrorPayload(error=str(exc), tool=fn_name), exclude_none=True)


async def _async_safe_call(fn_name: str, fn: Any) -> str:
    """Async version of _safe_call for async tool handlers."""
    try:
        return await fn()
    except Exception as exc:
        logger.exception("MCP tool %s failed", fn_name)
        return _json_payload(MCPErrorPayload(error=str(exc), tool=fn_name), exclude_none=True)


def _error_json(message: str, **extra: Any) -> str:
    """Return a JSON-encoded error dict."""
    return _json_payload(MCPErrorPayload(error=message, **extra), exclude_none=True)


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


def _get_repo():
    """Return the MCP repository from the configured runtime services."""
    return _get_runtime_services().get_repository()


def _get_config():
    """Return the MCP config from the configured runtime services."""
    return _get_runtime_services().get_config()


__all__ = [
    "ServerCallbacks",
    "_async_safe_call",
    "_clamp_limit",
    "_error_json",
    "_extract_fenced_code",
    "_get_config",
    "_get_repo",
    "_get_runtime_services",
    "_json_payload",
    "_safe_call",
    "_set_runtime_services",
]

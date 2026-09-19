"""Shared session-list projection contract for CLI and MCP.

Each row declares one session-list projection's public name, facade method,
MCP payload key, and CLI renderer family. Surface dispatchers derive their
vocabularies from these rows instead of maintaining parallel name branches.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

from polylogue.archive.viewport import READ_VIEW_PROFILE_BY_ID


@dataclass(frozen=True, slots=True)
class SessionListProjection:
    """One projection of a session into a bounded list of rows."""

    name: str
    method: str
    payload_key: str
    cli_handler: str


SESSION_LIST_PROJECTIONS: dict[str, SessionListProjection] = {
    projection.name: projection
    for projection in (
        SessionListProjection("events", "get_session_events", "events", cli_handler="events"),
        SessionListProjection("file-edits", "get_file_edits", "file_edits", cli_handler="file-edits"),
        SessionListProjection("agent-policies", "get_agent_policies", "agent_policies", cli_handler="agent-policies"),
        SessionListProjection(
            "web-content", "get_web_content_constructs", "web_content_constructs", cli_handler="web-content"
        ),
    )
}


def session_list_projection_names() -> tuple[str, ...]:
    """Return the current shared projection vocabulary in declaration order."""

    return tuple(SESSION_LIST_PROJECTIONS)


def mcp_read_view_names() -> tuple[str, ...]:
    """Return MCP ``read`` views, with list views derived from the shared table."""

    return ("summary", "topology", "messages", *session_list_projection_names())


def mcp_get_session_projection_names() -> tuple[str, ...]:
    """Return MCP ``get`` projections, with list views derived from the shared table."""

    return ("orchestration", *session_list_projection_names())


def is_mcp_read_view(value: object) -> bool:
    """Return whether ``value`` is one of the runtime-declared MCP read views."""

    return value is None or value in mcp_read_view_names()


def is_mcp_get_session_projection(value: object) -> bool:
    """Return whether ``value`` is one of the runtime-declared session projections."""

    return value is None or value in mcp_get_session_projection_names()


def validate_session_list_projection_cli_contract(cli_handler_ids: Collection[str]) -> None:
    """Reject a projection that MCP can serve but CLI cannot dispatch."""

    missing = sorted(set(SESSION_LIST_PROJECTIONS) - set(cli_handler_ids))
    if missing:
        raise RuntimeError(f"session projections without CLI read handlers: {', '.join(missing)}")


# Compatibility aliases hold the import-time snapshot. Production dispatchers
# call the functions above so the table remains the live vocabulary source.
SESSION_LIST_PROJECTION_NAMES = session_list_projection_names()
MCP_READ_VIEW_NAMES = mcp_read_view_names()
MCP_GET_SESSION_PROJECTION_NAMES = mcp_get_session_projection_names()

# These aliases are evaluated when the module loads, after the table above is
# declared.  A table addition therefore reaches MCP's public Literal schema
# without a duplicate hand-maintained type list. Mypy cannot evaluate a
# runtime tuple expansion as a type expression, but Python and Pydantic can.
MCPReadView: TypeAlias = cast(Any, Literal.__getitem__(mcp_read_view_names())) | None  # type: ignore[valid-type]
MCPGetSessionProjection: TypeAlias = cast(Any, Literal.__getitem__(mcp_get_session_projection_names())) | None  # type: ignore[valid-type]

_UNDECLARED = set(SESSION_LIST_PROJECTIONS) - set(READ_VIEW_PROFILE_BY_ID)
if _UNDECLARED:
    raise RuntimeError(f"session projections not in the shared read-view vocabulary: {sorted(_UNDECLARED)}")


__all__ = [
    "MCP_GET_SESSION_PROJECTION_NAMES",
    "MCP_READ_VIEW_NAMES",
    "MCPGetSessionProjection",
    "MCPReadView",
    "SESSION_LIST_PROJECTION_NAMES",
    "SESSION_LIST_PROJECTIONS",
    "SessionListProjection",
    "is_mcp_get_session_projection",
    "is_mcp_read_view",
    "mcp_get_session_projection_names",
    "mcp_read_view_names",
    "session_list_projection_names",
    "validate_session_list_projection_cli_contract",
]

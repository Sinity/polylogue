"""Shared session-list projection contract for CLI and MCP.

Each row declares one session-list projection's public name, facade method,
MCP payload key, and CLI renderer family. Surface dispatchers derive their
vocabularies from these rows instead of maintaining parallel name branches.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import TypeAlias

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

MCPReadView: TypeAlias = str | None

_UNDECLARED = set(SESSION_LIST_PROJECTIONS) - set(READ_VIEW_PROFILE_BY_ID)
if _UNDECLARED:
    raise RuntimeError(f"session projections not in the shared read-view vocabulary: {sorted(_UNDECLARED)}")


__all__ = [
    "MCP_GET_SESSION_PROJECTION_NAMES",
    "MCP_READ_VIEW_NAMES",
    "MCPReadView",
    "SESSION_LIST_PROJECTION_NAMES",
    "SESSION_LIST_PROJECTIONS",
    "SessionListProjection",
    "mcp_get_session_projection_names",
    "mcp_read_view_names",
    "session_list_projection_names",
    "validate_session_list_projection_cli_contract",
]

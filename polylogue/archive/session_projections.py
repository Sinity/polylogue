"""Shared session-list projection contract for CLI and MCP.

Each row declares one session-list projection's public name, facade method,
MCP payload key, and CLI renderer family. Surface dispatchers derive their
vocabularies from these rows instead of maintaining parallel name branches.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, TypeVar, cast


@dataclass(frozen=True, slots=True)
class SessionListProjection:
    """One projection of a session into a bounded list of rows."""

    name: str
    method: str
    payload_key: str
    cli_handler: str


_SESSION_LIST_RENDERERS = (
    SessionListProjection("events", "get_session_events", "events", cli_handler="events"),
    SessionListProjection("file-edits", "get_file_edits", "file_edits", cli_handler="file-edits"),
    SessionListProjection("agent-policies", "get_agent_policies", "agent_policies", cli_handler="agent-policies"),
    SessionListProjection(
        "web-content", "get_web_content_constructs", "web_content_constructs", cli_handler="web-content"
    ),
    SessionListProjection("materials", "get_session_materials", "materials", cli_handler="materials"),
)

SESSION_LIST_PROJECTIONS: dict[str, SessionListProjection] = {
    projection.name: projection for projection in _SESSION_LIST_RENDERERS
}

Contract = TypeVar("Contract")


def bind_session_list_projection_contracts(
    templates: Mapping[str, Contract], rename: Callable[[Contract, str], Contract]
) -> dict[str, Contract]:
    """Bind public projection names to their existing renderer contracts.

    Templates are implementation families, not an independent public vocabulary.
    Preserve template order and require every row to name a real renderer.
    """
    renderer_ids = {row.cli_handler for row in _SESSION_LIST_RENDERERS}
    for name, row in SESSION_LIST_PROJECTIONS.items():
        if row.name != name or row.cli_handler not in renderer_ids or row.cli_handler not in templates:
            raise RuntimeError(f"invalid session projection contract: {name!r}")
        if name in templates and name not in renderer_ids:
            raise RuntimeError(f"session projection collides with read view: {name!r}")
    result: dict[str, Contract] = {}
    for name, template in templates.items():
        if name not in renderer_ids:
            result[name] = template
        else:
            for row in SESSION_LIST_PROJECTIONS.values():
                if row.cli_handler == name:
                    result[row.name] = rename(template, row.name)
    return result


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


# These aliases are evaluated when the module loads, after the table above is
# declared.  A table addition therefore reaches MCP's public Literal schema
# without a duplicate hand-maintained type list. Mypy cannot evaluate a
# runtime tuple expansion as a type expression, but Python and Pydantic can.
MCPReadView: TypeAlias = cast(Any, Literal.__getitem__(mcp_read_view_names())) | None  # type: ignore[valid-type]
MCPGetSessionProjection: TypeAlias = cast(Any, Literal.__getitem__(mcp_get_session_projection_names())) | None  # type: ignore[valid-type]

__all__ = [
    "MCPGetSessionProjection",
    "MCPReadView",
    "SESSION_LIST_PROJECTIONS",
    "SessionListProjection",
    "bind_session_list_projection_contracts",
    "is_mcp_get_session_projection",
    "is_mcp_read_view",
    "mcp_get_session_projection_names",
    "mcp_read_view_names",
    "session_list_projection_names",
    "validate_session_list_projection_cli_contract",
]

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


SESSION_LIST_PROJECTIONS: dict[str, SessionListProjection] = {
    projection.name: projection
    for projection in (
        SessionListProjection("events", "get_session_events", "events", cli_handler="events"),
        SessionListProjection("file-edits", "get_file_edits", "file_edits", cli_handler="file-edits"),
        SessionListProjection("agent-policies", "get_agent_policies", "agent_policies", cli_handler="agent-policies"),
        SessionListProjection(
            "web-content", "get_web_content_constructs", "web_content_constructs", cli_handler="web-content"
        ),
        SessionListProjection("materials", "get_session_materials", "materials", cli_handler="materials"),
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

    missing = sorted(
        {projection.cli_handler for projection in SESSION_LIST_PROJECTIONS.values()} - set(cli_handler_ids)
    )
    if missing:
        raise RuntimeError(f"session projections without CLI read handlers: {', '.join(missing)}")


_Declaration = TypeVar("_Declaration")


def bind_session_list_projections(
    declarations: Mapping[str, _Declaration],
    *,
    rename: Callable[[str, _Declaration], _Declaration],
) -> dict[str, _Declaration]:
    """Bind public projection names to each surface's renderer-family contract."""
    by_handler: dict[str, list[SessionListProjection]] = {}
    for name, projection in SESSION_LIST_PROJECTIONS.items():
        if name != projection.name:
            raise RuntimeError(f"session projection key {name!r} differs from its declared name")
        by_handler.setdefault(projection.cli_handler, []).append(projection)
    validate_session_list_projection_cli_contract(declarations.keys())
    result: dict[str, _Declaration] = {}
    for family, declaration in declarations.items():
        projections = by_handler.get(family)
        bindings = (
            [(family, declaration)]
            if projections is None
            else [(projection.name, rename(projection.name, declaration)) for projection in projections]
        )
        for name, binding in bindings:
            if name in result:
                raise RuntimeError(f"session projection {name!r} collides with another read view")
            result[name] = binding
    return result


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
    "bind_session_list_projections",
    "is_mcp_get_session_projection",
    "is_mcp_read_view",
    "mcp_get_session_projection_names",
    "mcp_read_view_names",
    "session_list_projection_names",
    "validate_session_list_projection_cli_contract",
]

"""One declared table for the MCP session-list projections (polylogue-mjupn).

``get(ref, projection=X)`` used to carry four hand-written ``if projection ==
"..." and session_id is not None`` branches, each independently re-spelling the
projection name, the archive method it calls, the payload key it writes and the
envelope shape around it -- against a CLI ``read --view`` vocabulary that is
already registry-checked. Nothing tied the two together, so the surfaces could
drift on both the name set and the payload shape for a given name.

The table below is the single declaration, and its names are asserted at import
to be a subset of the shared read-view vocabulary in ``archive/viewport``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from polylogue.archive.viewport import READ_VIEW_PROFILE_BY_ID


@dataclass(frozen=True, slots=True)
class SessionListProjection:
    """One projection of a session into a bounded list of rows."""

    #: The projection name, shared with ``read --view`` and the HTTP surface.
    name: str
    #: The ``Polylogue`` coroutine that returns the rows, or ``None`` when the
    #: session does not exist.
    method: str
    #: The key the rows are published under in the response payload.
    payload_key: str


SESSION_LIST_PROJECTIONS: dict[str, SessionListProjection] = {
    projection.name: projection
    for projection in (
        SessionListProjection("events", "get_session_events", "events"),
        SessionListProjection("file-edits", "get_file_edits", "file_edits"),
        SessionListProjection("agent-policies", "get_agent_policies", "agent_policies"),
        SessionListProjection("web-content", "get_web_content_constructs", "web_content_constructs"),
    )
}

# The table is also the vocabulary owned jointly by MCP ``read`` and ``get``.
# The two routes retain their non-list views, but neither keeps a second copy
# of the session-list names.
SESSION_LIST_PROJECTION_NAMES = tuple(SESSION_LIST_PROJECTIONS)
MCP_READ_VIEW_NAMES = ("summary", "topology", "messages", *SESSION_LIST_PROJECTION_NAMES)
MCP_GET_SESSION_PROJECTION_NAMES = ("orchestration", *SESSION_LIST_PROJECTION_NAMES)

# The vocabulary is table-derived and validated at the operation boundary
# below.  Static type checkers cannot represent a dynamically constructed
# ``Literal`` alias, so keep the annotation broad while preserving the exact
# runtime contract in ``server_cutover``.
MCPReadView: TypeAlias = str | None

_UNDECLARED = set(SESSION_LIST_PROJECTIONS) - set(READ_VIEW_PROFILE_BY_ID)
if _UNDECLARED:
    raise RuntimeError(f"MCP session projections not in the shared read-view vocabulary: {sorted(_UNDECLARED)}")


__all__ = [
    "MCP_GET_SESSION_PROJECTION_NAMES",
    "MCP_READ_VIEW_NAMES",
    "MCPReadView",
    "SESSION_LIST_PROJECTION_NAMES",
    "SESSION_LIST_PROJECTIONS",
    "SessionListProjection",
]

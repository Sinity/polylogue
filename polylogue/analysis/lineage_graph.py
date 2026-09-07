"""Compact seed-relative lineage graph (polylogue-4ts.9).

A lineage *lookup* answers "which sessions are related". These models answer
"how", carrying the relationship meaning a row needs to be actionable: the
seed-relative role of every node and edge, the link type, whether the child
inherits the parent's leading prefix or was spawned fresh, the branch point,
how the edge was derived and with what confidence, whether it resolved, and
how many of a session's messages are its own rather than inherited.

The graph is a projection over ``sessions`` and ``session_links``. It is
deliberately separate from transcript composition: nothing here reads message
bodies, so a 130-session family is a bounded read.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from polylogue.core.types import MessageId, SessionId

#: Default node/edge window. A caller asking for a family it cannot render is
#: better served by a first page plus ``has_more`` than by a truncated total.
DEFAULT_LINEAGE_PAGE_LIMIT = 50


class LineageNodeRole(str, Enum):
    """A node's relationship to the seed session.

    ``FAMILY`` is a member of the same rooted tree that is none of the others
    (a cousin, an uncle's descendant): related, but not on the seed's own
    ancestry or descent path.
    """

    SEED = "seed"
    ANCESTOR = "ancestor"
    DESCENDANT = "descendant"
    SIBLING = "sibling"
    FAMILY = "family"


class LineageEdgeRole(str, Enum):
    """An edge's relationship to the seed session."""

    SEED_PARENT = "seed-parent"
    SEED_CHILD = "seed-child"
    FAMILY = "family"


class LineageEdgeResolution(str, Enum):
    """Whether an edge names a stored session, and how a resolver marked it.

    ``UNRESOLVED`` is an edge whose destination is a provider-native id no
    stored session carries — retained so a late-arriving parent reconciles
    deterministically. ``QUARANTINED`` and ``AUTHORITY_CONTRADICTED`` edges are
    excluded from composition; they appear here because hiding them would make
    a page silently lossy.
    """

    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    REPAIRED = "repaired"
    QUARANTINED = "quarantined"
    AUTHORITY_CONTRADICTED = "authority_contradicted"


class LineageAccountingStatus(str, Enum):
    """Whether unique/inherited counts are known to match composition."""

    KNOWN = "known"
    UNKNOWN = "unknown"


class LineageMessageAccounting(BaseModel):
    """How many of a session's composed messages are its own.

    ``unique`` is what the session stores; ``inherited`` is the composed
    prefix it replays from its parent. ``UNKNOWN`` means composition cannot be
    reproduced from stored rows (a dangling branch point, a chain past the
    depth limit), and then both counts are ``None`` rather than a plausible
    number.
    """

    model_config = ConfigDict(frozen=True)

    status: LineageAccountingStatus = LineageAccountingStatus.UNKNOWN
    unique: int | None = None
    inherited: int | None = None
    reason: str | None = None

    @property
    def composed(self) -> int | None:
        if self.unique is None or self.inherited is None:
            return None
        return self.unique + self.inherited


class CompactLineageNode(BaseModel):
    """One session in the compact graph, positioned relative to the seed."""

    model_config = ConfigDict(frozen=True)

    session_id: SessionId
    origin: str = ""
    title: str | None = None
    role: LineageNodeRole = LineageNodeRole.FAMILY
    depth_from_root: int = 0
    #: Signed distance from the seed: negative for ancestors, positive for
    #: descendants, ``None`` where the seed is neither (siblings, cousins).
    depth_from_seed: int | None = None
    is_root: bool = False
    is_seed: bool = False
    branch_type: str | None = None
    accounting: LineageMessageAccounting = Field(default_factory=LineageMessageAccounting)


class CompactLineageEdge(BaseModel):
    """One parent→child relationship, with the meaning a row needs."""

    model_config = ConfigDict(frozen=True)

    child_id: SessionId
    parent_id: SessionId | None = None
    parent_native_id: str | None = None
    parent_origin: str | None = None
    link_type: str = "unknown"
    inheritance: str | None = None
    branch_point_message_id: MessageId | None = None
    method: str | None = None
    confidence: float = 1.0
    resolution: LineageEdgeResolution = LineageEdgeResolution.UNRESOLVED
    role: LineageEdgeRole = LineageEdgeRole.FAMILY


class LineagePage(BaseModel):
    """One stable window over an independently paged collection."""

    model_config = ConfigDict(frozen=True)

    offset: int = 0
    limit: int | None = None
    returned: int = 0
    total: int = 0

    @property
    def has_more(self) -> bool:
        return self.offset + self.returned < self.total


class CompactLineageGraph(BaseModel):
    """Seed-relative lineage graph with nodes and edges paged independently.

    The seed node is always present regardless of sort or page: a page that
    omitted the session the caller asked about would make every relationship
    on it unreadable.
    """

    model_config = ConfigDict(frozen=True)

    seed_id: SessionId
    root_id: SessionId
    nodes: tuple[CompactLineageNode, ...] = ()
    edges: tuple[CompactLineageEdge, ...] = ()
    node_page: LineagePage = Field(default_factory=LineagePage)
    edge_page: LineagePage = Field(default_factory=LineagePage)
    cycle_detected: bool = False

    def seed_node(self) -> CompactLineageNode:
        """Return the seed's node. Always present by construction."""

        return next(node for node in self.nodes if node.is_seed)


__all__ = [
    "DEFAULT_LINEAGE_PAGE_LIMIT",
    "CompactLineageEdge",
    "CompactLineageGraph",
    "CompactLineageNode",
    "LineageAccountingStatus",
    "LineageEdgeResolution",
    "LineageEdgeRole",
    "LineageMessageAccounting",
    "LineageNodeRole",
    "LineagePage",
]

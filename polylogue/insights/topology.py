"""Session topology read model — session lineage graph.

Materializes the durable lineage graph that relates sessions through
resume/branch/subagent/sidechain forks. Built over the existing archive
substrate (``sessions.parent_session_id`` + ``branch_type`` +
``messages.parent_message_id``) rather than a new DDL bundle: the topology is
derivable from the canonical parent edges already persisted at ingest time.

The graph answers the classes of questions called out in issue #866:

- parent and root membership for an arbitrary session;
- ancestors / descendants / siblings;
- edge classification (continuation, sidechain, fork, subagent);
- unresolved native parent edges from ``session_links`` rows that did not
  resolve to a stored session;
- cycle detection — a cycle is structurally invalid and is surfaced through
  ``SessionTopology.cycle_detected``.

This module defines only the typed models; the derivation logic lives in
``polylogue.storage.insights.topology`` and is exposed through the
``RepositoryInsightTopologyReadMixin``.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from polylogue.archive.session.branch_type import BranchType
from polylogue.types import SessionId


class TopologyEdgeKind(str, Enum):
    """Edge classification for the session lineage graph.

    Mirrors :class:`BranchType` for the resolved-parent case and adds two
    extra states that exist only at the topology layer:

    - ``UNRESOLVED_NATIVE`` — a native parent ID was observed on the child
      through ``session_links`` but no session with that ID is present in the
      archive. The edge is preserved so late-arriving parents can be reconciled
      deterministically.
    - ``UNKNOWN`` — a parent ID resolves to a stored session but the
      child carries no ``branch_type``. Treated as a structural edge
      without semantic classification.
    """

    CONTINUATION = "continuation"
    SIDECHAIN = "sidechain"
    FORK = "fork"
    SUBAGENT = "subagent"
    UNKNOWN = "unknown"
    UNRESOLVED_NATIVE = "unresolved_native"

    @classmethod
    def from_branch_type(cls, branch_type: BranchType | None) -> TopologyEdgeKind:
        """Promote a ``BranchType`` into the topology edge taxonomy."""

        if branch_type is None:
            return cls.UNKNOWN
        return cls(branch_type.value)


class SessionRef(BaseModel):
    """Typed reference to a session, returned by topology read APIs.

    Carries the minimal projection that lineage consumers need (id +
    provider + title + depth) without forcing them to hydrate the full
    :class:`~polylogue.archive.session.domain_models.Session`. Slice D
    of #866: surfaces such as the MCP topology tool, future reader panes,
    and context packs consume this ref instead of re-walking parent
    pointers.
    """

    model_config = ConfigDict(frozen=True)

    session_id: SessionId
    source_name: str = ""
    title: str | None = None
    depth: int = 0
    """Distance from the topology root (root has depth 0)."""


class TopologyNode(BaseModel):
    """One session in the topology graph."""

    model_config = ConfigDict(frozen=True)

    session_id: SessionId
    source_name: str = ""
    title: str | None = None
    depth: int = 0
    """Distance from the topology root (root has depth 0)."""

    is_root: bool = False

    def as_ref(self) -> SessionRef:
        """Project this node into a :class:`SessionRef`."""

        return SessionRef(
            session_id=self.session_id,
            source_name=self.source_name,
            title=self.title,
            depth=self.depth,
        )


class TopologyEdge(BaseModel):
    """One parent → child edge in the topology graph.

    For unresolved native edges, ``parent_id`` is ``None`` and
    ``parent_native_id`` carries the provider-native pointer that could not
    be reconciled to a stored session.
    """

    model_config = ConfigDict(frozen=True)

    child_id: SessionId
    parent_id: SessionId | None = None
    parent_native_id: str | None = None
    kind: TopologyEdgeKind
    resolved: bool = True


class SessionTopology(BaseModel):
    """Resolved lineage graph rooted at one session.

    The graph is always presented as a rooted tree-with-back-edges view: a
    ``root_id`` plus every descendant reachable through resolved
    parent/child relationships. Unresolved native edges and detected
    cycles are reported separately so consumers do not have to re-walk the
    underlying tables.

    Attributes:
        target_id: the session the topology was requested for.
        root_id: the topology root (the ancestor with no resolved parent).
        nodes: every node in the rooted subtree, ordered by BFS from root.
        edges: resolved parent/child edges plus unresolved native edges
            observed inside this rooted subtree.
        cycle_detected: ``True`` when the ancestry walk revisited an
            already-seen session. Cycles are surfaced rather than
            silently broken so the operator can quarantine the archive
            slice (see #866 acceptance criteria).
    """

    model_config = ConfigDict(frozen=True)

    target_id: SessionId
    root_id: SessionId
    nodes: tuple[TopologyNode, ...] = Field(default_factory=tuple)
    edges: tuple[TopologyEdge, ...] = Field(default_factory=tuple)
    cycle_detected: bool = False

    def node_ids(self) -> tuple[SessionId, ...]:
        return tuple(node.session_id for node in self.nodes)

    def ancestors(self, session_id: str) -> tuple[SessionId, ...]:
        """Return ancestors of ``session_id`` ordered root → parent."""

        parent_lookup = {edge.child_id: edge.parent_id for edge in self.edges if edge.resolved and edge.parent_id}
        chain: list[SessionId] = []
        seen: set[str] = set()
        current = SessionId(str(session_id))
        while current in parent_lookup:
            parent = parent_lookup[current]
            if str(parent) in seen:
                break
            seen.add(str(parent))
            chain.append(parent)
            current = parent
        return tuple(reversed(chain))

    def descendants(self, session_id: str) -> tuple[SessionId, ...]:
        """Return descendants of ``session_id`` in BFS order."""

        children_lookup: dict[str, list[SessionId]] = {}
        for edge in self.edges:
            if not edge.resolved or edge.parent_id is None:
                continue
            children_lookup.setdefault(str(edge.parent_id), []).append(edge.child_id)
        out: list[SessionId] = []
        queue: list[str] = [str(session_id)]
        seen: set[str] = {str(session_id)}
        while queue:
            current = queue.pop(0)
            for child in children_lookup.get(current, ()):
                child_key = str(child)
                if child_key in seen:
                    continue
                seen.add(child_key)
                out.append(child)
                queue.append(child_key)
        return tuple(out)

    def siblings(self, session_id: str) -> tuple[SessionId, ...]:
        """Return other children of ``session_id``'s resolved parent."""

        parent_of: SessionId | None = None
        for edge in self.edges:
            if str(edge.child_id) == str(session_id) and edge.resolved:
                parent_of = edge.parent_id
                break
        if parent_of is None:
            return ()
        return tuple(
            edge.child_id
            for edge in self.edges
            if edge.resolved
            and edge.parent_id is not None
            and str(edge.parent_id) == str(parent_of)
            and str(edge.child_id) != str(session_id)
        )

    def unresolved_edges(self) -> tuple[TopologyEdge, ...]:
        """Return only the unresolved (native-pointer-only) edges."""

        return tuple(edge for edge in self.edges if not edge.resolved)

    # ------------------------------------------------------------------
    # Typed projection surface (#1261 / #866 slice D)
    #
    # These methods return :class:`SessionRef` lists rather than raw
    # ``SessionId`` tuples so callers (Python API, MCP, future reader
    # panes) get title/provider context without re-fetching session
    # rows. They are layered on top of the existing structural helpers so
    # ordering semantics stay identical.
    # ------------------------------------------------------------------

    def _node_index(self) -> dict[str, TopologyNode]:
        return {str(node.session_id): node for node in self.nodes}

    def _refs_for(self, ids: tuple[SessionId, ...]) -> list[SessionRef]:
        index = self._node_index()
        refs: list[SessionRef] = []
        for conv_id in ids:
            node = index.get(str(conv_id))
            if node is not None:
                refs.append(node.as_ref())
        return refs

    def ancestor_refs(self, session_id: str) -> list[SessionRef]:
        """Return ancestor :class:`SessionRef`s ordered root → parent."""

        return self._refs_for(self.ancestors(session_id))

    def descendant_refs(self, session_id: str) -> list[SessionRef]:
        """Return descendant :class:`SessionRef`s in BFS order."""

        return self._refs_for(self.descendants(session_id))

    def sibling_refs(self, session_id: str) -> list[SessionRef]:
        """Return sibling :class:`SessionRef`s (shared resolved parent)."""

        return self._refs_for(self.siblings(session_id))

    def thread_refs(self, session_id: str) -> list[SessionRef]:
        """Return the full lineage thread ordered ancestors → self → descendants.

        Use this when the consumer wants a single linearization of the
        whole sub-lineage rooted at the topology root and reaching every
        descendant of ``session_id``. The self node is included
        between the ancestors and the descendants — this preserves the
        topological order ancestors → self → descendants that
        chronological readers expect.
        """

        index = self._node_index()
        target_node = index.get(str(session_id))
        if target_node is None:
            return []
        thread: list[SessionRef] = []
        thread.extend(self._refs_for(self.ancestors(session_id)))
        thread.append(target_node.as_ref())
        thread.extend(self._refs_for(self.descendants(session_id)))
        return thread


class LogicalSession(BaseModel):
    """Compact read-pull view of one logical session lineage."""

    model_config = ConfigDict(frozen=True)

    session_id: SessionId
    root_id: SessionId
    thread: tuple[SessionRef, ...] = Field(default_factory=tuple)
    siblings: tuple[SessionRef, ...] = Field(default_factory=tuple)
    descendants: tuple[SessionRef, ...] = Field(default_factory=tuple)
    cycle_detected: bool = False


__all__ = [
    "SessionRef",
    "LogicalSession",
    "SessionTopology",
    "TopologyEdge",
    "TopologyEdgeKind",
    "TopologyNode",
]

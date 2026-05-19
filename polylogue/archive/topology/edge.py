"""Closed enums and the runtime record for cross-conversation topology edges.

Drop site context
-----------------
Before this module existed, ``polylogue/pipeline/prepare_enrichment.py``
silently dropped a parsed ``parent_conversation_provider_id`` whenever the
referenced parent conversation was not present in the in-batch
``PrepareCache.known_ids`` set (or in the database). That meant sidechain,
subagent, continuation, and branch edges were lost whenever the parent had
not yet been ingested (out-of-order ingestion) or had been hard-deleted.

The ``topology_edges`` table — populated from ``TopologyEdgeRecord`` rows
built at ingest time — preserves every parent reference, including those that
cannot be resolved at write time. A subsequent ingest of the parent flips the
edge from ``unresolved`` to ``resolved`` via
``resolve_topology_edges_for_conversation``.

The pre-existing fast-path (``conversations.parent_conversation_id`` set when
the parent is known at child-write time) is preserved unchanged; the topology
edge table is an *additional* durable record that always carries the original
provider-native parent id.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from polylogue.archive.conversation.branch_type import BranchType
from polylogue.types import ConversationId


class TopologyEdgeType(str, Enum):
    """Closed vocabulary of cross-conversation edge kinds (#1258, #866).

    ``CONTINUATION`` / ``SIDECHAIN`` / ``SUBAGENT`` / ``FORK`` mirror the
    existing ``BranchType`` enum. ``BRANCH`` is reserved for ChatGPT-style
    explicit branch points whose parent is referenced but absent.
    ``RESUME`` is reserved for Codex/Antigravity ``resume`` semantics where
    the new session is a logical continuation of an explicitly closed parent.
    ``REPAIRED`` denotes an edge whose parent row was hard-deleted but whose
    historical existence has been re-established by an evidence-driven
    repair pass.

    Slice A (#1258) emits ``CONTINUATION`` / ``SIDECHAIN`` / ``SUBAGENT`` /
    ``FORK`` / ``BRANCH`` and never produces ``RESUME`` or ``REPAIRED`` from
    the parser path; those values exist so that future repair / cross-source
    resolvers can land without another enum change.
    """

    CONTINUATION = "continuation"
    SIDECHAIN = "sidechain"
    SUBAGENT = "subagent"
    BRANCH = "branch"
    FORK = "fork"
    RESUME = "resume"
    REPAIRED = "repaired"

    def __str__(self) -> str:
        return self.value


class TopologyEdgeStatus(str, Enum):
    """Closed lifecycle vocabulary for a topology edge.

    ``QUARANTINED`` (#1260 / #866 slice C) marks an edge whose resolution
    would introduce a cycle (e.g. A → B → A) into the
    ``conversations.parent_conversation_id`` fast-path graph. The resolver
    refuses to backfill the cycle-creating edge; the edge row is preserved
    so the operator can audit the cycle path that was rejected. The cycle
    path is recorded in ``raw_evidence`` as JSON.
    """

    UNRESOLVED = "unresolved"
    RESOLVED = "resolved"
    REPAIRED = "repaired"
    QUARANTINED = "quarantined"

    def __str__(self) -> str:
        return self.value


def branch_type_to_edge_type(
    branch_type: BranchType | None,
    *,
    default: TopologyEdgeType = TopologyEdgeType.CONTINUATION,
) -> TopologyEdgeType:
    """Map a parsed ``BranchType`` to the canonical ``TopologyEdgeType``.

    Centralized so future grep-locality stays cheap. ``BranchType`` carries
    four values today (continuation, sidechain, fork, subagent); the mapping
    is the identity on those.  When ``branch_type`` is ``None`` (parser
    asserted a parent but did not classify the relationship), the fallback
    is ``CONTINUATION`` — the conservative default that matches how
    ``conversations.parent_conversation_id`` semantically behaves for
    unclassified continuations.
    """
    if branch_type is None:
        return default
    return {
        BranchType.CONTINUATION: TopologyEdgeType.CONTINUATION,
        BranchType.SIDECHAIN: TopologyEdgeType.SIDECHAIN,
        BranchType.SUBAGENT: TopologyEdgeType.SUBAGENT,
        BranchType.FORK: TopologyEdgeType.FORK,
    }[branch_type]


def _now_isoformat() -> str:
    return datetime.now(timezone.utc).isoformat()


class TopologyEdgeRecord(BaseModel):
    """Runtime row for the ``topology_edges`` table.

    Identity (``src_conversation_id``, ``dst_provider_native_id``, ``edge_type``)
    is enforced as a SQL ``UNIQUE`` constraint so re-ingesting the same child
    twice is idempotent — the second ingest upserts the same row.
    """

    model_config = ConfigDict(use_enum_values=False)

    src_conversation_id: ConversationId
    dst_provider_native_id: str
    dst_provider_name: str
    edge_type: TopologyEdgeType
    resolved_dst_conversation_id: ConversationId | None = None
    raw_evidence: str | None = None
    confidence: float = 1.0
    status: TopologyEdgeStatus = TopologyEdgeStatus.UNRESOLVED
    observed_at: str = Field(default_factory=_now_isoformat)
    resolved_at: str | None = None

    @field_validator("src_conversation_id", "dst_provider_native_id", "dst_provider_name")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("topology edge identity column cannot be empty")
        return value

    @field_validator("confidence")
    @classmethod
    def _confidence_bounded(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        return value


__all__ = [
    "TopologyEdgeRecord",
    "TopologyEdgeStatus",
    "TopologyEdgeType",
    "branch_type_to_edge_type",
]

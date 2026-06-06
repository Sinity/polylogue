"""Closed enums and the runtime record for cross-session topology edges.

Drop site context
-----------------
Before this module existed, ``polylogue/pipeline/prepare_enrichment.py``
silently dropped a parsed ``parent_session_provider_id`` whenever the
referenced parent session was not present in the in-batch
``PrepareCache.known_ids`` set (or in the database). That meant sidechain,
subagent, continuation, and branch edges were lost whenever the parent had
not yet been ingested (out-of-order ingestion) or had been hard-deleted.

The ``topology_edges`` table — populated from ``TopologyEdgeRecord`` rows
built at ingest time — preserves every parent reference, including those that
cannot be resolved at write time. A subsequent ingest of the parent flips the
edge from ``unresolved`` to ``resolved`` via
``resolve_topology_edges_for_session``.

The pre-existing fast-path (``sessions.parent_session_id`` set when
the parent is known at child-write time) is preserved unchanged; the topology
edge table is an *additional* durable record that always carries the original
provider-native parent id.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator

from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import LinkType as TopologyEdgeType
from polylogue.core.enums import TopologyEdgeStatus
from polylogue.types import SessionId


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
    ``sessions.parent_session_id`` semantically behaves for
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

    Identity (``src_session_id``, ``dst_provider_native_id``, ``edge_type``)
    is enforced as a SQL ``UNIQUE`` constraint so re-ingesting the same child
    twice is idempotent — the second ingest upserts the same row.
    """

    model_config = ConfigDict(use_enum_values=False)

    src_session_id: SessionId
    dst_provider_native_id: str
    dst_provider_name: str
    edge_type: TopologyEdgeType
    resolved_dst_session_id: SessionId | None = None
    raw_evidence: str | None = None
    confidence: float = 1.0
    status: TopologyEdgeStatus = TopologyEdgeStatus.UNRESOLVED
    observed_at: str = Field(default_factory=_now_isoformat)
    resolved_at: str | None = None

    @field_validator("src_session_id", "dst_provider_native_id", "dst_provider_name")
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

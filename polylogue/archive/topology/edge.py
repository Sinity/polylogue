"""Closed enums and runtime record for cross-session session links."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator

from polylogue.archive.session.branch_type import BranchType
from polylogue.core.enums import LinkType as TopologyEdgeType
from polylogue.core.enums import Origin, TopologyEdgeStatus
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
    """Runtime row for the ``session_links`` table.

    Identity is the natural unresolved assertion:
    ``(src_session_id, dst_origin, dst_native_id, link_type)``.
    """

    model_config = ConfigDict(use_enum_values=False)

    src_session_id: SessionId
    dst_origin: Origin
    dst_native_id: str
    link_type: TopologyEdgeType
    resolved_dst_session_id: SessionId | None = None
    evidence_json: str = "[]"
    confidence: float = 1.0
    status: TopologyEdgeStatus = TopologyEdgeStatus.UNRESOLVED
    observed_at: str = Field(default_factory=_now_isoformat)
    resolved_at: str | None = None

    @field_validator("src_session_id", "dst_native_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("session link identity column cannot be empty")
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

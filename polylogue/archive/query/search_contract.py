"""Canonical retrieval-lane availability and result metadata.

This module is deliberately small: provider setup and lane outcome are query
semantics, not surface policy.  Every surface can therefore distinguish an
executed empty lane from a lane that was never executable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

    from polylogue.config import Config
    from polylogue.core.protocols import VectorProvider

LaneName = Literal["text", "action", "vector"]
LaneFailureKind = Literal["unavailable", "construction_failed", "execution_failed"]


@dataclass(frozen=True, slots=True)
class LaneFailure:
    lane: LaneName
    kind: LaneFailureKind
    reason: str
    advisory: str


@dataclass(frozen=True, slots=True)
class SearchExecution:
    requested_lanes: tuple[LaneName, ...]
    executed_lanes: tuple[LaneName, ...]
    unavailable_lanes: tuple[LaneName, ...] = ()
    failed_lanes: tuple[LaneFailure, ...] = ()
    lane_ranks: dict[str, dict[str, int | None]] | None = None

    @property
    def degraded(self) -> bool:
        return bool(self.unavailable_lanes or self.failed_lanes)

    @property
    def advisories(self) -> tuple[str, ...]:
        return tuple(f.advisory for f in self.failed_lanes) + tuple(
            f"{lane} retrieval is unavailable; configure embeddings and retry" for lane in self.unavailable_lanes
        )


def resolve_vector_provider(
    config: Config | None,
    *,
    archive_root: Path,
    provider: VectorProvider | None = None,
) -> tuple[VectorProvider | None, LaneFailure | None]:
    """Resolve the vector backend once, retaining construction evidence."""
    if provider is not None:
        return provider, None
    from polylogue.storage.search_providers import create_vector_provider

    try:
        resolved = create_vector_provider(
            config,
            db_path=archive_root / "embeddings.db",
            archive_root=archive_root,
        )
    except Exception as exc:
        return None, LaneFailure(
            "vector",
            "construction_failed",
            type(exc).__name__,
            "vector provider construction failed; repair embedding configuration and retry",
        )
    if resolved is None:
        return None, LaneFailure(
            "vector",
            "unavailable",
            "no configured/constructible vector backend",
            "vector retrieval is unavailable; configure Voyage/sqlite-vec and retry",
        )
    return resolved, None


__all__ = ["LaneFailure", "SearchExecution", "resolve_vector_provider"]

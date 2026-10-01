"""Cold-build promotion must not drop a session the active index serves.

An explicit ``polylogued run --cold-build-index`` builds a candidate over a
populated active generation. The candidate is filled by ordinary intake over
the configured source roots, so a session the active generation serves from a
raw outside that baseline (a manual import, a transcript deleted or rotated out
of its source root) never reaches it, and raw materialization stays suspended
while the build is unsettled. Promoting such a candidate would drop that session
from every read with no route re-deriving it. The daemon therefore promotes
through :func:`promote_cold_build_covering_active_index`, which refuses with
:class:`ColdBuildCoverageError` while any such session is missing.

This lives outside ``sources/live/cold_build.py`` on purpose: that module is in
the derived index identity closure, and a daemon settlement policy must not
move the index schema identity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.logging import WARNING, emit

if TYPE_CHECKING:
    from polylogue.sources.live.cold_build import ColdBuildGeneration
    from polylogue.storage.index_generation import IndexGeneration, PreparedIndexPromotion

__all__ = [
    "ColdBuildCoverageError",
    "promote_cold_build_covering_active_index",
]


class ColdBuildCoverageError(RuntimeError):
    """The candidate lacks sessions the active generation serves from retained raws.

    The refusal leaves the candidate inactive and the active generation
    serving. Settlement re-evaluates it when the candidate or the source
    evidence changes.
    """

    def __init__(self, *, missing_count: int, first_missing_session_id: str) -> None:
        self.missing_count = missing_count
        self.first_missing_session_id = first_missing_session_id
        super().__init__(
            f"cold-build candidate lacks {missing_count} session(s) the active index serves from retained raws "
            f"(first: {first_missing_session_id})"
        )


def promote_cold_build_covering_active_index(
    generation: ColdBuildGeneration, prepared: PreparedIndexPromotion
) -> IndexGeneration:
    """Promote with the off-gate coverage/reference proof still retained."""
    if prepared.missing_session_count and prepared.first_missing_session_id is not None:
        emit(
            "daemon.cold_build.coverage_refused",
            level=WARNING,
            outcome="degraded",
            reason="active_coverage_incomplete",
            generation_id=generation.generation_id,
            sessions=prepared.missing_session_count,
        )
        raise ColdBuildCoverageError(
            missing_count=prepared.missing_session_count,
            first_missing_session_id=prepared.first_missing_session_id,
        )
    return generation.promote_prepared(prepared)

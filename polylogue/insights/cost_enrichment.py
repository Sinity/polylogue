"""Cost-insight enrichment.

The archive ``index.db`` ``session_profiles`` row stores only a scalar
``cost_usd`` plus a coarse provenance token. That is enough to answer "how
much did this session cost?" but it drops the model identity, catalog
pricing basis, per-model breakdown, and the precise ``missing_reasons`` that
the public :class:`~polylogue.insights.archive.SessionCostInsight` contract
exposes (the ``insights costs`` ``model``/``normalized_model`` fields and the
``insights cost-rollups`` grouping both key on those).

Rather than reconstruct that detail from the lossy scalar, this module
re-derives the full :class:`CostEstimatePayload` from the session's actual
session via :func:`estimate_session_cost`, then grafts it onto the
archive-built insight so the durable provenance, title, and timestamps are
preserved. This keeps cost normalization in the insight layer instead of the
storage read path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polylogue.archive.semantic.pricing import estimate_session_cost
from polylogue.insights.archive import SessionCostInsight

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore


def enrich_session_cost_insight(
    archive: ArchiveStore,
    insight: SessionCostInsight,
) -> SessionCostInsight:
    """Return ``insight`` with its estimate re-derived from the session.

    Falls back to the original insight when the session cannot be read (e.g.
    it was deleted between the list query and this read), so a transient gap
    never turns a cost listing into an error.

    Re-derivation augments the lossy stored scalar (model identity, catalog
    basis, missing-reason taxonomy). But a confident stored estimate
    (``exact`` / ``priced``, already materialized into ``session_profiles``) is
    never downgraded: when the session carries no cost evidence the
    re-derived estimate would be ``unavailable``, so the stored estimate is kept.
    """
    from polylogue.api.archive import _archive_session_to_session

    try:
        session = _archive_session_to_session(archive.read_session(insight.session_id))
    except KeyError:
        return insight
    estimate = estimate_session_cost(session)
    confident = ("exact", "priced")
    if insight.estimate.status in confident and estimate.status not in confident:
        return insight
    return insight.model_copy(
        update={
            "estimate": estimate.model_copy(
                update={
                    "origin": insight.estimate.origin,
                    "session_id": insight.session_id,
                }
            )
        }
    )


def enrich_session_cost_insights(
    archive: ArchiveStore,
    insights: list[SessionCostInsight],
) -> list[SessionCostInsight]:
    """Enrich every archive cost insight with a full re-derived estimate."""
    return [enrich_session_cost_insight(archive, insight) for insight in insights]

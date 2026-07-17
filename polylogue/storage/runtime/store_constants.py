"""Storage-layer materializer and schema version constants."""

from __future__ import annotations

SESSION_EVENT_MATERIALIZER_VERSION = 1
SESSION_INSIGHT_MATERIALIZER_VERSION = 14
SESSION_INFERENCE_VERSION = 2
SESSION_INFERENCE_FAMILY = "heuristic_session_semantics"
SESSION_ENRICHMENT_VERSION = 1
SESSION_ENRICHMENT_FAMILY = "scored_session_enrichment"

# Iterative lineage composition uses this only as a runaway backstop; cycle
# detection is the primary guard. Keep sync-writer and async-reader composition
# aligned so a valid deep branch cannot be normalized differently on write and
# read merely because one path stopped walking ancestors earlier.
LINEAGE_ITERATIVE_DEPTH_LIMIT = 1024


__all__ = [
    "LINEAGE_ITERATIVE_DEPTH_LIMIT",
    "SESSION_EVENT_MATERIALIZER_VERSION",
    "SESSION_ENRICHMENT_FAMILY",
    "SESSION_ENRICHMENT_VERSION",
    "SESSION_INFERENCE_FAMILY",
    "SESSION_INFERENCE_VERSION",
    "SESSION_INSIGHT_MATERIALIZER_VERSION",
]

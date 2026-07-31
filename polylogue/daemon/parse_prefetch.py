"""Daemon-facing re-export of the shared off-writer-hold parse-stage engine.

polylogue-m6tp phase (a); relocated to substrate at
``polylogue.sources.census_parse_stage`` (polylogue-czq2) so the offline
rebuild engine (``maintenance/rebuild_index.py``) can consume the exact same
``CensusParseStage``/``RawParsePrefetchCache`` machinery this module used to
own exclusively, instead of only the daemon's own bulk-rebuild loop
(``daemon/bulk_rebuild.py``) ever getting a warmed prefetch cache while the
offline CLI and the daemon's own ``/api/maintenance/rebuild-index`` HTTP
route silently threaded ``prefetch_cache=None``.

Every name below is the SAME object as its ``polylogue.sources.
census_parse_stage`` counterpart -- this module adds no behavior, only
preserves the daemon's existing import path
(``from polylogue.daemon.parse_prefetch import DaemonParseStage``) so every
pre-existing daemon call site and test keeps working unchanged.
"""

from __future__ import annotations

from polylogue.sources.census_parse_stage import (
    CensusParseStage as DaemonParseStage,
)
from polylogue.sources.census_parse_stage import (
    daemon_parse_stage_max_cached_tree_bytes,
    daemon_parse_stage_max_inflight_bytes,
    daemon_parse_stage_warm_timeout_seconds,
    daemon_parse_stage_worker_count,
    estimate_parsed_tree_bytes,
)

__all__ = [
    "DaemonParseStage",
    "daemon_parse_stage_max_cached_tree_bytes",
    "daemon_parse_stage_max_inflight_bytes",
    "daemon_parse_stage_warm_timeout_seconds",
    "daemon_parse_stage_worker_count",
    "estimate_parsed_tree_bytes",
]

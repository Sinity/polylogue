"""Daemon-facing re-export of the shared off-writer-hold parse-stage engine.

The implementation lives in ``polylogue.sources.census_parse_stage``.

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

"""Shared models for ingest planning."""

from __future__ import annotations

from dataclasses import dataclass

from polylogue.storage.state_views import PlanResult


@dataclass
class IngestPlan:
    """Internal plan consumed by runtime execution."""

    summary: PlanResult
    validate_raw_ids: list[str]
    parse_ready_raw_ids: list[str]


__all__ = ["IngestPlan"]

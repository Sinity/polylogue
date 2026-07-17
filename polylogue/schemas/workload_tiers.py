"""Stable names for schema-derived workload scale and selectivity projections."""

from __future__ import annotations

from enum import Enum


class WorkloadScaleTier(str, Enum):
    """Named archive projections with stable activation semantics."""

    CI_ACTIVATION = "ci-activation"
    ARCHIVE_1X = "archive-1x"
    ARCHIVE_10X = "archive-10x"


class WorkloadSelectivityTier(str, Enum):
    """Named cardinality targets resolved from an archive workload profile."""

    EXACT_ONE = "exact-one"
    OBSERVED_P50 = "observed-p50"
    OBSERVED_P99 = "observed-p99"


__all__ = ["WorkloadScaleTier", "WorkloadSelectivityTier"]

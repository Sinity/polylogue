"""Shared support helpers for semantic-proof report models."""

from __future__ import annotations


def empty_metric_counts() -> dict[str, int]:
    return {"preserved": 0, "declared_loss": 0, "critical_loss": 0}


__all__ = ["empty_metric_counts"]

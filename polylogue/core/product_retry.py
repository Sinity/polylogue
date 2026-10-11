"""Exact identities and outcomes for one admitted derived product attempt."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ProductOutcome(StrEnum):
    CURRENT = "current"
    VALID_EMPTY = "valid_empty"
    BLOCKED = "blocked"
    FAILED = "failed"
    STALE = "stale"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ProductAttempt:
    product: str
    target: str
    ops_incarnation: str
    demand_identity: str
    attempt_id: str


@dataclass(frozen=True, slots=True)
class ProductObservation:
    demand_identity: str
    needs_work: bool
    valid_empty: bool = False

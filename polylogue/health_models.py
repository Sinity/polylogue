"""Shared health reporting types."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .lib.outcomes import OutcomeCheck, OutcomeReport, OutcomeStatus

HealthCheck = OutcomeCheck
VerifyStatus = OutcomeStatus

HEALTH_TTL_SECONDS = 600


@dataclass
class HealthReport(OutcomeReport):
    """Comprehensive health and verification report."""

    timestamp: int = field(default_factory=lambda: int(time.time()))
    cached: bool = False
    age_seconds: int = 0

    @property
    def summary(self) -> dict[str, int]:
        return self.summary_counts()

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cached": self.cached,
            "age_seconds": self.age_seconds,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "count": check.count,
                    "detail": check.summary,
                    "breakdown": check.breakdown,
                }
                for check in self.checks
            ],
            "summary": self.summary,
        }


__all__ = [
    "HEALTH_TTL_SECONDS",
    "HealthCheck",
    "HealthReport",
    "VerifyStatus",
]

"""Shared outcome grammar for checks, audits, and verification results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OutcomeStatus(str, Enum):
    """Shared status levels for operator-facing checks and reports."""

    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    SKIP = "skip"

    @classmethod
    def from_string(cls, value: str | OutcomeStatus) -> OutcomeStatus:
        if isinstance(value, cls):
            return value
        return cls(str(value).strip().lower())

    def __str__(self) -> str:
        return self.value


@dataclass
class OutcomeCheck:
    """Typed, reusable check outcome with optional structured evidence."""

    name: str
    status: OutcomeStatus
    summary: str = ""
    count: int = 0
    details: list[str] = field(default_factory=list)
    breakdown: dict[str, int] = field(default_factory=dict)

    @property
    def detail(self) -> str:
        return self.summary

    @property
    def message(self) -> str:
        return self.summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "summary": self.summary,
            "count": self.count,
            "details": list(self.details),
            "breakdown": dict(self.breakdown),
        }

    def format_line(
        self,
        *,
        labels: Mapping[OutcomeStatus, str] | None = None,
        icons: Mapping[OutcomeStatus, str] | None = None,
    ) -> str:
        label_map = labels or {}
        icon_map = icons or {}
        label = label_map.get(self.status, self.status.value.upper())
        icon = icon_map.get(self.status)
        prefix = f"{icon} " if icon else ""
        return f"{prefix}[{label}] {self.name}: {self.summary}"


@dataclass
class OutcomeReport:
    """Shared report container for collections of outcome checks."""

    checks: list[OutcomeCheck] = field(default_factory=list)

    def count(self, status: OutcomeStatus) -> int:
        return sum(1 for check in self.checks if check.status is status)

    def summary_counts(self, *, include_skip: bool = False) -> dict[str, int]:
        statuses = [
            OutcomeStatus.OK,
            OutcomeStatus.WARNING,
            OutcomeStatus.ERROR,
        ]
        if include_skip:
            statuses.append(OutcomeStatus.SKIP)
        return {status.value: self.count(status) for status in statuses}

    @property
    def ok_count(self) -> int:
        return self.count(OutcomeStatus.OK)

    @property
    def warning_count(self) -> int:
        return self.count(OutcomeStatus.WARNING)

    @property
    def error_count(self) -> int:
        return self.count(OutcomeStatus.ERROR)

    @property
    def skip_count(self) -> int:
        return self.count(OutcomeStatus.SKIP)

    @property
    def all_ok(self) -> bool:
        return all(check.status in {OutcomeStatus.OK, OutcomeStatus.SKIP} for check in self.checks)


__all__ = [
    "OutcomeCheck",
    "OutcomeReport",
    "OutcomeStatus",
]

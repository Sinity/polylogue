"""Shared outcome grammar for checks, audits, and verification results."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from polylogue.core.json import JSONDocument, json_document


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


OutcomeCheckFn = Callable[[], "OutcomeCheck"]


class OutcomeOwner(Protocol):
    """The small callable/result seam used by independently owned checks."""

    @property
    def name(self) -> str: ...

    @property
    def check(self) -> OutcomeCheckFn | None: ...


class OutcomeCompositionFailureKind(str, Enum):
    """Why an owner could not contribute a result to a composition."""

    OWNER_OMITTED = "owner-omitted"
    OWNER_RAISED = "owner-raised"


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

    def to_dict(self) -> JSONDocument:
        return json_document(
            {
                "name": self.name,
                "status": self.status.value,
                "summary": self.summary,
                "count": self.count,
                "details": list(self.details),
                "breakdown": dict(self.breakdown),
            }
        )

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
class OutcomeCompositionFailure(OutcomeCheck):
    """Typed failure result that does not stop sibling owners from running."""

    failure_kind: OutcomeCompositionFailureKind = OutcomeCompositionFailureKind.OWNER_RAISED

    def to_dict(self) -> JSONDocument:
        payload = dict(super().to_dict())
        payload["failure_kind"] = self.failure_kind.value
        return json_document(payload)


@dataclass(frozen=True)
class BoundOutcomeOwner:
    """Named domain adapter bound to the inputs for one verification run."""

    name: str
    check: OutcomeCheckFn | None


def compose_outcome_checks(owners: Iterable[OutcomeOwner]) -> OutcomeReport:
    """Run named owners independently, preserving typed failures and siblings."""
    results: list[OutcomeCheck] = []
    for owner in owners:
        if owner.check is None:
            results.append(
                OutcomeCompositionFailure(
                    name=owner.name,
                    status=OutcomeStatus.ERROR,
                    summary="domain owner omitted its check",
                    failure_kind=OutcomeCompositionFailureKind.OWNER_OMITTED,
                )
            )
            continue
        try:
            results.append(owner.check())
        except Exception as exc:
            results.append(
                OutcomeCompositionFailure(
                    name=owner.name,
                    status=OutcomeStatus.ERROR,
                    summary=f"domain owner raised {type(exc).__name__}: {exc}",
                    failure_kind=OutcomeCompositionFailureKind.OWNER_RAISED,
                )
            )
    return OutcomeReport(checks=results)


@dataclass(frozen=True)
class OutcomeCounts:
    """Typed status-count payload shared by outcome reports."""

    ok: int = 0
    warning: int = 0
    error: int = 0
    skip: int | None = None

    def to_json(self, *, include_skip: bool = False) -> JSONDocument:
        payload: JSONDocument = {
            OutcomeStatus.OK.value: self.ok,
            OutcomeStatus.WARNING.value: self.warning,
            OutcomeStatus.ERROR.value: self.error,
        }
        if include_skip:
            payload[OutcomeStatus.SKIP.value] = self.skip or 0
        return payload


@dataclass
class OutcomeReport:
    """Shared report container for collections of outcome checks."""

    checks: list[OutcomeCheck] = field(default_factory=list)

    def count(self, status: OutcomeStatus) -> int:
        return sum(1 for check in self.checks if check.status is status)

    def counts(self) -> OutcomeCounts:
        return OutcomeCounts(
            ok=self.count(OutcomeStatus.OK),
            warning=self.count(OutcomeStatus.WARNING),
            error=self.count(OutcomeStatus.ERROR),
            skip=self.count(OutcomeStatus.SKIP),
        )

    def summary_counts(self, *, include_skip: bool = False) -> dict[str, int]:
        counts = self.counts()
        payload = {
            OutcomeStatus.OK.value: counts.ok,
            OutcomeStatus.WARNING.value: counts.warning,
            OutcomeStatus.ERROR.value: counts.error,
        }
        if include_skip:
            payload[OutcomeStatus.SKIP.value] = counts.skip or 0
        return payload

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
    "BoundOutcomeOwner",
    "OutcomeCheck",
    "OutcomeCounts",
    "OutcomeCompositionFailure",
    "OutcomeCompositionFailureKind",
    "OutcomeOwner",
    "OutcomeReport",
    "OutcomeStatus",
    "compose_outcome_checks",
]

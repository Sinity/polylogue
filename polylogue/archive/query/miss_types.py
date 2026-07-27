"""Shared dataclasses for query-miss diagnosis.

Split out from :mod:`polylogue.archive.query.miss_diagnostics` so the
predicate-attribution probes in :mod:`polylogue.archive.query.miss_predicates`
can construct :class:`QueryMissReason` values without importing
``miss_diagnostics`` (which imports the probes back) and creating a cycle.
``miss_diagnostics`` re-exports both names, so existing imports are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from polylogue.core.json import JSONDocument

Severity = Literal["info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class QueryMissReason:
    """One observed reason a query may have returned no sessions."""

    code: str
    severity: Severity
    summary: str
    detail: str | None = None
    count: int | None = None

    def to_dict(self) -> JSONDocument:
        payload: JSONDocument = {
            "code": self.code,
            "severity": self.severity,
            "summary": self.summary,
        }
        if self.detail:
            payload["detail"] = self.detail
        if self.count is not None:
            payload["count"] = self.count
        return payload


@dataclass(frozen=True, slots=True)
class QueryMissDiagnostics:
    """Structured no-result diagnosis shared by CLI and MCP surfaces."""

    message: str
    filters: tuple[str, ...]
    reasons: tuple[QueryMissReason, ...]
    archive_session_count: int | None = None
    raw_session_count: int | None = None

    def to_dict(self) -> JSONDocument:
        payload: JSONDocument = {
            "message": self.message,
            "filters": list(self.filters),
            "reasons": [reason.to_dict() for reason in self.reasons],
        }
        if self.archive_session_count is not None:
            payload["archive_session_count"] = self.archive_session_count
        if self.raw_session_count is not None:
            payload["raw_session_count"] = self.raw_session_count
        return payload

    def human_reason_lines(self) -> list[str]:
        """Return concise human-facing reason lines."""
        lines: list[str] = []
        for reason in self.reasons:
            lines.append(reason.summary)
            if reason.detail:
                lines.append(f"  {reason.detail}")
        return lines


__all__ = ["QueryMissDiagnostics", "QueryMissReason", "Severity"]

"""Health-oriented MCP payloads."""

from __future__ import annotations

from typing import Any

from polylogue.mcp.payload_base import MCPPayload


class MCPHealthCheckPayload(MCPPayload):
    name: str
    status: str
    count: int | None = None
    detail: str | None = None

    @classmethod
    def from_check(cls, check: Any, *, include_counts: bool, include_detail: bool) -> MCPHealthCheckPayload:
        return cls(
            name=check.name,
            status=check.status.value if hasattr(check.status, "value") else str(check.status),
            count=check.count if include_counts else None,
            detail=check.detail if include_detail else None,
        )


def _extract_source(report: Any) -> str | None:
    provenance = getattr(report, "provenance", None)
    if provenance is None:
        return None
    source = getattr(provenance, "source", None)
    if source is None:
        return None
    return getattr(source, "value", str(source))


class MCPHealthReportPayload(MCPPayload):
    checks: list[MCPHealthCheckPayload]
    summary: str
    source: str | None = None

    @classmethod
    def from_report(
        cls,
        report: Any,
        *,
        include_counts: bool,
        include_detail: bool,
        include_cached: bool,
    ) -> MCPHealthReportPayload:
        return cls(
            checks=[
                MCPHealthCheckPayload.from_check(
                    check,
                    include_counts=include_counts,
                    include_detail=include_detail,
                )
                for check in report.checks
            ],
            summary=report.summary,
            source=_extract_source(report) if include_cached else None,
        )


__all__ = ["MCPHealthCheckPayload", "MCPHealthReportPayload"]

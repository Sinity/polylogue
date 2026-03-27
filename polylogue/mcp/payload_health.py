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


class MCPHealthReportPayload(MCPPayload):
    checks: list[MCPHealthCheckPayload]
    summary: str
    source: str | None = None
    cache_age_seconds: int | None = None
    cache_ttl_seconds: int | None = None

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
            source=(
                getattr(getattr(report, "provenance", None), "source", None).value
                if include_cached and getattr(report, "provenance", None) is not None
                else None
            ),
            cache_age_seconds=(
                getattr(getattr(report, "provenance", None), "cache_age_seconds", None)
                if include_cached
                else None
            ),
            cache_ttl_seconds=(
                getattr(getattr(report, "provenance", None), "cache_ttl_seconds", None)
                if include_cached
                else None
            ),
        )


__all__ = ["MCPHealthCheckPayload", "MCPHealthReportPayload"]

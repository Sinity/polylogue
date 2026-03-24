"""Health cache read/write helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from polylogue.logging import get_logger

from .health_models import HEALTH_TTL_SECONDS, HealthCheck, HealthReport, VerifyStatus
from .maintenance_models import (
    ArchiveDebtStatus,
    DerivedModelStatus,
    MaintenanceCategory,
    ReportProvenance,
    TruthSource,
)

logger = get_logger(__name__)


def cache_path(archive_root: Path) -> Path:
    return archive_root / "health.json"


def load_cached(archive_root: Path) -> dict[str, Any] | None:
    path = cache_path(archive_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        logger.warning("Failed to load health cache: %s", exc)
    return None


def write_cache(archive_root: Path, report: HealthReport | dict[str, Any]) -> None:
    path = cache_path(archive_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = report.to_dict() if isinstance(report, HealthReport) else report
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write health cache: %s", exc)


def cached_health_summary(archive_root: Path) -> str:
    """Get a concise summary of the cached health state."""
    cached_data = load_cached(archive_root)
    if not cached_data:
        return "not run"
    try:
        ts = int(cached_data.get("timestamp", 0))
    except (TypeError, ValueError):
        return "unknown"
    age = int(time.time()) - ts
    summary = cached_data.get("summary", {})
    if not summary:
        return f"cached {age}s ago"

    parts = []
    for status in (VerifyStatus.OK, VerifyStatus.WARNING, VerifyStatus.ERROR):
        if count := summary.get(status.value):
            parts.append(f"{status.value}={count}")
    return f"cached {age}s ago ({', '.join(parts)})"


def load_cached_report(archive_root: Path) -> HealthReport | None:
    cached_data = load_cached(archive_root)
    now = int(time.time())
    path = cache_path(archive_root)
    if not cached_data:
        return None
    try:
        ts = int(cached_data.get("timestamp", 0))
    except (TypeError, ValueError):
        return None
    if (now - ts) >= HEALTH_TTL_SECONDS:
        return None
    try:
        checks = [
            HealthCheck(
                name=check["name"],
                status=VerifyStatus(check["status"]),
                count=check.get("count", 0),
                summary=check.get("detail", check.get("summary", "")),
                breakdown=check.get("breakdown", {}),
            )
            for check in cached_data.get("checks", [])
        ]
    except (KeyError, ValueError, TypeError):
        return None
    try:
        derived_models = {
            str(name): DerivedModelStatus(
                name=str(payload["name"]),
                ready=bool(payload["ready"]),
                detail=str(payload["detail"]),
                source_documents=int(payload.get("source_documents", 0) or 0),
                materialized_documents=int(payload.get("materialized_documents", 0) or 0),
                source_rows=int(payload.get("source_rows", 0) or 0),
                materialized_rows=int(payload.get("materialized_rows", 0) or 0),
                pending_documents=int(payload.get("pending_documents", 0) or 0),
                pending_rows=int(payload.get("pending_rows", 0) or 0),
                stale_rows=int(payload.get("stale_rows", 0) or 0),
                orphan_rows=int(payload.get("orphan_rows", 0) or 0),
                missing_provenance_rows=int(payload.get("missing_provenance_rows", 0) or 0),
                materializer_version=(
                    int(payload["materializer_version"])
                    if payload.get("materializer_version") is not None
                    else None
                ),
                matches_version=(
                    bool(payload["matches_version"])
                    if payload.get("matches_version") is not None
                    else None
                ),
            )
            for name, payload in (cached_data.get("derived_models") or {}).items()
            if isinstance(payload, dict)
        }
    except (KeyError, TypeError, ValueError):
        return None
    try:
        archive_debt = {
            str(name): ArchiveDebtStatus(
                name=str(payload["name"]),
                category=MaintenanceCategory(payload["category"]),
                destructive=bool(payload["destructive"]),
                issue_count=int(payload.get("issue_count", 0) or 0),
                detail=str(payload["detail"]),
                maintenance_target=str(payload.get("maintenance_target") or payload["name"]),
            )
            for name, payload in (cached_data.get("archive_debt") or {}).items()
            if isinstance(payload, dict)
        }
    except (KeyError, TypeError, ValueError):
        return None
    return HealthReport(
        checks=checks,
        timestamp=ts,
        provenance=ReportProvenance(
            source=TruthSource.CACHE,
            cache_age_seconds=now - ts,
            cache_ttl_seconds=HEALTH_TTL_SECONDS,
            cache_path=str(path),
        ),
        derived_models=derived_models,
        archive_debt=archive_debt,
    )

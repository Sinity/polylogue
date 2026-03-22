"""Health cache read/write helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from polylogue.logging import get_logger

from .health_models import HEALTH_TTL_SECONDS, HealthCheck, HealthReport, VerifyStatus

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
    if not cached_data:
        return None
    ts = cached_data.get("timestamp", 0)
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
    return HealthReport(
        checks=checks,
        timestamp=ts,
        cached=True,
        age_seconds=now - ts,
    )

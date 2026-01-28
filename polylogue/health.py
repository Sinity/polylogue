from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast

from .config import Config
from .core.log import get_logger
from .ingestion.drive_client import default_credentials_path, default_token_path
from .storage.backends.sqlite import connection_context, open_connection
from .storage.index import index_status

LOGGER = get_logger(__name__)
HEALTH_TTL_SECONDS = 600


class VerifyStatus(str, Enum):
    """Status levels for health and verification checks."""

    OK = "ok"
    WARNING = "warning"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


@dataclass
class HealthCheck:
    """Result of a single health or verification check."""

    name: str
    status: VerifyStatus
    count: int = 0
    detail: str = ""
    breakdown: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "count": self.count,
            "detail": self.detail,
            "breakdown": self.breakdown,
        }


@dataclass
class HealthReport:
    """Comprehensive health and verification report."""

    checks: list[HealthCheck]
    summary: dict[str, int]
    timestamp: int = field(default_factory=lambda: int(time.time()))
    cached: bool = False
    age_seconds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cached": self.cached,
            "age_seconds": self.age_seconds,
            "checks": [c.to_dict() for c in self.checks],
            "summary": self.summary,
        }


def _cache_path(archive_root: Path) -> Path:
    return archive_root / "health.json"


def _load_cached(archive_root: Path) -> dict[str, Any] | None:
    path = _cache_path(archive_root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception as exc:
        LOGGER.warning("Failed to load health cache: %s", exc)
    return None


def _write_cache(archive_root: Path, report: HealthReport | dict[str, Any]) -> None:
    path = _cache_path(archive_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = report.to_dict() if isinstance(report, HealthReport) else report
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        LOGGER.warning("Failed to write health cache: %s", exc)


def run_health(config: Config) -> HealthReport:
    """Run comprehensive system health and data verification checks."""
    checks: list[HealthCheck] = []

    # 1. Environment & Paths
    checks.append(HealthCheck("config", VerifyStatus.OK, detail="Zero-config (XDG paths)"))

    for path_name in ("archive_root", "render_root"):
        path = getattr(config, path_name)
        if path.exists():
            checks.append(HealthCheck(path_name, VerifyStatus.OK, detail=str(path)))
        else:
            checks.append(HealthCheck(path_name, VerifyStatus.WARNING, detail=f"Missing {path}"))

    # 2. Database Reachability & Integrity
    try:
        with open_connection(None) as conn:
            checks.append(HealthCheck("database", VerifyStatus.OK, detail="DB reachable"))
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            checks.append(
                HealthCheck(
                    "sqlite_integrity",
                    VerifyStatus.OK if integrity == "ok" else VerifyStatus.ERROR,
                    detail=integrity,
                )
            )
    except Exception as exc:
        checks.append(HealthCheck("database", VerifyStatus.ERROR, detail=f"DB error: {exc}"))

    # 3. Search Index Status
    idx = index_status()
    if idx["exists"]:
        checks.append(
            HealthCheck(
                "index", VerifyStatus.OK, count=int(cast(Any, idx["count"])), detail=f"messages indexed: {idx['count']}"
            )
        )
    else:
        checks.append(HealthCheck("index", VerifyStatus.WARNING, detail="index not built"))

    # 4. Data Quality (from former verify.py)
    with connection_context(None) as conn:
        # Orphaned messages
        orphan_count = conn.execute(
            """
            SELECT COUNT(*) FROM messages m
            WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = m.conversation_id)
        """
        ).fetchone()[0]
        checks.append(
            HealthCheck(
                "orphaned_messages",
                VerifyStatus.OK if orphan_count == 0 else VerifyStatus.ERROR,
                count=orphan_count,
                detail="No orphaned messages" if orphan_count == 0 else f"{orphan_count:,} orphaned messages",
            )
        )

        # Duplicate conversations
        dup_conv = conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT conversation_id FROM conversations GROUP BY conversation_id HAVING COUNT(*) > 1
            )
        """
        ).fetchone()[0]
        checks.append(
            HealthCheck(
                "duplicate_conversations",
                VerifyStatus.OK if dup_conv == 0 else VerifyStatus.ERROR,
                count=dup_conv,
                detail="No duplicates" if dup_conv == 0 else f"{dup_conv} duplicate conversation IDs",
            )
        )

    # 5. Source Accessibility
    for source in config.sources:
        if source.folder:
            cred_path = default_credentials_path(config.drive_config)
            token_path = default_token_path(config.drive_config)
            cred_status = VerifyStatus.OK if cred_path.exists() else VerifyStatus.WARNING
            token_status = VerifyStatus.OK if token_path.exists() else VerifyStatus.WARNING
            checks.append(
                HealthCheck(
                    f"source:{source.name}",
                    cred_status,
                    detail=f"drive folder '{source.folder}' credentials: {cred_path}",
                )
            )
            checks.append(
                HealthCheck(
                    f"source:{source.name}:token",
                    token_status,
                    detail=f"drive token: {token_path}",
                )
            )
        else:
            if source.path and source.path.exists():
                checks.append(HealthCheck(f"source:{source.name}", VerifyStatus.OK, detail=str(source.path)))
            else:
                checks.append(
                    HealthCheck(f"source:{source.name}", VerifyStatus.WARNING, detail=f"missing path: {source.path}")
                )

    # Build summary
    summary = {"ok": 0, "warning": 0, "error": 0}
    for check in checks:
        summary[check.status.value] += 1

    report = HealthReport(checks=checks, summary=summary)
    _write_cache(config.archive_root, report)
    return report


def get_health(config: Config) -> HealthReport:
    """Get health report, using cache if valid."""
    cached_data = _load_cached(config.archive_root)
    now = int(time.time())
    if cached_data:
        ts = cached_data.get("timestamp", 0)
        if (now - ts) < HEALTH_TTL_SECONDS:
            checks = [
                HealthCheck(
                    name=c["name"],
                    status=VerifyStatus(c["status"]),
                    count=c.get("count", 0),
                    detail=c.get("detail", ""),
                    breakdown=c.get("breakdown", {}),
                )
                for c in cached_data.get("checks", [])
            ]
            return HealthReport(
                checks=checks,
                summary=cached_data.get("summary", {}),
                timestamp=ts,
                cached=True,
                age_seconds=now - ts,
            )

    report = run_health(config)
    report.cached = False
    report.age_seconds = 0
    return report


def cached_health_summary(archive_root: Path) -> str:
    """Get a concise summary of the cached health state."""
    cached_data = _load_cached(archive_root)
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
    for s in (VerifyStatus.OK, VerifyStatus.WARNING, VerifyStatus.ERROR):
        if count := summary.get(s.value):
            parts.append(f"{s.value}={count}")
    return f"cached {age}s ago ({', '.join(parts)})"


__all__ = ["get_health", "run_health", "HealthCheck", "HealthReport", "VerifyStatus", "cached_health_summary"]

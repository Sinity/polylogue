"""Health checks, verification, and repair operations."""

from __future__ import annotations

from .config import Config
from .health_archive import run_archive_health
from .health_cache import (
    cache_path as _cache_path_impl,
)
from .health_cache import (
    cached_health_summary,
    load_cached_report,
)
from .health_cache import (
    load_cached as _load_cached_impl,
)
from .health_cache import (
    write_cache as _write_cache_impl,
)
from .health_models import (
    HEALTH_TTL_SECONDS as _HEALTH_TTL_SECONDS,
)
from .health_models import (
    HealthCheck as _HealthCheck,
)
from .health_models import (
    HealthReport as _HealthReport,
)
from .health_models import (
    VerifyStatus as _VerifyStatus,
)
from .health_runtime import run_runtime_health

HEALTH_TTL_SECONDS = _HEALTH_TTL_SECONDS
HealthCheck = _HealthCheck
HealthReport = _HealthReport
VerifyStatus = _VerifyStatus
_cache_path = _cache_path_impl
_load_cached = _load_cached_impl
_write_cache = _write_cache_impl


def run_health(config: Config, *, deep: bool = False) -> HealthReport:
    return run_archive_health(config, deep=deep)


def get_health(config: Config, *, deep: bool = False) -> HealthReport:
    """Get health report, using cache if valid.

    Args:
        config: Application configuration
        deep: If True, skip cache and run PRAGMA integrity_check (slow).
    """
    if not deep:
        cached_report = load_cached_report(config.archive_root)
        if cached_report is not None:
            return cached_report

    report = run_health(config, deep=deep)
    report.cached = False
    report.age_seconds = 0
    return report


__all__ = [
    "get_health",
    "run_health",
    "run_runtime_health",
    "HealthCheck",
    "HealthReport",
    "VerifyStatus",
    "cached_health_summary",
]

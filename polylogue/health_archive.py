"""Archive/data health checks."""

from __future__ import annotations

from .config import Config
from .health_archive_runtime import build_archive_runtime_checks
from .health_archive_sources import build_schema_health_checks, build_source_health_checks
from .health_cache import load_cached_report, write_cache
from .health_models import HealthCheck, HealthReport, VerifyStatus
from .storage.derived_status import collect_derived_model_statuses_sync


def run_archive_health(config: Config, *, deep: bool = False) -> HealthReport:
    checks: list[HealthCheck] = []
    checks.append(HealthCheck("config", VerifyStatus.OK, summary="Zero-config (XDG paths)"))

    for path_name in ("archive_root", "render_root"):
        path = getattr(config, path_name)
        if path.exists():
            checks.append(HealthCheck(path_name, VerifyStatus.OK, summary=str(path)))
        else:
            checks.append(HealthCheck(path_name, VerifyStatus.WARNING, summary=f"Missing {path}"))

    runtime_checks, derived_statuses, archive_debt, _db_error = build_archive_runtime_checks(
        config,
        deep=deep,
        collect_derived_statuses=collect_derived_model_statuses_sync,
    )
    checks.extend(runtime_checks)
    checks.extend(build_source_health_checks(config))
    checks.extend(build_schema_health_checks())

    report = HealthReport(checks=checks, derived_models=derived_statuses, archive_debt=archive_debt)
    write_cache(config.archive_root, report)
    return report


def get_health(config: Config, *, deep: bool = False, use_cached: bool = False) -> HealthReport:
    """Get an archive health report, optionally using the cached report."""
    if use_cached and not deep:
        cached_report = load_cached_report(config.archive_root)
        if cached_report is not None:
            return cached_report

    report = run_archive_health(config, deep=deep)
    return report


__all__ = ["collect_derived_model_statuses_sync", "get_health", "run_archive_health"]

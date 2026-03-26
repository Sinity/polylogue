"""Source and schema health check families."""

from __future__ import annotations

from polylogue.config import Config
from polylogue.health_models import HealthCheck, VerifyStatus
from polylogue.lib.provider_identity import CORE_SCHEMA_PROVIDERS
from polylogue.sources.drive_auth import default_credentials_path, default_token_path


def build_source_health_checks(config: Config) -> list[HealthCheck]:
    checks: list[HealthCheck] = []
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
                    summary=f"drive folder '{source.folder}' credentials: {cred_path}",
                )
            )
            checks.append(
                HealthCheck(
                    f"source:{source.name}:token",
                    token_status,
                    summary=f"drive token: {token_path}",
                )
            )
        elif source.path and source.path.exists():
            checks.append(HealthCheck(f"source:{source.name}", VerifyStatus.OK, summary=str(source.path)))
        else:
            checks.append(
                HealthCheck(
                    f"source:{source.name}",
                    VerifyStatus.WARNING,
                    summary=f"missing path: {source.path}",
                )
            )
    return checks


def build_schema_health_checks() -> list[HealthCheck]:
    checks: list[HealthCheck] = []
    try:
        from polylogue.schemas.registry import SchemaRegistry

        registry = SchemaRegistry()
        known_providers = list(CORE_SCHEMA_PROVIDERS)
        available = registry.list_providers()
        missing = [provider for provider in known_providers if provider not in available]

        if missing:
            checks.append(
                HealthCheck(
                    "schemas_missing",
                    VerifyStatus.WARNING,
                    count=len(missing),
                    summary=f"Missing schemas for: {', '.join(missing)}",
                )
            )
        else:
            checks.append(
                HealthCheck(
                    "schemas_coverage",
                    VerifyStatus.OK,
                    count=len(available),
                    summary=f"All {len(available)} provider schemas present",
                )
            )

        stale_providers = []
        for provider in available:
            age = registry.get_schema_age_days(provider)
            if age is not None and age > 30:
                stale_providers.append(f"{provider} ({age}d)")
        if stale_providers:
            checks.append(
                HealthCheck(
                    "schemas_freshness",
                    VerifyStatus.WARNING,
                    count=len(stale_providers),
                    summary=f"Stale schemas (>30d): {', '.join(stale_providers)}",
                )
            )
        else:
            checks.append(
                HealthCheck(
                    "schemas_freshness",
                    VerifyStatus.OK,
                    summary="All schemas current",
                )
            )
    except Exception as exc:
        checks.append(
            HealthCheck("schemas", VerifyStatus.WARNING, summary=f"Schema check failed: {exc}")
        )
    return checks


__all__ = ["build_schema_health_checks", "build_source_health_checks"]

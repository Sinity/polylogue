"""Support helpers for schema CLI commands."""

from __future__ import annotations

from pathlib import Path

from polylogue.schemas.operator_models import JSONDocument
from polylogue.schemas.privacy_config import PrivacyConfig, PrivacyConfigSection, PrivacyLevel


def _privacy_config_payload(config: PrivacyConfig) -> JSONDocument:
    payload: JSONDocument = {
        "level": config.level,
        "safe_enum_max_length": config.safe_enum_max_length,
        "high_entropy_min_length": config.high_entropy_min_length,
        "cross_conv_min_count": config.cross_conv_min_count,
        "cross_conv_proportional": config.cross_conv_proportional,
    }
    if config.field_overrides:
        payload["field_overrides"] = dict(config.field_overrides)
    if config.allow_value_patterns:
        payload["allow_value_patterns"] = list(config.allow_value_patterns)
    if config.deny_value_patterns:
        payload["deny_value_patterns"] = list(config.deny_value_patterns)
    return payload


def _privacy_level(value: str) -> PrivacyLevel:
    if value == "strict":
        return "strict"
    if value == "standard":
        return "standard"
    if value == "permissive":
        return "permissive"
    return "standard"


def build_schema_privacy_config(
    *,
    privacy: str | None,
    privacy_config_path: Path | None,
) -> JSONDocument | None:
    """Resolve schema-generation privacy config from CLI options."""
    from polylogue.schemas.privacy_config import load_privacy_config

    cli_overrides: PrivacyConfigSection = {}
    if privacy:
        cli_overrides["level"] = privacy
    if privacy_config_path:
        return _privacy_config_payload(
            load_privacy_config(
                cli_overrides=cli_overrides,
                project_path=privacy_config_path.parent,
            )
        )
    if privacy is not None:
        return _privacy_config_payload(PrivacyConfig(level=_privacy_level(privacy)))
    return None


__all__ = ["build_schema_privacy_config"]

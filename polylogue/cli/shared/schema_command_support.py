"""Support helpers for schema CLI commands."""

from __future__ import annotations

from pathlib import Path

from polylogue.schemas.operator.models import JSONDocument, operator_json_document
from polylogue.schemas.privacy_config import PrivacyConfig, PrivacyConfigSection, PrivacyLevel


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
        return operator_json_document(
            load_privacy_config(
                cli_overrides=cli_overrides,
                project_path=privacy_config_path.parent,
            ).to_payload()
        )
    if privacy is not None:
        return operator_json_document(PrivacyConfig(level=_privacy_level(privacy)).to_payload())
    return None


__all__ = ["build_schema_privacy_config"]

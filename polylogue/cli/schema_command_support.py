"""Support helpers for schema CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_schema_privacy_config(
    *,
    privacy: str | None,
    privacy_config_path: Path | None,
) -> Any:
    """Resolve schema-generation privacy config from CLI options."""
    from polylogue.schemas.privacy_config import PrivacyConfig, load_privacy_config

    cli_overrides: dict[str, Any] = {}
    if privacy:
        cli_overrides["level"] = privacy
    if privacy_config_path:
        return load_privacy_config(
            cli_overrides=cli_overrides,
            project_path=privacy_config_path.parent,
        )
    if cli_overrides:
        return PrivacyConfig(**cli_overrides)
    return None


__all__ = ["build_schema_privacy_config"]

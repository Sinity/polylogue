"""Configuration using Pydantic Settings for automatic env var support."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..paths import CONFIG_HOME, DATA_HOME


class OutputPathsConfig(BaseSettings):
    """Output paths configuration."""

    render: Path = Field(default=DATA_HOME / "archive" / "render")
    sync_drive: Path = Field(default=DATA_HOME / "archive" / "gemini")
    sync_codex: Path = Field(default=DATA_HOME / "archive" / "codex")
    sync_claude_code: Path = Field(default=DATA_HOME / "archive" / "claude-code")
    import_chatgpt: Path = Field(default=DATA_HOME / "archive" / "chatgpt")
    import_claude: Path = Field(default=DATA_HOME / "archive" / "claude")

    @field_validator("*", mode="before")
    @classmethod
    def expand_path(cls, v: Any) -> Path:
        if isinstance(v, str):
            return Path(v).expanduser()
        if isinstance(v, Path):
            return v.expanduser()
        return v


class DefaultsConfig(BaseSettings):
    """Default settings for the application."""

    collapse_threshold: int = Field(default=25, ge=0)
    html_previews: bool = Field(default=True)
    html_theme: str = Field(default="dark")

    model_config = SettingsConfigDict(
        env_prefix="POLYLOGUE_",
        env_nested_delimiter="__",
    )


class IndexConfig(BaseSettings):
    """Search index configuration."""

    backend: str = Field(default="sqlite")
    qdrant_url: Optional[str] = Field(default=None)
    qdrant_api_key: Optional[str] = Field(default=None)
    qdrant_collection: str = Field(default="polylogue")
    qdrant_vector_size: Optional[int] = Field(default=None, ge=1)

    model_config = SettingsConfigDict(
        env_prefix="POLYLOGUE_INDEX_",
        env_nested_delimiter="__",
    )


class PathsConfig(BaseSettings):
    """Path configuration."""

    input_root: Path = Field(default=DATA_HOME / "inbox")
    output_root: Path = Field(default=DATA_HOME / "archive")
    config_home: Path = Field(default=CONFIG_HOME)

    @field_validator("*", mode="before")
    @classmethod
    def expand_path(cls, v: Any) -> Path:
        if isinstance(v, str):
            return Path(v).expanduser()
        if isinstance(v, Path):
            return v.expanduser()
        return v

    model_config = SettingsConfigDict(
        env_prefix="POLYLOGUE_",
        env_nested_delimiter="__",
    )


class AppConfigV2(BaseSettings):
    """Main application configuration with Pydantic Settings.

    Supports:
    - JSON config files
    - Environment variables (POLYLOGUE_*)
    - .env files
    - Automatic type validation
    """

    paths: PathsConfig = Field(default_factory=PathsConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)

    # Additional config for compatibility
    raw: Dict[str, Any] = Field(default_factory=dict)
    config_path: Optional[Path] = Field(default=None)

    model_config = SettingsConfigDict(
        env_prefix="POLYLOGUE_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="allow",
    )

    @classmethod
    def from_json_file(cls, path: Path) -> "AppConfigV2":
        """Load configuration from a JSON file."""
        if not path.exists():
            return cls()

        try:
            data = json.loads(path.read_text(encoding="utf-8"))

            # Parse nested structures
            config_dict: Dict[str, Any] = {}

            if "paths" in data:
                config_dict["paths"] = PathsConfig(**data["paths"])

            if "defaults" in data or "ui" in data:
                defaults_data = data.get("defaults") or data.get("ui") or {}
                config_dict["defaults"] = DefaultsConfig(**defaults_data)

            if "index" in data:
                config_dict["index"] = IndexConfig(**data["index"])

            # Store raw data for compatibility
            config_dict["raw"] = data
            config_dict["config_path"] = path

            return cls(**config_dict)
        except Exception as e:
            # Fallback to default config on parse error
            print(f"Warning: Failed to parse config file {path}: {e}")
            return cls()

    @classmethod
    def load(cls) -> "AppConfigV2":
        """Load configuration from standard locations."""
        # Check for explicit config path
        import os
        env_path = os.environ.get("POLYLOGUE_CONFIG")
        if env_path:
            path = Path(env_path).expanduser()
            if path.exists():
                return cls.from_json_file(path)

        # Check default locations
        for path in [CONFIG_HOME / "config.json"]:
            if path.exists():
                return cls.from_json_file(path)

        # Return default config (will still load from env vars)
        return cls()

    def get_output_paths(self) -> OutputPathsConfig:
        """Get output paths based on configuration."""
        output_root = self.paths.output_root
        return OutputPathsConfig(
            render=output_root / "render",
            sync_drive=output_root / "gemini",
            sync_codex=output_root / "codex",
            sync_claude_code=output_root / "claude-code",
            import_chatgpt=output_root / "chatgpt",
            import_claude=output_root / "claude",
        )


# Convenience function for backward compatibility
def load_configuration_v2() -> AppConfigV2:
    """Load the application configuration."""
    return AppConfigV2.load()

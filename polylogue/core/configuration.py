"""Configuration using Pydantic Settings for automatic env var support."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..paths import CONFIG_HOME, DATA_HOME

# Configuration constants
CONFIG_ENV = "POLYLOGUE_CONFIG"
DEFAULT_CONFIG_LOCATIONS = [CONFIG_HOME / "config.json"]


class ExportsConfig(BaseSettings):
    """Export source paths configuration."""

    chatgpt: Path = Field(default=DATA_HOME / "inbox")
    claude: Path = Field(default=DATA_HOME / "inbox")

    @field_validator("*", mode="before")
    @classmethod
    def expand_path(cls, v: Any) -> Path:
        if isinstance(v, str):
            return Path(v).expanduser()
        if isinstance(v, Path):
            return v.expanduser()
        return v

    model_config = SettingsConfigDict(
        env_prefix="POLYLOGUE_EXPORTS_",
        env_nested_delimiter="__",
    )


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


class LabeledRootPathsConfig(BaseSettings):
    """Per-label output root overrides."""

    render: Path
    sync_drive: Path
    sync_codex: Path
    sync_claude_code: Path
    import_chatgpt: Path
    import_claude: Path

    @field_validator("*", mode="before")
    @classmethod
    def expand_path(cls, v: Any) -> Path:
        if isinstance(v, str):
            return Path(v).expanduser()
        if isinstance(v, Path):
            return v.expanduser()
        return v

    model_config = SettingsConfigDict(extra="allow")


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
    roots: Dict[str, LabeledRootPathsConfig] = Field(default_factory=dict)

    @field_validator("input_root", "output_root", "config_home", mode="before")
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
        extra="allow",
    )


class AppConfig(BaseSettings):
    """Main application configuration with Pydantic Settings.

    Supports:
    - JSON config files
    - Environment variables (POLYLOGUE_*)
    - Automatic type validation
    """

    paths: PathsConfig = Field(default_factory=PathsConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    exports: ExportsConfig = Field(default_factory=ExportsConfig)

    # Additional config for compatibility
    raw: Dict[str, Any] = Field(default_factory=dict)
    config_path: Optional[Path] = Field(default=None)

    model_config = SettingsConfigDict(
        env_prefix="POLYLOGUE_",
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
                # Map legacy field names from ui section
                if "html" in defaults_data and "html_previews" not in defaults_data:
                    defaults_data["html_previews"] = defaults_data.pop("html")
                if "theme" in defaults_data and "html_theme" not in defaults_data:
                    defaults_data["html_theme"] = defaults_data.pop("theme")
                config_dict["defaults"] = DefaultsConfig(**defaults_data)

            if "index" in data:
                index_data = dict(data["index"])
                # Flatten nested qdrant object to flat field names
                if "qdrant" in index_data and isinstance(index_data["qdrant"], dict):
                    qdrant = index_data.pop("qdrant")
                    if "url" in qdrant and "qdrant_url" not in index_data:
                        index_data["qdrant_url"] = qdrant["url"]
                    if "api_key" in qdrant and "qdrant_api_key" not in index_data:
                        index_data["qdrant_api_key"] = qdrant["api_key"]
                    if "collection" in qdrant and "qdrant_collection" not in index_data:
                        index_data["qdrant_collection"] = qdrant["collection"]
                    if "vector_size" in qdrant and "qdrant_vector_size" not in index_data:
                        index_data["qdrant_vector_size"] = qdrant["vector_size"]
                config_dict["index"] = IndexConfig(**index_data)

            if "exports" in data:
                exports_data = data.get("exports") or {}
                if isinstance(exports_data, dict):
                    config_dict["exports"] = ExportsConfig(**exports_data)

            # Store raw data for compatibility
            config_dict["raw"] = data
            config_dict["config_path"] = path

            return cls(**config_dict)
        except Exception as e:
            # Fallback to default config on parse error
            import sys
            print(f"Warning: Failed to parse config file {path}: {e}", file=sys.stderr)
            return cls()

    @classmethod
    def load(cls) -> "AppConfig":
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


# Type aliases for backward compatibility
Defaults = DefaultsConfig
OutputPaths = OutputPathsConfig


# Convenience function for backward compatibility
def load_configuration() -> AppConfig:
    """Load the application configuration."""
    return AppConfig.load()

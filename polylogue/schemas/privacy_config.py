"""Config-driven privacy for schema generation.

Provides a ``PrivacyConfig`` dataclass that controls all privacy heuristics.
Supports three presets (strict/standard/permissive), field-level overrides,
and value-level glob patterns.  Configuration cascades from XDG base →
project-level → CLI flags.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from polylogue.paths import config_root

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


# ---------------------------------------------------------------------------
# Typing helpers
# ---------------------------------------------------------------------------

PrivacyLevel: TypeAlias = Literal["strict", "standard", "permissive"]
PrivacySettingValue: TypeAlias = str | int | bool | float | list[str] | dict[str, str] | None
PrivacyConfigSection: TypeAlias = dict[str, PrivacySettingValue]
FieldOverride: TypeAlias = dict[str, str]
PatternList: TypeAlias = list[str]


@runtime_checkable
class SchemaPrivacyConfig(Protocol):
    """Runtime contract shared by schema generation privacy guards."""

    @property
    def level(self) -> PrivacyLevel: ...

    @property
    def safe_enum_max_length(self) -> int: ...

    def field_override(self, path: str) -> str | None: ...

    def is_value_allowed(self, value: str) -> bool | None: ...


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------

_PRESETS: dict[PrivacyLevel, dict[str, int | bool]] = {
    "strict": {
        "safe_enum_max_length": 30,
        "high_entropy_min_length": 8,
        "cross_conv_min_count": 5,
        "cross_conv_proportional": True,
    },
    "standard": {
        "safe_enum_max_length": 50,
        "high_entropy_min_length": 10,
        "cross_conv_min_count": 3,
        "cross_conv_proportional": False,
    },
    "permissive": {
        "safe_enum_max_length": 80,
        "high_entropy_min_length": 16,
        "cross_conv_min_count": 1,
        "cross_conv_proportional": False,
    },
}


@dataclass
class PrivacyConfig:
    """Controls all privacy heuristics in schema generation.

    Attributes:
        level: Preset name ("strict", "standard", "permissive").
        safe_enum_max_length: Max string length for enum values.
        high_entropy_min_length: Min token length for high-entropy detection.
        cross_conv_min_count: Minimum conversations a value must appear in.
        cross_conv_proportional: If True, threshold = max(3, corpus_size * 0.02).
        field_overrides: Path glob → "allow" | "deny" | "default".
        allow_value_patterns: Glob patterns for values to always include.
        deny_value_patterns: Glob patterns for values to always reject.
    """

    level: Literal["strict", "standard", "permissive"] = "standard"
    safe_enum_max_length: int = 50
    high_entropy_min_length: int = 10
    cross_conv_min_count: int = 3
    cross_conv_proportional: bool = False
    field_overrides: dict[str, str] = field(default_factory=dict)
    allow_value_patterns: list[str] = field(default_factory=list)
    deny_value_patterns: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Apply preset values as defaults (explicit overrides take precedence)
        preset = _PRESETS.get(self.level, _PRESETS["standard"])
        for attr, _default_val in preset.items():
            # Only apply preset if attribute is still at the dataclass default
            # (i.e. caller didn't override it explicitly)
            # We detect this by checking if the value matches the standard preset
            standard = _PRESETS["standard"]
            if getattr(self, attr) == standard.get(attr) and attr in preset:
                setattr(self, attr, preset[attr])

    def effective_cross_conv_threshold(self, corpus_size: int) -> int:
        """Compute the effective cross-conversation threshold."""
        if self.cross_conv_proportional:
            return max(3, int(corpus_size * 0.02))
        return self.cross_conv_min_count

    def field_override(self, path: str) -> str | None:
        """Return "allow", "deny", or None for a field path."""
        for pattern, action in self.field_overrides.items():
            if fnmatch(path, pattern):
                return action
        return None

    def is_value_allowed(self, value: str) -> bool | None:
        """Check value against allow/deny patterns.

        Returns True (force-allow), False (force-deny), or None (use heuristics).
        """
        for pattern in self.deny_value_patterns:
            if fnmatch(value, pattern):
                return False
        for pattern in self.allow_value_patterns:
            if fnmatch(value, pattern):
                return True
        return None


def load_privacy_config(
    *,
    cli_overrides: PrivacyConfigSection | None = None,
    project_path: Path | None = None,
) -> PrivacyConfig:
    """Load privacy config with XDG base + project-level + CLI cascade.

    Merge order: XDG defaults → project file → CLI flags.
    Field overrides from all levels merge (later wins per path).
    """
    merged: PrivacyConfigSection = {}
    merged_field_overrides: FieldOverride = {}
    merged_allow_patterns: PatternList = []
    merged_deny_patterns: PatternList = []

    # 1. XDG base config
    xdg_path = config_root() / "polylogue" / "schemas.toml"
    if xdg_path.exists():
        section = _load_toml_section(xdg_path)
        _merge_into(section, merged, merged_field_overrides, merged_allow_patterns, merged_deny_patterns)

    # 2. Project-level config
    if project_path is None:
        project_path = Path.cwd()
    project_file = project_path / "polylogue-schemas.toml"
    if project_file.exists():
        section = _load_toml_section(project_file)
        _merge_into(section, merged, merged_field_overrides, merged_allow_patterns, merged_deny_patterns)

    # 3. CLI overrides
    if cli_overrides:
        _merge_into(cli_overrides, merged, merged_field_overrides, merged_allow_patterns, merged_deny_patterns)

    # Build final config
    if merged_field_overrides:
        merged["field_overrides"] = merged_field_overrides
    if merged_allow_patterns:
        merged["allow_value_patterns"] = merged_allow_patterns
    if merged_deny_patterns:
        merged["deny_value_patterns"] = merged_deny_patterns

    return PrivacyConfig(
        level=_privacy_level_value(merged.get("level")),
        safe_enum_max_length=_int_config_value(merged.get("safe_enum_max_length"), default=50),
        high_entropy_min_length=_int_config_value(merged.get("high_entropy_min_length"), default=10),
        cross_conv_min_count=_int_config_value(merged.get("cross_conv_min_count"), default=3),
        cross_conv_proportional=_bool_config_value(merged.get("cross_conv_proportional"), default=False),
        field_overrides=dict(merged_field_overrides),
        allow_value_patterns=list(merged_allow_patterns),
        deny_value_patterns=list(merged_deny_patterns),
    )


def _load_toml_section(path: Path) -> PrivacyConfigSection:
    """Load the [schema.privacy] section from a TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return dict(data.get("schema", {}).get("privacy", {}))


def _merge_into(
    source: PrivacyConfigSection,
    merged: PrivacyConfigSection,
    field_overrides: FieldOverride,
    allow_patterns: PatternList,
    deny_patterns: PatternList,
) -> None:
    """Merge a source dict into the accumulated config."""
    for key, value in source.items():
        if key == "field_overrides" and isinstance(value, dict):
            field_overrides.update({str(pattern): str(action) for pattern, action in value.items()})
        elif key == "allow_value_patterns" and isinstance(value, list):
            allow_patterns.extend(str(item) for item in value)
        elif key == "deny_value_patterns" and isinstance(value, list):
            deny_patterns.extend(str(item) for item in value)
        else:
            merged[key] = value


def _privacy_level_value(value: PrivacySettingValue) -> PrivacyLevel:
    if value == "strict":
        return "strict"
    if value == "standard":
        return "standard"
    if value == "permissive":
        return "permissive"
    return "standard"


def _int_config_value(value: PrivacySettingValue, *, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _bool_config_value(value: PrivacySettingValue, *, default: bool) -> bool:
    return value if isinstance(value, bool) else default


__all__ = [
    "PrivacyConfig",
    "SchemaPrivacyConfig",
    "load_privacy_config",
]

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .paths import CONFIG_HOME, DATA_HOME


CONFIG_VERSION = 1
DEFAULT_CONFIG_NAME = "config.json"
DEFAULT_PROFILE_NAME = "default"
DEFAULT_ARCHIVE_ROOT = DATA_HOME / "archive"

_ALLOWED_TOP_LEVEL_KEYS = {"version", "archive_root", "sources", "profiles"}
_ALLOWED_SOURCE_KEYS = {"name", "type", "path", "folder"}
_ALLOWED_PROFILE_KEYS = {"attachments", "html", "index", "sanitize_html"}
_ALLOWED_SOURCE_TYPES = {"drive", "codex", "claude", "chatgpt", "claude-code"}
_ALLOWED_ATTACHMENTS = {"download", "link", "skip"}
_ALLOWED_HTML = {"auto", "on", "off"}


class ConfigError(RuntimeError):
    pass


@dataclass
class Source:
    name: str
    type: str
    path: Optional[Path] = None
    folder: Optional[str] = None

    def as_dict(self) -> dict:
        payload = {"name": self.name, "type": self.type}
        if self.path is not None:
            payload["path"] = str(self.path)
        if self.folder is not None:
            payload["folder"] = self.folder
        return payload


@dataclass
class Profile:
    attachments: str = "download"
    html: str = "auto"
    index: bool = True
    sanitize_html: bool = False

    def as_dict(self) -> dict:
        return {
            "attachments": self.attachments,
            "html": self.html,
            "index": self.index,
            "sanitize_html": self.sanitize_html,
        }


@dataclass
class Config:
    version: int
    archive_root: Path
    sources: List[Source]
    profiles: Dict[str, Profile]
    path: Path

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "archive_root": str(self.archive_root),
            "sources": [source.as_dict() for source in self.sources],
            "profiles": {name: profile.as_dict() for name, profile in self.profiles.items()},
        }


def _config_path(explicit: Optional[Path] = None) -> Path:
    env_path = os.environ.get("POLYLOGUE_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    if explicit:
        return explicit.expanduser()
    return CONFIG_HOME / DEFAULT_CONFIG_NAME


def _ensure_keys(data: dict, *, allowed: Iterable[str], context: str) -> None:
    unknown = set(data.keys()) - set(allowed)
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise ConfigError(f"Unknown {context} key(s): {keys}")


def _parse_source(raw: dict) -> Source:
    _ensure_keys(raw, allowed=_ALLOWED_SOURCE_KEYS, context="source")
    name = raw.get("name")
    source_type = raw.get("type")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("Source 'name' must be a non-empty string")
    if not isinstance(source_type, str) or source_type not in _ALLOWED_SOURCE_TYPES:
        raise ConfigError(f"Source '{name}' has invalid type '{source_type}'")
    path = raw.get("path")
    folder = raw.get("folder")
    if source_type == "drive":
        if not isinstance(folder, str) or not folder.strip():
            raise ConfigError(f"Drive source '{name}' requires 'folder'")
        if path is not None:
            raise ConfigError(f"Drive source '{name}' must not set 'path'")
    else:
        if not isinstance(path, str) or not path.strip():
            raise ConfigError(f"Source '{name}' requires 'path'")
        if folder is not None:
            raise ConfigError(f"Source '{name}' must not set 'folder'")
    return Source(
        name=name.strip(),
        type=source_type,
        path=Path(path).expanduser() if isinstance(path, str) else None,
        folder=folder.strip() if isinstance(folder, str) else None,
    )


def _parse_profile(name: str, raw: dict) -> Profile:
    if not isinstance(raw, dict):
        raise ConfigError(f"Profile '{name}' must be a mapping")
    _ensure_keys(raw, allowed=_ALLOWED_PROFILE_KEYS, context=f"profile '{name}'")
    attachments = raw.get("attachments", "download")
    html = raw.get("html", "auto")
    index = raw.get("index", True)
    sanitize_html = raw.get("sanitize_html", False)
    if attachments not in _ALLOWED_ATTACHMENTS:
        raise ConfigError(f"Profile '{name}' has invalid attachments policy '{attachments}'")
    if html not in _ALLOWED_HTML:
        raise ConfigError(f"Profile '{name}' has invalid html mode '{html}'")
    if not isinstance(index, bool):
        raise ConfigError(f"Profile '{name}' has invalid index flag")
    if not isinstance(sanitize_html, bool):
        raise ConfigError(f"Profile '{name}' has invalid sanitize_html flag")
    return Profile(
        attachments=attachments,
        html=html,
        index=index,
        sanitize_html=sanitize_html,
    )


def default_config(path: Optional[Path] = None, *, archive_root: Optional[Path] = None) -> Config:
    config_path = _config_path(path)
    env_root = os.environ.get("POLYLOGUE_ARCHIVE_ROOT")
    if archive_root:
        root = archive_root.expanduser()
    elif env_root:
        root = Path(env_root).expanduser()
    else:
        root = DEFAULT_ARCHIVE_ROOT
    profiles = {DEFAULT_PROFILE_NAME: Profile()}
    return Config(
        version=CONFIG_VERSION,
        archive_root=root,
        sources=[],
        profiles=profiles,
        path=config_path,
    )


def load_config(path: Optional[Path] = None) -> Config:
    config_path = _config_path(path)
    if not config_path.exists():
        raise ConfigError(f"Config not found: {config_path}. Run 'polylogue config init'.")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError("Config payload must be a JSON object")
    _ensure_keys(raw, allowed=_ALLOWED_TOP_LEVEL_KEYS, context="config")
    version = raw.get("version")
    if version != CONFIG_VERSION:
        raise ConfigError(f"Unsupported config version '{version}', expected {CONFIG_VERSION}")
    archive_root = raw.get("archive_root")
    if not isinstance(archive_root, str) or not archive_root.strip():
        raise ConfigError("Config 'archive_root' must be a non-empty string")
    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, list):
        raise ConfigError("Config 'sources' must be a list")
    sources = [_parse_source(entry) for entry in sources_raw]
    profiles_raw = raw.get("profiles")
    if not isinstance(profiles_raw, dict) or not profiles_raw:
        raise ConfigError("Config 'profiles' must be a non-empty object")
    profiles = {name: _parse_profile(name, payload) for name, payload in profiles_raw.items()}

    env_root = os.environ.get("POLYLOGUE_ARCHIVE_ROOT")
    root = Path(env_root).expanduser() if env_root else Path(archive_root).expanduser()

    return Config(
        version=CONFIG_VERSION,
        archive_root=root,
        sources=sources,
        profiles=profiles,
        path=config_path,
    )


def resolve_profile(config: Config, override: Optional[str] = None) -> Tuple[str, Profile]:
    name = override or os.environ.get("POLYLOGUE_PROFILE") or DEFAULT_PROFILE_NAME
    if name not in config.profiles:
        raise ConfigError(f"Profile '{name}' not found in config")
    return name, config.profiles[name]


def write_config(config: Config) -> None:
    config.path.parent.mkdir(parents=True, exist_ok=True)
    config.path.write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")


def update_config(config: Config, *, archive_root: Optional[Path] = None) -> Config:
    if archive_root is not None:
        config.archive_root = archive_root.expanduser()
    return config


def update_profile(config: Config, profile_name: str, field: str, value: str) -> Config:
    if profile_name not in config.profiles:
        raise ConfigError(f"Profile '{profile_name}' not found in config")
    profile = config.profiles[profile_name]
    if field == "attachments":
        if value not in _ALLOWED_ATTACHMENTS:
            raise ConfigError(f"Invalid attachments policy '{value}'")
        profile.attachments = value
    elif field == "html":
        if value not in _ALLOWED_HTML:
            raise ConfigError(f"Invalid html mode '{value}'")
        profile.html = value
    elif field == "index":
        if value.lower() not in {"true", "false"}:
            raise ConfigError("Index must be true/false")
        profile.index = value.lower() == "true"
    elif field == "sanitize_html":
        if value.lower() not in {"true", "false"}:
            raise ConfigError("sanitize_html must be true/false")
        profile.sanitize_html = value.lower() == "true"
    else:
        raise ConfigError(f"Unknown profile field '{field}'")
    return config


def update_source(config: Config, source_name: str, field: str, value: str) -> Config:
    matches = [source for source in config.sources if source.name == source_name]
    if not matches:
        raise ConfigError(f"Source '{source_name}' not found")
    source = matches[0]
    if field == "path":
        if source.type == "drive":
            raise ConfigError("Drive sources do not accept 'path'")
        source.path = Path(value).expanduser()
    elif field == "folder":
        if source.type != "drive":
            raise ConfigError("Non-drive sources do not accept 'folder'")
        source.folder = value
    elif field == "type":
        if value not in _ALLOWED_SOURCE_TYPES:
            raise ConfigError(f"Invalid source type '{value}'")
        source.type = value
    else:
        raise ConfigError(f"Unknown source field '{field}'")
    return config

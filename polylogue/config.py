from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .paths import CONFIG_HOME, DATA_HOME


CONFIG_VERSION = 2
DEFAULT_CONFIG_NAME = "config.json"
DEFAULT_ARCHIVE_ROOT = DATA_HOME / "archive"
DEFAULT_INBOX_ROOT = DATA_HOME / "inbox"

_ALLOWED_TOP_LEVEL_KEYS = {"version", "archive_root", "render_root", "sources"}
_ALLOWED_SOURCE_KEYS = {"name", "path", "folder"}


class ConfigError(RuntimeError):
    pass


@dataclass
class Source:
    name: str
    path: Optional[Path] = None
    folder: Optional[str] = None

    def as_dict(self) -> dict:
        payload = {"name": self.name}
        if self.path is not None:
            payload["path"] = str(self.path)
        if self.folder is not None:
            payload["folder"] = self.folder
        return payload

    @property
    def is_drive(self) -> bool:
        return self.folder is not None


@dataclass
class Config:
    version: int
    archive_root: Path
    render_root: Path
    sources: List[Source]
    path: Path

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "archive_root": str(self.archive_root),
            "render_root": str(self.render_root),
            "sources": [source.as_dict() for source in self.sources],
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
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("Source 'name' must be a non-empty string")
    path = raw.get("path")
    folder = raw.get("folder")
    if path is not None and folder is not None:
        raise ConfigError(f"Source '{name}' must not set both 'path' and 'folder'")
    if folder is not None:
        if not isinstance(folder, str) or not folder.strip():
            raise ConfigError(f"Drive source '{name}' requires non-empty 'folder'")
    else:
        if not isinstance(path, str) or not path.strip():
            raise ConfigError(f"Source '{name}' requires non-empty 'path'")
    return Source(
        name=name.strip(),
        path=Path(path).expanduser() if isinstance(path, str) else None,
        folder=folder.strip() if isinstance(folder, str) else None,
    )


def default_config(
    path: Optional[Path] = None,
    *,
    archive_root: Optional[Path] = None,
    render_root: Optional[Path] = None,
) -> Config:
    config_path = _config_path(path)
    env_root = os.environ.get("POLYLOGUE_ARCHIVE_ROOT")
    env_render_root = os.environ.get("POLYLOGUE_RENDER_ROOT")
    if archive_root:
        root = archive_root.expanduser()
    elif env_root:
        root = Path(env_root).expanduser()
    else:
        root = DEFAULT_ARCHIVE_ROOT
    if render_root:
        resolved_render_root = render_root.expanduser()
    elif env_render_root:
        resolved_render_root = Path(env_render_root).expanduser()
    else:
        resolved_render_root = root / "render"
    sources = [Source(name="inbox", path=DEFAULT_INBOX_ROOT)]
    return Config(
        version=CONFIG_VERSION,
        archive_root=root,
        render_root=resolved_render_root,
        sources=sources,
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
    render_root = raw.get("render_root")
    if render_root is not None and (not isinstance(render_root, str) or not render_root.strip()):
        raise ConfigError("Config 'render_root' must be a non-empty string when provided")
    sources_raw = raw.get("sources")
    if not isinstance(sources_raw, list):
        raise ConfigError("Config 'sources' must be a list")
    sources = [_parse_source(entry) for entry in sources_raw]
    names = [source.name for source in sources]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        dup_list = ", ".join(sorted(duplicates))
        raise ConfigError(f"Duplicate source name(s): {dup_list}")

    env_root = os.environ.get("POLYLOGUE_ARCHIVE_ROOT")
    root = Path(env_root).expanduser() if env_root else Path(archive_root).expanduser()
    env_render_root = os.environ.get("POLYLOGUE_RENDER_ROOT")
    if env_render_root:
        resolved_render_root = Path(env_render_root).expanduser()
    elif isinstance(render_root, str) and render_root.strip():
        resolved_render_root = Path(render_root).expanduser()
    else:
        resolved_render_root = root / "render"

    return Config(
        version=CONFIG_VERSION,
        archive_root=root,
        render_root=resolved_render_root,
        sources=sources,
        path=config_path,
    )


def write_config(config: Config) -> None:
    config.path.parent.mkdir(parents=True, exist_ok=True)
    config.path.write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")


def update_config(
    config: Config,
    *,
    archive_root: Optional[Path] = None,
    render_root: Optional[Path] = None,
) -> Config:
    if archive_root is not None:
        config.archive_root = archive_root.expanduser()
    if render_root is not None:
        config.render_root = render_root.expanduser()
    return config


def update_source(config: Config, source_name: str, field: str, value: str) -> Config:
    matches = [source for source in config.sources if source.name == source_name]
    if not matches:
        raise ConfigError(f"Source '{source_name}' not found")
    source = matches[0]
    if field == "path":
        source.path = Path(value).expanduser()
        source.folder = None
    elif field == "folder":
        source.folder = value
        source.path = None
    else:
        raise ConfigError(f"Unknown source field '{field}'")
    return config

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .paths import CONFIG_HOME, DATA_HOME

CONFIG_VERSION = 2
DEFAULT_CONFIG_NAME = "config.json"
DEFAULT_ARCHIVE_ROOT = DATA_HOME / "archive"
DEFAULT_INBOX_ROOT = DATA_HOME / "inbox"

_ALLOWED_TOP_LEVEL_KEYS = {"version", "archive_root", "render_root", "sources", "template_path"}
_ALLOWED_SOURCE_KEYS = {"name", "path", "folder"}


class ConfigError(RuntimeError):
    pass


@dataclass
class Source:
    name: str
    path: Path | None = None
    folder: str | None = None

    def __post_init__(self):
        """Validate source configuration."""
        # Name validation
        if not self.name or not self.name.strip():
            raise ValueError("Source name cannot be empty")
        self.name = self.name.strip()

        # Path/folder validation
        has_path = self.path is not None
        has_folder = self.folder is not None and self.folder.strip()

        if not has_path and not has_folder:
            raise ValueError(f"Source '{self.name}' must have either 'path' or 'folder'")
        if has_path and has_folder:
            raise ValueError(f"Source '{self.name}' cannot have both 'path' and 'folder' (ambiguous)")

        # Normalize folder
        if self.folder:
            self.folder = self.folder.strip()

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
    sources: list[Source]
    path: Path
    template_path: Path | None = None

    def __post_init__(self):
        """Validate config invariants."""
        # Check for duplicate source names
        names = [s.name for s in self.sources]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            dup_list = ", ".join(sorted(duplicates))
            raise ConfigError(f"Duplicate source name(s): {dup_list}")

    def as_dict(self) -> dict:
        payload = {
            "version": self.version,
            "archive_root": str(self.archive_root),
            "render_root": str(self.render_root),
            "sources": [source.as_dict() for source in self.sources],
        }
        if self.template_path:
            payload["template_path"] = str(self.template_path)
        return payload


def _config_path(explicit: Path | None = None) -> Path:
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
    path: Path | None = None,
    *,
    archive_root: Path | None = None,
    render_root: Path | None = None,
    template_path: Path | None = None,
) -> Config:
    config_path = _config_path(path)
    env_root = os.environ.get("POLYLOGUE_ARCHIVE_ROOT")
    env_render_root = os.environ.get("POLYLOGUE_RENDER_ROOT")
    env_template_path = os.environ.get("POLYLOGUE_TEMPLATE_PATH")
    
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
        
    if template_path:
        resolved_template_path = template_path.expanduser()
    elif env_template_path:
        resolved_template_path = Path(env_template_path).expanduser()
    else:
        resolved_template_path = None

    sources = [Source(name="inbox", path=DEFAULT_INBOX_ROOT)]
    return Config(
        version=CONFIG_VERSION,
        archive_root=root,
        render_root=resolved_render_root,
        sources=sources,
        path=config_path,
        template_path=resolved_template_path,
    )


def load_config(path: Path | None = None) -> Config:
    config_path = _config_path(path)
    if not config_path.exists():
        raise ConfigError(f"Config not found: {config_path}. Run 'polylogue config init'.")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in config file {config_path}: {exc}") from exc
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
    
    template_path_raw = raw.get("template_path")
    if template_path_raw is not None and (not isinstance(template_path_raw, str) or not template_path_raw.strip()):
        raise ConfigError("Config 'template_path' must be a non-empty string when provided")

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
        
    env_template_path = os.environ.get("POLYLOGUE_TEMPLATE_PATH")
    if env_template_path:
        resolved_template_path = Path(env_template_path).expanduser()
    elif isinstance(template_path_raw, str) and template_path_raw.strip():
        resolved_template_path = Path(template_path_raw).expanduser()
    else:
        resolved_template_path = None

    return Config(
        version=CONFIG_VERSION,
        archive_root=root,
        render_root=resolved_render_root,
        sources=sources,
        path=config_path,
        template_path=resolved_template_path,
    )


def write_config(config: Config) -> None:
    config.path.parent.mkdir(parents=True, exist_ok=True)
    config.path.write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")


def update_config(
    config: Config,
    *,
    archive_root: Path | None = None,
    render_root: Path | None = None,
) -> Config:
    """Update config paths, returning a new Config instance.

    This function returns a new Config object with updated values rather than
    mutating the input config in place. This makes the API more predictable
    and prevents unintended side effects.

    Args:
        config: The config to update (not modified).
        archive_root: If provided, set as the new archive_root (will be expanded).
        render_root: If provided, set as the new render_root (will be expanded).

    Returns:
        A new Config object with the updated values. The original config is unchanged.
    """
    from dataclasses import replace

    updates = {}
    if archive_root is not None:
        updates["archive_root"] = archive_root.expanduser()
    if render_root is not None:
        updates["render_root"] = render_root.expanduser()

    return replace(config, **updates) if updates else config


def update_source(config: Config, source_name: str, field: str, value: str) -> Config:
    """Update a source's field by mutating the config.sources list in place.

    WARNING: Unlike update_config(), this function mutates the config object's sources
    list by modifying the matching Source object in place. The config parameter is
    returned for convenience, but it has been modified.

    Args:
        config: The config to update (WILL BE MUTATED - sources list modified in place).
        source_name: Name of the source to update.
        field: Field to update ('path' or 'folder').
        value: New value for the field.

    Returns:
        The same config object that was passed in (now mutated).

    Raises:
        ConfigError: If source_name is not found or field is unknown.
    """
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

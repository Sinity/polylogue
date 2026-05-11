"""Runtime configuration derived from filesystem defaults and env overrides.

Precedence (highest wins):
  1. CLI flag overrides
  2. POLYLOGUE_* environment variables
  3. polylogue.toml (XDG_CONFIG_HOME or project root)
  4. Hardcoded defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import tomllib

from .errors import PolylogueError
from .paths import (
    GEMINI_DRIVE_FOLDER,
    archive_root,
    config_home,
    drive_cache_path,
    drive_credentials_path,
    drive_token_path,
    render_root,
)
from .paths import (
    db_path as default_db_path,
)


class ConfigError(PolylogueError):
    """Configuration error."""


@dataclass
class Source:
    """A conversation source (local path, Drive folder, or both)."""

    name: str
    path: Path | None = None
    folder: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Source name cannot be empty")
        self.name = self.name.strip()
        has_path = self.path is not None
        has_folder = self.folder is not None and self.folder.strip()
        if not has_path and not has_folder:
            raise ValueError(f"Source '{self.name}' must have either 'path' or 'folder'")
        if self.folder:
            self.folder = self.folder.strip()

    @property
    def is_drive(self) -> bool:
        return self.folder is not None


@dataclass
class DriveConfig:
    """Google Drive OAuth configuration."""

    credentials_path: Path = field(default_factory=drive_credentials_path)
    token_path: Path = field(default_factory=drive_token_path)
    retry_count: int = 3
    timeout: int = 30


@dataclass
class IndexConfig:
    """Search indexing configuration."""

    voyage_api_key: str | None = None

    @classmethod
    def from_env(cls) -> IndexConfig:
        """Load IndexConfig from environment variables."""
        return cls(
            voyage_api_key=os.environ.get("VOYAGE_API_KEY"),
        )


@dataclass
class Config:
    """Application configuration derived from paths and source discovery."""

    archive_root: Path
    render_root: Path
    sources: list[Source]
    db_path: Path = field(default_factory=default_db_path)
    drive_config: DriveConfig | None = None
    index_config: IndexConfig | None = None

    def __post_init__(self) -> None:
        # Paths must be absolute. Relative paths are interpreted against the
        # caller's CWD and silently change meaning across processes (CLI vs
        # service vs MCP server). Catch the misuse at construction.
        for attr in ("archive_root", "render_root", "db_path"):
            value = getattr(self, attr)
            if not isinstance(value, Path):
                raise ConfigError(f"Config.{attr} must be a Path, got {type(value).__name__}")
            if not value.is_absolute():
                raise ConfigError(f"Config.{attr} must be an absolute path, got {value!r}")

    def with_sources(self, sources: list[Source]) -> Config:
        return Config(
            archive_root=self.archive_root,
            render_root=self.render_root,
            sources=sources,
            db_path=self.db_path,
            drive_config=self.drive_config,
            index_config=self.index_config,
        )


def get_sources() -> list[Source]:
    """Return the configured conversation sources.

    Delegates to ``default_sources()`` for local watch roots (Claude Code,
    Codex, inbox), then adds Drive/Gemini if configured.  Daemon and CLI
    share the same source discovery through this function.
    """
    from polylogue.sources.live.watcher import default_sources

    watch_sources = default_sources()
    sources: list[Source] = [Source(name=ws.name, path=ws.root) for ws in watch_sources if ws.exists()]

    gemini_cache = drive_cache_path() / "gemini"
    if gemini_cache.exists() or drive_credentials_path().exists() or drive_token_path().exists():
        sources.append(
            Source(
                name="aistudio",
                folder=GEMINI_DRIVE_FOLDER,
                path=gemini_cache,
            )
        )
    return sources


def get_drive_config() -> DriveConfig:
    """Return Drive configuration with default paths."""
    return DriveConfig()


def get_index_config() -> IndexConfig:
    """Return index configuration from environment."""
    return IndexConfig.from_env()


def get_config() -> Config:
    """Return the effective runtime configuration."""
    return Config(
        archive_root=archive_root(),
        render_root=render_root(),
        sources=get_sources(),
        db_path=default_db_path(),
        drive_config=get_drive_config(),
        index_config=get_index_config(),
    )


# ---------------------------------------------------------------------------
# TOML config file support (#829)
# ---------------------------------------------------------------------------


@dataclass
class PolylogueConfig:
    """Typed configuration loaded from TOML with env/CLI overrides.

    Wraps a resolved config dict with attribute access for known keys.
    The underlying dict is available via ``raw`` for unknown keys.
    """

    _data: dict[str, object] = field(default_factory=dict)

    @property
    def raw(self) -> dict[str, object]:
        return self._data

    @property
    def archive_root(self) -> str:
        return str(self._data.get("archive_root", ""))

    @property
    def daemon_url(self) -> str:
        return str(self._data.get("daemon_url", "http://127.0.0.1:8766"))

    @property
    def daemon_host(self) -> str:
        return str(self._data.get("daemon_host", "127.0.0.1"))

    @property
    def daemon_port(self) -> int:
        return int(str(self._data.get("daemon_port", 8766)))

    @property
    def embedding_enabled(self) -> bool:
        return bool(self._data.get("embedding_enabled"))

    @property
    def embedding_model(self) -> str:
        return str(self._data.get("embedding_model", "voyage-4"))

    @property
    def embedding_dimension(self) -> int:
        return int(str(self._data.get("embedding_dimension", 1024)))

    @property
    def embedding_max_cost_usd(self) -> float:
        return float(str(self._data.get("embedding_max_cost_usd", 0.0)))

    @property
    def voyage_api_key(self) -> str | None:
        v = self._data.get("voyage_api_key")
        return v if isinstance(v, str) and v else None

    @property
    def log_level(self) -> str:
        return str(self._data.get("log_level", "INFO"))

    @property
    def force_plain(self) -> bool:
        return bool(self._data.get("force_plain"))

    @property
    def schema_validation(self) -> str:
        return str(self._data.get("schema_validation", "advisory"))

    @property
    def slow_query_notice_seconds(self) -> float | None:
        v = self._data.get("slow_query_notice_seconds")
        return float(str(v)) if v is not None else None

    @property
    def api_host(self) -> str:
        return str(self._data.get("api_host", "127.0.0.1"))

    @property
    def api_port(self) -> int:
        return int(str(self._data.get("api_port", 8766)))

    @property
    def api_auth_token(self) -> str | None:
        v = self._data.get("api_auth_token")
        return v if isinstance(v, str) and v else None

    @property
    def browser_capture_port(self) -> int:
        return int(str(self._data.get("browser_capture_port", 8765)))

    @property
    def browser_capture_allowed_origins(self) -> str:
        return str(self._data.get("browser_capture_allowed_origins", "127.0.0.1"))

    @property
    def notification_backend(self) -> str:
        return str(self._data.get("notification_backend", "log"))

    @property
    def health_check_interval_s(self) -> int:
        return int(str(self._data.get("health_check_interval_s", 300)))

    @property
    def health_check_tiers(self) -> str:
        return str(self._data.get("health_check_tiers", "fast,medium"))

    @property
    def watch_debounce_s(self) -> float:
        return float(str(self._data.get("watch_debounce_s", 2.0)))

    @property
    def browser_capture_host(self) -> str:
        return str(self._data.get("browser_capture_host", "127.0.0.1"))

    @property
    def browser_capture_spool_path(self) -> str:
        return str(self._data.get("browser_capture_spool_path", ""))

    @property
    def browser_capture_auth_token(self) -> str | None:
        v = self._data.get("browser_capture_auth_token")
        return v if isinstance(v, str) and v else None

    @property
    def browser_capture_allow_remote(self) -> bool:
        return bool(self._data.get("browser_capture_allow_remote"))

    @property
    def source_roots(self) -> tuple[str, ...]:
        v = self._data.get("source_roots")
        if isinstance(v, (list, tuple)):
            return tuple(str(item) for item in v)
        if isinstance(v, str) and v.strip():
            return tuple(s.strip() for s in v.split(",") if s.strip())
        return ()

    def get(self, key: str, default: object = None) -> object:
        return self._data.get(key, default)


def _config_file_path() -> Path | None:
    """Resolve the polylogue.toml path.

    Returns the first existing path from:
      1. ``POLYLOGUE_CONFIG`` env var
      2. ``{XDG_CONFIG_HOME}/polylogue/polylogue.toml``
      3. ``{project_root}/polylogue.toml``
    Returns None if no file exists.
    """
    override = os.environ.get("POLYLOGUE_CONFIG")
    if override:
        p = Path(override)
        return p if p.is_file() else None

    xdg_path = config_home() / "polylogue.toml"
    if xdg_path.is_file():
        return xdg_path

    project_path = Path("polylogue.toml")
    if project_path.is_file():
        return project_path

    return None


def load_polylogue_config(
    *,
    config_path: Path | None = None,
    cli_overrides: dict[str, object] | None = None,
) -> PolylogueConfig:
    """Load resolved Polylogue config with four-layer precedence.

    Returns a typed ``PolylogueConfig`` with attribute access for
    known keys and ``.get()`` / ``.raw`` for everything else.
    """
    cfg: dict[str, object] = {
        "archive_root": str(archive_root()),
        "daemon_url": "http://127.0.0.1:8766",
        "daemon_host": "127.0.0.1",
        "daemon_port": 8766,
        "api_host": "127.0.0.1",
        "api_port": 8766,
        "api_auth_token": None,
        "browser_capture_port": 8765,
        "browser_capture_allowed_origins": "127.0.0.1",
        # Stays opt-in: the daemon embed stage is gated separately on a
        # config TOML flag or POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS so that
        # supplying VOYAGE_API_KEY (e.g., for one-off CLI use) does not
        # incur ongoing daemon-driven spend.
        "embedding_enabled": False,
        "embedding_model": "voyage-4",
        "embedding_dimension": 1024,
        # Soft monthly cap on embedding spend. 0 = unlimited; the default
        # below is intentionally low enough to act as a safety net for a
        # first-time user without an explicit configuration.
        "embedding_max_cost_usd": 5.0,
        "log_level": "INFO",
        "force_plain": False,
        "schema_validation": "advisory",
        "notification_backend": "log",
        "health_check_interval_s": 300,
        "health_check_tiers": "fast,medium",
        "watch_debounce_s": 2.0,
        "browser_capture_host": "127.0.0.1",
        "browser_capture_spool_path": "",
        "browser_capture_auth_token": None,
        "browser_capture_allow_remote": False,
        "source_roots": (),
    }

    # Layer 2: TOML file
    path = config_path or _config_file_path()
    if path is not None:
        try:
            with open(path, "rb") as fh:
                toml_data = tomllib.load(fh)
            _merge_toml(cfg, toml_data)
        except (OSError, tomllib.TOMLDecodeError):
            pass

    # Layer 3: Environment variables
    _apply_env_overrides(cfg)

    # Layer 4: CLI overrides
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                cfg[key] = value

    return PolylogueConfig(_data=cfg)


def _merge_toml(cfg: dict[str, object], toml_data: dict[str, object]) -> None:
    """Merge TOML sections into the flat config dict.

    Each section maps short keys (as written in TOML) to canonical flat keys
    in ``cfg``. For example, ``[archive] root = "/x"`` maps to
    ``cfg["archive_root"] = "/x"``.
    """
    section_keys: dict[str, dict[str, str]] = {
        "archive": {"root": "archive_root"},
        "daemon": {
            "host": "daemon_host",
            "port": "daemon_port",
        },
        "daemon.api": {
            "host": "api_host",
            "port": "api_port",
            "auth_token": "api_auth_token",
        },
        "daemon.browser_capture": {
            "host": "browser_capture_host",
            "port": "browser_capture_port",
            "allowed_origins": "browser_capture_allowed_origins",
            "allow_remote": "browser_capture_allow_remote",
            "auth_token": "browser_capture_auth_token",
        },
        "daemon.watch": {
            "debounce_s": "watch_debounce_s",
        },
        "sources": {
            "roots": "source_roots",
        },
        "embedding": {
            "enabled": "embedding_enabled",
            "model": "embedding_model",
            "dimension": "embedding_dimension",
            "max_cost_usd": "embedding_max_cost_usd",
        },
        "logging": {
            "level": "log_level",
            "force_plain": "force_plain",
        },
        "notifications": {"backend": "notification_backend"},
        "health": {
            "check_interval_s": "health_check_interval_s",
            "check_tiers": "health_check_tiers",
        },
    }
    for section, key_map in section_keys.items():
        # Walk dotted paths for nested TOML sections like [daemon.api]
        section_data: object = toml_data
        for part in section.split("."):
            if isinstance(section_data, dict):
                section_data = section_data.get(part)
            else:
                section_data = None
                break
        if isinstance(section_data, dict):
            for short_key, flat_key in key_map.items():
                if short_key in section_data:
                    value = section_data[short_key]
                    if isinstance(value, list):
                        cfg[flat_key] = tuple(value)
                    else:
                        cfg[flat_key] = value


def _apply_env_overrides(cfg: dict[str, object]) -> None:
    """Apply POLYLOGUE_* environment variable overrides."""
    env_map = {
        "POLYLOGUE_ARCHIVE_ROOT": "archive_root",
        "POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS": "embedding_enabled",
        "VOYAGE_API_KEY": "voyage_api_key",
        "POLYLOGUE_FORCE_PLAIN": "force_plain",
        "POLYLOGUE_SLOW_QUERY_NOTICE_SECONDS": "slow_query_notice_seconds",
        "POLYLOGUE_SCHEMA_VALIDATION": "schema_validation",
        "POLYLOGUE_NOTIFICATION_BACKEND": "notification_backend",
        "POLYLOGUE_HEALTH_CHECK_INTERVAL_S": "health_check_interval_s",
        "POLYLOGUE_HEALTH_CHECK_TIERS": "health_check_tiers",
        "POLYLOGUE_API_HOST": "api_host",
        "POLYLOGUE_API_PORT": "api_port",
        "POLYLOGUE_API_AUTH_TOKEN": "api_auth_token",
        "POLYLOGUE_BROWSER_CAPTURE_PORT": "browser_capture_port",
        "POLYLOGUE_BROWSER_CAPTURE_HOST": "browser_capture_host",
        "POLYLOGUE_WATCH_DEBOUNCE_S": "watch_debounce_s",
    }
    # Keys that must be stored as int
    _int_keys = {"api_port", "daemon_port", "browser_capture_port", "health_check_interval_s"}
    for env_var, cfg_key in env_map.items():
        value = os.environ.get(env_var)
        if value is not None:
            # Coerce booleans
            if value.lower() in ("1", "true", "yes"):
                cfg[cfg_key] = True
            elif value.lower() in ("0", "false", "no"):
                cfg[cfg_key] = False
            elif cfg_key in _int_keys:
                with __import__("contextlib").suppress(ValueError):
                    cfg[cfg_key] = int(value)
            else:
                cfg[cfg_key] = value


def format_config_toml(cfg: dict[str, object]) -> str:
    """Render loaded config as TOML for display.

    Round-trips with :func:`_merge_toml`: the generated TOML uses short keys
    inside their sections (``[archive] root = ...``) so the output can be
    written back as a ``polylogue.toml`` file and re-loaded.
    """
    # (section_name, [(short_key, flat_key), ...])
    sections_layout: list[tuple[str, list[tuple[str, str]]]] = [
        ("archive", [("root", "archive_root")]),
        (
            "daemon",
            [
                ("host", "daemon_host"),
                ("port", "daemon_port"),
            ],
        ),
        (
            "daemon.api",
            [
                ("host", "api_host"),
                ("port", "api_port"),
                ("auth_token", "api_auth_token"),
            ],
        ),
        (
            "daemon.browser_capture",
            [
                ("host", "browser_capture_host"),
                ("port", "browser_capture_port"),
                ("allowed_origins", "browser_capture_allowed_origins"),
                ("allow_remote", "browser_capture_allow_remote"),
                ("auth_token", "browser_capture_auth_token"),
            ],
        ),
        (
            "daemon.watch",
            [("debounce_s", "watch_debounce_s")],
        ),
        (
            "sources",
            [("roots", "source_roots")],
        ),
        (
            "embedding",
            [
                ("enabled", "embedding_enabled"),
                ("model", "embedding_model"),
                ("dimension", "embedding_dimension"),
                ("max_cost_usd", "embedding_max_cost_usd"),
                ("voyage_api_key", "voyage_api_key"),
            ],
        ),
        (
            "logging",
            [
                ("level", "log_level"),
                ("force_plain", "force_plain"),
            ],
        ),
        ("notifications", [("backend", "notification_backend")]),
        (
            "health",
            [
                ("check_interval_s", "health_check_interval_s"),
                ("check_tiers", "health_check_tiers"),
            ],
        ),
    ]

    lines: list[str] = []
    for section, key_pairs in sections_layout:
        section_lines: list[str] = []
        for short_key, flat_key in key_pairs:
            if flat_key not in cfg:
                continue
            value = cfg[flat_key]
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                items = ", ".join(f'"{v}"' for v in value)
                section_lines.append(f"{short_key} = [{items}]")
            elif isinstance(value, str):
                section_lines.append(f'{short_key} = "{value}"')
            elif isinstance(value, bool):
                section_lines.append(f"{short_key} = {str(value).lower()}")
            else:
                section_lines.append(f"{short_key} = {value}")
        if section_lines:
            lines.append(f"[{section}]")
            lines.extend(section_lines)
            lines.append("")

    return "\n".join(lines)


__all__ = [
    "Config",
    "ConfigError",
    "DriveConfig",
    "IndexConfig",
    "PolylogueConfig",
    "Source",
    "format_config_toml",
    "get_config",
    "get_drive_config",
    "get_index_config",
    "get_sources",
    "load_polylogue_config",
]

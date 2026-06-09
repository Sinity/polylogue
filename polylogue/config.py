"""Runtime configuration derived from filesystem defaults and env overrides.

Precedence (highest wins):
  1. CLI flag overrides
  2. POLYLOGUE_* environment variables
  3. polylogue.toml (XDG_CONFIG_HOME or project root)
  4. Hardcoded defaults
"""

from __future__ import annotations

import os
from collections.abc import Mapping
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
    active_index_db_path as default_db_path,
)


class ConfigError(PolylogueError):
    """Configuration error."""


@dataclass
class Source:
    """A session source (local path, Drive folder, or both)."""

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
    """Return the configured session sources.

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
    _layers: dict[str, str] = field(default_factory=dict)

    @property
    def raw(self) -> dict[str, object]:
        return self._data

    @property
    def layers(self) -> dict[str, str]:
        """Map of config-key -> originating layer name.

        Layer names: ``default``, ``site``, ``user``, ``env``, ``cli``.
        Only keys that have been resolved through the loader appear here;
        keys present only in ``raw`` from external construction may be
        absent.
        """
        return dict(self._layers)

    def layer_of(self, key: str) -> str:
        """Return the layer name that supplied ``key`` (default: ``default``)."""
        return self._layers.get(key, "default")

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
    def observability_enabled(self) -> bool:
        """Return whether the OTLP HTTP receiver routes are accepted.

        The receiver is OFF by default (closes #1604 — the routes were
        previously unconditionally enabled in front of the auth gate
        despite a code comment claiming otherwise). Operators opt in
        via TOML ``observability_enabled = true`` or the env var
        ``POLYLOGUE_OBSERVABILITY_ENABLED=1``.
        """
        return bool(self._data.get("observability_enabled"))

    @property
    def otlp_max_body_bytes(self) -> int:
        """Maximum accepted Content-Length for OTLP POST bodies.

        Default 8 MiB matches typical OTLP exporter batch sizes; clients
        sending more receive 413. Configurable via TOML
        ``otlp_max_body_bytes`` or env ``POLYLOGUE_OTLP_MAX_BODY_BYTES``.
        """
        return int(str(self._data.get("otlp_max_body_bytes", 8 * 1024 * 1024)))

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
    def theme(self) -> str:
        """Resolved CLI theme: ``"dark"``, ``"light"``, or ``"auto"`` (#1274).

        Empty string means "not configured" and lets
        :func:`polylogue.ui.theme.resolve_theme_mode` fall through to
        environment/auto-detection.
        """
        return str(self._data.get("theme", "")).strip().lower()

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
    def notification_webhook_url(self) -> str | None:
        v = self._data.get("notification_webhook_url")
        return v if isinstance(v, str) and v else None

    @property
    def notification_webhook_secret(self) -> str | None:
        v = self._data.get("notification_webhook_secret")
        return v if isinstance(v, str) and v else None

    @property
    def notification_apprise_urls(self) -> tuple[str, ...]:
        v = self._data.get("notification_apprise_urls")
        if isinstance(v, (list, tuple)):
            return tuple(str(s) for s in v if str(s).strip())
        if isinstance(v, str) and v.strip():
            return tuple(s.strip() for s in v.split(",") if s.strip())
        return ()

    @property
    def notification_email_host(self) -> str | None:
        v = self._data.get("notification_email_host")
        return v if isinstance(v, str) and v else None

    @property
    def notification_email_port(self) -> int:
        return int(str(self._data.get("notification_email_port", 587)))

    @property
    def notification_email_from(self) -> str | None:
        v = self._data.get("notification_email_from")
        return v if isinstance(v, str) and v else None

    @property
    def notification_email_to(self) -> tuple[str, ...]:
        v = self._data.get("notification_email_to")
        if isinstance(v, (list, tuple)):
            return tuple(str(s) for s in v if str(s).strip())
        if isinstance(v, str) and v.strip():
            return tuple(s.strip() for s in v.split(",") if s.strip())
        return ()

    @property
    def health_check_interval_s(self) -> int:
        return int(str(self._data.get("health_check_interval_s", 300)))

    @property
    def health_check_tiers(self) -> str:
        return str(self._data.get("health_check_tiers", "fast"))

    @property
    def health_fts_auto_restore(self) -> bool:
        """Whether the daemon health loop should auto-restore missing FTS triggers (#1229).

        When true, :func:`polylogue.daemon.health._check_fts_trigger_drift_fast`
        drops/re-creates the six canonical FTS sync triggers and runs the
        FTS5 ``rebuild`` command in place when drift is detected. The
        check still emits a WARNING-level alert so the operator is told
        the recovery happened. When false (default), drift surfaces as
        CRITICAL with an operator-facing restore command.
        """
        return bool(self._data.get("health_fts_auto_restore", False))

    @property
    def health_blob_integrity_sample_size(self) -> int:
        """Bounded blob-integrity sample size for daemon health checks (#1231)."""
        return int(str(self._data.get("health_blob_integrity_sample_size", 100)))

    @property
    def health_convergence_debt(self) -> dict[str, object]:
        """Raw ``[health.convergence_debt]`` table from polylogue.toml.

        Returned verbatim as the underlying TOML dict (with nested
        ``families`` sub-table) so :mod:`polylogue.daemon.convergence_debt_alert`
        can decode it into typed thresholds without the config layer
        owning the alert vocabulary.
        """
        raw = self._data.get("health_convergence_debt")
        if isinstance(raw, dict):
            return dict(raw)
        return {}

    @property
    def health_cursor_lag(self) -> dict[str, object]:
        """Raw ``[health.cursor_lag]`` table from polylogue.toml (#1232).

        Returned verbatim as the underlying TOML dict (with nested
        ``families`` sub-table) so :mod:`polylogue.daemon.cursor_lag_alert`
        can decode it into typed thresholds without the config layer
        owning the alert vocabulary.
        """
        raw = self._data.get("health_cursor_lag")
        if isinstance(raw, dict):
            return dict(raw)
        return {}

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

    @property
    def subscription_plans(self) -> tuple[dict[str, object], ...]:
        """User-supplied subscription plan rows from ``[[cost.subscription.plans]]``.

        Returns the raw row dicts so callers in :mod:`polylogue.cost.plans` can
        validate them through the typed :class:`SubscriptionPlan` model. The
        config layer deliberately stays loose so a malformed plan entry fails
        loudly at the cost-cluster boundary (with a useful name) rather than
        silently dropping fields here.
        """
        raw = self._data.get("subscription_plans")
        if not isinstance(raw, (list, tuple)):
            return ()
        rows: list[dict[str, object]] = []
        for entry in raw:
            if isinstance(entry, dict):
                rows.append(dict(entry))
        return tuple(rows)


#: Default site-wide configuration path. Overridable through the
#: ``POLYLOGUE_SITE_CONFIG`` env var (primarily for tests and packaging).
DEFAULT_SITE_CONFIG_PATH = Path("/etc/polylogue/polylogue.toml")


def _site_config_path() -> Path | None:
    """Resolve the site-wide ``polylogue.toml`` (layer 2).

    Resolution order:
      1. ``POLYLOGUE_SITE_CONFIG`` env var (explicit override; ``""`` disables)
      2. ``/etc/polylogue/polylogue.toml`` if it exists

    Returns ``None`` when no site config is available.
    """
    override = os.environ.get("POLYLOGUE_SITE_CONFIG")
    if override is not None:
        if not override:
            return None
        p = Path(override)
        return p if p.is_file() else None

    return DEFAULT_SITE_CONFIG_PATH if DEFAULT_SITE_CONFIG_PATH.is_file() else None


def _user_config_path() -> Path | None:
    """Resolve the user-scoped ``polylogue.toml`` (layer 3).

    Resolution order:
      1. ``POLYLOGUE_CONFIG`` env var (explicit override)
      2. ``{XDG_CONFIG_HOME}/polylogue/polylogue.toml``
      3. ``{cwd}/polylogue.toml`` (project-local fallback)

    Returns ``None`` when no user config is found.
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


def _config_file_path() -> Path | None:
    """Back-compat alias for :func:`_user_config_path`.

    Retained because earlier tests reference it indirectly via the loader's
    behavior; new callers should pick the explicit user/site helper.
    """
    return _user_config_path()


def _default_config_values() -> dict[str, object]:
    """Built-in defaults (layer 1).

    Pure function — does not touch the filesystem, env, or CLI state.
    The one exception is ``archive_root``, which derives from XDG paths
    via :func:`archive_root`; that resolution is itself env-driven but
    represents the documented built-in default for an unconfigured
    install.
    """
    return {
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
        # OTLP HTTP receiver and other observability routes (see
        # ``polylogue/daemon/otlp_receiver.py``). Default off; the
        # receiver was previously documented-but-not-actually gated,
        # which made it an unauthenticated write surface under
        # ``--insecure-allow-remote`` (closes #1604).
        "observability_enabled": False,
        "otlp_max_body_bytes": 8 * 1024 * 1024,
        "embedding_model": "voyage-4",
        "embedding_dimension": 1024,
        # Soft monthly cap on embedding spend. 0 = unlimited; the default
        # below is intentionally low enough to act as a safety net for a
        # first-time user without an explicit configuration.
        "embedding_max_cost_usd": 5.0,
        "log_level": "INFO",
        "force_plain": False,
        "theme": "",
        "schema_validation": "advisory",
        "notification_backend": "log",
        "notification_webhook_url": None,
        "notification_webhook_secret": None,
        "notification_apprise_urls": (),
        "notification_email_host": None,
        "notification_email_port": 587,
        "notification_email_username": None,
        "notification_email_password": None,
        "notification_email_from": None,
        "notification_email_to": (),
        "notification_email_subject_prefix": "[polylogue]",
        "notification_email_use_tls": True,
        "notification_email_use_starttls": True,
        "notification_email_max_per_hour": 12,
        "health_check_interval_s": 300,
        "health_check_tiers": "fast",
        "health_fts_auto_restore": False,
        "health_blob_integrity_sample_size": 100,
        "watch_debounce_s": 2.0,
        "browser_capture_host": "127.0.0.1",
        "browser_capture_spool_path": "",
        "browser_capture_auth_token": None,
        "browser_capture_allow_remote": False,
        "source_roots": (),
    }


def _apply_toml_layer(
    cfg: dict[str, object],
    layers: dict[str, str],
    path: Path,
    layer_name: str,
) -> None:
    """Load ``path`` as TOML and merge it into ``cfg``, recording layer source.

    Errors (missing file race, malformed TOML) are swallowed silently so a
    broken site config cannot brick a user's CLI. Surface diagnostics live
    in ``polylogue config --show-layers`` via :func:`describe_config_layers`.
    """
    try:
        with open(path, "rb") as fh:
            toml_data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return

    before = dict(cfg)
    _merge_toml(cfg, toml_data)
    for key, value in cfg.items():
        if before.get(key, _MISSING) != value:
            layers[key] = layer_name


_MISSING = object()


def load_polylogue_config(
    *,
    config_path: Path | None = None,
    site_config_path: Path | None = None,
    cli_overrides: dict[str, object] | None = None,
) -> PolylogueConfig:
    """Load resolved Polylogue config with four-layer precedence.

    Precedence (low → high), per #829:

      1. Built-in defaults (:func:`_default_config_values`).
      2. Site config: ``/etc/polylogue/polylogue.toml`` (overridable via
         ``POLYLOGUE_SITE_CONFIG`` env var or the ``site_config_path``
         keyword).
      3. User config: ``$XDG_CONFIG_HOME/polylogue/polylogue.toml`` or a
         ``polylogue.toml`` in the current working directory; overridable
         via ``POLYLOGUE_CONFIG`` env var or the ``config_path`` keyword.
      4. ``POLYLOGUE_*`` environment variables.
      5. CLI overrides (highest precedence).

    Returns a typed :class:`PolylogueConfig` with attribute access for
    known keys plus ``layer_of()`` / ``layers`` for provenance.
    """
    cfg = _default_config_values()
    layers: dict[str, str] = dict.fromkeys(cfg, "default")

    # Layer 2: site TOML.
    site_path = site_config_path if site_config_path is not None else _site_config_path()
    if site_path is not None and site_path.is_file():
        _apply_toml_layer(cfg, layers, site_path, "site")

    # Layer 3: user TOML.
    user_path = config_path if config_path is not None else _user_config_path()
    if user_path is not None and user_path.is_file():
        _apply_toml_layer(cfg, layers, user_path, "user")

    # Layer 4: environment variables.
    before_env = dict(cfg)
    _apply_env_overrides(cfg)
    for key, value in cfg.items():
        if before_env.get(key, _MISSING) != value:
            layers[key] = "env"

    # Layer 5: CLI overrides (highest precedence).
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                cfg[key] = value
                layers[key] = "cli"

    return PolylogueConfig(_data=cfg, _layers=layers)


def describe_config_layers(
    *,
    config_path: Path | None = None,
    site_config_path: Path | None = None,
) -> dict[str, object]:
    """Return a structured description of the active config layer paths.

    Used by ``polylogue config --show-layers`` to report which physical
    files (if any) supplied the site and user layers. The shape is a
    plain dict so it can be JSON-serialized verbatim.
    """
    site = site_config_path if site_config_path is not None else _site_config_path()
    user = config_path if config_path is not None else _user_config_path()
    return {
        "site": {
            "path": str(site) if site is not None else None,
            "exists": bool(site is not None and site.is_file()),
        },
        "user": {
            "path": str(user) if user is not None else None,
            "exists": bool(user is not None and user.is_file()),
        },
    }


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
        "ui": {
            "theme": "theme",
        },
        "notifications": {
            "backend": "notification_backend",
            "webhook_url": "notification_webhook_url",
            "webhook_secret": "notification_webhook_secret",
            "apprise_urls": "notification_apprise_urls",
        },
        "notifications.email": {
            "host": "notification_email_host",
            "port": "notification_email_port",
            "username": "notification_email_username",
            "password": "notification_email_password",
            "from": "notification_email_from",
            "to": "notification_email_to",
            "subject_prefix": "notification_email_subject_prefix",
            "use_tls": "notification_email_use_tls",
            "use_starttls": "notification_email_use_starttls",
            "max_per_hour": "notification_email_max_per_hour",
        },
        "health": {
            "check_interval_s": "health_check_interval_s",
            "check_tiers": "health_check_tiers",
            "fts_auto_restore": "health_fts_auto_restore",
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

    # [health.convergence_debt] — typed nested table with per-family overrides.
    # See polylogue/daemon/convergence_debt_alert.py for shape and semantics.
    health_section = toml_data.get("health")
    debt_section = health_section.get("convergence_debt") if isinstance(health_section, dict) else None
    if isinstance(debt_section, dict):
        # Stored as a nested dict on the flat config under a reserved key.
        cfg["health_convergence_debt"] = dict(debt_section)

    # [health.cursor_lag] — per-source-family cursor-lag SLO thresholds (#1232).
    # See polylogue/daemon/cursor_lag_alert.py for shape and semantics.
    cursor_lag_section = health_section.get("cursor_lag") if isinstance(health_section, dict) else None
    if isinstance(cursor_lag_section, dict):
        cfg["health_cursor_lag"] = dict(cursor_lag_section)

    # Array-of-tables: [[cost.subscription.plans]].
    cost_section = toml_data.get("cost")
    sub_section = cost_section.get("subscription") if isinstance(cost_section, dict) else None
    plans = sub_section.get("plans") if isinstance(sub_section, dict) else None
    if isinstance(plans, list):
        cfg["subscription_plans"] = tuple(p for p in plans if isinstance(p, dict))


def _apply_env_overrides(cfg: dict[str, object]) -> None:
    """Apply POLYLOGUE_* environment variable overrides."""
    env_map = {
        "POLYLOGUE_ARCHIVE_ROOT": "archive_root",
        "POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS": "embedding_enabled",
        "POLYLOGUE_OBSERVABILITY_ENABLED": "observability_enabled",
        "POLYLOGUE_OTLP_MAX_BODY_BYTES": "otlp_max_body_bytes",
        "VOYAGE_API_KEY": "voyage_api_key",
        "POLYLOGUE_FORCE_PLAIN": "force_plain",
        "POLYLOGUE_THEME": "theme",
        "POLYLOGUE_SLOW_QUERY_NOTICE_SECONDS": "slow_query_notice_seconds",
        "POLYLOGUE_SCHEMA_VALIDATION": "schema_validation",
        "POLYLOGUE_NOTIFICATION_BACKEND": "notification_backend",
        "POLYLOGUE_NOTIFICATION_WEBHOOK_URL": "notification_webhook_url",
        "POLYLOGUE_NOTIFICATION_WEBHOOK_SECRET": "notification_webhook_secret",
        "POLYLOGUE_NOTIFICATION_APPRISE_URLS": "notification_apprise_urls",
        "POLYLOGUE_NOTIFICATION_EMAIL_HOST": "notification_email_host",
        "POLYLOGUE_NOTIFICATION_EMAIL_PORT": "notification_email_port",
        "POLYLOGUE_NOTIFICATION_EMAIL_USERNAME": "notification_email_username",
        "POLYLOGUE_NOTIFICATION_EMAIL_PASSWORD": "notification_email_password",
        "POLYLOGUE_NOTIFICATION_EMAIL_FROM": "notification_email_from",
        "POLYLOGUE_NOTIFICATION_EMAIL_TO": "notification_email_to",
        "POLYLOGUE_HEALTH_CHECK_INTERVAL_S": "health_check_interval_s",
        "POLYLOGUE_HEALTH_CHECK_TIERS": "health_check_tiers",
        "POLYLOGUE_HEALTH_FTS_AUTO_RESTORE": "health_fts_auto_restore",
        "POLYLOGUE_HEALTH_BLOB_INTEGRITY_SAMPLE_SIZE": "health_blob_integrity_sample_size",
        "POLYLOGUE_API_HOST": "api_host",
        "POLYLOGUE_API_PORT": "api_port",
        "POLYLOGUE_API_AUTH_TOKEN": "api_auth_token",
        "POLYLOGUE_BROWSER_CAPTURE_PORT": "browser_capture_port",
        "POLYLOGUE_BROWSER_CAPTURE_HOST": "browser_capture_host",
        "POLYLOGUE_WATCH_DEBOUNCE_S": "watch_debounce_s",
    }
    # Keys that must be stored as int
    _int_keys = {
        "api_port",
        "daemon_port",
        "browser_capture_port",
        "health_check_interval_s",
        "health_blob_integrity_sample_size",
        "notification_email_port",
        "notification_email_max_per_hour",
    }
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


#: Flat config keys whose values are secrets and must never be rendered in
#: cleartext on any display surface (``polylogue config`` toml/json/layers,
#: logs, bug reports). The redaction is keyed on the flat config key so it
#: applies regardless of the display section/short-key mapping.
SECRET_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "voyage_api_key",
        "notification_webhook_secret",
        "notification_email_password",
        "api_auth_token",
        "browser_capture_auth_token",
    }
)

#: Placeholder substituted for a secret that is set to a non-empty value.
SECRET_SET_PLACEHOLDER = "<set>"
#: Placeholder substituted for a secret that is unset / empty.
SECRET_UNSET_PLACEHOLDER = "<unset>"


def is_secret_config_key(flat_key: str) -> bool:
    """Return True when ``flat_key`` holds a secret that must be redacted.

    Matches the explicit :data:`SECRET_CONFIG_KEYS` set plus any key that
    ends in a sensitive suffix (``_api_key``, ``_secret``, ``_password``,
    ``_auth_token``, ``_token``) or a bare ``auth_token``/``password``/
    ``secret`` so newly added secret-bearing keys are redacted by default
    rather than leaking until the set is updated.
    """
    if flat_key in SECRET_CONFIG_KEYS:
        return True
    sensitive_suffixes = ("_api_key", "_secret", "_password", "_auth_token", "_token")
    return flat_key in {"auth_token", "password", "secret"} or flat_key.endswith(sensitive_suffixes)


def redact_secret_value(value: object) -> str:
    """Map a secret config value to a non-revealing presence placeholder."""
    if value is None:
        return SECRET_UNSET_PLACEHOLDER
    if isinstance(value, str) and value == "":
        return SECRET_UNSET_PLACEHOLDER
    return SECRET_SET_PLACEHOLDER


def redact_config_mapping(cfg: Mapping[str, object]) -> dict[str, object]:
    """Return a copy of ``cfg`` with every secret-bearing key redacted.

    Used by JSON and layer-source display surfaces so a secret value never
    appears verbatim in shared config output.
    """
    redacted: dict[str, object] = {}
    for key, value in cfg.items():
        if is_secret_config_key(key):
            redacted[key] = redact_secret_value(value)
        else:
            redacted[key] = value
    return redacted


def format_config_toml(cfg: dict[str, object]) -> str:
    """Render loaded config as TOML for display.

    Round-trips with :func:`_merge_toml`: the generated TOML uses short keys
    inside their sections (``[archive] root = ...``) so the output can be
    written back as a ``polylogue.toml`` file and re-loaded.

    Secret-bearing keys are redacted to a presence placeholder
    (:data:`SECRET_SET_PLACEHOLDER` / :data:`SECRET_UNSET_PLACEHOLDER`) so the
    rendered TOML can be safely pasted into bug reports, and values are
    serialized with a real TOML emitter (``tomli_w``) rather than raw f-string
    interpolation so quotes/backslashes/newlines cannot corrupt the output.
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
        (
            "ui",
            [
                ("theme", "theme"),
            ],
        ),
        (
            "notifications",
            [
                ("backend", "notification_backend"),
                ("webhook_url", "notification_webhook_url"),
                ("webhook_secret", "notification_webhook_secret"),
                ("apprise_urls", "notification_apprise_urls"),
            ],
        ),
        (
            "notifications.email",
            [
                ("host", "notification_email_host"),
                ("port", "notification_email_port"),
                ("username", "notification_email_username"),
                ("password", "notification_email_password"),
                ("from", "notification_email_from"),
                ("to", "notification_email_to"),
                ("subject_prefix", "notification_email_subject_prefix"),
                ("use_tls", "notification_email_use_tls"),
                ("use_starttls", "notification_email_use_starttls"),
                ("max_per_hour", "notification_email_max_per_hour"),
            ],
        ),
        (
            "health",
            [
                ("check_interval_s", "health_check_interval_s"),
                ("check_tiers", "health_check_tiers"),
                ("fts_auto_restore", "health_fts_auto_restore"),
            ],
        ),
    ]

    import tomli_w

    lines: list[str] = []
    for section, key_pairs in sections_layout:
        section_table: dict[str, object] = {}
        for short_key, flat_key in key_pairs:
            if flat_key not in cfg:
                continue
            value = cfg[flat_key]
            if value is None:
                continue
            if is_secret_config_key(flat_key):
                section_table[short_key] = redact_secret_value(value)
                continue
            if isinstance(value, tuple):
                section_table[short_key] = list(value)
            else:
                section_table[short_key] = value
        if section_table:
            lines.append(f"[{section}]")
            # ``tomli_w`` escapes quotes/backslashes/newlines correctly, so a
            # value containing TOML metacharacters cannot break the output.
            rendered = tomli_w.dumps(section_table).rstrip("\n")
            if rendered:
                lines.append(rendered)
            lines.append("")

    return "\n".join(lines)


__all__ = [
    "Config",
    "ConfigError",
    "DEFAULT_SITE_CONFIG_PATH",
    "DriveConfig",
    "IndexConfig",
    "PolylogueConfig",
    "SECRET_CONFIG_KEYS",
    "SECRET_SET_PLACEHOLDER",
    "SECRET_UNSET_PLACEHOLDER",
    "Source",
    "describe_config_layers",
    "format_config_toml",
    "get_config",
    "get_drive_config",
    "get_index_config",
    "get_sources",
    "is_secret_config_key",
    "load_polylogue_config",
    "redact_config_mapping",
    "redact_secret_value",
]

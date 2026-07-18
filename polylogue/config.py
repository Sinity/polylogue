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
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import tomllib

from .core.errors import PolylogueError
from .core.loopback import bind_hosts_overlap, is_loopback_host
from .paths import GEMINI_DRIVE_FOLDER


class ConfigError(PolylogueError):
    """Configuration error."""


@dataclass(frozen=True, slots=True)
class Source:
    """A session source (local path, Drive folder, or both)."""

    name: str
    path: Path | None = None
    folder: str | None = None

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        if not normalized_name:
            raise ValueError("Source name cannot be empty")
        normalized_folder = self.folder.strip() if self.folder is not None else None
        if self.path is None and not normalized_folder:
            raise ValueError(f"Source '{normalized_name}' must have either 'path' or 'folder'")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "folder", normalized_folder)

    @property
    def is_drive(self) -> bool:
        return self.folder is not None


@dataclass(frozen=True, slots=True)
class DriveConfig:
    """Google Drive OAuth configuration projected from resolved authority."""

    credentials_path: Path
    token_path: Path
    retry_count: int = 3
    timeout: int = 30


@dataclass(frozen=True, slots=True)
class IndexConfig:
    """Search indexing configuration projected from resolved authority."""

    voyage_api_key: str | None = None


class Config:
    """Compatibility projection of an already-resolved runtime snapshot.

    ``Config`` never reads the environment, current directory, home directory,
    or :mod:`polylogue.paths`.  When ``db_path`` is omitted it is derived only
    from the explicit ``archive_root`` supplied by the caller.

    Handwritten (not ``@dataclass``) so that the constructor can accept an
    optional ``db_path`` while the resolved attribute is always a concrete
    ``Path`` -- a dataclass field cannot carry two different static types for
    "what the constructor accepts" versus "what got stored" under one name.
    """

    archive_root: Path
    render_root: Path
    sources: list[Source]
    db_path: Path
    drive_config: DriveConfig | None
    index_config: IndexConfig | None

    def __init__(
        self,
        archive_root: Path,
        render_root: Path,
        sources: list[Source],
        db_path: Path | None = None,
        drive_config: DriveConfig | None = None,
        index_config: IndexConfig | None = None,
    ) -> None:
        self.archive_root = archive_root
        self.render_root = render_root
        self.sources = sources
        self.db_path = db_path if db_path is not None else archive_root / "index.db"
        self.drive_config = drive_config
        self.index_config = index_config
        for attr in ("archive_root", "render_root", "db_path"):
            value = getattr(self, attr)
            if not isinstance(value, Path):
                raise ConfigError(f"Config.{attr} must be a Path, got {type(value).__name__}")
            if not value.is_absolute():
                raise ConfigError(f"Config.{attr} must be an absolute path, got {value!r}")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Config):
            return NotImplemented
        return (
            self.archive_root == other.archive_root
            and self.render_root == other.render_root
            and self.sources == other.sources
            and self.db_path == other.db_path
            and self.drive_config == other.drive_config
            and self.index_config == other.index_config
        )

    def __repr__(self) -> str:
        return (
            f"Config(archive_root={self.archive_root!r}, render_root={self.render_root!r}, "
            f"sources={self.sources!r}, db_path={self.db_path!r}, "
            f"drive_config={self.drive_config!r}, index_config={self.index_config!r})"
        )

    def with_sources(self, sources: list[Source]) -> Config:
        return Config(
            archive_root=self.archive_root,
            render_root=self.render_root,
            sources=sources,
            db_path=self.db_path,
            drive_config=self.drive_config,
            index_config=self.index_config,
        )


def get_sources(runtime: ResolvedRuntimeConfig) -> list[Source]:
    """Return a defensive source list from an already-resolved runtime."""
    return list(runtime.sources)


def get_drive_config(runtime: ResolvedRuntimeConfig) -> DriveConfig:
    """Return the resolved Drive projection."""
    return runtime.drive_config


def get_index_config(runtime: ResolvedRuntimeConfig) -> IndexConfig:
    """Return the resolved index projection."""
    return runtime.index_config


def get_config() -> Config:
    """Return the effective runtime configuration. Compat wrapper, see :func:`get_sources`."""
    return resolve_runtime_config().as_config()


# ---------------------------------------------------------------------------
# TOML config file support (#829)
# ---------------------------------------------------------------------------


def _freeze_config_value(value: object) -> object:
    """Recursively freeze loader output so one snapshot cannot drift in-place."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_config_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config_value(item) for item in value)
    return value


def _thaw_config_value(value: object) -> object:
    """Return a defensive mutable copy of a frozen configuration value."""
    if isinstance(value, Mapping):
        return {str(key): _thaw_config_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw_config_value(item) for item in value)
    return deepcopy(value)


@dataclass(frozen=True, slots=True)
class PolylogueConfig:
    """Typed configuration loaded from TOML with env/CLI overrides.

    Wraps a resolved config dict with attribute access for known keys.
    The underlying dict is available via ``raw`` for unknown keys.
    """

    _data: Mapping[str, object] = field(default_factory=dict)
    _layers: Mapping[str, str] = field(default_factory=dict)
    _layer_paths: Mapping[str, Path | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frozen_data = {str(key): _freeze_config_value(value) for key, value in self._data.items()}
        object.__setattr__(self, "_data", MappingProxyType(frozen_data))
        object.__setattr__(self, "_layers", MappingProxyType(dict(self._layers)))
        object.__setattr__(self, "_layer_paths", MappingProxyType(dict(self._layer_paths)))

    @property
    def raw(self) -> dict[str, object]:
        return {key: _thaw_config_value(value) for key, value in self._data.items()}

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
    def layer_paths(self) -> dict[str, Path | None]:
        """Physical site/user paths captured by this resolution."""
        return dict(self._layer_paths)

    @property
    def archive_root(self) -> str:
        return str(self._data.get("archive_root", ""))

    @property
    def daemon_url(self) -> str:
        return str(self._data.get("daemon_url", "http://127.0.0.1:8766"))

    @property
    def daemon_client_mode(self) -> str:
        return str(self._data.get("daemon_client_mode", "auto")).strip().lower() or "auto"

    @property
    def no_daemon(self) -> bool:
        return bool(self._data.get("no_daemon"))

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
        via TOML ``[observability] enabled = true`` or the env var
        ``POLYLOGUE_OBSERVABILITY_ENABLED=1``.
        """
        return bool(self._data.get("observability_enabled"))

    @property
    def otlp_max_body_bytes(self) -> int:
        """Maximum accepted Content-Length for OTLP POST bodies.

        Default 8 MiB matches typical OTLP exporter batch sizes; clients
        sending more receive 413. Configurable via TOML
        ``[observability] otlp_max_body_bytes = ...`` or env ``POLYLOGUE_OTLP_MAX_BODY_BYTES``.
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
    def sinex_mode(self) -> str:
        """Sinex-backed evidence-mode authority profile: ``off``/``mirror``/``primary``.

        See ``polylogue.sinex.models.PublicationMode`` and
        ``docs/sinex-interop.md``. Default ``off``: standalone SQLite is
        canonical and permanent, per operator directive.
        """
        return str(self._data.get("sinex_mode", "off")).strip().lower() or "off"

    @property
    def log_level(self) -> str:
        return str(self._data.get("log_level", "INFO"))

    @property
    def force_plain(self) -> bool:
        return bool(self._data.get("force_plain"))

    @property
    def no_color(self) -> bool:
        return bool(self._data.get("no_color"))

    @property
    def theme(self) -> str:
        """Resolved CLI theme: ``"dark"``, ``"light"``, or ``"auto"`` (#1274).

        Empty string means "not configured" and lets
        :func:`polylogue.ui.theme.resolve_theme_mode` fall through to
        environment/auto-detection.
        """
        return str(self._data.get("theme", "")).strip().lower()

    @property
    def debug_timing(self) -> bool:
        return bool(self._data.get("debug_timing"))

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
        return str(self._data.get("browser_capture_allowed_origins", "chrome-extension://*"))

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
    def notification_email_username(self) -> str | None:
        value = self._data.get("notification_email_username")
        return value if isinstance(value, str) and value else None

    @property
    def notification_email_password(self) -> str | None:
        value = self._data.get("notification_email_password")
        return value if isinstance(value, str) and value else None

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
    def notification_email_subject_prefix(self) -> str:
        return str(self._data.get("notification_email_subject_prefix", "[polylogue]"))

    @property
    def notification_email_use_tls(self) -> bool:
        return bool(self._data.get("notification_email_use_tls", True))

    @property
    def notification_email_use_starttls(self) -> bool:
        return bool(self._data.get("notification_email_use_starttls", True))

    @property
    def notification_email_max_per_hour(self) -> int:
        return int(str(self._data.get("notification_email_max_per_hour", 12)))

    @property
    def health_check_interval_s(self) -> int:
        return int(str(self._data.get("health_check_interval_s", 300)))

    @property
    def health_check_tiers(self) -> str:
        return str(self._data.get("health_check_tiers", "fast"))

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
        if isinstance(raw, Mapping):
            thawed = _thaw_config_value(raw)
            return thawed if isinstance(thawed, dict) else {}
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
        if isinstance(raw, Mapping):
            thawed = _thaw_config_value(raw)
            return thawed if isinstance(thawed, dict) else {}
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
    def browser_capture_allow_no_auth(self) -> bool:
        return bool(self._data.get("browser_capture_allow_no_auth"))

    @property
    def source_roots(self) -> tuple[str, ...]:
        v = self._data.get("source_roots")
        if isinstance(v, (list, tuple)):
            return tuple(str(item) for item in v)
        if isinstance(v, str) and v.strip():
            return tuple(s.strip() for s in v.split(",") if s.strip())
        return ()

    @property
    def hermes_root(self) -> str:
        """Optional layered override for the Hermes runtime root."""
        return str(self._data.get("hermes_root", ""))

    @property
    def drive_credentials_path(self) -> str:
        return str(self._data.get("drive_credentials_path", ""))

    @property
    def drive_token_path(self) -> str:
        return str(self._data.get("drive_token_path", ""))

    @property
    def hook_sidecar_dir(self) -> str:
        return str(self._data.get("hook_sidecar_dir", ""))

    @property
    def backup_verify_tmpdir(self) -> str | None:
        value = self._data.get("backup_verify_tmpdir")
        return value if isinstance(value, str) and value else None

    @property
    def antigravity_language_server(self) -> str | None:
        value = self._data.get("antigravity_language_server")
        return value if isinstance(value, str) and value else None

    @property
    def ingest_commit_batch_messages(self) -> int:
        return int(str(self._data.get("ingest_commit_batch_messages", 8000)))

    @property
    def ingest_parse_workers(self) -> int:
        return max(1, int(str(self._data.get("ingest_parse_workers", 1))))

    @property
    def live_full_ingest_workers(self) -> int:
        return max(1, int(str(self._data.get("live_full_ingest_workers", 1))))

    def get(self, key: str, default: object = None) -> object:
        value = self._data.get(key, default)
        return _thaw_config_value(value)

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
            if isinstance(entry, Mapping):
                thawed = _thaw_config_value(entry)
                if isinstance(thawed, dict):
                    rows.append(thawed)
        return tuple(rows)


#: Default site-wide configuration path. Overridable through the
#: ``POLYLOGUE_SITE_CONFIG`` env var (primarily for tests and packaging).
DEFAULT_SITE_CONFIG_PATH = Path("/etc/polylogue/polylogue.toml")


_MISSING = object()


@dataclass(frozen=True, slots=True)
class ConfigInventoryEntry:
    """Single public configuration knob and its inspection metadata.

    The inventory owns TOML/env/CLI surface metadata. Loader/rendering code
    below consumes the same entries, so the list is not a decorative ledger
    that can drift away from implementation.
    """

    key: str
    owner_class: str
    reload_behavior: str
    toml_path: str | None = None
    env_var: str | None = None
    cli_override: str | None = None
    description: str = ""
    toml_kind: str = "scalar"

    @property
    def effective_path(self) -> str:
        return f"polylogue config --format json values.{self.key}"


_CONFIG_INVENTORY: tuple[ConfigInventoryEntry, ...] = (
    ConfigInventoryEntry(
        "archive_root",
        toml_path="archive.root",
        env_var="POLYLOGUE_ARCHIVE_ROOT",
        owner_class="path-layout",
        reload_behavior="startup-bound",
        description="Archive root containing source/index/embeddings/user/ops stores.",
    ),
    ConfigInventoryEntry(
        "daemon_url",
        toml_path="daemon.url",
        env_var="POLYLOGUE_DAEMON_URL",
        cli_override="polylogue status --daemon-url",
        owner_class="network-security",
        reload_behavior="per-invocation-client",
        description="Client-side base URL used by CLI surfaces that call the daemon API.",
    ),
    ConfigInventoryEntry(
        "daemon_client_mode",
        toml_path="client.daemon",
        env_var="POLYLOGUE_DAEMON",
        owner_class="deployment-policy",
        reload_behavior="per-invocation-client",
        description="Daemon client routing mode; 'off' forces direct archive access.",
    ),
    ConfigInventoryEntry(
        "no_daemon",
        toml_path="client.no_daemon",
        env_var="POLYLOGUE_NO_DAEMON",
        cli_override="polylogue --no-daemon",
        owner_class="deployment-policy",
        reload_behavior="per-invocation-client",
        description="Disable daemon client routing for one resolved invocation.",
    ),
    ConfigInventoryEntry(
        "daemon_host",
        toml_path="daemon.host",
        env_var="POLYLOGUE_DAEMON_HOST",
        owner_class="network-security",
        reload_behavior="startup-bound",
        description="Legacy daemon listen host alias kept for TOML/Nix compatibility.",
    ),
    ConfigInventoryEntry(
        "daemon_port",
        toml_path="daemon.port",
        env_var="POLYLOGUE_DAEMON_PORT",
        owner_class="network-security",
        reload_behavior="startup-bound",
        description="Legacy daemon listen port alias kept for TOML/Nix compatibility.",
    ),
    ConfigInventoryEntry(
        "api_host",
        toml_path="daemon.api.host",
        env_var="POLYLOGUE_API_HOST",
        cli_override="polylogued run --api-host",
        owner_class="network-security",
        reload_behavior="startup-bound",
        description="Daemon HTTP API listen host.",
    ),
    ConfigInventoryEntry(
        "api_port",
        toml_path="daemon.api.port",
        env_var="POLYLOGUE_API_PORT",
        cli_override="polylogued run --api-port",
        owner_class="network-security",
        reload_behavior="startup-bound",
        description="Daemon HTTP API listen port.",
    ),
    ConfigInventoryEntry(
        "api_auth_token",
        toml_path="daemon.api.auth_token",
        env_var="POLYLOGUE_API_AUTH_TOKEN",
        cli_override="polylogued run --api-auth-token",
        owner_class="network-security",
        reload_behavior="startup-bound",
        description="Bearer token required when the daemon API is exposed beyond loopback.",
    ),
    ConfigInventoryEntry(
        "browser_capture_host",
        toml_path="daemon.browser_capture.host",
        env_var="POLYLOGUE_BROWSER_CAPTURE_HOST",
        cli_override="polylogued run --host",
        owner_class="network-security",
        reload_behavior="startup-bound",
        description="Browser-capture receiver listen host.",
    ),
    ConfigInventoryEntry(
        "browser_capture_port",
        toml_path="daemon.browser_capture.port",
        env_var="POLYLOGUE_BROWSER_CAPTURE_PORT",
        cli_override="polylogued run --port",
        owner_class="network-security",
        reload_behavior="startup-bound",
        description="Browser-capture receiver listen port.",
    ),
    ConfigInventoryEntry(
        "browser_capture_allowed_origins",
        toml_path="daemon.browser_capture.allowed_origins",
        env_var="POLYLOGUE_BROWSER_CAPTURE_ALLOWED_ORIGINS",
        cli_override="polylogued run --browser-capture-origin",
        owner_class="network-security",
        reload_behavior="startup-bound",
        description="Comma-separated browser-capture CORS origins.",
    ),
    ConfigInventoryEntry(
        "browser_capture_allow_remote",
        toml_path="daemon.browser_capture.allow_remote",
        env_var="POLYLOGUE_BROWSER_CAPTURE_ALLOW_REMOTE",
        cli_override="polylogued run --insecure-allow-remote",
        owner_class="network-security",
        reload_behavior="startup-bound",
        description="Explicit opt-in for non-loopback browser-capture/API binding.",
    ),
    ConfigInventoryEntry(
        "browser_capture_auth_token",
        toml_path="daemon.browser_capture.auth_token",
        env_var="POLYLOGUE_BROWSER_CAPTURE_AUTH_TOKEN",
        cli_override="polylogued run --browser-capture-auth-token",
        owner_class="network-security",
        reload_behavior="startup-bound",
        description="Bearer token for browser-capture requests; auto-minted/loaded from a 0600 file if unset.",
    ),
    ConfigInventoryEntry(
        "browser_capture_allow_no_auth",
        toml_path="daemon.browser_capture.allow_no_auth",
        env_var="POLYLOGUE_BROWSER_CAPTURE_ALLOW_NO_AUTH",
        cli_override="polylogued run --browser-capture-allow-no-auth",
        owner_class="network-security",
        reload_behavior="startup-bound",
        description="Explicit opt-out of the auto-minted receiver bearer token (receiver serves unauthenticated).",
    ),
    ConfigInventoryEntry(
        "browser_capture_spool_path",
        toml_path="daemon.browser_capture.spool_path",
        env_var="POLYLOGUE_BROWSER_CAPTURE_SPOOL_PATH",
        cli_override="polylogued run --spool",
        owner_class="path-layout",
        reload_behavior="startup-bound",
        description="Spool directory for browser-capture JSONL before archive ingestion.",
    ),
    ConfigInventoryEntry(
        "watch_debounce_s",
        toml_path="daemon.watch.debounce_s",
        env_var="POLYLOGUE_WATCH_DEBOUNCE_S",
        cli_override="polylogued run --debounce-s",
        owner_class="resource-policy",
        reload_behavior="startup-bound",
        description="Quiet period before the live watcher parses a changed file.",
    ),
    ConfigInventoryEntry(
        "source_roots",
        toml_path="sources.roots",
        cli_override="polylogued run --root",
        owner_class="path-layout",
        reload_behavior="startup-bound",
        description="Additional source roots watched by the daemon.",
    ),
    ConfigInventoryEntry(
        "hermes_root",
        toml_path="sources.hermes.root",
        env_var="POLYLOGUE_HERMES_ROOT",
        owner_class="path-layout",
        reload_behavior="startup-bound",
        description="Hermes runtime root containing state, snapshots, and observability artifacts.",
    ),
    ConfigInventoryEntry(
        "embedding_enabled",
        toml_path="embedding.enabled",
        env_var="POLYLOGUE_DAEMON_ENABLE_EMBEDDINGS",
        owner_class="provider-cost-control",
        reload_behavior="daemon-loop",
        description="Explicit opt-in for daemon-driven embedding catch-up and spend.",
    ),
    ConfigInventoryEntry(
        "embedding_model",
        toml_path="embedding.model",
        owner_class="provider-cost-control",
        reload_behavior="daemon-loop",
        description="Embedding provider model name.",
    ),
    ConfigInventoryEntry(
        "embedding_dimension",
        toml_path="embedding.dimension",
        owner_class="provider-cost-control",
        reload_behavior="daemon-loop",
        description="Embedding vector dimension expected in the archive.",
    ),
    ConfigInventoryEntry(
        "embedding_max_cost_usd",
        toml_path="embedding.max_cost_usd",
        owner_class="provider-cost-control",
        reload_behavior="daemon-loop",
        description="Soft spend cap for embedding work.",
    ),
    ConfigInventoryEntry(
        "voyage_api_key",
        toml_path="embedding.voyage_api_key",
        env_var="VOYAGE_API_KEY",
        owner_class="provider-cost-control",
        reload_behavior="daemon-loop",
        description="Voyage provider API key presence; always redacted in inspection output.",
    ),
    ConfigInventoryEntry(
        "sinex_mode",
        toml_path="sinex.mode",
        env_var="POLYLOGUE_SINEX_MODE",
        owner_class="network-security",
        reload_behavior="daemon-startup",
        description=(
            "Sinex-backed evidence-mode authority profile: off (default; SQLite is "
            "canonical, zero Sinex transport work), mirror (durable local commit plus "
            "a best-effort publication obligation), or primary (local projection "
            "advance waits for an allowed durable Sinex receipt: confirmed persistence, "
            "durable debt, or lossless spool acceptance). Mirror/primary are wired "
            "through ingest and daemon convergence, and require deployment composition "
            "to register a concrete Sinex transport; no reference transport is selected "
            "automatically. See docs/sinex-interop.md and polylogue/sinex/__init__.py."
        ),
    ),
    ConfigInventoryEntry(
        "observability_enabled",
        toml_path="observability.enabled",
        env_var="POLYLOGUE_OBSERVABILITY_ENABLED",
        owner_class="network-security",
        reload_behavior="request-time",
        description="Enable OTLP/observability HTTP ingestion routes.",
    ),
    ConfigInventoryEntry(
        "otlp_max_body_bytes",
        toml_path="observability.otlp_max_body_bytes",
        env_var="POLYLOGUE_OTLP_MAX_BODY_BYTES",
        owner_class="resource-policy",
        reload_behavior="request-time",
        description="Maximum accepted OTLP request body size.",
    ),
    ConfigInventoryEntry(
        "log_level",
        toml_path="logging.level",
        owner_class="presentation-preference",
        reload_behavior="startup-bound",
        description="Python logging verbosity.",
    ),
    ConfigInventoryEntry(
        "force_plain",
        toml_path="logging.force_plain",
        env_var="POLYLOGUE_FORCE_PLAIN",
        cli_override="polylogue --plain",
        owner_class="presentation-preference",
        reload_behavior="per-invocation-client",
        description="Force plain output and avoid Rich layout primitives.",
    ),
    ConfigInventoryEntry(
        "no_color",
        env_var="NO_COLOR",
        owner_class="presentation-preference",
        reload_behavior="per-invocation-client",
        description="Standards-based terminal request to suppress ANSI color output.",
    ),
    ConfigInventoryEntry(
        "theme",
        toml_path="ui.theme",
        env_var="POLYLOGUE_THEME",
        owner_class="presentation-preference",
        reload_behavior="per-invocation-client",
        description="Terminal/web semantic theme mode: dark, light, or auto.",
    ),
    ConfigInventoryEntry(
        "debug_timing",
        toml_path="ui.debug_timing",
        env_var="POLYLOGUE_DEBUG_TIMING",
        owner_class="presentation-preference",
        reload_behavior="per-invocation-client",
        description="Emit CLI phase timing diagnostics.",
    ),
    ConfigInventoryEntry(
        "schema_validation",
        toml_path="schema.validation",
        env_var="POLYLOGUE_SCHEMA_VALIDATION",
        owner_class="deployment-policy",
        reload_behavior="per-invocation-client",
        description="Schema validation mode used by CLI/import surfaces.",
    ),
    ConfigInventoryEntry(
        "slow_query_notice_seconds",
        toml_path="ui.slow_query_notice_seconds",
        env_var="POLYLOGUE_SLOW_QUERY_NOTICE_SECONDS",
        owner_class="presentation-preference",
        reload_behavior="per-invocation-client",
        description="Threshold for slow-query user notices.",
    ),
    ConfigInventoryEntry(
        "notification_backend",
        toml_path="notifications.backend",
        env_var="POLYLOGUE_NOTIFICATION_BACKEND",
        owner_class="deployment-policy",
        reload_behavior="daemon-loop",
        description="Health notification backend: log/stdout/webhook/apprise/email.",
    ),
    ConfigInventoryEntry(
        "notification_webhook_url",
        toml_path="notifications.webhook_url",
        env_var="POLYLOGUE_NOTIFICATION_WEBHOOK_URL",
        owner_class="deployment-policy",
        reload_behavior="daemon-loop",
        description="Webhook endpoint for daemon health notifications.",
    ),
    ConfigInventoryEntry(
        "notification_webhook_secret",
        toml_path="notifications.webhook_secret",
        env_var="POLYLOGUE_NOTIFICATION_WEBHOOK_SECRET",
        owner_class="deployment-policy",
        reload_behavior="daemon-loop",
        description="Shared webhook secret; always redacted in inspection output.",
    ),
    ConfigInventoryEntry(
        "notification_apprise_urls",
        toml_path="notifications.apprise_urls",
        env_var="POLYLOGUE_NOTIFICATION_APPRISE_URLS",
        owner_class="deployment-policy",
        reload_behavior="daemon-loop",
        description="Apprise notification URLs.",
    ),
    ConfigInventoryEntry(
        "notification_email_host",
        toml_path="notifications.email.host",
        env_var="POLYLOGUE_NOTIFICATION_EMAIL_HOST",
        owner_class="deployment-policy",
        reload_behavior="daemon-loop",
        description="SMTP host for health notifications.",
    ),
    ConfigInventoryEntry(
        "notification_email_port",
        toml_path="notifications.email.port",
        env_var="POLYLOGUE_NOTIFICATION_EMAIL_PORT",
        owner_class="deployment-policy",
        reload_behavior="daemon-loop",
        description="SMTP port for health notifications.",
    ),
    ConfigInventoryEntry(
        "notification_email_username",
        toml_path="notifications.email.username",
        env_var="POLYLOGUE_NOTIFICATION_EMAIL_USERNAME",
        owner_class="deployment-policy",
        reload_behavior="daemon-loop",
        description="SMTP username.",
    ),
    ConfigInventoryEntry(
        "notification_email_password",
        toml_path="notifications.email.password",
        env_var="POLYLOGUE_NOTIFICATION_EMAIL_PASSWORD",
        owner_class="deployment-policy",
        reload_behavior="daemon-loop",
        description="SMTP password; always redacted in inspection output.",
    ),
    ConfigInventoryEntry(
        "notification_email_from",
        toml_path="notifications.email.from",
        env_var="POLYLOGUE_NOTIFICATION_EMAIL_FROM",
        owner_class="deployment-policy",
        reload_behavior="daemon-loop",
        description="SMTP sender address.",
    ),
    ConfigInventoryEntry(
        "notification_email_to",
        toml_path="notifications.email.to",
        env_var="POLYLOGUE_NOTIFICATION_EMAIL_TO",
        owner_class="deployment-policy",
        reload_behavior="daemon-loop",
        description="SMTP recipient list.",
    ),
    ConfigInventoryEntry(
        "notification_email_subject_prefix",
        toml_path="notifications.email.subject_prefix",
        env_var="POLYLOGUE_NOTIFICATION_EMAIL_SUBJECT_PREFIX",
        owner_class="deployment-policy",
        reload_behavior="daemon-loop",
        description="SMTP subject prefix.",
    ),
    ConfigInventoryEntry(
        "notification_email_use_tls",
        toml_path="notifications.email.use_tls",
        env_var="POLYLOGUE_NOTIFICATION_EMAIL_USE_TLS",
        owner_class="deployment-policy",
        reload_behavior="daemon-loop",
        description="Use implicit TLS for SMTP notifications.",
    ),
    ConfigInventoryEntry(
        "notification_email_use_starttls",
        toml_path="notifications.email.use_starttls",
        env_var="POLYLOGUE_NOTIFICATION_EMAIL_USE_STARTTLS",
        owner_class="deployment-policy",
        reload_behavior="daemon-loop",
        description="Use STARTTLS for SMTP notifications.",
    ),
    ConfigInventoryEntry(
        "notification_email_max_per_hour",
        toml_path="notifications.email.max_per_hour",
        env_var="POLYLOGUE_NOTIFICATION_EMAIL_MAX_PER_HOUR",
        owner_class="deployment-policy",
        reload_behavior="daemon-loop",
        description="Throttle health notification email volume.",
    ),
    ConfigInventoryEntry(
        "health_check_interval_s",
        toml_path="health.check_interval_s",
        env_var="POLYLOGUE_HEALTH_CHECK_INTERVAL_S",
        owner_class="resource-policy",
        reload_behavior="daemon-loop",
        description="Periodic health-check interval.",
    ),
    ConfigInventoryEntry(
        "health_check_tiers",
        toml_path="health.check_tiers",
        env_var="POLYLOGUE_HEALTH_CHECK_TIERS",
        owner_class="resource-policy",
        reload_behavior="daemon-loop",
        description="Comma-separated health-check tiers.",
    ),
    ConfigInventoryEntry(
        "health_blob_integrity_sample_size",
        toml_path="health.blob_integrity_sample_size",
        env_var="POLYLOGUE_HEALTH_BLOB_INTEGRITY_SAMPLE_SIZE",
        owner_class="resource-policy",
        reload_behavior="daemon-loop",
        description="Bounded sample size for blob-integrity health checks.",
    ),
    ConfigInventoryEntry(
        "health_convergence_debt",
        toml_path="health.convergence_debt",
        owner_class="resource-policy",
        reload_behavior="daemon-loop",
        description="Nested convergence-debt SLO thresholds.",
        toml_kind="table",
    ),
    ConfigInventoryEntry(
        "health_cursor_lag",
        toml_path="health.cursor_lag",
        owner_class="resource-policy",
        reload_behavior="daemon-loop",
        description="Nested cursor-lag SLO thresholds.",
        toml_kind="table",
    ),
    ConfigInventoryEntry(
        "drive_credentials_path",
        toml_path="drive.credentials_path",
        env_var="POLYLOGUE_CREDENTIAL_PATH",
        owner_class="path-layout",
        reload_behavior="startup-bound",
        description="Google Drive OAuth client credentials path.",
    ),
    ConfigInventoryEntry(
        "drive_token_path",
        toml_path="drive.token_path",
        env_var="POLYLOGUE_TOKEN_PATH",
        owner_class="path-layout",
        reload_behavior="startup-bound",
        description="Google Drive OAuth token path.",
    ),
    ConfigInventoryEntry(
        "hook_sidecar_dir",
        toml_path="sources.hook_sidecar_dir",
        env_var="POLYLOGUE_HOOK_SIDECAR_DIR",
        owner_class="path-layout",
        reload_behavior="startup-bound",
        description="Durable hook-event sidecar/spool directory.",
    ),
    ConfigInventoryEntry(
        "backup_verify_tmpdir",
        toml_path="maintenance.backup_verify_tmpdir",
        env_var="POLYLOGUE_BACKUP_VERIFY_TMPDIR",
        owner_class="path-layout",
        reload_behavior="startup-bound",
        description="Preferred parent directory for backup verification scratch data.",
    ),
    ConfigInventoryEntry(
        "antigravity_language_server",
        toml_path="sources.antigravity.language_server",
        env_var="POLYLOGUE_ANTIGRAVITY_LANGUAGE_SERVER",
        owner_class="path-layout",
        reload_behavior="startup-bound",
        description="Antigravity language-server executable override.",
    ),
    ConfigInventoryEntry(
        "ingest_commit_batch_messages",
        toml_path="pipeline.ingest.commit_batch_messages",
        env_var="POLYLOGUE_INGEST_COMMIT_BATCH_MESSAGES",
        owner_class="resource-policy",
        reload_behavior="startup-bound",
        description="Message threshold for grouped index commits; <=0 restores per-session commits.",
    ),
    ConfigInventoryEntry(
        "ingest_parse_workers",
        toml_path="pipeline.ingest.parse_workers",
        env_var="POLYLOGUE_INGEST_PARSE_WORKERS",
        owner_class="resource-policy",
        reload_behavior="startup-bound",
        description="Process-worker count for CPU-bound source parsing.",
    ),
    ConfigInventoryEntry(
        "live_full_ingest_workers",
        toml_path="pipeline.live.full_ingest_workers",
        env_var="POLYLOGUE_LIVE_FULL_INGEST_WORKERS",
        owner_class="resource-policy",
        reload_behavior="startup-bound",
        description="Maximum concurrent workers for live full-artifact ingestion.",
    ),
    ConfigInventoryEntry(
        "subscription_plans",
        toml_path="cost.subscription.plans",
        owner_class="provider-cost-control",
        reload_behavior="per-invocation-client",
        description="Subscription plan rows used by cost/outlook reporting.",
        toml_kind="array-table",
    ),
)

_CONFIG_INVENTORY_BY_KEY = {entry.key: entry for entry in _CONFIG_INVENTORY}
_ENV_CONFIG_KEY_MAP = {entry.env_var: entry.key for entry in _CONFIG_INVENTORY if entry.env_var}
_INT_CONFIG_KEYS = frozenset(
    {
        "api_port",
        "daemon_port",
        "browser_capture_port",
        "embedding_dimension",
        "health_check_interval_s",
        "health_blob_integrity_sample_size",
        "notification_email_port",
        "notification_email_max_per_hour",
        "otlp_max_body_bytes",
        "ingest_commit_batch_messages",
        "ingest_parse_workers",
        "live_full_ingest_workers",
    }
)
_FLOAT_CONFIG_KEYS = frozenset({"embedding_max_cost_usd", "slow_query_notice_seconds", "watch_debounce_s"})
_BOOL_CONFIG_KEYS = frozenset(
    {
        "browser_capture_allow_remote",
        "browser_capture_allow_no_auth",
        "embedding_enabled",
        "force_plain",
        "no_color",
        "no_daemon",
        "debug_timing",
        "notification_email_use_tls",
        "notification_email_use_starttls",
        "observability_enabled",
    }
)


def _toml_section_layout() -> list[tuple[str, list[tuple[str, str]]]]:
    sections: dict[str, list[tuple[str, str]]] = {}
    for entry in _CONFIG_INVENTORY:
        if not entry.toml_path or entry.toml_kind != "scalar":
            continue
        section, _, short_key = entry.toml_path.rpartition(".")
        if not section:
            continue
        sections.setdefault(section, []).append((short_key, entry.key))
    return list(sections.items())


@dataclass(frozen=True, slots=True)
class _BootstrapPaths:
    """Process boundary captured exactly once for one resolution."""

    environment: Mapping[str, str]
    cwd: Path
    home: Path
    config_root: Path
    data_root: Path
    cache_root: Path
    state_root: Path
    runtime_root: Path

    @property
    def config_home(self) -> Path:
        return self.config_root / "polylogue"

    @property
    def data_home(self) -> Path:
        return self.data_root / "polylogue"

    @property
    def cache_home(self) -> Path:
        return self.cache_root / "polylogue"

    @property
    def state_home(self) -> Path:
        return self.state_root / "polylogue"


def _expand_bootstrap_path(value: str | Path, *, home: Path, cwd: Path) -> Path:
    raw = str(value)
    if raw == "~":
        path = home
    elif raw.startswith("~/"):
        path = home / raw[2:]
    else:
        path = Path(raw)
    if not path.is_absolute():
        path = cwd / path
    return Path(os.path.abspath(path))


def _snapshot_bootstrap(
    *,
    environment: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    home: Path | None = None,
) -> _BootstrapPaths:
    env = dict(os.environ if environment is None else environment)
    captured_cwd = Path(os.path.abspath(cwd if cwd is not None else Path.cwd()))
    if home is not None:
        captured_home = _expand_bootstrap_path(home, home=home, cwd=captured_cwd)
    elif env.get("HOME"):
        raw_home = Path(env["HOME"])
        captured_home = Path(os.path.abspath(raw_home if raw_home.is_absolute() else captured_cwd / raw_home))
    else:
        captured_home = Path(os.path.abspath(Path.home()))

    def xdg(name: str, fallback: Path) -> Path:
        raw = env.get(name, "").strip()
        return _expand_bootstrap_path(raw, home=captured_home, cwd=captured_cwd) if raw else fallback

    config_root = xdg("XDG_CONFIG_HOME", captured_home / ".config")
    data_root = xdg("XDG_DATA_HOME", captured_home / ".local" / "share")
    cache_root = xdg("XDG_CACHE_HOME", captured_home / ".cache")
    state_root = xdg("XDG_STATE_HOME", captured_home / ".local" / "state")
    runtime_root = xdg("XDG_RUNTIME_DIR", Path(f"/run/user/{os.getuid()}"))
    return _BootstrapPaths(
        environment=MappingProxyType(env),
        cwd=captured_cwd,
        home=captured_home,
        config_root=config_root,
        data_root=data_root,
        cache_root=cache_root,
        state_root=state_root,
        runtime_root=runtime_root,
    )


def _site_config_path(bootstrap: _BootstrapPaths | None = None) -> Path | None:
    """Resolve the site-wide ``polylogue.toml`` (layer 2)."""
    captured = bootstrap or _snapshot_bootstrap()
    override = captured.environment.get("POLYLOGUE_SITE_CONFIG")
    if override is not None:
        if not override:
            return None
        path = _expand_bootstrap_path(override, home=captured.home, cwd=captured.cwd)
        return path if path.is_file() else None
    return DEFAULT_SITE_CONFIG_PATH if DEFAULT_SITE_CONFIG_PATH.is_file() else None


def _user_config_path(bootstrap: _BootstrapPaths | None = None) -> Path | None:
    """Resolve the user-scoped ``polylogue.toml`` (layer 3)."""
    captured = bootstrap or _snapshot_bootstrap()
    override = captured.environment.get("POLYLOGUE_CONFIG")
    if override:
        path = _expand_bootstrap_path(override, home=captured.home, cwd=captured.cwd)
        return path if path.is_file() else None

    xdg_path = captured.config_home / "polylogue.toml"
    if xdg_path.is_file():
        return xdg_path
    project_path = captured.cwd / "polylogue.toml"
    return project_path if project_path.is_file() else None


def _default_config_values(bootstrap: _BootstrapPaths | None = None) -> dict[str, object]:
    """Built-in defaults (layer 1) captured from one bootstrap context."""
    captured = bootstrap or _snapshot_bootstrap()
    default_parse_workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    return {
        "archive_root": str(captured.data_home),
        "daemon_url": "http://127.0.0.1:8766",
        "daemon_client_mode": "auto",
        "no_daemon": False,
        "daemon_host": "127.0.0.1",
        "daemon_port": 8766,
        "api_host": "127.0.0.1",
        "api_port": 8766,
        "api_auth_token": None,
        "browser_capture_port": 8765,
        "browser_capture_allowed_origins": "chrome-extension://*",
        "embedding_enabled": False,
        "observability_enabled": False,
        "otlp_max_body_bytes": 8 * 1024 * 1024,
        "embedding_model": "voyage-4",
        "embedding_dimension": 1024,
        "embedding_max_cost_usd": 5.0,
        "voyage_api_key": None,
        "sinex_mode": "off",
        "log_level": "INFO",
        "force_plain": False,
        "no_color": False,
        "theme": "",
        "debug_timing": False,
        "schema_validation": "advisory",
        "slow_query_notice_seconds": None,
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
        "health_blob_integrity_sample_size": 100,
        "health_convergence_debt": {},
        "health_cursor_lag": {},
        "watch_debounce_s": 2.0,
        "browser_capture_host": "127.0.0.1",
        "browser_capture_spool_path": "",
        "browser_capture_auth_token": None,
        "browser_capture_allow_remote": False,
        "browser_capture_allow_no_auth": False,
        "source_roots": (),
        "hermes_root": "",
        "drive_credentials_path": str(captured.config_home / "polylogue-credentials.json"),
        "drive_token_path": str(captured.state_home / "token.json"),
        "hook_sidecar_dir": str(captured.data_home / "hooks"),
        "backup_verify_tmpdir": None,
        "antigravity_language_server": None,
        "ingest_commit_batch_messages": 8000,
        "ingest_parse_workers": default_parse_workers,
        "live_full_ingest_workers": 1,
        "subscription_plans": (),
    }


def _apply_toml_layer(
    cfg: dict[str, object],
    layers: dict[str, str],
    path: Path,
    layer_name: str,
    *,
    strict: bool,
) -> None:
    """Load one TOML layer, optionally failing for an explicitly selected file."""
    try:
        with open(path, "rb") as fh:
            toml_data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        if strict:
            raise ConfigError(f"cannot load explicitly selected {layer_name} config {path}: {exc}") from exc
        return

    before = deepcopy(cfg)
    _merge_toml(cfg, toml_data)
    for key, value in cfg.items():
        if before.get(key, _MISSING) != value:
            layers[key] = layer_name


def load_polylogue_config(
    *,
    config_path: Path | None = None,
    site_config_path: Path | None = None,
    cli_overrides: dict[str, object] | None = None,
    environment: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    home: Path | None = None,
    _bootstrap: _BootstrapPaths | None = None,
) -> PolylogueConfig:
    """Load resolved Polylogue config with five-layer precedence.

    Precedence (low → high): built-in defaults, site TOML, user TOML,
    environment, then CLI overrides.  The bootstrap boundary is captured once
    so all defaults, discovery paths, and environment values belong to the same
    immutable resolution.
    """
    bootstrap = _bootstrap or _snapshot_bootstrap(environment=environment, cwd=cwd, home=home)
    cfg = _default_config_values(bootstrap)
    layers: dict[str, str] = dict.fromkeys(cfg, "default")

    explicit_site = site_config_path is not None or bool(bootstrap.environment.get("POLYLOGUE_SITE_CONFIG"))
    site_path: Path | None
    if site_config_path is not None:
        site_path = _expand_bootstrap_path(site_config_path, home=bootstrap.home, cwd=bootstrap.cwd)
    else:
        site_path = _site_config_path(bootstrap)
    if site_path is not None and site_path.is_file():
        _apply_toml_layer(cfg, layers, site_path, "site", strict=explicit_site)

    explicit_user = config_path is not None or bool(bootstrap.environment.get("POLYLOGUE_CONFIG"))
    user_path: Path | None
    if config_path is not None:
        user_path = _expand_bootstrap_path(config_path, home=bootstrap.home, cwd=bootstrap.cwd)
    else:
        user_path = _user_config_path(bootstrap)
    if user_path is not None and user_path.is_file():
        _apply_toml_layer(cfg, layers, user_path, "user", strict=explicit_user)

    before_env = deepcopy(cfg)
    _apply_env_overrides(cfg, bootstrap.environment)
    for key, value in cfg.items():
        if before_env.get(key, _MISSING) != value:
            layers[key] = "env"

    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                cfg[key] = value
                layers[key] = "cli"

    return PolylogueConfig(
        _data=cfg,
        _layers=layers,
        _layer_paths={"site": site_path, "user": user_path},
    )


def describe_config_layers(
    *,
    cfg: PolylogueConfig | None = None,
    config_path: Path | None = None,
    site_config_path: Path | None = None,
    environment: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    home: Path | None = None,
) -> dict[str, object]:
    """Return the physical site/user paths associated with one resolution."""
    if cfg is not None:
        captured = cfg.layer_paths
        site = captured.get("site")
        user = captured.get("user")
    else:
        bootstrap = _snapshot_bootstrap(environment=environment, cwd=cwd, home=home)
        site = (
            _expand_bootstrap_path(site_config_path, home=bootstrap.home, cwd=bootstrap.cwd)
            if site_config_path is not None
            else _site_config_path(bootstrap)
        )
        user = (
            _expand_bootstrap_path(config_path, home=bootstrap.home, cwd=bootstrap.cwd)
            if config_path is not None
            else _user_config_path(bootstrap)
        )
    return {
        "site": {"path": str(site) if site is not None else None, "exists": bool(site and site.is_file())},
        "user": {"path": str(user) if user is not None else None, "exists": bool(user and user.is_file())},
    }


def _toml_value_at_path(toml_data: Mapping[str, object], path: str) -> object:
    value: object = toml_data
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _deep_merge_table(existing: Mapping[str, object], incoming: Mapping[str, object]) -> dict[str, object]:
    """Recursively merge TOML tables while replacing scalar/list leaves."""
    merged = {str(key): deepcopy(value) for key, value in existing.items()}
    for key, value in incoming.items():
        current = merged.get(str(key), _MISSING)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[str(key)] = _deep_merge_table(current, value)
        else:
            merged[str(key)] = deepcopy(value)
    return merged


def _merge_toml(cfg: dict[str, object], toml_data: dict[str, object]) -> None:
    """Merge TOML into the flat inventory using kind-aware semantics.

    Scalar leaves replace lower layers. Nested tables deep-merge recursively,
    including their ``families`` sub-tables. Arrays of tables replace the lower
    layer as one TOML value, preserving the subscription-plan contract.
    """
    for entry in _CONFIG_INVENTORY:
        if not entry.toml_path:
            continue
        value = _toml_value_at_path(toml_data, entry.toml_path)
        if value is _MISSING:
            continue
        if entry.toml_kind == "table":
            if isinstance(value, Mapping):
                existing = cfg.get(entry.key)
                base = existing if isinstance(existing, Mapping) else {}
                cfg[entry.key] = _deep_merge_table(base, value)
            continue
        if entry.toml_kind == "array-table":
            if isinstance(value, list):
                cfg[entry.key] = tuple(dict(item) for item in value if isinstance(item, Mapping))
            continue
        cfg[entry.key] = tuple(value) if isinstance(value, list) else value

    # Back-compat for early observability TOML examples that used top-level
    # scalar keys before the inventory made the section explicit.
    for legacy_key in ("observability_enabled", "otlp_max_body_bytes"):
        if legacy_key in toml_data:
            cfg[legacy_key] = toml_data[legacy_key]


def _coerce_env_value(cfg_key: str, value: str) -> object:
    """Coerce environment values according to the config inventory key type."""
    if cfg_key in _BOOL_CONFIG_KEYS:
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
        return value
    if cfg_key in _INT_CONFIG_KEYS:
        try:
            return int(value)
        except ValueError:
            return _MISSING
    if cfg_key in _FLOAT_CONFIG_KEYS:
        try:
            return float(value)
        except ValueError:
            return _MISSING
    return value


def _apply_env_overrides(cfg: dict[str, object], environment: Mapping[str, str]) -> None:
    """Apply public environment overrides from the captured bootstrap."""
    for env_var, cfg_key in _ENV_CONFIG_KEY_MAP.items():
        value = environment.get(env_var)
        if value is None:
            continue
        coerced = _coerce_env_value(cfg_key, value)
        if coerced is _MISSING:
            continue
        cfg[cfg_key] = coerced


@dataclass(frozen=True, slots=True)
class ResolvedArchivePaths:
    """Immutable archive/tier path projection derived from resolved settings."""

    archive_root: Path
    source_db: Path
    index_db: Path
    embeddings_db: Path
    user_db: Path
    ops_db: Path
    render_root: Path
    blob_root: Path
    inbox_root: Path
    browser_capture_spool_root: Path
    browser_capture_receiver_token_path: Path
    hook_sidecar_root: Path
    drive_cache_root: Path

    @property
    def active_index_db(self) -> Path:
        return self.index_db


@dataclass(frozen=True, slots=True)
class ResolvedSourcePaths:
    """Immutable local source discovery roots for one runtime."""

    claude_code: Path
    codex: Path
    gemini_cli: Path
    hermes: Path
    antigravity: Path
    browser_capture: Path
    inbox: Path
    hooks_pending: Path
    explicit: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ResolvedRuntimeConfig:
    """One immutable authority shared by every runtime composition root."""

    settings: PolylogueConfig
    paths: ResolvedArchivePaths
    source_paths: ResolvedSourcePaths
    sources: tuple[Source, ...]
    drive_config: DriveConfig
    index_config: IndexConfig
    cwd: Path
    home: Path
    config_home: Path
    data_home: Path
    cache_home: Path
    state_home: Path
    runtime_root: Path
    backup_verify_tmpdir: Path | None
    antigravity_language_server: Path | None

    def as_config(self) -> Config:
        """Return a defensive legacy projection without ambient re-resolution."""
        return Config(
            archive_root=self.paths.archive_root,
            render_root=self.paths.render_root,
            sources=list(self.sources),
            db_path=self.paths.index_db,
            drive_config=self.drive_config,
            index_config=self.index_config,
        )


def _resolved_runtime_path(value: str | Path | None, *, bootstrap: _BootstrapPaths, fallback: Path) -> Path:
    if value is None or not str(value).strip():
        return fallback
    return _expand_bootstrap_path(value, home=bootstrap.home, cwd=bootstrap.cwd)


def resolve_runtime_config(
    *,
    config_path: Path | None = None,
    site_config_path: Path | None = None,
    cli_overrides: dict[str, object] | None = None,
    environment: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    home: Path | None = None,
) -> ResolvedRuntimeConfig:
    """Resolve all five layers once and project every runtime filesystem value."""
    bootstrap = _snapshot_bootstrap(environment=environment, cwd=cwd, home=home)
    settings = load_polylogue_config(
        config_path=config_path,
        site_config_path=site_config_path,
        cli_overrides=cli_overrides,
        _bootstrap=bootstrap,
    )
    archive = _resolved_runtime_path(settings.archive_root, bootstrap=bootstrap, fallback=bootstrap.data_home)
    render = archive / "render"
    browser_spool = _resolved_runtime_path(
        settings.browser_capture_spool_path,
        bootstrap=bootstrap,
        fallback=bootstrap.data_home / "browser-capture",
    )
    hook_sidecar = _resolved_runtime_path(
        settings.hook_sidecar_dir,
        bootstrap=bootstrap,
        fallback=bootstrap.data_home / "hooks",
    )
    drive_credentials = _resolved_runtime_path(
        settings.drive_credentials_path,
        bootstrap=bootstrap,
        fallback=bootstrap.config_home / "polylogue-credentials.json",
    )
    drive_token = _resolved_runtime_path(
        settings.drive_token_path,
        bootstrap=bootstrap,
        fallback=bootstrap.state_home / "token.json",
    )
    drive_cache = bootstrap.data_home / "drive-cache"
    paths = ResolvedArchivePaths(
        archive_root=archive,
        source_db=archive / "source.db",
        index_db=archive / "index.db",
        embeddings_db=archive / "embeddings.db",
        user_db=archive / "user.db",
        ops_db=archive / "ops.db",
        render_root=render,
        blob_root=archive / "blob",
        inbox_root=archive / "inbox",
        browser_capture_spool_root=browser_spool,
        browser_capture_receiver_token_path=bootstrap.state_home / "browser-capture-receiver-token",
        hook_sidecar_root=hook_sidecar,
        drive_cache_root=drive_cache,
    )
    explicit_roots = tuple(
        _resolved_runtime_path(value, bootstrap=bootstrap, fallback=bootstrap.cwd) for value in settings.source_roots
    )
    source_paths = ResolvedSourcePaths(
        claude_code=bootstrap.home / ".claude" / "projects",
        codex=bootstrap.home / ".codex" / "sessions",
        gemini_cli=bootstrap.home / ".gemini" / "tmp",
        hermes=_resolved_runtime_path(
            settings.hermes_root,
            bootstrap=bootstrap,
            fallback=bootstrap.home / ".hermes",
        ),
        antigravity=bootstrap.home / ".gemini" / "antigravity",
        browser_capture=browser_spool,
        inbox=paths.inbox_root,
        hooks_pending=hook_sidecar / "pending",
        explicit=explicit_roots,
    )
    local_candidates = (
        ("claude-code", source_paths.claude_code),
        ("codex", source_paths.codex),
        ("gemini-cli", source_paths.gemini_cli),
        ("hermes", source_paths.hermes),
        ("antigravity", source_paths.antigravity),
        ("browser-capture", source_paths.browser_capture),
        ("inbox", source_paths.inbox),
        ("hooks", source_paths.hooks_pending),
    )
    sources = [Source(name=name, path=path) for name, path in local_candidates if path.exists()]
    gemini_cache = drive_cache / "gemini"
    if gemini_cache.exists() or drive_credentials.exists() or drive_token.exists():
        sources.append(Source(name="aistudio", folder=GEMINI_DRIVE_FOLDER, path=gemini_cache))

    backup_tmp = (
        _resolved_runtime_path(settings.backup_verify_tmpdir, bootstrap=bootstrap, fallback=archive)
        if settings.backup_verify_tmpdir
        else None
    )
    antigravity_server = (
        _resolved_runtime_path(settings.antigravity_language_server, bootstrap=bootstrap, fallback=bootstrap.cwd)
        if settings.antigravity_language_server
        else None
    )
    return ResolvedRuntimeConfig(
        settings=settings,
        paths=paths,
        source_paths=source_paths,
        sources=tuple(sources),
        drive_config=DriveConfig(credentials_path=drive_credentials, token_path=drive_token),
        index_config=IndexConfig(voyage_api_key=settings.voyage_api_key),
        cwd=bootstrap.cwd,
        home=bootstrap.home,
        config_home=bootstrap.config_home,
        data_home=bootstrap.data_home,
        cache_home=bootstrap.cache_home,
        state_home=bootstrap.state_home,
        runtime_root=bootstrap.runtime_root,
        backup_verify_tmpdir=backup_tmp,
        antigravity_language_server=antigravity_server,
    )


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


def _json_safe_config_value(value: object) -> object:
    """Return a JSON/TOML-display-safe value without losing scalar types."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe_config_value(item) for item in value]
    if isinstance(value, list):
        return [_json_safe_config_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_config_value(item) for key, item in value.items()}
    return value


def config_display_value(key: str, value: object) -> object:
    """Return a public inspection value, redacting secret-bearing keys."""
    if is_secret_config_key(key):
        return redact_secret_value(value)
    return _json_safe_config_value(value)


def config_inventory() -> tuple[ConfigInventoryEntry, ...]:
    """Return the maintained public configuration inventory."""
    return _CONFIG_INVENTORY


def config_inventory_by_key() -> dict[str, ConfigInventoryEntry]:
    """Return configuration inventory entries keyed by flat config key."""
    return dict(_CONFIG_INVENTORY_BY_KEY)


def _inventory_entry_payload(entry: ConfigInventoryEntry, *, default: object = _MISSING) -> dict[str, object]:
    secret = is_secret_config_key(entry.key)
    payload: dict[str, object] = {
        "key": entry.key,
        "toml_path": entry.toml_path,
        "toml_kind": entry.toml_kind,
        "env_var": entry.env_var,
        "cli_override": entry.cli_override,
        "owner_class": entry.owner_class,
        "secret": secret,
        "redaction": "presence" if secret else "none",
        "reload_behavior": entry.reload_behavior,
        "effective_path": entry.effective_path,
        "description": entry.description,
    }
    payload["default"] = None if default is _MISSING else config_display_value(entry.key, default)
    return payload


def config_inventory_payload() -> list[dict[str, object]]:
    """Return inventory rows ready for JSON/docs consumption."""
    defaults = _default_config_values()
    return [_inventory_entry_payload(entry, default=defaults.get(entry.key, _MISSING)) for entry in _CONFIG_INVENTORY]


def _config_diagnostic(
    *,
    code: str,
    severity: str,
    key: str,
    message: str,
    next_action: str,
    cfg: PolylogueConfig | None = None,
    value_key: str | None = None,
    related_keys: tuple[str, ...] = (),
) -> dict[str, object]:
    entry = _CONFIG_INVENTORY_BY_KEY.get(key)
    payload: dict[str, object] = {
        "code": code,
        "severity": severity,
        "key": key,
        "toml_path": entry.toml_path if entry is not None else None,
        "env_var": entry.env_var if entry is not None else None,
        "message": message,
        "next_action": next_action,
    }
    if cfg is not None:
        display_key = value_key or key
        value = cfg.raw.get(display_key)
        secret = is_secret_config_key(display_key)
        payload.update(
            {
                "source_layer": cfg.layer_of(display_key),
                "value": config_display_value(display_key, value),
                "secret": secret,
                "secret_present": bool(secret and redact_secret_value(value) == SECRET_SET_PLACEHOLDER),
            }
        )
    if related_keys:
        payload["related_keys"] = list(related_keys)
    return payload


def _iter_path_config_values(key: str, value: object) -> list[str]:
    if value in (None, "", ()):
        return []
    if key == "source_roots":
        if isinstance(value, str):
            return [value]
        if isinstance(value, (tuple, list)):
            return [str(item) for item in value if str(item)]
        return [str(value)]
    return [str(value)]


def _expand_config_path(raw_path: str) -> Path | None:
    try:
        return Path(raw_path).expanduser()
    except RuntimeError:
        return None


def _config_path_diagnostics(resolved: PolylogueConfig) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    for entry in _CONFIG_INVENTORY:
        if entry.owner_class != "path-layout":
            continue
        value = resolved.raw.get(entry.key)
        # Defaults may legitimately point at not-yet-created first-run paths.
        # Operator-provided paths must be explicit enough to audit.
        if resolved.layer_of(entry.key) == "default" and not (entry.env_var and entry.env_var in os.environ):
            continue
        for raw_path in _iter_path_config_values(entry.key, value):
            path = _expand_config_path(raw_path)
            if path is None:
                diagnostics.append(
                    _config_diagnostic(
                        code="config_path_invalid",
                        severity="error",
                        key=entry.key,
                        message=f"{entry.key} resolves to an invalid path: {raw_path!r}.",
                        next_action=(
                            f"Set {entry.toml_path or entry.key} to an absolute path with a valid home expansion"
                            + (f" or override {entry.env_var}" if entry.env_var else "")
                            + "."
                        ),
                        cfg=resolved,
                    )
                )
                continue
            if not path.is_absolute():
                diagnostics.append(
                    _config_diagnostic(
                        code="config_path_not_absolute",
                        severity="error",
                        key=entry.key,
                        message=f"{entry.key} resolves to a relative path: {raw_path!r}.",
                        next_action=(
                            f"Set {entry.toml_path or entry.key} to an absolute path"
                            + (f" or override {entry.env_var}" if entry.env_var else "")
                            + "."
                        ),
                        cfg=resolved,
                    )
                )
                continue
            if entry.key == "source_roots" and not path.exists():
                diagnostics.append(
                    _config_diagnostic(
                        code="configured_source_root_missing",
                        severity="warning",
                        key=entry.key,
                        message=f"Configured source root does not exist: {raw_path}.",
                        next_action="Remove the stale source root or create/mount it before running the daemon.",
                        cfg=resolved,
                    )
                )
    return diagnostics


def _configured_browser_capture_origins(resolved: PolylogueConfig) -> tuple[str, ...]:
    return tuple(origin.strip() for origin in resolved.browser_capture_allowed_origins.split(",") if origin.strip())


def _is_browser_extension_origin_pattern(origin: str) -> bool:
    return origin == "chrome-extension://*" or origin.startswith("chrome-extension://")


def _config_network_diagnostics(resolved: PolylogueConfig) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    api_is_remote = not is_loopback_host(resolved.api_host)
    browser_capture_is_remote = not is_loopback_host(resolved.browser_capture_host)

    if api_is_remote and not resolved.browser_capture_allow_remote:
        diagnostics.append(
            _config_diagnostic(
                code="api_remote_bind_requires_allow_remote",
                severity="error",
                key="api_host",
                message=f"Daemon API host {resolved.api_host!r} is not loopback but remote binding is not enabled.",
                next_action=(
                    "Set daemon.browser_capture.allow_remote = true / POLYLOGUE_BROWSER_CAPTURE_ALLOW_REMOTE=true "
                    "only with an API auth token, or bind daemon.api.host to 127.0.0.1."
                ),
                cfg=resolved,
                related_keys=("browser_capture_allow_remote", "api_auth_token"),
            )
        )
    if api_is_remote and resolved.browser_capture_allow_remote and not resolved.api_auth_token:
        diagnostics.append(
            _config_diagnostic(
                code="api_remote_bind_requires_auth_token",
                severity="error",
                key="api_auth_token",
                message=f"Daemon API host {resolved.api_host!r} is remote-enabled without an API bearer token.",
                next_action=(
                    "Set daemon.api.auth_token / POLYLOGUE_API_AUTH_TOKEN or bind daemon.api.host to a loopback address."
                ),
                cfg=resolved,
                related_keys=("api_host", "browser_capture_allow_remote"),
            )
        )

    if browser_capture_is_remote and not resolved.browser_capture_allow_remote:
        diagnostics.append(
            _config_diagnostic(
                code="browser_capture_remote_bind_requires_allow_remote",
                severity="error",
                key="browser_capture_host",
                message=(
                    f"Browser-capture host {resolved.browser_capture_host!r} is not loopback "
                    "but remote binding is not enabled."
                ),
                next_action=(
                    "Set daemon.browser_capture.allow_remote = true / POLYLOGUE_BROWSER_CAPTURE_ALLOW_REMOTE=true "
                    "only with a browser-capture auth token, or bind daemon.browser_capture.host to 127.0.0.1."
                ),
                cfg=resolved,
                related_keys=("browser_capture_allow_remote", "browser_capture_auth_token"),
            )
        )
    if resolved.browser_capture_allow_remote and not resolved.browser_capture_auth_token:
        diagnostics.append(
            _config_diagnostic(
                code="browser_capture_remote_requires_auth_token",
                severity="error",
                key="browser_capture_auth_token",
                message="Remote browser-capture opt-in is set without a browser-capture bearer token.",
                next_action=(
                    "Set daemon.browser_capture.auth_token / POLYLOGUE_BROWSER_CAPTURE_AUTH_TOKEN, "
                    "or disable daemon.browser_capture.allow_remote."
                ),
                cfg=resolved,
                related_keys=("browser_capture_allow_remote", "browser_capture_host"),
            )
        )

    web_origins = tuple(
        origin
        for origin in _configured_browser_capture_origins(resolved)
        if not _is_browser_extension_origin_pattern(origin)
    )
    if web_origins and not resolved.browser_capture_auth_token:
        diagnostics.append(
            _config_diagnostic(
                code="browser_capture_web_origin_requires_auth_token",
                severity="error",
                key="browser_capture_auth_token",
                message=(
                    "Browser-capture web origins require a bearer token; "
                    f"{len(web_origins)} non-extension origin(s) are configured."
                ),
                next_action=(
                    "Set daemon.browser_capture.auth_token / POLYLOGUE_BROWSER_CAPTURE_AUTH_TOKEN, "
                    "or remove non-extension origins from daemon.browser_capture.allowed_origins."
                ),
                cfg=resolved,
                related_keys=("browser_capture_allowed_origins",),
            )
        )

    if resolved.api_port == resolved.browser_capture_port and bind_hosts_overlap(
        resolved.api_host, resolved.browser_capture_host
    ):
        diagnostics.append(
            _config_diagnostic(
                code="daemon_api_browser_capture_port_conflict",
                severity="error",
                key="api_port",
                message=(
                    f"Daemon API {resolved.api_host}:{resolved.api_port} conflicts with browser-capture "
                    f"receiver {resolved.browser_capture_host}:{resolved.browser_capture_port}."
                ),
                next_action=(
                    "Set distinct daemon.api.port and daemon.browser_capture.port values, "
                    "or bind one component to a non-overlapping host."
                ),
                cfg=resolved,
                related_keys=("browser_capture_port", "api_host", "browser_capture_host"),
            )
        )

    return diagnostics


def config_diagnostics(
    cfg: PolylogueConfig | ResolvedRuntimeConfig | None = None,
) -> list[dict[str, object]]:
    """Return typed diagnostics for the resolved effective configuration."""
    resolved = cfg.settings if isinstance(cfg, ResolvedRuntimeConfig) else (cfg or load_polylogue_config())
    diagnostics = _config_path_diagnostics(resolved)
    diagnostics.extend(_config_network_diagnostics(resolved))
    if resolved.embedding_enabled and not resolved.voyage_api_key:
        diagnostics.append(
            _config_diagnostic(
                code="embedding_enabled_without_voyage_key",
                severity="error",
                key="voyage_api_key",
                message="Embedding convergence is enabled but no Voyage API key is configured.",
                next_action=(
                    "Set VOYAGE_API_KEY, run `polylogue ops embed enable --voyage-api-key ...`, "
                    "or disable embedding convergence."
                ),
                cfg=resolved,
            )
        )
    diagnostics.extend(_sinex_mode_diagnostics(resolved))
    return diagnostics


#: Recognized ``[sinex] mode`` / ``POLYLOGUE_SINEX_MODE`` values. Kept as a
#: literal tuple (not an import of ``polylogue.sinex.models.PublicationMode``)
#: so ``config.py`` stays free of internal-package imports -- it is the
#: bootstrap layer every other package imports, not the reverse.
_KNOWN_SINEX_MODES = ("off", "mirror", "primary")


def _sinex_mode_diagnostics(resolved: PolylogueConfig) -> list[dict[str, object]]:
    """Report invalid Sinex authority-mode values before daemon startup."""
    mode = resolved.sinex_mode
    if mode not in _KNOWN_SINEX_MODES:
        return [
            _config_diagnostic(
                code="sinex_mode_unrecognized",
                severity="error",
                key="sinex_mode",
                message=f"sinex_mode={mode!r} is not a recognized value (expected one of {_KNOWN_SINEX_MODES}).",
                next_action="Set [sinex] mode (or POLYLOGUE_SINEX_MODE) to off, mirror, or primary.",
                cfg=resolved,
            )
        ]
    return []


def effective_config_payload(
    cfg: PolylogueConfig | ResolvedRuntimeConfig | None = None,
    *,
    include_inventory: bool = True,
) -> dict[str, object]:
    """Return redacted effective config values with source/layer metadata.

    This is the machine contract behind ``polylogue config --format json``.
    Secret-bearing values expose presence only (``<set>``/``<unset>``) while
    still reporting which layer supplied the value.
    """
    if isinstance(cfg, ResolvedRuntimeConfig):
        runtime: ResolvedRuntimeConfig | None = cfg
        resolved: PolylogueConfig = cfg.settings
    else:
        runtime = None
        resolved = cfg or load_polylogue_config()
    if runtime is None:
        defaults = _default_config_values()
    else:
        bootstrap = _BootstrapPaths(
            environment=MappingProxyType({}),
            cwd=runtime.cwd,
            home=runtime.home,
            config_root=runtime.config_home.parent,
            data_root=runtime.data_home.parent,
            cache_root=runtime.cache_home.parent,
            state_root=runtime.state_home.parent,
            runtime_root=runtime.runtime_root,
        )
        defaults = _default_config_values(bootstrap)
    values: dict[str, object] = {}

    ordered_keys = [entry.key for entry in _CONFIG_INVENTORY]
    ordered_keys.extend(sorted(key for key in resolved.raw if key not in _CONFIG_INVENTORY_BY_KEY))

    for key in ordered_keys:
        entry = _CONFIG_INVENTORY_BY_KEY.get(key)
        value = resolved.raw.get(key, defaults.get(key))
        layer = resolved.layer_of(key)
        secret = is_secret_config_key(key)
        secret_present = bool(secret and redact_secret_value(value) == SECRET_SET_PLACEHOLDER)
        values[key] = {
            "value": config_display_value(key, value),
            "source_layer": layer,
            # Kept for the original --show-layers JSON wording and simple jq paths.
            "layer": layer,
            "secret": secret,
            "secret_present": secret_present,
            "toml_path": entry.toml_path if entry is not None else None,
            "env_var": entry.env_var if entry is not None else None,
            "cli_override": entry.cli_override if entry is not None else None,
            "owner_class": entry.owner_class if entry is not None else "unknown",
            "reload_behavior": entry.reload_behavior if entry is not None else "unknown",
            "effective_path": entry.effective_path
            if entry is not None
            else f"polylogue config --format json values.{key}",
            "default": config_display_value(key, defaults.get(key)) if key in defaults else None,
        }

    layer_paths = describe_config_layers(cfg=resolved)
    payload: dict[str, object] = {
        "layers": {
            "default": "built-in defaults",
            "site": layer_paths["site"],
            "user": layer_paths["user"],
            "env": "POLYLOGUE_*, provider credential, and presentation environment variables",
            "cli": "CLI overrides (per-invocation)",
        },
        "values": values,
        "diagnostics": config_diagnostics(resolved),
    }
    if runtime is not None:
        payload["resolved_paths"] = {
            "archive_root": str(runtime.paths.archive_root),
            "source_db": str(runtime.paths.source_db),
            "index_db": str(runtime.paths.index_db),
            "embeddings_db": str(runtime.paths.embeddings_db),
            "user_db": str(runtime.paths.user_db),
            "ops_db": str(runtime.paths.ops_db),
            "render_root": str(runtime.paths.render_root),
            "blob_root": str(runtime.paths.blob_root),
            "inbox_root": str(runtime.paths.inbox_root),
        }
    if include_inventory:
        payload["inventory"] = config_inventory_payload()
    return payload


def format_config_toml(cfg: dict[str, object]) -> str:
    """Render loaded config as TOML for display.

    Round-trips with :func:`_merge_toml`: scalar keys are rendered from the
    same inventory-driven TOML layout that the loader consumes.

    Secret-bearing keys are redacted to a presence placeholder
    (:data:`SECRET_SET_PLACEHOLDER` / :data:`SECRET_UNSET_PLACEHOLDER`) so the
    rendered TOML can be safely pasted into bug reports, and values are
    serialized with a real TOML emitter (``tomli_w``) rather than raw f-string
    interpolation so quotes/backslashes/newlines cannot corrupt the output.
    """
    import tomli_w

    lines: list[str] = []
    for section, key_pairs in _toml_section_layout():
        section_table: dict[str, object] = {}
        for short_key, flat_key in key_pairs:
            if flat_key not in cfg:
                continue
            value = cfg[flat_key]
            if value is None:
                continue
            section_table[short_key] = config_display_value(flat_key, value)
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
    "ConfigInventoryEntry",
    "DEFAULT_SITE_CONFIG_PATH",
    "DriveConfig",
    "IndexConfig",
    "PolylogueConfig",
    "SECRET_CONFIG_KEYS",
    "SECRET_SET_PLACEHOLDER",
    "SECRET_UNSET_PLACEHOLDER",
    "Source",
    "config_display_value",
    "config_diagnostics",
    "config_inventory",
    "config_inventory_by_key",
    "config_inventory_payload",
    "describe_config_layers",
    "effective_config_payload",
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

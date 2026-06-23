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
        env_var="POLYLOGUE_DAEMON_URL",
        cli_override="polylogue status --daemon-url",
        owner_class="network-security",
        reload_behavior="per-invocation-client",
        description="Client-side base URL used by CLI surfaces that call the daemon API.",
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
        description="Bearer token for browser-capture requests when web origins or remote binds are enabled.",
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
        "health_fts_auto_restore",
        toml_path="health.fts_auto_restore",
        env_var="POLYLOGUE_HEALTH_FTS_AUTO_RESTORE",
        owner_class="resource-policy",
        reload_behavior="daemon-loop",
        description="Allow health checks to repair FTS trigger drift automatically.",
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
    }
)
_FLOAT_CONFIG_KEYS = frozenset({"embedding_max_cost_usd", "slow_query_notice_seconds", "watch_debounce_s"})
_BOOL_CONFIG_KEYS = frozenset(
    {
        "browser_capture_allow_remote",
        "embedding_enabled",
        "force_plain",
        "no_color",
        "health_fts_auto_restore",
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
        "browser_capture_allowed_origins": "chrome-extension://*",
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
        "voyage_api_key": None,
        "log_level": "INFO",
        "force_plain": False,
        "no_color": False,
        "theme": "",
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
        "health_fts_auto_restore": False,
        "health_blob_integrity_sample_size": 100,
        "health_convergence_debt": {},
        "health_cursor_lag": {},
        "watch_debounce_s": 2.0,
        "browser_capture_host": "127.0.0.1",
        "browser_capture_spool_path": "",
        "browser_capture_auth_token": None,
        "browser_capture_allow_remote": False,
        "source_roots": (),
        "subscription_plans": (),
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


def load_polylogue_config(
    *,
    config_path: Path | None = None,
    site_config_path: Path | None = None,
    cli_overrides: dict[str, object] | None = None,
) -> PolylogueConfig:
    """Load resolved Polylogue config with five-layer precedence.

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

    The scalar mapping is generated from :data:`_CONFIG_INVENTORY`, keeping
    loader and inspection metadata on the same rails. Specialized nested
    tables remain explicit because their shape is owned by downstream typed
    decoders rather than by this flat config layer.
    """
    for section, key_pairs in _toml_section_layout():
        # Walk dotted paths for nested TOML sections like [daemon.api]
        section_data: object = toml_data
        for part in section.split("."):
            if isinstance(section_data, dict):
                section_data = section_data.get(part)
            else:
                section_data = None
                break
        if isinstance(section_data, dict):
            for short_key, flat_key in key_pairs:
                if short_key in section_data:
                    value = section_data[short_key]
                    if isinstance(value, list):
                        cfg[flat_key] = tuple(value)
                    else:
                        cfg[flat_key] = value

    # Back-compat for early observability TOML examples that used
    # top-level scalar keys before the inventory made the section explicit.
    for legacy_key in ("observability_enabled", "otlp_max_body_bytes"):
        if legacy_key in toml_data:
            cfg[legacy_key] = toml_data[legacy_key]

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


def _apply_env_overrides(cfg: dict[str, object]) -> None:
    """Apply public environment variable overrides from the config inventory."""
    for env_var, cfg_key in _ENV_CONFIG_KEY_MAP.items():
        value = os.environ.get(env_var)
        if value is None:
            continue
        coerced = _coerce_env_value(cfg_key, value)
        if coerced is _MISSING:
            continue
        cfg[cfg_key] = coerced


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
) -> dict[str, object]:
    entry = _CONFIG_INVENTORY_BY_KEY.get(key)
    return {
        "code": code,
        "severity": severity,
        "key": key,
        "toml_path": entry.toml_path if entry is not None else None,
        "env_var": entry.env_var if entry is not None else None,
        "message": message,
        "next_action": next_action,
    }


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


def _config_path_diagnostics(resolved: PolylogueConfig) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    for entry in _CONFIG_INVENTORY:
        if entry.owner_class != "path-layout":
            continue
        value = resolved.raw.get(entry.key)
        # Defaults may legitimately point at not-yet-created first-run paths.
        # Operator-provided paths must be explicit enough to audit.
        if resolved.layer_of(entry.key) == "default":
            continue
        for raw_path in _iter_path_config_values(entry.key, value):
            path = Path(raw_path).expanduser()
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
                    )
                )
    return diagnostics


def config_diagnostics(cfg: PolylogueConfig | None = None) -> list[dict[str, object]]:
    """Return typed diagnostics for the resolved effective configuration."""
    resolved = cfg if cfg is not None else load_polylogue_config()
    diagnostics = _config_path_diagnostics(resolved)
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
            )
        )
    return diagnostics


def effective_config_payload(
    cfg: PolylogueConfig | None = None,
    *,
    include_inventory: bool = True,
) -> dict[str, object]:
    """Return redacted effective config values with source/layer metadata.

    This is the machine contract behind ``polylogue config --format json``.
    Secret-bearing values expose presence only (``<set>``/``<unset>``) while
    still reporting which layer supplied the value.
    """
    resolved = cfg if cfg is not None else load_polylogue_config()
    defaults = _default_config_values()
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

    layer_paths = describe_config_layers()
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

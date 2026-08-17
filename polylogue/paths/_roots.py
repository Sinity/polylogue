"""Filesystem root and path accessors."""

from __future__ import annotations

import os
from pathlib import Path


def _xdg_path(env_var: str, fallback: Path) -> Path:
    raw = os.environ.get(env_var, "").strip()
    if raw:
        return Path(raw).expanduser()
    return fallback


def config_root() -> Path:
    """XDG_CONFIG_HOME (default: ~/.config)."""
    return _xdg_path("XDG_CONFIG_HOME", Path.home() / ".config")


def data_root() -> Path:
    """XDG_DATA_HOME (default: ~/.local/share)."""
    return _xdg_path("XDG_DATA_HOME", Path.home() / ".local/share")


def cache_root() -> Path:
    """XDG_CACHE_HOME (default: ~/.cache)."""
    return _xdg_path("XDG_CACHE_HOME", Path.home() / ".cache")


def state_root() -> Path:
    """XDG_STATE_HOME (default: ~/.local/state)."""
    return _xdg_path("XDG_STATE_HOME", Path.home() / ".local/state")


def config_home() -> Path:
    """Polylogue config directory."""
    return config_root() / "polylogue"


def data_home() -> Path:
    """Polylogue data directory."""
    return data_root() / "polylogue"


def cache_home() -> Path:
    """Polylogue cache directory."""
    return cache_root() / "polylogue"


def state_home() -> Path:
    """Polylogue state directory."""
    return state_root() / "polylogue"


def db_path() -> Path:
    """Default archive index database path."""
    return index_db_path()


def source_db_path() -> Path:
    """source-log database path."""
    return archive_root() / "source.db"


def index_db_path() -> Path:
    """projection/index database path."""
    return archive_root() / "index.db"


def embeddings_db_path() -> Path:
    """semantic-index database path."""
    return archive_root() / "embeddings.db"


def browser_capture_spool_root() -> Path:
    """Browser-capture source artifact spool, scoped under the archive root.

    Must track ``archive_root()`` (not ``data_home()``) so a
    ``POLYLOGUE_ARCHIVE_ROOT`` override genuinely isolates a scratch/test
    archive's capture spool from the default archive's.
    """
    return archive_root() / "browser-capture"


def browser_capture_receiver_token_path() -> Path:
    """Path for the auto-minted browser-capture receiver bearer token, scoped under the archive root.

    Must track ``archive_root()`` (not ``state_home()``) so two ``polylogued``
    instances with different ``POLYLOGUE_ARCHIVE_ROOT`` values never share one
    token. See :func:`browser_capture_receiver_identity_path` for the
    receiver's stable pairing identity, which is minted and persisted
    independently of this token (polylogue-jlme.5): ordinary token rotation
    must not force every paired browser profile through a full re-pair.
    """
    return archive_root() / "browser-capture-receiver-token"


def browser_capture_receiver_identity_path() -> Path:
    """Path for the auto-minted, non-secret browser-capture receiver identity.

    Scoped under ``archive_root()`` like the bearer token above, so two
    ``polylogued`` instances with different ``POLYLOGUE_ARCHIVE_ROOT`` values
    get distinct identities. Deliberately a *separate* file from the bearer
    token: the identity is the receiver's stable pairing anchor and must
    survive ordinary token rotation (polylogue-jlme.5); hashing the token
    itself (the pre-jlme.5 design) made every rotation an identity change and
    forced every paired extension profile through an unnecessary re-pair.
    """
    return archive_root() / "browser-capture-receiver-id"


def api_auth_token_path() -> Path:
    """Path for the auto-minted daemon HTTP API bearer token, scoped under the archive root.

    Mirrors :func:`browser_capture_receiver_token_path` (polylogue-rzve): must
    track ``archive_root()`` (not ``state_home()``) so two ``polylogued``
    instances with different ``POLYLOGUE_ARCHIVE_ROOT`` values never share one
    API token, and so an explicit ``--api-auth-token``/config value always
    takes precedence over whatever is minted here.
    """
    return archive_root() / "api-auth-token"


def browser_capture_pairing_state_path() -> Path:
    """Path for the receiver's ephemeral one-time pairing-code state.

    Scoped under ``archive_root()`` like the bearer token and identity above,
    so two ``polylogued`` instances with different ``POLYLOGUE_ARCHIVE_ROOT``
    values never share a pairing window. Holds at most one pending code's
    hash + expiry + attempt count at a time (polylogue-gnie): a fresh
    ``pairing start`` overwrites any prior unredeemed code. The file never
    stores the plaintext code or the bearer token itself.
    """
    return archive_root() / "browser-capture-pairing"


CLOUD_SANDBOX_ARCHIVE_ROOT = "/tmp/polylogue-archive"
"""Archive root `.claude/settings.json` sets for cloud sandboxes."""


def archive_root() -> Path:
    """Archive root.

    Precedence (highest first): the ``POLYLOGUE_ARCHIVE_ROOT`` environment
    variable, then ``[archive] root`` in ``polylogue.toml`` (site then user
    config layer, same as :func:`polylogue.config.load_polylogue_config`),
    then the XDG data-home default. Two archive roots must never exist
    simultaneously and silently -- a process without the env var used to
    fall straight through to the XDG default even when a config file named a
    different root, splitting writes (hooks, browser-capture spool, inbox)
    across two disjoint directories that nothing reconciled (polylogue-4ma3).

    This module is intentionally stdlib-only, and :mod:`polylogue.config`
    itself imports from here (``GEMINI_DRIVE_FOLDER``), so a top-level
    import of ``polylogue.config`` would create an import cycle. The env-var
    fast path below never touches ``polylogue.config`` at all; the config
    lookup is a *lazy*, function-local import that only runs once the env
    var is absent, by which point both modules are already fully loaded.
    Nothing here is cached, so a test that monkeypatches
    ``POLYLOGUE_ARCHIVE_ROOT`` (or the config-selecting env vars) between
    calls sees the change immediately -- this must never regress, since many
    tests rely on a per-test ``POLYLOGUE_ARCHIVE_ROOT`` to isolate scratch
    archives from each other and from the real one.
    """
    raw = os.environ.get("POLYLOGUE_ARCHIVE_ROOT", "").strip()
    if raw and raw != CLOUD_SANDBOX_ARCHIVE_ROOT:
        return Path(raw).expanduser()

    from ..config import resolve_archive_root  # lazy: avoid paths<->config import cycle

    if not raw:
        return resolve_archive_root()

    # The cloud sentinel specifically. `.claude/settings.json` sets it so a
    # cloud sandbox has a writable archive, and agent subprocesses inherit it
    # on the workstation, where it silently retargets every read at an empty
    # stub instead of the real archive named by polylogue.toml. Same leak class
    # as POLYLOGUE_PYTEST_BASETEMP_ROOT, and the same treatment: honour it only
    # where nothing else is configured. Resolving with the variable scrubbed
    # tells us which world we are in -- a config layer naming a root means the
    # sentinel leaked, while falling through to the XDG default means this
    # really is a sandbox with no config.
    scrubbed = {key: value for key, value in os.environ.items() if key != "POLYLOGUE_ARCHIVE_ROOT"}
    configured = resolve_archive_root(environment=scrubbed)
    if configured != data_home():
        return configured
    return Path(raw).expanduser()


def render_root() -> Path:
    """Render output root under the archive root."""
    return archive_root() / "render"


def drive_credentials_path() -> Path:
    """Drive OAuth credentials path."""
    return config_home() / "polylogue-credentials.json"


def drive_token_path() -> Path:
    """Drive OAuth token path."""
    return state_home() / "token.json"


def claude_code_path() -> Path:
    """Claude Code sessions directory."""
    return Path.home() / ".claude" / "projects"


def claude_code_todos_path() -> Path:
    """Claude Code live plan-snapshot directory (polylogue-t0p).

    Sibling of ``claude_code_path()``, not nested under it: Claude Code
    writes one ``<session-id>[-agent-<agent-id>].json`` file per session
    (rewritten wholesale on every ``TodoWrite`` call, never appended), and
    prunes old entries on its own schedule -- unread state here is
    eventually lost, not merely delayed.
    """
    return Path.home() / ".claude" / "todos"


def codex_path() -> Path:
    """Codex sessions directory."""
    return Path.home() / ".codex" / "sessions"


def gemini_cli_path() -> Path:
    """Gemini CLI local session workspace directory."""
    return Path.home() / ".gemini" / "tmp"


def hermes_sessions_path() -> Path:
    """Hermes agent state directory."""
    return Path.home() / ".hermes"


def antigravity_path() -> Path:
    """Antigravity local state directory."""
    return Path.home() / ".gemini" / "antigravity"


GEMINI_DRIVE_FOLDER = "Google AI Studio"


def blob_store_root() -> Path:
    """Content-addressed blob store root directory."""
    return archive_root() / "blob"


def hooks_sidecar_dir() -> Path:
    """Directory where hook scripts drop session event data, scoped under the archive root.

    AI coding agent hooks (Claude Code, Codex) write structured JSONL
    event records here. The daemon watcher picks them up for ingestion.

    Must track ``archive_root()`` (not ``data_home()``), mirroring
    ``browser_capture_spool_root()``: a daemon started against a scratch or
    test ``POLYLOGUE_ARCHIVE_ROOT`` must resolve to that scratch archive's own
    hook spool, never the real global one (polylogue-o7hx -- a daemon pointed
    at a scratch archive drained the real spool four separate times before
    this was fixed). For the default production configuration the archive
    root IS ``data_home()``, so this resolves to the exact same path as
    before -- zero migration for real deployments.
    """
    return archive_root() / "hooks"


def drive_cache_path() -> Path:
    """Local cache directory for Drive-sourced files."""
    return data_home() / "drive-cache"

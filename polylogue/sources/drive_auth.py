from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from polylogue.logging import get_logger

from ..paths import drive_credentials_path, drive_token_path
from .drive_auth_support import (
    DriveAuthPrompter,
    UIAuthPrompter,
    _PromptBridge,
    import_auth_module,
    load_cached_credentials,
    persist_token,
    refresh_credentials_if_needed,
    run_manual_auth_flow,
)
from .drive_types import SCOPES, DriveAuthError
from .token_store import TokenStore, create_token_store

logger = get_logger(__name__)
_import_auth_module = import_auth_module


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------


def default_credentials_path(config: object | None = None) -> Path:
    """Get default credentials path, optionally from DriveConfig."""
    if config is not None and hasattr(config, "credentials_path"):
        cred_path = getattr(config, "credentials_path", None)
        if cred_path:
            return Path(cred_path)
    return drive_credentials_path()


def default_token_path(config: object | None = None) -> Path:
    """Get default token path, optionally from DriveConfig."""
    if config is not None and hasattr(config, "token_path"):
        token_path = getattr(config, "token_path", None)
        if token_path:
            return Path(token_path)
    return drive_token_path()


def _resolve_credentials_path(ui: object | None, config: object | None = None) -> Path:
    """Resolve credentials path from config, environment, or defaults."""
    if config is not None and hasattr(config, "credentials_path"):
        cred_path = getattr(config, "credentials_path", None)
        if cred_path:
            return Path(cred_path)

    env_path = os.environ.get("POLYLOGUE_CREDENTIAL_PATH")
    if env_path:
        return Path(env_path).expanduser()

    default_path = default_credentials_path(config)
    if default_path.exists():
        return default_path

    if ui is not None and not getattr(ui, "plain", True):
        prompt = f"Path to Google OAuth client JSON (default {default_path}):"
        response = getattr(ui, "input", lambda p, default: None)(prompt, default=str(default_path))
        if response:
            candidate = Path(response).expanduser()
            if candidate.exists():
                default_path.parent.mkdir(parents=True, exist_ok=True)
                if candidate != default_path:
                    shutil.copy(candidate, default_path)
                return default_path

    raise DriveAuthError(
        f"Drive credentials not found. Set POLYLOGUE_CREDENTIAL_PATH or place a client JSON at {default_path}."
    )


def _resolve_token_path(config: object | None = None) -> Path:
    """Resolve token path from config, environment, or defaults."""
    if config is not None and hasattr(config, "token_path"):
        token_path = getattr(config, "token_path", None)
        if token_path:
            return Path(token_path)

    env_path = os.environ.get("POLYLOGUE_TOKEN_PATH")
    if env_path:
        return Path(env_path).expanduser()

    return default_token_path(config)


class DriveAuthManager:
    """Owns Drive credential acquisition and token lifecycle."""

    def __init__(
        self,
        *,
        ui: object | None = None,
        credentials_path: Path | None = None,
        token_path: Path | None = None,
        config: object | None = None,
    ) -> None:
        self._credentials_path = credentials_path
        self._token_path = token_path
        self._config = config
        self._prompter: DriveAuthPrompter | None = None
        if ui is not None and not getattr(ui, "plain", True):
            self._prompter = UIAuthPrompter(ui)

        resolved_token_path = token_path or _resolve_token_path(config)
        self._token_store: TokenStore = create_token_store(resolved_token_path.parent)

    def _resolved_token_path(self) -> Path:
        return self._token_path or _resolve_token_path(self._config)

    def _load_cached_credentials(self, credentials_cls: Any, token_path: Path):
        return load_cached_credentials(
            token_store=self._token_store,
            credentials_cls=credentials_cls,
            token_path=token_path,
        )

    def _persist_token(self, creds: Any, token_path: Path) -> None:
        persist_token(token_store=self._token_store, creds=creds, token_path=token_path)

    def _refresh_credentials_if_needed(self, creds: Any, token_path: Path) -> Any:
        return refresh_credentials_if_needed(
            creds=creds,
            token_path=token_path,
            token_store=self._token_store,
        )

    def _run_manual_auth_flow(self, flow: Any) -> Any:
        return run_manual_auth_flow(flow=flow, prompter=self._prompter)

    def load_credentials(self) -> Any:
        """Load or acquire Google OAuth credentials. Triggers interactive auth if needed."""
        credentials_cls = _import_auth_module("google.oauth2.credentials").Credentials
        installed_app_flow_cls = _import_auth_module("google_auth_oauthlib.flow").InstalledAppFlow

        token_path = self._resolved_token_path()
        cached = self._load_cached_credentials(credentials_cls, token_path)
        creds = self._refresh_credentials_if_needed(cached.creds, token_path)
        if creds and creds.valid:
            return creds
        if cached.had_invalid_token_path and self._prompter is None:
            raise DriveAuthError(
                f"Drive token at {token_path} is invalid or expired. "
                "Delete it and re-run with --interactive to re-authorize."
            )

        credentials_path = self._credentials_path or _resolve_credentials_path(
            # Pass None when no interactive prompter — resolution will error if creds missing
            None if self._prompter is None else _PromptBridge(self._prompter),
            self._config,
        )
        if not credentials_path.exists():
            raise DriveAuthError(f"Drive credentials not found: {credentials_path}")
        if self._prompter is None:
            raise DriveAuthError(
                "Drive authorization required but no interactive UI is available. "
                "Run with --interactive or set POLYLOGUE_TOKEN_PATH with a valid token."
            )
        flow = installed_app_flow_cls.from_client_secrets_file(str(credentials_path), SCOPES)
        try:
            creds = flow.run_local_server(open_browser=False, port=0)
        except OSError as exc:
            logger.info("Local server auth unavailable (%s). Using manual flow.", exc)
            creds = self._run_manual_auth_flow(flow)
        except Exception as exc:
            logger.warning("Unexpected auth error: %s. Falling back to manual flow.", exc)
            creds = self._run_manual_auth_flow(flow)
        self._persist_token(creds, token_path)
        return creds

    def revoke(self) -> None:
        """Delete all stored tokens."""
        token_path = self._resolved_token_path()
        self._token_store.delete("drive_token")
        if token_path.exists():
            token_path.unlink()


__all__ = [
    "DriveAuthManager",
    "DriveAuthPrompter",
    "UIAuthPrompter",
    "_resolve_credentials_path",
    "_resolve_token_path",
    "default_credentials_path",
    "default_token_path",
]

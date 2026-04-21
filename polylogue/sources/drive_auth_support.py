"""Support helpers for Drive OAuth flows."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType
from typing import Protocol

from polylogue.logging import get_logger
from polylogue.sources.drive_types import (
    SCOPES,
    CachedCredentialState,
    DriveAuthError,
    DriveAuthFlowLike,
    DriveCredentialLike,
    DriveCredentialsFactory,
    DriveTokenStoreLike,
    DriveUILike,
)

logger = get_logger(__name__)


class DriveAuthPrompter(Protocol):
    """Interface for interactive OAuth prompts."""

    def announce_auth_url(self, url: str) -> None: ...

    def request_authorization_code(self) -> str | None: ...


class UIAuthPrompter:
    """Adapter bridging a generic UI object to DriveAuthPrompter."""

    def __init__(self, ui: DriveUILike) -> None:
        self._ui = ui

    def announce_auth_url(self, url: str) -> None:
        console = getattr(self._ui, "console", None)
        if console:
            getattr(console, "print", print)("Open this URL in your browser to authorize Drive access:")
            getattr(console, "print", print)(url)

    def request_authorization_code(self) -> str | None:
        code = getattr(self._ui, "input", lambda x: None)("Paste the authorization code")
        return code or None


class _PromptBridge:
    """Minimal bridge so credential resolution sees a non-plain UI when a prompter exists."""

    def __init__(self, prompter: DriveAuthPrompter) -> None:
        self._prompter = prompter
        self.console = None

    @property
    def plain(self) -> bool:
        return False

    def input(self, prompt: str, *, default: str | None = None) -> str | None:
        del prompt
        del default
        return None


def import_auth_module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise DriveAuthError(
            "Drive dependencies are not available. "
            "Install google-api-python-client + google-auth-oauthlib "
            "or run Polylogue from a Nix build/dev shell."
        ) from exc


def load_cached_credentials(
    *,
    token_store: DriveTokenStoreLike,
    credentials_cls: DriveCredentialsFactory,
    token_path: Path,
) -> CachedCredentialState:
    creds = None
    had_invalid_token_path = False

    token_data = token_store.load("drive_token")
    if token_data:
        try:
            creds = credentials_cls.from_authorized_user_info(json.loads(token_data), SCOPES)
        except (OSError, ValueError, json.JSONDecodeError):
            creds = None

    if creds is None and token_path.exists():
        try:
            creds = credentials_cls.from_authorized_user_file(str(token_path), SCOPES)
        except (OSError, ValueError):
            had_invalid_token_path = True
            creds = None

    return CachedCredentialState(creds=creds, had_invalid_token_path=had_invalid_token_path)


def persist_token(
    *,
    token_store: DriveTokenStoreLike,
    creds: DriveCredentialLike,
    token_path: Path,
) -> None:
    token_store.save("drive_token", creds.to_json())
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(creds.to_json(), encoding="utf-8")
    token_path.chmod(0o600)


def refresh_credentials_if_needed(
    *,
    creds: DriveCredentialLike | None,
    token_path: Path,
    token_store: DriveTokenStoreLike,
) -> DriveCredentialLike | None:
    if creds and creds.expired and creds.refresh_token:
        try:
            import google.auth.transport.requests as _gtr

            transport = _gtr.Request()
            transport.session.timeout = 30
            creds.refresh(transport)
        except Exception as exc:
            raise DriveAuthError(
                f"Failed to refresh OAuth token: {exc}. Try re-authenticating with 'polylogue auth'."
            ) from exc

    if creds and creds.valid:
        persist_token(token_store=token_store, creds=creds, token_path=token_path)
        return creds

    if creds and not creds.valid and not creds.refresh_token:
        raise DriveAuthError(
            f"Drive token at {token_path} is invalid and cannot be refreshed "
            "(no refresh token). Delete it and re-run with --interactive to re-authorize."
        )

    return creds


def run_manual_auth_flow(
    *,
    flow: DriveAuthFlowLike,
    prompter: DriveAuthPrompter | None,
) -> DriveCredentialLike:
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
    if prompter is not None:
        prompter.announce_auth_url(auth_url)
    code = prompter.request_authorization_code() if prompter is not None else None
    if not code:
        raise DriveAuthError("Drive authorization cancelled.") from None
    try:
        flow.fetch_token(code=code)
    except Exception as exc:
        raise DriveAuthError(f"Drive authorization failed: {exc}") from exc
    return flow.credentials


__all__ = [
    "DriveAuthPrompter",
    "UIAuthPrompter",
    "_PromptBridge",
    "import_auth_module",
    "load_cached_credentials",
    "persist_token",
    "refresh_credentials_if_needed",
    "run_manual_auth_flow",
]

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from os import PathLike
from typing import Protocol, runtime_checkable

from ...errors import PolylogueError

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
GEMINI_PROMPT_MIME_TYPE = "application/vnd.google-makersuite.prompt"


class DriveError(PolylogueError):
    pass


class DriveAuthError(DriveError):
    pass


class DriveNotFoundError(DriveError):
    pass


@dataclass
class DriveFile:
    file_id: str
    name: str
    mime_type: str
    modified_time: str | None
    size_bytes: int | None


@dataclass(frozen=True, slots=True)
class DriveRetryPolicy:
    retries: int
    retry_base: float


class DriveConfigLike(Protocol):
    @property
    def credentials_path(self) -> str | PathLike[str] | None: ...

    @property
    def token_path(self) -> str | PathLike[str] | None: ...

    @property
    def retry_count(self) -> int | None: ...


class DriveConsoleLike(Protocol):
    def print(self, *args: object, **kwargs: object) -> None: ...


@runtime_checkable
class DriveUILike(Protocol):
    @property
    def plain(self) -> bool: ...

    @property
    def console(self) -> DriveConsoleLike | None: ...

    def input(self, prompt: str, *, default: str | None = None) -> str | None: ...


class DriveCredentialLike(Protocol):
    valid: bool
    expired: bool
    refresh_token: str | None

    def to_json(self) -> str: ...

    def refresh(self, request: object) -> None: ...


class DriveCredentialsFactory(Protocol):
    @classmethod
    def from_authorized_user_info(
        cls,
        info: object,
        scopes: Sequence[str],
    ) -> DriveCredentialLike: ...

    @classmethod
    def from_authorized_user_file(
        cls,
        filename: str,
        scopes: Sequence[str],
    ) -> DriveCredentialLike: ...


class DriveAuthFlowLike(Protocol):
    credentials: DriveCredentialLike

    def authorization_url(self, *, prompt: str, access_type: str) -> tuple[str, object]: ...

    def fetch_token(self, *, code: str) -> None: ...


class DriveLocalServerFlowLike(DriveAuthFlowLike, Protocol):
    def run_local_server(self, *, open_browser: bool, port: int) -> DriveCredentialLike: ...


class DriveAuthFlowFactory(Protocol):
    @classmethod
    def from_client_secrets_file(
        cls,
        filename: str,
        scopes: Sequence[str],
    ) -> DriveLocalServerFlowLike: ...


class DriveTokenStoreLike(Protocol):
    def load(self, key: str) -> str | None: ...

    def save(self, key: str, value: str) -> None: ...

    def delete(self, key: str) -> None: ...


@dataclass(frozen=True)
class CachedCredentialState:
    creds: DriveCredentialLike | None
    had_invalid_token_path: bool


__all__ = [
    "FOLDER_MIME_TYPE",
    "GEMINI_PROMPT_MIME_TYPE",
    "SCOPES",
    "CachedCredentialState",
    "DriveAuthFlowFactory",
    "DriveAuthFlowLike",
    "DriveLocalServerFlowLike",
    "DriveAuthError",
    "DriveConfigLike",
    "DriveConsoleLike",
    "DriveCredentialLike",
    "DriveCredentialsFactory",
    "DriveError",
    "DriveFile",
    "DriveNotFoundError",
    "DriveRetryPolicy",
    "DriveTokenStoreLike",
    "DriveUILike",
]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import PolylogueError

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


@dataclass(frozen=True)
class CachedCredentialState:
    creds: Any | None
    had_invalid_token_path: bool


__all__ = [
    "FOLDER_MIME_TYPE",
    "GEMINI_PROMPT_MIME_TYPE",
    "SCOPES",
    "CachedCredentialState",
    "DriveAuthError",
    "DriveError",
    "DriveFile",
    "DriveNotFoundError",
]

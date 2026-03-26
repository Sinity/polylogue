"""Sources package — unified parsing from all AI providers.

- sources/parsers/: JSON → ParsedConversation for each provider
- sources/providers/: Pydantic models for provider export formats
- sources/source_parsing.py: Source walking and parsed conversation iteration
- sources/source_acquisition.py: Raw source acquisition iteration
- sources/drive_*.py / drive.py: Google Drive auth, gateway, and source access
"""

from __future__ import annotations

# Drive integration
from .drive import download_drive_files, iter_drive_conversations
from .drive_source import DriveSourceAPI, DriveSourceClient, build_drive_source_client
from .drive_types import (
    DriveAuthError,
    DriveError,
    DriveFile,
    DriveNotFoundError,
)

# Core parsing
from .parsers.base import ParsedAttachment, ParsedConversation, ParsedMessage
from .source_parsing import iter_source_conversations

__all__ = [
    "DriveAuthError",
    "DriveError",
    "DriveFile",
    "DriveNotFoundError",
    "DriveSourceAPI",
    "DriveSourceClient",
    "ParsedAttachment",
    "ParsedConversation",
    "ParsedMessage",
    "build_drive_source_client",
    "download_drive_files",
    "iter_drive_conversations",
    "iter_source_conversations",
]

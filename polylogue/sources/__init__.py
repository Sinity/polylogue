"""Sources package — unified parsing from all AI providers.

- sources/parsers/: JSON → ParsedConversation for each provider
- sources/providers/: Pydantic models for provider export formats
- sources/source.py: Source walking and raw conversation iteration
- sources/drive*.py: Google Drive integration
"""

from __future__ import annotations

# Drive integration
from .drive import download_drive_files, iter_drive_conversations
from .drive_client import DriveClient
from .drive_types import (
    DriveAuthError,
    DriveError,
    DriveFile,
    DriveNotFoundError,
)

# Core parsing
from .source import (
    ParsedAttachment,
    ParsedConversation,
    ParsedMessage,
    iter_source_conversations,
)

__all__ = [
    "DriveAuthError",
    "DriveClient",
    "DriveError",
    "DriveFile",
    "DriveNotFoundError",
    "ParsedAttachment",
    "ParsedConversation",
    "ParsedMessage",
    "download_drive_files",
    "iter_drive_conversations",
    "iter_source_conversations",
]

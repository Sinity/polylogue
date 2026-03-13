"""Sources package — unified parsing from all AI providers.

- sources/parsers/: JSON → ParsedConversation for each provider
- sources/providers/: Pydantic models for provider export formats
- sources/source.py: File/Drive reading → RecordBundle
- sources/drive*.py: Google Drive integration
"""

from __future__ import annotations

# Drive integration
from .drive import download_drive_files, iter_drive_conversations
from .drive_types import (
    DriveAuthError,
    DriveError,
    DriveFile,
    DriveNotFoundError,
)
from .drive_client import DriveClient

# Core parsing
from .source import (
    ParsedAttachment,
    ParsedConversation,
    ParsedMessage,
    RecordBundle,
    SaveResult,
    iter_source_conversations,
    parse_drive_payload,
    save_bundle,
)

__all__ = [
    "DriveAuthError",
    "DriveClient",
    "DriveError",
    "DriveFile",
    "DriveNotFoundError",
    "RecordBundle",
    "SaveResult",
    "ParsedAttachment",
    "ParsedConversation",
    "ParsedMessage",
    "download_drive_files",
    "save_bundle",
    "iter_drive_conversations",
    "iter_source_conversations",
    "parse_drive_payload",
]

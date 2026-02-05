"""Sources package — unified ingestion from all AI providers.

Merges the former importers/, ingestion/, and providers/ packages.
- sources/parsers/: JSON → ParsedConversation for each provider
- sources/providers/: Pydantic models for provider export formats
- sources/source.py: File/Drive reading → IngestBundle
- sources/drive*.py: Google Drive integration
"""

from __future__ import annotations

# Drive integration
from .drive import download_drive_files, iter_drive_conversations
from .drive_client import (
    DriveAuthError,
    DriveClient,
    DriveError,
    DriveFile,
    DriveNotFoundError,
)

# Core ingestion
from .source import (
    IngestBundle,
    IngestResult,
    ParsedAttachment,
    ParsedConversation,
    ParsedMessage,
    ingest_bundle,
    iter_source_conversations,
    parse_drive_payload,
)

__all__ = [
    "DriveAuthError",
    "DriveClient",
    "DriveError",
    "DriveFile",
    "DriveNotFoundError",
    "IngestBundle",
    "IngestResult",
    "ParsedAttachment",
    "ParsedConversation",
    "ParsedMessage",
    "download_drive_files",
    "ingest_bundle",
    "iter_drive_conversations",
    "iter_source_conversations",
    "parse_drive_payload",
]

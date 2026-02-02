"""Ingestion package for Polylogue.

This package handles all data ingestion operations including:
- Local file ingestion (JSON, JSONL, ZIP)
- Google Drive integration
- Provider detection and parsing
"""

from __future__ import annotations

# Drive integration
from .drive import DriveDownloadResult, download_drive_files, iter_drive_conversations
from .drive_client import (
    DriveAuthError,
    DriveClient,
    DriveError,
    DriveFile,
    DriveNotFoundError,
    default_credentials_path,
    default_token_path,
)

# Core ingestion
from .source import (
    IngestBundle,
    IngestResult,
    ParsedAttachment,
    ParsedConversation,
    ParsedMessage,
    RawConversationData,
    ingest_bundle,
    iter_source_conversations,
    iter_source_conversations_with_raw,
    parse_drive_payload,
)

__all__ = [
    # Core ingestion
    "IngestBundle",
    "IngestResult",
    "ingest_bundle",
    # Source ingestion
    "ParsedAttachment",
    "ParsedConversation",
    "ParsedMessage",
    "RawConversationData",
    "iter_source_conversations",
    "iter_source_conversations_with_raw",
    "parse_drive_payload",
    # Drive integration
    "DriveDownloadResult",
    "download_drive_files",
    "iter_drive_conversations",
    "DriveAuthError",
    "DriveClient",
    "DriveError",
    "DriveFile",
    "DriveNotFoundError",
    "default_credentials_path",
    "default_token_path",
]

"""Claude Code record types beyond messages.

Claude Code exports contain multiple record types:

MESSAGE RECORDS (conversation content):
    - user: User input messages
    - assistant: Assistant responses with content blocks
    - system: System messages (context, instructions)

METADATA RECORDS (session state/progress):
    - progress: Hook execution and tool progress events
    - file-history-snapshot: File state at points in time
    - queue-operation: Message queue operations (enqueue/remove)
    - custom-title: Conversation title changes

This module defines extractors for the metadata records, complementing
the message extraction in extractors.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from polylogue.lib.timestamps import parse_timestamp


class RecordType(str, Enum):
    """All Claude Code record types."""

    # Message types
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    # Metadata types
    PROGRESS = "progress"
    FILE_HISTORY_SNAPSHOT = "file-history-snapshot"
    QUEUE_OPERATION = "queue-operation"
    CUSTOM_TITLE = "custom-title"

    @classmethod
    def is_message(cls, record_type: str) -> bool:
        """Check if record type is a conversation message."""
        return record_type in (cls.USER.value, cls.ASSISTANT.value, cls.SYSTEM.value)

    @classmethod
    def is_metadata(cls, record_type: str) -> bool:
        """Check if record type is metadata (not a message)."""
        return not cls.is_message(record_type)


class HookEvent(str, Enum):
    """Hook events tracked in progress records."""

    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"
    SUBAGENT_STOP = "SubagentStop"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    PRE_COMPACT = "PreCompact"
    NOTIFICATION = "Notification"


class QueueOperation(str, Enum):
    """Queue operations."""

    ENQUEUE = "enqueue"
    REMOVE = "remove"


# =============================================================================
# Metadata Record Types
# =============================================================================


@dataclass
class ProgressRecord:
    """Hook/tool execution progress event.

    These records track:
    - Hook lifecycle events (SessionStart, PreToolUse, etc.)
    - Tool execution progress
    - Associated tool use IDs for correlation
    """

    hook_event: str | None
    hook_name: str | None
    tool_use_id: str | None
    parent_tool_use_id: str | None
    timestamp: datetime | None
    session_id: str | None

    # Original data
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> ProgressRecord:
        """Extract ProgressRecord from raw Claude Code record."""
        data = raw.get("data", {})
        return cls(
            hook_event=data.get("hookEvent"),
            hook_name=data.get("hookName"),
            tool_use_id=raw.get("toolUseID"),
            parent_tool_use_id=raw.get("parentToolUseID"),
            timestamp=parse_timestamp(raw.get("timestamp")),
            session_id=raw.get("sessionId"),
            raw=raw,
        )


@dataclass
class FileBackup:
    """A single file backup entry."""

    path: str
    content_hash: str | None = None
    # Could add: size, modified_time if available


@dataclass
class FileHistorySnapshot:
    """File state snapshot at a point in time.

    These records capture:
    - Which files were being tracked
    - File content hashes (for change detection)
    - Associated with a specific message ID
    """

    message_id: str
    timestamp: datetime | None
    tracked_files: list[FileBackup]
    is_snapshot_update: bool

    # Original data
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> FileHistorySnapshot:
        """Extract FileHistorySnapshot from raw Claude Code record."""
        snapshot = raw.get("snapshot", {})
        backups = snapshot.get("trackedFileBackups", {})

        tracked = [
            FileBackup(path=path, content_hash=data.get("hash") if isinstance(data, dict) else None)
            for path, data in backups.items()
        ]

        return cls(
            message_id=raw.get("messageId", ""),
            timestamp=parse_timestamp(snapshot.get("timestamp")),
            tracked_files=tracked,
            is_snapshot_update=raw.get("isSnapshotUpdate", False),
            raw=raw,
        )


@dataclass
class QueueOperationRecord:
    """Message queue operation.

    Tracks when messages are enqueued for processing or removed.
    Useful for understanding message flow and timing.
    """

    operation: str
    timestamp: datetime | None
    session_id: str | None
    content: Any | None  # The queued content, if present

    # Original data
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> QueueOperationRecord:
        """Extract QueueOperationRecord from raw Claude Code record."""
        return cls(
            operation=raw.get("operation", ""),
            timestamp=parse_timestamp(raw.get("timestamp")),
            session_id=raw.get("sessionId"),
            content=raw.get("content"),
            raw=raw,
        )


# =============================================================================
# Dispatcher
# =============================================================================


def extract_metadata_record(
    raw: dict[str, Any],
) -> ProgressRecord | FileHistorySnapshot | QueueOperationRecord | None:
    """Extract metadata record from raw Claude Code data.

    Returns None if the record is a message type (use extract_message instead).
    """
    record_type = raw.get("type", "")

    if record_type == RecordType.PROGRESS.value:
        return ProgressRecord.from_raw(raw)
    elif record_type == RecordType.FILE_HISTORY_SNAPSHOT.value:
        return FileHistorySnapshot.from_raw(raw)
    elif record_type == RecordType.QUEUE_OPERATION.value:
        return QueueOperationRecord.from_raw(raw)
    elif RecordType.is_message(record_type):
        return None  # Use extract_message for these
    else:
        return None  # Unknown type


def classify_record(raw: dict[str, Any]) -> tuple[str, str]:
    """Classify a Claude Code record.

    Returns:
        (category, record_type) where category is "message" or "metadata"
    """
    record_type = raw.get("type", "unknown")
    category = "message" if RecordType.is_message(record_type) else "metadata"
    return category, record_type

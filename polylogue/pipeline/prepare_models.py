"""Typed models and caches for raw-to-record preparation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from polylogue.storage.state_views import ExistingConversation
from polylogue.storage.store import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
)
from polylogue.types import ConversationId, MessageId

if TYPE_CHECKING:
    from polylogue.storage.backends.async_sqlite import SQLiteBackend


class RecordBundle(BaseModel):
    conversation: ConversationRecord
    messages: list[MessageRecord]
    attachments: list[AttachmentRecord]
    content_blocks: list[ContentBlockRecord] = Field(default_factory=list)


class SaveResult(BaseModel):
    conversations: int
    messages: int
    attachments: int
    skipped_conversations: int
    skipped_messages: int
    skipped_attachments: int


def _timestamp_sort_key(ts: str | None) -> float | None:
    """Convert a timestamp string to a numeric sort key."""
    if ts is None:
        return None
    try:
        value = float(ts)
        if value > 32503680000:
            value = value / 1000
        return value
    except (ValueError, TypeError):
        pass

    from datetime import datetime, timezone

    try:
        normalized = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


@dataclass
class PrepareCache:
    """Pre-loaded batch data for prepare_records."""

    existing: dict[str, ExistingConversation] = field(default_factory=dict)
    known_ids: set[str] = field(default_factory=set)
    message_ids: dict[str, dict[str, MessageId]] = field(default_factory=dict)

    @classmethod
    async def load(cls, backend: SQLiteBackend, candidate_cids: set[str]) -> PrepareCache:
        cache = cls()
        if not candidate_cids:
            return cache

        cid_list = list(candidate_cids)
        for chunk_start in range(0, len(cid_list), 500):
            chunk = cid_list[chunk_start : chunk_start + 500]
            placeholders = ", ".join("?" for _ in chunk)
            async with backend.connection() as conn:
                cursor = await conn.execute(
                    f"SELECT conversation_id, content_hash FROM conversations "
                    f"WHERE conversation_id IN ({placeholders})",
                    tuple(chunk),
                )
                rows = await cursor.fetchall()
            for row in rows:
                cid = row["conversation_id"]
                cache.existing[cid] = ExistingConversation(
                    conversation_id=cid,
                    content_hash=row["content_hash"],
                )
                cache.known_ids.add(cid)

        existing_cids = list(cache.known_ids)
        for chunk_start in range(0, len(existing_cids), 500):
            chunk = existing_cids[chunk_start : chunk_start + 500]
            placeholders = ", ".join("?" for _ in chunk)
            async with backend.connection() as conn:
                cursor = await conn.execute(
                    f"SELECT conversation_id, provider_message_id, message_id "
                    f"FROM messages WHERE conversation_id IN ({placeholders}) "
                    f"AND provider_message_id IS NOT NULL",
                    tuple(chunk),
                )
                rows = await cursor.fetchall()
            for row in rows:
                cid = row["conversation_id"]
                if cid not in cache.message_ids:
                    cache.message_ids[cid] = {}
                if row["provider_message_id"]:
                    cache.message_ids[cid][str(row["provider_message_id"])] = MessageId(row["message_id"])

        return cache


@dataclass
class AttachmentMaterializationPlan:
    move_before_save: list[tuple[Path, Path]] = field(default_factory=list)
    delete_after_save: list[Path] = field(default_factory=list)


@dataclass
class TransformResult:
    bundle: RecordBundle
    materialization_plan: AttachmentMaterializationPlan
    content_hash: str
    candidate_cid: ConversationId
    message_id_map: dict[str, MessageId]


@dataclass(frozen=True)
class PreparedBundle:
    bundle: RecordBundle
    materialization_plan: AttachmentMaterializationPlan
    cid: ConversationId
    changed: bool


@dataclass(frozen=True)
class PersistedConversationResult:
    conversation_id: ConversationId
    save_result: SaveResult
    content_changed: bool

    @property
    def counts(self) -> dict[str, int]:
        return self.save_result.model_dump()


__all__ = [
    "AttachmentMaterializationPlan",
    "PersistedConversationResult",
    "PreparedBundle",
    "PrepareCache",
    "RecordBundle",
    "SaveResult",
    "TransformResult",
    "_timestamp_sort_key",
]

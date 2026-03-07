"""Async parse preparation logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.lib.log import get_logger
from polylogue.pipeline.enrichment import enrich_message_metadata
from polylogue.pipeline.ids import (
    attachment_content_id,
    conversation_content_hash,
    materialize_attachment_path,
    message_content_hash,
    move_attachment_to_archive,
)
from polylogue.pipeline.ids import (
    conversation_id as make_conversation_id,
)
from polylogue.pipeline.ids import (
    message_id as make_message_id,
)
from polylogue.sources.source import RecordBundle, SaveResult, save_bundle
from polylogue.storage.store import AttachmentRecord, ConversationRecord, ExistingConversation, MessageRecord
from polylogue.types import AttachmentId, ConversationId, MessageId

if TYPE_CHECKING:
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository

logger = get_logger(__name__)


def _timestamp_sort_key(ts: str | None) -> float | None:
    """Convert a timestamp string to a numeric sort key.

    Handles both Unix epoch (numeric) and ISO-8601 formats.
    Returns None for None timestamps (sorted last by queries).
    """
    if ts is None:
        return None
    # Fast path: numeric (epoch seconds or milliseconds)
    try:
        val = float(ts)
        # Values > year 3000 in seconds are likely milliseconds
        if val > 32503680000:
            val = val / 1000
        return val
    except (ValueError, TypeError):
        pass
    # Slow path: ISO-8601 → epoch
    from datetime import datetime, timezone

    try:
        # Handle 'Z' suffix and various ISO formats
        normalized = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


@dataclass
class PrepareCache:
    """Pre-loaded batch data for prepare_records, replacing per-conversation DB queries.

    Instead of 3 queries per conversation (existing lookup, parent check, message IDs),
    we bulk-load all needed data for an entire batch in 2 queries total.
    """

    # {conversation_id: ExistingConversation}
    existing: dict[str, ExistingConversation] = field(default_factory=dict)
    # Set of all conversation_ids that exist in the DB (for parent FK check)
    known_ids: set[str] = field(default_factory=set)
    # {conversation_id: {provider_message_id: message_id}}
    message_ids: dict[str, dict[str, MessageId]] = field(default_factory=dict)

    @classmethod
    async def load(cls, backend: SQLiteBackend, candidate_cids: set[str]) -> PrepareCache:
        """Bulk-load all data needed for a batch of conversations.

        Replaces N per-conversation queries with 2 bulk queries:
        1. Existing conversations (id + content_hash)
        2. Message ID mappings for all known conversations
        """
        cache = cls()
        if not candidate_cids:
            return cache

        cid_list = list(candidate_cids)

        # Query 1: Existing conversations — bulk fetch by conversation_id
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
                    conversation_id=cid, content_hash=row["content_hash"],
                )
                cache.known_ids.add(cid)

        # Query 2: Message ID mappings for existing conversations
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
    """Filesystem actions needed to align attachment paths with archive storage."""

    move_before_save: list[tuple[Path, Path]] = field(default_factory=list)
    delete_after_save: list[Path] = field(default_factory=list)


def _plan_attachment_materialization(
    source_path: str | None,
    target_path: str | None,
) -> AttachmentMaterializationPlan:
    """Decide how an attachment path should be materialized, if at all."""
    if not source_path or not target_path or source_path == target_path:
        return AttachmentMaterializationPlan()

    source = Path(source_path)
    target = Path(target_path)
    if not source.exists():
        return AttachmentMaterializationPlan()
    if target.exists():
        return AttachmentMaterializationPlan(delete_after_save=[source])
    return AttachmentMaterializationPlan(move_before_save=[(source, target)])


async def prepare_records(
    convo,
    source_name: str,
    *,
    archive_root: Path,
    backend: SQLiteBackend | None = None,
    repository: ConversationRepository | None = None,
    raw_id: str | None = None,
    cache: PrepareCache | None = None,
) -> tuple[str, dict[str, int], bool]:
    """Async version of prepare_records for converting ParsedConversation to records.

    Args:
        convo: ParsedConversation to prepare
        source_name: Name of the source
        archive_root: Root directory for archived conversations
        backend: SQLiteBackend for database lookups
        repository: ConversationRepository for saving
        raw_id: Optional raw conversation ID
        cache: Optional PrepareCache for batch lookups (avoids per-conversation queries)

    Returns:
        Tuple of (conversation_id, result_counts, content_changed)
    """
    if repository is None and backend is None:
        raise ValueError("prepare_records requires a repository or backend")
    if repository is None:
        from polylogue.storage.repository import ConversationRepository

        repository = ConversationRepository(backend=backend)
    if backend is None:
        backend = repository.backend

    # Skip conversations with no messages — these are empty shells from
    # parse filtering (e.g. JSONL files with only metadata records)
    if not convo.messages:
        cid = make_conversation_id(convo.provider_name, convo.provider_conversation_id)
        logger.debug("Skipping empty conversation (no messages)", conversation_id=cid)
        return (
            cid,
            {"conversations": 0, "messages": 0, "attachments": 0,
             "skipped_conversations": 1, "skipped_messages": 0, "skipped_attachments": 0},
            False,
        )

    content_hash = conversation_content_hash(convo)
    candidate_cid = make_conversation_id(convo.provider_name, convo.provider_conversation_id)

    # Look up existing conversation — use batch cache if available, else query DB
    existing = None
    if cache is not None:
        existing = cache.existing.get(candidate_cid)
    elif backend:
        async with backend.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT conversation_id, content_hash
                FROM conversations
                WHERE conversation_id = ?
                LIMIT 1
                """,
                (candidate_cid,),
            )
            row = await cursor.fetchone()
        if row:
            existing = ExistingConversation(conversation_id=row["conversation_id"], content_hash=row["content_hash"])

    if existing:
        cid: ConversationId = ConversationId(existing.conversation_id)
        changed = existing.content_hash != content_hash
    else:
        cid = candidate_cid
        changed = False

    # Resolve parent conversation ID if present (provider ID → internal polylogue ID)
    # Only set FK if the parent conversation already exists in the database,
    # otherwise the FK constraint fails when child is parsed before parent.
    parent_conversation_id = None
    if convo.parent_conversation_provider_id:
        candidate_parent = make_conversation_id(convo.provider_name, convo.parent_conversation_provider_id)
        if cache is not None:
            if candidate_parent in cache.known_ids:
                parent_conversation_id = candidate_parent
        elif backend:
            async with backend.connection() as conn:
                cursor = await conn.execute(
                    "SELECT 1 FROM conversations WHERE conversation_id = ?",
                    (candidate_parent,),
                )
                if await cursor.fetchone():
                    parent_conversation_id = candidate_parent

    # Merge source into provider_meta rather than overwriting
    merged_provider_meta: dict[str, object] = {"source": source_name}
    if convo.provider_meta:
        merged_provider_meta.update(convo.provider_meta)

    conversation_record = ConversationRecord(
        conversation_id=cid,
        provider_name=convo.provider_name,
        provider_conversation_id=convo.provider_conversation_id,
        title=convo.title,
        created_at=convo.created_at,
        updated_at=convo.updated_at,
        sort_key=_timestamp_sort_key(convo.updated_at),
        content_hash=content_hash,
        provider_meta=merged_provider_meta,
        parent_conversation_id=parent_conversation_id,
        branch_type=convo.branch_type,
        raw_id=raw_id,
    )

    messages: list[MessageRecord] = []
    message_ids: dict[str, MessageId] = {}

    # Retrieve existing message ID mapping — use batch cache if available
    existing_message_ids: dict[str, MessageId] = {}
    if cache is not None:
        existing_message_ids = cache.message_ids.get(cid, {})
    elif backend:
        async with backend.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT provider_message_id, message_id
                FROM messages
                WHERE conversation_id = ? AND provider_message_id IS NOT NULL
                """,
                (cid,),
            )
            rows = await cursor.fetchall()
        existing_message_ids = {
            str(row["provider_message_id"]): MessageId(row["message_id"]) for row in rows if row["provider_message_id"]
        }

    # First pass: build complete message_ids mapping (needed for parent resolution)
    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        mid: MessageId = existing_message_ids.get(provider_message_id) or make_message_id(cid, provider_message_id)
        message_ids[str(provider_message_id)] = mid

    # Second pass: create MessageRecords with resolved parent IDs
    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        mid = message_ids[str(provider_message_id)]
        message_hash = message_content_hash(msg, provider_message_id)

        # Resolve parent message ID if present
        parent_message_id: MessageId | None = None
        if msg.parent_message_provider_id:
            parent_message_id = message_ids.get(str(msg.parent_message_provider_id))

        # Enrich provider_meta with code language detection
        enriched_meta = enrich_message_metadata(msg.provider_meta)

        messages.append(
            MessageRecord(
                message_id=mid,
                conversation_id=cid,
                provider_message_id=provider_message_id,
                role=msg.role,
                text=msg.text,
                timestamp=msg.timestamp,
                sort_key=_timestamp_sort_key(msg.timestamp),
                content_hash=message_hash,
                provider_meta=enriched_meta,
                parent_message_id=parent_message_id,
                branch_index=msg.branch_index,
            )
        )

    attachments: list[AttachmentRecord] = []
    materialization_plan = AttachmentMaterializationPlan()
    for att in convo.attachments:
        aid, updated_meta, updated_path = attachment_content_id(convo.provider_name, att, archive_root=archive_root)
        # Merge updated metadata with provider_id if present
        meta: dict[str, object] = dict(updated_meta or {})
        if att.provider_attachment_id:
            meta.setdefault("provider_id", att.provider_attachment_id)
        attachment_plan = _plan_attachment_materialization(att.path, updated_path)
        materialization_plan.move_before_save.extend(attachment_plan.move_before_save)
        materialization_plan.delete_after_save.extend(attachment_plan.delete_after_save)
        message_id_val: MessageId | None = (
            message_ids.get(att.message_provider_id or "") if att.message_provider_id else None
        )
        attachments.append(
            AttachmentRecord(
                attachment_id=AttachmentId(aid),
                conversation_id=cid,
                message_id=message_id_val,
                mime_type=att.mime_type,
                size_bytes=att.size_bytes,
                path=updated_path,
                provider_meta=meta,
            )
        )

    applied_moves: list[tuple[Path, Path]] = []
    try:
        for source_path, target_path in materialization_plan.move_before_save:
            materialize_attachment_path(source_path, target_path)
            applied_moves.append((source_path, target_path))

        result = await save_bundle(
            RecordBundle(
                conversation=conversation_record,
                messages=messages,
                attachments=attachments,
            ),
            repository=repository,
        )
    except Exception:
        for source_path, target_path in reversed(applied_moves):
            if target_path.exists():
                move_attachment_to_archive(target_path, source_path)
        raise

    for duplicate_source in materialization_plan.delete_after_save:
        if duplicate_source.exists():
            duplicate_source.unlink()

    return (
        cid,
        {
            "conversations": result.conversations,
            "messages": result.messages,
            "attachments": result.attachments,
            "skipped_conversations": result.skipped_conversations,
            "skipped_messages": result.skipped_messages,
            "skipped_attachments": result.skipped_attachments,
        },
        changed,
    )


__all__ = [
    "PrepareCache",
    "_timestamp_sort_key",
    "prepare_records",
]

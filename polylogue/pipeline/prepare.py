"""Async parse preparation logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from polylogue.lib.json import dumps as json_dumps
from polylogue.lib.viewports import ToolCategory, classify_tool
from polylogue.logging import get_logger
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
from polylogue.pipeline.semantic import extract_tool_metadata
from polylogue.schemas.code_detection import detect_language
from polylogue.schemas.unified import harmonize_parsed_message
from polylogue.sources.parsers.base import ParsedContentBlock
from polylogue.storage.store import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    ExistingConversation,
    MessageRecord,
)
from polylogue.types import AttachmentId, ConversationId, MessageId

if TYPE_CHECKING:
    from polylogue.sources.parsers.base import ParsedConversation, ParsedMessage
    from polylogue.storage.backends.async_sqlite import SQLiteBackend
    from polylogue.storage.repository import ConversationRepository

logger = get_logger(__name__)


class RecordBundle(BaseModel):
    conversation: ConversationRecord
    messages: list[MessageRecord]
    attachments: list[AttachmentRecord]
    content_blocks: list[ContentBlockRecord] = []


class SaveResult(BaseModel):
    conversations: int
    messages: int
    attachments: int
    skipped_conversations: int
    skipped_messages: int
    skipped_attachments: int


async def save_bundle(bundle: RecordBundle, repository: ConversationRepository) -> SaveResult:
    """Save a bundle of prepared records into the repository."""
    counts = await repository.save_conversation(
        conversation=bundle.conversation,
        messages=bundle.messages,
        attachments=bundle.attachments,
        content_blocks=bundle.content_blocks,
    )
    return SaveResult(**counts)


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


@dataclass
class TransformResult:
    """Output of transform_to_records: pure records derived from a ParsedConversation."""

    bundle: RecordBundle
    materialization_plan: AttachmentMaterializationPlan
    content_hash: str
    candidate_cid: ConversationId
    # provider_message_id → MessageId mapping built from the transform
    message_id_map: dict[str, MessageId]


@dataclass
class EnrichedBundle:
    """Output of enrich_bundle_from_db: bundle with DB-resolved IDs and change flag."""

    bundle: RecordBundle
    materialization_plan: AttachmentMaterializationPlan
    cid: ConversationId
    changed: bool


def _parsed_block_from_harmonized(block) -> ParsedContentBlock | None:
    metadata: dict[str, object] | None = dict(block.raw) if isinstance(block.raw, dict) and block.raw else None

    if block.type.name == "TOOL_USE" and block.tool_call is not None:
        return ParsedContentBlock(
            type="tool_use",
            text=block.text,
            tool_name=block.tool_call.name,
            tool_id=block.tool_call.id,
            tool_input=block.tool_call.input or None,
            metadata=metadata,
        )
    if block.type.name == "TOOL_RESULT":
        tool_id = None
        if isinstance(block.raw, dict):
            raw_tool_id = block.raw.get("tool_use_id") or block.raw.get("tool_id")
            if isinstance(raw_tool_id, str) and raw_tool_id:
                tool_id = raw_tool_id
        return ParsedContentBlock(
            type="tool_result",
            text=block.text,
            tool_id=tool_id,
            metadata=metadata,
        )
    if block.type.name == "CODE":
        if block.language:
            metadata = dict(metadata or {})
            metadata.setdefault("language", block.language)
        return ParsedContentBlock(type="code", text=block.text, metadata=metadata)
    if block.type.name == "THINKING":
        return ParsedContentBlock(type="thinking", text=block.text, metadata=metadata)
    if block.type.name == "IMAGE":
        return ParsedContentBlock(
            type="image",
            text=block.text,
            media_type=block.mime_type,
            metadata=metadata,
        )
    if block.type.name in {"FILE", "AUDIO", "VIDEO"}:
        return ParsedContentBlock(
            type="document",
            text=block.text,
            media_type=block.mime_type,
            metadata=metadata,
        )
    if block.type.name in {"TEXT", "SYSTEM", "ERROR", "UNKNOWN"}:
        return ParsedContentBlock(type="text", text=block.text, metadata=metadata)
    return None


def _canonicalize_message_content(
    provider_name: str,
    message: ParsedMessage,
) -> ParsedMessage:
    harmonized = harmonize_parsed_message(
        provider_name,
        message.provider_meta,
        message_id=message.provider_message_id,
        role=str(message.role),
        text=message.text,
        timestamp=message.timestamp,
    )
    if harmonized is None:
        return message

    updates: dict[str, object] = {}
    if not message.text and harmonized.text:
        updates["text"] = harmonized.text
    if not message.content_blocks and harmonized.content_blocks:
        content_blocks = [
            parsed_block
            for block in harmonized.content_blocks
            if (parsed_block := _parsed_block_from_harmonized(block)) is not None
        ]
        if content_blocks:
            updates["content_blocks"] = content_blocks
    if not updates:
        return message
    return message.model_copy(update=updates)


def _canonicalize_conversation_content(convo: ParsedConversation) -> ParsedConversation:
    messages = [_canonicalize_message_content(str(convo.provider_name), message) for message in convo.messages]
    if all(original == updated for original, updated in zip(convo.messages, messages, strict=True)):
        return convo
    return convo.model_copy(update={"messages": messages})


def transform_to_records(
    convo: ParsedConversation,
    source_name: str,
    *,
    archive_root: Path,
) -> TransformResult:
    """Build all storage records from a ParsedConversation without any DB access.

    All IDs are freshly computed (no stable-ID reuse from the DB).
    The returned TransformResult carries enough context for enrich_bundle_from_db
    to apply DB-derived ID mappings and determine the change flag.
    """
    convo = _canonicalize_conversation_content(convo)
    content_hash = conversation_content_hash(convo)
    candidate_cid = make_conversation_id(convo.provider_name, convo.provider_conversation_id)

    # Merge source into provider_meta rather than overwriting
    merged_provider_meta: dict[str, object] = {"source": source_name}
    if convo.provider_meta:
        merged_provider_meta.update(convo.provider_meta)

    # Build placeholder conversation record; cid and parent_conversation_id are
    # refined by enrich_bundle_from_db after DB lookups.
    conversation_record = ConversationRecord(
        conversation_id=candidate_cid,
        provider_name=convo.provider_name,
        provider_conversation_id=convo.provider_conversation_id,
        title=convo.title,
        created_at=convo.created_at,
        updated_at=convo.updated_at,
        sort_key=_timestamp_sort_key(convo.updated_at),
        content_hash=content_hash,
        provider_meta=merged_provider_meta,
        parent_conversation_id=None,
        branch_type=convo.branch_type,
        raw_id=None,
    )

    # First pass: build complete message_ids mapping (needed for parent resolution)
    message_id_map: dict[str, MessageId] = {}
    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        mid: MessageId = make_message_id(candidate_cid, provider_message_id)
        message_id_map[str(provider_message_id)] = mid

    messages: list[MessageRecord] = []
    content_block_records: list[ContentBlockRecord] = []

    # Second pass: create MessageRecords with resolved parent IDs
    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        mid = message_id_map[str(provider_message_id)]
        message_hash = message_content_hash(msg, provider_message_id)

        # Resolve parent message ID if present
        parent_message_id: MessageId | None = None
        if msg.parent_message_provider_id:
            parent_message_id = message_id_map.get(str(msg.parent_message_provider_id))

        # Precompute analytics fields from parsed content blocks
        _block_types = {blk.type for blk in msg.content_blocks}
        _msg_word_count = len(msg.text.split()) if msg.text and msg.text.strip() else 0
        _has_tool_use = 1 if (_block_types & {"tool_use", "tool_result"}) or msg.role == "tool" else 0
        _has_thinking = 1 if "thinking" in _block_types else 0

        messages.append(
            MessageRecord(
                message_id=mid,
                conversation_id=candidate_cid,
                provider_message_id=provider_message_id,
                role=msg.role,
                text=msg.text,
                sort_key=_timestamp_sort_key(msg.timestamp),
                content_hash=message_hash,
                parent_message_id=parent_message_id,
                branch_index=msg.branch_index,
                provider_name=convo.provider_name,
                word_count=_msg_word_count,
                has_tool_use=_has_tool_use,
                has_thinking=_has_thinking,
            )
        )

        # Create ContentBlockRecords from structured content blocks
        for block_idx, block in enumerate(msg.content_blocks):
            tool_input_json = None
            if block.tool_input is not None:
                tool_input_json = json_dumps(block.tool_input)

            # Compute semantic_type and enrich metadata at ingest time
            semantic_type: str | None = None
            semantic_metadata: dict | None = block.metadata  # start with existing block metadata

            if block.type == "tool_use" and block.tool_name:
                category = classify_tool(block.tool_name, block.tool_input or {})
                semantic_type = None if category is ToolCategory.OTHER else category.value
                # Extract structured metadata for semantically meaningful tool calls
                tool_meta = extract_tool_metadata(block.tool_name, block.tool_input or {})
                if tool_meta is not None:
                    # Merge with existing block metadata (tool_meta takes precedence for semantic keys)
                    base = dict(block.metadata) if isinstance(block.metadata, dict) else {}
                    base.update(tool_meta)
                    semantic_metadata = base
            elif block.type == "thinking":
                semantic_type = "thinking"
            elif block.type == "code" and block.text and semantic_metadata is None:
                # Detect programming language for code blocks that don't have metadata
                detected_lang = detect_language(block.text)
                if detected_lang:
                    semantic_metadata = {"language": detected_lang}

            metadata_json = None
            if semantic_metadata is not None:
                metadata_json = json_dumps(semantic_metadata)

            content_block_records.append(
                ContentBlockRecord(
                    block_id=ContentBlockRecord.make_id(mid, block_idx),
                    message_id=MessageId(mid),
                    conversation_id=candidate_cid,
                    block_index=block_idx,
                    type=block.type,
                    text=block.text,
                    tool_name=block.tool_name,
                    tool_id=block.tool_id,
                    tool_input=tool_input_json,
                    media_type=block.media_type,
                    metadata=metadata_json,
                    semantic_type=semantic_type,
                )
            )

    attachments: list[AttachmentRecord] = []
    materialization_plan = AttachmentMaterializationPlan()
    for att in convo.attachments:
        aid, updated_meta, updated_path = attachment_content_id(convo.provider_name, att, archive_root=archive_root)
        meta: dict[str, object] = dict(updated_meta or {})
        if att.provider_attachment_id:
            meta.setdefault("provider_id", att.provider_attachment_id)
        attachment_plan = _plan_attachment_materialization(att.path, updated_path)
        materialization_plan.move_before_save.extend(attachment_plan.move_before_save)
        materialization_plan.delete_after_save.extend(attachment_plan.delete_after_save)
        message_id_val: MessageId | None = (
            message_id_map.get(att.message_provider_id or "") if att.message_provider_id else None
        )
        attachments.append(
            AttachmentRecord(
                attachment_id=AttachmentId(aid),
                conversation_id=candidate_cid,
                message_id=message_id_val,
                mime_type=att.mime_type,
                size_bytes=att.size_bytes,
                path=updated_path,
                provider_meta=meta,
            )
        )

    bundle = RecordBundle(
        conversation=conversation_record,
        messages=messages,
        attachments=attachments,
        content_blocks=content_block_records,
    )
    return TransformResult(
        bundle=bundle,
        materialization_plan=materialization_plan,
        content_hash=content_hash,
        candidate_cid=candidate_cid,
        message_id_map=message_id_map,
    )


def enrich_bundle_from_db(
    convo: ParsedConversation,
    source_name: str,
    transform: TransformResult,
    cache: PrepareCache,
    *,
    raw_id: str | None = None,
) -> EnrichedBundle:
    """Apply DB-derived lookups to a TransformResult, producing a save-ready EnrichedBundle.

    Uses the pre-loaded PrepareCache — no writes, no async I/O.
    Resolves: stable conversation/message IDs, parent FKs, change detection.
    """
    candidate_cid = transform.candidate_cid
    content_hash = transform.content_hash

    # Resolve conversation identity and change flag from cache
    existing = cache.existing.get(candidate_cid)
    if existing:
        cid: ConversationId = ConversationId(existing.conversation_id)
        changed = existing.content_hash != content_hash
    else:
        cid = candidate_cid
        changed = False

    # Resolve parent conversation ID FK
    parent_conversation_id = None
    if convo.parent_conversation_provider_id:
        candidate_parent = make_conversation_id(convo.provider_name, convo.parent_conversation_provider_id)
        if candidate_parent in cache.known_ids:
            parent_conversation_id = candidate_parent

    # Build stable message ID map (existing DB IDs take priority)
    existing_message_ids = cache.message_ids.get(cid, {})
    stable_message_id_map: dict[str, MessageId] = {}
    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        key = str(provider_message_id)
        stable_message_id_map[key] = existing_message_ids.get(key) or transform.message_id_map[key]

    # Patch the conversation record with enriched fields
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

    # Patch messages: stable IDs, corrected conversation_id and parent_message_id
    patched_messages: list[MessageRecord] = []
    for idx, (msg_rec, msg) in enumerate(zip(transform.bundle.messages, convo.messages, strict=True), start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        key = str(provider_message_id)
        mid = stable_message_id_map[key]

        parent_message_id: MessageId | None = None
        if msg.parent_message_provider_id:
            parent_message_id = stable_message_id_map.get(str(msg.parent_message_provider_id))

        patched_messages.append(
            MessageRecord(
                message_id=mid,
                conversation_id=cid,
                provider_message_id=msg_rec.provider_message_id,
                role=msg_rec.role,
                text=msg_rec.text,
                sort_key=msg_rec.sort_key,
                content_hash=msg_rec.content_hash,
                parent_message_id=parent_message_id,
                branch_index=msg_rec.branch_index,
                provider_name=msg_rec.provider_name,
                word_count=msg_rec.word_count,
                has_tool_use=msg_rec.has_tool_use,
                has_thinking=msg_rec.has_thinking,
            )
        )

    # Patch content blocks: stable message_id and conversation_id
    patched_blocks: list[ContentBlockRecord] = []
    for block_rec in transform.bundle.content_blocks:
        patched_blocks.append(block_rec)

    # Build reverse map: old_mid → stable_mid
    reverse_mid: dict[MessageId, MessageId] = {}
    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        key = str(provider_message_id)
        old = transform.message_id_map[key]
        new = stable_message_id_map[key]
        reverse_mid[old] = new

    patched_blocks = [
        ContentBlockRecord(
            block_id=ContentBlockRecord.make_id(reverse_mid.get(b.message_id, b.message_id), b.block_index),
            message_id=reverse_mid.get(b.message_id, b.message_id),
            conversation_id=cid,
            block_index=b.block_index,
            type=b.type,
            text=b.text,
            tool_name=b.tool_name,
            tool_id=b.tool_id,
            tool_input=b.tool_input,
            media_type=b.media_type,
            metadata=b.metadata,
            semantic_type=b.semantic_type,
        )
        for b in transform.bundle.content_blocks
    ]

    # Patch attachments: stable cid and message_id
    patched_attachments: list[AttachmentRecord] = []
    for att_rec in transform.bundle.attachments:
        att_message_id: MessageId | None = None
        if att_rec.message_id is not None:
            att_message_id = reverse_mid.get(att_rec.message_id, att_rec.message_id)
        patched_attachments.append(
            AttachmentRecord(
                attachment_id=att_rec.attachment_id,
                conversation_id=cid,
                message_id=att_message_id,
                mime_type=att_rec.mime_type,
                size_bytes=att_rec.size_bytes,
                path=att_rec.path,
                provider_meta=att_rec.provider_meta,
            )
        )

    enriched_bundle = RecordBundle(
        conversation=conversation_record,
        messages=patched_messages,
        attachments=patched_attachments,
        content_blocks=patched_blocks,
    )
    return EnrichedBundle(
        bundle=enriched_bundle,
        materialization_plan=transform.materialization_plan,
        cid=cid,
        changed=changed,
    )


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
    """Convert a ParsedConversation to storage records and persist them.

    Thin orchestration: delegates pure transformation to transform_to_records,
    DB-dependent enrichment to enrich_bundle_from_db, then saves.

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

    # Pure transform — no DB
    transform = transform_to_records(convo, source_name, archive_root=archive_root)

    # Build or supplement the cache for single-conversation use (no cache provided)
    if cache is None:
        cache = await _build_single_cache(backend, convo, transform.candidate_cid, transform.candidate_cid)

    # DB-enriched bundle — no writes
    enriched = enrich_bundle_from_db(convo, source_name, transform, cache, raw_id=raw_id)

    # Execute filesystem moves and save to DB
    applied_moves: list[tuple[Path, Path]] = []
    try:
        for source_path, target_path in enriched.materialization_plan.move_before_save:
            materialize_attachment_path(source_path, target_path)
            applied_moves.append((source_path, target_path))

        result = await save_bundle(enriched.bundle, repository=repository)
    except Exception:
        for source_path, target_path in reversed(applied_moves):
            if target_path.exists():
                move_attachment_to_archive(target_path, source_path)
        raise

    for duplicate_source in enriched.materialization_plan.delete_after_save:
        if duplicate_source.exists():
            duplicate_source.unlink()

    return (
        enriched.cid,
        {
            "conversations": result.conversations,
            "messages": result.messages,
            "attachments": result.attachments,
            "skipped_conversations": result.skipped_conversations,
            "skipped_messages": result.skipped_messages,
            "skipped_attachments": result.skipped_attachments,
        },
        enriched.changed,
    )


async def _build_single_cache(
    backend: SQLiteBackend,
    convo,
    candidate_cid: ConversationId,
    _unused: ConversationId,
) -> PrepareCache:
    """Build a PrepareCache for a single conversation without a pre-loaded batch cache."""
    cache = PrepareCache()

    async with backend.connection() as conn:
        cursor = await conn.execute(
            "SELECT conversation_id, content_hash FROM conversations WHERE conversation_id = ? LIMIT 1",
            (candidate_cid,),
        )
        row = await cursor.fetchone()
    if row:
        cid = row["conversation_id"]
        cache.existing[cid] = ExistingConversation(conversation_id=cid, content_hash=row["content_hash"])
        cache.known_ids.add(cid)

    if convo.parent_conversation_provider_id:
        from polylogue.pipeline.ids import conversation_id as make_conv_id

        candidate_parent = make_conv_id(convo.provider_name, convo.parent_conversation_provider_id)
        async with backend.connection() as conn:
            cursor = await conn.execute(
                "SELECT 1 FROM conversations WHERE conversation_id = ?",
                (candidate_parent,),
            )
            if await cursor.fetchone():
                cache.known_ids.add(candidate_parent)

    # Retrieve existing message IDs for this conversation
    existing_cid = candidate_cid if candidate_cid in cache.known_ids else None
    if existing_cid:
        async with backend.connection() as conn:
            cursor = await conn.execute(
                "SELECT provider_message_id, message_id FROM messages "
                "WHERE conversation_id = ? AND provider_message_id IS NOT NULL",
                (existing_cid,),
            )
            rows = await cursor.fetchall()
        cache.message_ids[existing_cid] = {
            str(r["provider_message_id"]): MessageId(r["message_id"])
            for r in rows
            if r["provider_message_id"]
        }

    return cache


__all__ = [
    "RecordBundle",
    "SaveResult",
    "PrepareCache",
    "TransformResult",
    "EnrichedBundle",
    "_timestamp_sort_key",
    "save_bundle",
    "transform_to_records",
    "enrich_bundle_from_db",
    "prepare_records",
]

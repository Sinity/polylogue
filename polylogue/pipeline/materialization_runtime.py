"""Shared conversation materialization helpers for prepare and ingest paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias

from polylogue.archive.message.paste_detection import detect_paste
from polylogue.archive.message.types import MessageType
from polylogue.lib.conversation.branch_type import BranchType
from polylogue.lib.json import JSONDocument, json_document
from polylogue.lib.json import dumps as json_dumps
from polylogue.lib.roles import Role
from polylogue.lib.viewport.viewports import ToolCategory, classify_tool
from polylogue.pipeline.ids import (
    attachment_content_id,
    conversation_content_hash,
    message_content_hash,
)
from polylogue.pipeline.ids import conversation_id as make_conversation_id
from polylogue.pipeline.ids import message_id as make_message_id
from polylogue.pipeline.prepare_transform_content import canonicalize_conversation_content
from polylogue.pipeline.semantic_metadata import ToolInputPayload, extract_tool_metadata
from polylogue.schemas.code_detection.detection import detect_language
from polylogue.sources.parsers.base import ParsedConversation
from polylogue.types import (
    AttachmentId,
    ContentBlockType,
    ContentHash,
    ConversationId,
    MessageId,
    Provider,
    SemanticBlockType,
)

ProviderMetadata: TypeAlias = dict[str, object]
BlockMetadata: TypeAlias = JSONDocument


@dataclass(frozen=True, slots=True)
class MaterializedContentBlock:
    block_id: str
    block_index: int
    type: ContentBlockType
    text: str | None
    tool_name: str | None
    tool_id: str | None
    tool_input_json: str | None
    media_type: str | None
    metadata_json: str | None
    semantic_type: SemanticBlockType | None


@dataclass(frozen=True, slots=True)
class MaterializedMessage:
    message_id: MessageId
    provider_message_id: str
    role: Role
    text: str | None
    sort_key: float | None
    content_hash: ContentHash
    parent_message_id: MessageId | None
    branch_index: int
    word_count: int
    has_tool_use: int
    has_thinking: int
    has_paste: int
    message_type: MessageType
    blocks: list[MaterializedContentBlock]


@dataclass(frozen=True, slots=True)
class MaterializedAttachment:
    attachment_id: AttachmentId
    message_id: MessageId | None
    mime_type: str | None
    size_bytes: int | None
    source_path: str | None
    path: str | None
    provider_meta: ProviderMetadata


@dataclass(frozen=True, slots=True)
class MaterializedConversationStats:
    message_count: int
    word_count: int
    tool_use_count: int
    thinking_count: int
    paste_count: int


@dataclass(frozen=True, slots=True)
class MaterializedConversation:
    conversation_id: ConversationId
    provider_name: Provider
    provider_conversation_id: str
    title: str | None
    created_at: str | None
    updated_at: str | None
    sort_key: float | None
    content_hash: ContentHash
    provider_meta: ProviderMetadata
    parent_conversation_id: ConversationId | None
    branch_type: BranchType | None
    messages: list[MaterializedMessage]
    attachments: list[MaterializedAttachment]
    stats: MaterializedConversationStats


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


def _merged_conversation_provider_meta(
    convo: ParsedConversation,
    *,
    source_name: str,
) -> ProviderMetadata:
    merged_provider_meta: ProviderMetadata = {"source": source_name}
    if convo.provider_meta:
        merged_provider_meta.update(convo.provider_meta)
    return merged_provider_meta


def _attachment_provider_meta(
    base_meta: ProviderMetadata | None,
    *,
    provider_attachment_id: str | None,
) -> ProviderMetadata:
    meta: ProviderMetadata = dict(base_meta or {})
    if provider_attachment_id:
        meta.setdefault("provider_id", provider_attachment_id)
    return meta


def _tool_input_payload(tool_input: Mapping[str, object] | JSONDocument | None) -> ToolInputPayload:
    if tool_input is None:
        return {}
    return json_document(tool_input if isinstance(tool_input, dict) else dict(tool_input))


def _block_metadata(metadata: Mapping[str, object] | JSONDocument | None) -> BlockMetadata | None:
    if metadata is None:
        return None
    return json_document(metadata if isinstance(metadata, dict) else dict(metadata))


def _build_message_ids(convo: ParsedConversation, conversation_id: ConversationId) -> dict[str, MessageId]:
    message_id_map: dict[str, MessageId] = {}
    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        message_id_map[str(provider_message_id)] = make_message_id(conversation_id, provider_message_id)
    return message_id_map


def _materialize_content_block(
    message_id: MessageId,
    block_index: int,
    block: ParsedContentBlockLike,
) -> MaterializedContentBlock:
    tool_input = _tool_input_payload(block.tool_input)
    tool_input_json = json_dumps(tool_input) if block.tool_input is not None else None
    semantic_type: SemanticBlockType | None = None
    semantic_metadata: BlockMetadata | None = _block_metadata(block.metadata)

    if block.type == "tool_use" and block.tool_name:
        category = classify_tool(block.tool_name, tool_input)
        semantic_type = None if category is ToolCategory.OTHER else SemanticBlockType.from_string(category.value)
        tool_meta = extract_tool_metadata(
            block.tool_name,
            tool_input,
        )
        if tool_meta is not None:
            base = semantic_metadata or {}
            base.update(tool_meta)
            semantic_metadata = base
    elif block.type == "thinking":
        semantic_type = SemanticBlockType.THINKING
    elif block.type == "code" and block.text and semantic_metadata is None:
        detected_lang = detect_language(block.text)
        if detected_lang:
            semantic_metadata = {"language": detected_lang}

    metadata_json = json_dumps(semantic_metadata) if semantic_metadata is not None else None

    from polylogue.storage.runtime import ContentBlockRecord

    return MaterializedContentBlock(
        block_id=ContentBlockRecord.make_id(message_id, block_index),
        block_index=block_index,
        type=block.type,
        text=block.text,
        tool_name=block.tool_name,
        tool_id=block.tool_id,
        tool_input_json=tool_input_json,
        media_type=block.media_type,
        metadata_json=metadata_json,
        semantic_type=semantic_type,
    )


def materialize_conversation(
    convo: ParsedConversation,
    *,
    source_name: str,
    archive_root: Path,
) -> MaterializedConversation:
    normalized_convo = canonicalize_conversation_content(convo)
    content_hash = conversation_content_hash(normalized_convo)
    conversation_id = make_conversation_id(
        normalized_convo.provider_name,
        normalized_convo.provider_conversation_id,
    )
    parent_conversation_id = (
        make_conversation_id(normalized_convo.provider_name, normalized_convo.parent_conversation_provider_id)
        if normalized_convo.parent_conversation_provider_id
        else None
    )
    provider_meta = _merged_conversation_provider_meta(
        normalized_convo,
        source_name=source_name,
    )
    message_id_map = _build_message_ids(normalized_convo, conversation_id)

    messages: list[MaterializedMessage] = []
    total_word_count = 0
    total_tool_use = 0
    total_thinking = 0
    total_paste = 0

    for idx, msg in enumerate(normalized_convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        message_id = message_id_map[str(provider_message_id)]
        parent_message_id = (
            message_id_map.get(str(msg.parent_message_provider_id)) if msg.parent_message_provider_id else None
        )
        block_types = {block.type for block in msg.content_blocks}
        message_type = msg.message_type
        word_count = len(msg.text.split()) if msg.text and msg.text.strip() else 0
        has_tool_use = 1 if (block_types & {"tool_use", "tool_result"}) or msg.role == "tool" else 0
        has_thinking = 1 if "thinking" in block_types else 0
        has_paste = detect_paste(msg.text)
        if message_type == MessageType.MESSAGE:
            if "thinking" in block_types:
                message_type = MessageType.THINKING
            elif "tool_result" in block_types or msg.role == "tool":
                message_type = MessageType.TOOL_RESULT
            elif "tool_use" in block_types:
                message_type = MessageType.TOOL_USE

        total_word_count += word_count
        total_tool_use += has_tool_use
        total_thinking += has_thinking
        total_paste += has_paste

        blocks = [
            _materialize_content_block(message_id, block_index, block)
            for block_index, block in enumerate(msg.content_blocks)
        ]

        messages.append(
            MaterializedMessage(
                message_id=message_id,
                provider_message_id=provider_message_id,
                role=msg.role,
                text=msg.text,
                sort_key=_timestamp_sort_key(msg.timestamp),
                content_hash=message_content_hash(msg, provider_message_id),
                parent_message_id=parent_message_id,
                branch_index=msg.branch_index,
                word_count=word_count,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                has_paste=has_paste,
                message_type=message_type,
                blocks=blocks,
            )
        )

    attachments: list[MaterializedAttachment] = []
    for attachment in normalized_convo.attachments:
        raw_attachment_id, updated_meta, updated_path = attachment_content_id(
            normalized_convo.provider_name,
            attachment,
            archive_root=archive_root,
        )
        attachments.append(
            MaterializedAttachment(
                attachment_id=AttachmentId(raw_attachment_id),
                message_id=(
                    message_id_map.get(attachment.message_provider_id or "") if attachment.message_provider_id else None
                ),
                mime_type=attachment.mime_type,
                size_bytes=attachment.size_bytes,
                source_path=attachment.path,
                path=updated_path,
                provider_meta=_attachment_provider_meta(
                    dict(updated_meta or {}),
                    provider_attachment_id=attachment.provider_attachment_id,
                ),
            )
        )

    return MaterializedConversation(
        conversation_id=conversation_id,
        provider_name=normalized_convo.provider_name,
        provider_conversation_id=normalized_convo.provider_conversation_id,
        title=normalized_convo.title,
        created_at=normalized_convo.created_at,
        updated_at=normalized_convo.updated_at,
        sort_key=_timestamp_sort_key(normalized_convo.updated_at),
        content_hash=content_hash,
        provider_meta=provider_meta,
        parent_conversation_id=parent_conversation_id,
        branch_type=normalized_convo.branch_type,
        messages=messages,
        attachments=attachments,
        stats=MaterializedConversationStats(
            message_count=len(messages),
            word_count=total_word_count,
            tool_use_count=total_tool_use,
            thinking_count=total_thinking,
            paste_count=total_paste,
        ),
    )


ParsedToolInput: TypeAlias = Mapping[str, object]
ParsedBlockMetadata: TypeAlias = dict[str, object]


class ParsedContentBlockLike(Protocol):
    type: ContentBlockType
    text: str | None
    tool_name: str | None
    tool_id: str | None
    tool_input: ParsedToolInput | None
    media_type: str | None
    metadata: ParsedBlockMetadata | None


__all__ = [
    "BlockMetadata",
    "MaterializedAttachment",
    "MaterializedContentBlock",
    "MaterializedConversation",
    "MaterializedConversationStats",
    "MaterializedMessage",
    "ProviderMetadata",
    "_timestamp_sort_key",
    "materialize_conversation",
]

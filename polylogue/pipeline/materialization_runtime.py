"""Shared session materialization helpers for prepare and ingest paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias

from polylogue.archive.message.artifacts import classify_text_message_type
from polylogue.archive.message.paste_detection import detect_paste
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.archive.session.branch_type import BranchType
from polylogue.archive.viewport.viewports import ToolCategory, classify_tool
from polylogue.core.json import JSONDocument, json_document
from polylogue.core.json import dumps as json_dumps
from polylogue.core.timestamps import canonical_timestamp_text, parse_timestamp
from polylogue.pipeline.ids import (
    attachment_content_id,
    message_content_hash,
    provider_event_id,
    session_content_hashes,
)
from polylogue.pipeline.ids import message_id as make_message_id
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.pipeline.prepare_transform_content import canonicalize_session_content
from polylogue.pipeline.semantic_metadata import ToolInputPayload, extract_tool_metadata
from polylogue.schemas.code_detection.detection import detect_language
from polylogue.sources.parsers.base import ParsedSession
from polylogue.types import (
    AttachmentId,
    ContentBlockType,
    ContentHash,
    MessageId,
    Provider,
    ProviderEventId,
    SemanticBlockType,
    SessionId,
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
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    model_name: str | None = None
    paste_boundary_state: str | None = None


@dataclass(frozen=True, slots=True)
class MaterializedAttachment:
    attachment_id: AttachmentId
    message_id: MessageId | None
    mime_type: str | None
    size_bytes: int | None
    source_path: str | None
    path: str | None
    provider_meta: ProviderMetadata
    provider_attachment_id: str | None = None
    provider_file_id: str | None = None
    provider_drive_id: str | None = None
    upload_origin: str | None = None


@dataclass(frozen=True, slots=True)
class MaterializedProviderEvent:
    event_id: ProviderEventId
    session_id: SessionId
    source_name: Provider
    event_index: int
    event_type: str
    timestamp: str | None
    sort_key: float | None
    payload: ProviderMetadata
    source_message_id: MessageId | None = None


@dataclass(frozen=True, slots=True)
class MaterializedSessionStats:
    message_count: int
    word_count: int
    tool_use_count: int
    thinking_count: int
    paste_count: int


@dataclass(frozen=True, slots=True)
class MaterializedSession:
    session_id: SessionId
    source_name: Provider
    provider_session_id: str
    title: str | None
    created_at: str | None
    updated_at: str | None
    sort_key: float | None
    content_hash: ContentHash
    provider_meta: ProviderMetadata
    parent_session_id: SessionId | None
    branch_type: BranchType | None
    messages: list[MaterializedMessage]
    attachments: list[MaterializedAttachment]
    provider_events: list[MaterializedProviderEvent]
    stats: MaterializedSessionStats
    working_directories_json: str | None = None
    git_branch: str | None = None
    git_repository_url: str | None = None


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
    parsed = parse_timestamp(ts)
    if parsed is not None:
        return parsed.timestamp()
    return None


def _merged_session_provider_meta(
    convo: ParsedSession,
    *,
    source_name: str,
) -> ProviderMetadata:
    merged_provider_meta: ProviderMetadata = {"source": source_name}
    if convo.provider_meta:
        merged_provider_meta.update(
            {key: value for key, value in convo.provider_meta.items() if key != "context_compactions"}
        )
    return merged_provider_meta


def _materialize_provider_events(
    convo: ParsedSession,
    *,
    session_id: SessionId,
    message_id_map: dict[str, MessageId],
) -> list[MaterializedProviderEvent]:
    events: list[MaterializedProviderEvent] = []
    for event_index, event in enumerate(convo.provider_events):
        source_message_id = (
            message_id_map.get(event.source_message_provider_id)
            if event.source_message_provider_id is not None
            else None
        )
        events.append(
            MaterializedProviderEvent(
                event_id=provider_event_id(session_id, event_index),
                session_id=session_id,
                source_name=convo.source_name,
                event_index=event_index,
                event_type=event.event_type,
                timestamp=event.timestamp,
                sort_key=_timestamp_sort_key(event.timestamp),
                payload=event.payload,
                source_message_id=source_message_id,
            )
        )
    return events


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


def _build_message_ids(convo: ParsedSession, session_id: SessionId) -> dict[str, MessageId]:
    message_id_map: dict[str, MessageId] = {}
    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        message_id_map[str(provider_message_id)] = make_message_id(session_id, provider_message_id)
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

    # #1240: media_type is no longer a dedicated column; preserve it under
    # the existing block-metadata JSON for image/document blocks so the
    # render/hash roundtrip stays lossless.
    if block.media_type:
        base = dict(semantic_metadata) if semantic_metadata else {}
        base.setdefault("media_type", block.media_type)
        semantic_metadata = base

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
        metadata_json=metadata_json,
        semantic_type=semantic_type,
    )


def materialize_session(
    convo: ParsedSession,
    *,
    source_name: str,
    archive_root: Path,
) -> MaterializedSession:
    normalized_convo = canonicalize_session_content(convo)
    content_hash, message_hashes = session_content_hashes(normalized_convo)
    session_id = make_session_id(
        normalized_convo.source_name,
        normalized_convo.provider_session_id,
    )
    parent_session_id = (
        make_session_id(normalized_convo.source_name, normalized_convo.parent_session_provider_id)
        if normalized_convo.parent_session_provider_id
        else None
    )
    provider_meta = _merged_session_provider_meta(
        normalized_convo,
        source_name=source_name,
    )
    message_id_map = _build_message_ids(normalized_convo, session_id)

    messages: list[MaterializedMessage] = []

    for idx, msg in enumerate(normalized_convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        message_id = message_id_map[str(provider_message_id)]
        parent_message_id = (
            message_id_map.get(str(msg.parent_message_provider_id)) if msg.parent_message_provider_id else None
        )
        has_tool_block = False
        has_tool_result_block = False
        has_thinking_block = False
        for block in msg.content_blocks:
            if block.type == "thinking":
                has_thinking_block = True
            elif block.type == "tool_use":
                has_tool_block = True
            elif block.type == "tool_result":
                has_tool_result_block = True
        message_type = msg.message_type
        # Aggregate session stats are rebuilt later, but these per-message
        # flags are part of the archive row itself and drive query filters.
        # word_count is the dialogue word count — counts words in msg.text
        # (the joined human-readable text). Tool-only messages get 0
        # naturally because msg.text is empty/None for those. Was previously
        # hard-coded to 0, leaving every messages row with word_count=0
        # across 2.4M rows; downstream analytics (cost, substantive ratio,
        # productivity rollups) saw a dead signal.
        word_count = len((msg.text or "").split())
        has_tool_use = int(
            has_tool_block
            or has_tool_result_block
            or msg.role == "tool"
            or message_type in {MessageType.TOOL_USE, MessageType.TOOL_RESULT}
        )
        has_thinking = int(has_thinking_block or message_type == MessageType.THINKING)
        # #1583: history.jsonl sidecar evidence overrides a False heuristic
        # so that user messages whose assembled text no longer carries the
        # ``[Pasted text #N]`` marker still resolve to has_paste=1 when the
        # Claude Code history sidecar recorded a paste.
        meta_paste_evidence = bool((msg.provider_meta or {}).get("claude_code_history_paste"))
        has_paste = 1 if (detect_paste(msg.text) or meta_paste_evidence) else 0
        # #1655: resolve paste boundary state from strongest available evidence.
        from polylogue.archive.message.paste_detection import resolve_paste_boundary_state

        paste_boundary_state = (
            resolve_paste_boundary_state(
                message_text=msg.text,
                history_has_paste=meta_paste_evidence,
                history_has_content=bool((msg.provider_meta or {}).get("claude_code_history_paste_content")),
                hook_has_paste=bool((msg.provider_meta or {}).get("claude_code_hook_paste")),
            )
            if has_paste
            else None
        )
        if message_type == MessageType.MESSAGE:
            if has_thinking_block:
                message_type = MessageType.THINKING
            elif has_tool_result_block or msg.role == "tool":
                message_type = MessageType.TOOL_RESULT
            elif has_tool_block:
                message_type = MessageType.TOOL_USE
        if message_type == MessageType.MESSAGE:
            classified = classify_text_message_type(msg.text)
            if classified is not None:
                message_type = classified

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
                content_hash=message_hashes.get(provider_message_id) or message_content_hash(msg, provider_message_id),
                parent_message_id=parent_message_id,
                branch_index=msg.branch_index,
                word_count=word_count,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
                has_paste=has_paste,
                paste_boundary_state=paste_boundary_state,
                message_type=message_type,
                blocks=blocks,
                # Token counts flow through from parsers that populated them
                # (e.g., claude code's record.message.usage). Other parsers
                # leave the defaults of 0 until they're extended similarly.
                input_tokens=getattr(msg, "input_tokens", 0) or 0,
                output_tokens=getattr(msg, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(msg, "cache_read_tokens", 0) or 0,
                cache_write_tokens=getattr(msg, "cache_write_tokens", 0) or 0,
                model_name=getattr(msg, "model_name", None),
            )
        )

    attachments: list[MaterializedAttachment] = []
    for attachment in normalized_convo.attachments:
        raw_attachment_id, updated_meta, updated_path = attachment_content_id(
            normalized_convo.source_name,
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
                provider_attachment_id=attachment.provider_attachment_id,
                provider_file_id=attachment.provider_file_id,
                provider_drive_id=attachment.provider_drive_id,
                upload_origin=attachment.upload_origin,
            )
        )

    # Extract promoted provider_meta fields for canonical columns (#864).
    import json as _json

    working_directories_json: str | None = None
    git_branch: str | None = None
    git_repository_url: str | None = None
    if provider_meta:
        wds = provider_meta.get("working_directories")
        if isinstance(wds, list):
            working_directories_json = _json.dumps(wds)
        cwd = provider_meta.get("cwd")
        if isinstance(cwd, str) and working_directories_json is None:
            working_directories_json = _json.dumps([cwd])
        gb = provider_meta.get("gitBranch")
        if isinstance(gb, str):
            git_branch = gb
        git_obj = provider_meta.get("git")
        if isinstance(git_obj, dict):
            git_branch = git_branch or git_obj.get("branch")
            git_repository_url = git_obj.get("repository_url")

    return MaterializedSession(
        session_id=session_id,
        source_name=normalized_convo.source_name,
        provider_session_id=normalized_convo.provider_session_id,
        title=normalized_convo.title,
        created_at=canonical_timestamp_text(normalized_convo.created_at),
        updated_at=canonical_timestamp_text(normalized_convo.updated_at),
        sort_key=_timestamp_sort_key(normalized_convo.updated_at),
        content_hash=content_hash,
        provider_meta=provider_meta,
        parent_session_id=parent_session_id,
        branch_type=normalized_convo.branch_type,
        messages=messages,
        attachments=attachments,
        provider_events=_materialize_provider_events(
            normalized_convo,
            session_id=session_id,
            message_id_map=message_id_map,
        ),
        stats=MaterializedSessionStats(
            message_count=len(messages),
            word_count=0,
            tool_use_count=0,
            thinking_count=0,
            paste_count=0,
        ),
        working_directories_json=working_directories_json,
        git_branch=git_branch,
        git_repository_url=git_repository_url,
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
    "MaterializedSession",
    "MaterializedSessionStats",
    "MaterializedMessage",
    "MaterializedProviderEvent",
    "ProviderMetadata",
    "_timestamp_sort_key",
    "materialize_session",
]

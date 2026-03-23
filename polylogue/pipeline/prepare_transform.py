"""Pure raw-to-record transformation helpers for parsed conversations."""

from __future__ import annotations

from pathlib import Path

from polylogue.lib.json import dumps as json_dumps
from polylogue.lib.viewports import ToolCategory, classify_tool
from polylogue.pipeline.ids import (
    attachment_content_id,
    conversation_content_hash,
    message_content_hash,
)
from polylogue.pipeline.ids import conversation_id as make_conversation_id
from polylogue.pipeline.ids import message_id as make_message_id
from polylogue.pipeline.prepare_models import (
    AttachmentMaterializationPlan,
    RecordBundle,
    TransformResult,
    _timestamp_sort_key,
)
from polylogue.pipeline.semantic import extract_tool_metadata
from polylogue.schemas.code_detection import detect_language
from polylogue.schemas.unified import harmonize_parsed_message
from polylogue.sources.parsers.base import ParsedContentBlock
from polylogue.storage.store import (
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
)
from polylogue.types import AttachmentId, MessageId


def _plan_attachment_materialization(
    source_path: str | None,
    target_path: str | None,
) -> AttachmentMaterializationPlan:
    if not source_path or not target_path or source_path == target_path:
        return AttachmentMaterializationPlan()

    source = Path(source_path)
    target = Path(target_path)
    if not source.exists():
        return AttachmentMaterializationPlan()
    if target.exists():
        return AttachmentMaterializationPlan(delete_after_save=[source])
    return AttachmentMaterializationPlan(move_before_save=[(source, target)])


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
        return ParsedContentBlock(type="tool_result", text=block.text, tool_id=tool_id, metadata=metadata)
    if block.type.name == "CODE":
        if block.language:
            metadata = dict(metadata or {})
            metadata.setdefault("language", block.language)
        return ParsedContentBlock(type="code", text=block.text, metadata=metadata)
    if block.type.name == "THINKING":
        return ParsedContentBlock(type="thinking", text=block.text, metadata=metadata)
    if block.type.name == "IMAGE":
        return ParsedContentBlock(type="image", text=block.text, media_type=block.mime_type, metadata=metadata)
    if block.type.name in {"FILE", "AUDIO", "VIDEO"}:
        return ParsedContentBlock(type="document", text=block.text, media_type=block.mime_type, metadata=metadata)
    if block.type.name in {"TEXT", "SYSTEM", "ERROR", "UNKNOWN"}:
        return ParsedContentBlock(type="text", text=block.text, metadata=metadata)
    return None


def _canonicalize_message_content(provider_name: str, message) -> object:
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


def _canonicalize_conversation_content(convo) -> object:
    messages = [_canonicalize_message_content(str(convo.provider_name), message) for message in convo.messages]
    if all(original == updated for original, updated in zip(convo.messages, messages, strict=True)):
        return convo
    return convo.model_copy(update={"messages": messages})


def transform_to_records(convo, source_name: str, *, archive_root: Path) -> TransformResult:
    convo = _canonicalize_conversation_content(convo)
    content_hash = conversation_content_hash(convo)
    candidate_cid = make_conversation_id(convo.provider_name, convo.provider_conversation_id)

    merged_provider_meta: dict[str, object] = {"source": source_name}
    if convo.provider_meta:
        merged_provider_meta.update(convo.provider_meta)

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

    message_id_map: dict[str, MessageId] = {}
    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        message_id_map[str(provider_message_id)] = make_message_id(candidate_cid, provider_message_id)

    messages: list[MessageRecord] = []
    content_block_records: list[ContentBlockRecord] = []
    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        mid = message_id_map[str(provider_message_id)]
        message_hash = message_content_hash(msg, provider_message_id)
        parent_message_id: MessageId | None = None
        if msg.parent_message_provider_id:
            parent_message_id = message_id_map.get(str(msg.parent_message_provider_id))

        block_types = {blk.type for blk in msg.content_blocks}
        word_count = len(msg.text.split()) if msg.text and msg.text.strip() else 0
        has_tool_use = 1 if (block_types & {"tool_use", "tool_result"}) or msg.role == "tool" else 0
        has_thinking = 1 if "thinking" in block_types else 0
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
                word_count=word_count,
                has_tool_use=has_tool_use,
                has_thinking=has_thinking,
            )
        )

        for block_idx, block in enumerate(msg.content_blocks):
            tool_input_json = json_dumps(block.tool_input) if block.tool_input is not None else None
            semantic_type: str | None = None
            semantic_metadata: dict | None = block.metadata

            if block.type == "tool_use" and block.tool_name:
                category = classify_tool(block.tool_name, block.tool_input or {})
                semantic_type = None if category is ToolCategory.OTHER else category.value
                tool_meta = extract_tool_metadata(block.tool_name, block.tool_input or {})
                if tool_meta is not None:
                    base = dict(block.metadata) if isinstance(block.metadata, dict) else {}
                    base.update(tool_meta)
                    semantic_metadata = base
            elif block.type == "thinking":
                semantic_type = "thinking"
            elif block.type == "code" and block.text and semantic_metadata is None:
                detected_lang = detect_language(block.text)
                if detected_lang:
                    semantic_metadata = {"language": detected_lang}

            metadata_json = json_dumps(semantic_metadata) if semantic_metadata is not None else None
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


__all__ = [
    "transform_to_records",
]

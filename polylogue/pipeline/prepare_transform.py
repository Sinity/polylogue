"""Pure raw-to-record transformation helpers for parsed conversations."""

from __future__ import annotations

from pathlib import Path

from polylogue.pipeline.materialization_runtime import materialize_conversation
from polylogue.pipeline.prepare_models import (
    AttachmentMaterializationPlan,
    RecordBundle,
    TransformResult,
)
from polylogue.sources.parsers.base import ParsedConversation
from polylogue.storage.runtime import (
    PROVIDER_EVENT_MATERIALIZER_VERSION,
    AttachmentRecord,
    ContentBlockRecord,
    ConversationRecord,
    MessageRecord,
    ProviderEventRecord,
)
from polylogue.types import MessageId


def plan_attachment_materialization(
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


def transform_to_records(convo: ParsedConversation, source_name: str, *, archive_root: Path) -> TransformResult:
    materialized = materialize_conversation(
        convo,
        source_name=source_name,
        archive_root=archive_root,
    )

    conversation_record = ConversationRecord(
        conversation_id=materialized.conversation_id,
        provider_name=materialized.provider_name,
        provider_conversation_id=materialized.provider_conversation_id,
        title=materialized.title,
        created_at=materialized.created_at,
        updated_at=materialized.updated_at,
        sort_key=materialized.sort_key,
        content_hash=materialized.content_hash,
        provider_meta=materialized.provider_meta,
        parent_conversation_id=materialized.parent_conversation_id,
        branch_type=materialized.branch_type,
        raw_id=None,
    )

    messages: list[MessageRecord] = [
        MessageRecord(
            message_id=message.message_id,
            conversation_id=materialized.conversation_id,
            provider_message_id=message.provider_message_id,
            role=message.role,
            text=None if message.blocks else message.text,
            sort_key=message.sort_key,
            content_hash=message.content_hash,
            parent_message_id=message.parent_message_id,
            branch_index=message.branch_index,
            provider_name=materialized.provider_name,
            word_count=message.word_count,
            has_tool_use=message.has_tool_use,
            has_thinking=message.has_thinking,
            has_paste=message.has_paste,
            message_type=message.message_type,
        )
        for message in materialized.messages
    ]

    content_block_records: list[ContentBlockRecord] = []
    message_id_map: dict[str, MessageId] = {}
    for message in materialized.messages:
        message_id_map[message.provider_message_id] = message.message_id
        for block in message.blocks:
            content_block_records.append(
                ContentBlockRecord(
                    block_id=block.block_id,
                    message_id=message.message_id,
                    conversation_id=materialized.conversation_id,
                    block_index=block.block_index,
                    type=block.type,
                    text=block.text,
                    tool_name=block.tool_name,
                    tool_id=block.tool_id,
                    tool_input=block.tool_input_json,
                    metadata=block.metadata_json,
                    semantic_type=block.semantic_type,
                )
            )

    attachments: list[AttachmentRecord] = []
    materialization_plan = AttachmentMaterializationPlan()
    for attachment in materialized.attachments:
        attachment_plan = plan_attachment_materialization(attachment.source_path, attachment.path)
        materialization_plan.move_before_save.extend(attachment_plan.move_before_save)
        materialization_plan.delete_after_save.extend(attachment_plan.delete_after_save)
        attachments.append(
            AttachmentRecord(
                attachment_id=attachment.attachment_id,
                conversation_id=materialized.conversation_id,
                message_id=attachment.message_id,
                mime_type=attachment.mime_type,
                size_bytes=attachment.size_bytes,
                path=attachment.path,
                provider_meta=attachment.provider_meta,
                provider_attachment_id=attachment.provider_attachment_id,
                provider_file_id=attachment.provider_file_id,
                provider_drive_id=attachment.provider_drive_id,
                upload_origin=attachment.upload_origin,
            )
        )

    provider_events = [
        ProviderEventRecord(
            event_id=event.event_id,
            conversation_id=event.conversation_id,
            provider_name=str(event.provider_name),
            event_index=event.event_index,
            event_type=event.event_type,
            timestamp=event.timestamp,
            sort_key=event.sort_key,
            payload=event.payload,
            source_message_id=event.source_message_id,
            materializer_version=PROVIDER_EVENT_MATERIALIZER_VERSION,
        )
        for event in materialized.provider_events
    ]

    bundle = RecordBundle(
        conversation=conversation_record,
        messages=messages,
        attachments=attachments,
        content_blocks=content_block_records,
        provider_events=provider_events,
    )
    return TransformResult(
        bundle=bundle,
        materialization_plan=materialization_plan,
        content_hash=materialized.content_hash,
        candidate_cid=materialized.conversation_id,
        message_id_map=message_id_map,
    )


__all__ = [
    "transform_to_records",
]

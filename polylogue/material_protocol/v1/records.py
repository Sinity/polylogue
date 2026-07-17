"""Record-id formulas and record-payload builders for material protocol v1.

Each ``*_record`` function returns a plain JSON-compatible dict carrying
``kind`` and ``record_id`` but *not* ``seq`` -- the encoder assigns ``seq``
once it has settled the full ordered record list for a revision (see
``encode.py``). Id formulas mirror the ``index.db`` generated columns.
"""

from __future__ import annotations

from polylogue.core.json import JSONValue
from polylogue.material_protocol.v1.input_model import (
    AttachmentInput,
    BlockInput,
    LineageInput,
    MessageInput,
    SessionEventInput,
    SessionMaterial,
    UsageInput,
)


def message_native_component(message: MessageInput) -> str:
    """The COALESCE(native_id, position||'.'||variant_index) component."""
    return message.native_id if message.native_id is not None else f"{message.position}.{message.variant_index}"


def message_id_for(session_id: str, message: MessageInput) -> str:
    return f"{session_id}:{message_native_component(message)}"


def block_id_for(message_id: str, block: BlockInput) -> str:
    return f"{message_id}:{block.position}"


def event_id_for(session_id: str, position: int) -> str:
    return f"{session_id}:{position}"


def attachment_ref_id_for(message_id: str, attachment: AttachmentInput) -> str:
    return f"{message_id}:attachment:{attachment.position}"


def lineage_record_id_for(session_id: str, lineage: LineageInput) -> str:
    return f"{session_id}:lineage:{lineage.dst_origin.value}:{lineage.dst_native_id}:{lineage.link_type.value}"


def usage_record_id_for(session_id: str, usage: UsageInput) -> str:
    return f"{session_id}:usage:{usage.model_name}"


def session_record(material: SessionMaterial) -> dict[str, JSONValue]:
    session_id = material.session_id
    return {
        "kind": "session",
        "record_id": session_id,
        "session_id": session_id,
        "origin": material.origin.value,
        "native_id": material.native_id,
        "title": material.title,
        "session_kind": material.session_kind.value,
        "created_at_ms": material.created_at_ms,
        "updated_at_ms": material.updated_at_ms,
        "git_branch": material.git_branch,
        "git_repository_url": material.git_repository_url,
        "provider_project_ref": material.provider_project_ref,
        "working_directories": list(material.working_directories),
        "metadata": dict(material.metadata),
        "tags": list(material.tags),
        "message_count": len(material.messages),
    }


def lineage_record(session_id: str, lineage: LineageInput) -> dict[str, JSONValue]:
    record_id = lineage_record_id_for(session_id, lineage)
    return {
        "kind": "lineage",
        "record_id": record_id,
        "src_session_id": session_id,
        "dst_origin": lineage.dst_origin.value,
        "dst_native_id": lineage.dst_native_id,
        "link_type": lineage.link_type.value,
        "branch_point_message_native_id": lineage.branch_point_message_native_id,
        "inheritance": lineage.inheritance,
        "status": lineage.status,
        "confidence": lineage.confidence,
        "observed_at_ms": lineage.observed_at_ms,
    }


def usage_record(session_id: str, usage: UsageInput) -> dict[str, JSONValue]:
    record_id = usage_record_id_for(session_id, usage)
    return {
        "kind": "usage",
        "record_id": record_id,
        "session_id": session_id,
        "model_name": usage.model_name,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "cost_usd": usage.cost_usd,
        "cost_credits": usage.cost_credits,
        "cost_provenance": usage.cost_provenance,
    }


def message_record(session_id: str, message: MessageInput) -> dict[str, JSONValue]:
    message_id = message_id_for(session_id, message)
    parent_message_id = f"{session_id}:{message.parent_native_id}" if message.parent_native_id is not None else None
    return {
        "kind": "message",
        "record_id": message_id,
        "session_id": session_id,
        "message_id": message_id,
        "native_id": message.native_id,
        "position": message.position,
        "variant_index": message.variant_index,
        "role": message.role.value,
        "message_type": message.message_type.value,
        "material_origin": message.material_origin.value,
        "text": message.text,
        "occurred_at_ms": message.occurred_at_ms,
        "model_name": message.model_name,
        "parent_message_id": parent_message_id,
        "usage": {
            "input_tokens": message.input_tokens,
            "output_tokens": message.output_tokens,
            "cache_read_tokens": message.cache_read_tokens,
            "cache_write_tokens": message.cache_write_tokens,
            "duration_ms": message.duration_ms,
        },
        "block_count": len(message.blocks),
    }


def block_record(session_id: str, message_id: str, block: BlockInput) -> dict[str, JSONValue]:
    block_id = block_id_for(message_id, block)
    return {
        "kind": "block",
        "record_id": block_id,
        "session_id": session_id,
        "message_id": message_id,
        "block_id": block_id,
        "position": block.position,
        "block_type": block.block_type.value,
        "text": block.text,
        "tool_name": block.tool_name,
        "tool_id": block.tool_id,
        "tool_input": block.tool_input,
        "tool_result_is_error": block.tool_result_is_error,
        "tool_result_exit_code": block.tool_result_exit_code,
        "semantic_type": block.semantic_type,
        "media_type": block.media_type,
        "language": block.language,
    }


def attachment_record(session_id: str, message_id: str, attachment: AttachmentInput) -> dict[str, JSONValue]:
    ref_id = attachment_ref_id_for(message_id, attachment)
    return {
        "kind": "attachment",
        "record_id": ref_id,
        "session_id": session_id,
        "message_id": message_id,
        "position": attachment.position,
        "attachment_id": attachment.attachment_id,
        "display_name": attachment.display_name,
        "media_type": attachment.media_type,
        "byte_count": attachment.byte_count,
        "blob_sha256": attachment.blob_sha256,
        "acquisition_status": attachment.acquisition_status,
        "upload_origin": attachment.upload_origin,
        "source_url": attachment.source_url,
        "caption": attachment.caption,
    }


def session_event_record(
    session_id: str, event: SessionEventInput, *, source_message_id: str | None
) -> dict[str, JSONValue]:
    event_id = event_id_for(session_id, event.position)
    return {
        "kind": "session_event",
        "record_id": event_id,
        "session_id": session_id,
        "position": event.position,
        "event_type": event.event_type,
        "summary": event.summary,
        "payload": dict(event.payload),
        "source_message_id": source_message_id,
        "occurred_at_ms": event.occurred_at_ms,
    }


__all__ = [
    "attachment_record",
    "attachment_ref_id_for",
    "block_id_for",
    "block_record",
    "event_id_for",
    "lineage_record",
    "lineage_record_id_for",
    "message_id_for",
    "message_native_component",
    "message_record",
    "session_event_record",
    "session_record",
    "usage_record",
    "usage_record_id_for",
]

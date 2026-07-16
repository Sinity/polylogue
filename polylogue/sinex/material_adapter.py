"""Adapt accepted Polylogue session material to material-protocol v1.

The production adapter consumes the parser's full ``ParsedSession`` before the
batch releases it.  Every material-protocol unit available at that boundary is
encoded: session, message, block, attachment, lineage, usage, and session
event.  Normalized fields that v1 cannot represent are named in fidelity gaps
rather than silently discarded.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum

from polylogue.archive.models import Session
from polylogue.core.enums import (
    BlockType,
    LinkType,
    MaterialOrigin,
    MessageType,
    Origin,
    Role,
    SessionKind,
)
from polylogue.core.json import JSONValue
from polylogue.core.web_urls import native_id_from_session_id
from polylogue.material_protocol.v1 import (
    AttachmentInput,
    BlockInput,
    FidelityGapInput,
    LineageInput,
    MaterialProtocolError,
    MessageInput,
    SessionEventInput,
    SessionMaterial,
    UsageInput,
    decode_session_revision,
    encode_session_revision,
    verify_revision,
)
from polylogue.material_protocol.v1.canonical import canonical_bytes
from polylogue.sinex.models import PublicationPayload
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
)

_PROTOCOL_VERSION = "polylogue.material-protocol/v1"


class PublicationEncodingError(RuntimeError):
    """Accepted normalized material could not be reconciled to exact wire bytes."""


class PublicationBackpressureError(PublicationEncodingError):
    """Exact publication bytes exceeded the bounded ingest staging budget."""


def _attr(value: object, *names: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _items(value: object) -> tuple[object, ...]:
    """Return a bounded sequence view without treating scalars as records."""
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Iterable):
        return ()
    return tuple(value)


def _int(value: object, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _timestamp_ms(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(normalized.timestamp() * 1000)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return int(parsed.timestamp() * 1000)
    return None


def _revision_created_at(value: object) -> str:
    milliseconds = _timestamp_ms(value)
    if milliseconds is None:
        return "1970-01-01T00:00:00+00:00"
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC).isoformat()


def _json_safe(value: object) -> JSONValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        safe_items = [_json_safe(item) for item in value]
        return sorted(safe_items, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump(mode="json"))
    return str(value)


def _json_object(value: object) -> dict[str, JSONValue]:
    safe = _json_safe(value)
    if not isinstance(safe, dict):
        return {"value": safe}
    return safe


def _origin_for_session_id(session_id: str) -> Origin:
    prefix, separator, _native_id = session_id.partition(":")
    if not separator:
        raise ValueError(f"session_id {session_id!r} is not a well-formed 'origin:native_id' session id")
    try:
        return Origin(prefix)
    except ValueError as exc:
        raise PublicationEncodingError(f"session_id {session_id!r} uses an unknown Origin token") from exc


def _role(value: object) -> Role:
    if isinstance(value, Role):
        return value
    return Role.normalize(str(value) if value is not None else "unknown")


def _message_type(value: object) -> MessageType:
    return MessageType.normalize(value)


def _material_origin(value: object) -> MaterialOrigin:
    return MaterialOrigin.normalize(value)


def _session_kind(value: object) -> SessionKind:
    return SessionKind.normalize(value)


def _link_type(value: object) -> LinkType:
    if isinstance(value, LinkType):
        return value
    candidate = str(value).strip().lower() if value is not None else LinkType.BRANCH.value
    try:
        return LinkType(candidate)
    except ValueError:
        return LinkType.BRANCH


def _block_input(position: int, block: object) -> BlockInput | None:
    raw_type = _attr(block, "type", "block_type")
    if raw_type is None:
        return None
    try:
        block_type = raw_type if isinstance(raw_type, BlockType) else BlockType.from_string(str(raw_type))
    except ValueError:
        return None
    tool_input = _attr(block, "tool_input", "input")
    text = _attr(block, "text")
    tool_name = _attr(block, "tool_name", "name")
    tool_id = _attr(block, "tool_id")
    is_error = _attr(block, "tool_result_is_error", "is_error")
    exit_code = _attr(block, "tool_result_exit_code", "exit_code")
    metadata = _attr(block, "metadata", default={})
    semantic_type = _attr(block, "semantic_type")
    if semantic_type is None and isinstance(metadata, Mapping):
        semantic_type = metadata.get("semantic_type")
    media_type = _attr(block, "media_type", "mime_type")
    language = _attr(block, "language")
    if language is None and isinstance(metadata, Mapping):
        language = metadata.get("language")
    return BlockInput(
        position=position,
        block_type=block_type,
        text=text if isinstance(text, str) else None,
        tool_name=tool_name if isinstance(tool_name, str) else None,
        tool_id=tool_id if isinstance(tool_id, str) else None,
        tool_input=_json_object(tool_input) if isinstance(tool_input, Mapping) else None,
        tool_result_is_error=is_error if isinstance(is_error, bool) else None,
        tool_result_exit_code=(exit_code if isinstance(exit_code, int) and not isinstance(exit_code, bool) else None),
        semantic_type=semantic_type if isinstance(semantic_type, str) else None,
        media_type=media_type if isinstance(media_type, str) else None,
        language=language if isinstance(language, str) else None,
    )


def _dropped_block_gap(
    session_id: str,
    message_index: int,
    block_position: int,
    block: object,
) -> FidelityGapInput:
    return FidelityGapInput(
        scope="block",
        record_id=f"{session_id}:message[{message_index}]:block[{block_position}]",
        gap_kind="dropped_block",
        detail=f"block type {_attr(block, 'type', 'block_type')!r} is missing or unsupported",
    )


def _block_fidelity_gap(
    session_id: str,
    message_index: int,
    block_position: int,
    block: object,
) -> FidelityGapInput | None:
    unsupported: list[str] = []
    metadata = _attr(block, "metadata", default={})
    if isinstance(metadata, Mapping):
        represented_metadata = {"language", "semantic_type"}
        unsupported.extend(f"metadata.{key}" for key in sorted(set(map(str, metadata)) - represented_metadata))
    if _items(_attr(block, "web_constructs", default=())):
        unsupported.append("web_constructs")
    if not unsupported:
        return None
    return FidelityGapInput(
        scope="block",
        record_id=f"{session_id}:message[{message_index}]:block[{block_position}]",
        gap_kind="unsupported_normalized_fields",
        detail="material-protocol v1 has no field for: " + ", ".join(unsupported),
    )


def _parsed_block_input(position: int, block: ParsedContentBlock) -> BlockInput:
    metadata = block.metadata or {}
    semantic_type = metadata.get("semantic_type")
    language = metadata.get("language")
    return BlockInput(
        position=position,
        block_type=block.type,
        text=block.text,
        tool_name=block.tool_name,
        tool_id=block.tool_id,
        tool_input=_json_object(block.tool_input) if block.tool_input is not None else None,
        tool_result_is_error=block.is_error,
        tool_result_exit_code=block.exit_code,
        semantic_type=semantic_type if isinstance(semantic_type, str) else None,
        media_type=block.media_type,
        language=language if isinstance(language, str) else None,
    )


def _parsed_block_fidelity_gap(
    session_id: str,
    message_index: int,
    block_position: int,
    block: ParsedContentBlock,
) -> FidelityGapInput | None:
    metadata = block.metadata or {}
    represented_metadata = {"language", "semantic_type"}
    unsupported = [f"metadata.{key}" for key in sorted(set(metadata) - represented_metadata)]
    if block.web_constructs:
        unsupported.append("web_constructs")
    if not unsupported:
        return None
    return FidelityGapInput(
        scope="block",
        record_id=f"{session_id}:message[{message_index}]:block[{block_position}]",
        gap_kind="unsupported_normalized_fields",
        detail="material-protocol v1 has no field for: " + ", ".join(unsupported),
    )


def _parsed_attachment_input(position: int, attachment: ParsedAttachment) -> AttachmentInput:
    blob_sha = hashlib.sha256(attachment.inline_bytes).hexdigest() if attachment.inline_bytes is not None else None
    byte_count = attachment.size_bytes
    if byte_count is None and attachment.inline_bytes is not None:
        byte_count = len(attachment.inline_bytes)
    return AttachmentInput(
        position=position,
        attachment_id=attachment.provider_attachment_id,
        display_name=attachment.name,
        media_type=attachment.mime_type,
        byte_count=max(0, byte_count or 0),
        blob_sha256=blob_sha,
        acquisition_status="acquired" if attachment.inline_bytes is not None else "unfetched",
        upload_origin=attachment.upload_origin,
        source_url=attachment.source_url,
        caption=attachment.caption,
    )


def _parsed_attachment_fidelity_gap(
    session_id: str,
    attachment: ParsedAttachment,
) -> FidelityGapInput | None:
    unsupported = [
        field
        for field, value in (
            ("path", attachment.path),
            ("provider_file_id", attachment.provider_file_id),
            ("provider_drive_id", attachment.provider_drive_id),
            ("attachment_kind", attachment.attachment_kind),
        )
        if value not in (None, "")
    ]
    if not unsupported:
        return None
    return FidelityGapInput(
        scope="attachment",
        record_id=f"{session_id}:attachment:{attachment.provider_attachment_id}",
        gap_kind="unsupported_normalized_fields",
        detail="material-protocol v1 has no field for: " + ", ".join(unsupported),
    )


def _message_anchor(attachment: object) -> tuple[str, object] | None:
    provider_id = _attr(
        attachment,
        "message_provider_id",
        "source_message_provider_id",
        "provider_message_id",
        "message_native_id",
    )
    if provider_id:
        return ("native", str(provider_id))
    message_position = _attr(attachment, "message_position", "message_index")
    if isinstance(message_position, int) and not isinstance(message_position, bool):
        return ("position", message_position)
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    return None


def _usage_inputs(messages: Sequence[ParsedMessage], session: ParsedSession) -> tuple[UsageInput, ...]:
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    )
    for message in messages:
        model_name = message.model_name
        token_values = {
            "input": message.input_tokens,
            "output": message.output_tokens,
            "cache_read": message.cache_read_tokens,
            "cache_write": message.cache_write_tokens,
        }
        has_tokens = any(value != 0 for value in token_values.values())
        if model_name is None and not has_tokens:
            continue
        key = str(model_name or "unknown")
        for name, value in token_values.items():
            totals[key][name] += max(0, value)
    cost_usd = _number(session.reported_cost_usd)
    if not totals and cost_usd is not None:
        totals["unknown"]
    usages: list[UsageInput] = []
    for index, (model_name, values) in enumerate(sorted(totals.items())):
        usages.append(
            UsageInput(
                model_name=model_name,
                input_tokens=values["input"],
                output_tokens=values["output"],
                cache_read_tokens=values["cache_read"],
                cache_write_tokens=values["cache_write"],
                cost_usd=cost_usd if index == 0 else None,
                cost_provenance="reported" if index == 0 and cost_usd is not None else None,
            )
        )
    return tuple(usages)


def _session_metadata(parsed_session: ParsedSession) -> dict[str, JSONValue]:
    values: dict[str, object] = {
        "active_leaf_message_provider_id": parsed_session.active_leaf_message_provider_id,
        "branch_type": parsed_session.branch_type,
        "git_commit_hash": parsed_session.git_commit_hash,
        "instructions_text": parsed_session.instructions_text,
        "models_used": parsed_session.models_used,
        "reported_duration_ms": parsed_session.reported_duration_ms,
        "source_name": parsed_session.source_name,
        "title_source": parsed_session.title_source,
    }
    metadata: dict[str, JSONValue] = {}
    for field, value in values.items():
        if value not in (None, (), [], {}, ""):
            metadata[field] = _json_safe(value)
    return metadata


def _parsed_event_input(position: int, event: ParsedSessionEvent) -> SessionEventInput:
    payload = _json_object(event.payload)
    summary_value = payload.get("summary")
    summary = summary_value if isinstance(summary_value, str) else event.event_type
    return SessionEventInput(
        position=position,
        event_type=event.event_type,
        summary=summary,
        payload=payload,
        source_message_native_id=event.source_message_provider_id,
        occurred_at_ms=_timestamp_ms(event.timestamp),
    )


def session_material_from_parsed_session(parsed_session: ParsedSession, *, session_id: str) -> SessionMaterial:
    """Build complete v1 material from the real ingest ``ParsedSession``."""
    native_id = native_id_from_session_id(session_id)
    if native_id is None:
        raise ValueError(f"session_id {session_id!r} is not a well-formed 'origin:native_id' session id")
    origin = _origin_for_session_id(session_id)
    raw_messages = parsed_session.messages
    attachments_by_message: dict[str, list[ParsedAttachment]] = defaultdict(list)
    unanchored_attachments: list[ParsedAttachment] = []
    for attachment in parsed_session.attachments:
        if attachment.message_provider_id is None:
            unanchored_attachments.append(attachment)
        else:
            attachments_by_message[attachment.message_provider_id].append(attachment)

    fidelity_gaps: list[FidelityGapInput] = []
    messages: list[MessageInput] = []
    for index, message in enumerate(raw_messages):
        position = message.position if message.position is not None else index
        native_message_id = message.provider_message_id
        blocks = [_parsed_block_input(block_index, block) for block_index, block in enumerate(message.blocks)]
        for block_index, block in enumerate(message.blocks):
            gap = _parsed_block_fidelity_gap(session_id, index, block_index, block)
            if gap is not None:
                fidelity_gaps.append(gap)
        anchored = attachments_by_message.pop(native_message_id, [])
        attachment_inputs: list[AttachmentInput] = []
        for attachment_position, attachment in enumerate(anchored):
            attachment_inputs.append(_parsed_attachment_input(attachment_position, attachment))
            gap = _parsed_attachment_fidelity_gap(session_id, attachment)
            if gap is not None:
                fidelity_gaps.append(gap)
        message_fields = {
            "delivery_status": message.delivery_status,
            "end_turn": message.end_turn,
            "is_active_leaf": message.is_active_leaf,
            "is_active_path": message.is_active_path,
            "model_effort": message.model_effort,
            "recipient": message.recipient,
            "sender_name": message.sender_name,
            "user_context_text": message.user_context_text,
        }
        unsupported_message_fields = [field for field, value in message_fields.items() if value not in (None, "")]
        if message.branch_index != 0:
            unsupported_message_fields.append("branch_index")
        if message.paste_spans:
            unsupported_message_fields.append("paste_spans")
        if unsupported_message_fields:
            fidelity_gaps.append(
                FidelityGapInput(
                    scope="message",
                    record_id=f"{session_id}:message[{index}]",
                    gap_kind="unsupported_normalized_fields",
                    detail="material-protocol v1 has no field for: " + ", ".join(unsupported_message_fields),
                )
            )
        messages.append(
            MessageInput(
                native_id=native_message_id,
                position=position,
                role=message.role,
                text=message.text,
                variant_index=message.variant_index or 0,
                message_type=message.message_type,
                material_origin=message.material_origin,
                occurred_at_ms=message.occurred_at_ms or _timestamp_ms(message.timestamp),
                model_name=message.model_name,
                parent_native_id=message.parent_message_provider_id,
                input_tokens=message.input_tokens,
                output_tokens=message.output_tokens,
                cache_read_tokens=message.cache_read_tokens,
                cache_write_tokens=message.cache_write_tokens,
                duration_ms=message.duration_ms,
                blocks=tuple(blocks),
                attachments=tuple(attachment_inputs),
            )
        )

    for message_provider_id, orphaned in attachments_by_message.items():
        unanchored_attachments.extend(orphaned)
        fidelity_gaps.append(
            FidelityGapInput(
                scope="attachment",
                record_id=f"{session_id}:attachment-anchor:native:{message_provider_id}",
                gap_kind="unresolved_anchor",
                detail="attachment referenced a message anchor absent from the accepted session",
            )
        )
    for index, attachment in enumerate(unanchored_attachments):
        if attachment.message_provider_id is not None:
            continue
        fidelity_gaps.append(
            FidelityGapInput(
                scope="attachment",
                record_id=attachment.provider_attachment_id or f"{session_id}:attachment[{index}]",
                gap_kind="unresolved_anchor",
                detail="material-protocol v1 requires a message anchor; source attachment had none",
            )
        )

    lineage: list[LineageInput] = []
    parent_native_id = parsed_session.parent_session_provider_id
    if parent_native_id:
        lineage.append(
            LineageInput(
                dst_origin=origin,
                dst_native_id=str(parent_native_id),
                link_type=_link_type(parsed_session.branch_type),
                inheritance="prefix-sharing",
                status="unresolved",
                confidence=1.0,
                observed_at_ms=_timestamp_ms(parsed_session.updated_at or parsed_session.created_at),
            )
        )

    events = [_parsed_event_input(index, event) for index, event in enumerate(parsed_session.session_events)]

    tags = tuple(parsed_session.ingest_flags)
    return SessionMaterial(
        origin=origin,
        native_id=native_id,
        title=parsed_session.title,
        session_kind=parsed_session.session_kind,
        created_at_ms=_timestamp_ms(parsed_session.created_at),
        updated_at_ms=_timestamp_ms(parsed_session.updated_at),
        git_branch=parsed_session.git_branch,
        git_repository_url=parsed_session.git_repository_url,
        provider_project_ref=parsed_session.provider_project_ref,
        working_directories=tuple(parsed_session.working_directories),
        metadata=_session_metadata(parsed_session),
        tags=tags,
        messages=tuple(messages),
        lineage=tuple(lineage),
        usage=_usage_inputs(parsed_session.messages, parsed_session),
        session_events=tuple(events),
        fidelity_gaps=tuple(fidelity_gaps),
    )


def session_material_from_session(session: Session) -> SessionMaterial:
    """Build best-available material from a hydrated archive session tree."""
    native_id = native_id_from_session_id(session.id)
    if native_id is None:
        raise ValueError(f"session.id {session.id!r} is not a well-formed 'origin:native_id' session id")
    dropped_block_gaps: list[FidelityGapInput] = []
    messages: list[MessageInput] = []
    for index, message in enumerate(session.messages):
        blocks: list[BlockInput] = []
        for block_position, raw_block in enumerate(message.blocks):
            block = _block_input(block_position, raw_block)
            if block is None:
                dropped_block_gaps.append(_dropped_block_gap(session.id, index, block_position, raw_block))
            else:
                blocks.append(block)
        messages.append(
            MessageInput(
                native_id=None,
                position=index,
                role=message.role,
                text=message.text,
                message_type=message.message_type,
                material_origin=message.material_origin,
                occurred_at_ms=_timestamp_ms(message.timestamp),
                model_name=message.model_name,
                input_tokens=message.input_tokens,
                output_tokens=message.output_tokens,
                cache_read_tokens=message.cache_read_tokens,
                cache_write_tokens=message.cache_write_tokens,
                duration_ms=message.duration_ms or None,
                blocks=tuple(blocks),
            )
        )
    return SessionMaterial(
        origin=session.origin,
        native_id=native_id,
        title=session.title,
        session_kind=session.session_kind,
        created_at_ms=_timestamp_ms(session.created_at),
        updated_at_ms=_timestamp_ms(session.updated_at),
        git_branch=session.git_branch,
        git_repository_url=session.git_repository_url,
        provider_project_ref=session.provider_project_ref,
        working_directories=session.working_directories,
        metadata={key: _json_safe(value) for key, value in session.metadata.items()},
        tags=session.tags_m2m,
        messages=tuple(messages),
        lineage=(),
        usage=(),
        session_events=(),
        fidelity_gaps=(
            FidelityGapInput(
                scope="session",
                record_id=session.id,
                gap_kind="omitted_relation",
                detail="hydrated Session does not include attachment/lineage/usage/event repository relations",
            ),
            *dropped_block_gaps,
        ),
    )


def _minimum_text_bytes(value: object, *, stop_after: int) -> int:
    """Count definitely-emitted text without constructing serialized protocol bytes."""
    total = 0
    stack = [value]
    while stack and total <= stop_after:
        item = stack.pop()
        if item is None or isinstance(item, (bool, int, float, bytes, Enum, Decimal)):
            continue
        if isinstance(item, str):
            total += len(item)
            continue
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            stack.extend(getattr(item, field.name) for field in dataclasses.fields(item))
            continue
        if isinstance(item, Mapping):
            stack.extend(str(key) for key in item)
            stack.extend(item.values())
            continue
        if isinstance(item, Iterable):
            stack.extend(item)
    return total


def encode_parsed_session_publication(
    parsed_session: ParsedSession,
    *,
    session_id: str,
    max_payload_bytes: int | None = None,
) -> PublicationPayload:
    """Encode, verify, and return the exact bytes staged by production ingest."""
    try:
        material = session_material_from_parsed_session(parsed_session, session_id=session_id)
        if max_payload_bytes is not None and (
            max_payload_bytes < 0 or _minimum_text_bytes(material, stop_after=max_payload_bytes) > max_payload_bytes
        ):
            raise PublicationBackpressureError(
                f"Sinex exact-payload staging budget exceeded before encoding: limit_bytes={max_payload_bytes}"
            )
        revision_time = parsed_session.updated_at or parsed_session.created_at
        encoded = encode_session_revision(
            material,
            revision_created_at=_revision_created_at(revision_time),
        )
        manifest = encoded.manifest
        raw_segments = encoded.segments
        names = encoded.segment_filenames()
        # Run the protocol's byte/digest/count/anchor/vocabulary/semantic closure
        # verifier before these bytes become durable publication evidence.
        verify_revision(manifest, raw_segments)
        decoded = decode_session_revision(manifest, raw_segments)
        decoded_session_id = decoded.session.get("session_id")
        if decoded_session_id != session_id:
            raise PublicationEncodingError(
                f"encoded revision session_id mismatch: expected={session_id!r} actual={decoded_session_id!r}"
            )
        # This exactly matches material_protocol.v1.io.write_revision.
        manifest_bytes = canonical_bytes(manifest.to_dict()) + b"\n"
        revision_id = manifest.revision_id
        protocol_version = manifest.protocol_version
        segments = tuple((names[index], bytes(raw_segments[index])) for index in sorted(raw_segments))
    except PublicationEncodingError:
        raise
    except (MaterialProtocolError, TypeError, ValueError, OverflowError, AssertionError) as exc:
        raise PublicationEncodingError(f"material protocol reconciliation failed: {type(exc).__name__}: {exc}") from exc
    payload = PublicationPayload(
        object_id=session_id,
        protocol_version=str(protocol_version),
        revision_id=str(revision_id),
        manifest_digest=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest_bytes=manifest_bytes,
        segments=segments,
    )
    if max_payload_bytes is not None and payload.size_bytes > max_payload_bytes:
        raise PublicationBackpressureError(
            "Sinex exact-payload staging budget exceeded after encoding: "
            f"payload_bytes={payload.size_bytes} limit_bytes={max_payload_bytes}"
        )
    return payload


__all__ = [
    "PublicationBackpressureError",
    "PublicationEncodingError",
    "encode_parsed_session_publication",
    "session_material_from_parsed_session",
    "session_material_from_session",
]

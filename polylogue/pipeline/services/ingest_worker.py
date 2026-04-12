"""Unified subprocess worker: decode → validate → parse → transform in one pass.

Runs inside ProcessPoolExecutor. Returns plain tuples for direct SQL executemany,
avoiding Pydantic serialization overhead across the process boundary.

Performance: eliminates double blob decode (was: validate decodes, then parse decodes
the same blob again). Moves transform into subprocess for true parallelism.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
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
from polylogue.pipeline.prepare_transform_content import canonicalize_conversation_content
from polylogue.pipeline.semantic_metadata import extract_tool_metadata
from polylogue.schemas.code_detection import detect_language
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.store import RawConversationRecord, _json_or_none
from polylogue.types import ValidationMode, ValidationStatus

_SOURCE_HASH_SUFFIX = re.compile(r"-(?:[0-9a-f]{16,64})$", re.IGNORECASE)
_SCHEMA_REGISTRY = None


# ---------------------------------------------------------------------------
# Result dataclasses — cheap to pickle (no Pydantic)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ConversationData:
    """All DB-ready data for one conversation, as plain tuples for executemany."""

    conversation_id: str
    content_hash: str
    provider_name: str

    # Tuple matching INSERT INTO conversations column order
    conversation_tuple: tuple

    # list[tuple] matching INSERT INTO messages column order
    message_tuples: list[tuple] = field(default_factory=list)

    # list[tuple] matching INSERT INTO content_blocks column order
    block_tuples: list[tuple] = field(default_factory=list)

    # list[tuple] matching INSERT INTO action_events column order
    action_event_tuples: list[tuple] = field(default_factory=list)

    # (conversation_id, provider_name, msg_count, word_count, tool_use_count, thinking_count)
    stats_tuple: tuple = ()

    # Attachments are rare; keep as list of simple tuples
    # Each: (attachment_id, conversation_id, message_id, mime_type, size_bytes, path, provider_meta_json)
    attachment_tuples: list[tuple] = field(default_factory=list)
    attachment_ref_tuples: list[tuple] = field(default_factory=list)

    # Source metadata
    source_name: str = ""
    raw_id: str | None = None


@dataclass(slots=True)
class IngestRecordResult:
    """Result from processing one raw record in a subprocess."""

    raw_id: str
    payload_provider: str | None = None
    validation_status: str = "skipped"  # ValidationStatus value
    validation_error: str | None = None
    error: str | None = None
    conversations: list[ConversationData] = field(default_factory=list)
    source_name: str | None = None
    serialized_size_bytes: int | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fallback_id(source_path: str | None, raw_id: str) -> str:
    if not source_path:
        return raw_id
    normalized = source_path.replace("\\", "/")
    entry_path = normalized.rsplit(":", 1)[-1]
    stem = Path(entry_path).stem
    if not stem:
        return raw_id
    cleaned = _SOURCE_HASH_SUFFIX.sub("", stem).strip("._- ")
    return cleaned or stem


def _timestamp_sort_key(ts: str | None) -> float | None:
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


def _make_ref_id(
    attachment_id: str,
    conversation_id: str,
    message_id: str | None,
) -> str:
    from hashlib import sha256

    key = f"{attachment_id}:{conversation_id}:{message_id or ''}"
    return sha256(key.encode()).hexdigest()[:32]


def _runtime_schema_registry():
    global _SCHEMA_REGISTRY
    if _SCHEMA_REGISTRY is None:
        from polylogue.schemas.runtime_registry import SchemaRegistry

        _SCHEMA_REGISTRY = SchemaRegistry()
    return _SCHEMA_REGISTRY


def _finalize_result(result: IngestRecordResult, *, measure_serialized_size: bool) -> IngestRecordResult:
    if not measure_serialized_size:
        return result
    result.serialized_size_bytes = len(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
    return result


# ---------------------------------------------------------------------------
# Main worker function — runs in subprocess
# ---------------------------------------------------------------------------


def ingest_record(
    raw_record: RawConversationRecord,
    archive_root_str: str,
    validation_mode_value: str = "strict",
    measure_serialized_size: bool = False,
    *,
    blob_root_str: str | None = None,
) -> IngestRecordResult:
    """Decode + validate + parse + transform one raw record in a single pass.

    Returns DB-ready tuples, not Pydantic models. This function runs in a
    subprocess via ProcessPoolExecutor and must be self-contained (no shared
    state, no DB access).
    """
    from polylogue.lib.raw_payload import build_raw_payload_envelope
    from polylogue.paths import blob_store_root
    from polylogue.schemas.validator import SchemaValidator
    from polylogue.sources.dispatch import parse_payload
    archive_root = Path(archive_root_str)
    validation_mode = ValidationMode.from_string(validation_mode_value)

    stored_payload_provider = raw_record.payload_provider
    if not isinstance(stored_payload_provider, str) or not stored_payload_provider.strip():
        stored_payload_provider = None

    resolved_blob_root = Path(blob_root_str) if blob_root_str is not None else blob_store_root()
    blob_store = BlobStore(resolved_blob_root)
    raw_source = blob_store.blob_path(raw_record.raw_id)

    # ── Phase 1: Decode blob (ONE decode, not two) ────────────────────
    try:
        envelope = build_raw_payload_envelope(
            raw_source,
            source_path=raw_record.source_path,
            fallback_provider=raw_record.provider_name,
            payload_provider=stored_payload_provider,
        )
    except Exception as exc:
        return _finalize_result(
            IngestRecordResult(
                raw_id=raw_record.raw_id,
                payload_provider=stored_payload_provider,
                validation_status=ValidationStatus.FAILED.value,
                validation_error=f"decode: {exc}",
                error=f"decode: {exc}",
            ),
            measure_serialized_size=measure_serialized_size,
        )

    payload_provider = str(envelope.provider)

    if not envelope.artifact.parse_as_conversation:
        return _finalize_result(
            IngestRecordResult(
                raw_id=raw_record.raw_id,
                payload_provider=payload_provider,
                validation_status=ValidationStatus.SKIPPED.value,
            ),
            measure_serialized_size=measure_serialized_size,
        )

    # ── Phase 2: Validate schema (inline, reuses decoded payload) ─────
    v_status = ValidationStatus.PASSED
    v_error: str | None = None
    schema_resolution = None

    if validation_mode is not ValidationMode.OFF and envelope.artifact.schema_eligible:
        malformed_lines = envelope.malformed_jsonl_lines
        if malformed_lines and validation_mode is ValidationMode.STRICT:
            return _finalize_result(
                IngestRecordResult(
                    raw_id=raw_record.raw_id,
                    payload_provider=payload_provider,
                    validation_status=ValidationStatus.FAILED.value,
                    validation_error=f"Malformed JSONL lines: {malformed_lines}",
                    error=f"Malformed JSONL lines: {malformed_lines}",
                ),
                measure_serialized_size=measure_serialized_size,
            )

        schema_resolution = _runtime_schema_registry().resolve_payload(
            envelope.provider,
            envelope.payload,
            source_path=raw_record.source_path,
        )

        try:
            validator = SchemaValidator.for_payload(
                envelope.provider,
                envelope.payload,
                source_path=raw_record.source_path,
                schema_resolution=schema_resolution,
            )
        except (FileNotFoundError, ImportError):
            validator = None
            v_status = ValidationStatus.SKIPPED

        if validator is not None:
            samples = validator.validation_samples(envelope.payload)
            if samples:
                collected_errors: list[str] = []
                for sample in samples:
                    sample_result = validator.validate(sample, include_drift=False)
                    if not sample_result.is_valid:
                        collected_errors.extend(sample_result.errors[:2])

                if collected_errors and validation_mode is ValidationMode.STRICT:
                    first_error = collected_errors[0]
                    return _finalize_result(
                        IngestRecordResult(
                            raw_id=raw_record.raw_id,
                            payload_provider=payload_provider,
                            validation_status=ValidationStatus.FAILED.value,
                            validation_error=f"Schema validation failed: {first_error}",
                            error=f"Schema validation failed: {first_error}",
                        ),
                        measure_serialized_size=measure_serialized_size,
                    )
    elif validation_mode is ValidationMode.OFF:
        v_status = ValidationStatus.SKIPPED

    # ── Phase 3: Parse (provider-specific conversation extraction) ─────
    if schema_resolution is None and envelope.artifact.schema_eligible:
        schema_resolution = _runtime_schema_registry().resolve_payload(
            envelope.provider,
            envelope.payload,
            source_path=raw_record.source_path,
        )

    try:
        parsed_conversations = parse_payload(
            envelope.provider,
            envelope.payload,
            _fallback_id(raw_record.source_path, raw_record.raw_id),
            schema_resolution=schema_resolution,
        )
    except Exception as exc:
        return _finalize_result(
            IngestRecordResult(
                raw_id=raw_record.raw_id,
                payload_provider=payload_provider,
                validation_status=v_status.value,
                error=f"parse: {exc}",
            ),
            measure_serialized_size=measure_serialized_size,
        )

    # The decoded payload can be much larger than the normalized conversation
    # objects; drop it before tuple transformation to avoid overlapping heaps.
    del envelope
    schema_resolution = None

    # Apply raw record defaults (timestamps) and transform immediately.
    fallback_timestamp = raw_record.file_mtime
    source_name = raw_record.source_name or raw_record.source_path or ""
    result_convos: list[ConversationData] = []
    for convo in parsed_conversations:
        updates: dict[str, object] = {}
        if convo.created_at is None and fallback_timestamp:
            updates["created_at"] = fallback_timestamp
        effective_created = updates.get("created_at", convo.created_at)
        if convo.updated_at is None and isinstance(effective_created, str) and effective_created:
            updates["updated_at"] = effective_created
        normalized_convo = convo.model_copy(update=updates) if updates else convo
        try:
            cdata = _transform_to_tuples(
                normalized_convo,
                source_name=source_name,
                archive_root=archive_root,
                raw_id=raw_record.raw_id,
            )
            result_convos.append(cdata)
        except Exception as exc:
            return _finalize_result(
                IngestRecordResult(
                    raw_id=raw_record.raw_id,
                    payload_provider=payload_provider,
                    validation_status=v_status.value,
                    error=f"transform: {exc}",
                ),
                measure_serialized_size=measure_serialized_size,
            )
    del parsed_conversations

    return _finalize_result(
        IngestRecordResult(
            raw_id=raw_record.raw_id,
            payload_provider=payload_provider,
            validation_status=v_status.value,
            validation_error=v_error,
            conversations=result_convos,
            source_name=source_name,
        ),
        measure_serialized_size=measure_serialized_size,
    )


# ---------------------------------------------------------------------------
# Transform — converts ParsedConversation to plain tuples
# ---------------------------------------------------------------------------


def _transform_to_tuples(
    convo,
    *,
    source_name: str,
    archive_root: Path,
    raw_id: str | None,
) -> ConversationData:
    """Convert a ParsedConversation to DB-ready tuples.

    Equivalent to transform_to_records + enrich_bundle_from_db but produces
    plain tuples instead of Pydantic models. Skips PrepareCache entirely
    since conversation_id and message_id are deterministic hashes.
    """
    convo = canonicalize_conversation_content(convo)
    content_hash = conversation_content_hash(convo)
    cid = make_conversation_id(convo.provider_name, convo.provider_conversation_id)

    merged_provider_meta: dict[str, object] = {"source": source_name}
    if convo.provider_meta:
        merged_provider_meta.update(convo.provider_meta)

    # Parent conversation — always compute it (no PrepareCache needed).
    # If parent doesn't exist in DB yet, that's fine — no FK constraint.
    parent_conversation_id = None
    if convo.parent_conversation_provider_id:
        parent_conversation_id = make_conversation_id(
            convo.provider_name,
            convo.parent_conversation_provider_id,
        )

    sort_key = _timestamp_sort_key(convo.updated_at)

    # Conversation tuple: matches INSERT INTO conversations column order
    conv_tuple = (
        cid,  # conversation_id
        convo.provider_name,  # provider_name
        convo.provider_conversation_id,  # provider_conversation_id
        convo.title,  # title
        convo.created_at,  # created_at
        convo.updated_at,  # updated_at
        sort_key,  # sort_key
        content_hash,  # content_hash
        _json_or_none(merged_provider_meta),  # provider_meta
        "{}",  # metadata
        1,  # version
        parent_conversation_id,  # parent_conversation_id
        convo.branch_type,  # branch_type
        raw_id,  # raw_id
    )

    # Build message ID map
    message_id_map: dict[str, str] = {}
    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        message_id_map[str(provider_message_id)] = make_message_id(cid, provider_message_id)

    # Message tuples + content block tuples
    msg_tuples: list[tuple] = []
    block_tuples: list[tuple] = []
    total_word_count = 0
    total_tool_use = 0
    total_thinking = 0

    for idx, msg in enumerate(convo.messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        mid = message_id_map[str(provider_message_id)]
        message_hash = message_content_hash(msg, provider_message_id)

        parent_message_id: str | None = None
        if msg.parent_message_provider_id:
            parent_message_id = message_id_map.get(str(msg.parent_message_provider_id))

        block_types = {blk.type for blk in msg.content_blocks}
        word_count = len(msg.text.split()) if msg.text and msg.text.strip() else 0
        has_tool_use = 1 if (block_types & {"tool_use", "tool_result"}) or msg.role == "tool" else 0
        has_thinking = 1 if "thinking" in block_types else 0

        total_word_count += word_count
        total_tool_use += has_tool_use
        total_thinking += has_thinking

        msg_tuples.append(
            (
                mid,  # message_id
                cid,  # conversation_id
                provider_message_id,  # provider_message_id
                msg.role,  # role
                msg.text,  # text
                _timestamp_sort_key(msg.timestamp),  # sort_key
                message_hash,  # content_hash
                1,  # version
                parent_message_id,  # parent_message_id
                msg.branch_index,  # branch_index
                convo.provider_name,  # provider_name
                word_count,  # word_count
                has_tool_use,  # has_tool_use
                has_thinking,  # has_thinking
            )
        )

        # Content blocks for this message
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

            from polylogue.storage.store import ContentBlockRecord

            block_id = ContentBlockRecord.make_id(mid, block_idx)
            block_tuples.append(
                (
                    block_id,  # block_id
                    mid,  # message_id
                    cid,  # conversation_id
                    block_idx,  # block_index
                    block.type,  # type
                    block.text,  # text
                    block.tool_name,  # tool_name
                    block.tool_id,  # tool_id
                    tool_input_json,  # tool_input
                    block.media_type,  # media_type
                    metadata_json,  # metadata
                    semantic_type,  # semantic_type
                )
            )

    # Conversation stats tuple
    stats_tuple = (
        cid,
        convo.provider_name,
        len(convo.messages),
        total_word_count,
        total_tool_use,
        total_thinking,
    )

    # Action events — build from content blocks
    action_event_tuples = _build_action_event_tuples(
        cid,
        convo.provider_name,
        convo.messages,
        message_id_map,
        msg_tuples,
    )

    # Attachments
    attachment_tuples: list[tuple] = []
    attachment_ref_tuples: list[tuple] = []
    for att in convo.attachments:
        aid, updated_meta, updated_path = attachment_content_id(
            convo.provider_name,
            att,
            archive_root=archive_root,
        )
        meta: dict[str, object] = dict(updated_meta or {})
        if att.provider_attachment_id:
            meta.setdefault("provider_id", att.provider_attachment_id)
        message_id_val = message_id_map.get(att.message_provider_id or "") if att.message_provider_id else None
        meta_json = _json_or_none(meta)

        attachment_tuples.append(
            (
                aid,  # attachment_id
                att.mime_type,  # mime_type
                att.size_bytes,  # size_bytes
                updated_path,  # path
                0,  # ref_count (updated after ref insert)
                meta_json,  # provider_meta
            )
        )
        ref_id = _make_ref_id(aid, cid, message_id_val)
        attachment_ref_tuples.append(
            (
                ref_id,  # ref_id
                aid,  # attachment_id
                cid,  # conversation_id
                message_id_val,  # message_id
                meta_json,  # provider_meta
            )
        )

    return ConversationData(
        conversation_id=cid,
        content_hash=content_hash,
        provider_name=convo.provider_name,
        conversation_tuple=conv_tuple,
        message_tuples=msg_tuples,
        block_tuples=block_tuples,
        action_event_tuples=action_event_tuples,
        stats_tuple=stats_tuple,
        attachment_tuples=attachment_tuples,
        attachment_ref_tuples=attachment_ref_tuples,
        source_name=source_name,
        raw_id=raw_id,
    )


def _build_action_event_tuples(
    conversation_id: str,
    provider_name: str,
    messages: list,
    message_id_map: dict[str, str],
    msg_tuples: list[tuple],
) -> list[tuple]:
    """Build action event tuples for all messages in a conversation.

    Uses the lightweight action event builder that works from content blocks
    without needing full Pydantic MessageRecord hydration.
    """
    from polylogue.lib.action_events import build_action_events, build_tool_calls_from_content_blocks
    from polylogue.storage.hydrators import message_from_record
    from polylogue.storage.store import (
        ACTION_EVENT_MATERIALIZER_VERSION,
        ContentBlockRecord,
        MessageRecord,
    )
    from polylogue.types import MessageId, Provider

    provider = Provider.from_string(provider_name)
    action_tuples: list[tuple] = []

    for idx, msg in enumerate(messages, start=1):
        provider_message_id = msg.provider_message_id or f"msg-{idx}"
        mid = message_id_map[str(provider_message_id)]
        msg_tuple = msg_tuples[idx - 1]
        msg_sort_key = msg_tuple[5]  # sort_key is at index 5

        # Build lightweight domain message for action event extraction
        block_types = {blk.type for blk in msg.content_blocks}
        has_tool_use = 1 if (block_types & {"tool_use", "tool_result"}) or msg.role == "tool" else 0
        if not has_tool_use:
            continue  # Skip messages without tool use — no action events

        # Reconstruct minimal MessageRecord for the hydrator
        word_count = len(msg.text.split()) if msg.text and msg.text.strip() else 0
        has_thinking = 1 if "thinking" in block_types else 0
        msg_record = MessageRecord(
            message_id=MessageId(mid),
            conversation_id=conversation_id,
            provider_message_id=provider_message_id,
            role=msg.role,
            text=msg.text,
            sort_key=msg_sort_key,
            content_hash=msg_tuple[6],
            provider_name=provider_name,
            word_count=word_count,
            has_tool_use=has_tool_use,
            has_thinking=has_thinking,
            content_blocks=[
                ContentBlockRecord(
                    block_id=ContentBlockRecord.make_id(mid, bi),
                    message_id=MessageId(mid),
                    conversation_id=conversation_id,
                    block_index=bi,
                    type=blk.type,
                    text=blk.text,
                    tool_name=blk.tool_name,
                    tool_id=blk.tool_id,
                    tool_input=json_dumps(blk.tool_input) if blk.tool_input is not None else None,
                    media_type=blk.media_type,
                    metadata=json_dumps(blk.metadata) if blk.metadata is not None else None,
                    semantic_type=None,
                )
                for bi, blk in enumerate(msg.content_blocks)
            ],
        )

        domain_message = message_from_record(msg_record, attachments=[], provider=provider)
        tool_calls = build_tool_calls_from_content_blocks(
            provider=provider,
            content_blocks=domain_message.content_blocks,
        )
        for event in build_action_events(domain_message, tool_calls):
            from datetime import timezone

            timestamp_iso = None
            if event.timestamp is not None:
                ts = event.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                timestamp_iso = ts.isoformat()

            affected_paths_json = json_dumps(list(event.affected_paths)) if event.affected_paths else None
            branch_names_json = json_dumps(list(event.branch_names)) if event.branch_names else None

            action_tuples.append(
                (
                    event.event_id,  # event_id
                    conversation_id,  # conversation_id
                    mid,  # message_id
                    ACTION_EVENT_MATERIALIZER_VERSION,  # materializer_version
                    event.raw.get("block_id") if isinstance(event.raw, dict) else None,  # source_block_id
                    timestamp_iso,  # timestamp
                    msg_sort_key,  # sort_key
                    event.sequence_index,  # sequence_index
                    provider_name,  # provider_name
                    event.kind.value,  # action_kind
                    event.tool_name,  # tool_name
                    event.normalized_tool_name,  # normalized_tool_name
                    event.tool_id,  # tool_id
                    affected_paths_json,  # affected_paths_json
                    event.cwd_path,  # cwd_path
                    branch_names_json,  # branch_names_json
                    event.command,  # command
                    event.query,  # query_text
                    event.url,  # url
                    event.output_text,  # output_text
                    event.search_text,  # search_text
                )
            )

    return action_tuples


__all__ = ["ConversationData", "IngestRecordResult", "ingest_record"]

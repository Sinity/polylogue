"""Pure semantic transcript construction over already-hydrated evidence."""

from __future__ import annotations

import difflib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from typing import Protocol

from polylogue.core.enums import BlockType
from polylogue.core.json import JSONDocument
from polylogue.core.tool_identity import parse_mcp_tool_name
from polylogue.rendering.block_models import RenderableBlock, coerce_renderable_blocks
from polylogue.rendering.semantic_card_models import (
    CardOutcomeState,
    LineageAuthority,
    LineageAvailability,
    LineageDescriptor,
    PreviewStrategy,
    SemanticCard,
    SemanticCardField,
    SemanticCardKind,
    SemanticCardOutcome,
    SemanticCardPreview,
    SemanticCardRawEvidence,
    SemanticCardSource,
    SemanticNoticeKind,
    SemanticTranscript,
    SemanticTranscriptEntry,
    TranscriptNotice,
    TranscriptNoticeSource,
    TranscriptProse,
)
from polylogue.rendering.semantic_card_registry import ToolClassification, classify_tool, normalize_provider_family

DEFAULT_PREVIEW_HEAD_LINES = 48
DEFAULT_PREVIEW_TAIL_LINES = 16
DEFAULT_PREVIEW_MAX_CHARS = 16_000

_THINKING_BLOCK_TYPES = {BlockType.THINKING.value, BlockType.REASONING.value}
_ATTACHMENT_BLOCK_TYPES = {BlockType.IMAGE.value, BlockType.DOCUMENT.value, "file"}
_TOOL_BLOCK_TYPES = {BlockType.TOOL_USE.value, BlockType.TOOL_RESULT.value}


class _TopologyEdgeLike(Protocol):
    child_id: object
    parent_id: object
    parent_native_id: str | None
    kind: object
    resolved: bool


class _SessionTopologyLike(Protocol):
    root_id: object
    edges: Sequence[_TopologyEdgeLike]
    cycle_detected: bool


@dataclass(frozen=True, slots=True)
class RenderableAttachment:
    """Provider-neutral attachment evidence retained by the renderer."""

    attachment_id: str | None = None
    name: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    path: str | None = None
    source_url: str | None = None
    caption: str | None = None
    upload_origin: str | None = None


@dataclass(frozen=True, slots=True)
class RenderableMessage:
    """Message projection needed by the pure renderer."""

    id: str
    role: str
    message_type: str
    text: str | None
    provider_family: str
    origin: str | None
    material_origin: str
    blocks: tuple[RenderableBlock, ...]
    attachments: tuple[RenderableAttachment, ...] = ()
    occurred_at: str | None = None
    duration_ms: int | None = None
    parent_id: str | None = None
    variant_index: int | None = None
    is_active_path: bool | None = None
    is_active_leaf: bool | None = None
    inherited_prefix: bool | None = None
    source_session_id: str | None = None


@dataclass(frozen=True, slots=True)
class _BlockCoordinate:
    message_index: int
    block_index: int


@dataclass(frozen=True, slots=True)
class _ResultMatch:
    coordinate: _BlockCoordinate
    message: RenderableMessage
    block: RenderableBlock


def coerce_renderable_message(value: Mapping[str, object] | object) -> RenderableMessage:
    """Normalize a domain/archive message or a message-shaped fixture."""

    if isinstance(value, Mapping):
        raw_id = value.get("id", value.get("message_id", ""))
        raw_role = value.get("role", "unknown")
        raw_message_type = value.get("message_type", "message")
        raw_text = value.get("text")
        raw_origin = value.get("origin")
        raw_provider = value.get("provider_family", value.get("provider", raw_origin or "unknown"))
        raw_material_origin = value.get("material_origin", "unknown")
        raw_blocks = value.get("blocks", value.get("content_blocks", ()))
        raw_attachments = value.get("attachments", ())
        raw_occurred_at = value.get("occurred_at", value.get("timestamp"))
        raw_duration = value.get("duration_ms")
        raw_parent_id = value.get("parent_id", value.get("parent_message_id"))
        raw_variant_index = value.get("variant_index", value.get("branch_index"))
        raw_active_path = value.get("is_active_path")
        raw_active_leaf = value.get("is_active_leaf")
        raw_inherited_prefix = value.get("inherited_prefix")
        raw_source_session_id = value.get("source_session_id")
    else:
        raw_id = getattr(value, "id", getattr(value, "message_id", ""))
        raw_role = getattr(value, "role", "unknown")
        raw_message_type = getattr(value, "message_type", "message")
        raw_text = getattr(value, "text", None)
        raw_origin = getattr(value, "origin", None)
        raw_provider = getattr(value, "provider_family", getattr(value, "provider", raw_origin or "unknown"))
        raw_material_origin = getattr(value, "material_origin", "unknown")
        raw_blocks = getattr(value, "blocks", getattr(value, "content_blocks", ()))
        raw_attachments = getattr(value, "attachments", ())
        raw_occurred_at = getattr(value, "occurred_at", getattr(value, "timestamp", None))
        raw_duration = getattr(value, "duration_ms", None)
        raw_parent_id = getattr(value, "parent_id", getattr(value, "parent_message_id", None))
        raw_variant_index = getattr(value, "variant_index", getattr(value, "branch_index", None))
        raw_active_path = getattr(value, "is_active_path", None)
        raw_active_leaf = getattr(value, "is_active_leaf", None)
        raw_inherited_prefix = getattr(value, "inherited_prefix", None)
        raw_source_session_id = getattr(value, "source_session_id", None)

    raw_role = _enum_like_value(raw_role)
    raw_message_type = _enum_like_value(raw_message_type)
    raw_origin = _enum_like_value(raw_origin)
    raw_provider = _enum_like_value(raw_provider)
    raw_material_origin = _enum_like_value(raw_material_origin)

    blocks_input: Sequence[object] | None
    if isinstance(raw_blocks, Sequence) and not isinstance(raw_blocks, (str, bytes, bytearray)):
        blocks_input = raw_blocks
    else:
        blocks_input = None

    attachments_input: Sequence[object] | None
    if isinstance(raw_attachments, Sequence) and not isinstance(raw_attachments, (str, bytes, bytearray)):
        attachments_input = raw_attachments
    else:
        attachments_input = None

    text = _optional_text(raw_text)
    origin = str(raw_origin) if raw_origin is not None else None
    return RenderableMessage(
        id=str(raw_id),
        role=str(raw_role),
        message_type=str(raw_message_type),
        text=text,
        provider_family=normalize_provider_family(raw_provider),
        origin=origin,
        material_origin=str(raw_material_origin or "unknown"),
        blocks=coerce_renderable_blocks(blocks_input),
        attachments=_coerce_renderable_attachments(attachments_input),
        occurred_at=_iso_timestamp(raw_occurred_at),
        duration_ms=_optional_int(raw_duration),
        parent_id=str(raw_parent_id) if raw_parent_id is not None else None,
        variant_index=_optional_int(raw_variant_index),
        is_active_path=_optional_bool(raw_active_path),
        is_active_leaf=_optional_bool(raw_active_leaf),
        inherited_prefix=_optional_bool(raw_inherited_prefix),
        source_session_id=str(raw_source_session_id) if raw_source_session_id is not None else None,
    )


def _coerce_renderable_attachments(values: Sequence[object] | None) -> tuple[RenderableAttachment, ...]:
    if values is None:
        return ()
    attachments: list[RenderableAttachment] = []
    for value in values:
        if isinstance(value, Mapping):
            get = value.get
        else:

            def get(key: str, default: object | None = None, *, item: object = value) -> object | None:
                return getattr(item, key, default)

        attachment_id = get("id", get("attachment_id"))
        name = get("name", get("display_name"))
        mime_type = get("mime_type", get("media_type"))
        size_bytes = get("size_bytes", get("byte_count"))
        source_url = get("source_url", get("url"))
        attachments.append(
            RenderableAttachment(
                attachment_id=str(attachment_id) if attachment_id is not None else None,
                name=str(name) if name is not None else None,
                mime_type=str(mime_type) if mime_type is not None else None,
                size_bytes=_optional_int(size_bytes),
                path=str(get("path")) if get("path") is not None else None,
                source_url=str(source_url) if source_url is not None else None,
                caption=str(get("caption")) if get("caption") is not None else None,
                upload_origin=str(get("upload_origin")) if get("upload_origin") is not None else None,
            )
        )
    return tuple(attachments)


def _enum_like_value(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    candidate = getattr(value, "value", None)
    return candidate if isinstance(candidate, str) else value


def _optional_text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return None


def _envelope_attribute(envelope: object, name: str, default: object | None = None) -> object | None:
    """Read a bounded archive-envelope field without coupling to storage types."""

    return getattr(envelope, name, default)


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    return None


def _iso_timestamp(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        rendered = isoformat()
        return str(rendered) if rendered is not None else None
    return str(value)


def lineage_descriptor_from_session(session: object) -> LineageDescriptor:
    """Project one session row without hydrating a full topology family."""

    session_id = str(getattr(session, "id", getattr(session, "session_id", "")))
    parent = getattr(session, "parent_id", getattr(session, "parent_session_id", None))
    branch = _enum_like_value(getattr(session, "branch_type", None))
    origin_value = _enum_like_value(getattr(session, "origin", None))
    parent_id = str(parent) if parent is not None else None
    relation = str(branch) if branch is not None else "unknown"
    availability = (
        LineageAvailability.PARTIAL
        if parent_id is not None or relation != "unknown"
        else LineageAvailability.UNAVAILABLE
    )
    return LineageDescriptor(
        session_id=session_id,
        provider_family=normalize_provider_family(origin_value),
        origin=str(origin_value) if origin_value is not None else None,
        parent_session_id=parent_id,
        relation=relation,
        resolved=None,
        lineage_complete=None,
        authority=LineageAuthority.SESSION_ROW,
        availability=availability,
    )


def lineage_descriptor_from_archive_envelope(envelope: object) -> LineageDescriptor:
    """Project exact bounded lineage/composition authority from an archive read."""

    session_id = str(_envelope_attribute(envelope, "session_id"))
    parent = _envelope_attribute(envelope, "parent_session_id")
    stored_root = _envelope_attribute(envelope, "root_session_id")
    root = str(stored_root) if stored_root is not None else (session_id if parent is None else None)
    branch = _envelope_attribute(envelope, "branch_type")
    relation = str(branch) if branch is not None else ("root" if parent is None else "unknown")
    complete = _optional_bool(_envelope_attribute(envelope, "lineage_complete"))
    truncation = _envelope_attribute(envelope, "lineage_truncation_reason")
    truncation_value = str(_enum_like_value(truncation)) if truncation is not None else None
    inheritance = _envelope_attribute(envelope, "lineage_inheritance")
    inherited_prefix = None if inheritance is None else str(inheritance) == "prefix-sharing"
    if inheritance is None and parent is None:
        inherited_prefix = False
    availability = LineageAvailability.COMPLETE
    if complete is False or root is None:
        availability = LineageAvailability.PARTIAL
    origin_value = _enum_like_value(_envelope_attribute(envelope, "origin"))
    return LineageDescriptor(
        session_id=session_id,
        provider_family=normalize_provider_family(origin_value),
        origin=str(origin_value) if origin_value is not None else None,
        root_session_id=root,
        parent_session_id=str(parent) if parent is not None else None,
        relation=relation,
        resolved=True if parent is not None and inherited_prefix is True and complete is True else None,
        lineage_complete=complete,
        lineage_truncation_reason=truncation_value,
        inherited_prefix=inherited_prefix,
        branch_point_message_id=_optional_string(getattr(envelope, "lineage_branch_point_message_id", None)),
        active_leaf_message_id=_optional_string(getattr(envelope, "active_leaf_message_id", None)),
        authority=LineageAuthority.ARCHIVE_ENVELOPE,
        availability=availability,
    )


def lineage_descriptor_from_topology(
    topology: _SessionTopologyLike,
    *,
    session_id: str,
) -> LineageDescriptor | None:
    """Project an already-read topology into bounded lineage-card input."""

    target_edge = next((edge for edge in topology.edges if str(edge.child_id) == session_id), None)
    if target_edge is None:
        if str(topology.root_id) != session_id and not topology.cycle_detected:
            return None
        return LineageDescriptor(
            session_id=session_id,
            root_session_id=str(topology.root_id),
            relation="root" if str(topology.root_id) == session_id else "unknown",
            cycle_detected=topology.cycle_detected,
            authority=LineageAuthority.TOPOLOGY,
            availability=LineageAvailability.PARTIAL,
        )
    return LineageDescriptor(
        session_id=session_id,
        root_session_id=str(topology.root_id),
        parent_session_id=str(target_edge.parent_id) if target_edge.parent_id is not None else None,
        parent_native_id=target_edge.parent_native_id,
        relation=str(_enum_like_value(target_edge.kind)),
        resolved=target_edge.resolved,
        cycle_detected=topology.cycle_detected,
        authority=LineageAuthority.TOPOLOGY,
        availability=LineageAvailability.PARTIAL,
    )


def build_semantic_transcript(
    messages: Sequence[Mapping[str, object] | object],
    *,
    session_id: str,
    lineage: LineageDescriptor | None = None,
    provider_family: str | None = None,
) -> SemanticTranscript:
    """Build one ordered document for prose, thinking, tools, and attachments.

    Tool pairing is structural by ``tool_id`` and FIFO coordinates, independent
    of serialization order.  Empty typed thinking/reasoning blocks compact into
    one notice per contiguous run while retaining every exact source coordinate.
    No card kind or outcome is inferred from prose.
    """

    normalized: list[RenderableMessage] = []
    session_provider = normalize_provider_family(provider_family) if provider_family else None
    for raw in messages:
        message = coerce_renderable_message(raw)
        inherited_prefix = message.inherited_prefix
        if inherited_prefix is None and message.source_session_id is not None:
            inherited_prefix = message.source_session_id != session_id
        if session_provider and message.provider_family in {"", "unknown"}:
            message = replace(message, provider_family=session_provider)
        if inherited_prefix != message.inherited_prefix:
            message = replace(message, inherited_prefix=inherited_prefix)
        normalized.append(message)

    result_by_use, paired_results = _pair_tool_results(normalized)
    entries: list[SemanticTranscriptEntry] = []
    pending_empty_thinking: list[TranscriptNoticeSource] = []

    def flush_empty_thinking() -> None:
        if not pending_empty_thinking:
            return
        entries.append(
            SemanticTranscriptEntry(
                notice=TranscriptNotice(
                    kind=SemanticNoticeKind.EMPTY_THINKING,
                    sources=tuple(pending_empty_thinking),
                )
            )
        )
        pending_empty_thinking.clear()

    if lineage is not None:
        entries.append(SemanticTranscriptEntry(card=_build_lineage_card(lineage)))

    for message_index, message in enumerate(normalized):
        if not message.blocks:
            flush_empty_thinking()
            if message.text and message.text.strip():
                entries.append(SemanticTranscriptEntry(prose=_prose_for_message(message, message.text)))
            for attachment in message.attachments:
                entries.append(
                    SemanticTranscriptEntry(
                        card=_build_attachment_envelope_card(
                            session_id=session_id,
                            message=message,
                            attachment=attachment,
                        )
                    )
                )
            continue

        emitted_block_attachment_keys: set[tuple[object, ...]] = set()
        for block_index, block in enumerate(message.blocks):
            coordinate = _BlockCoordinate(message_index, block_index)
            if block.type in _THINKING_BLOCK_TYPES:
                if block.text is None or not block.text.strip():
                    pending_empty_thinking.append(_notice_source(message, block_index, block))
                else:
                    flush_empty_thinking()
                    entries.append(
                        SemanticTranscriptEntry(prose=_prose_for_block(message, block_index=block_index, block=block))
                    )
                continue

            # Any typed non-thinking row terminates an absence run, even when the
            # row is subsequently absorbed into a paired card.
            flush_empty_thinking()
            if block.type == BlockType.TOOL_USE.value:
                entries.append(
                    SemanticTranscriptEntry(
                        card=_build_tool_card(
                            session_id=session_id,
                            message=message,
                            block_index=block_index,
                            block=block,
                            result=result_by_use.get(coordinate),
                        )
                    )
                )
                continue
            if block.type == BlockType.TOOL_RESULT.value:
                if coordinate not in paired_results:
                    entries.append(
                        SemanticTranscriptEntry(
                            card=_build_orphan_result_card(
                                session_id=session_id,
                                message=message,
                                block_index=block_index,
                                block=block,
                            )
                        )
                    )
                continue
            if block.type in _ATTACHMENT_BLOCK_TYPES:
                entries.append(
                    SemanticTranscriptEntry(
                        card=_build_attachment_block_card(
                            session_id=session_id,
                            message=message,
                            block_index=block_index,
                            block=block,
                        )
                    )
                )
                emitted_block_attachment_keys.add(_block_attachment_identity(block))
                continue
            if block.text is not None and block.text != "":
                entries.append(
                    SemanticTranscriptEntry(prose=_prose_for_block(message, block_index=block_index, block=block))
                )

        if _should_emit_message_text(message):
            flush_empty_thinking()
            assert message.text is not None
            entries.append(SemanticTranscriptEntry(prose=_prose_for_message(message, message.text)))

        for attachment in message.attachments:
            if _attachment_matches_block(attachment, emitted_block_attachment_keys):
                continue
            flush_empty_thinking()
            entries.append(
                SemanticTranscriptEntry(
                    card=_build_attachment_envelope_card(
                        session_id=session_id,
                        message=message,
                        attachment=attachment,
                    )
                )
            )

    flush_empty_thinking()
    return SemanticTranscript(session_id=session_id, entries=tuple(entries))


def _pair_tool_results(
    messages: Sequence[RenderableMessage],
) -> tuple[dict[_BlockCoordinate, _ResultMatch], frozenset[_BlockCoordinate]]:
    uses: dict[str, list[_BlockCoordinate]] = defaultdict(list)
    results: dict[str, list[_ResultMatch]] = defaultdict(list)
    for message_index, message in enumerate(messages):
        for block_index, block in enumerate(message.blocks):
            coordinate = _BlockCoordinate(message_index, block_index)
            if block.type == BlockType.TOOL_USE.value and block.tool_id:
                uses[block.tool_id].append(coordinate)
            elif block.type == BlockType.TOOL_RESULT.value and block.tool_id:
                results[block.tool_id].append(_ResultMatch(coordinate, message, block))

    result_by_use: dict[_BlockCoordinate, _ResultMatch] = {}
    paired_results: set[_BlockCoordinate] = set()
    for tool_id, use_coordinates in uses.items():
        for use_coordinate, result in zip(use_coordinates, results.get(tool_id, ()), strict=False):
            result_by_use[use_coordinate] = result
            paired_results.add(result.coordinate)
    return result_by_use, frozenset(paired_results)


def _notice_source(
    message: RenderableMessage,
    block_index: int,
    block: RenderableBlock,
) -> TranscriptNoticeSource:
    return TranscriptNoticeSource(
        message_id=message.id,
        block_index=block_index,
        block_type=block.type,
        block_id=block.block_id,
        role=message.role,
        message_type=message.message_type,
        provider_family=message.provider_family,
        origin=message.origin,
        material_origin=message.material_origin,
        occurred_at=message.occurred_at,
        duration_ms=message.duration_ms,
        parent_message_id=message.parent_id,
        variant_index=message.variant_index,
        is_active_path=message.is_active_path,
        is_active_leaf=message.is_active_leaf,
        inherited_prefix=message.inherited_prefix,
    )


def _prose_for_message(message: RenderableMessage, text: str) -> TranscriptProse:
    return TranscriptProse(
        message_id=message.id,
        role=message.role,
        message_type=message.message_type,
        provider_family=message.provider_family,
        origin=message.origin,
        material_origin=message.material_origin,
        text=text,
        occurred_at=message.occurred_at,
        duration_ms=message.duration_ms,
        parent_message_id=message.parent_id,
        variant_index=message.variant_index,
        is_active_path=message.is_active_path,
        is_active_leaf=message.is_active_leaf,
        inherited_prefix=message.inherited_prefix,
    )


def _prose_for_block(
    message: RenderableMessage,
    *,
    block_index: int,
    block: RenderableBlock,
) -> TranscriptProse:
    assert block.text is not None
    return TranscriptProse(
        message_id=message.id,
        role=message.role,
        message_type=message.message_type,
        provider_family=message.provider_family,
        origin=message.origin,
        material_origin=message.material_origin,
        text=block.text,
        block_id=block.block_id,
        block_index=block_index,
        block_type=block.type,
        language=block.language,
        occurred_at=message.occurred_at,
        duration_ms=message.duration_ms,
        parent_message_id=message.parent_id,
        variant_index=message.variant_index,
        is_active_path=message.is_active_path,
        is_active_leaf=message.is_active_leaf,
        inherited_prefix=message.inherited_prefix,
    )


def _should_emit_message_text(message: RenderableMessage) -> bool:
    text = message.text
    if text is None or not text.strip():
        return False
    block_texts = [block.text for block in message.blocks if block.text is not None and block.text != ""]
    exact_aggregates = set(block_texts)
    if block_texts:
        exact_aggregates.add("\n".join(block_texts))
        exact_aggregates.add("\n\n".join(block_texts))
    if text in exact_aggregates:
        return False
    has_typed_tool_use = any(block.type == BlockType.TOOL_USE.value for block in message.blocks)
    # Recipient/tool envelopes sometimes retain serialized invocation JSON in
    # message.text. The typed TOOL_USE block is the authority.
    return not (message.message_type == "tool_use" and has_typed_tool_use)


def _source_for_message(
    *,
    session_id: str,
    message: RenderableMessage,
    block: RenderableBlock | None = None,
    block_index: int | None = None,
    result: _ResultMatch | None = None,
    attachment_id: str | None = None,
) -> SemanticCardSource:
    result_message = result.message if result is not None else None
    result_block = result.block if result is not None else None
    return SemanticCardSource(
        session_id=session_id,
        provider_family=message.provider_family,
        origin=message.origin,
        message_id=message.id,
        block_id=block.block_id if block is not None else None,
        block_index=block_index,
        tool_name=block.tool_name if block is not None else None,
        tool_id=block.tool_id if block is not None else None,
        attachment_id=attachment_id,
        material_origin=message.material_origin,
        occurred_at=message.occurred_at,
        duration_ms=message.duration_ms,
        parent_message_id=message.parent_id,
        variant_index=message.variant_index,
        is_active_path=message.is_active_path,
        is_active_leaf=message.is_active_leaf,
        inherited_prefix=message.inherited_prefix,
        result_message_id=result_message.id if result_message is not None else None,
        result_block_id=result_block.block_id if result_block is not None else None,
        result_block_index=result.coordinate.block_index if result is not None else None,
        result_duration_ms=result_message.duration_ms if result_message is not None else None,
        result_material_origin=result_message.material_origin if result_message is not None else None,
        result_inherited_prefix=result_message.inherited_prefix if result_message is not None else None,
    )


def _build_tool_card(
    *,
    session_id: str,
    message: RenderableMessage,
    block_index: int,
    block: RenderableBlock,
    result: _ResultMatch | None,
) -> SemanticCard:
    result_block = result.block if result is not None else None
    classification = classify_tool(
        provider_family=message.provider_family,
        tool_name=block.tool_name,
        semantic_type=block.semantic_type,
    )
    source = _source_for_message(
        session_id=session_id,
        message=message,
        block=block,
        block_index=block_index,
        result=result,
    )
    outcome, outcome_caveats = _outcome_from_result(result_block)
    common_caveats: list[str] = list(outcome_caveats)
    if block.tool_input_raw:
        common_caveats.append("tool input was retained as raw text because it was not valid JSON")
    if result_block is None:
        common_caveats.append("no structurally paired tool result is present")

    if classification.card_kind is SemanticCardKind.SHELL:
        builder = _build_shell_card
    elif classification.card_kind is SemanticCardKind.FILE_READ:
        builder = _build_file_read_card
    elif classification.card_kind is SemanticCardKind.FILE_EDIT:
        builder = _build_file_edit_card
    elif classification.card_kind is SemanticCardKind.SEARCH:
        builder = _build_search_card
    elif classification.card_kind is SemanticCardKind.WEB:
        builder = _build_web_card
    elif classification.card_kind is SemanticCardKind.TASK:
        builder = _build_task_card
    elif classification.card_kind is SemanticCardKind.MCP:
        builder = _build_mcp_card
    else:
        builder = None
    if builder is not None:
        return builder(
            block,
            result_block,
            source,
            outcome,
            tuple(common_caveats),
            classification,
        )
    return _build_fallback_card(
        block,
        result_block,
        source,
        outcome,
        tuple(common_caveats),
        classification,
    )


def _base_tool_fields(
    block: RenderableBlock,
    classification: ToolClassification,
    *,
    duration_ms: int | None,
) -> list[SemanticCardField]:
    fields = [SemanticCardField("tool", block.tool_name or "unknown")]
    fields.append(SemanticCardField("semantic family", classification.semantic_type))
    fields.append(SemanticCardField("classification", classification.basis))
    if duration_ms is not None:
        fields.append(SemanticCardField("duration", f"{duration_ms} ms"))
    return fields


def _effective_duration(source: SemanticCardSource) -> int | None:
    return source.result_duration_ms if source.result_duration_ms is not None else source.duration_ms


def _build_shell_card(
    block: RenderableBlock,
    result: RenderableBlock | None,
    source: SemanticCardSource,
    outcome: SemanticCardOutcome,
    caveats: tuple[str, ...],
    classification: ToolClassification,
) -> SemanticCard:
    command = _first_scalar(block.tool_input, ("command", "cmd", "script"))
    cwd = _first_scalar(block.tool_input, ("cwd", "working_directory", "workdir"))
    fields = _base_tool_fields(block, classification, duration_ms=_effective_duration(source))
    if command:
        fields.append(SemanticCardField("command", command))
    if cwd:
        fields.append(SemanticCardField("cwd", cwd))
    local_caveats = caveats + (() if command else ("no exact command field is present in tool input",))
    previews = _result_previews(result, kind="output")
    return SemanticCard(
        kind=SemanticCardKind.SHELL,
        title="Shell command",
        summary=command,
        source=source,
        outcome=outcome,
        fields=tuple(fields),
        previews=previews,
        raw_evidence=_raw_evidence(block, result),
        caveats=_deduplicate(local_caveats + _preview_caveats(previews)),
    )


def _build_file_read_card(
    block: RenderableBlock,
    result: RenderableBlock | None,
    source: SemanticCardSource,
    outcome: SemanticCardOutcome,
    caveats: tuple[str, ...],
    classification: ToolClassification,
) -> SemanticCard:
    path = _first_scalar(block.tool_input, ("file_path", "path", "file", "filename"))
    offset = _first_scalar(block.tool_input, ("offset", "start", "line_start"))
    limit = _first_scalar(block.tool_input, ("limit", "count", "line_count", "end"))
    fields = _base_tool_fields(block, classification, duration_ms=_effective_duration(source))
    if path:
        fields.append(SemanticCardField("path", path))
    if offset:
        fields.append(SemanticCardField("offset", offset))
    if limit:
        fields.append(SemanticCardField("limit", limit))
    local_caveats = caveats + (() if path else ("no exact path field is present in tool input",))
    previews = _result_previews(result, kind="content")
    return SemanticCard(
        kind=SemanticCardKind.FILE_READ,
        title="File read",
        summary=path,
        source=source,
        outcome=outcome,
        fields=tuple(fields),
        previews=previews,
        raw_evidence=_raw_evidence(block, result),
        caveats=_deduplicate(local_caveats + _preview_caveats(previews)),
    )


def _build_file_edit_card(
    block: RenderableBlock,
    result: RenderableBlock | None,
    source: SemanticCardSource,
    outcome: SemanticCardOutcome,
    caveats: tuple[str, ...],
    classification: ToolClassification,
) -> SemanticCard:
    path = _first_scalar(block.tool_input, ("file_path", "path", "file", "filename"))
    diff_text = _edit_diff(block.tool_input, path=path)
    previews: list[SemanticCardPreview] = []
    local_caveats = list(caveats)
    if diff_text:
        previews.append(_bounded_preview(diff_text, kind="diff"))
    else:
        local_caveats.append("no exact diff could be constructed from the available tool input")
    if result is not None and result.text:
        previews.append(
            _bounded_preview(
                result.text,
                kind="result",
                encoding_replacements=result.text_encoding_replacements,
            )
        )
    fields = _base_tool_fields(block, classification, duration_ms=_effective_duration(source))
    if path:
        fields.append(SemanticCardField("path", path))
    else:
        local_caveats.append("no exact path field is present in tool input")
    return SemanticCard(
        kind=SemanticCardKind.FILE_EDIT,
        title="File write" if classification.semantic_type == "file_write" else "File edit",
        summary=path,
        source=source,
        outcome=outcome,
        fields=tuple(fields),
        previews=tuple(previews),
        raw_evidence=_raw_evidence(block, result),
        caveats=_deduplicate(tuple(local_caveats) + _preview_caveats(tuple(previews))),
    )


def _build_search_card(
    block: RenderableBlock,
    result: RenderableBlock | None,
    source: SemanticCardSource,
    outcome: SemanticCardOutcome,
    caveats: tuple[str, ...],
    classification: ToolClassification,
) -> SemanticCard:
    query = _first_scalar(block.tool_input, ("query", "pattern", "regex", "search_term", "glob"))
    scope = _first_scalar(block.tool_input, ("path", "root", "directory", "cwd", "include"))
    fields = _base_tool_fields(block, classification, duration_ms=_effective_duration(source))
    if query:
        fields.append(SemanticCardField("query", query))
    if scope:
        fields.append(SemanticCardField("scope", scope))
    local_caveats = caveats + (() if query else ("no exact query/pattern field is present in tool input",))
    previews = _result_previews(result, kind="matches")
    return SemanticCard(
        kind=SemanticCardKind.SEARCH,
        title="Search",
        summary=query or scope,
        source=source,
        outcome=outcome,
        fields=tuple(fields),
        previews=previews,
        raw_evidence=_raw_evidence(block, result),
        caveats=_deduplicate(local_caveats + _preview_caveats(previews)),
    )


def _build_web_card(
    block: RenderableBlock,
    result: RenderableBlock | None,
    source: SemanticCardSource,
    outcome: SemanticCardOutcome,
    caveats: tuple[str, ...],
    classification: ToolClassification,
) -> SemanticCard:
    url = _first_scalar(block.tool_input, ("url", "uri", "href"))
    query = _first_web_query(block.tool_input)
    fields = _base_tool_fields(block, classification, duration_ms=_effective_duration(source))
    if url:
        fields.append(SemanticCardField("url", url))
    if query:
        fields.append(SemanticCardField("query", query))
    local_caveats = caveats
    if not url and not query:
        local_caveats += ("no exact URL or query field is present in tool input",)
    previews = _result_previews(result, kind="result")
    return SemanticCard(
        kind=SemanticCardKind.WEB,
        title="Web request",
        summary=url or query,
        source=source,
        outcome=outcome,
        fields=tuple(fields),
        previews=previews,
        raw_evidence=_raw_evidence(block, result),
        caveats=_deduplicate(local_caveats + _preview_caveats(previews)),
    )


def _build_task_card(
    block: RenderableBlock,
    result: RenderableBlock | None,
    source: SemanticCardSource,
    outcome: SemanticCardOutcome,
    caveats: tuple[str, ...],
    classification: ToolClassification,
) -> SemanticCard:
    prompt = _first_scalar(block.tool_input, ("prompt", "task", "instructions", "description", "question"))
    agent_type = _first_scalar(block.tool_input, ("subagent_type", "agent_type", "agent"))
    fields = _base_tool_fields(block, classification, duration_ms=_effective_duration(source))
    if agent_type:
        fields.append(SemanticCardField("agent", agent_type))
    if prompt:
        fields.append(SemanticCardField("request", prompt))
    previews = _result_previews(result, kind="result")
    return SemanticCard(
        kind=SemanticCardKind.TASK,
        title="Task / delegation",
        summary=prompt or agent_type,
        source=source,
        outcome=outcome,
        fields=tuple(fields),
        previews=previews,
        raw_evidence=_raw_evidence(block, result),
        caveats=_deduplicate(caveats + _preview_caveats(previews)),
    )


def _build_mcp_card(
    block: RenderableBlock,
    result: RenderableBlock | None,
    source: SemanticCardSource,
    outcome: SemanticCardOutcome,
    caveats: tuple[str, ...],
    classification: ToolClassification,
) -> SemanticCard:
    identity = parse_mcp_tool_name(block.tool_name)
    fields = _base_tool_fields(block, classification, duration_ms=_effective_duration(source))
    if identity is not None:
        fields.append(SemanticCardField("MCP server", identity.server))
        fields.append(SemanticCardField("MCP tool", identity.tool))
    target = _first_scalar(
        block.tool_input,
        ("resource", "path", "url", "uri", "query", "command", "name", "id"),
    )
    if target:
        fields.append(SemanticCardField("target", target))
    input_preview = _input_preview(block)
    previews = (() if input_preview is None else (input_preview,)) + _result_previews(result, kind="result")
    return SemanticCard(
        kind=SemanticCardKind.MCP,
        title="MCP tool call",
        summary=(f"{identity.server}/{identity.tool}" if identity is not None else block.tool_name),
        source=source,
        outcome=outcome,
        fields=tuple(fields),
        previews=previews,
        raw_evidence=_raw_evidence(block, result),
        caveats=_deduplicate(caveats + _preview_caveats(previews)),
    )


def _build_fallback_card(
    block: RenderableBlock,
    result: RenderableBlock | None,
    source: SemanticCardSource,
    outcome: SemanticCardOutcome,
    caveats: tuple[str, ...],
    classification: ToolClassification,
) -> SemanticCard:
    input_preview = _input_preview(block)
    previews = (() if input_preview is None else (input_preview,)) + _result_previews(result, kind="result")
    fields = _base_tool_fields(block, classification, duration_ms=_effective_duration(source))
    return SemanticCard(
        kind=SemanticCardKind.FALLBACK,
        title=f"Tool evidence · {block.tool_name or 'unknown'}",
        source=source,
        outcome=outcome,
        fields=tuple(fields),
        previews=previews,
        raw_evidence=_raw_evidence(block, result),
        caveats=_deduplicate(
            (
                f"generic fallback policy: {classification.reason}",
                "raw evidence is shown without a guessed specialized card type",
            )
            + caveats
            + _preview_caveats(previews)
        ),
    )


def _build_orphan_result_card(
    *,
    session_id: str,
    message: RenderableMessage,
    block_index: int,
    block: RenderableBlock,
) -> SemanticCard:
    outcome, outcome_caveats = _outcome_from_result(block)
    previews = _result_previews(block, kind="result")
    source = _source_for_message(
        session_id=session_id,
        message=message,
        block=block,
        block_index=block_index,
    )
    return SemanticCard(
        kind=SemanticCardKind.FALLBACK,
        title="Unpaired tool result",
        source=source,
        outcome=outcome,
        fields=(SemanticCardField("tool id", block.tool_id or "unknown"),),
        previews=previews,
        raw_evidence=SemanticCardRawEvidence(result_preview=previews[0] if previews else None),
        caveats=_deduplicate(
            ("no matching tool-use block is present; result semantics were not guessed",)
            + outcome_caveats
            + _preview_caveats(previews)
        ),
    )


def _build_attachment_block_card(
    *,
    session_id: str,
    message: RenderableMessage,
    block_index: int,
    block: RenderableBlock,
) -> SemanticCard:
    fields: list[SemanticCardField] = []
    if block.name:
        fields.append(SemanticCardField("name", block.name))
    if block.mime_type:
        fields.append(SemanticCardField("media type", block.mime_type))
    if block.url:
        fields.append(SemanticCardField("url", block.url))
    return SemanticCard(
        kind=SemanticCardKind.ATTACHMENT,
        title="Attachment",
        summary=block.name or block.mime_type or block.type,
        source=_source_for_message(
            session_id=session_id,
            message=message,
            block=block,
            block_index=block_index,
            attachment_id=block.block_id,
        ),
        fields=tuple(fields),
        caveats=("attachment bytes are not embedded in the transcript document",),
    )


def _build_attachment_envelope_card(
    *,
    session_id: str,
    message: RenderableMessage,
    attachment: RenderableAttachment,
) -> SemanticCard:
    fields: list[SemanticCardField] = []
    if attachment.name:
        fields.append(SemanticCardField("name", attachment.name))
    if attachment.mime_type:
        fields.append(SemanticCardField("media type", attachment.mime_type))
    if attachment.size_bytes is not None:
        fields.append(SemanticCardField("size", f"{attachment.size_bytes} bytes"))
    if attachment.path:
        fields.append(SemanticCardField("path", attachment.path))
    if attachment.source_url:
        fields.append(SemanticCardField("source URL", attachment.source_url))
    if attachment.upload_origin:
        fields.append(SemanticCardField("upload origin", attachment.upload_origin))
    if attachment.caption:
        fields.append(SemanticCardField("caption", attachment.caption))
    return SemanticCard(
        kind=SemanticCardKind.ATTACHMENT,
        title="Attachment",
        summary=attachment.name or attachment.mime_type or attachment.attachment_id,
        source=_source_for_message(
            session_id=session_id,
            message=message,
            attachment_id=attachment.attachment_id,
        ),
        fields=tuple(fields),
        caveats=("attachment bytes are not embedded in the transcript document",),
    )


def _block_attachment_identity(block: RenderableBlock) -> tuple[object, ...]:
    return (block.block_id, block.url, block.name, block.mime_type)


def _attachment_matches_block(
    attachment: RenderableAttachment,
    block_keys: set[tuple[object, ...]],
) -> bool:
    if not block_keys:
        return False
    for block_id, url, name, mime_type in block_keys:
        if attachment.attachment_id is not None and attachment.attachment_id == block_id:
            return True
        comparable = tuple(value for value in (attachment.source_url, attachment.name, attachment.mime_type) if value)
        if comparable and comparable == tuple(value for value in (url, name, mime_type) if value):
            return True
    return False


def _build_lineage_card(lineage: LineageDescriptor) -> SemanticCard:
    fields = [
        SemanticCardField("session", f"session:{lineage.session_id}"),
        SemanticCardField("authority", lineage.authority.value),
        SemanticCardField("availability", lineage.availability.value),
        SemanticCardField("relation", lineage.relation),
    ]
    if lineage.root_session_id:
        fields.append(SemanticCardField("root", f"session:{lineage.root_session_id}"))
    if lineage.parent_session_id:
        fields.append(SemanticCardField("parent", f"session:{lineage.parent_session_id}"))
    elif lineage.parent_native_id:
        fields.append(SemanticCardField("native parent", lineage.parent_native_id))
    if lineage.lineage_complete is not None:
        fields.append(
            SemanticCardField("composed transcript", "complete" if lineage.lineage_complete else "incomplete")
        )
    if lineage.inherited_prefix is not None:
        fields.append(SemanticCardField("inherited prefix", "yes" if lineage.inherited_prefix else "no"))
    if lineage.branch_point_message_id:
        fields.append(SemanticCardField("branch point", f"message:{lineage.branch_point_message_id}"))
    if lineage.active_leaf_message_id:
        fields.append(SemanticCardField("active leaf", f"message:{lineage.active_leaf_message_id}"))

    caveats: list[str] = []
    if lineage.root_session_id is None:
        caveats.append("root identity is unavailable from this bounded lineage authority")
    if lineage.lineage_complete is None:
        caveats.append("transcript-composition completeness is unknown from this bounded lineage authority")
    if lineage.resolved is False:
        caveats.append("the provider-native parent has not resolved to a stored session")
    if lineage.relation == "unknown":
        caveats.append("the structural lineage relation is unknown")
    if lineage.cycle_detected:
        caveats.append("a cycle was detected in the session topology")
    if lineage.lineage_truncation_reason:
        caveats.append(f"composed transcript is truncated: {lineage.lineage_truncation_reason}")
    return SemanticCard(
        kind=SemanticCardKind.LINEAGE,
        title=f"Lineage boundary · {lineage.relation}",
        summary=f"session:{lineage.session_id}",
        source=SemanticCardSource(
            session_id=lineage.session_id,
            provider_family=lineage.provider_family,
            origin=lineage.origin,
        ),
        fields=tuple(fields),
        caveats=tuple(caveats),
    )


def _outcome_from_result(result: RenderableBlock | None) -> tuple[SemanticCardOutcome, tuple[str, ...]]:
    if result is None:
        return SemanticCardOutcome(CardOutcomeState.UNKNOWN), ()
    is_error = result.tool_result_is_error
    exit_code = result.tool_result_exit_code
    if is_error is True or (exit_code is not None and exit_code != 0):
        caveats: tuple[str, ...] = ()
        if is_error is False and exit_code is not None and exit_code != 0:
            caveats = ("source outcome fields disagree; non-zero exit code is treated as failure",)
        elif is_error is True and exit_code == 0:
            caveats = ("source outcome fields disagree; explicit error flag is treated as failure",)
        return SemanticCardOutcome(CardOutcomeState.FAILED, is_error=is_error, exit_code=exit_code), caveats
    if is_error is False or exit_code == 0:
        return SemanticCardOutcome(CardOutcomeState.SUCCEEDED, is_error=is_error, exit_code=exit_code), ()
    return (
        SemanticCardOutcome(CardOutcomeState.UNKNOWN, is_error=is_error, exit_code=exit_code),
        ("tool result exists but carries no structural success/failure outcome",),
    )


def _first_scalar(document: JSONDocument | None, keys: Iterable[str]) -> str | None:
    if document is None:
        return None
    for key in keys:
        value = document.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            return str(value)
    return None


def _first_web_query(document: JSONDocument | None) -> str | None:
    """Extract a query only from explicit web-tool input structure."""

    scalar = _first_scalar(document, ("query", "q", "search_query"))
    if scalar is not None or document is None:
        return scalar
    search_queries = document.get("search_query")
    if not isinstance(search_queries, list):
        return None
    for item in search_queries:
        if not isinstance(item, dict):
            continue
        value = item.get("q", item.get("query"))
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            return str(value)
    return None


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _edit_diff(document: JSONDocument | None, *, path: str | None) -> str | None:
    if document is None:
        return None
    explicit = _first_scalar(document, ("patch", "diff"))
    if explicit:
        return explicit
    old = _first_scalar(document, ("old_string", "old_text", "before"))
    new = _first_scalar(document, ("new_string", "new_text", "after", "content"))
    if new is None:
        return None
    label = path or "file"
    lines = difflib.unified_diff(
        (old or "").splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{label}",
        tofile=f"b/{label}",
        lineterm="",
    )
    return "\n".join(lines)


def _input_preview(block: RenderableBlock) -> SemanticCardPreview | None:
    if block.tool_input is not None:
        rendered = json.dumps(block.tool_input, indent=2, sort_keys=True, ensure_ascii=False)
        return _bounded_preview(rendered, kind="input", encoding_replacements=rendered.count("\ufffd"))
    if block.tool_input_raw is not None:
        return _bounded_preview(
            block.tool_input_raw,
            kind="raw_input",
            encoding_replacements=block.tool_input_raw.count("\ufffd"),
        )
    return None


def _result_previews(result: RenderableBlock | None, *, kind: str) -> tuple[SemanticCardPreview, ...]:
    if result is None or not result.text:
        return ()
    return (
        _bounded_preview(
            result.text,
            kind=kind,
            encoding_replacements=result.text_encoding_replacements,
        ),
    )


def _raw_evidence(block: RenderableBlock, result: RenderableBlock | None) -> SemanticCardRawEvidence:
    previews = _result_previews(result, kind="raw_result")
    return SemanticCardRawEvidence(
        tool_input=block.tool_input,
        tool_input_raw=block.tool_input_raw,
        result_preview=previews[0] if previews else None,
    )


def _bounded_preview(
    text: str,
    *,
    kind: str,
    encoding_replacements: int = 0,
    head_lines: int = DEFAULT_PREVIEW_HEAD_LINES,
    tail_lines: int = DEFAULT_PREVIEW_TAIL_LINES,
    max_chars: int = DEFAULT_PREVIEW_MAX_CHARS,
) -> SemanticCardPreview:
    lines = text.splitlines()
    line_count = len(lines)
    if line_count > head_lines + tail_lines:
        selected = lines[:head_lines] + lines[-tail_lines:]
        rendered = "\n".join(selected)
        if len(rendered) <= max_chars:
            return SemanticCardPreview(
                kind=kind,
                text=rendered,
                line_count=line_count,
                omitted_lines=line_count - len(selected),
                truncated=True,
                strategy=PreviewStrategy.HEAD_TAIL,
                encoding_replacements=encoding_replacements,
            )
    if len(text) > max_chars:
        head_chars = max_chars * 3 // 4
        tail_chars = max_chars - head_chars
        rendered = text[:head_chars] + "\n… [character-bounded preview] …\n" + text[-tail_chars:]
        return SemanticCardPreview(
            kind=kind,
            text=rendered,
            line_count=line_count,
            omitted_characters=len(text) - max_chars,
            truncated=True,
            strategy=PreviewStrategy.CHARACTER_BOUNDED,
            encoding_replacements=encoding_replacements,
        )
    return SemanticCardPreview(
        kind=kind,
        text=text,
        line_count=line_count,
        strategy=PreviewStrategy.FULL,
        encoding_replacements=encoding_replacements,
    )


def _preview_caveats(previews: Sequence[SemanticCardPreview]) -> tuple[str, ...]:
    caveats: list[str] = []
    for preview in previews:
        if preview.omitted_lines:
            caveats.append(f"{preview.omitted_lines} {preview.kind} lines are omitted from the bounded preview")
        if preview.omitted_characters:
            caveats.append(
                f"{preview.omitted_characters} {preview.kind} characters are omitted from the bounded preview"
            )
        if preview.encoding_replacements:
            caveats.append(
                f"{preview.encoding_replacements} invalid UTF-8 sequence(s) were rendered with replacement characters"
            )
    return tuple(caveats)


def _deduplicate(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


__all__ = [
    "build_semantic_transcript",
    "coerce_renderable_message",
    "lineage_descriptor_from_archive_envelope",
    "lineage_descriptor_from_session",
    "lineage_descriptor_from_topology",
    "RenderableAttachment",
    "RenderableMessage",
]

"""Shared Claude parser helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, WebConstructType
from polylogue.core.timestamps import parse_timestamp

from ..base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSessionEvent,
    ParsedWebConstruct,
    attachment_from_meta,
    content_blocks_from_segments,
)

CLAUDE_MISSING_MESSAGE_ID_INGEST_FLAG = "degraded:claude-missing-message-id"
CLAUDE_DUPLICATE_MESSAGE_ID_INGEST_FLAG = "diagnostic:claude-duplicate-message-id"
CLAUDE_LINEAGE_CYCLE_INGEST_FLAG = "degraded:claude-lineage-cycle"


@dataclass(frozen=True, slots=True)
class ClaudeMessageNormalization:
    """Normalized Claude chat-message evidence shared by export and browser routes."""

    messages: list[ParsedMessage]
    attachments: list[ParsedAttachment]
    active_leaf_message_provider_id: str | None
    models_used: list[str]
    session_events: list[ParsedSessionEvent]
    ingest_flags: list[str]
    reported_duration_ms: int | None


@dataclass(frozen=True, slots=True)
class _ClaudeMessageEvidence:
    provider_message_id: str
    raw: Mapping[str, object]
    original_index: int
    role: Role
    text: str | None
    timestamp: str | None
    updated_at: str | None
    blocks: list[ParsedContentBlock]
    attachments: list[ParsedAttachment]
    parent_message_provider_id: str | None
    explicit_position: int | None
    explicit_branch_index: int | None
    explicit_variant_index: int | None
    explicit_is_active_path: bool | None
    explicit_is_active_leaf: bool | None
    model_name: str | None
    model_effort: str | None
    duration_ms: int | None
    delivery_status: str | None
    end_turn: bool | None
    thinking_configuration: dict[str, object] | None

    @property
    def has_material(self) -> bool:
        return bool(self.text or self.blocks or self.attachments)


def _optional_non_negative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 else None
    if isinstance(value, str):
        try:
            parsed = int(float(value))
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def _metadata_mapping(item: Mapping[str, object]) -> Mapping[str, object]:
    metadata = item.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _first_string_field(item: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    metadata = _metadata_mapping(item)
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _first_identity_field(item: Mapping[str, object], *keys: str) -> str | None:
    for source in (item, _metadata_mapping(item)):
        for key in keys:
            value = source.get(key)
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (str, int, float)):
                normalized = str(value).strip()
                if normalized:
                    return normalized
    return None


def _first_bool_field(item: Mapping[str, object], *keys: str) -> bool | None:
    for source in (item, _metadata_mapping(item)):
        for key in keys:
            value = source.get(key)
            if isinstance(value, bool):
                return value
    return None


def _first_non_negative_int_field(item: Mapping[str, object], *keys: str) -> int | None:
    for source in (item, _metadata_mapping(item)):
        for key in keys:
            if key not in source:
                continue
            value = _optional_non_negative_int(source.get(key))
            if value is not None:
                return value
    return None


def _message_model_name(item: Mapping[str, object]) -> str | None:
    return _first_string_field(item, "model", "model_name", "modelName", "model_slug")


def _message_model_effort(item: Mapping[str, object]) -> str | None:
    return _first_string_field(item, "effort", "model_effort", "modelEffort")


def _message_duration_ms(item: Mapping[str, object]) -> int | None:
    return _first_non_negative_int_field(item, "durationMs", "duration_ms", "elapsed_ms")


def _message_parent_id(item: Mapping[str, object]) -> str | None:
    return _first_identity_field(
        item,
        "parent_message_uuid",
        "parent_uuid",
        "parent_message_id",
        "parentMessageId",
        "parent_id",
        "parent",
    )


def _message_delivery_status(item: Mapping[str, object]) -> str | None:
    return _first_string_field(item, "delivery_status", "deliveryStatus", "status")


def _message_end_turn(item: Mapping[str, object]) -> bool | None:
    return _first_bool_field(item, "end_turn", "endTurn")


def _raw_role(item: Mapping[str, object]) -> object:
    role = item.get("sender") or item.get("role")
    if role is not None:
        return role
    author = item.get("author")
    if isinstance(author, Mapping):
        return author.get("role")
    return None


def _thinking_configuration(item: Mapping[str, object]) -> dict[str, object] | None:
    sources = (item, _metadata_mapping(item))
    for source in sources:
        for key in ("thinking_config", "thinkingConfig", "thinking", "extended_thinking"):
            value = source.get(key)
            if isinstance(value, Mapping):
                return dict(value)
            if isinstance(value, bool):
                return {"enabled": value}
            if isinstance(value, str) and value:
                return {"mode": value}

    config: dict[str, object] = {}
    for source in sources:
        for key in ("thinking_enabled", "thinkingEnabled", "enable_thinking"):
            value = source.get(key)
            if isinstance(value, bool):
                config["enabled"] = value
                break
        if "enabled" in config:
            break
    for source in sources:
        for key in ("thinking_budget_tokens", "thinking_budget", "budget_tokens", "budgetTokens"):
            value = _optional_non_negative_int(source.get(key))
            if value is not None:
                config["budget_tokens"] = value
                break
        if "budget_tokens" in config:
            break
    return config or None


def reclassify_tool_result_envelope(role: Role, content_blocks: list[ParsedContentBlock]) -> Role:
    """Reclassify a ``role: user`` envelope whose content is all ``tool_result`` to ``Role.TOOL``.

    The Anthropic API protocol requires ``tool_result`` blocks to be carried by
    ``role: user`` messages — the assistant emits ``tool_use`` blocks and the
    runtime replies with corresponding ``tool_result`` blocks under the
    protocol-mandated ``user`` role. Polylogue's outer-envelope role
    normalization classifies these as ``Role.USER``, polluting
    role-scoped message queries with non-typed content.

    See `#428 <https://github.com/Sinity/polylogue/issues/428>`_.
    """
    if role is not Role.USER:
        return role
    if not content_blocks:
        return role
    if all(block.type == BlockType.TOOL_RESULT for block in content_blocks):
        return Role.TOOL
    return role


def extract_text_from_segments(segments: list[object]) -> str | None:
    lines: list[str] = []
    for segment in segments:
        if isinstance(segment, str):
            if segment:
                lines.append(segment)
            continue
        if not isinstance(segment, dict):
            continue
        seg_type = segment.get("type")
        if seg_type in {"tool_use", "tool_result"}:
            lines.append(json.dumps(segment, sort_keys=True))
            continue
        if seg_type == "thinking":
            seg_thinking = segment.get("thinking")
            if isinstance(seg_thinking, str):
                lines.append(f"<thinking>{seg_thinking}</thinking>")
                continue
        seg_text = segment.get("text")
        if isinstance(seg_text, str):
            lines.append(seg_text)
            continue
        seg_content = segment.get("content")
        if isinstance(seg_content, str):
            lines.append(seg_content)
            continue
    combined = "\n".join(line for line in lines if line)
    return combined or None


def normalize_timestamp(ts: int | float | str | None) -> str | None:
    if ts is None:
        return None
    try:
        val = float(ts)
        if val > 1e11:
            val = val / 1000.0
        dt = parse_timestamp(val)
        return dt.isoformat() if dt is not None else None
    except (ValueError, TypeError):
        pass
    if isinstance(ts, str):
        dt = parse_timestamp(ts)
        if dt is not None:
            return dt.isoformat()
    return None


def _citation_construct(raw: object) -> ParsedWebConstruct | None:
    if not isinstance(raw, Mapping):
        return None
    details = raw.get("details")
    details_mapping = details if isinstance(details, Mapping) else {}

    def first_string(*keys: str) -> str | None:
        for source in (raw, details_mapping):
            for key in keys:
                value = source.get(key)
                if isinstance(value, str) and value:
                    return value
        return None

    url = first_string("url", "source_url", "sourceUrl")
    title = first_string("title", "name")
    text = first_string("text", "snippet", "quote")
    source_id = first_string("uuid", "id", "source_id", "sourceId")
    provider_key = first_string("type", "source_type", "sourceType") or "claude_citation"
    start_index = _first_non_negative_int_field(raw, "start_index", "startIndex")
    end_index = _first_non_negative_int_field(raw, "end_index", "endIndex")
    if not any((url, title, text, source_id, start_index is not None, end_index is not None)):
        return None
    return ParsedWebConstruct(
        construct_type=WebConstructType.CONTENT_REFERENCE,
        provider_key=provider_key,
        title=title,
        url=url,
        text=text,
        source_id=source_id,
        start_index=start_index,
        end_index=end_index,
    )


def _artifact_construct(segment: Mapping[str, object]) -> ParsedWebConstruct | None:
    if segment.get("type") != "tool_use":
        return None
    raw_input = segment.get("input")
    if not isinstance(raw_input, Mapping):
        return None
    mime_type = raw_input.get("type")
    if not isinstance(mime_type, str) or not mime_type.startswith("application/vnd.ant."):
        return None
    title = raw_input.get("title")
    content = raw_input.get("content") or raw_input.get("code")
    source_id = raw_input.get("version_uuid") or raw_input.get("id")
    return ParsedWebConstruct(
        construct_type=WebConstructType.CANVAS,
        provider_key=mime_type,
        title=str(title) if title is not None else None,
        text=str(content) if content is not None else None,
        source_id=str(source_id) if source_id is not None else None,
        mime_type=mime_type,
    )


def _claude_content_blocks(content: object) -> list[ParsedContentBlock]:
    if not isinstance(content, list):
        return content_blocks_from_segments(content)

    known_segment_types = {
        "text",
        "thinking",
        "tool_use",
        "tool_result",
        "image",
        "document",
        "token_budget",
        "voice_note",
        "code",
    }
    blocks: list[ParsedContentBlock] = []
    for raw_segment in content:
        if not isinstance(raw_segment, Mapping):
            blocks.extend(content_blocks_from_segments([raw_segment]))
            continue

        segment = dict(raw_segment)
        provider_type = segment.get("type")
        if isinstance(provider_type, str) and provider_type not in known_segment_types:
            # Keep a non-semantic structural witness for provider block types
            # Polylogue does not yet understand. Raw source evidence remains
            # authoritative for the opaque fields.
            segment_blocks = [
                ParsedContentBlock(
                    type=BlockType.TEXT,
                    metadata={
                        "provider_type": provider_type,
                        "raw_preserved_in_source": True,
                    },
                )
            ]
        else:
            segment_blocks = content_blocks_from_segments([raw_segment])

        constructs: list[ParsedWebConstruct] = []
        citations = segment.get("citations")
        if isinstance(citations, list):
            constructs.extend(
                construct for citation in citations if (construct := _citation_construct(citation)) is not None
            )
        artifact = _artifact_construct(segment)
        if artifact is not None:
            constructs.append(artifact)

        if not segment_blocks and isinstance(provider_type, str) and provider_type:
            segment_blocks = [
                ParsedContentBlock(
                    type=BlockType.TEXT,
                    metadata={
                        "provider_type": provider_type,
                        "raw_preserved_in_source": True,
                    },
                )
            ]
        if constructs and segment_blocks:
            first = segment_blocks[0]
            segment_blocks[0] = first.model_copy(update={"web_constructs": [*first.web_constructs, *constructs]})
        blocks.extend(segment_blocks)
    return blocks


def _extract_message_text(item: Mapping[str, object]) -> str | None:
    text = item.get("text")
    if isinstance(text, str) and text:
        return text
    content = item.get("content")
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        return extract_text_from_segments(content)
    if isinstance(content, Mapping):
        nested_text = content.get("text")
        if isinstance(nested_text, str) and nested_text:
            return nested_text
        parts = content.get("parts")
        if isinstance(parts, list):
            combined = "\n".join(str(part) for part in parts if isinstance(part, str) and part)
            return combined or None
    return None


def _message_attachments(item: Mapping[str, object], message_id: str) -> list[ParsedAttachment]:
    raw_attachments: list[object] = []
    for key in ("attachments", "files"):
        value = item.get(key)
        if isinstance(value, list):
            raw_attachments.extend(value)
    attachments: list[ParsedAttachment] = []
    for index, meta in enumerate(raw_attachments, start=1):
        attachment = attachment_from_meta(meta, message_id, index)
        if attachment is not None:
            attachments.append(attachment)
    return attachments


def _canonical_record(item: Mapping[str, object]) -> str:
    return json.dumps(dict(item), sort_keys=True, separators=(",", ":"), default=str)


def _evidence_richness(evidence: _ClaudeMessageEvidence) -> tuple[int, float, str]:
    parsed_updated = parse_timestamp(evidence.updated_at) if evidence.updated_at is not None else None
    updated = parsed_updated.timestamp() if parsed_updated is not None else float("-inf")
    score = (
        (8 if evidence.text else 0)
        + len(evidence.blocks) * 6
        + len(evidence.attachments) * 5
        + (3 if evidence.parent_message_provider_id else 0)
        + (2 if evidence.model_name else 0)
        + (2 if evidence.delivery_status else 0)
        + (2 if evidence.thinking_configuration else 0)
    )
    return score, updated, _canonical_record(evidence.raw)


def _timestamp_sort_value(timestamp: str | None) -> float:
    if timestamp is None:
        return float("inf")
    parsed = parse_timestamp(timestamp)
    return parsed.timestamp() if parsed is not None else float("inf")


def _sibling_sort_key(evidence: _ClaudeMessageEvidence) -> tuple[int, int, float, float, str]:
    explicit_variant = evidence.explicit_variant_index
    explicit_branch = evidence.explicit_branch_index
    return (
        0 if explicit_variant is not None else 1,
        explicit_variant if explicit_variant is not None else explicit_branch or 0,
        _timestamp_sort_value(evidence.timestamp),
        _timestamp_sort_value(evidence.updated_at),
        evidence.provider_message_id,
    )


def _merge_attachment_rows(attachments: list[ParsedAttachment]) -> list[ParsedAttachment]:
    merged: dict[str, ParsedAttachment] = {}
    for candidate in attachments:
        existing = merged.get(candidate.provider_attachment_id)
        if existing is None:
            merged[candidate.provider_attachment_id] = candidate
            continue
        preferred = candidate if candidate.inline_bytes is not None and existing.inline_bytes is None else existing
        other = existing if preferred is candidate else candidate
        merged[candidate.provider_attachment_id] = preferred.model_copy(
            update={
                "message_provider_id": preferred.message_provider_id or other.message_provider_id,
                "name": preferred.name or other.name,
                "mime_type": preferred.mime_type or other.mime_type,
                "size_bytes": preferred.size_bytes if preferred.size_bytes is not None else other.size_bytes,
                "provider_file_id": preferred.provider_file_id or other.provider_file_id,
                "provider_drive_id": preferred.provider_drive_id or other.provider_drive_id,
                "source_url": preferred.source_url or other.source_url,
            }
        )
    return list(merged.values())


def _lineage_depths(
    evidence_by_id: Mapping[str, _ClaudeMessageEvidence],
) -> tuple[dict[str, int], bool]:
    """Compute parent depth iteratively and degrade cycles to depth zero."""

    depths: dict[str, int] = {}
    cycle_detected = False
    for start_id in evidence_by_id:
        if start_id in depths:
            continue

        chain: list[str] = []
        chain_position: dict[str, int] = {}
        cursor: str | None = start_id
        base_depth = -1
        while cursor is not None and cursor in evidence_by_id and cursor not in depths:
            cycle_start = chain_position.get(cursor)
            if cycle_start is not None:
                cycle_detected = True
                for cycle_id in chain[cycle_start:]:
                    depths[cycle_id] = 0
                chain = chain[:cycle_start]
                base_depth = 0
                break
            chain_position[cursor] = len(chain)
            chain.append(cursor)
            cursor = evidence_by_id[cursor].parent_message_provider_id
        else:
            if cursor is not None and cursor in depths:
                base_depth = depths[cursor]

        for message_id in reversed(chain):
            base_depth += 1
            depths[message_id] = base_depth

    return depths, cycle_detected


def _active_path_state(
    evidence_by_id: Mapping[str, _ClaudeMessageEvidence],
    emitted_ids: set[str],
    *,
    flat_mode: bool,
    explicit_active_leaf_message_provider_id: str | None,
    order_key_by_id: Mapping[str, tuple[int, int, str]],
) -> tuple[dict[str, bool | None], dict[str, bool | None], str | None]:
    leaf_id = explicit_active_leaf_message_provider_id
    if leaf_id not in evidence_by_id:
        leaf_id = None

    explicit_leaf_ids = [
        evidence.provider_message_id for evidence in evidence_by_id.values() if evidence.explicit_is_active_leaf is True
    ]
    if leaf_id is None and len(explicit_leaf_ids) == 1:
        leaf_id = explicit_leaf_ids[0]

    path_values = {
        message_id: evidence.explicit_is_active_path
        for message_id, evidence in evidence_by_id.items()
        if message_id in emitted_ids
    }
    leaf_values = {
        message_id: evidence.explicit_is_active_leaf
        for message_id, evidence in evidence_by_id.items()
        if message_id in emitted_ids
    }

    if leaf_id is None and any(value is True for value in path_values.values()):
        active_ids = {message_id for message_id, value in path_values.items() if value is True}
        active_children = {
            evidence.parent_message_provider_id
            for evidence in evidence_by_id.values()
            if evidence.provider_message_id in active_ids and evidence.parent_message_provider_id in active_ids
        }
        candidates = sorted(active_ids - active_children)
        if len(candidates) == 1:
            leaf_id = candidates[0]

    if leaf_id is None and flat_mode and emitted_ids:
        leaf_id = max(emitted_ids, key=order_key_by_id.__getitem__)

    if flat_mode:
        if not any(value is not None for value in path_values.values()):
            path_values = dict.fromkeys(emitted_ids, True)
        if leaf_id is not None:
            leaf_values = {message_id: message_id == leaf_id for message_id in emitted_ids}
        return path_values, leaf_values, leaf_id if leaf_id in emitted_ids else None

    if leaf_id is None:
        parent_ids = {
            evidence.parent_message_provider_id
            for evidence in evidence_by_id.values()
            if evidence.provider_message_id in emitted_ids and evidence.parent_message_provider_id in emitted_ids
        }
        terminal_ids = emitted_ids - parent_ids
        if len(terminal_ids) == 1:
            leaf_id = next(iter(terminal_ids))

    if leaf_id is not None:
        active_path_ids: set[str] = set()
        cursor: str | None = leaf_id
        while cursor is not None and cursor not in active_path_ids:
            active_path_ids.add(cursor)
            evidence = evidence_by_id.get(cursor)
            cursor = evidence.parent_message_provider_id if evidence is not None else None
        path_values = {message_id: message_id in active_path_ids for message_id in emitted_ids}
        leaf_values = {message_id: message_id == leaf_id for message_id in emitted_ids}
        return path_values, leaf_values, leaf_id if leaf_id in emitted_ids else None

    return path_values, leaf_values, None


def normalize_chat_messages(
    chat_messages: list[object],
    *,
    session_model: str | None = None,
    session_effort: str | None = None,
    session_thinking_configuration: dict[str, object] | None = None,
    session_created_at: str | None = None,
    session_updated_at: str | None = None,
    active_leaf_message_provider_id: str | None = None,
) -> ClaudeMessageNormalization:
    """Normalize Claude web messages without splitting strict and loose shapes.

    Native IDs and parent pointers are authoritative. Array order is used only
    when flat records lack lineage, explicit positions, and usable timestamps.
    """

    raw_evidence: list[_ClaudeMessageEvidence] = []
    ingest_flags: list[str] = []
    for index, raw_item in enumerate(chat_messages, start=1):
        if not isinstance(raw_item, Mapping):
            continue
        item = dict(raw_item)
        message_id = _first_identity_field(
            item,
            "uuid",
            "id",
            "message_id",
            "messageId",
            "provider_message_id",
        )
        if message_id is None:
            message_id = f"msg-{index}"
            ingest_flags.append(CLAUDE_MISSING_MESSAGE_ID_INGEST_FLAG)

        raw_role = _raw_role(item)
        role = Role.normalize(str(raw_role)) if isinstance(raw_role, str) and raw_role else Role.UNKNOWN
        text = _extract_message_text(item)
        raw_content = item.get("content")
        content_blocks = _claude_content_blocks(raw_content)
        if not content_blocks and text:
            content_blocks = [ParsedContentBlock(type=BlockType.TEXT, text=text)]
        role = reclassify_tool_result_envelope(role, content_blocks)

        raw_created_at = item.get("created_at") or item.get("create_time") or item.get("timestamp")
        raw_updated_at = item.get("updated_at") or item.get("update_time") or item.get("edited_at")
        attachments = _message_attachments(item, message_id)
        raw_evidence.append(
            _ClaudeMessageEvidence(
                provider_message_id=message_id,
                raw=item,
                original_index=index,
                role=role,
                text=text,
                timestamp=normalize_timestamp(
                    raw_created_at if isinstance(raw_created_at, (int, float, str)) else None
                ),
                updated_at=normalize_timestamp(
                    raw_updated_at if isinstance(raw_updated_at, (int, float, str)) else None
                ),
                blocks=content_blocks,
                attachments=attachments,
                parent_message_provider_id=_message_parent_id(item),
                explicit_position=_first_non_negative_int_field(item, "position"),
                explicit_branch_index=_first_non_negative_int_field(item, "branch_index", "branchIndex"),
                explicit_variant_index=_first_non_negative_int_field(item, "variant_index", "variantIndex"),
                explicit_is_active_path=_first_bool_field(item, "is_active_path", "isActivePath", "active_path"),
                explicit_is_active_leaf=_first_bool_field(item, "is_active_leaf", "isActiveLeaf", "active_leaf"),
                model_name=_message_model_name(item) or session_model,
                model_effort=_message_model_effort(item) or session_effort,
                duration_ms=_message_duration_ms(item),
                delivery_status=_message_delivery_status(item),
                end_turn=_message_end_turn(item),
                thinking_configuration=_thinking_configuration(item),
            )
        )

    evidence_by_id: dict[str, _ClaudeMessageEvidence] = {}
    duplicate_ids: set[str] = set()
    for evidence in raw_evidence:
        existing = evidence_by_id.get(evidence.provider_message_id)
        if existing is None:
            evidence_by_id[evidence.provider_message_id] = evidence
            continue
        duplicate_ids.add(evidence.provider_message_id)
        evidence_by_id[evidence.provider_message_id] = max((existing, evidence), key=_evidence_richness)
    if duplicate_ids:
        ingest_flags.append(CLAUDE_DUPLICATE_MESSAGE_ID_INGEST_FLAG)

    emitted = [evidence for evidence in evidence_by_id.values() if evidence.has_material]
    emitted_ids = {evidence.provider_message_id for evidence in emitted}
    flat_mode = not any(evidence.parent_message_provider_id for evidence in evidence_by_id.values())

    branch_index_by_id: dict[str, int] = {}
    if flat_mode:
        ordered_flat = sorted(
            emitted,
            key=lambda evidence: (
                evidence.explicit_position if evidence.explicit_position is not None else 2**31,
                0 if evidence.timestamp is not None else 1,
                _timestamp_sort_value(evidence.timestamp),
                evidence.provider_message_id if evidence.timestamp is not None else "",
                evidence.original_index,
            ),
        )
        position_by_id = {
            evidence.provider_message_id: (
                evidence.explicit_position if evidence.explicit_position is not None else position
            )
            for position, evidence in enumerate(ordered_flat)
        }
        for evidence in emitted:
            branch_index_by_id[evidence.provider_message_id] = evidence.explicit_branch_index or 0
    else:
        depths, cycle_detected = _lineage_depths(evidence_by_id)
        if cycle_detected:
            ingest_flags.append(CLAUDE_LINEAGE_CYCLE_INGEST_FLAG)
        minimum_emitted_depth = min((depths[evidence.provider_message_id] for evidence in emitted), default=0)
        position_by_id = {
            evidence.provider_message_id: (
                evidence.explicit_position
                if evidence.explicit_position is not None
                else max(0, depths[evidence.provider_message_id] - minimum_emitted_depth)
            )
            for evidence in emitted
        }
        siblings_by_parent: dict[str | None, list[_ClaudeMessageEvidence]] = defaultdict(list)
        for evidence in evidence_by_id.values():
            siblings_by_parent[evidence.parent_message_provider_id].append(evidence)
        for siblings in siblings_by_parent.values():
            for rank, evidence in enumerate(sorted(siblings, key=_sibling_sort_key)):
                branch_index_by_id[evidence.provider_message_id] = (
                    evidence.explicit_branch_index if evidence.explicit_branch_index is not None else rank
                )

    variant_index_by_id = {
        evidence.provider_message_id: (
            evidence.explicit_variant_index
            if evidence.explicit_variant_index is not None
            else branch_index_by_id.get(evidence.provider_message_id, 0)
        )
        for evidence in emitted
    }
    order_key_by_id = {
        evidence.provider_message_id: (
            position_by_id[evidence.provider_message_id],
            variant_index_by_id[evidence.provider_message_id],
            evidence.provider_message_id,
        )
        for evidence in emitted
    }
    path_values, leaf_values, normalized_active_leaf = _active_path_state(
        evidence_by_id,
        emitted_ids,
        flat_mode=flat_mode,
        explicit_active_leaf_message_provider_id=active_leaf_message_provider_id,
        order_key_by_id=order_key_by_id,
    )

    messages = [
        ParsedMessage(
            provider_message_id=evidence.provider_message_id,
            role=evidence.role,
            text=evidence.text,
            timestamp=evidence.timestamp,
            blocks=evidence.blocks,
            parent_message_provider_id=evidence.parent_message_provider_id,
            position=position_by_id[evidence.provider_message_id],
            branch_index=branch_index_by_id.get(evidence.provider_message_id, 0),
            variant_index=variant_index_by_id[evidence.provider_message_id],
            is_active_path=path_values.get(evidence.provider_message_id),
            is_active_leaf=leaf_values.get(evidence.provider_message_id),
            model_name=evidence.model_name,
            model_effort=evidence.model_effort,
            duration_ms=evidence.duration_ms,
            delivery_status=evidence.delivery_status,
            end_turn=evidence.end_turn,
        )
        for evidence in sorted(emitted, key=lambda row: order_key_by_id[row.provider_message_id])
    ]

    attachments = _merge_attachment_rows([attachment for evidence in emitted for attachment in evidence.attachments])
    models_used: list[str] = []
    for model_name in [session_model, *(message.model_name for message in messages)]:
        if model_name and model_name not in models_used:
            models_used.append(model_name)

    session_events: list[ParsedSessionEvent] = []
    if session_model or session_effort or session_thinking_configuration:
        configuration_payload: dict[str, object] = {}
        if session_model:
            configuration_payload["model"] = session_model
        if session_effort:
            configuration_payload["effort"] = session_effort
        if session_thinking_configuration:
            configuration_payload["thinking"] = session_thinking_configuration
        session_events.append(
            ParsedSessionEvent(
                event_type="model_configuration",
                timestamp=session_updated_at or session_created_at,
                payload=configuration_payload,
            )
        )

    for evidence in sorted(emitted, key=lambda row: order_key_by_id[row.provider_message_id]):
        if evidence.thinking_configuration:
            payload: dict[str, object] = {"thinking": evidence.thinking_configuration}
            if evidence.model_name:
                payload["model"] = evidence.model_name
            if evidence.model_effort:
                payload["effort"] = evidence.model_effort
            session_events.append(
                ParsedSessionEvent(
                    event_type="model_configuration",
                    timestamp=evidence.updated_at or evidence.timestamp,
                    source_message_provider_id=evidence.provider_message_id,
                    payload=payload,
                )
            )
        if evidence.updated_at and evidence.updated_at != evidence.timestamp:
            update_payload: dict[str, object] = {"updated_at": evidence.updated_at}
            if evidence.timestamp:
                update_payload["created_at"] = evidence.timestamp
            if evidence.delivery_status:
                update_payload["status"] = evidence.delivery_status
            revision_id = _first_identity_field(evidence.raw, "version_uuid", "revision_id", "revisionId")
            explicitly_edited = _first_bool_field(evidence.raw, "is_edited", "isEdited", "edited") is True
            has_edited_timestamp = _first_string_field(evidence.raw, "edited_at", "editedAt") is not None
            if revision_id:
                update_payload["revision_id"] = revision_id
            # A changed provider timestamp is observable update evidence, but it
            # is not necessarily a user edit. Only claim a revision when Claude
            # supplied an explicit revision/edit marker; otherwise keep the
            # event neutral and preserve the timestamps/status verbatim.
            session_events.append(
                ParsedSessionEvent(
                    event_type=(
                        "message_revision"
                        if revision_id or explicitly_edited or has_edited_timestamp
                        else "provider_message_update"
                    ),
                    timestamp=evidence.updated_at,
                    source_message_provider_id=evidence.provider_message_id,
                    payload=update_payload,
                )
            )

    if duplicate_ids:
        session_events.append(
            ParsedSessionEvent(
                event_type="normalization_diagnostic",
                timestamp=session_updated_at or session_created_at,
                payload={
                    "diagnostic": "duplicate_message_ids",
                    "provider_message_ids": sorted(duplicate_ids),
                    "resolution": "richest_structured_record",
                },
            )
        )

    duration_values = [message.duration_ms for message in messages if message.duration_ms is not None]
    return ClaudeMessageNormalization(
        messages=messages,
        attachments=attachments,
        active_leaf_message_provider_id=normalized_active_leaf,
        models_used=models_used,
        session_events=session_events,
        ingest_flags=list(dict.fromkeys(ingest_flags)),
        reported_duration_ms=sum(duration_values) if duration_values else None,
    )


def extract_messages_from_chat_messages(
    chat_messages: list[object],
) -> tuple[list[ParsedMessage], list[ParsedAttachment]]:
    normalized = normalize_chat_messages(chat_messages)
    return normalized.messages, normalized.attachments


def extract_message_text(message_content: object) -> str | None:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        return extract_text_from_segments(message_content)
    if isinstance(message_content, dict):
        text = message_content.get("text")
        if isinstance(text, str):
            return text
        parts = message_content.get("parts")
        if isinstance(parts, list):
            return "\n".join(str(p) for p in parts if p)
    return None


__all__ = [
    "CLAUDE_DUPLICATE_MESSAGE_ID_INGEST_FLAG",
    "CLAUDE_LINEAGE_CYCLE_INGEST_FLAG",
    "CLAUDE_MISSING_MESSAGE_ID_INGEST_FLAG",
    "ClaudeMessageNormalization",
    "extract_message_text",
    "extract_messages_from_chat_messages",
    "extract_text_from_segments",
    "normalize_chat_messages",
    "normalize_timestamp",
]

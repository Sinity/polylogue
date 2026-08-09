"""Shared Claude parser helpers."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from polylogue.archive.message.artifacts import classify_block_message_type, classify_material_origin
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MessageType, WebConstructType
from polylogue.core.timestamps import parse_timestamp

from ..base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSessionEvent,
    ParsedWebConstruct,
    attachment_from_meta,
    content_blocks_from_segments,
    human_authored_override,
    synthetic_message_id,
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
    evidence_key: str
    native_provider_message_id: str
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


def _claude_ai_web_tool_evidence(segment: Mapping[str, object]) -> dict[str, object] | None:
    """Project Claude AI web tool_use/tool_result fields that ``content_blocks_from_segments``
    does not read (that helper is shared with Claude Code, which never emits them).

    ``start_timestamp``/``stop_timestamp`` give real per-block wall-clock
    timing (distinct from message-level timestamps); ``integration_name``/
    ``integration_icon_url``/``is_mcp_app``/``mcp_server_url`` identify which
    connected MCP app served the call; ``approval_key``/``approval_options``
    are the human-in-the-loop permission gate Claude AI shows before running a
    connected-app action; ``display_content`` is the provider's own rendered
    summary of the call (distinct from ``message``, a shorter one-line label).
    All are only ever observed non-null on ``tool_use``/``tool_result``
    segments in the live corpus (2026-07-29 triage).
    """
    evidence: dict[str, object] = {}
    raw_start_timestamp = segment.get("start_timestamp")
    start_timestamp = normalize_timestamp(
        raw_start_timestamp if isinstance(raw_start_timestamp, (int, float, str)) else None
    )
    if start_timestamp is not None:
        evidence["start_timestamp"] = start_timestamp
    raw_stop_timestamp = segment.get("stop_timestamp")
    stop_timestamp = normalize_timestamp(
        raw_stop_timestamp if isinstance(raw_stop_timestamp, (int, float, str)) else None
    )
    if stop_timestamp is not None:
        evidence["stop_timestamp"] = stop_timestamp
    integration_name = segment.get("integration_name")
    if isinstance(integration_name, str) and integration_name:
        evidence["integration_name"] = integration_name
    integration_icon_url = segment.get("integration_icon_url")
    if isinstance(integration_icon_url, str) and integration_icon_url:
        evidence["integration_icon_url"] = integration_icon_url
    is_mcp_app = segment.get("is_mcp_app")
    if isinstance(is_mcp_app, bool):
        evidence["is_mcp_app"] = is_mcp_app
    mcp_server_url = segment.get("mcp_server_url")
    if isinstance(mcp_server_url, str) and mcp_server_url:
        evidence["mcp_server_url"] = mcp_server_url
    approval_key = segment.get("approval_key")
    if isinstance(approval_key, str) and approval_key:
        evidence["approval_key"] = approval_key
    approval_options = segment.get("approval_options")
    if isinstance(approval_options, list) and approval_options:
        evidence["approval_options"] = [option for option in approval_options if isinstance(option, str)]
    display_content = segment.get("display_content")
    if isinstance(display_content, Mapping):
        display_type = display_content.get("type")
        display_text = display_content.get("text")
        if isinstance(display_type, str) or isinstance(display_text, str):
            evidence["display_content"] = {
                "type": display_type if isinstance(display_type, str) else None,
                "text": display_text if isinstance(display_text, str) else None,
            }
    return evidence or None


# polylogue-9x22: ``ParsedContentBlock.metadata`` is never persisted -- the
# ``blocks`` table has no metadata column and the only key the write path
# reads back out of it is ``language`` (``storage/sqlite/archive_tiers/
# write.py:_block_language``). ``_claude_ai_web_tool_evidence`` above merges
# its fields into ``first_block.metadata`` as an in-process carrier from
# ``_claude_content_blocks`` up to ``normalize_chat_messages`` (below), but
# without this projection step the evidence was silently dropped at write
# time despite parsing correctly. Route it through ``session_events``
# instead, keyed to the message via ``source_message_provider_id`` -- the
# same precedent ``hermes_spans.py`` uses for tool-availability evidence
# (polylogue-5o05). One event per tool_use/tool_result block, not one event
# per field.
_CLAUDE_AI_WEB_TOOL_EVIDENCE_KEYS = frozenset(
    {
        "start_timestamp",
        "stop_timestamp",
        "integration_name",
        "integration_icon_url",
        "is_mcp_app",
        "mcp_server_url",
        "approval_key",
        "approval_options",
        "display_content",
    }
)


def _web_tool_evidence_from_block_metadata(metadata: Mapping[str, object] | None) -> dict[str, object] | None:
    if not metadata:
        return None
    evidence = {key: value for key, value in metadata.items() if key in _CLAUDE_AI_WEB_TOOL_EVIDENCE_KEYS}
    return evidence or None


def _web_tool_evidence_events(evidence: _ClaudeMessageEvidence) -> list[ParsedSessionEvent]:
    events: list[ParsedSessionEvent] = []
    for block_index, block in enumerate(evidence.blocks):
        block_evidence = _web_tool_evidence_from_block_metadata(block.metadata)
        if block_evidence is None:
            continue
        start_timestamp = block_evidence.get("start_timestamp")
        events.append(
            ParsedSessionEvent(
                event_type="claude_ai_web_tool_evidence",
                timestamp=start_timestamp if isinstance(start_timestamp, str) else evidence.timestamp,
                source_message_provider_id=evidence.native_provider_message_id,
                payload={"block_index": block_index, **block_evidence},
            )
        )
    return events


# Claude AI (`claude-ai`) parser-diff triage disposition notes, 2026-07-29
# (bd polylogue-2qx.3/polylogue-cgfy). Fields not otherwise mentioned in this
# module's docstrings:
#
#   chat_messages[].attachments[].extracted_content/file_name/file_size/
#   file_type, chat_messages[].files[].file_name/file_uuid
#       FALSE POSITIVE in the parser-diff scan -- already read by the shared
#       ``attachment_from_meta`` (base_support.py, not in the tool's per-
#       provider module list, hence invisible to its AST scan).
#   chat_messages[].content[].is_error, .tool_use_id
#       Same false-positive class: read by ``content_blocks_from_segments``
#       (base_support.py) for every ``tool_result`` segment.
#   chat_messages[].content[].input.*  (dozens of per-tool-call parameter
#   names: query, command, path, calendar_id, ...)
#       FALSE POSITIVE: the whole ``input`` dict is captured verbatim as
#       ``ParsedContentBlock.tool_input`` regardless of which keys it has --
#       parser-diff's name-based scan cannot see a wholesale dict copy.
#   chat_messages[].content[].start_timestamp/stop_timestamp,
#   integration_name/integration_icon_url, approval_key/approval_options,
#   display_content, is_mcp_app, mcp_server_url
#       READ as of this pass -- see ``_claude_ai_web_tool_evidence`` above.
#   summary (session-level)
#       READ as of this pass -- see ``parse_ai``'s ``claude_ai_conversation_summary``
#       event.
#   chat_messages[].content[].content[].* (doc_uuid, uri, extras.*,
#   prompt_context_metadata.*, metadata.site_domain/site_name/favicon_url,
#   is_citable, is_missing, ingestion_date, file_path)
#       DEFERRED, not dropped: this is a Google Workspace/Drive connected-app
#       tool_result's own nested document-citation records (distinct from the
#       message-level ``citations`` list ``_citation_construct`` already
#       projects). It needs its own construct-projection design (a citation
#       has a stable url/title/text; a Drive doc reference has drive-specific
#       identity/provenance fields with no equivalent slot on
#       ``ParsedWebConstruct`` today) rather than a same-pass bolt-on. Filed as
#       a to-acquire item, not silently dropped.
#   chat_messages[].content[].context.tools[].server_uuid/tool_name,
#   .cut_off, .icon_name, .message, .meta, .remaining, .signature,
#   .structured_content, .summaries[].summary, .truncated
#       DELIBERATELY DROPPED for this pass: each is a single low-frequency
#       field on the same tool_use/tool_result segment already covered above,
#       with no corpus evidence yet of carrying information beyond what
#       ``tool_input``/``display_content``/``integration_name`` already
#       capture (``message`` in particular duplicates ``display_content.text``
#       in every sampled instance). Re-audit if a future corpus pass finds
#       divergent values.
#   account (session-level)
#       DELIBERATELY DROPPED: provider account identity is out of scope for
#       per-message/session content evidence and risks conflating multiple
#       real accounts' PII into one field; the archive already scopes by
#       origin/session, not by account.
#   chat_messages[].content[].flags
#       DELIBERATELY DROPPED: the 99% "encountered" figure parser-diff reports
#       is presence-of-key, not presence-of-signal -- the observed-distribution
#       schema shows it null in 19,491 of 19,509 observations (non_null in
#       only 4 of 525 documents, 0.8%), and even those 4 documents' values are
#       a single-element array whose one string is always the same length (14
#       chars) with estimated-distinct 1 across all 18 occurrences -- i.e. one
#       constant opaque flag, not a signal-bearing field. Re-audit if a larger
#       corpus sample ever shows more than one distinct value.
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
            if provider_type in ("tool_use", "tool_result") and segment_blocks:
                web_tool_evidence = _claude_ai_web_tool_evidence(segment)
                if web_tool_evidence is not None:
                    first_block = segment_blocks[0]
                    segment_blocks[0] = first_block.model_copy(
                        update={"metadata": {**(first_block.metadata or {}), **web_tool_evidence}}
                    )

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
    for meta in raw_attachments:
        attachment = attachment_from_meta(meta, message_id)
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
        evidence.evidence_key,
    )


def _resolve_variant_index(
    start_id: str,
    evidence_by_id: Mapping[str, _ClaudeMessageEvidence],
    branch_index_by_id: Mapping[str, int],
    resolved: dict[str, int],
) -> int:
    """Resolve a tree-mode message's variant index, composing rank-0 inheritance.

    A message with an explicit provider variant index, or a nonzero rank among
    its siblings, keeps that value -- those are real branch points. A rank-0
    ("primary") child is just the continuation of whichever variant its parent
    belongs to, so it inherits the parent's resolved variant index rather than
    resetting to 0. This composes through chains of rank-0 descendants (a
    rank-0 child of a rank-0 child of a variant sibling still inherits that
    variant), walking up until an explicit/nonzero variant, the root, or a
    cycle is found.

    Without this, two sibling variants at the same depth each contribute a
    rank-0 continuation at the *same* (position, variant_index=0) coordinate --
    the exact collision shape that silently drops a message via
    `INSERT OR REPLACE` on the messages table's
    ``PRIMARY KEY(session_id, position, variant_index)``.

    Cycles degrade to variant 0 (mirrors `_lineage_depths`' cycle handling)
    instead of looping forever; the final uniqueness pass below is the safety
    net if that ever produces a residual collision anyway.
    """
    chain: list[str] = []
    cursor: str | None = start_id
    seen: set[str] = set()
    value = 0
    while cursor is not None:
        if cursor in resolved:
            value = resolved[cursor]
            break
        if cursor in seen or cursor not in evidence_by_id:
            value = 0
            break
        seen.add(cursor)
        evidence = evidence_by_id[cursor]
        if evidence.explicit_variant_index is not None:
            value = evidence.explicit_variant_index
            resolved[cursor] = value
            break
        rank = branch_index_by_id.get(cursor, 0)
        if rank != 0:
            value = rank
            resolved[cursor] = value
            break
        chain.append(cursor)
        cursor = evidence.parent_message_provider_id
    for message_id in chain:
        resolved[message_id] = value
    return value


def _deduplicate_variant_collisions(
    emitted: list[_ClaudeMessageEvidence],
    position_by_id: Mapping[str, int],
    variant_index_by_id: dict[str, int],
) -> None:
    """Guarantee (position, variant_index) uniqueness per session -- the safety net.

    Parser-level variant assignment (explicit provider values, sibling rank, or
    rank-0 inheritance) is a heuristic and cannot be proven collision-free for
    every exotic input tree shape. No two emitted messages may share
    (position, variant_index): the messages table is unique on exactly that
    pair, so a silent collision here becomes a silently dropped message
    downstream (`INSERT OR REPLACE`) whose blocks then orphan against a
    foreign key that no longer resolves.

    Mutates ``variant_index_by_id`` in place. Positions with no collision are
    left untouched. A colliding position group is fully renumbered in
    order_key order (current variant index, then timestamp, then provider
    message id) so the result is deterministic regardless of which heuristic
    produced the original assignment.
    """
    by_position: dict[int, list[_ClaudeMessageEvidence]] = defaultdict(list)
    for evidence in emitted:
        by_position[position_by_id[evidence.evidence_key]].append(evidence)
    for group in by_position.values():
        variants = [variant_index_by_id[evidence.evidence_key] for evidence in group]
        if len(set(variants)) == len(variants):
            continue
        ordered = sorted(
            group,
            key=lambda evidence: (
                variant_index_by_id[evidence.evidence_key],
                _timestamp_sort_value(evidence.timestamp),
                evidence.evidence_key,
            ),
        )
        for rank, evidence in enumerate(ordered):
            variant_index_by_id[evidence.evidence_key] = rank


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
                "message_position": preferred.message_position
                if preferred.message_position is not None
                else other.message_position,
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
        evidence.evidence_key for evidence in evidence_by_id.values() if evidence.explicit_is_active_leaf is True
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
            if evidence.evidence_key in active_ids and evidence.parent_message_provider_id in active_ids
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
            if evidence.evidence_key in emitted_ids and evidence.parent_message_provider_id in emitted_ids
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
    evidence_key_counts: dict[str, int] = {}
    ingest_flags: list[str] = []
    for index, raw_item in enumerate(chat_messages, start=1):
        if not isinstance(raw_item, Mapping):
            continue
        item = dict(raw_item)
        native_message_id = _first_identity_field(
            item,
            "uuid",
            "id",
            "message_id",
            "messageId",
            "provider_message_id",
        )
        if native_message_id is None:
            native_message_id = ""
            ingest_flags.append(CLAUDE_MISSING_MESSAGE_ID_INGEST_FLAG)

        raw_role = _raw_role(item)
        role = Role.normalize(str(raw_role)) if isinstance(raw_role, str) and raw_role else Role.UNKNOWN
        text = _extract_message_text(item)
        raw_content = item.get("content")
        # polylogue-0qfy: do NOT synthesize a content_blocks entry that just
        # duplicates `text` when `content` itself produced no blocks -- that
        # made `message.blocks` presence (and therefore the message's
        # identity hash, `pipeline/ids.py:_message_hash_payload`) depend on
        # whether a given export vintage's raw record happened to carry a
        # structured `content` field or only a top-level `text` field, for
        # the exact same conversation content. The write path
        # (`storage/sqlite/archive_tiers/write.py:_message_blocks`) already
        # falls back to a text block for storage whenever `message.blocks`
        # is empty, so this synthesis was pure redundancy with no storage
        # benefit and a real comparison-stability cost.
        content_blocks = _claude_content_blocks(raw_content)
        role = reclassify_tool_result_envelope(role, content_blocks)

        raw_created_at = item.get("created_at") or item.get("create_time") or item.get("timestamp")
        raw_updated_at = item.get("updated_at") or item.get("update_time") or item.get("edited_at")
        timestamp = normalize_timestamp(raw_created_at if isinstance(raw_created_at, (int, float, str)) else None)
        base_evidence_key = native_message_id or synthetic_message_id(
            role=role,
            text=text,
            timestamp=timestamp,
            kind="claude-web-evidence",
        )
        occurrence = evidence_key_counts.get(base_evidence_key, 0)
        evidence_key_counts[base_evidence_key] = occurrence + 1
        evidence_key = (
            base_evidence_key
            if native_message_id or occurrence == 0
            else f"{base_evidence_key}:occurrence:{occurrence}"
        )
        attachments = _message_attachments(item, native_message_id)
        raw_evidence.append(
            _ClaudeMessageEvidence(
                evidence_key=evidence_key,
                native_provider_message_id=native_message_id,
                raw=item,
                original_index=index,
                role=role,
                text=text,
                timestamp=timestamp,
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
        evidence_id = evidence.evidence_key
        existing = evidence_by_id.get(evidence_id)
        if existing is None:
            evidence_by_id[evidence_id] = evidence
            continue
        duplicate_ids.add(evidence.native_provider_message_id)
        evidence_by_id[evidence_id] = max((existing, evidence), key=_evidence_richness)
    if duplicate_ids:
        ingest_flags.append(CLAUDE_DUPLICATE_MESSAGE_ID_INGEST_FLAG)

    emitted = [evidence for evidence in evidence_by_id.values() if evidence.has_material]
    emitted_ids = {evidence.evidence_key for evidence in emitted}
    flat_mode = not any(evidence.parent_message_provider_id for evidence in evidence_by_id.values())

    branch_index_by_id: dict[str, int] = {}
    if flat_mode:
        ordered_flat = sorted(
            emitted,
            key=lambda evidence: (
                evidence.explicit_position if evidence.explicit_position is not None else 2**31,
                0 if evidence.timestamp is not None else 1,
                _timestamp_sort_value(evidence.timestamp),
                evidence.evidence_key if evidence.timestamp is not None else "",
                evidence.original_index,
            ),
        )
        position_by_id = {
            evidence.evidence_key: (evidence.explicit_position if evidence.explicit_position is not None else position)
            for position, evidence in enumerate(ordered_flat)
        }
        for evidence in emitted:
            branch_index_by_id[evidence.evidence_key] = evidence.explicit_branch_index or 0
    else:
        depths, cycle_detected = _lineage_depths(evidence_by_id)
        if cycle_detected:
            ingest_flags.append(CLAUDE_LINEAGE_CYCLE_INGEST_FLAG)
        minimum_emitted_depth = min((depths[evidence.evidence_key] for evidence in emitted), default=0)
        position_by_id = {
            evidence.evidence_key: (
                evidence.explicit_position
                if evidence.explicit_position is not None
                else max(0, depths[evidence.evidence_key] - minimum_emitted_depth)
            )
            for evidence in emitted
        }
        siblings_by_parent: dict[str | None, list[_ClaudeMessageEvidence]] = defaultdict(list)
        for evidence in evidence_by_id.values():
            siblings_by_parent[evidence.parent_message_provider_id].append(evidence)
        for siblings in siblings_by_parent.values():
            for rank, evidence in enumerate(sorted(siblings, key=_sibling_sort_key)):
                branch_index_by_id[evidence.evidence_key] = (
                    evidence.explicit_branch_index if evidence.explicit_branch_index is not None else rank
                )

    if flat_mode:
        variant_index_by_id = {
            evidence.evidence_key: (
                evidence.explicit_variant_index
                if evidence.explicit_variant_index is not None
                else branch_index_by_id.get(evidence.evidence_key, 0)
            )
            for evidence in emitted
        }
    else:
        resolved_variants: dict[str, int] = {}
        variant_index_by_id = {
            evidence.evidence_key: _resolve_variant_index(
                evidence.evidence_key,
                evidence_by_id,
                branch_index_by_id,
                resolved_variants,
            )
            for evidence in emitted
        }
    # Safety net regardless of mode: no input tree shape may leave two emitted
    # messages sharing (position, variant_index) -- see
    # `_deduplicate_variant_collisions` docstring.
    _deduplicate_variant_collisions(emitted, position_by_id, variant_index_by_id)
    order_key_by_id = {
        evidence.evidence_key: (
            position_by_id[evidence.evidence_key],
            variant_index_by_id[evidence.evidence_key],
            evidence.evidence_key,
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

    def _evidence_message_type(evidence: _ClaudeMessageEvidence) -> MessageType:
        block_message_type = classify_block_message_type(tuple(block.type for block in evidence.blocks))
        return block_message_type if block_message_type is not None else MessageType.MESSAGE

    messages = [
        ParsedMessage(
            provider_message_id=evidence.native_provider_message_id,
            role=evidence.role,
            text=evidence.text,
            timestamp=evidence.timestamp,
            blocks=evidence.blocks,
            parent_message_provider_id=evidence.parent_message_provider_id,
            position=position_by_id[evidence.evidence_key],
            branch_index=branch_index_by_id.get(evidence.evidence_key, 0),
            variant_index=variant_index_by_id[evidence.evidence_key],
            is_active_path=path_values.get(evidence.evidence_key),
            is_active_leaf=leaf_values.get(evidence.evidence_key),
            model_name=evidence.model_name,
            model_effort=evidence.model_effort,
            duration_ms=evidence.duration_ms,
            delivery_status=evidence.delivery_status,
            end_turn=evidence.end_turn,
            message_type=(message_type := _evidence_message_type(evidence)),
            # polylogue-gzgyl: ordinary claude.ai chat_messages carries no
            # agent/subagent ambiguity -- a plain role=user message here IS
            # positive human evidence, mirroring the Codex/ChatGPT override
            # for the shared classify_material_origin no-fallthrough (#2502).
            material_origin=human_authored_override(
                evidence.role,
                message_type,
                classify_material_origin(
                    role=evidence.role,
                    message_type=message_type,
                    text=evidence.text,
                    block_types=tuple(block.type for block in evidence.blocks),
                ),
            ),
        )
        for evidence in sorted(emitted, key=lambda row: order_key_by_id[row.evidence_key])
    ]

    attachments = _merge_attachment_rows(
        [
            attachment.model_copy(update={"message_position": position_by_id[evidence.evidence_key]})
            for evidence in emitted
            for attachment in evidence.attachments
        ]
    )
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

    for evidence in sorted(emitted, key=lambda row: order_key_by_id[row.evidence_key]):
        session_events.extend(_web_tool_evidence_events(evidence))
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
                    source_message_provider_id=evidence.native_provider_message_id,
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
                    source_message_provider_id=evidence.native_provider_message_id,
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
        active_leaf_message_provider_id=(
            evidence_by_id[normalized_active_leaf].native_provider_message_id
            if normalized_active_leaf is not None
            else None
        ),
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

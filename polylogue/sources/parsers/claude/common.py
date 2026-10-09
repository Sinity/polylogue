"""Shared Claude parser helpers."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableSequence
from dataclasses import dataclass
from typing import Protocol

from polylogue.archive.message.artifacts import classify_block_message_type, classify_material_origin
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, MessageType, Provider, StopReason, WebConstructType
from polylogue.core.hashing import hash_payload
from polylogue.core.message_owner import MessageOwnerCoordinate
from polylogue.core.timestamps import parse_timestamp

from ..base import (
    AdmissionLedger,
    AdmissionUnit,
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
from ..base_models import ParseAccounting, upgrade_chat_export_user_authorship
from ..base_support import tool_result_media_attachments
from .lineage_graph import ClaudeLineageGraph, LineageNode

CLAUDE_MISSING_MESSAGE_ID_INGEST_FLAG = "degraded:claude-missing-message-id"
CLAUDE_DUPLICATE_MESSAGE_ID_INGEST_FLAG = "diagnostic:claude-duplicate-message-id"
CLAUDE_LINEAGE_CYCLE_INGEST_FLAG = "degraded:claude-lineage-cycle"


@dataclass(frozen=True, slots=True)
class ClaudeMessageNormalization:
    """Normalized Claude chat-message evidence shared by export and browser routes."""

    messages: MutableSequence[ParsedMessage]
    attachments: ClaudeAttachmentRows
    active_leaf_message_provider_id: str | None
    models_used: list[str]
    session_events: MutableSequence[ParsedSessionEvent]
    ingest_flags: list[str]
    reported_duration_ms: int | None
    unit_accounting: ParseAccounting


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
    stop_reason: str | None
    thinking_configuration: dict[str, object] | None
    owner_stable_key: str | None

    @property
    def has_material(self) -> bool:
        return bool(self.text or self.blocks or self.attachments)


class ClaudeEvidenceStore(Protocol):
    """Per-record Claude evidence addressed by its 1-based array index."""

    def put(self, evidence: _ClaudeMessageEvidence) -> None: ...

    def raw(self, original_index: int) -> Mapping[str, object]: ...

    def get(
        self,
        original_index: int,
        rebuild: Callable[[dict[str, object]], _ClaudeMessageEvidence],
    ) -> _ClaudeMessageEvidence: ...


class ClaudeAttachmentRows(Protocol):
    """Merged attachment rows keyed by provider attachment id, in first-seen order."""

    def get(self, provider_attachment_id: str) -> ParsedAttachment | None: ...

    def put(self, attachment: ParsedAttachment) -> None: ...

    def descriptor_owners(self) -> Callable[[str, str | None], str | None]:
        """Freeze the current rows' descriptors: the one id named and typed so, if unique."""
        ...

    def __iter__(self) -> Iterator[ParsedAttachment]: ...


class _ResidentEvidence:
    def __init__(self) -> None:
        self._evidence: dict[int, _ClaudeMessageEvidence] = {}

    def put(self, evidence: _ClaudeMessageEvidence) -> None:
        self._evidence[evidence.original_index] = evidence

    def raw(self, original_index: int) -> Mapping[str, object]:
        return self._evidence[original_index].raw

    def get(
        self,
        original_index: int,
        rebuild: Callable[[dict[str, object]], _ClaudeMessageEvidence],
    ) -> _ClaudeMessageEvidence:
        return self._evidence[original_index]


class _ResidentAttachmentRows:
    def __init__(self) -> None:
        self._rows: dict[str, ParsedAttachment] = {}

    def get(self, provider_attachment_id: str) -> ParsedAttachment | None:
        return self._rows.get(provider_attachment_id)

    def put(self, attachment: ParsedAttachment) -> None:
        self._rows[attachment.provider_attachment_id] = attachment

    def descriptor_owners(self) -> Callable[[str, str | None], str | None]:
        owners: dict[tuple[str | None, str | None], list[str]] = defaultdict(list)
        for row in self._rows.values():
            owners[(row.name, row.mime_type)].append(row.provider_attachment_id)

        def owner(name: str, mime_type: str | None) -> str | None:
            ids = owners.get((name, mime_type), [])
            return ids[0] if len(ids) == 1 else None

        return owner

    def __iter__(self) -> Iterator[ParsedAttachment]:
        return iter(list(self._rows.values()))


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


#: ``chat_messages[].stop_reason`` -> ``messages.stop_reason``. The web wire
#: carries Anthropic's own vocabulary plus product-surface additions
#: (``user_canceled``, ``conversation_length_limit``, ``tool_use_limit``,
#: ``error``) that name no :class:`StopReason` member, so those leave the
#: column NULL rather than widening a guess into it.
_CLAUDE_WEB_STOP_REASONS: dict[str, StopReason] = {member.value: member for member in StopReason}


def _message_stop_reason(item: Mapping[str, object]) -> str | None:
    """Return the provider's own terminal-state signal for this turn."""
    raw = item.get("stop_reason") or item.get("stopReason")
    mapped = _CLAUDE_WEB_STOP_REASONS.get(raw) if isinstance(raw, str) else None
    return mapped.value if mapped is not None else None


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
    if any(block.type is BlockType.TOOL_RESULT for block in content_blocks) and all(
        block.type is BlockType.TOOL_RESULT or (block.metadata or {}).get("tool_result_media") is True
        for block in content_blocks
    ):
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
def _claude_content_blocks(content: object, *, admission: AdmissionLedger | None = None) -> list[ParsedContentBlock]:
    if not isinstance(content, list):
        return content_blocks_from_segments(content, admission=admission)

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
            blocks.extend(content_blocks_from_segments([raw_segment], admission=admission))
            continue

        segment = dict(raw_segment)
        provider_type = segment.get("type")
        if isinstance(provider_type, str) and provider_type not in known_segment_types:
            # Keep a non-semantic structural witness for provider block types
            # Polylogue does not yet understand. Raw source evidence remains
            # authoritative for the opaque fields.
            if admission is not None:
                part_ordinal = admission.next_ordinal(AdmissionUnit.PART)
                admission.expect(AdmissionUnit.PART, 1)
                admission.unknown(AdmissionUnit.PART, part_ordinal, provider_type)
                block_ordinal = admission.next_ordinal(AdmissionUnit.BLOCK)
                admission.expect(AdmissionUnit.BLOCK, 1)
                admission.unknown(AdmissionUnit.BLOCK, block_ordinal, provider_type)
            segment_blocks = [
                ParsedContentBlock(
                    type=BlockType.TEXT,
                    metadata={"provider_type": provider_type, "raw_preserved_in_source": True},
                )
            ]
        else:
            segment_blocks = content_blocks_from_segments([raw_segment], admission=admission)
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


def _message_attachments(
    item: Mapping[str, object],
    message_id: str,
    *,
    role: Role,
) -> list[ParsedAttachment]:
    raw_attachments: list[object] = []
    for key in ("attachments", "files"):
        value = item.get(key)
        if isinstance(value, list):
            raw_attachments.extend(value)
    attachments: list[ParsedAttachment] = []
    for meta in raw_attachments:
        attachment = attachment_from_meta(meta, message_id, role=role)
        if attachment is not None:
            attachments.append(attachment)
    return attachments


def _owner_stable_key(
    item: Mapping[str, object],
    *,
    parent_message_provider_id: str | None,
    explicit_position: int | None,
    explicit_branch_index: int | None,
    explicit_variant_index: int | None,
    blocks: list[ParsedContentBlock],
    attachments: list[ParsedAttachment],
) -> str | None:
    """Derive reorder-stable private evidence for duplicate owner anchors."""
    evidence: dict[str, object] = {}
    for field, value in (
        ("parent", parent_message_provider_id),
        ("position", explicit_position),
        ("branch", explicit_branch_index),
        ("variant", explicit_variant_index),
    ):
        if value is not None:
            evidence[field] = value
    for field in ("message_key", "messageKey", "turn_id", "turnId", "sequence_id", "sequenceId"):
        value = _first_identity_field(item, field)
        if value is not None:
            evidence[field] = value
    if (summary := _compaction_summary_material(item)) is not None:
        evidence["compaction_summary"] = summary[0]
    block_ids = sorted(block.tool_id for block in blocks if block.tool_id)
    if block_ids:
        evidence["tool_ids"] = block_ids
    # Attachment identity is deliberately excluded from owner evidence. A
    # Claude export can move an idless file between same-role, same-timestamp
    # turns while retaining the file id. Including that id makes the file's
    # stable key follow the file instead of the owning turn, so the session
    # hash can remain unchanged and re-ingest can skip the reassignment. The
    # parser-provided position/variant coordinate is the independent fallback.
    if not evidence:
        return None
    return f"claude-owner-evidence:{hash_payload(evidence)}"


def _canonical_record(item: Mapping[str, object]) -> str:
    return json.dumps(dict(item), sort_keys=True, separators=(",", ":"), default=str)


def _evidence_richness_score(evidence: _ClaudeMessageEvidence) -> tuple[int, float]:
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
    return score, updated


def _timestamp_sort_value(timestamp: str | None) -> float:
    if timestamp is None:
        return float("inf")
    parsed = parse_timestamp(timestamp)
    return parsed.timestamp() if parsed is not None else float("inf")


def _merged_attachment(existing: ParsedAttachment, candidate: ParsedAttachment) -> ParsedAttachment:
    preferred = candidate if candidate.inline_bytes is not None and existing.inline_bytes is None else existing
    other = existing if preferred is candidate else candidate
    return preferred.model_copy(
        update={
            "message_provider_id": preferred.message_provider_id or other.message_provider_id,
            "message_position": preferred.message_position
            if preferred.message_position is not None
            else other.message_position,
            "message_variant_index": preferred.message_variant_index
            if preferred.message_variant_index is not None
            else other.message_variant_index,
            "owner_coordinate": preferred.owner_coordinate or other.owner_coordinate,
            "name": preferred.name or other.name,
            "mime_type": preferred.mime_type or other.mime_type,
            "size_bytes": preferred.size_bytes if preferred.size_bytes is not None else other.size_bytes,
            "provider_file_id": preferred.provider_file_id or other.provider_file_id,
            "provider_drive_id": preferred.provider_drive_id or other.provider_drive_id,
            "direction": preferred.direction or other.direction,
            "producer_ref": preferred.producer_ref or other.producer_ref,
            "source_url": preferred.source_url or other.source_url,
        }
    )


def merge_attachment_row(rows: ClaudeAttachmentRows, candidate: ParsedAttachment) -> None:
    """Fold one attachment into rows that share its provider attachment id."""
    existing = rows.get(candidate.provider_attachment_id)
    rows.put(candidate if existing is None else _merged_attachment(existing, candidate))


def resident_attachment_rows() -> ClaudeAttachmentRows:
    return _ResidentAttachmentRows()


#: Identity namespace for a Claude web tool call the provider left unnamed.
#: ``toolu_``-prefixed provider ids can never collide with it.
_STRUCTURAL_TOOL_ID_PREFIX = "structural:claude-web"


def _pair_idless_tool_blocks(blocks: list[ParsedContentBlock], *, message_key: str) -> list[ParsedContentBlock]:
    """Give one deterministic id to each id-less tool call and its answer.

    The Claude web transcript emits a tool_use segment followed by its
    tool_result segment inside one message's ``content`` array and puts no id
    on either -- the pair is expressed by position and tool name alone. Every
    downstream relation joins a call to its result by ``tool_id``, so leaving
    both NULL discards a link the source states unambiguously. The id is
    derived from the owning message and the call's ordinal within it, so it is
    stable across re-ingest; an unanswered call keeps its own id and stays
    ``no_result``, and a result with no preceding unanswered call keeps no id
    rather than borrowing one.
    """
    pending: list[int] = []
    assigned: dict[int, str] = {}
    ordinal = 0
    for index, block in enumerate(blocks):
        if block.tool_id:
            continue
        if block.type is BlockType.TOOL_USE:
            assigned[index] = f"{_STRUCTURAL_TOOL_ID_PREFIX}:{message_key}:{ordinal}"
            ordinal += 1
            pending.append(index)
        elif block.type is BlockType.TOOL_RESULT:
            match = next(
                (
                    candidate
                    for candidate in reversed(pending)
                    if not (block.tool_name and blocks[candidate].tool_name)
                    or block.tool_name == blocks[candidate].tool_name
                ),
                None,
            )
            if match is None:
                continue
            pending.remove(match)
            assigned[index] = assigned[match]
    if not assigned:
        return blocks
    return [
        block.model_copy(update={"tool_id": assigned[index]}) if index in assigned else block
        for index, block in enumerate(blocks)
    ]


def _occurrence_base(evidence_key: str) -> str:
    """The identity an occurrence-suffixed evidence key repeats."""
    return evidence_key.split(":occurrence:", 1)[0]


def _compaction_summary_material(
    item: Mapping[str, object],
) -> tuple[dict[str, object], str | None, str | None] | None:
    summary_blocks = item.get("compaction_summary")
    if not isinstance(summary_blocks, list):
        return None
    texts: list[str] = []
    start_timestamp: str | None = None
    stop_timestamp: str | None = None
    for block in summary_blocks:
        if not isinstance(block, Mapping):
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            texts.append(text)
        start = block.get("start_timestamp")
        stop = block.get("stop_timestamp")
        if start_timestamp is None and isinstance(start, str) and start:
            start_timestamp = start
        if isinstance(stop, str) and stop:
            stop_timestamp = stop
    if not texts:
        return None
    payload: dict[str, object] = {"summary": "\n\n".join(texts)}
    if start_timestamp is not None:
        payload["start_timestamp"] = start_timestamp
    if stop_timestamp is not None:
        payload["stop_timestamp"] = stop_timestamp
    return payload, start_timestamp, stop_timestamp


def _compaction_summary_event(evidence: _ClaudeMessageEvidence) -> ParsedSessionEvent | None:
    """The ``claude_ai_compaction_summary`` event of one message, when it carries one.

    When claude.ai compacts a conversation it stores the summary it carries
    forward on the message where compaction took effect (``compaction_summary``:
    text blocks with start/stop timestamps). The text is kept, keyed to that
    message. It is deliberately not a ``compaction`` event: those carry
    boundaries and a materialized summary message that effective-context reads
    apply, and placing a summary message into claude.ai's branched message tree
    (variant and attachment-owner coordinates) is not done here.

    One event per emitted message, in the normalizer's canonical message
    order: a repeated native id is one message per occurrence and keeps its
    own summary, and an export listing the same messages in another array
    order yields the same events in the same order.
    """
    material = _compaction_summary_material(evidence.raw)
    if material is None:
        return None
    payload, start_timestamp, stop_timestamp = material
    return ParsedSessionEvent(
        event_type="claude_ai_compaction_summary",
        timestamp=stop_timestamp or start_timestamp or evidence.timestamp or evidence.updated_at,
        source_message_provider_id=evidence.native_provider_message_id or None,
        payload=payload,
    )


def _message_evidence(
    item: dict[str, object],
    index: int,
    *,
    evidence_key_for: Callable[[str], str],
    session_model: str | None,
    session_effort: str | None,
    admission: AdmissionLedger | None = None,
) -> _ClaudeMessageEvidence:
    native_message_id = (
        _first_identity_field(
            item,
            "uuid",
            "id",
            "message_id",
            "messageId",
            "provider_message_id",
        )
        or ""
    )
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
    content_blocks = _claude_content_blocks(raw_content, admission=admission)
    role = reclassify_tool_result_envelope(role, content_blocks)

    raw_created_at = item.get("created_at") or item.get("create_time") or item.get("timestamp")
    raw_updated_at = item.get("updated_at") or item.get("update_time") or item.get("edited_at")
    timestamp = normalize_timestamp(raw_created_at if isinstance(raw_created_at, (int, float, str)) else None)
    evidence_key = evidence_key_for(
        native_message_id
        or synthetic_message_id(
            role=role,
            text=text,
            timestamp=timestamp,
            kind="claude-web-evidence",
        )
    )
    content_blocks = _pair_idless_tool_blocks(content_blocks, message_key=evidence_key)
    attachments = _message_attachments(item, native_message_id, role=role)
    attachments.extend(tool_result_media_attachments(raw_content, native_message_id, role=role))
    parent_message_provider_id = _message_parent_id(item)
    explicit_position = _first_non_negative_int_field(item, "position")
    explicit_branch_index = _first_non_negative_int_field(item, "branch_index", "branchIndex")
    explicit_variant_index = _first_non_negative_int_field(item, "variant_index", "variantIndex")
    return _ClaudeMessageEvidence(
        evidence_key=evidence_key,
        native_provider_message_id=native_message_id,
        raw=item,
        original_index=index,
        role=role,
        text=text,
        timestamp=timestamp,
        updated_at=normalize_timestamp(raw_updated_at if isinstance(raw_updated_at, (int, float, str)) else None),
        blocks=content_blocks,
        attachments=attachments,
        parent_message_provider_id=parent_message_provider_id,
        explicit_position=explicit_position,
        explicit_branch_index=explicit_branch_index,
        explicit_variant_index=explicit_variant_index,
        explicit_is_active_path=_first_bool_field(item, "is_active_path", "isActivePath", "active_path"),
        explicit_is_active_leaf=_first_bool_field(item, "is_active_leaf", "isActiveLeaf", "active_leaf"),
        model_name=_message_model_name(item) or session_model,
        model_effort=_message_model_effort(item) or session_effort,
        duration_ms=_message_duration_ms(item),
        delivery_status=_message_delivery_status(item),
        end_turn=_message_end_turn(item),
        stop_reason=_message_stop_reason(item),
        thinking_configuration=_thinking_configuration(item),
        owner_stable_key=_owner_stable_key(
            item,
            parent_message_provider_id=parent_message_provider_id,
            explicit_position=explicit_position,
            explicit_branch_index=explicit_branch_index,
            explicit_variant_index=explicit_variant_index,
            blocks=content_blocks,
            attachments=attachments,
        ),
    )


def _message_update_event(evidence: _ClaudeMessageEvidence) -> ParsedSessionEvent | None:
    if not evidence.updated_at or evidence.updated_at == evidence.timestamp:
        return None
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
    return ParsedSessionEvent(
        event_type=(
            "message_revision"
            if revision_id or explicitly_edited or has_edited_timestamp
            else "provider_message_update"
        ),
        timestamp=evidence.updated_at,
        source_message_provider_id=evidence.native_provider_message_id,
        payload=update_payload,
    )


def _lineage_node(evidence: _ClaudeMessageEvidence) -> LineageNode:
    score, updated_score = _evidence_richness_score(evidence)
    return LineageNode(
        evidence_key=evidence.evidence_key,
        native_id=evidence.native_provider_message_id,
        original_index=evidence.original_index,
        timestamp=evidence.timestamp,
        updated_at=evidence.updated_at,
        parent=evidence.parent_message_provider_id,
        explicit_position=evidence.explicit_position,
        explicit_branch=evidence.explicit_branch_index,
        explicit_variant=evidence.explicit_variant_index,
        explicit_active_path=evidence.explicit_is_active_path,
        explicit_active_leaf=evidence.explicit_is_active_leaf,
        has_material=evidence.has_material,
        has_attachments=bool(evidence.attachments),
        has_compaction_summary=isinstance(evidence.raw.get("compaction_summary"), list),
        score=score,
        updated_score=updated_score,
        ts_sort=_timestamp_sort_value(evidence.timestamp),
        updated_sort=_timestamp_sort_value(evidence.updated_at),
    )


def _add_summary(
    graph: ClaudeLineageGraph, evidence_key: str, original_index: int, position: int, event: ParsedSessionEvent
) -> None:
    graph.add_summary(
        _occurrence_base(evidence_key),
        evidence_key,
        original_index,
        position,
        json.dumps(event.payload, sort_keys=True),
        event.timestamp,
    )


def normalize_chat_messages(
    chat_messages: Iterable[object],
    *,
    session_model: str | None = None,
    session_effort: str | None = None,
    session_thinking_configuration: dict[str, object] | None = None,
    session_created_at: str | None = None,
    session_updated_at: str | None = None,
    active_leaf_message_provider_id: str | None = None,
    evidence_store: ClaudeEvidenceStore | None = None,
    messages: MutableSequence[ParsedMessage] | None = None,
    session_events: MutableSequence[ParsedSessionEvent] | None = None,
    attachment_rows: ClaudeAttachmentRows | None = None,
    graph_connection: sqlite3.Connection | None = None,
) -> ClaudeMessageNormalization:
    """Normalize Claude web messages without splitting strict and loose shapes.

    Native IDs and parent pointers are authoritative. Array order is used only
    when flat records lack lineage, explicit positions, and usable timestamps.

    ``chat_messages`` is read once. Each record's content goes to
    ``evidence_store`` and its identity and ranking fields to a lineage graph
    in ``graph_connection``; messages, events and merged attachments are
    written in canonical order to the supplied sequences. The defaults are
    resident (an in-memory graph), so the object-returning parser and a
    disk-backed preparation share one normalization.
    """

    store: ClaudeEvidenceStore = evidence_store if evidence_store is not None else _ResidentEvidence()
    message_rows: MutableSequence[ParsedMessage] = messages if messages is not None else []
    event_rows: MutableSequence[ParsedSessionEvent] = session_events if session_events is not None else []
    attachments: ClaudeAttachmentRows = attachment_rows if attachment_rows is not None else _ResidentAttachmentRows()
    graph = ClaudeLineageGraph(graph_connection)
    try:
        return _normalize_through_graph(
            chat_messages,
            graph,
            store=store,
            message_rows=message_rows,
            event_rows=event_rows,
            attachments=attachments,
            session_model=session_model,
            session_effort=session_effort,
            session_thinking_configuration=session_thinking_configuration,
            session_created_at=session_created_at,
            session_updated_at=session_updated_at,
            active_leaf_message_provider_id=active_leaf_message_provider_id,
        )
    finally:
        graph.close()


def _normalize_through_graph(
    chat_messages: Iterable[object],
    graph: ClaudeLineageGraph,
    *,
    store: ClaudeEvidenceStore,
    message_rows: MutableSequence[ParsedMessage],
    event_rows: MutableSequence[ParsedSessionEvent],
    attachments: ClaudeAttachmentRows,
    session_model: str | None,
    session_effort: str | None,
    session_thinking_configuration: dict[str, object] | None,
    session_created_at: str | None,
    session_updated_at: str | None,
    active_leaf_message_provider_id: str | None,
) -> ClaudeMessageNormalization:
    admission = AdmissionLedger()
    for index, raw_item in enumerate(chat_messages, start=1):
        if not isinstance(raw_item, Mapping):
            continue
        evidence = _message_evidence(
            dict(raw_item),
            index,
            evidence_key_for=graph.occurrence_key,
            session_model=session_model,
            session_effort=session_effort,
            admission=admission,
        )
        store.put(evidence)
        graph.observe(
            _lineage_node(evidence),
            lambda candidate, retained: (
                _canonical_record(store.raw(candidate)) > _canonical_record(store.raw(retained))
            ),
        )

    def load(original_index: int, evidence_key: str) -> _ClaudeMessageEvidence:
        return store.get(
            original_index,
            lambda item: _message_evidence(
                item,
                original_index,
                evidence_key_for=lambda _base: evidence_key,
                session_model=session_model,
                session_effort=session_effort,
            ),
        )

    ingest_flags: list[str] = []
    if graph.missing_native_id:
        ingest_flags.append(CLAUDE_MISSING_MESSAGE_ID_INGEST_FLAG)
    duplicate_ids = graph.duplicate_native_ids()
    if duplicate_ids:
        ingest_flags.append(CLAUDE_DUPLICATE_MESSAGE_ID_INGEST_FLAG)
    cycle_detected, leaf_id, walked_path, all_active = graph.resolve(active_leaf_message_provider_id)
    if cycle_detected:
        ingest_flags.append(CLAUDE_LINEAGE_CYCLE_INGEST_FLAG)

    if session_model or session_effort or session_thinking_configuration:
        configuration_payload: dict[str, object] = {}
        if session_model:
            configuration_payload["model"] = session_model
        if session_effort:
            configuration_payload["effort"] = session_effort
        if session_thinking_configuration:
            configuration_payload["thinking"] = session_thinking_configuration
        event_rows.append(
            ParsedSessionEvent(
                event_type="model_configuration",
                timestamp=session_updated_at or session_created_at,
                payload=configuration_payload,
            )
        )

    models_used: list[str] = [session_model] if session_model else []
    duration_total: int | None = None
    for node in graph.emitted():
        evidence = load(node.original_index, node.evidence_key)
        block_types = tuple(block.type for block in evidence.blocks)
        block_message_type = classify_block_message_type(block_types)
        message_type = block_message_type if block_message_type is not None else MessageType.MESSAGE
        if all_active:
            is_active_path: bool | None = True
        elif walked_path:
            is_active_path = node.on_path
        else:
            is_active_path = node.explicit_active_path
        message_rows.append(
            # The session validator applies this to resident rows; a disk
            # sink is written before its session exists.
            upgrade_chat_export_user_authorship(
                Provider.CLAUDE_AI,
                ParsedMessage(
                    provider_message_id=evidence.native_provider_message_id,
                    role=evidence.role,
                    text=evidence.text,
                    timestamp=evidence.timestamp,
                    blocks=evidence.blocks,
                    parent_message_provider_id=evidence.parent_message_provider_id,
                    owner_coordinate=MessageOwnerCoordinate(
                        stable_key=evidence.owner_stable_key,
                        position=node.position,
                        variant_index=node.variant_index,
                    ),
                    position=node.position,
                    branch_index=node.branch_index,
                    variant_index=node.variant_index,
                    is_active_path=is_active_path,
                    is_active_leaf=(node.evidence_key == leaf_id if leaf_id is not None else node.explicit_active_leaf),
                    model_name=evidence.model_name,
                    model_effort=evidence.model_effort,
                    duration_ms=evidence.duration_ms,
                    delivery_status=evidence.delivery_status,
                    end_turn=evidence.end_turn,
                    stop_reason=evidence.stop_reason,
                    message_type=message_type,
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
                            block_types=block_types,
                        ),
                    ),
                ),
            )
        )
        if evidence.model_name and evidence.model_name not in models_used:
            models_used.append(evidence.model_name)
        if evidence.duration_ms is not None:
            duration_total = (duration_total or 0) + evidence.duration_ms

        event_owner = MessageOwnerCoordinate(
            stable_key=evidence.owner_stable_key,
            position=node.position,
            variant_index=node.variant_index,
        )
        event_rows.extend(
            event.model_copy(update={"owner_coordinate": event_owner}) for event in _web_tool_evidence_events(evidence)
        )
        if (compaction_summary := _compaction_summary_event(evidence)) is not None:
            _add_summary(graph, node.evidence_key, node.original_index, node.position, compaction_summary)
        if evidence.thinking_configuration:
            payload: dict[str, object] = {"thinking": evidence.thinking_configuration}
            if evidence.model_name:
                payload["model"] = evidence.model_name
            if evidence.model_effort:
                payload["effort"] = evidence.model_effort
            event_rows.append(
                ParsedSessionEvent(
                    event_type="model_configuration",
                    timestamp=evidence.updated_at or evidence.timestamp,
                    source_message_provider_id=evidence.native_provider_message_id,
                    owner_coordinate=event_owner,
                    payload=payload,
                )
            )
        if (update_event := _message_update_event(evidence)) is not None:
            event_rows.append(update_event.model_copy(update={"owner_coordinate": event_owner}))
    # Compaction summaries follow the messages. The occurrences of one
    # repeated identity (a native id, or an ID-less record's synthetic key)
    # get their suffixes, and so their positions and variants, in array
    # order, so they are grouped at the identity's first position and ordered
    # by their own content. A summary on a record with no other material, and
    # so no message of its own, comes after them.
    for evidence_key, original_index in graph.unemitted_summaries():
        if (event := _compaction_summary_event(load(original_index, evidence_key))) is not None:
            _add_summary(graph, evidence_key, original_index, 2**31, event)
    for evidence_key, original_index in graph.ordered_summaries():
        evidence = load(original_index, evidence_key)
        summary = _compaction_summary_event(evidence)
        assert summary is not None
        coordinate = graph.emitted_coordinate(evidence_key)
        if coordinate is not None:
            summary = summary.model_copy(
                update={
                    "owner_coordinate": MessageOwnerCoordinate(
                        stable_key=evidence.owner_stable_key,
                        position=coordinate[0],
                        variant_index=coordinate[1],
                    )
                }
            )
        event_rows.append(summary)

    if duplicate_ids:
        event_rows.append(
            ParsedSessionEvent(
                event_type="normalization_diagnostic",
                timestamp=session_updated_at or session_created_at,
                payload={
                    "diagnostic": "duplicate_message_ids",
                    "provider_message_ids": duplicate_ids,
                    "resolution": "richest_structured_record",
                },
            )
        )

    # Attachment merging keeps first-seen order over records, not canonical
    # message order, so it is a separate pass over the records that carry any.
    for node in graph.emitted(first_seen_with_attachments=True):
        evidence = load(node.original_index, node.evidence_key)
        for attachment in evidence.attachments:
            merge_attachment_row(
                attachments,
                attachment.model_copy(
                    update={
                        "message_position": node.position,
                        "message_variant_index": node.variant_index,
                        "owner_coordinate": MessageOwnerCoordinate(
                            stable_key=evidence.owner_stable_key,
                            position=node.position,
                            variant_index=node.variant_index,
                        ),
                    }
                ),
            )

    return ClaudeMessageNormalization(
        messages=message_rows,
        attachments=attachments,
        active_leaf_message_provider_id=(
            graph.native_id(leaf_id) if leaf_id is not None and graph.is_emitted(leaf_id) else None
        ),
        models_used=models_used,
        session_events=event_rows,
        ingest_flags=ingest_flags,
        reported_duration_ms=duration_total,
        unit_accounting=admission.close(),
    )


def extract_messages_from_chat_messages(
    chat_messages: list[object],
) -> tuple[list[ParsedMessage], list[ParsedAttachment]]:
    normalized = normalize_chat_messages(chat_messages)
    return list(normalized.messages), list(normalized.attachments)


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
    "ClaudeAttachmentRows",
    "ClaudeEvidenceStore",
    "ClaudeMessageNormalization",
    "extract_message_text",
    "extract_messages_from_chat_messages",
    "extract_text_from_segments",
    "merge_attachment_row",
    "normalize_chat_messages",
    "normalize_timestamp",
    "resident_attachment_rows",
]

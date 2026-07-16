from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

from polylogue.archive.message.artifacts import classify_material_origin, classify_text_message_type
from polylogue.archive.message.roles import Role
from polylogue.archive.message.types import MessageType
from polylogue.core.enums import BlockType, Provider, SessionKind, WebConstructType
from polylogue.core.timestamps import parse_timestamp

from .base import (
    ParsedAttachment,
    ParsedContentBlock,
    ParsedMessage,
    ParsedSession,
    ParsedSessionEvent,
    ParsedWebConstruct,
)

SHARED_CONVERSATION_INDEX_INGEST_FLAG = "capture:chatgpt-shared-index-shell"


@dataclass(frozen=True)
class _GenerationTiming:
    message_provider_id: str
    elapsed_duration_ms: int
    started_at_ms: int | None
    ended_at_ms: int | None
    event_timestamp: str | None
    fidelity: str


def _coerce_float(value: object) -> float | None:
    # Exclude bool explicitly (bool is a subclass of int)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            pass
        parsed = parse_timestamp(value)
        if parsed is not None:
            return parsed.timestamp()
    return None


def _non_negative_finite_float(value: object) -> float | None:
    parsed = _coerce_float(value)
    if parsed is None or not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def _generation_branch_key(mapping: Mapping[str, object], node_id: str) -> str:
    """Return the first assistant-side node below the nearest user ancestor.

    ChatGPT repeats run-wide reasoning metadata across thought, tool, recap,
    and final-answer nodes. Grouping by this branch root deduplicates those
    copies while preserving regenerated alternatives beneath the same user
    message as distinct generations.
    """

    current_id = node_id
    seen: set[str] = set()
    while current_id not in seen:
        seen.add(current_id)
        current = mapping.get(current_id)
        if not isinstance(current, Mapping):
            break
        parent_raw = current.get("parent")
        if not isinstance(parent_raw, str) or not parent_raw:
            break
        parent = mapping.get(parent_raw)
        if not isinstance(parent, Mapping):
            break
        parent_message = parent.get("message")
        parent_author = parent_message.get("author") if isinstance(parent_message, Mapping) else None
        parent_role = parent_author.get("role") if isinstance(parent_author, Mapping) else None
        if parent_role == "user":
            return current_id
        current_id = parent_raw
    return current_id


def _extract_generation_timings(mapping: Mapping[str, object]) -> list[_GenerationTiming]:
    """Select one authoritative lifecycle timing per ChatGPT generation.

    Native conversation payloads commonly copy ``reasoning_start_time`` and
    ``finished_duration_sec`` onto many nodes. A complete ``reasoning_recap``
    is preferred over those partial copies; otherwise the best complete node
    on the same assistant branch owns the timing. Provider-finished duration
    is authoritative, with a valid start/end delta as the derived fallback.
    """

    candidates: dict[str, list[tuple[tuple[int, int, int, int], _GenerationTiming]]] = {}
    for position, (node_id, raw_node) in enumerate(mapping.items()):
        if not isinstance(raw_node, Mapping):
            continue
        raw_message = raw_node.get("message")
        if not isinstance(raw_message, Mapping):
            continue
        raw_metadata = raw_message.get("metadata")
        if not isinstance(raw_metadata, Mapping):
            continue

        start_sec = _non_negative_finite_float(raw_metadata.get("reasoning_start_time"))
        end_sec = _non_negative_finite_float(raw_metadata.get("reasoning_end_time"))
        finished_sec = _non_negative_finite_float(raw_metadata.get("finished_duration_sec"))
        if finished_sec is not None:
            elapsed_ms = round(finished_sec * 1000)
            fidelity = "exact"
        elif start_sec is not None and end_sec is not None and end_sec >= start_sec:
            elapsed_ms = round((end_sec - start_sec) * 1000)
            fidelity = "derived"
        else:
            continue

        content = raw_message.get("content")
        content_type = content.get("content_type") if isinstance(content, Mapping) else None
        message_id_raw = raw_message.get("id") or raw_node.get("id") or node_id
        message_id = str(message_id_raw)
        timing = _GenerationTiming(
            message_provider_id=message_id,
            elapsed_duration_ms=elapsed_ms,
            started_at_ms=round(start_sec * 1000) if start_sec is not None else None,
            ended_at_ms=round(end_sec * 1000) if end_sec is not None else None,
            event_timestamp=str(end_sec) if end_sec is not None else None,
            fidelity=fidelity,
        )
        score = (
            int(content_type == "reasoning_recap"),
            int(start_sec is not None and end_sec is not None),
            int(raw_message.get("end_turn") is True),
            position,
        )
        branch_key = _generation_branch_key(mapping, str(node_id))
        candidates.setdefault(branch_key, []).append((score, timing))

    return [max(branch_candidates, key=lambda item: item[0])[1] for branch_candidates in candidates.values()]


def _string_value(payload: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    return None


def _int_value(payload: Mapping[str, object], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                continue
    return None


def _iter_mapping_items(value: object) -> list[Mapping[str, object]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _construct_from_reference(
    item: Mapping[str, object],
    *,
    construct_type: WebConstructType,
    provider_key: str,
    rank: int | None = None,
    group_id: str | None = None,
    group_title: str | None = None,
) -> ParsedWebConstruct:
    # File-search citation rows nest their source identity one or two levels
    # down (item.metadata carries the file name/id/source, item.metadata.extra
    # carries cited_message_id, library_file_id, source_url, ...). Consult the
    # nested layers whenever the top level misses — otherwise a file citation
    # keeps only its answer-side anchor span and loses which document it cites.
    metadata = item.get("metadata")
    nested = metadata if isinstance(metadata, Mapping) else {}
    extra_value = nested.get("extra")
    extra = extra_value if isinstance(extra_value, Mapping) else {}

    def pick(*keys: str) -> str | None:
        return _string_value(item, *keys) or _string_value(nested, *keys) or _string_value(extra, *keys)

    return ParsedWebConstruct(
        construct_type=construct_type,
        provider_key=provider_key,
        title=pick("title", "name", "source_name", "source_label"),
        url=pick("url", "link", "source_url", "cloud_doc_url"),
        text=pick("snippet", "text", "content", "description", "quote"),
        source_id=pick("id", "source_id", "ref_id", "attribution_id", "textdoc_id", "library_file_id"),
        group_id=group_id,
        group_title=group_title,
        asset_pointer=pick("asset_pointer"),
        mime_type=pick("mime_type", "media_type"),
        rank=rank if rank is not None else _int_value(item, "rank", "index"),
        start_index=_int_value(item, "start_index", "start_idx", "start_ix", "start"),
        end_index=_int_value(item, "end_index", "end_idx", "end_ix", "end"),
    )


def _constructs_from_chatgpt_metadata(msg_metadata: object) -> list[ParsedWebConstruct]:
    if not isinstance(msg_metadata, Mapping):
        return []
    constructs: list[ParsedWebConstruct] = []
    for item in _iter_mapping_items(msg_metadata.get("canvas")):
        constructs.append(
            ParsedWebConstruct(
                construct_type=WebConstructType.CANVAS,
                provider_key="canvas",
                title=_string_value(item, "title", "name"),
                text=_string_value(item, "text", "content"),
                source_id=_string_value(item, "id", "canvas_id", "textdoc_id"),
                status=_string_value(item, "status"),
            )
        )
    for provider_key in ("content_references", "citations", "_cite_metadata"):
        value = msg_metadata.get(provider_key)
        for rank, item in enumerate(_iter_mapping_items(value)):
            constructs.append(
                _construct_from_reference(
                    item,
                    construct_type=WebConstructType.CONTENT_REFERENCE,
                    provider_key=provider_key,
                    rank=rank,
                )
            )
    search_queries = msg_metadata.get("search_queries")
    if isinstance(search_queries, list):
        for rank, item in enumerate(search_queries):
            if isinstance(item, str) and item:
                constructs.append(
                    ParsedWebConstruct(
                        construct_type=WebConstructType.SEARCH_QUERY,
                        provider_key="search_queries",
                        query=item,
                        rank=rank,
                    )
                )
            elif isinstance(item, Mapping):
                constructs.append(
                    ParsedWebConstruct(
                        construct_type=WebConstructType.SEARCH_QUERY,
                        provider_key="search_queries",
                        query=_string_value(item, "query", "text"),
                        title=_string_value(item, "title"),
                        rank=rank,
                    )
                )
    for group_rank, group in enumerate(_iter_mapping_items(msg_metadata.get("search_result_groups"))):
        group_id = _string_value(group, "id", "group_id") or str(group_rank)
        group_title = _string_value(group, "title", "name", "query")
        candidates = (
            group.get("results") or group.get("items") or group.get("search_results") or group.get("sources") or []
        )
        for rank, item in enumerate(_iter_mapping_items(candidates)):
            constructs.append(
                _construct_from_reference(
                    item,
                    construct_type=WebConstructType.SEARCH_RESULT,
                    provider_key="search_result_groups",
                    rank=rank,
                    group_id=group_id,
                    group_title=group_title,
                )
            )
    for rank, item in enumerate(_iter_mapping_items(msg_metadata.get("selected_sources"))):
        constructs.append(
            _construct_from_reference(
                item,
                construct_type=WebConstructType.SELECTED_SOURCE,
                provider_key="selected_sources",
                rank=rank,
            )
        )
    for rank, item in enumerate(_iter_mapping_items(msg_metadata.get("image_results"))):
        constructs.append(
            _construct_from_reference(
                item,
                construct_type=WebConstructType.IMAGE_RESULT,
                provider_key="image_results",
                rank=rank,
            )
        )
    async_task_type = _string_value(msg_metadata, "async_task_type")
    async_task_id = _string_value(msg_metadata, "async_task_id")
    async_task_title = _string_value(msg_metadata, "async_task_title")
    if async_task_type or async_task_id or async_task_title:
        constructs.append(
            ParsedWebConstruct(
                construct_type=WebConstructType.ASYNC_TASK,
                provider_key="async_task",
                title=async_task_title,
                task_id=async_task_id,
                task_type=async_task_type,
            )
        )
    for item in _iter_mapping_items(msg_metadata.get("aggregate_result")):
        constructs.append(
            ParsedWebConstruct(
                construct_type=WebConstructType.ASYNC_TASK,
                provider_key="aggregate_result",
                title=_string_value(item, "title"),
                text=_string_value(item, "output", "text", "stdout", "stderr"),
                status=_string_value(item, "status", "exit_code"),
            )
        )
    return constructs


def _active_path_node_ids(mapping: Mapping[str, object], current_node: str | None) -> list[str]:
    """Return the active ChatGPT path from root to ``current_node``.

    ChatGPT exports preserve regenerated and edited branches in ``mapping`` and
    use ``current_node`` only to identify the leaf the user last saw. The v1
    parser contract keeps every branch and carries the active path explicitly
    instead of using it as a lossy filter (#1743).
    """
    if current_node and current_node in mapping:
        path: list[str] = []
        seen: set[str] = set()
        node_id: str | None = current_node
        while node_id is not None and node_id in mapping and node_id not in seen:
            seen.add(node_id)
            path.append(node_id)
            node = mapping[node_id]
            node_id = node.get("parent") if isinstance(node, dict) else None
        path.reverse()
        return path

    return []


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if value >= 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


# ChatGPT embeds inline citation anchors in assistant text as private-use
# unicode spans: U+E200 opens, U+E202 separates reference tokens, U+E201
# closes (e.g. "\ue200filecite\ue202turn3file14\ue202L180-L293\ue201").
# The span carries no human-readable text -- the resolvable citation rows
# live in message metadata (`citations`/`content_references`) and are
# preserved as web constructs. The raw markers otherwise leak invisible
# glyphs into search text and rendered transcripts; the untouched original
# remains in the source-tier raw payload.
_CITATION_MARKER_RE = re.compile("\ue200.*?\ue201|[\ue200\ue201\ue202]")
_CITATION_MARKER_SPAN_RE = re.compile("\ue200(.*?)\ue201")


def _strip_citation_markers(text: str) -> str:
    return _CITATION_MARKER_RE.sub("", text)


_SANDBOX_FILE_RE = re.compile(r"sandbox:(/mnt/data/[^\s)\]\"'>]+)")


def _sandbox_file_paths(text: str) -> list[str]:
    """Ordered, deduplicated ``/mnt/data`` paths linked in assistant text.

    Trailing prose punctuation is stripped so ``(sandbox:/mnt/data/kit.zip).``
    yields ``/mnt/data/kit.zip``. Directory links keep their trailing slash in
    the returned path.
    """

    seen: dict[str, None] = {}
    for match in _SANDBOX_FILE_RE.finditer(text):
        path = match.group(1).rstrip(".,;:!?*`")
        if path != "/mnt/data/":
            seen.setdefault(path)
    return list(seen)


def _extract_content_text(content: Mapping[str, object]) -> str:
    """Extract message text from a ChatGPT content block.

    Handles the common ``parts`` array (strings and structured dicts carrying
    ``text``) and falls back to non-``parts`` content shapes — ``code`` and
    ``execution_output`` carry a top-level ``text``, browsing display carries a
    ``result``. Without this fallback those messages have empty text and are
    dropped entirely (#1744).
    """
    parts = content.get("parts")
    if isinstance(parts, list):
        text_parts: list[str] = []
        for part in parts:
            if isinstance(part, str) and part:
                text_parts.append(part)
            elif isinstance(part, dict):
                # Extract text from structured parts (e.g. tether_quote dicts)
                t = part.get("text")
                if isinstance(t, str) and t:
                    text_parts.append(t)
                # Skip image_asset_pointer and other non-text dicts
        if text_parts:
            return "\n".join(text_parts)
    # Non-parts content shapes: code / execution_output carry top-level text;
    # browsing display carries a result string.
    top_text = content.get("text")
    if isinstance(top_text, str) and top_text:
        return top_text
    result = content.get("result")
    if isinstance(result, str) and result:
        return result
    return ""


def extract_messages_from_mapping(
    mapping: Mapping[str, object],
    current_node: str | None = None,
) -> tuple[list[ParsedMessage], list[ParsedAttachment]]:
    entries: list[tuple[float | None, int, ParsedMessage]] = []
    attachments: list[ParsedAttachment] = []
    active_path_ids = _active_path_node_ids(mapping, current_node)
    active_path_id_set = set(active_path_ids)
    emitted_by_node_id: dict[str, str] = {}
    for idx, node_id in enumerate(mapping.keys(), start=1):
        node = mapping.get(node_id)
        if not isinstance(node, dict):
            continue
        msg = node.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts") or []
        raw_text = _extract_content_text(content)
        text = _strip_citation_markers(raw_text)
        # Role is required - skip messages without one
        author = msg.get("author")
        raw_role = author.get("role") if isinstance(author, dict) else None
        if not raw_role or not isinstance(raw_role, str):
            continue
        role = Role.normalize(str(raw_role))
        timestamp = msg.get("create_time")
        msg_id = msg.get("id") or node.get("id") or ""
        if not msg_id:
            msg_id = f"msg-{idx}"

        # Extract parent message reference and calculate branch index
        parent_id = node.get("parent")
        parent_message_provider_id = str(parent_id) if parent_id else None
        branch_index = 0

        # Calculate branch_index from parent's children array position
        if parent_message_provider_id:
            parent_node = mapping.get(str(parent_id))
            if isinstance(parent_node, dict):
                children = parent_node.get("children")
                if isinstance(children, list):
                    current_node_id = node.get("id")
                    if current_node_id in children:
                        branch_index = children.index(current_node_id)

        # Extract attachments from message metadata
        msg_metadata = msg.get("metadata") or {}
        if isinstance(msg_metadata, dict):
            msg_attachments = msg_metadata.get("attachments") or []
            if isinstance(msg_attachments, list):
                for attach in msg_attachments:
                    if isinstance(attach, dict) and attach.get("id"):
                        # #1252: ChatGPT attachments arrive through the OAuth-
                        # authenticated export; the only native identifier is
                        # `id`. file_id is recorded when the export carries one
                        # (some private deployments surface it).
                        file_id_raw = attach.get("file_id") or attach.get("fileId")
                        attachments.append(
                            ParsedAttachment(
                                provider_attachment_id=str(attach["id"]),
                                message_provider_id=str(msg_id),
                                name=str(attach["name"]) if attach.get("name") else None,
                                mime_type=str(attach["mime_type"]) if attach.get("mime_type") else None,
                                size_bytes=int(attach["size"])
                                if isinstance(attach.get("size"), (int, float))
                                else None,
                                provider_file_id=str(file_id_raw)
                                if isinstance(file_id_raw, str) and file_id_raw
                                else None,
                                upload_origin="oauth",
                            )
                        )

        # Assistant-generated downloadable files (#sandbox links). Code
        # Interpreter deliverables surface only as `sandbox:/mnt/data/...`
        # links inside assistant prose; the export/capture carries no bytes
        # and no metadata attachment row for them, and the links expire with
        # the sandbox container. Record each as an unfetchable attachment so
        # the archive knows the file existed, its name, and which message
        # produced it. attachment_kind="sandbox_file" keeps every acquisition
        # path away from it (there is nothing local to fetch).
        if role is Role.ASSISTANT and text:
            for sandbox_path in _sandbox_file_paths(text):
                attachments.append(
                    ParsedAttachment(
                        provider_attachment_id=f"sandbox:{msg_id}:{sandbox_path}",
                        message_provider_id=str(msg_id),
                        name=sandbox_path.rsplit("/", 1)[-1] or None,
                        attachment_kind="sandbox_file",
                        source_url=f"sandbox:{sandbox_path}",
                    )
                )

        model_slug: object = None
        duration_raw: object = None

        # Extract message-level metadata from typed fields
        if isinstance(msg_metadata, dict):
            model_slug = msg_metadata.get("model_slug")
            duration_raw = msg_metadata.get("durationMs")
            if duration_raw is None:
                duration_raw = msg_metadata.get("duration_ms")
        model_name = str(model_slug) if isinstance(model_slug, str) and model_slug else None
        duration_ms = _non_negative_int(duration_raw)

        # A non-"all" recipient marks a tool invocation (e.g. the web-search/
        # browsing tool). Computed here (rather than where ParsedMessage is
        # built below) so the content-block builder can use it to recognize
        # a JSON-encoded tool-call payload instead of storing it as raw text.
        recipient_val = msg.get("recipient")
        recipient = (
            recipient_val if isinstance(recipient_val, str) and recipient_val and recipient_val != "all" else None
        )
        tool_call_input: Mapping[str, object] | None = None
        if recipient is not None and text:
            try:
                parsed_tool_json = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                parsed_tool_json = None
            if isinstance(parsed_tool_json, dict):
                tool_call_input = parsed_tool_json

        # Build structured content blocks
        content_blocks: list[ParsedContentBlock] = []
        forced_message_type: MessageType | None = None
        content_type = content.get("content_type", "text")
        if tool_call_input is not None:
            # Recipient-addressed tool call whose content is a JSON payload
            # (e.g. ChatGPT's web-search tool: {"search_query": [...]}) --
            # a proper TOOL_USE block instead of raw JSON as BlockType.TEXT
            # (#e2yk). The reader already folds tool_use blocks by default.
            content_blocks.append(
                ParsedContentBlock(
                    type=BlockType.TOOL_USE,
                    tool_name=recipient,
                    tool_input=tool_call_input,
                    metadata={"content_type": content_type},
                )
            )
        elif content_type in ("thoughts", "reasoning_recap"):
            # ChatGPT thinking/reasoning blocks
            content_blocks.append(
                ParsedContentBlock(
                    type=BlockType.THINKING,
                    text=text,
                    metadata={"content_type": content_type},
                )
            )
        elif content_type == "code":
            # Code-interpreter input — top-level text, no parts (#1744).
            content_blocks.append(
                ParsedContentBlock(
                    type=BlockType.CODE,
                    text=text,
                    metadata={"content_type": content_type},
                )
            )
        elif content_type == "execution_output":
            # Code-interpreter output — top-level text, no parts (#1744).
            content_blocks.append(
                ParsedContentBlock(
                    type=BlockType.TOOL_RESULT,
                    text=text,
                    metadata={"content_type": content_type},
                )
            )
        elif content_type in ("user_editable_context", "model_editable_context"):
            # System-injected conversation context (#runtime evidence): custom
            # instructions / user profile (`user_editable_context`) and the
            # ChatGPT memory payload (`model_set_context`). These carry no
            # `parts`, so without this branch the messages are dropped and the
            # archive loses what context the provider injected. Empty payloads
            # (e.g. memory feature on but no memories) still drop.
            context_fields = (
                ("user_profile", "user_instructions")
                if content_type == "user_editable_context"
                else ("model_set_context",)
            )
            context_texts = [
                value for key in context_fields if isinstance(value := content.get(key), str) and value.strip()
            ]
            if context_texts:
                text = "\n\n".join(context_texts)
                forced_message_type = MessageType.CONTEXT
                content_blocks.append(
                    ParsedContentBlock(
                        type=BlockType.TEXT,
                        text=text,
                        metadata={"content_type": content_type},
                    )
                )
        elif parts:
            for part in parts:
                if isinstance(part, str) and part:
                    content_blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=_strip_citation_markers(part)))
                elif isinstance(part, dict) and part.get("content_type") == "image_asset_pointer":
                    content_blocks.append(
                        ParsedContentBlock(
                            type=BlockType.IMAGE,
                            metadata={"asset_pointer": str(part.get("asset_pointer", ""))},
                        )
                    )
                elif isinstance(part, dict) and part.get("content_type") in {
                    "audio_asset_pointer",
                    "audio_transcription",
                    "real_time_user_audio_video_asset_pointer",
                }:
                    part_text = part.get("text")
                    content_type = str(part.get("content_type"))
                    content_blocks.append(
                        ParsedContentBlock(
                            type=BlockType.DOCUMENT,
                            text=part_text if isinstance(part_text, str) and part_text else None,
                            media_type=_string_value(part, "mime_type", "media_type"),
                            web_constructs=[
                                ParsedWebConstruct(
                                    construct_type=(
                                        WebConstructType.AUDIO_TRANSCRIPTION
                                        if content_type == "audio_transcription"
                                        else WebConstructType.AUDIO_ASSET
                                    ),
                                    provider_key=content_type,
                                    text=part_text if isinstance(part_text, str) and part_text else None,
                                    asset_pointer=_string_value(part, "asset_pointer"),
                                    mime_type=_string_value(part, "mime_type", "media_type"),
                                )
                            ],
                        )
                    )

        web_constructs = _constructs_from_chatgpt_metadata(msg_metadata)
        # Inline citation anchors are stripped from stored text (invisible
        # glyphs), but their reference tokens — turn/file pointers and line
        # ranges like "filecite turn3file14 L180-L293" — carry source-location
        # detail the metadata rows sometimes lack (line_range is often null).
        # Preserve each span as a construct anchored at its original-text
        # offset, the same coordinate system the citation rows' start_ix/
        # end_ix use.
        web_constructs.extend(
            ParsedWebConstruct(
                construct_type=WebConstructType.CONTENT_REFERENCE,
                provider_key="inline_citation_marker",
                text=" ".join(token for token in marker.group(1).split("\ue202") if token),
                start_index=marker.start(),
                end_index=marker.end(),
            )
            for marker in _CITATION_MARKER_SPAN_RE.finditer(raw_text)
        )
        if web_constructs:
            if not content_blocks:
                content_blocks.append(ParsedContentBlock(type=BlockType.TEXT))
            first_block = content_blocks[0]
            first_block.web_constructs.extend(web_constructs)
        if not text and not content_blocks:
            continue

        status_val = msg.get("status")
        end_turn_val = msg.get("end_turn")
        user_context_val = msg_metadata.get("user_context_message_data") if isinstance(msg_metadata, Mapping) else None
        message_type = forced_message_type or classify_text_message_type(text) or MessageType.MESSAGE
        parsed = ParsedMessage(
            provider_message_id=str(msg_id),
            role=role,
            text=text,
            timestamp=str(timestamp) if timestamp is not None else None,
            blocks=content_blocks,
            message_type=message_type,
            material_origin=classify_material_origin(
                role=role,
                message_type=message_type,
                text=text,
                block_types=tuple(block.type for block in content_blocks),
            ),
            parent_message_provider_id=parent_message_provider_id,
            position=idx - 1,
            branch_index=branch_index,
            variant_index=branch_index,
            is_active_path=node_id in active_path_id_set if active_path_ids else None,
            model_name=model_name,
            duration_ms=duration_ms,
            sender_name=_string_value(author, "name") if isinstance(author, Mapping) else None,
            recipient=recipient,
            delivery_status=status_val if isinstance(status_val, str) and status_val else None,
            end_turn=end_turn_val if isinstance(end_turn_val, bool) else None,
            user_context_text=(
                _string_value(user_context_val, "about_user_message", "text", "content")
                if isinstance(user_context_val, Mapping)
                else None
            ),
        )
        emitted_by_node_id[node_id] = parsed.provider_message_id
        entries.append((_coerce_float(timestamp), idx, parsed))
    if any(value is not None for value, _, _ in entries):
        # Use explicit None check instead of `or` to handle zero/negative timestamps correctly
        entries.sort(key=lambda item: (item[0] is None, item[0] if item[0] is not None else 0.0, item[1]))
    messages = [entry[2] for entry in entries]
    emitted_message_ids = {message.provider_message_id for message in messages}
    messages = [
        message.model_copy(
            update={
                "parent_message_provider_id": (
                    emitted_by_node_id.get(
                        message.parent_message_provider_id,
                        message.parent_message_provider_id
                        if message.parent_message_provider_id in emitted_message_ids
                        else None,
                    )
                )
            }
        )
        if message.parent_message_provider_id is not None
        else message
        for message in messages
    ]
    active_leaf_message_provider_id = next(
        (emitted_by_node_id[node_id] for node_id in reversed(active_path_ids) if node_id in emitted_by_node_id),
        None,
    )
    if active_leaf_message_provider_id is not None:
        messages = [
            message.model_copy(
                update={"is_active_leaf": message.provider_message_id == active_leaf_message_provider_id}
            )
            for message in messages
        ]
    return (messages, attachments)


def looks_like(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return isinstance(payload.get("mapping"), dict)


def parse(payload: Mapping[str, object], fallback_id: str) -> ParsedSession:
    mapping = payload.get("mapping") or {}
    if not isinstance(mapping, dict):
        mapping = {}
    current_node = payload.get("current_node")
    current_node = current_node if isinstance(current_node, str) else None
    messages, attachments = extract_messages_from_mapping(mapping, current_node)
    generation_timings = _extract_generation_timings(mapping)
    timing_by_message_id = {timing.message_provider_id: timing for timing in generation_timings}
    if timing_by_message_id:
        messages = [
            message.model_copy(
                update={"duration_ms": timing_by_message_id[message.provider_message_id].elapsed_duration_ms}
            )
            if message.provider_message_id in timing_by_message_id
            else message
            for message in messages
        ]
    session_events = [
        ParsedSessionEvent(
            event_type="generation_lifecycle",
            timestamp=timing.event_timestamp,
            source_message_provider_id=timing.message_provider_id,
            payload={
                "state": "completed",
                "evidence_source": "provider_native",
                "fidelity": timing.fidelity,
                "duration_semantics": "provider_reported_elapsed",
                "elapsed_duration_ms": timing.elapsed_duration_ms,
                **({"started_at_ms": timing.started_at_ms} if timing.started_at_ms is not None else {}),
                **({"ended_at_ms": timing.ended_at_ms} if timing.ended_at_ms is not None else {}),
            },
        )
        for timing in generation_timings
    ]
    duration_values = [message.duration_ms for message in messages if message.duration_ms is not None]
    title = payload.get("title") or payload.get("name") or fallback_id
    conv_id = payload.get("id") or payload.get("uuid") or payload.get("conversation_id")
    ingest_flags: list[str] = []
    if not messages and payload.get("conversation_id") and payload.get("id") and "mapping" not in payload:
        ingest_flags.append(SHARED_CONVERSATION_INDEX_INGEST_FLAG)
    if payload.get("is_temporary") is True:
        ingest_flags.append("capture:temporary-chat")
    session_kind = SessionKind.TEMPORARY if payload.get("is_temporary") is True else SessionKind.STANDARD

    # ChatGPT "project" token (g-p-<id>): present in project-scoped conversations
    # as gizmo_id / conversation_template_id. A bare g-<id> is a custom GPT, not a
    # project, so only the g-p- prefix is treated as a workspace/project ref.
    project_raw = payload.get("conversation_template_id") or payload.get("gizmo_id")
    provider_project_ref = str(project_raw) if isinstance(project_raw, str) and project_raw.startswith("g-p-") else None

    return ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id=str(conv_id or fallback_id),
        title=str(title),
        session_kind=session_kind,
        provider_project_ref=provider_project_ref,
        created_at=str(payload.get("create_time")) if payload.get("create_time") is not None else None,
        updated_at=str(payload.get("update_time")) if payload.get("update_time") is not None else None,
        messages=messages,
        active_leaf_message_provider_id=next(
            (message.provider_message_id for message in messages if message.is_active_leaf),
            None,
        ),
        attachments=attachments,
        session_events=session_events,
        reported_duration_ms=sum(duration_values) if duration_values else None,
        ingest_flags=ingest_flags,
    )

from __future__ import annotations

from collections.abc import Mapping

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider, SessionKind, WebConstructType
from polylogue.core.timestamps import parse_timestamp

from .base import ParsedAttachment, ParsedContentBlock, ParsedMessage, ParsedSession, ParsedWebConstruct

SHARED_CONVERSATION_INDEX_INGEST_FLAG = "capture:chatgpt-shared-index-shell"


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
    return ParsedWebConstruct(
        construct_type=construct_type,
        provider_key=provider_key,
        title=_string_value(item, "title", "name", "source_name"),
        url=_string_value(item, "url", "link", "source_url"),
        text=_string_value(item, "snippet", "text", "content", "description"),
        source_id=_string_value(item, "id", "source_id", "ref_id", "attribution_id", "textdoc_id"),
        group_id=group_id,
        group_title=group_title,
        asset_pointer=_string_value(item, "asset_pointer"),
        mime_type=_string_value(item, "mime_type", "media_type"),
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
        text = _extract_content_text(content)
        if not text:
            continue
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

        # Build structured content blocks
        content_blocks: list[ParsedContentBlock] = []
        content_type = content.get("content_type", "text")
        if content_type in ("thoughts", "reasoning_recap"):
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
        elif parts:
            for part in parts:
                if isinstance(part, str) and part:
                    content_blocks.append(ParsedContentBlock(type=BlockType.TEXT, text=part))
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
        if web_constructs:
            if not content_blocks:
                content_blocks.append(ParsedContentBlock(type=BlockType.TEXT))
            first_block = content_blocks[0]
            first_block.web_constructs.extend(web_constructs)

        recipient_val = msg.get("recipient")
        status_val = msg.get("status")
        end_turn_val = msg.get("end_turn")
        user_context_val = msg_metadata.get("user_context_message_data") if isinstance(msg_metadata, Mapping) else None

        parsed = ParsedMessage(
            provider_message_id=str(msg_id),
            role=role,
            text=text,
            timestamp=str(timestamp) if timestamp is not None else None,
            blocks=content_blocks,
            parent_message_provider_id=parent_message_provider_id,
            position=idx - 1,
            branch_index=branch_index,
            variant_index=branch_index,
            is_active_path=node_id in active_path_id_set if active_path_ids else None,
            model_name=model_name,
            duration_ms=duration_ms,
            sender_name=_string_value(author, "name") if isinstance(author, Mapping) else None,
            recipient=recipient_val
            if isinstance(recipient_val, str) and recipient_val and recipient_val != "all"
            else None,
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
        ingest_flags=ingest_flags,
    )

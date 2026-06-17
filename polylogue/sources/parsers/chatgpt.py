from __future__ import annotations

from collections.abc import Mapping

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.core.timestamps import parse_timestamp

from .base import ParsedAttachment, ParsedContentBlock, ParsedMessage, ParsedSession


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

        # Promote message-level metadata into content_block metadata so it
        # survives parsing → materialization → storage → hydration. These
        # are prefixed with chatgpt_ to distinguish provider-specific facts
        # from canonical block semantics.
        _chatgpt_block_meta: dict[str, object] = {}
        if isinstance(msg_metadata, dict):
            model_slug_val = msg_metadata.get("model_slug")
            if model_slug_val:
                _chatgpt_block_meta["chatgpt_model"] = model_slug_val
            citations_val = msg_metadata.get("citations") or msg_metadata.get("_cite_metadata")
            if isinstance(citations_val, (list, dict)) and citations_val:
                _chatgpt_block_meta["chatgpt_citations"] = citations_val
            aggregate_result_val = msg_metadata.get("aggregate_result")
            if isinstance(aggregate_result_val, dict) and aggregate_result_val:
                _chatgpt_block_meta["chatgpt_code_execution"] = aggregate_result_val
            user_context_val = msg_metadata.get("user_context_message_data")
            if isinstance(user_context_val, dict) and user_context_val:
                _chatgpt_block_meta["chatgpt_user_context"] = user_context_val
        if isinstance(author, dict):
            author_name = author.get("name")
            if isinstance(author_name, str) and author_name:
                _chatgpt_block_meta["chatgpt_author_name"] = author_name
        recipient_val = msg.get("recipient")
        if isinstance(recipient_val, str) and recipient_val and recipient_val != "all":
            _chatgpt_block_meta["chatgpt_recipient"] = recipient_val
        status_val = msg.get("status")
        if isinstance(status_val, str) and status_val:
            _chatgpt_block_meta["chatgpt_status"] = status_val
        end_turn_val = msg.get("end_turn")
        if isinstance(end_turn_val, bool):
            _chatgpt_block_meta["chatgpt_end_turn"] = end_turn_val

        if _chatgpt_block_meta:
            for block in content_blocks:
                if block.metadata is None:
                    block.metadata = {}
                block.metadata.update(_chatgpt_block_meta)

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

    return ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id=str(conv_id or fallback_id),
        title=str(title),
        created_at=str(payload.get("create_time")) if payload.get("create_time") is not None else None,
        updated_at=str(payload.get("update_time")) if payload.get("update_time") is not None else None,
        messages=messages,
        active_leaf_message_provider_id=next(
            (message.provider_message_id for message in messages if message.is_active_leaf),
            None,
        ),
        attachments=attachments,
    )

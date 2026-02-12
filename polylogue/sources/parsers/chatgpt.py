from __future__ import annotations

from .base import ParsedAttachment, ParsedConversation, ParsedMessage, normalize_role


def _coerce_float(value: object) -> float | None:
    # Exclude bool explicitly (bool is a subclass of int)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def extract_messages_from_mapping(mapping: dict[str, object]) -> tuple[list[ParsedMessage], list[ParsedAttachment]]:
    entries: list[tuple[float | None, int, ParsedMessage]] = []
    attachments: list[ParsedAttachment] = []
    for idx, node in enumerate(mapping.values(), start=1):
        if not isinstance(node, dict):
            continue
        msg = node.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts") or []
        if not isinstance(parts, list):
            continue
        text_parts = []
        for part in parts:
            if isinstance(part, str) and part:
                text_parts.append(part)
            elif isinstance(part, dict):
                # Extract text from structured parts (e.g. tether_quote dicts)
                t = part.get("text")
                if isinstance(t, str) and t:
                    text_parts.append(t)
                # Skip image_asset_pointer and other non-text dicts
        text = "\n".join(text_parts)
        if not text:
            continue
        # Role is required - skip messages without one
        author = msg.get("author")
        raw_role = author.get("role") if isinstance(author, dict) else None
        if not raw_role:
            continue
        role = normalize_role(raw_role)
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
                        attachments.append(ParsedAttachment(
                            provider_attachment_id=str(attach["id"]),
                            message_provider_id=str(msg_id),
                            name=str(attach["name"]) if attach.get("name") else None,
                            mime_type=str(attach["mime_type"]) if attach.get("mime_type") else None,
                            size_bytes=int(attach["size"]) if isinstance(attach.get("size"), (int, float)) else None,
                            provider_meta={"raw": attach},
                        ))

        # Build provider_meta with structured content_blocks
        meta: dict[str, object] = {"raw": msg}

        # Extract message-level metadata from typed fields
        if isinstance(msg_metadata, dict):
            model_slug = msg_metadata.get("model_slug")
            if model_slug:
                meta["model"] = model_slug
            # Citations from web browsing
            citations = msg_metadata.get("citations") or msg_metadata.get("_cite_metadata")
            if isinstance(citations, (list, dict)) and citations:
                meta["citations"] = citations
            # Aggregate result from code interpreter
            aggregate_result = msg_metadata.get("aggregate_result")
            if isinstance(aggregate_result, dict) and aggregate_result:
                meta["code_execution"] = aggregate_result
            # User editable context (memory/instructions)
            user_context = msg_metadata.get("user_context_message_data")
            if isinstance(user_context, dict) and user_context:
                meta["user_context"] = user_context

        # Author name (identifies tools like dalle, browser, python)
        if isinstance(author, dict):
            author_name = author.get("name")
            if isinstance(author_name, str) and author_name:
                meta["author_name"] = author_name
        # Message recipient (e.g., "dalle.text2im", "browser", "python")
        recipient_val = msg.get("recipient")
        if isinstance(recipient_val, str) and recipient_val and recipient_val != "all":
            meta["recipient"] = recipient_val
        # Message status and end_turn
        status = msg.get("status")
        if isinstance(status, str) and status:
            meta["status"] = status
        end_turn = msg.get("end_turn")
        if isinstance(end_turn, bool):
            meta["end_turn"] = end_turn

        # Extract structured content blocks for semantic detection
        content_type = content.get("content_type", "text")
        if content_type in ("thoughts", "reasoning_recap"):
            # ChatGPT thinking/reasoning blocks
            meta["content_blocks"] = [{
                "type": "thinking",
                "content_type": content_type,
                "text": text,
            }]
        elif parts:
            # Regular content - preserve as text blocks
            content_blocks = []
            for part in parts:
                if isinstance(part, str) and part:
                    content_blocks.append({
                        "type": "text",
                        "text": part,
                    })
                elif isinstance(part, dict) and part.get("content_type") == "image_asset_pointer":
                    # Preserve image attachment references
                    content_blocks.append({
                        "type": "image",
                        "asset_pointer": str(part.get("asset_pointer", "")),
                    })
            if content_blocks:
                meta["content_blocks"] = content_blocks

        parsed = ParsedMessage(
            provider_message_id=str(msg_id),
            role=role,
            text=text,
            timestamp=str(timestamp) if timestamp is not None else None,
            parent_message_provider_id=parent_message_provider_id,
            branch_index=branch_index,
            provider_meta=meta,
        )
        entries.append((_coerce_float(timestamp), idx, parsed))
    if any(value is not None for value, _, _ in entries):
        # Use explicit None check instead of `or` to handle zero/negative timestamps correctly
        entries.sort(key=lambda item: (item[0] is None, item[0] if item[0] is not None else 0.0, item[1]))
    return ([entry[2] for entry in entries], attachments)


def looks_like(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return isinstance(payload.get("mapping"), dict)


def parse(payload: dict[str, object], fallback_id: str) -> ParsedConversation:
    mapping = payload.get("mapping") or {}
    if not isinstance(mapping, dict):
        mapping = {}
    messages, attachments = extract_messages_from_mapping(mapping)
    title = payload.get("title") or payload.get("name") or fallback_id
    conv_id = payload.get("id") or payload.get("uuid") or payload.get("conversation_id")

    # Build conversation-level provider_meta from rich fields
    conv_meta: dict[str, object] = {}
    default_model = payload.get("default_model_slug")
    if isinstance(default_model, str) and default_model:
        conv_meta["default_model"] = default_model
    gizmo_id = payload.get("gizmo_id")
    if isinstance(gizmo_id, str) and gizmo_id:
        conv_meta["gizmo_id"] = gizmo_id
    gizmo_type = payload.get("gizmo_type")
    if isinstance(gizmo_type, str) and gizmo_type:
        conv_meta["gizmo_type"] = gizmo_type
    is_archived = payload.get("is_archived")
    if isinstance(is_archived, bool) and is_archived:
        conv_meta["is_archived"] = True

    return ParsedConversation(
        provider_name="chatgpt",
        provider_conversation_id=str(conv_id or fallback_id),
        title=str(title),
        created_at=str(payload.get("create_time")) if payload.get("create_time") is not None else None,
        updated_at=str(payload.get("update_time")) if payload.get("update_time") is not None else None,
        messages=messages,
        attachments=attachments,
        provider_meta=conv_meta if conv_meta else None,
    )

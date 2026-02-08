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
        text = "\n".join(str(part) for part in parts if part)
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
    return ParsedConversation(
        provider_name="chatgpt",
        provider_conversation_id=str(conv_id or fallback_id),
        title=str(title),
        created_at=str(payload.get("create_time")) if payload.get("create_time") is not None else None,
        updated_at=str(payload.get("update_time")) if payload.get("update_time") is not None else None,
        messages=messages,
        attachments=attachments,
    )

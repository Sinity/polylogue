from __future__ import annotations

import json

from .base import ParsedAttachment, ParsedConversation, ParsedMessage, attachment_from_meta, normalize_role


def extract_text_from_segments(segments: list) -> str | None:
    lines: list[str] = []
    for segment in segments:
        if isinstance(segment, str):
            if segment:
                lines.append(segment)
            continue
        if not isinstance(segment, dict):
            continue
        # Check type first - tool_use/tool_result should be serialized as JSON
        seg_type = segment.get("type")
        if seg_type in {"tool_use", "tool_result"}:
            lines.append(json.dumps(segment, sort_keys=True))
            continue
        # Handle thinking blocks - wrap in XML tags for semantic detection
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


def extract_messages_from_chat_messages(chat_messages: list) -> tuple[list[ParsedMessage], list[ParsedAttachment]]:
    messages: list[ParsedMessage] = []
    attachments: list[ParsedAttachment] = []
    for idx, item in enumerate(chat_messages, start=1):
        if not isinstance(item, dict):
            continue
        message_id = str(item.get("uuid") or item.get("id") or item.get("message_id") or f"msg-{idx}")
        role = normalize_role(item.get("sender") or item.get("role"))
        timestamp = item.get("created_at") or item.get("create_time") or item.get("timestamp")
        # Check for text field directly first (Claude AI format)
        text = item.get("text") if isinstance(item.get("text"), str) else None
        # Then check content field
        if text is None:
            content = item.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = extract_text_from_segments(content)
            elif isinstance(content, dict):
                text = content.get("text") if isinstance(content.get("text"), str) else None
                if text is None and isinstance(content.get("parts"), list):
                    text = "\n".join(str(part) for part in content["parts"] if part)
        if text:
            messages.append(
                ParsedMessage(
                    provider_message_id=message_id,
                    role=role,
                    text=text,
                    timestamp=str(timestamp) if timestamp is not None else None,
                    provider_meta={"raw": item},
                )
            )
        for att_idx, meta in enumerate(item.get("attachments") or item.get("files") or [], start=1):
            attachment = attachment_from_meta(meta, message_id, att_idx)
            if attachment:
                attachments.append(attachment)
    return messages, attachments


def looks_like_ai(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return isinstance(payload.get("chat_messages"), list)


def looks_like_code(payload: list) -> bool:
    if not isinstance(payload, list):
        return False
    for item in payload:
        if not isinstance(item, dict):
            continue
        if any(key in item for key in ("parentUuid", "leafUuid", "sessionId", "session_id")):
            return True
    return False


def _extract_message_text(message_content: object) -> str | None:
    """Extract text from claude-code message content structure."""
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


def parse_code(payload: list, fallback_id: str) -> ParsedConversation:
    """Parse claude-code JSONL format (list of message objects)."""
    messages: list[ParsedMessage] = []
    timestamps: list[str] = []
    session_id: str | None = None

    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue

        msg_type = item.get("type")
        # Skip summary entries and other non-message types
        if msg_type in ("summary", "init"):
            continue

        # Extract session ID for conversation grouping
        if not session_id:
            session_id = item.get("sessionId") or item.get("session_id")

        # Get message UUID
        msg_id = str(item.get("uuid") or item.get("id") or f"msg-{idx}")

        # Map type to role
        if msg_type in ("user", "human"):
            role = "user"
        elif msg_type == "assistant":
            role = "assistant"
        else:
            role = msg_type or "unknown"

        # Get timestamp
        timestamp = item.get("timestamp")
        if timestamp:
            timestamps.append(str(timestamp))

        # Extract text from nested message.content structure
        msg_obj = item.get("message", {})
        text = None
        content_list = None
        if isinstance(msg_obj, dict):
            content_raw = msg_obj.get("content")
            text = _extract_message_text(content_raw)
            # Preserve content list for structured block extraction
            if isinstance(content_raw, list):
                content_list = content_raw
        elif isinstance(msg_obj, str):
            text = msg_obj

        # Build provider_meta with useful fields
        meta: dict = {"raw": item}
        if item.get("costUSD"):
            meta["costUSD"] = item.get("costUSD")
        if item.get("durationMs"):
            meta["durationMs"] = item.get("durationMs")
        if item.get("isSidechain"):
            meta["isSidechain"] = True
        if item.get("isMeta"):
            meta["isMeta"] = True

        # Extract structured content blocks for semantic detection
        if content_list:
            content_blocks = []
            for seg in content_list:
                if isinstance(seg, dict):
                    block_type = seg.get("type")
                    if block_type == "thinking":
                        content_blocks.append({
                            "type": "thinking",
                            "text": seg.get("thinking"),
                        })
                    elif block_type == "tool_use":
                        content_blocks.append({
                            "type": "tool_use",
                            "name": seg.get("name"),
                            "id": seg.get("id"),
                            "input": seg.get("input"),
                        })
                    elif block_type == "tool_result":
                        content_blocks.append({
                            "type": "tool_result",
                            "tool_use_id": seg.get("tool_use_id"),
                        })
                    elif block_type == "text":
                        content_blocks.append({
                            "type": "text",
                            "text": seg.get("text"),
                        })
                    else:
                        # For text field without explicit type
                        text_content = seg.get("text") or seg.get("content")
                        if text_content:
                            content_blocks.append({
                                "type": "text",
                                "text": text_content,
                            })
                elif isinstance(seg, str):
                    content_blocks.append({
                        "type": "text",
                        "text": seg,
                    })
            if content_blocks:
                meta["content_blocks"] = content_blocks

        messages.append(
            ParsedMessage(
                provider_message_id=msg_id,
                role=role,
                text=text,
                timestamp=str(timestamp) if timestamp else None,
                provider_meta=meta,
            )
        )

    # Derive conversation timestamps from messages
    created_at = min(timestamps) if timestamps else None
    updated_at = max(timestamps) if timestamps else None

    # Use session_id as conversation ID if available
    conv_id = session_id or fallback_id

    return ParsedConversation(
        provider_name="claude-code",
        provider_conversation_id=str(conv_id),
        title=str(conv_id),  # Claude-code doesn't have titles, use session ID
        created_at=created_at,
        updated_at=updated_at,
        messages=messages,
    )


def parse_ai(payload: dict, fallback_id: str) -> ParsedConversation:
    messages, attachments = extract_messages_from_chat_messages(payload.get("chat_messages", []))
    title = payload.get("title") or payload.get("name") or fallback_id
    conv_id = payload.get("id") or payload.get("uuid") or payload.get("conversation_id")
    return ParsedConversation(
        provider_name="claude",
        provider_conversation_id=str(conv_id or fallback_id),
        title=str(title),
        created_at=str(payload.get("created_at")) if payload.get("created_at") else None,
        updated_at=str(payload.get("updated_at")) if payload.get("updated_at") else None,
        messages=messages,
        attachments=attachments,
    )

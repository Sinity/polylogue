from __future__ import annotations

import json
from typing import List, Tuple, Optional
from .base import ParsedMessage, ParsedAttachment, ParsedConversation, normalize_role

def extract_text_from_segments(segments: list) -> Optional[str]:
    lines: List[str] = []
    for segment in segments:
        if isinstance(segment, str):
            if segment:
                lines.append(segment)
            continue
        if not isinstance(segment, dict):
            continue
        seg_text = segment.get("text")
        if isinstance(seg_text, str):
            lines.append(seg_text)
            continue
        seg_content = segment.get("content")
        if isinstance(seg_content, str):
            lines.append(seg_content)
            continue
        seg_type = segment.get("type")
        if seg_type in {"tool_use", "tool_result"}:
            lines.append(json.dumps(segment, sort_keys=True))
    combined = "\n".join(line for line in lines if line)
    return combined or None

def extract_messages_from_chat_messages(chat_messages: list) -> Tuple[List[ParsedMessage], List[ParsedAttachment]]:
    messages: List[ParsedMessage] = []
    attachments: List[ParsedAttachment] = []
    for idx, item in enumerate(chat_messages, start=1):
        if not isinstance(item, dict):
            continue
        message_id = str(item.get("uuid") or item.get("id") or item.get("message_id") or f"msg-{idx}")
        role = normalize_role(item.get("sender") or item.get("role"))
        timestamp = item.get("created_at") or item.get("create_time") or item.get("timestamp")
        content = item.get("content")
        text = None
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
    return messages, attachments

def looks_like_ai(payload: dict) -> bool:
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

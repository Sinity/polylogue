from __future__ import annotations

import json
from typing import List, Tuple, Optional
from .base import ParsedMessage, ParsedAttachment, ParsedConversation, normalize_role

def _coerce_float(value: object) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None

def extract_messages_from_mapping(mapping: dict) -> List[ParsedMessage]:
    entries: List[Tuple[Optional[float], int, ParsedMessage]] = []
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
        role = normalize_role(msg.get("author", {{}}).get("role") or "user")
        timestamp = msg.get("create_time")
        msg_id = msg.get("id") or node.get("id") or ""
        if not msg_id:
            msg_id = f"msg-{idx}"
        parsed = ParsedMessage(
            provider_message_id=str(msg_id),
            role=role,
            text=text,
            timestamp=str(timestamp) if timestamp is not None else None,
            provider_meta={"raw": msg},
        )
        entries.append((_coerce_float(timestamp), idx, parsed))
    if any(value is not None for value, _, _ in entries):
        entries.sort(key=lambda item: (item[0] is None, item[0] or 0.0, item[1]))
    return [entry[2] for entry in entries]

def looks_like(payload: dict) -> bool:
    return isinstance(payload.get("mapping"), dict)

def parse(payload: dict, fallback_id: str) -> ParsedConversation:
    messages = extract_messages_from_mapping(payload.get("mapping", {{}}))
    title = payload.get("title") or payload.get("name") or fallback_id
    conv_id = payload.get("id") or payload.get("uuid") or payload.get("conversation_id")
    return ParsedConversation(
        provider_name="chatgpt",
        provider_conversation_id=str(conv_id or fallback_id),
        title=str(title),
        created_at=str(payload.get("create_time")) if payload.get("create_time") else None,
        updated_at=str(payload.get("update_time")) if payload.get("update_time") else None,
        messages=messages,
    )

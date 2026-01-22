from __future__ import annotations

from .base import ParsedConversation, ParsedMessage, normalize_role


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


def extract_messages_from_mapping(mapping: dict) -> list[ParsedMessage]:
    entries: list[tuple[float | None, int, ParsedMessage]] = []
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
        role = normalize_role((msg.get("author") or {}).get("role") or "user")
        timestamp = msg.get("create_time")
        msg_id = msg.get("id") or node.get("id") or ""
        if not msg_id:
            msg_id = f"msg-{idx}"

        # Build provider_meta with structured content_blocks
        meta: dict = {"raw": msg}

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
            if content_blocks:
                meta["content_blocks"] = content_blocks

        parsed = ParsedMessage(
            provider_message_id=str(msg_id),
            role=role,
            text=text,
            timestamp=str(timestamp) if timestamp is not None else None,
            provider_meta=meta,
        )
        entries.append((_coerce_float(timestamp), idx, parsed))
    if any(value is not None for value, _, _ in entries):
        # Use explicit None check instead of `or` to handle zero/negative timestamps correctly
        entries.sort(key=lambda item: (item[0] is None, item[0] if item[0] is not None else 0.0, item[1]))
    return [entry[2] for entry in entries]


def looks_like(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return isinstance(payload.get("mapping"), dict)


def parse(payload: dict, fallback_id: str) -> ParsedConversation:
    messages = extract_messages_from_mapping(payload.get("mapping") or {})
    title = payload.get("title") or payload.get("name") or fallback_id
    conv_id = payload.get("id") or payload.get("uuid") or payload.get("conversation_id")
    return ParsedConversation(
        provider_name="chatgpt",
        provider_conversation_id=str(conv_id or fallback_id),
        title=str(title),
        created_at=str(payload.get("create_time")) if payload.get("create_time") is not None else None,
        updated_at=str(payload.get("update_time")) if payload.get("update_time") is not None else None,
        messages=messages,
    )

from __future__ import annotations

from .base import ParsedConversation, ParsedMessage


def looks_like(payload: list[object]) -> bool:
    """Detect Codex JSONL format.

    New format (JSONL with session metadata + messages):
        {"id":"...","timestamp":"...","git":{...}}
        {"record_type":"state"}
        {"type":"message","role":"user","content":[...]}

    Old format (JSON array with prompt/completion pairs):
        [{"prompt": "...", "completion": "..."}]
    """
    if not isinstance(payload, list):
        return False

    for item in payload:
        if not isinstance(item, dict):
            continue
        # New format: session metadata or message records
        if "id" in item and "timestamp" in item:
            return True
        if item.get("type") == "message":
            return True
        # Old format: prompt/completion pairs
        if "prompt" in item and "completion" in item:
            return True

    return False


def parse(payload: list[object], fallback_id: str) -> ParsedConversation:
    """Parse Codex JSONL session file.

    Format:
        Line 1: {"id":"session-id","timestamp":"...","git":{...}}
        Line N: {"record_type":"state"} or {"type":"message",...}

    Messages have structure:
        {"type":"message","role":"user/assistant","content":[{"type":"input_text","text":"..."}]}
    """
    messages: list[ParsedMessage] = []
    session_id = fallback_id
    session_timestamp = None

    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue

        # Skip state markers
        if item.get("record_type") == "state":
            continue

        # Extract session metadata (first line)
        if "id" in item and "timestamp" in item and not item.get("type"):
            session_id = item["id"]
            session_timestamp = item.get("timestamp")
            continue

        # Parse message records
        if item.get("type") == "message":
            role = item.get("role")
            content = item.get("content", [])

            if not role or not content:
                continue

            # Extract text from content array
            # Content is list of {"type":"input_text","text":"..."}
            text_parts = []
            for content_item in content:
                if isinstance(content_item, dict):
                    if content_item.get("type") == "input_text":
                        text = content_item.get("text", "")
                        if text:
                            text_parts.append(text)

            if text_parts:
                messages.append(
                    ParsedMessage(
                        provider_message_id=item.get("id") or f"msg-{idx}",
                        role=role,
                        text="\n".join(text_parts),
                        timestamp=item.get("timestamp"),
                    )
                )

        # Fallback: old format with prompt/completion
        elif "prompt" in item or "completion" in item:
            prompt = item.get("prompt")
            completion = item.get("completion")
            timestamp = item.get("timestamp")

            if isinstance(prompt, str) and prompt:
                messages.append(
                    ParsedMessage(
                        provider_message_id=f"prompt-{idx}",
                        role="user",
                        text=prompt,
                        timestamp=str(timestamp) if timestamp else None,
                    )
                )
            if isinstance(completion, str) and completion:
                messages.append(
                    ParsedMessage(
                        provider_message_id=f"completion-{idx}",
                        role="assistant",
                        text=completion,
                        timestamp=str(timestamp) if timestamp else None,
                    )
                )

    return ParsedConversation(
        provider_name="codex",
        provider_conversation_id=session_id,
        title=session_id,
        created_at=session_timestamp,
        updated_at=None,
        messages=messages,
    )

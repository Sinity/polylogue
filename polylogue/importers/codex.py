from __future__ import annotations

from .base import ParsedConversation, ParsedMessage


def looks_like(payload: list[object]) -> bool:
    if not isinstance(payload, list):
        return False
    for item in payload:
        if not isinstance(item, dict):
            continue
        if "prompt" in item and "completion" in item:
            return True
    return False


def parse(payload: list[object], fallback_id: str) -> ParsedConversation:
    messages: list[ParsedMessage] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        prompt = item.get("prompt")
        completion = item.get("completion")
        if isinstance(prompt, str) and prompt:
            messages.append(
                ParsedMessage(
                    provider_message_id=f"prompt-{idx}",
                    role="user",
                    text=prompt,
                    timestamp=str(item.get("timestamp")) if item.get("timestamp") else None,
                )
            )
        if isinstance(completion, str) and completion:
            messages.append(
                ParsedMessage(
                    provider_message_id=f"completion-{idx}",
                    role="assistant",
                    text=completion,
                    timestamp=str(item.get("timestamp")) if item.get("timestamp") else None,
                )
            )
    return ParsedConversation(
        provider_name="codex",
        provider_conversation_id=fallback_id,
        title=fallback_id,
        created_at=None,
        updated_at=None,
        messages=messages,
    )

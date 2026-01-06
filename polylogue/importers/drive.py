from __future__ import annotations

import json
from typing import List, Optional
from .base import ParsedMessage, ParsedAttachment, ParsedConversation, normalize_role

def extract_text_from_chunk(chunk: dict) -> Optional[str]:
    for key in ("text", "content", "message", "markdown", "data"):
        value = chunk.get(key)
        if isinstance(value, str):
            return value
    parts = chunk.get("parts")
    if isinstance(parts, list):
        return "\n".join(str(part) for part in parts if part)
    return None

def parse_chunked_prompt(provider: str, payload: dict, fallback_id: str) -> ParsedConversation:
    prompt = payload.get("chunkedPrompt")
    chunks = []
    if isinstance(prompt, dict):
        chunks = prompt.get("chunks") or []
    elif isinstance(payload.get("chunks"), list):
        chunks = payload.get("chunks")
    
    messages: List[ParsedMessage] = []
    for idx, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, str):
            chunk_obj = {"text": chunk}
        elif isinstance(chunk, dict):
            chunk_obj = chunk
        else:
            continue
        text = extract_text_from_chunk(chunk_obj)
        if not text:
            continue
        role = normalize_role(chunk_obj.get("role") or chunk_obj.get("author"))
        messages.append(
            ParsedMessage(
                provider_message_id=str(chunk_obj.get("id") or f"chunk-{idx}"),
                role=role,
                text=text,
                timestamp=None,
            )
        )
    
    title = payload.get("title") or payload.get("displayName") or fallback_id
    return ParsedConversation(
        provider_name=provider,
        provider_conversation_id=str(payload.get("id") or fallback_id),
        title=str(title),
        created_at=str(payload.get("createTime")) if payload.get("createTime") else None,
        updated_at=str(payload.get("updateTime")) if payload.get("updateTime") else None,
        messages=messages,
    )

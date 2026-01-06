from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import List, Optional

@dataclass
class ParsedMessage:
    provider_message_id: str
    role: str
    text: str
    timestamp: Optional[str]
    provider_meta: Optional[dict] = None

@dataclass
class ParsedAttachment:
    provider_attachment_id: str
    message_provider_id: Optional[str]
    name: Optional[str]
    mime_type: Optional[str]
    size_bytes: Optional[int]
    path: Optional[str]
    provider_meta: Optional[dict] = None

@dataclass
class ParsedConversation:
    provider_name: str
    provider_conversation_id: str
    title: str
    created_at: Optional[str]
    updated_at: Optional[str]
    messages: List[ParsedMessage]
    attachments: List[ParsedAttachment] = field(default_factory=list)

def hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()

def normalize_role(role: Optional[str]) -> str:
    if not role:
        return "message"
    lowered = str(role).strip().lower()
    if lowered in {"assistant", "model"}:
        return "assistant"
    if lowered in {"user", "human"}:
        return "user"
    if lowered in {"system"}:
        return "system"
    return lowered

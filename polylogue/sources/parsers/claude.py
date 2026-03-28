"""Claude parser using typed Pydantic models.

Uses ClaudeCodeRecord from polylogue.sources.providers.claude_code for type-safe parsing
with automatic validation and normalized property access.
"""

from __future__ import annotations

from .base import ParsedConversation
from .claude_ai_parser import looks_like_ai as _looks_like_ai
from .claude_ai_parser import parse_ai as _parse_ai
from .claude_code_detection import looks_like_code
from .claude_code_parser import parse_code
from .claude_common import (
    extract_messages_from_chat_messages as _extract_messages_from_chat_messages,
)
from .claude_common import extract_text_from_segments as _extract_text_from_segments
from .claude_common import normalize_timestamp as _normalize_timestamp
from .claude_index import (
    SessionIndexEntry,
)
from .claude_index import (
    enrich_conversation_from_index as _enrich_conversation_from_index,
)
from .claude_index import (
    find_sessions_index as _find_sessions_index,
)
from .claude_index import (
    parse_sessions_index as _parse_sessions_index,
)


def looks_like_ai(payload: object) -> bool:
    return _looks_like_ai(payload)


def parse_sessions_index(index_path):
    return _parse_sessions_index(index_path)


def find_sessions_index(session_path):
    return _find_sessions_index(session_path)


def enrich_conversation_from_index(conv: ParsedConversation, index_entry: SessionIndexEntry) -> ParsedConversation:
    return _enrich_conversation_from_index(conv, index_entry)


def extract_text_from_segments(segments: list[object]) -> str | None:
    return _extract_text_from_segments(segments)


def extract_messages_from_chat_messages(chat_messages: list[object]):
    return _extract_messages_from_chat_messages(chat_messages)


def normalize_timestamp(timestamp: int | float | str | None) -> str | None:
    return _normalize_timestamp(timestamp)


def parse_ai(payload: dict[str, object], fallback_id: str) -> ParsedConversation:
    return _parse_ai(payload, fallback_id)


# Symmetric aliases — makes the claude module conform to the same interface
# as chatgpt.py and codex.py (parse + looks_like at module level).
# parse_code / looks_like_code remain for explicit dispatch in source.py.
parse = parse_code
looks_like = looks_like_code

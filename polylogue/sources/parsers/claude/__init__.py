"""Claude parser using typed Pydantic models.

Uses ClaudeCodeRecord from polylogue.sources.providers.claude_code for type-safe parsing
with automatic validation and normalized property access.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..base import ParsedSession
from .ai_parser import looks_like_ai as _looks_like_ai
from .ai_parser import parse_ai as _parse_ai
from .code_detection import looks_like_code
from .code_parser import parse_code, parse_code_stream, reconcile_code_session_chunks
from .common import (
    extract_messages_from_chat_messages,
    extract_text_from_segments,
    normalize_timestamp,
)
from .index import (
    SessionIndexEntry,
    enrich_session_from_index,
    find_sessions_index,
    parse_sessions_index,
)
from .orchestration import inventory_claude_orchestration_artifacts, parse_claude_orchestration_artifact


def looks_like_ai(payload: object) -> bool:
    return _looks_like_ai(payload)


def parse_ai(payload: Mapping[str, object], fallback_id: str) -> ParsedSession:
    return _parse_ai(payload, fallback_id)


# Symmetric aliases — makes the claude module conform to the same interface
# as chatgpt.py and codex.py (parse + looks_like at module level).
# parse_code / looks_like_code remain for explicit dispatch in source.py.
parse = parse_code
parse_stream = parse_code_stream
looks_like = looks_like_code

__all__ = [
    "SessionIndexEntry",
    "enrich_session_from_index",
    "extract_messages_from_chat_messages",
    "extract_text_from_segments",
    "find_sessions_index",
    "looks_like",
    "looks_like_ai",
    "looks_like_code",
    "normalize_timestamp",
    "inventory_claude_orchestration_artifacts",
    "parse",
    "parse_ai",
    "parse_code",
    "parse_code_stream",
    "reconcile_code_session_chunks",
    "parse_sessions_index",
    "parse_claude_orchestration_artifact",
    "parse_stream",
]

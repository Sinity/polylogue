"""Conversation builders and wire-format shaping for synthetic corpora."""

from __future__ import annotations

from polylogue.schemas.synthetic.build_batch import (
    _generate_conversation,
    _role_cycle,
    generate_batch,
)
from polylogue.schemas.synthetic.build_records import (
    _generate_jsonl_records,
    _generate_linear_json,
    _generate_tree_json,
)
from polylogue.schemas.synthetic.build_wire_formats import (
    _ensure_wire_chatgpt,
    _ensure_wire_claude_ai,
    _ensure_wire_claude_code,
    _ensure_wire_codex,
    _ensure_wire_format,
    _ensure_wire_gemini,
)

__all__ = [
    "_ensure_wire_chatgpt",
    "_ensure_wire_claude_ai",
    "_ensure_wire_claude_code",
    "_ensure_wire_codex",
    "_ensure_wire_format",
    "_ensure_wire_gemini",
    "_generate_conversation",
    "_generate_jsonl_records",
    "_generate_linear_json",
    "_generate_tree_json",
    "_role_cycle",
    "generate_batch",
]

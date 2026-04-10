"""Provider wire-format shaping helpers for synthetic corpora."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone

from polylogue.schemas.synthetic.semantic_values import _text_for_role
from polylogue.schemas.synthetic.showcase import ConversationTheme


def _ensure_wire_format(
    self,
    data: dict,
    role: str,
    rng: random.Random,
    index: int,
    base_ts: float = 1700000000.0,
    theme: ConversationTheme | None = None,
) -> None:
    ts = base_ts + index * 60
    match self.provider:
        case "chatgpt":
            self._ensure_wire_chatgpt(data, role, rng, ts, index=index, theme=theme)
        case "claude-ai":
            self._ensure_wire_claude_ai(data, role, rng, ts, index=index, theme=theme)
        case "claude-code":
            self._ensure_wire_claude_code(data, role, rng, ts, index=index, theme=theme)
        case "codex":
            self._ensure_wire_codex(data, role, rng, ts, index=index, theme=theme)
        case "gemini":
            self._ensure_wire_gemini(data, role, rng, index=index, theme=theme)


def _ensure_wire_chatgpt(
    self, data: dict, role: str, rng: random.Random, ts: float, *, index: int, theme: ConversationTheme | None
) -> None:
    msg = data.get("message")
    if not isinstance(msg, dict):
        msg = {"id": str(uuid.UUID(int=rng.getrandbits(128), version=4))}
        data["message"] = msg
    author = msg.setdefault("author", {})
    if isinstance(author, dict):
        author.setdefault("role", role)
    if not isinstance(msg.get("content"), dict):
        msg["content"] = {}
    content = msg.setdefault("content", {})
    if isinstance(content, dict):
        if "parts" not in content or not content["parts"]:
            content["parts"] = [_text_for_role(rng, role, turn_index=index, theme=theme)]
        content.setdefault("content_type", "text")
    msg.setdefault("create_time", ts)
    msg.setdefault("id", str(uuid.UUID(int=rng.getrandbits(128), version=4)))


def _ensure_wire_claude_ai(
    self, data: dict, role: str, rng: random.Random, ts: float, *, index: int, theme: ConversationTheme | None
) -> None:
    data.setdefault("uuid", str(uuid.UUID(int=rng.getrandbits(128), version=4)))
    data.setdefault("sender", role)
    if not data.get("text"):
        data["text"] = _text_for_role(rng, role, turn_index=index, theme=theme)
    if "created_at" not in data:
        data["created_at"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _ensure_wire_claude_code(
    self, data: dict, role: str, rng: random.Random, ts: float, *, index: int, theme: ConversationTheme | None
) -> None:
    data.setdefault("type", role)
    if not isinstance(data.get("message"), dict):
        data["message"] = {}
    msg = data["message"]
    msg.setdefault("role", role)
    if "content" not in msg:
        msg["content"] = [{"type": "text", "text": _text_for_role(rng, role, turn_index=index, theme=theme)}]
    if "timestamp" not in data:
        data["timestamp"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _ensure_wire_codex(
    self, data: dict, role: str, rng: random.Random, ts: float, *, index: int, theme: ConversationTheme | None
) -> None:
    data["type"] = "message"
    data.setdefault("role", role)
    if "content" not in data:
        content_type = "input_text" if role == "user" else "output_text"
        data["content"] = [{"type": content_type, "text": _text_for_role(rng, role, turn_index=index, theme=theme)}]
    data.setdefault("id", str(uuid.UUID(int=rng.getrandbits(128), version=4)))
    if "timestamp" not in data:
        data["timestamp"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    data.pop("payload", None)


def _ensure_wire_gemini(
    self, data: dict, role: str, rng: random.Random, *, index: int, theme: ConversationTheme | None
) -> None:
    data.setdefault("role", role)
    if not data.get("text"):
        data["text"] = _text_for_role(rng, role, turn_index=index, theme=theme)


__all__ = [
    "_ensure_wire_chatgpt",
    "_ensure_wire_claude_ai",
    "_ensure_wire_claude_code",
    "_ensure_wire_codex",
    "_ensure_wire_format",
    "_ensure_wire_gemini",
]

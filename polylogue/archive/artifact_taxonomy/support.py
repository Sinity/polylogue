"""Shared artifact-taxonomy heuristics."""

from __future__ import annotations

from pathlib import Path

from polylogue.core.json import JSONDocument, JSONValue, json_document

_PATH_ONLY_SIDECARS = {
    "bridge-pointer.json": "bridge pointer sidecar",
    "sessions-index.json": "session index sidecar",
}
_SUBAGENT_SUFFIXES = (".jsonl", ".jsonl.txt", ".ndjson")
_SCALAR_TYPES = (str, int, float, bool, type(None))
_RECORDISH_KEYS = frozenset(
    {
        "type",
        "record_type",
        "sessionId",
        "parentUuid",
        "message",
        "payload",
        "tool_name",
        "tool_input",
    }
)
_MESSAGE_KEYS = frozenset({"role", "content", "text", "parts", "author"})


def path_only_sidecars() -> dict[str, str]:
    return _PATH_ONLY_SIDECARS


def looks_like_conversation_document(payload: JSONDocument) -> bool:
    if payload.get("polylogue_capture_kind") == "browser_llm_session":
        return True
    if isinstance(payload.get("mapping"), dict):
        return True
    if isinstance(payload.get("chat_messages"), list):
        return True
    if isinstance(payload.get("chunkedPrompt"), dict):
        return True
    if isinstance(payload.get("chunks"), list):
        return True

    messages = payload.get("messages")
    return isinstance(messages, list) and any(looks_like_message_entry(item) for item in messages[:12])


def looks_like_record_stream(payload: list[JSONDocument]) -> bool:
    if not payload:
        return False
    recordish = sum(1 for item in payload if looks_like_record_entry(item))
    return recordish / max(len(payload), 1) >= 0.5


def looks_like_record_entry(payload: JSONDocument) -> bool:
    if any(key in payload for key in _RECORDISH_KEYS):
        return True
    if "role" in payload and any(key in payload for key in ("content", "text")) and len(payload) <= 16:
        return True
    nested_message = json_document(payload.get("message"))
    return bool(nested_message) and any(key in nested_message for key in _MESSAGE_KEYS)


def looks_like_message_entry(payload: object) -> bool:
    return isinstance(payload, dict) and any(key in payload for key in _MESSAGE_KEYS)


def looks_metadataish_dict(payload: JSONDocument) -> bool:
    if not payload:
        return True
    if len(payload) > 20:
        return False
    if looks_like_record_entry(payload):
        return False
    if looks_like_conversation_document(payload):
        return False
    return all(is_scalarish(value) for value in payload.values())


def looks_metadataish_list(payload: list[JSONValue]) -> bool:
    if not payload:
        return True
    if len(payload) > 512:
        return False
    return all(
        isinstance(item, _SCALAR_TYPES) or (isinstance(item, dict) and looks_metadataish_dict(item))
        for item in payload[:64]
    )


def is_scalarish(value: object, *, depth: int = 0) -> bool:
    if isinstance(value, _SCALAR_TYPES):
        return True
    if depth >= 2:
        return False
    if isinstance(value, list):
        return len(value) <= 32 and all(is_scalarish(item, depth=depth + 1) for item in value)
    if isinstance(value, dict):
        return len(value) <= 8 and all(
            isinstance(key, str) and is_scalarish(item, depth=depth + 1) for key, item in value.items()
        )
    return False


def is_subagent_path(source_path: str | Path | None) -> bool:
    normalized = normalize_source_path(source_path)
    if not normalized:
        return False
    inner = normalized.rsplit(":", 1)[-1]
    inner_lower = inner.lower()
    name = Path(inner).name.lower()
    return "/subagents/" in inner_lower or (name.startswith("agent-") and name.endswith(_SUBAGENT_SUFFIXES))


def normalize_source_path(source_path: str | Path | None) -> str:
    if source_path is None:
        return ""
    return str(source_path).replace("\\", "/")

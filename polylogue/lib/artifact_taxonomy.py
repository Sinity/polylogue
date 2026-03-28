"""Heuristic raw-artifact taxonomy for conversation-bearing payloads.

The taxonomy intentionally favors payload shape over path names. Path hints are
used only as strong evidence for well-known sidecars and weak evidence for
subagent streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from polylogue.types import Provider

_PATH_ONLY_SIDEcars = {
    "bridge-pointer.json": "bridge pointer sidecar",
    "sessions-index.json": "session index sidecar",
}
_SUBAGENT_SUFFIXES = (".jsonl", ".jsonl.txt", ".ndjson")
_SCALAR_TYPES = (str, int, float, bool, type(None))
_RECORDISH_KEYS = frozenset({
    "type",
    "record_type",
    "sessionId",
    "parentUuid",
    "message",
    "payload",
    "tool_name",
    "tool_input",
})
_MESSAGE_KEYS = frozenset({"role", "content", "text", "parts", "author"})


class ArtifactKind(StrEnum):
    CONVERSATION_DOCUMENT = "conversation_document"
    CONVERSATION_RECORD_STREAM = "conversation_record_stream"
    SUBAGENT_CONVERSATION_STREAM = "subagent_conversation_stream"
    AGENT_SIDECAR_META = "agent_sidecar_meta"
    SESSION_INDEX = "session_index"
    BRIDGE_POINTER = "bridge_pointer"
    METADATA_DOCUMENT = "metadata_document"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ArtifactClassification:
    """Heuristic classification of a raw payload or document sample."""

    provider: Provider
    kind: ArtifactKind
    parse_as_conversation: bool
    schema_eligible: bool
    default_priority: int
    reason: str

    @property
    def cohort(self) -> str:
        return self.kind.value


def classify_artifact_path(
    source_path: str | Path | None,
    *,
    provider: str | Provider,
) -> ArtifactClassification | None:
    """Classify obvious sidecars using only the source path."""
    provider_token = Provider.from_string(provider)
    normalized = _normalize_source_path(source_path)
    if not normalized:
        return None

    inner_name = Path(normalized.rsplit(":", 1)[-1]).name.lower()
    if inner_name in _PATH_ONLY_SIDEcars:
        kind = (
            ArtifactKind.BRIDGE_POINTER
            if inner_name == "bridge-pointer.json"
            else ArtifactKind.SESSION_INDEX
        )
        return ArtifactClassification(
            provider=provider_token,
            kind=kind,
            parse_as_conversation=False,
            schema_eligible=False,
            default_priority=0,
            reason=_PATH_ONLY_SIDEcars[inner_name],
        )

    if inner_name.startswith("agent-") and inner_name.endswith(".meta.json"):
        return ArtifactClassification(
            provider=provider_token,
            kind=ArtifactKind.AGENT_SIDECAR_META,
            parse_as_conversation=False,
            schema_eligible=False,
            default_priority=0,
            reason="agent sidecar metadata path",
        )

    return None


def classify_artifact(
    payload: Any,
    *,
    provider: str | Provider,
    source_path: str | Path | None = None,
) -> ArtifactClassification:
    """Classify a payload/document into a conversation or sidecar cohort."""
    provider_token = Provider.from_string(provider)
    explicit = classify_artifact_path(source_path, provider=provider_token)
    if explicit is not None:
        return explicit

    if isinstance(payload, list):
        return _classify_list(payload, provider=provider_token, source_path=source_path)
    if isinstance(payload, dict):
        return _classify_dict(payload, provider=provider_token, source_path=source_path)
    return ArtifactClassification(
        provider=provider_token,
        kind=ArtifactKind.UNKNOWN,
        parse_as_conversation=False,
        schema_eligible=False,
        default_priority=0,
        reason="non-object payload",
    )


def _classify_list(
    payload: list[Any],
    *,
    provider: Provider,
    source_path: str | Path | None,
) -> ArtifactClassification:
    dict_items = [item for item in payload[:32] if isinstance(item, dict)]
    if not payload:
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.METADATA_DOCUMENT,
            parse_as_conversation=False,
            schema_eligible=False,
            default_priority=0,
            reason="empty list payload",
        )

    if dict_items and _looks_like_record_stream(dict_items):
        is_subagent = _is_subagent_path(source_path)
        kind = (
            ArtifactKind.SUBAGENT_CONVERSATION_STREAM
            if is_subagent
            else ArtifactKind.CONVERSATION_RECORD_STREAM
        )
        return ArtifactClassification(
            provider=provider,
            kind=kind,
            parse_as_conversation=True,
            schema_eligible=True,
            default_priority=90 if is_subagent else 120,
            reason="record-like JSONL stream",
        )

    if dict_items and any(_looks_like_conversation_document(item) for item in dict_items):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.CONVERSATION_DOCUMENT,
            parse_as_conversation=True,
            schema_eligible=True,
            default_priority=120,
            reason="bundle of conversation documents",
        )

    if _looks_metadataish_list(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.METADATA_DOCUMENT,
            parse_as_conversation=False,
            schema_eligible=False,
            default_priority=0,
            reason="metadata-oriented list payload",
        )

    return ArtifactClassification(
        provider=provider,
        kind=ArtifactKind.UNKNOWN,
        parse_as_conversation=False,
        schema_eligible=False,
        default_priority=0,
        reason="unrecognized list payload",
    )


def _classify_dict(
    payload: dict[str, Any],
    *,
    provider: Provider,
    source_path: str | Path | None,
) -> ArtifactClassification:
    if _looks_like_conversation_document(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.CONVERSATION_DOCUMENT,
            parse_as_conversation=True,
            schema_eligible=True,
            default_priority=120,
            reason="conversation-bearing document",
        )

    if _is_subagent_path(source_path) and _looks_like_record_entry(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.SUBAGENT_CONVERSATION_STREAM,
            parse_as_conversation=True,
            schema_eligible=True,
            default_priority=90,
            reason="subagent record payload",
        )

    if _looks_metadataish_dict(payload):
        return ArtifactClassification(
            provider=provider,
            kind=ArtifactKind.METADATA_DOCUMENT,
            parse_as_conversation=False,
            schema_eligible=False,
            default_priority=0,
            reason="metadata-oriented document",
        )

    return ArtifactClassification(
        provider=provider,
        kind=ArtifactKind.UNKNOWN,
        parse_as_conversation=False,
        schema_eligible=False,
        default_priority=0,
        reason="unrecognized document payload",
    )


def _looks_like_conversation_document(payload: dict[str, Any]) -> bool:
    if isinstance(payload.get("mapping"), dict):
        return True
    if isinstance(payload.get("chat_messages"), list):
        return True
    if isinstance(payload.get("chunkedPrompt"), dict):
        return True
    if isinstance(payload.get("chunks"), list):
        return True

    messages = payload.get("messages")
    if isinstance(messages, list) and any(_looks_like_message_entry(item) for item in messages[:12]):
        return True
    return False


def _looks_like_record_stream(payload: list[dict[str, Any]]) -> bool:
    if not payload:
        return False
    recordish = sum(1 for item in payload if _looks_like_record_entry(item))
    return recordish / max(len(payload), 1) >= 0.5


def _looks_like_record_entry(payload: dict[str, Any]) -> bool:
    if any(key in payload for key in _RECORDISH_KEYS):
        return True
    if "role" in payload and any(key in payload for key in ("content", "text")) and len(payload) <= 16:
        return True
    nested_message = payload.get("message")
    return isinstance(nested_message, dict) and any(key in nested_message for key in _MESSAGE_KEYS)


def _looks_like_message_entry(payload: Any) -> bool:
    return isinstance(payload, dict) and any(key in payload for key in _MESSAGE_KEYS)


def _looks_metadataish_dict(payload: dict[str, Any]) -> bool:
    if not payload:
        return True
    if len(payload) > 20:
        return False
    if _looks_like_record_entry(payload):
        return False
    if _looks_like_conversation_document(payload):
        return False
    return all(_is_scalarish(value) for value in payload.values())


def _looks_metadataish_list(payload: list[Any]) -> bool:
    if not payload:
        return True
    if len(payload) > 512:
        return False
    return all(
        isinstance(item, _SCALAR_TYPES)
        or (isinstance(item, dict) and _looks_metadataish_dict(item))
        for item in payload[:64]
    )


def _is_scalarish(value: Any, *, depth: int = 0) -> bool:
    if isinstance(value, _SCALAR_TYPES):
        return True
    if depth >= 2:
        return False
    if isinstance(value, list):
        return len(value) <= 32 and all(_is_scalarish(item, depth=depth + 1) for item in value)
    if isinstance(value, dict):
        return len(value) <= 8 and all(isinstance(key, str) and _is_scalarish(item, depth=depth + 1) for key, item in value.items())
    return False


def _is_subagent_path(source_path: str | Path | None) -> bool:
    normalized = _normalize_source_path(source_path)
    if not normalized:
        return False
    inner = normalized.rsplit(":", 1)[-1]
    inner_lower = inner.lower()
    name = Path(inner).name.lower()
    return "/subagents/" in inner_lower or (
        name.startswith("agent-") and name.endswith(_SUBAGENT_SUFFIXES)
    )


def _normalize_source_path(source_path: str | Path | None) -> str:
    if source_path is None:
        return ""
    return str(source_path).replace("\\", "/")


__all__ = [
    "ArtifactClassification",
    "ArtifactKind",
    "classify_artifact",
    "classify_artifact_path",
]

"""Shared artifact-taxonomy heuristics."""

from __future__ import annotations

from itertools import islice
from pathlib import Path

from polylogue.core.json import JSONDocument, JSONValue, json_document

_PATH_ONLY_SIDECARS = {
    "bridge-pointer.json": "bridge pointer sidecar",
    "sessions-index.json": "session index sidecar",
    "logs.json": "agent log sidecar",
}
_SUBAGENT_SUFFIXES = (".jsonl", ".jsonl.txt", ".ndjson")
_SCALAR_TYPES = (str, int, float, bool, type(None))
#: Keys specific enough that their bare presence alone is positive evidence of
#: a provider conversation/tool record. Deliberately does NOT include bare
#: ``"type"`` (see ``_TYPE_ENVELOPE_MARKERS`` below): a generic ``"type"``
#: field shows up on all kinds of non-conversational structured data (graph
#: edges, index rows, run manifests), so "record has *a* type key" is not
#: discriminating evidence on its own -- treating it as sufficient is exactly
#: how a third-party analysis artifact like ``conversation_relationships.jsonl``
#: (rows shaped ``{"conversation", "parent", "child", "type", "timestamp"}``,
#: no envelope at all) misclassified as a session-record stream (polylogue-9ykn).
_RECORDISH_KEYS = frozenset(
    {
        "record_type",
        "sessionId",
        "parentUuid",
        "message",
        "payload",
        "tool_name",
        "tool_input",
    }
)
#: A bare ``"type"`` key only counts as positive record evidence when it
#: co-occurs with at least one of these genuine provider-record envelope
#: markers (mirrors ``sources/parsers/claude/code_detection.py``'s
#: ``looks_like_code``, which has the same "type-only is too weak" defect and
#: the same fix shape).
_TYPE_ENVELOPE_MARKERS = frozenset({"uuid", "sessionId", "parentUuid", "message", "payload", "cwd", "version"})
_MESSAGE_KEYS = frozenset({"role", "content", "text", "parts", "author"})
#: Third-party graph/relationship-index JSONL rows (e.g. a sinex analysis
#: artifact recording conversation parent/child edges) that happen to sit
#: under a watched Claude Code directory tree. Both observed field-name
#: variants are guarded explicitly rather than relying solely on the
#: ``_TYPE_ENVELOPE_MARKERS`` fix above, so this shape is refused even if a
#: future provider record legitimately grows one of the envelope markers.
_RELATIONSHIP_INDEX_KEYS = frozenset({"session", "parent", "child", "type", "timestamp"})
_RELATIONSHIP_INDEX_KEYS_CONVERSATION = frozenset({"conversation", "parent", "child", "type", "timestamp"})
_HOOK_EVENT_KEYS = frozenset({"event_type", "session_id", "timestamp", "provider"})
_BEADS_INTERACTION_KEYS = frozenset({"id", "kind", "created_at", "issue_id", "extra"})
#: A Claude Code ``projects/<proj>/<session-uuid>.jsonl`` file whose only
#: records carry these ``type`` values is a pure file-history checkpoint
#: stream, never a conversation (polylogue-omsw). Mirrors the type set
#: ``archive/raw_materialization.py``'s ``parsed_non_session_artifact_reason``
#: already checks post-parse ("Claude Code file-history snapshot").
_FILE_HISTORY_SNAPSHOT_ONLY_TYPES = frozenset({"file-history-snapshot", "progress"})


def path_only_sidecar_reason(name: str) -> str | None:
    lowered = name.lower()
    if lowered in _PATH_ONLY_SIDECARS:
        return _PATH_ONLY_SIDECARS[lowered]
    if lowered.startswith("request_dump_") and lowered.endswith(".json"):
        return "Hermes request dump sidecar"
    return None


def looks_like_session_document(payload: JSONDocument) -> bool:
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
    # An Antigravity language-server markdown export carries its whole
    # conversation in one `markdown` string rather than a message list, so
    # every value is a scalar and `looks_metadataish_dict` would otherwise
    # classify it as a metadata document. Raw replay wraps the single document
    # in a one-item list before classifying, so that verdict reached
    # `looks_metadataish_list` and made replay drop the session -- every
    # antigravity raw in the live archive is quarantined for this reason.
    if (
        payload.get("source") == "antigravity_language_server"
        and isinstance(payload.get("cascadeId"), str)
        and isinstance(payload.get("markdown"), str)
    ):
        return True

    messages = payload.get("messages")
    return isinstance(messages, list) and any(looks_like_message_entry(item) for item in messages[:12])


def looks_like_record_stream(payload: list[JSONDocument]) -> bool:
    if not payload:
        return False
    recordish = sum(1 for item in payload if looks_like_record_entry(item))
    return recordish / max(len(payload), 1) >= 0.5


def looks_like_record_entry(payload: JSONDocument) -> bool:
    has_envelope_marker = any(key in payload for key in _TYPE_ENVELOPE_MARKERS)
    if (
        _RELATIONSHIP_INDEX_KEYS.issubset(payload) or _RELATIONSHIP_INDEX_KEYS_CONVERSATION.issubset(payload)
    ) and not has_envelope_marker:
        return False
    if any(key in payload for key in _RECORDISH_KEYS):
        return True
    if "type" in payload and has_envelope_marker:
        return True
    if "role" in payload and any(key in payload for key in ("content", "text")) and len(payload) <= 16:
        return True
    nested_message = json_document(payload.get("message"))
    return bool(nested_message) and any(key in nested_message for key in _MESSAGE_KEYS)


def looks_like_hook_event(payload: object) -> bool:
    """Detect if a payload is a hook event record.

    Hook events have a canonical shape with event_type, session_id,
    timestamp, and provider fields. This detects both Claude Code (16
    events) and Codex (6 events) hook artifacts.
    """
    if not isinstance(payload, dict):
        return False
    if not isinstance(payload.get("event_type"), str):
        return False
    if not isinstance(payload.get("session_id"), str):
        return False
    if not isinstance(payload.get("timestamp"), str):
        return False
    provider = payload.get("provider")
    return isinstance(provider, str) and provider in ("claude-code", "codex")


def looks_like_hook_event_stream(payload: list[JSONDocument]) -> bool:
    """Detect if a JSONL list is a stream of hook event records."""
    if not payload:
        return False
    recordish = sum(1 for item in payload if looks_like_hook_event(item))
    return recordish == len(payload) and recordish >= 1


def looks_like_beads_interaction(payload: object) -> bool:
    """Return whether a record is one append-only Beads interaction."""
    if not isinstance(payload, dict) or not _BEADS_INTERACTION_KEYS.issubset(payload):
        return False
    return (
        isinstance(payload.get("id"), str)
        and isinstance(payload.get("kind"), str)
        and isinstance(payload.get("created_at"), str)
        and isinstance(payload.get("issue_id"), str)
        and isinstance(payload.get("extra"), dict)
    )


def looks_like_file_history_snapshot_only_stream(dict_items: list[JSONDocument]) -> bool:
    """True when every decoded record's ``type`` is a file-history checkpoint.

    ``dict_items`` should be the decoded records of a Claude Code
    ``projects/<proj>/<uuid>.jsonl`` stream (or a bounded prefix of one).
    Empty input is not positive evidence either way.
    """
    if not dict_items:
        return False
    types = {item.get("type") for item in dict_items if isinstance(item.get("type"), str)}
    return bool(types) and types <= _FILE_HISTORY_SNAPSHOT_ONLY_TYPES


def looks_like_message_entry(payload: object) -> bool:
    return isinstance(payload, dict) and any(key in payload for key in _MESSAGE_KEYS)


def looks_metadataish_dict(payload: JSONDocument) -> bool:
    if not payload:
        return True
    if len(payload) > 20:
        return False
    if looks_like_record_entry(payload):
        return False
    if looks_like_session_document(payload):
        return False
    return all(is_scalarish(value) for value in payload.values())


def looks_metadataish_list(payload: list[JSONValue]) -> bool:
    if not payload:
        return True
    # A bounded 513-item peek answers "more than 512 items?" without forcing
    # ``len(payload)`` -- for a lazy full-corpus record stream
    # (``ReplayableRecordSamples``) that would otherwise decode the entire
    # backing file just to classify one artifact.
    head = list(islice(payload, 513))
    if len(head) > 512:
        return False
    return all(
        isinstance(item, _SCALAR_TYPES) or (isinstance(item, dict) and looks_metadataish_dict(item))
        for item in head[:64]
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

"""Shared artifact-taxonomy heuristics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from polylogue.core.json import JSONDocument, JSONValue, json_document

if TYPE_CHECKING:
    from polylogue.sources.detection_projection import DetectorProjection

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
#: Top-level keys whose string value names the transcript a record's content
#: was copied out of. A generated extract carries this reference because its
#: rows are not its own content; a provider's wire record never names an
#: external transcript as the origin of the turn it transmits, because it
#: carries that turn. This is the positive evidence that separates an
#: analytical derivative from the session it was derived from, at any path.
_EXTRACTED_PROVENANCE_KEYS = frozenset({"file", "source_file", "source_path", "transcript", "session_file"})
_TRANSCRIPT_REFERENCE_SUFFIXES = (".jsonl", ".jsonl.txt", ".ndjson", ".json")
_COPIED_CONTENT_KEYS = ("content", "text", "message_text", "body")


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

    # A claude.ai project (``projects/<uuid>.json``) and account-memory record
    # carry no message list, yet each is a session the claude.ai parser
    # admits. Deciding them here with the parser's own detectors keeps the
    # classifier from turning them away before any parser runs. Deferred for
    # the same artifact-taxonomy/sources import cycle ``runtime.py`` notes.
    from polylogue.sources.parsers.claude import looks_like_claude_memories, looks_like_claude_project

    if looks_like_claude_project(payload) or looks_like_claude_memories(payload):
        return True

    messages = payload.get("messages")
    return isinstance(messages, list) and any(looks_like_message_entry(item) for item in messages)


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


def looks_like_extracted_transcript_record(payload: object) -> bool:
    """Return whether one record is conversation text copied out of a named
    transcript rather than a turn a provider transmitted.

    Three conditions, all required: no provider-native record envelope (the
    record is not itself a turn), a declared reference naming the transcript
    the content came from, and the copied content itself. A pointer/index row
    carrying provenance but no copied turn is somebody else's shape --
    ``looks_like_record_entry`` already refuses it -- and is deliberately not
    claimed here.
    """
    if not isinstance(payload, dict):
        return False
    if any(key in payload for key in _TYPE_ENVELOPE_MARKERS):
        return False
    reference = next(
        (value for key in _EXTRACTED_PROVENANCE_KEYS if isinstance(value := payload.get(key), str)),
        None,
    )
    if reference is None or not reference.lower().endswith(_TRANSCRIPT_REFERENCE_SUFFIXES):
        return False
    return any(isinstance(payload.get(key), str) and payload[key] for key in _COPIED_CONTENT_KEYS)


def record_carries_provider_envelope(payload: object) -> bool:
    """True when a record carries a provider-native envelope marker.

    One such record disqualifies a stream from the extracted-corpus rule, so
    a bounded scan may stop at the first one it sees.
    """
    return isinstance(payload, dict) and any(key in payload for key in _TYPE_ENVELOPE_MARKERS)


def looks_like_extracted_transcript_corpus(dict_items: Iterable[JSONDocument]) -> bool:
    """True when decoded records are an extract of other transcripts.

    At least one record must carry the positive extraction evidence, and no
    record may carry a provider-native envelope marker: one genuine wire
    record disqualifies the whole stream, so a real session can never be
    refused by this rule, and a stream mixing extracted rows with rows this
    taxonomy cannot name stays refused rather than guessing.
    """
    extracted = False
    for item in dict_items:
        if record_carries_provider_envelope(item):
            return False
        extracted = extracted or looks_like_extracted_transcript_record(item)
    return extracted


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


def looks_like_file_history_snapshot_only_stream(dict_items: Iterable[JSONDocument]) -> bool:
    """True when every decoded record's ``type`` is a file-history checkpoint.

    ``dict_items`` must be the decoded records of a **complete** Claude Code
    ``projects/<proj>/<uuid>.jsonl`` stream. A bounded prefix is not admissible
    evidence: this predicate's caller uses a positive result to override a
    positive session verdict, so a prefix of checkpoints followed by real
    conversational records would drop a genuine session from ingest with no
    gap recorded.

    The scan is lazy and exits on the first record that is not a checkpoint, so
    a real session -- which reaches a ``user``/``assistant`` record almost
    immediately -- costs no more than the old 32-record prefix did.

    Empty input is not positive evidence either way.
    """
    saw_checkpoint = False
    for item in dict_items:
        if not item:
            continue
        record_type = item.get("type")
        if not isinstance(record_type, str):
            continue
        if record_type not in _FILE_HISTORY_SNAPSHOT_ONLY_TYPES:
            return False
        saw_checkpoint = True
    return saw_checkpoint


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
    complete_values = getattr(payload, "metadata_values_scalarish", None)
    if isinstance(complete_values, bool):
        return complete_values
    return all(is_scalarish(value) for value in payload.values())


def looks_metadataish_list(payload: Sequence[JSONValue]) -> bool:
    """Require metadata evidence from every element, without a prefix verdict."""
    return all(
        isinstance(item, _SCALAR_TYPES) or (isinstance(item, dict) and looks_metadataish_dict(item)) for item in payload
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


def record_candidacy_projection() -> DetectorProjection:
    """Declare the fields and complete folds used by artifact candidacy.

    This projection supplies admission evidence, never schema validation or
    parser material. The canonical parser still consumes the original bytes.
    """
    from polylogue.sources.detection_projection import DetectorProjection
    from polylogue.sources.parsers import grok
    from polylogue.sources.parsers.chatgpt_codex_sidecar import looks_like as looks_like_codex_task

    scalar = DetectorProjection()
    fields: dict[str, DetectorProjection | None] = dict.fromkeys(
        _RECORDISH_KEYS
        | _TYPE_ENVELOPE_MARKERS
        | _MESSAGE_KEYS
        | _RELATIONSHIP_INDEX_KEYS
        | _RELATIONSHIP_INDEX_KEYS_CONVERSATION
        | _HOOK_EVENT_KEYS
        | _BEADS_INTERACTION_KEYS
        | _EXTRACTED_PROVENANCE_KEYS
        | frozenset(_COPIED_CONTENT_KEYS)
        | {
            "id",
            "polylogue_capture_kind",
            "mapping",
            "chat_messages",
            "chunkedPrompt",
            "chunks",
            "source",
            "cascadeId",
            "markdown",
            "account_uuid",
            "conversations_memory",
            "project_memories",
            "docs",
            "prompt_template",
            "is_starter_project",
            "atof_version",
            "kind",
            "name",
            "schema_version",
            "steps",
            "polylogue_artifact",
        },
        scalar,
    )
    fields["messages"] = DetectorProjection(
        item=DetectorProjection(fields=dict.fromkeys(_MESSAGE_KEYS)),
        array_fold="any",
        array_predicate=looks_like_message_entry,
    )
    fields["turns"] = DetectorProjection(
        item=DetectorProjection(fields={"id": scalar, "role": scalar}),
        array_fold="all",
        array_predicate=lambda item: looks_like_codex_task({"id": "task_e_projection", "turns": [item]}),
    )
    fields.update(grok.detection_projection().fields or {})
    fields.update(grok.native_detection_projection().fields or {})
    return DetectorProjection(fields=fields, preserve_mapping_size=True, capture_metadata_values=True)

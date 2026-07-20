"""Parser for xAI Grok account-data export documents.

Wire shape (no official xAI schema publication exists; this is reconstructed
from independent evidence and cross-checked across sources rather than taken
from a single blog post):

* https://github.com/beejaksharam/grok-export-viewer — an open-source tool
  whose ``grok_export_viewer/core.py`` parses the real export file (named
  ``prod-grok-backend.json`` by xAI) with executable, tested code:
  ``data["conversations"]`` -> each item has ``item["conversation"]["title"]``
  and ``item["responses"]``, each response nested as
  ``r["response"]["sender"]`` / ``r["response"]["message"]``.
* https://ai-chat-importer.com/blog/how-to-export-grok-conversations — shows
  the same top-level ``conversations`` / ``conversation`` / ``responses``
  shape (with a flatter per-response layout: ``sender``/``message`` directly
  on the response entry rather than nested under ``response``) and documents
  ``create_time`` as MongoDB extended JSON (``{"$date": {"$numberLong":
  "<epoch-ms>"}}``) plus the explicit absence of a native conversation id or
  attachment/image data in the export.
* A live grok.com API-scraping userscript
  (https://greasyfork.org/en/scripts/571847) independently confirms
  ``sender``/``message``/``createTime`` field names and ``sender == "human"``
  for the user turn.

This parser accepts both the nested (``{"response": {...}}``) and flat
per-response shapes, and both MongoDB extended-JSON and plain (ISO/epoch)
timestamps, since the shape is reconstructed from secondary sources rather
than one authoritative spec.

Neither the conversation nor its responses carry a native id in any of the
confirmed shapes above, so ``provider_session_id``/``provider_message_id``
are synthesized from the dispatch-provided fallback id and positional index
within the export file -- the same convention used elsewhere in this codebase
for id-less sources (e.g. Antigravity brain-artifact fragments in
``antigravity.py``). This is stable across re-ingest of the same physical
export file (whose per-conversation position does not change) but is not a
provider-native identity.
"""

from __future__ import annotations

from collections.abc import Mapping

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.core.timestamps import canonical_timestamp_text

from .base import ParsedContentBlock, ParsedMessage, ParsedSession

_SENDER_ROLE: dict[str, Role] = {
    "human": Role.USER,
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "grok": Role.ASSISTANT,
}


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _response_fields(entry: object) -> Mapping[str, object]:
    """Unwrap the ``{"response": {...}}`` nesting used by the real export.

    Some secondary documentation shows responses flattened one level
    (``{"sender": ..., "message": ...}`` directly on the entry); both shapes
    are accepted.
    """
    entry_map = _mapping(entry)
    nested = entry_map.get("response")
    return _mapping(nested) if isinstance(nested, Mapping) else entry_map


def _timestamp_text(value: object) -> str | None:
    if isinstance(value, Mapping):
        date_value = value.get("$date")
        if isinstance(date_value, Mapping):
            number_long = date_value.get("$numberLong")
            if number_long is None:
                return None
            try:
                epoch_ms = int(number_long)
            except (TypeError, ValueError):
                return None
            return canonical_timestamp_text(epoch_ms / 1000)
        if isinstance(date_value, (str, int, float)):
            return canonical_timestamp_text(date_value)
        return None
    if isinstance(value, (str, int, float)):
        return canonical_timestamp_text(value)
    return None


def _role_for_sender(sender: object) -> Role:
    if not isinstance(sender, str) or not sender:
        return Role.UNKNOWN
    role = _SENDER_ROLE.get(sender.strip().lower())
    return role if role is not None else Role.normalize(sender)


def looks_like_conversation(payload: object) -> bool:
    """Return whether ``payload`` is a single Grok export conversation entry."""
    if not isinstance(payload, Mapping):
        return False
    conversation = payload.get("conversation")
    responses = payload.get("responses")
    return isinstance(conversation, Mapping) and isinstance(responses, list)


def looks_like_export(payload: object) -> bool:
    """Return whether ``payload`` is a top-level Grok account-data export document."""
    if not isinstance(payload, Mapping):
        return False
    conversations = payload.get("conversations")
    if not isinstance(conversations, list):
        return False
    if not conversations:
        return True
    return any(looks_like_conversation(item) for item in conversations)


def parse_conversation(payload: Mapping[str, object], fallback_id: str) -> ParsedSession:
    """Parse a single Grok export conversation entry into a session."""
    conversation = _mapping(payload.get("conversation"))
    responses_raw = payload.get("responses")
    responses = responses_raw if isinstance(responses_raw, list) else []

    title_raw = conversation.get("title")
    title = title_raw if isinstance(title_raw, str) and title_raw else fallback_id
    created_at = _timestamp_text(conversation.get("create_time"))

    messages: list[ParsedMessage] = []
    for index, entry in enumerate(responses):
        fields = _response_fields(entry)
        text_raw = fields.get("message")
        text = text_raw if isinstance(text_raw, str) else None
        if not text:
            continue
        provider_message_id = f"{fallback_id}:{index}"
        messages.append(
            ParsedMessage(
                provider_message_id=provider_message_id,
                role=_role_for_sender(fields.get("sender")),
                text=text,
                timestamp=_timestamp_text(fields.get("create_time")),
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                position=len(messages),
                variant_index=0,
                is_active_path=True,
            )
        )

    active_leaf_message_provider_id = messages[-1].provider_message_id if messages else None
    if active_leaf_message_provider_id is not None:
        messages = [
            message.model_copy(
                update={"is_active_leaf": message.provider_message_id == active_leaf_message_provider_id}
            )
            for message in messages
        ]
    updated_at = messages[-1].timestamp if messages and messages[-1].timestamp else created_at

    return ParsedSession(
        source_name=Provider.GROK,
        provider_session_id=fallback_id,
        title=title,
        created_at=created_at,
        updated_at=updated_at,
        messages=messages,
        active_leaf_message_provider_id=active_leaf_message_provider_id,
    )


__all__ = [
    "looks_like_conversation",
    "looks_like_export",
    "parse_conversation",
]

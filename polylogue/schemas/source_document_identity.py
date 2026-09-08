"""Provider-declared document identities for source revision selection."""

from __future__ import annotations

from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.core.json import JSONDocument
from polylogue.sources.parsers.hermes_identity import (
    profile_key,
    profile_root_for_session_snapshot,
    qualified_session_id,
)
from polylogue.sources.parsers.hermes_spans import atif_session_provider_id, looks_like_atif_payload
from polylogue.sources.parsers.local_agent import gemini_cli_chat_identity

DOCUMENT_UPDATE_FIELDS: dict[Provider, tuple[str, ...]] = {
    Provider.GEMINI: ("updateTime",),
    Provider.DRIVE: ("updateTime",),
    Provider.GEMINI_CLI: ("lastUpdated",),
    Provider.HERMES: ("last_updated",),
    Provider.ANTIGRAVITY: ("lastModifiedTime",),
}


def native_document_identity(provider: Provider, payload: JSONDocument, source_path: Path) -> str | None:
    """Use the parser's identity coordinates without materializing session content."""
    if provider is Provider.GEMINI_CLI:
        session_id = payload.get("sessionId")
        if isinstance(session_id, str) and session_id:
            return f"gemini-cli:{gemini_cli_chat_identity(payload, session_id)}"
    elif provider is Provider.HERMES:
        session_id = payload.get("session_id")
        if isinstance(session_id, str) and session_id:
            if looks_like_atif_payload(payload):
                identity = atif_session_provider_id(session_id, profile_key(source_path.parent))
            else:
                identity = qualified_session_id(session_id, profile_key(profile_root_for_session_snapshot(source_path)))
            return f"hermes:{identity}"
    elif provider in {Provider.GEMINI, Provider.DRIVE}:
        session_id = payload.get("id")
        if isinstance(session_id, str) and session_id:
            return f"gemini:{session_id}"
    elif provider is Provider.ANTIGRAVITY:
        session_id = payload.get("cascadeId")
        if isinstance(session_id, str) and session_id:
            return f"antigravity:{session_id}"
    return None

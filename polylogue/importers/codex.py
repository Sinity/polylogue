"""Codex JSONL session importer using typed Pydantic models.

Uses CodexRecord from polylogue.providers.codex for type-safe parsing
with automatic validation and normalization.
"""

from __future__ import annotations

from pydantic import ValidationError

from polylogue.providers.codex import CodexRecord
from .base import ParsedConversation, ParsedMessage


def looks_like(payload: list[object]) -> bool:
    """Detect Codex JSONL format using typed validation.

    Newest format (envelope with typed payloads):
        {"type":"session_meta","payload":{"id":"...","timestamp":"...","git":{...}}}
        {"type":"response_item","payload":{"type":"message","role":"user","content":[...]}}

    Intermediate format (JSONL with session metadata + messages):
        {"id":"...","timestamp":"...","git":{...}}
        {"record_type":"state"}
        {"type":"message","role":"user","content":[...]}
    """
    if not isinstance(payload, list):
        return False

    for item in payload:
        if not isinstance(item, dict):
            continue

        try:
            record = CodexRecord.model_validate(item)
            # Check for known Codex format indicators
            if record.format_type in ("envelope", "direct", "state"):
                return True
            # Session metadata (first line of intermediate format)
            if record.id and record.timestamp:
                return True
        except ValidationError:
            continue

    return False


def parse(payload: list[object], fallback_id: str) -> ParsedConversation:
    """Parse Codex JSONL session file using typed CodexRecord model.

    Supports two format generations via CodexRecord.format_type:
    - "envelope": {"type":"session_meta"|"response_item", "payload":{...}}
    - "direct": {"type":"message", "role":"...", "content":[...]}
    - "state": {"record_type":"state"} (skip markers)

    The CodexRecord model handles format normalization via properties:
    - effective_role: Normalized role from any format
    - text_content: Extracted text from any format
    - format_type: Detected format generation
    """
    messages: list[ParsedMessage] = []
    session_id = fallback_id
    session_timestamp: str | None = None
    session_metas_seen: list[str] = []  # Collect all session_meta IDs for parent tracking

    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue

        try:
            record = CodexRecord.model_validate(item)
        except ValidationError:
            # Skip invalid records
            continue

        # Handle session metadata (envelope format)
        if record.type == "session_meta" and record.payload:
            meta_id = record.payload.get("id")
            if meta_id and meta_id not in session_metas_seen:
                session_metas_seen.append(meta_id)
                # First session_meta sets the conversation ID
                if len(session_metas_seen) == 1:
                    session_id = str(meta_id)
                    session_timestamp = record.payload.get("timestamp")
            continue

        # Handle session metadata (intermediate format - first line with id+timestamp)
        if record.id and record.timestamp and not record.type:
            if record.id not in session_metas_seen:
                session_metas_seen.append(record.id)
                if len(session_metas_seen) == 1:
                    session_id = record.id
                    session_timestamp = record.timestamp
            continue

        # Skip state markers
        if record.format_type == "state":
            continue

        # Handle messages - either from envelope or direct format
        if record.type == "response_item" and record.payload:
            # Envelope format: unwrap payload and create new record for message
            inner_payload = record.payload
            if inner_payload.get("type") == "message":
                try:
                    record = CodexRecord.model_validate(inner_payload)
                except ValidationError:
                    continue

        # Parse message records using typed properties
        if record.is_message:
            role = record.effective_role
            text = record.text_content

            if not role or role == "unknown" or not text:
                continue

            # Get message ID from the record
            msg_id = record.id or f"msg-{idx}"

            messages.append(
                ParsedMessage(
                    provider_message_id=msg_id,
                    role=role,
                    text=text,
                    timestamp=record.timestamp,
                    provider_meta={"raw": item},  # Preserve original for re-parsing
                )
            )

    # Second session_meta ID (if present) is the parent session
    parent_id = session_metas_seen[1] if len(session_metas_seen) > 1 else None
    branch_type = "continuation" if parent_id else None

    return ParsedConversation(
        provider_name="codex",
        provider_conversation_id=session_id,
        title=session_id,
        created_at=session_timestamp,
        updated_at=None,
        messages=messages,
        parent_conversation_provider_id=parent_id,
        branch_type=branch_type,
    )

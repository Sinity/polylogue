"""Codex JSONL session parser using typed Pydantic models.

Uses CodexRecord from polylogue.sources.providers.codex for type-safe parsing
with automatic validation and normalization.
"""

from __future__ import annotations

from pydantic import ValidationError

from polylogue.lib.log import get_logger
from polylogue.sources.providers.codex import CodexRecord

from .base import ParsedConversation, ParsedMessage, normalize_role

logger = get_logger(__name__)


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
    session_git: dict[str, object] | None = None  # Git context from session metadata
    session_instructions: str | None = None  # System instructions from session metadata

    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue

        try:
            record = CodexRecord.model_validate(item)
        except ValidationError as exc:
            logger.debug("Skipping invalid record at index %d: %s", idx, exc)
            continue

        # Handle session metadata (envelope format)
        if record.type == "session_meta" and record.payload:
            meta_id = record.payload.get("id")
            if meta_id and meta_id not in session_metas_seen:
                session_metas_seen.append(meta_id)
                # First session_meta sets the conversation ID and captures session-level data
                if len(session_metas_seen) == 1:
                    session_id = str(meta_id)
                    session_timestamp = record.payload.get("timestamp")
            # Capture git/instructions from envelope payload (may appear in inner record)
            inner = CodexRecord.model_validate(record.payload) if isinstance(record.payload, dict) else None
            if inner and inner.git and not session_git:
                session_git = inner.git.model_dump(exclude_none=True) or None
            if inner and inner.instructions and not session_instructions:
                session_instructions = inner.instructions
            continue

        # Handle session metadata (intermediate format - first line with id+timestamp)
        if record.id and record.timestamp and not record.type:
            if record.id not in session_metas_seen:
                session_metas_seen.append(record.id)
                if len(session_metas_seen) == 1:
                    session_id = record.id
                    session_timestamp = record.timestamp
            # Capture git/instructions from top-level record
            if record.git and not session_git:
                session_git = record.git.model_dump(exclude_none=True) or None
            if record.instructions and not session_instructions:
                session_instructions = record.instructions
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
                except ValidationError as exc:
                    logger.debug("Skipping invalid envelope payload at index %d: %s", idx, exc)
                    continue

        # Parse message records using typed properties
        if record.is_message:
            raw_role = record.effective_role
            text = record.text_content

            if not raw_role or raw_role == "unknown" or not text:
                continue
            role = normalize_role(raw_role)

            # Get message ID from the record
            msg_id = record.id or f"msg-{idx}"

            # Build provider_meta with raw data and structured metadata
            msg_meta: dict[str, object] = {"raw": item}
            if record.git:
                git_info = record.git.model_dump(exclude_none=True)
                if git_info:
                    msg_meta["git"] = git_info
            if record.instructions:
                msg_meta["instructions"] = record.instructions

            messages.append(
                ParsedMessage(
                    provider_message_id=msg_id,
                    role=role,
                    text=text,
                    timestamp=record.timestamp,
                    provider_meta=msg_meta,
                )
            )

    # Second session_meta ID (if present) is the parent session
    parent_id = session_metas_seen[1] if len(session_metas_seen) > 1 else None
    branch_type = "continuation" if parent_id else None

    # Build conversation-level provider_meta with session context
    conv_meta: dict[str, object] | None = None
    if session_git or session_instructions:
        conv_meta = {}
        if session_git:
            conv_meta["git"] = session_git
        if session_instructions:
            conv_meta["instructions"] = session_instructions

    return ParsedConversation(
        provider_name="codex",
        provider_conversation_id=session_id,
        title=session_id,
        created_at=session_timestamp,
        updated_at=None,
        messages=messages,
        provider_meta=conv_meta,
        parent_conversation_provider_id=parent_id,
        branch_type=branch_type,
    )

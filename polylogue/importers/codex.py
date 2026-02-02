from __future__ import annotations

from .base import ParsedConversation, ParsedMessage


def looks_like(payload: list[object]) -> bool:
    """Detect Codex JSONL format.

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
        # Newest format: envelope types
        if item.get("type") in ("session_meta", "response_item"):
            return True
        # Intermediate format: session metadata or message records
        if "id" in item and "timestamp" in item:
            return True
        if item.get("type") == "message":
            return True

    return False


def parse(payload: list[object], fallback_id: str) -> ParsedConversation:
    """Parse Codex JSONL session file.

    Supports two format generations:

    Newest (envelope format):
        {"type":"session_meta","payload":{"id":"...","timestamp":"...","git":{...}}}
        {"type":"response_item","payload":{"type":"message","role":"user","content":[...]}}

    Intermediate (direct records):
        {"id":"...","timestamp":"...","git":{...}}
        {"record_type":"state"}
        {"type":"message","role":"user","content":[...]}
    """
    messages: list[ParsedMessage] = []
    session_id = fallback_id
    session_timestamp = None
    session_metas_seen: list[str] = []  # Collect all session_meta IDs for parent tracking

    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue

        # Newest format: envelope with "session_meta" type
        # Collect ALL session_meta IDs - child sessions include parent session_meta
        if item.get("type") == "session_meta":
            envelope_payload = item.get("payload", {})
            if isinstance(envelope_payload, dict):
                meta_id = envelope_payload.get("id")
                if meta_id and meta_id not in session_metas_seen:
                    session_metas_seen.append(meta_id)
                    # First session_meta sets the conversation ID
                    if len(session_metas_seen) == 1:
                        session_id = meta_id
                        session_timestamp = envelope_payload.get("timestamp", session_timestamp)
            continue

        # Newest format: envelope with "response_item" type
        if item.get("type") == "response_item":
            envelope_payload = item.get("payload", {})
            if isinstance(envelope_payload, dict) and envelope_payload.get("type") == "message":
                # Unwrap and process as message
                item = envelope_payload
                # Fall through to message processing below

        # Skip state markers (intermediate format)
        if item.get("record_type") == "state":
            continue

        # Extract session metadata (intermediate format - first line)
        if "id" in item and "timestamp" in item and not item.get("type"):
            meta_id = item["id"]
            if meta_id and meta_id not in session_metas_seen:
                session_metas_seen.append(meta_id)
                # First session_meta sets the conversation ID
                if len(session_metas_seen) == 1:
                    session_id = meta_id
                    session_timestamp = item.get("timestamp")
            continue

        # Parse message records (both newest and intermediate formats)
        if item.get("type") == "message":
            role = item.get("role")
            content = item.get("content", [])

            if not role or not content:
                continue

            # Extract text from content array
            # Content is list of {"type":"input_text","text":"..."}
            text_parts = []
            for content_item in content:
                if isinstance(content_item, dict):
                    if content_item.get("type") == "input_text":
                        text = content_item.get("text", "")
                        if text:
                            text_parts.append(text)

            if text_parts:
                messages.append(
                    ParsedMessage(
                        provider_message_id=item.get("id") or f"msg-{idx}",
                        role=role,
                        text="\n".join(text_parts),
                        timestamp=item.get("timestamp"),
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

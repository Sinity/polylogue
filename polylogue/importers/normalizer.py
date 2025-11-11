"""Helper utilities for building normalised message records in importers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set, Tuple

from ..branching import MessageRecord

AttachmentLink = Tuple[str, object]


def _hash_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stringify_link(link: object) -> str:
    if isinstance(link, Path):
        return str(link)
    return str(link) if link is not None else ""


def ensure_message_id(
    candidate: Optional[str],
    *,
    fallback_prefix: str,
    index: int,
    seen: Set[str],
) -> str:
    """Return a deterministic message id that does not collide with previous ids."""

    base = candidate or f"{fallback_prefix}-msg-{index:04d}"
    base = str(base)
    if not base:
        base = f"{fallback_prefix}-msg-{index:04d}"
    message_id = base
    suffix = 1
    while message_id in seen:
        message_id = f"{base}-dup{suffix}"
        suffix += 1
    seen.add(message_id)
    return message_id


def build_message_record(
    *,
    provider: str,
    conversation_id: Optional[str],
    chunk_index: int,
    chunk: Dict[str, Any],
    raw_metadata: Optional[Dict[str, Any]],
    attachments: Sequence[AttachmentLink],
    tool_calls: Optional[Sequence[Dict[str, Any]]] = None,
    seen_ids: Optional[Set[str]] = None,
    fallback_prefix: Optional[str] = None,
) -> MessageRecord:
    """Create a normalised MessageRecord with consistent metadata and hashes."""

    seen = seen_ids if seen_ids is not None else set()
    prefix = fallback_prefix or conversation_id or provider or "conversation"
    raw_id = chunk.get("messageId") or (raw_metadata or {}).get("id")
    message_id = ensure_message_id(raw_id, fallback_prefix=prefix, index=chunk_index, seen=seen)
    chunk.setdefault("messageId", message_id)

    text = chunk.get("text") or ""
    token_count = int(chunk.get("tokenCount") or 0)
    branch_hint = chunk.get("branchParent")
    parent_id = chunk.get("parentId") or (raw_metadata or {}).get("parent_id")

    attachment_meta = [
        {"name": name, "link": _stringify_link(link)}
        for name, link in attachments
    ]
    metadata: Dict[str, Any] = {
        "provider": provider,
        "attachments": attachment_meta,
    }
    if raw_metadata:
        metadata["raw"] = raw_metadata
    if tool_calls:
        metadata["tool_calls"] = [dict(call) for call in tool_calls]
    if branch_hint:
        metadata["branch_hint"] = branch_hint

    return MessageRecord(
        message_id=message_id,
        parent_id=parent_id,
        role=chunk.get("role") or "assistant",
        text=text,
        token_count=token_count,
        word_count=len(text.split()),
        timestamp=chunk.get("timestamp"),
        attachments=len(attachments),
        chunk=chunk,
        links=list(attachments),
        metadata=metadata,
        branch_hint=branch_hint,
        content_hash=_hash_text(text),
    )


__all__ = ["AttachmentLink", "build_message_record", "ensure_message_id"]

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import Source


@dataclass
class ParsedMessage:
    provider_message_id: str
    role: str
    text: str
    timestamp: Optional[str]
    provider_meta: Optional[dict]


@dataclass
class ParsedAttachment:
    provider_attachment_id: str
    message_provider_id: Optional[str]
    name: Optional[str]
    mime_type: Optional[str]
    size_bytes: Optional[int]
    path: Optional[str]
    provider_meta: Optional[dict]


@dataclass
class ParsedConversation:
    provider_name: str
    provider_conversation_id: str
    title: str
    created_at: Optional[str]
    updated_at: Optional[str]
    messages: List[ParsedMessage]
    attachments: List[ParsedAttachment] = field(default_factory=list)


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _load_json_from_path(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        items = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return items
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_from_zip(path: Path) -> Any:
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith("conversations.json"):
                with zf.open(name) as handle:
                    return json.loads(handle.read().decode("utf-8"))
            if name.endswith(".jsonl"):
                with zf.open(name) as handle:
                    lines = handle.read().decode("utf-8").splitlines()
                    items = []
                    for line in lines:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                    return items
    raise RuntimeError(f"No supported JSON payload in {path}")


def _extract_messages_from_mapping(mapping: dict) -> List[ParsedMessage]:
    nodes = []
    for node in mapping.values():
        if not isinstance(node, dict):
            continue
        msg = node.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts") or []
        if not isinstance(parts, list):
            continue
        text = "\n".join(str(part) for part in parts if part)
        role = msg.get("author", {}).get("role") or "user"
        timestamp = msg.get("create_time")
        msg_id = msg.get("id") or node.get("id") or ""
        if not msg_id:
            msg_id = f"msg-{len(nodes) + 1}"
        nodes.append(
            ParsedMessage(
                provider_message_id=str(msg_id),
                role=str(role),
                text=text,
                timestamp=str(timestamp) if timestamp is not None else None,
                provider_meta={"raw": msg},
            )
        )
    return nodes


def _extract_messages_from_list(items: list) -> List[ParsedMessage]:
    messages: List[ParsedMessage] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        text = None
        role = None
        timestamp = item.get("timestamp") or item.get("created_at") or item.get("create_time")
        if "text" in item and isinstance(item["text"], str):
            text = item["text"]
        elif "content" in item:
            content = item["content"]
            if isinstance(content, str):
                text = content
            elif isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list):
                    text = "\n".join(str(part) for part in parts)
        if "role" in item:
            role = item.get("role")
        if text:
            messages.append(
                ParsedMessage(
                    provider_message_id=str(item.get("id") or f"msg-{idx}"),
                    role=str(role or "message"),
                    text=text,
                    timestamp=str(timestamp) if timestamp is not None else None,
                    provider_meta={"raw": item},
                )
            )
    return messages


def _normalize_role(role: Optional[str]) -> str:
    if not role:
        return "message"
    lowered = str(role).strip().lower()
    if lowered in {"assistant", "model"}:
        return "assistant"
    if lowered in {"user", "human"}:
        return "user"
    if lowered in {"system"}:
        return "system"
    return lowered


def _extract_text_from_chunk(chunk: dict) -> Optional[str]:
    for key in ("text", "content", "message", "markdown", "data"):
        value = chunk.get(key)
        if isinstance(value, str):
            return value
    parts = chunk.get("parts")
    if isinstance(parts, list):
        return "\n".join(str(part) for part in parts if part)
    return None


def _collect_drive_docs(payload: object) -> List[dict | str]:
    docs: List[dict | str] = []
    if not isinstance(payload, dict):
        return docs
    for key in ("driveDocument", "driveDocuments", "drive_document"):
        value = payload.get(key)
        if isinstance(value, (dict, str)):
            docs.append(value)
        elif isinstance(value, list):
            docs.extend([item for item in value if isinstance(item, (dict, str))])
    nested = payload.get("metadata")
    if isinstance(nested, dict):
        docs.extend(_collect_drive_docs(nested))
    return docs


def _attachment_from_doc(doc: dict | str, message_id: Optional[str]) -> Optional[ParsedAttachment]:
    if isinstance(doc, str):
        doc_id = doc
        meta = {"id": doc_id}
        return ParsedAttachment(
            provider_attachment_id=doc_id,
            message_provider_id=message_id,
            name=None,
            mime_type=None,
            size_bytes=None,
            path=None,
            provider_meta=meta,
        )
    if not isinstance(doc, dict):
        return None
    doc_id = doc.get("id") or doc.get("fileId") or doc.get("driveId")
    if not isinstance(doc_id, str) or not doc_id:
        return None
    size_raw = doc.get("sizeBytes") or doc.get("size")
    size_bytes = None
    if isinstance(size_raw, (int, str)):
        try:
            size_bytes = int(size_raw)
        except ValueError:
            size_bytes = None
    return ParsedAttachment(
        provider_attachment_id=doc_id,
        message_provider_id=message_id,
        name=doc.get("name") or doc.get("title"),
        mime_type=doc.get("mimeType") or doc.get("mime_type"),
        size_bytes=size_bytes,
        path=None,
        provider_meta=doc,
    )


def _parse_chunked_prompt(provider: str, payload: dict, fallback_id: str) -> ParsedConversation:
    prompt = payload.get("chunkedPrompt")
    chunks = []
    if isinstance(prompt, dict):
        chunks = prompt.get("chunks") or []
    elif isinstance(payload.get("chunks"), list):
        chunks = payload.get("chunks")  # type: ignore[assignment]
    messages: List[ParsedMessage] = []
    attachments: List[ParsedAttachment] = []
    for idx, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, str):
            chunk_obj = {"text": chunk}
        elif isinstance(chunk, dict):
            chunk_obj = chunk
        else:
            continue
        text = _extract_text_from_chunk(chunk_obj)
        if not text:
            continue
        role = _normalize_role(chunk_obj.get("role") or chunk_obj.get("author") or chunk_obj.get("speaker"))
        msg_id = chunk_obj.get("id") or chunk_obj.get("messageId") or f"chunk-{idx}"
        msg_id = str(msg_id)
        messages.append(
            ParsedMessage(
                provider_message_id=msg_id,
                role=role,
                text=text,
                timestamp=None,
                provider_meta={"raw": chunk_obj},
            )
        )
        for doc in _collect_drive_docs(chunk_obj):
            attachment = _attachment_from_doc(doc, msg_id)
            if attachment:
                attachments.append(attachment)
    if not messages:
        input_text = payload.get("inputText") or payload.get("prompt")
        output_text = payload.get("outputText") or payload.get("response")
        if isinstance(input_text, str) and input_text:
            messages.append(
                ParsedMessage(
                    provider_message_id="chunk-1",
                    role="user",
                    text=input_text,
                    timestamp=None,
                    provider_meta=None,
                )
            )
        if isinstance(output_text, str) and output_text:
            messages.append(
                ParsedMessage(
                    provider_message_id="chunk-2",
                    role="assistant",
                    text=output_text,
                    timestamp=None,
                    provider_meta=None,
                )
            )
    title = payload.get("title") or payload.get("displayName") or payload.get("name") or fallback_id
    return ParsedConversation(
        provider_name=provider,
        provider_conversation_id=str(payload.get("id") or payload.get("name") or fallback_id),
        title=str(title),
        created_at=str(payload.get("createTime")) if payload.get("createTime") else None,
        updated_at=str(payload.get("updateTime")) if payload.get("updateTime") else None,
        messages=messages,
        attachments=attachments,
    )


def _parse_conversation_obj(provider: str, obj: dict, fallback_id: str) -> ParsedConversation:
    mapping = obj.get("mapping") if isinstance(obj, dict) else None
    if isinstance(mapping, dict):
        messages = _extract_messages_from_mapping(mapping)
    else:
        msgs = obj.get("messages") if isinstance(obj, dict) else None
        if isinstance(msgs, list):
            messages = _extract_messages_from_list(msgs)
        else:
            messages = _extract_messages_from_list([obj])
    title = obj.get("title") if isinstance(obj, dict) else None
    conv_id = obj.get("id") if isinstance(obj, dict) else None
    return ParsedConversation(
        provider_name=provider,
        provider_conversation_id=str(conv_id or fallback_id),
        title=str(title or fallback_id),
        created_at=str(obj.get("create_time")) if isinstance(obj, dict) and obj.get("create_time") else None,
        updated_at=str(obj.get("update_time")) if isinstance(obj, dict) and obj.get("update_time") else None,
        messages=messages,
        attachments=[],
    )


def _parse_json_payload(provider: str, payload: Any, fallback_id: str) -> List[ParsedConversation]:
    conversations: List[ParsedConversation] = []
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and "mapping" in payload[0]:
            for idx, obj in enumerate(payload, start=1):
                if isinstance(obj, dict):
                    conversations.append(_parse_conversation_obj(provider, obj, f"{fallback_id}-{idx}"))
        elif payload and isinstance(payload[0], dict) and "messages" in payload[0]:
            for idx, obj in enumerate(payload, start=1):
                if isinstance(obj, dict):
                    conversations.append(_parse_conversation_obj(provider, obj, f"{fallback_id}-{idx}"))
        else:
            messages = _extract_messages_from_list(payload)
            conversations.append(
                ParsedConversation(
                    provider_name=provider,
                    provider_conversation_id=fallback_id,
                    title=fallback_id,
                    created_at=None,
                    updated_at=None,
                    messages=messages,
                )
            )
    elif isinstance(payload, dict):
        if "conversations" in payload and isinstance(payload["conversations"], list):
            for idx, obj in enumerate(payload["conversations"], start=1):
                if isinstance(obj, dict):
                    conversations.append(_parse_conversation_obj(provider, obj, f"{fallback_id}-{idx}"))
        else:
            conversations.append(_parse_conversation_obj(provider, payload, fallback_id))
    return conversations


def parse_drive_payload(provider: str, payload: Any, fallback_id: str) -> List[ParsedConversation]:
    conversations: List[ParsedConversation] = []
    if isinstance(payload, list):
        for idx, obj in enumerate(payload, start=1):
            if isinstance(obj, dict) and ("chunkedPrompt" in obj or "chunks" in obj):
                conversations.append(_parse_chunked_prompt(provider, obj, f"{fallback_id}-{idx}"))
            elif isinstance(obj, dict):
                conversations.append(_parse_conversation_obj(provider, obj, f"{fallback_id}-{idx}"))
    elif isinstance(payload, dict):
        if "chunkedPrompt" in payload or "chunks" in payload:
            conversations.append(_parse_chunked_prompt(provider, payload, fallback_id))
        else:
            conversations.append(_parse_conversation_obj(provider, payload, fallback_id))
    return conversations


def iter_source_conversations(source: Source) -> Iterable[ParsedConversation]:
    paths: List[Path] = []
    if source.type == "drive":
        return []
    if not source.path:
        return []
    base = source.path.expanduser()
    if base.is_dir():
        paths.extend(sorted(base.rglob("*.json")))
        paths.extend(sorted(base.rglob("*.jsonl")))
        paths.extend(sorted(base.rglob("*.zip")))
    elif base.is_file():
        paths.append(base)

    conversations: List[ParsedConversation] = []
    for path in paths:
        try:
            if path.suffix.lower() == ".zip":
                payload = _load_json_from_zip(path)
            else:
                payload = _load_json_from_path(path)
        except Exception:
            continue
        conversations.extend(_parse_json_payload(source.name, payload, path.stem))
    return conversations

__all__ = [
    "ParsedConversation",
    "ParsedMessage",
    "ParsedAttachment",
    "iter_source_conversations",
    "parse_drive_payload",
]

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
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
class ParsedConversation:
    provider_name: str
    provider_conversation_id: str
    title: str
    created_at: Optional[str]
    updated_at: Optional[str]
    messages: List[ParsedMessage]


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


def iter_source_conversations(source: Source) -> Iterable[ParsedConversation]:
    paths: List[Path] = []
    if source.type == "drive":
        if not source.folder:
            return []
        base = Path(source.folder).expanduser()
    else:
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
        conversations.extend(_parse_json_payload(source.type, payload, path.stem))
    return conversations


__all__ = ["ParsedConversation", "ParsedMessage", "iter_source_conversations"]

from __future__ import annotations

import ijson
import json
import logging
import zipfile
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, BinaryIO

from .config import Source

LOGGER = logging.getLogger(__name__)


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


def _iter_json_stream(handle: BinaryIO, path_name: str) -> Iterable[dict]:
    """Stream items from a JSON file (handling both lists and single objects)."""
    if path_name.lower().endswith(".jsonl") or path_name.lower().endswith(".jsonl.txt"):
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
        return

    # For standard JSON, we try to detect if it's a list or an object with a "conversations" list
    # We use ijson.items to stream if possible
    try:
        # Try to stream items from a root list
        for item in ijson.items(handle, "item"):
            yield item
        return
    except ijson.common.JSONError:
        handle.seek(0)
        try:
            # Try to stream items from "conversations" key in a root object
            for item in ijson.items(handle, "conversations.item"):
                yield item
            return
        except ijson.common.JSONError:
            handle.seek(0)
            # Fallback: load whole object if it's not a list we recognize for streaming
            try:
                data = json.load(handle)
                if isinstance(data, dict):
                    yield data
                elif isinstance(data, list):
                    yield from data
            except json.JSONDecodeError:
                LOGGER.warning("Failed to parse JSON from %s", path_name)


def iter_source_conversations(source: Source, *, cursor_state: Optional[dict] = None) -> Iterable[ParsedConversation]:
    paths: List[Path] = []
    if not source.path:
        return
    base = source.path.expanduser()
    if base.is_dir():
        paths.extend(sorted(base.rglob("*.json")))
        paths.extend(sorted(base.rglob("*.jsonl")))
        paths.extend(sorted(base.rglob("*.jsonl.txt")))
        paths.extend(sorted(base.rglob("*.zip")))
    elif base.is_file():
        paths.append(base)

    if cursor_state is not None:
        cursor_state["file_count"] = len(paths)
        if paths:
            try:
                latest = max(paths, key=lambda p: p.stat().st_mtime)
                cursor_state["latest_mtime"] = latest.stat().st_mtime
                cursor_state["latest_path"] = str(latest)
            except OSError:
                pass

    for path in paths:
        try:
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as zf:
                    target_names = [
                        name for name in zf.namelist() 
                        if name.endswith("conversations.json") or name.endswith(".jsonl")
                    ]
                    for name in target_names:
                        with zf.open(name) as handle:
                            for payload in _iter_json_stream(handle, name):
                                provider = detect_provider(payload, path) or source.name
                                for convo in _parse_json_payload(provider, payload, path.stem):
                                    yield convo
            else:
                with path.open("rb") as handle:
                    for payload in _iter_json_stream(handle, path.name):
                        provider = detect_provider(payload, path) or source.name
                        for convo in _parse_json_payload(provider, payload, path.stem):
                            yield convo
        except Exception as exc:
            LOGGER.warning("Failed to parse %s: %s", path, exc)
            continue

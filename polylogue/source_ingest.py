from __future__ import annotations

import ijson
import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Iterable, List, Optional, BinaryIO

from .config import Source
from .importers.base import ParsedConversation, ParsedMessage, ParsedAttachment
from .importers import chatgpt, claude, codex, drive

LOGGER = logging.getLogger(__name__)

def detect_provider(payload: Any, path: Path) -> Optional[str]:
    if isinstance(payload, dict):
        if chatgpt.looks_like(payload):
            return "chatgpt"
        if claude.looks_like_ai(payload):
            return "claude"
    if isinstance(payload, list):
        if claude.looks_like_code(payload):
            return "claude-code"
        if codex.looks_like(payload):
            return "codex"
    
    name = path.name.lower()
    if "chatgpt" in name: return "chatgpt"
    if "claude-code" in name or "claude_code" in name: return "claude-code"
    if "claude" in name: return "claude"
    if "codex" in name: return "codex"
    return None

def _parse_json_payload(provider: str, payload: Any, fallback_id: str) -> List[ParsedConversation]:
    if provider == "chatgpt":
        return [chatgpt.parse(payload, fallback_id)]
    if provider == "claude":
        return [claude.parse_ai(payload, fallback_id)]
    if provider == "codex":
        return [codex.parse(payload, fallback_id)]
    
    # Fallback / Generic
    if isinstance(payload, dict):
        if "conversations" in payload and isinstance(payload["conversations"], list):
            return [_parse_json_payload(provider, item, f"{fallback_id}-{i}")[0] 
                    for i, item in enumerate(payload["conversations"])]
        return [chatgpt.parse(payload, fallback_id) if chatgpt.looks_like(payload) else drive.parse_chunked_prompt(provider, payload, fallback_id)]
    
    return []

def parse_drive_payload(provider: str, payload: Any, fallback_id: str) -> List[ParsedConversation]:
    if isinstance(payload, list):
        results = []
        for i, item in enumerate(payload):
            results.extend(parse_drive_payload(provider, item, f"{fallback_id}-{i}"))
        return results
    if isinstance(payload, dict):
        if "chunkedPrompt" in payload or "chunks" in payload:
            return [drive.parse_chunked_prompt(provider, payload, fallback_id)]
        return [chatgpt.parse(payload, fallback_id) if chatgpt.looks_like(payload) else chatgpt.parse(payload, fallback_id)] # placeholder
    return []

def _iter_json_stream(handle: BinaryIO, path_name: str) -> Iterable[dict]:
    if path_name.lower().endswith((".jsonl", ".jsonl.txt")):
        for line in handle:
            line = line.strip()
            if not line: continue
            try: yield json.loads(line)
            except json.JSONDecodeError: continue
        return

    try:
        for item in ijson.items(handle, "item"): yield item
        return
    except ijson.common.JSONError:
        handle.seek(0)
        try:
            for item in ijson.items(handle, "conversations.item"): yield item
            return
        except ijson.common.JSONError:
            handle.seek(0)
            try:
                data = json.load(handle)
                if isinstance(data, dict): yield data
                elif isinstance(data, list): yield from data
            except json.JSONDecodeError:
                LOGGER.warning("Failed to parse JSON from %s", path_name)

def iter_source_conversations(source: Source, *, cursor_state: Optional[dict] = None) -> Iterable[ParsedConversation]:
    if not source.path: return
    base = source.path.expanduser()
    paths = []
    if base.is_dir():
        for ext in ("*.json", "*.jsonl", "*.jsonl.txt", "*.zip"):
            paths.extend(sorted(base.rglob(ext)))
    elif base.is_file():
        paths.append(base)

    if cursor_state is not None:
        cursor_state["file_count"] = len(paths)
        if paths:
            try:
                latest = max(paths, key=lambda p: p.stat().st_mtime)
                cursor_state["latest_mtime"] = latest.stat().st_mtime
                cursor_state["latest_path"] = str(latest)
            except OSError: pass

    for path in paths:
        try:
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as zf:
                    for name in zf.namelist():
                        if name.endswith(("conversations.json", ".jsonl")):
                            with zf.open(name) as handle:
                                for payload in _iter_json_stream(handle, name):
                                    provider = detect_provider(payload, path) or source.name
                                    yield from _parse_json_payload(provider, payload, path.stem)
            else:
                with path.open("rb") as handle:
                    for payload in _iter_json_stream(handle, path.name):
                        provider = detect_provider(payload, path) or source.name
                        yield from _parse_json_payload(provider, payload, path.stem)
        except Exception as exc:
            LOGGER.warning("Failed to parse %s: %s", path, exc)

__all__ = ["ParsedConversation", "ParsedMessage", "ParsedAttachment", "iter_source_conversations", "parse_drive_payload"]
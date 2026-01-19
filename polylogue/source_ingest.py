from __future__ import annotations

import json
import logging
import zipfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, BinaryIO

import ijson

from .config import Source
from .importers import chatgpt, claude, codex, drive
from .importers.base import ParsedAttachment, ParsedConversation, ParsedMessage, extract_messages_from_list

LOGGER = logging.getLogger(__name__)
_ENCODING_GUESSES: tuple[str, ...] = (
    "utf-8",
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "utf-32",
    "utf-32-le",
    "utf-32-be",
)


def _decode_json_bytes(blob: bytes) -> str | None:
    """Decode a JSON payload from bytes, trying multiple encodings."""

    for encoding in _ENCODING_GUESSES:
        try:
            decoded = blob.decode(encoding)
        except UnicodeError:
            continue
        cleaned = decoded.replace("\x00", "")
        if cleaned:
            return cleaned
    try:
        decoded = blob.decode("utf-8", errors="ignore").replace("\x00", "")
        return decoded if decoded else None
    except Exception:
        LOGGER.debug("Failed to coerce JSON bytes after fallbacks.")
        return None


def detect_provider(payload: Any, path: Path) -> str | None:
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
    if "chatgpt" in name:
        return "chatgpt"
    if "claude-code" in name or "claude_code" in name:
        return "claude-code"
    if "claude" in name:
        return "claude"
    if "codex" in name:
        return "codex"
    return None


def _parse_json_payload(provider: str, payload: Any, fallback_id: str) -> list[ParsedConversation]:
    if provider == "chatgpt":
        return [chatgpt.parse(payload, fallback_id)]
    if provider == "claude":
        return [claude.parse_ai(payload, fallback_id)]
    if provider == "claude-code":
        if isinstance(payload, list):
            return [claude.parse_code(payload, fallback_id)]
        # If payload is a dict with messages, extract them
        if isinstance(payload, dict) and isinstance(payload.get("messages"), list):
            return [claude.parse_code(payload["messages"], fallback_id)]
    if provider == "codex":
        return [codex.parse(payload, fallback_id)]

    # Fallback / Generic
    if isinstance(payload, dict):
        if "conversations" in payload and isinstance(payload["conversations"], list):
            return [
                _parse_json_payload(provider, item, f"{fallback_id}-{i}")[0]
                for i, item in enumerate(payload["conversations"])
            ]

        # Generic "messages" support
        if "messages" in payload and isinstance(payload["messages"], list):
            messages = extract_messages_from_list(payload["messages"])
            title = payload.get("title") or payload.get("name") or fallback_id
            return [
                ParsedConversation(
                    provider_name=provider,
                    provider_conversation_id=str(payload.get("id") or fallback_id),
                    title=str(title),
                    created_at=None,
                    updated_at=None,
                    messages=messages,
                )
            ]

        return [
            chatgpt.parse(payload, fallback_id)
            if chatgpt.looks_like(payload)
            else drive.parse_chunked_prompt(provider, payload, fallback_id)
        ]

    return []


def parse_drive_payload(provider: str, payload: Any, fallback_id: str) -> list[ParsedConversation]:
    if isinstance(payload, list):
        results = []
        for i, item in enumerate(payload):
            results.extend(parse_drive_payload(provider, item, f"{fallback_id}-{i}"))
        return results
    if isinstance(payload, dict):
        if "chunkedPrompt" in payload or "chunks" in payload:
            return [drive.parse_chunked_prompt(provider, payload, fallback_id)]
        return [
            chatgpt.parse(payload, fallback_id) if chatgpt.looks_like(payload) else chatgpt.parse(payload, fallback_id)
        ]  # placeholder
    return []


def _iter_json_stream(handle: BinaryIO, path_name: str) -> Iterable[dict]:
    if path_name.lower().endswith((".jsonl", ".jsonl.txt")):
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            if isinstance(raw, bytes):
                decoded = _decode_json_bytes(raw)
                if not decoded:
                    LOGGER.warning("Skipping undecodable line from %s", path_name)
                    continue
            else:
                decoded = raw
            try:
                yield json.loads(decoded)
            except json.JSONDecodeError:
                continue
        return

    # Strategy 1: Try streaming root list
    try:
        # LOGGER.info("Strategy 1: ijson items(item) for %s", path_name)
        found_any = False
        for item in ijson.items(handle, "item"):
            found_any = True
            yield item
        if found_any:
            return
    except (ijson.common.JSONError, Exception):
        # LOGGER.info("Strategy 1 failed for %s: %s", path_name, e)
        pass

    handle.seek(0)
    # Strategy 2: Try streaming conversations list
    try:
        # LOGGER.info("Strategy 2: ijson items(conversations.item) for %s", path_name)
        found_any = False
        for item in ijson.items(handle, "conversations.item"):
            found_any = True
            yield item
        if found_any:
            return
    except (ijson.common.JSONError, Exception):
        # LOGGER.info("Strategy 2 failed for %s: %s", path_name, e)
        pass

    handle.seek(0)
    # Strategy 3: Load full object (fallback for single dicts or unknown structures)
    try:
        # LOGGER.info("Strategy 3: json.load for %s", path_name)
        data = json.load(handle)
        if isinstance(data, dict):
            yield data
        elif isinstance(data, list):
            yield from data
    except json.JSONDecodeError:
        LOGGER.warning("Failed to parse JSON from %s", path_name)
    except Exception as e:
        LOGGER.warning("Strategy 3 failed for %s: %s", path_name, e)
        raise


def iter_source_conversations(source: Source, *, cursor_state: dict | None = None) -> Iterable[ParsedConversation]:
    if not source.path:
        return
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
            except OSError:
                pass

    for path in paths:
        try:
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as zf:
                    for name in zf.namelist():
                        if name.endswith(("conversations.json", ".jsonl")):
                            with zf.open(name) as handle:
                                for payload in _iter_json_stream(handle, name):
                                    try:
                                        provider = detect_provider(payload, path) or source.name
                                        yield from _parse_json_payload(provider, payload, path.stem)
                                    except Exception:
                                        LOGGER.exception("Error processing payload from %s", name)
                                        raise
            else:
                with path.open("rb") as handle:
                    for payload in _iter_json_stream(handle, path.name):
                        try:
                            provider = detect_provider(payload, path) or source.name
                            yield from _parse_json_payload(provider, payload, path.stem)
                        except Exception:
                            LOGGER.exception("Error processing payload from %s", path)
                            raise
        except Exception as exc:
            LOGGER.warning("Failed to parse %s: %s", path, exc)
            continue


__all__ = [
    "ParsedConversation",
    "ParsedMessage",
    "ParsedAttachment",
    "iter_source_conversations",
    "parse_drive_payload",
]

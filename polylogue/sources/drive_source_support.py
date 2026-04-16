from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from polylogue.logging import get_logger

from .drive_types import (
    FOLDER_MIME_TYPE,
    GEMINI_PROMPT_MIME_TYPE,
    DriveFile,
)

logger = get_logger(__name__)


def _parse_modified_time(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return None


def _parse_size(raw: str | int | None) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _looks_like_id(value: str) -> bool:
    if not value or " " in value:
        return False
    return all(ch.isalnum() or ch in "-_" for ch in value)


def _build_folder_lookup_query(folder_ref: str) -> str:
    escaped = folder_ref.replace("'", "\\'")
    return f"name = '{escaped}' and mimeType = '{FOLDER_MIME_TYPE}' and trashed = false"


def _is_newline_delimited_json_name(name: str) -> bool:
    name_lower = name.lower()
    return name_lower.endswith(".jsonl") or name_lower.endswith(".jsonl.txt") or name_lower.endswith(".ndjson")


def _is_supported_drive_payload(name: str, mime_type: str) -> bool:
    return name.lower().endswith((".json", ".jsonl", ".jsonl.txt", ".ndjson")) or mime_type == GEMINI_PROMPT_MIME_TYPE


def _parse_downloaded_json_payload(raw: bytes, *, name: str) -> object:
    if _is_newline_delimited_json_name(name):
        items = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON line in Drive file %s: %s", name, exc)
                continue
        return items

    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return json.loads(raw.decode("utf-8", errors="replace"))


def _needs_download(meta: DriveFile, dest: Path) -> bool:
    if not dest.exists():
        return True

    try:
        stat = dest.stat()
    except OSError:
        return True

    if meta.size_bytes is not None and stat.st_size != meta.size_bytes:
        return True

    modified_timestamp = _parse_modified_time(meta.modified_time)
    return modified_timestamp is not None and abs(stat.st_mtime - modified_timestamp) > 1


def _build_drive_file(meta: dict[str, Any], *, file_id_fallback: str = "") -> DriveFile:
    file_id = meta.get("id", file_id_fallback)
    return DriveFile(
        file_id=file_id,
        name=meta.get("name") or file_id or file_id_fallback,
        mime_type=meta.get("mimeType") or "",
        modified_time=meta.get("modifiedTime"),
        size_bytes=_parse_size(meta.get("size")),
    )

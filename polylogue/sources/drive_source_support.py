from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TypeAlias, cast

from polylogue.logging import get_logger

from .drive_gateway import DrivePayloadRecord
from .drive_types import (
    FOLDER_MIME_TYPE,
    GEMINI_PROMPT_MIME_TYPE,
    DriveFile,
)

logger = get_logger(__name__)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonPayload: TypeAlias = dict[str, "JsonPayload"] | list["JsonPayload"] | JsonScalar
JsonObject: TypeAlias = DrivePayloadRecord


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


def _parse_downloaded_json_payload(raw: bytes, *, name: str) -> JsonPayload:
    if _is_newline_delimited_json_name(name):
        items: list[JsonPayload] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(cast(JsonPayload, json.loads(line)))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping invalid JSON line in Drive file %s: %s", name, exc)
                continue
        return items

    try:
        return cast(JsonPayload, json.loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return cast(JsonPayload, json.loads(raw.decode("utf-8", errors="replace")))


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


def _extract_meta_string(meta: JsonObject, key: str, *, file_id_fallback: str = "") -> str:
    value = meta.get(key)
    if isinstance(value, str):
        return value
    return file_id_fallback


def _build_drive_file(meta: JsonObject, *, file_id_fallback: str = "") -> DriveFile:
    file_id = _extract_meta_string(meta, "id", file_id_fallback=file_id_fallback)
    name = _extract_meta_string(meta, "name", file_id_fallback=file_id_fallback)
    modified_time_raw = meta.get("modifiedTime")
    modified_time = modified_time_raw if isinstance(modified_time_raw, str) else None
    size_raw = meta.get("size")
    size_value = size_raw if isinstance(size_raw, (str, int)) else None
    return DriveFile(
        file_id=file_id,
        name=name or file_id or file_id_fallback,
        mime_type=_extract_meta_string(meta, "mimeType"),
        modified_time=modified_time,
        size_bytes=_parse_size(size_value),
    )

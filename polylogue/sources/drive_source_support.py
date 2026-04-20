from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TypeAlias

from polylogue.lib.json import JSONValue, is_json_value, loads
from polylogue.logging import get_logger

from .drive_gateway import DriveListFilesResponse, DrivePayloadRecord
from .drive_types import (
    FOLDER_MIME_TYPE,
    GEMINI_PROMPT_MIME_TYPE,
    DriveFile,
)

logger = get_logger(__name__)

DriveJSONPayload: TypeAlias = JSONValue
DriveJSONRecord: TypeAlias = DrivePayloadRecord
DriveJSONSequence: TypeAlias = list[DriveJSONPayload]


def _json_sequence(value: object) -> DriveJSONSequence:
    if not isinstance(value, list):
        return []
    items: DriveJSONSequence = []
    for item in value:
        if is_json_value(item):
            items.append(item)
    return items


def _response_files(response: DriveListFilesResponse) -> list[JSONValue]:
    return _json_sequence(response.get("files"))


def _response_page_token(response: DriveListFilesResponse) -> str | None:
    token = response.get("nextPageToken")
    return token if isinstance(token, str) else None


def _record_string(record: DrivePayloadRecord, key: str, *, default: str = "") -> str:
    value = record.get(key)
    return value if isinstance(value, str) else default


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


def _parse_downloaded_json_payload(raw: bytes, *, name: str) -> DriveJSONPayload:
    if _is_newline_delimited_json_name(name):
        items: DriveJSONSequence = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(loads(line))
            except Exception as exc:
                logger.warning("Skipping invalid JSON line in Drive file %s: %s", name, exc)
                continue
        return items

    try:
        return loads(raw)
    except Exception:
        return loads(raw.decode("utf-8", errors="replace"))


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


def _extract_meta_string(meta: DriveJSONRecord, key: str, *, file_id_fallback: str = "") -> str:
    return _record_string(meta, key, default=file_id_fallback)


def _build_drive_file(meta: DriveJSONRecord, *, file_id_fallback: str = "") -> DriveFile:
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


__all__ = [
    "DriveJSONPayload",
    "DriveJSONRecord",
    "DriveJSONSequence",
    "_record_string",
    "_build_drive_file",
    "_build_folder_lookup_query",
    "_response_files",
    "_response_page_token",
    "_is_supported_drive_payload",
    "_json_sequence",
    "_looks_like_id",
    "_needs_download",
    "_parse_downloaded_json_payload",
    "_parse_modified_time",
    "_parse_size",
]

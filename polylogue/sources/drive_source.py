from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ..lib.log import get_logger
from .drive_gateway import DriveServiceGateway
from .drive_types import (
    FOLDER_MIME_TYPE,
    GEMINI_PROMPT_MIME_TYPE,
    DriveFile,
    DriveNotFoundError,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


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
    return (
        name_lower.endswith(".jsonl")
        or name_lower.endswith(".jsonl.txt")
        or name_lower.endswith(".ndjson")
    )


def _is_supported_drive_payload(name: str, mime_type: str) -> bool:
    """Return whether a Drive file should be treated as a JSON payload export."""
    return (
        name.lower().endswith((".json", ".jsonl", ".jsonl.txt", ".ndjson"))
        or mime_type == GEMINI_PROMPT_MIME_TYPE
    )


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
    """Return whether a Drive file should be downloaded to the destination path."""
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


# ---------------------------------------------------------------------------
# DriveSourceAPI protocol
# ---------------------------------------------------------------------------


class DriveSourceAPI(Protocol):
    """Minimal interface that Drive ingestion code depends on."""

    def resolve_folder_id(self, folder_ref: str) -> str: ...
    def iter_json_files(self, folder_id: str) -> Iterable[DriveFile]: ...
    def download_json_payload(self, file_id: str, *, name: str) -> object: ...
    def download_to_path(self, file_id: str, dest: Path) -> DriveFile: ...
    def download_bytes(self, file_id: str) -> bytes: ...


# ---------------------------------------------------------------------------
# DriveSourceClient
# ---------------------------------------------------------------------------


class DriveSourceClient:
    """Owns Polylogue-specific Drive source semantics."""

    def __init__(self, *, gateway: DriveServiceGateway) -> None:
        self._gateway = gateway
        self._meta_cache: dict[str, DriveFile] = {}

    def _resolve_folder_by_id(self, folder_ref: str) -> str | None:
        meta = self._gateway.get_file(folder_ref, "id,name,mimeType")
        if meta and meta.get("mimeType") == FOLDER_MIME_TYPE:
            return str(meta["id"])
        return None

    def _resolve_folder_by_name(self, folder_ref: str) -> str:
        response = self._gateway.list_files(
            q=_build_folder_lookup_query(folder_ref),
            fields="files(id,name)",
            page_token=None,
            page_size=1000,
        )
        matches = response.get("files", [])
        if not matches:
            raise DriveNotFoundError(f"Folder not found: {folder_ref}")
        return str(matches[0]["id"])

    def resolve_folder_id(self, folder_ref: str) -> str:
        from .drive_gateway import _import_module as _gw_import
        http_error_cls = _gw_import("googleapiclient.errors").HttpError

        if _looks_like_id(folder_ref):
            try:
                resolved = self._resolve_folder_by_id(folder_ref)
                if resolved is not None:
                    return resolved
            except Exception as exc:
                if isinstance(exc, http_error_cls):
                    if exc.resp.status not in (404, 403):
                        logger.warning("Unexpected Drive API error resolving %s: %s", folder_ref, exc)
                else:
                    logger.warning("Error resolving folder ID %s: %s", folder_ref, exc)
        return self._resolve_folder_by_name(folder_ref)

    def iter_json_files(self, folder_id: str) -> Iterable[DriveFile]:
        page_token: str | None = None
        query = f"'{folder_id}' in parents and trashed = false"
        fields = "nextPageToken, files(id,name,mimeType,modifiedTime,size)"
        while True:
            response = self._gateway.list_files(
                q=query,
                fields=fields,
                page_token=page_token,
                page_size=1000,
            )
            for item in response.get("files", []):
                name = item.get("name") or ""
                mime_type = item.get("mimeType") or ""
                if not _is_supported_drive_payload(name, mime_type):
                    continue
                file_obj = _build_drive_file(item)
                if file_obj.file_id:
                    self._meta_cache[file_obj.file_id] = file_obj
                    yield file_obj
            page_token = response.get("nextPageToken")
            if not page_token:
                break

    def get_metadata(self, file_id: str) -> DriveFile:
        if file_id in self._meta_cache:
            return self._meta_cache[file_id]
        meta = self._gateway.get_file(file_id, "id,name,mimeType,modifiedTime,size")
        file_obj = _build_drive_file(meta, file_id_fallback=file_id)
        self._meta_cache[file_id] = file_obj
        return file_obj

    def download_bytes(self, file_id: str) -> bytes:
        def _download() -> bytes:
            buffer = io.BytesIO()
            self._gateway.download_file(file_id, buffer)
            return buffer.getvalue()

        return self._gateway.call_with_retry(_download)

    def download_to_path(self, file_id: str, dest: Path) -> DriveFile:
        meta = self.get_metadata(file_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if _needs_download(meta, dest):

            def _download_once() -> None:
                tmp_path: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as handle:
                        tmp_path = Path(handle.name)
                        self._gateway.download_file(file_id, handle)
                    tmp_path.replace(dest)
                except Exception:
                    if tmp_path is not None:
                        with contextlib.suppress(OSError):
                            tmp_path.unlink()
                    raise

            self._gateway.call_with_retry(_download_once)

        modified_timestamp = _parse_modified_time(meta.modified_time)
        if modified_timestamp is not None:
            os.utime(dest, (modified_timestamp, modified_timestamp))
        return meta

    def download_json_payload(self, file_id: str, *, name: str) -> object:
        raw = self.download_bytes(file_id)
        return _parse_downloaded_json_payload(raw, name=name)


__all__ = [
    "DriveSourceAPI",
    "DriveSourceClient",
    "_build_drive_file",
    "_build_folder_lookup_query",
    "_is_supported_drive_payload",
    "_looks_like_id",
    "_needs_download",
    "_parse_downloaded_json_payload",
    "_parse_modified_time",
    "_parse_size",
]

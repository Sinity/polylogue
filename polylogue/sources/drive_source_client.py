from __future__ import annotations

import contextlib
import io
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, BinaryIO, cast

from polylogue.logging import get_logger

from .drive_gateway import DriveServiceGateway
from .drive_source_support import (
    JsonObject,
    _build_drive_file,
    _build_folder_lookup_query,
    _is_supported_drive_payload,
    _looks_like_id,
    _needs_download,
    _parse_downloaded_json_payload,
    _parse_modified_time,
)
from .drive_types import (
    FOLDER_MIME_TYPE,
    DriveFile,
    DriveNotFoundError,
)

logger = get_logger(__name__)


class DriveSourceClient:
    """Owns Polylogue-specific Drive source semantics."""

    @staticmethod
    def _as_payload_dict(payload: Any) -> JsonObject:
        if isinstance(payload, dict):
            return payload
        return {}

    @staticmethod
    def _as_payload_list(payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return payload
        return []

    @staticmethod
    def _as_str(payload: Any, default: str = "") -> str:
        return payload if isinstance(payload, str) else default

    @classmethod
    def _as_mime_type(cls, payload: Any, default: str = "") -> str:
        return cls._as_str(payload, default=default)

    @classmethod
    def _as_file_id(cls, payload: Any, default: str = "") -> str:
        return cls._as_str(payload, default=default)

    @classmethod
    def _next_page_token(cls, payload: Any) -> str | None:
        if isinstance(payload, str):
            return payload
        return None

    @classmethod
    def _resolve_next_page_token(cls, payload: Any) -> str | None:
        return cls._next_page_token(payload)

    def __init__(self, *, gateway: DriveServiceGateway) -> None:
        self._gateway = gateway
        self._meta_cache: dict[str, DriveFile] = {}

    def _resolve_folder_by_id(self, folder_ref: str) -> str | None:
        meta = self._gateway.get_file(folder_ref, "id,name,mimeType")
        payload = self._as_payload_dict(meta)
        if self._as_mime_type(payload.get("mimeType")) == FOLDER_MIME_TYPE:
            return self._as_file_id(payload.get("id"))
        return None

    def _resolve_folder_by_name(self, folder_ref: str) -> str:
        response = self._gateway.list_files(
            q=_build_folder_lookup_query(folder_ref),
            fields="files(id,name)",
            page_token=None,
            page_size=1000,
        )
        payload = self._as_payload_dict(response)
        matches = self._as_payload_list(payload.get("files"))
        if not matches:
            raise DriveNotFoundError(f"Folder not found: {folder_ref}")
        first = self._as_payload_dict(matches[0])
        return self._as_file_id(first.get("id"), default=folder_ref)

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
            payload = self._as_payload_dict(response)
            for item in self._as_payload_list(payload.get("files")):
                item_payload = self._as_payload_dict(item)
                name = self._as_str(item_payload.get("name"))
                mime_type = self._as_str(item_payload.get("mimeType"))
                if not _is_supported_drive_payload(name, mime_type):
                    continue
                file_obj = _build_drive_file(item_payload)
                if file_obj.file_id:
                    self._meta_cache[file_obj.file_id] = file_obj
                    yield file_obj
            page_token = self._resolve_next_page_token(payload.get("nextPageToken"))
            if not page_token:
                break

    def get_metadata(self, file_id: str) -> DriveFile:
        if file_id in self._meta_cache:
            return self._meta_cache[file_id]
        meta = self._gateway.get_file(file_id, "id,name,mimeType,modifiedTime,size")
        file_obj = _build_drive_file(self._as_payload_dict(meta), file_id_fallback=file_id)
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
                        self._gateway.download_file(file_id, cast(BinaryIO, handle))
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

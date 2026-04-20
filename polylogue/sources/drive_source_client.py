from __future__ import annotations

import contextlib
import io
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO, cast

from polylogue.lib.json import JSONDocument, JSONValue
from polylogue.logging import get_logger

from .drive_gateway import DriveServiceGateway
from .drive_source_support import (
    _build_drive_file,
    _build_folder_lookup_query,
    _is_supported_drive_payload,
    _looks_like_id,
    _needs_download,
    _parse_downloaded_json_payload,
    _parse_modified_time,
    _record_string,
    _response_files,
    _response_page_token,
)
from .drive_types import (
    FOLDER_MIME_TYPE,
    DriveFile,
    DriveNotFoundError,
)

logger = get_logger(__name__)


class DriveSourceClient:
    """Owns Polylogue-specific Drive source semantics."""

    def __init__(self, *, gateway: DriveServiceGateway) -> None:
        self._gateway = gateway
        self._meta_cache: dict[str, DriveFile] = {}

    def _resolve_folder_by_id(self, folder_ref: str) -> str | None:
        meta = self._gateway.get_file(folder_ref, "id,name,mimeType")
        if _record_string(meta, "mimeType") == FOLDER_MIME_TYPE:
            file_id = _record_string(meta, "id")
            return file_id or None
        return None

    def _resolve_folder_by_name(self, folder_ref: str) -> str:
        response = self._gateway.list_files(
            q=_build_folder_lookup_query(folder_ref),
            fields="files(id,name)",
            page_token=None,
            page_size=1000,
        )
        matches = _response_files(response)
        if not matches:
            raise DriveNotFoundError(f"Folder not found: {folder_ref}")
        first = matches[0]
        if not isinstance(first, dict):
            return folder_ref
        file_id = _record_string(first, "id", default=folder_ref)
        return file_id or folder_ref

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
            for item in _response_files(response):
                if not isinstance(item, dict):
                    continue
                item_payload: JSONDocument = item
                name = _record_string(item_payload, "name")
                mime_type = _record_string(item_payload, "mimeType")
                if not _is_supported_drive_payload(name, mime_type):
                    continue
                file_obj = _build_drive_file(item_payload)
                if file_obj.file_id:
                    self._meta_cache[file_obj.file_id] = file_obj
                    yield file_obj
            page_token = _response_page_token(response)
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

    def download_json_payload(self, file_id: str, *, name: str) -> JSONValue:
        raw = self.download_bytes(file_id)
        return _parse_downloaded_json_payload(raw, name=name)

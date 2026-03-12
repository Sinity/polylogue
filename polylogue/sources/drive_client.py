"""Drive client — thin composition root over auth, gateway, and source layers.

For new code, prefer depending on DriveSourceAPI (protocol) from drive_source.py
rather than the concrete DriveClient. The concrete class exists for backwards
compatibility of instantiation sites and as the default construction path.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .drive_auth import (
    DriveAuthManager,
    _resolve_credentials_path,
    _resolve_token_path,
    default_credentials_path,
    default_token_path,
)
from .drive_gateway import (
    DEFAULT_DRIVE_RETRIES,
    DEFAULT_DRIVE_RETRY_BASE,
    DriveServiceGateway,
    _import_module,
    _resolve_retries,
    _resolve_retry_base,
)
from .drive_source import (
    DriveSourceClient,
    _build_folder_lookup_query,
    _is_supported_drive_payload,
    _looks_like_id,
    _needs_download,
    _parse_downloaded_json_payload,
    _parse_modified_time,
    _parse_size,
)
from .drive_types import (
    GEMINI_PROMPT_MIME_TYPE,
    SCOPES,
    DriveAuthError,
    DriveError,
    DriveFile,
    DriveNotFoundError,
)


class DriveClient:
    """Composition root: auth + gateway + source client in one object.

    Callers that need only the source operations should accept DriveSourceAPI
    (a protocol) instead of this concrete class.
    """

    def __init__(
        self,
        *,
        ui: object | None = None,
        credentials_path: Path | None = None,
        token_path: Path | None = None,
        retries: int | None = None,
        retry_base: float | None = None,
        config: object | None = None,
    ) -> None:
        self._auth_manager = DriveAuthManager(
            ui=ui,
            credentials_path=credentials_path,
            token_path=token_path,
            config=config,
        )
        resolved_retries = _resolve_retries(retries, config)
        resolved_retry_base = _resolve_retry_base(retry_base)
        self._gateway = DriveServiceGateway(
            auth_manager=self._auth_manager,
            retries=resolved_retries,
            retry_base=resolved_retry_base,
        )
        self._source = DriveSourceClient(gateway=self._gateway)

    # ------------------------------------------------------------------
    # Expose internals that tests still reach into (kept for compatibility)
    # ------------------------------------------------------------------

    @property
    def _service(self):
        return self._gateway._service

    @_service.setter
    def _service(self, value):
        self._gateway._service = value

    @property
    def _meta_cache(self):
        return self._source._meta_cache

    @property
    def _token_store(self):
        return self._auth_manager._token_store

    @_token_store.setter
    def _token_store(self, value):
        self._auth_manager._token_store = value

    # ------------------------------------------------------------------
    # Methods delegated to inner layers (kept for call-site compatibility)
    # ------------------------------------------------------------------

    def _call_with_retry(self, func, *args, **kwargs):
        return self._gateway.call_with_retry(func, *args, **kwargs)

    def _load_credentials(self):
        return self._auth_manager.load_credentials()

    def _load_cached_credentials(self, credentials_cls, token_path):
        return self._auth_manager._load_cached_credentials(credentials_cls, token_path)

    def _refresh_credentials_if_needed(self, creds, token_path):
        return self._auth_manager._refresh_credentials_if_needed(creds, token_path)

    def _run_manual_auth_flow(self, flow):
        return self._auth_manager._run_manual_auth_flow(flow)

    def _persist_token(self, creds, token_path):
        return self._auth_manager._persist_token(creds, token_path)

    def _service_handle(self):
        return self._gateway._service_handle()

    @staticmethod
    def _build_drive_file(meta, *, file_id_fallback=""):
        from .drive_source import _build_drive_file
        return _build_drive_file(meta, file_id_fallback=file_id_fallback)

    def _resolve_folder_by_id(self, service, folder_ref):
        # service arg ignored — gateway manages the service handle internally
        return self._source._resolve_folder_by_id(folder_ref)

    def _resolve_folder_by_name(self, service, folder_ref):
        # service arg ignored — gateway manages the service handle internally
        return self._source._resolve_folder_by_name(folder_ref)

    def _download_request(self, request, handle, downloader_cls, *, file_id):
        return self._gateway._download_request(request, handle, downloader_cls, file_id=file_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_folder_id(self, folder_ref: str) -> str:
        return self._source.resolve_folder_id(folder_ref)

    def iter_json_files(self, folder_id: str) -> Iterable[DriveFile]:
        return self._source.iter_json_files(folder_id)

    def get_metadata(self, file_id: str) -> DriveFile:
        return self._source.get_metadata(file_id)

    def download_bytes(self, file_id: str) -> bytes:
        return self._source.download_bytes(file_id)

    def download_to_path(self, file_id: str, dest: Path) -> DriveFile:
        return self._source.download_to_path(file_id, dest)

    def download_json_payload(self, file_id: str, *, name: str) -> object:
        return self._source.download_json_payload(file_id, name=name)


__all__ = [
    "DEFAULT_DRIVE_RETRIES",
    "DEFAULT_DRIVE_RETRY_BASE",
    "DriveAuthError",
    "DriveClient",
    "DriveError",
    "DriveFile",
    "DriveNotFoundError",
    "GEMINI_PROMPT_MIME_TYPE",
    "SCOPES",
    "_build_folder_lookup_query",
    "_import_module",
    "_is_supported_drive_payload",
    "_looks_like_id",
    "_needs_download",
    "_parse_downloaded_json_payload",
    "_parse_modified_time",
    "_parse_size",
    "_resolve_credentials_path",
    "_resolve_retries",
    "_resolve_retry_base",
    "_resolve_token_path",
    "default_credentials_path",
    "default_token_path",
]

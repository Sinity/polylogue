from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any, BinaryIO, TypeVar

from tenacity import (
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from polylogue.logging import get_logger

from .drive_types import DriveAuthError, DriveError, DriveNotFoundError

logger = get_logger(__name__)

T = TypeVar("T")

DEFAULT_DRIVE_RETRIES = 3
DEFAULT_DRIVE_RETRY_BASE = 0.5
ENV_DRIVE_RETRIES = "POLYLOGUE_DRIVE_RETRIES"
ENV_DRIVE_RETRY_BASE = "POLYLOGUE_DRIVE_RETRY_BASE"


def _import_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise DriveAuthError(
            "Drive dependencies are not available. "
            "Install google-api-python-client + google-auth-oauthlib "
            "or run Polylogue from a Nix build/dev shell."
        ) from exc


def _resolve_retries(value: int | None, config: object | None = None) -> int:
    """Resolve retry count from explicit value, config, environment, or default."""
    if value is not None:
        return max(0, int(value))

    if config is not None and hasattr(config, "retry_count"):
        return max(0, int(config.retry_count))

    env_value = os.environ.get(ENV_DRIVE_RETRIES)
    if env_value:
        try:
            return max(0, int(env_value))
        except ValueError:
            pass

    return DEFAULT_DRIVE_RETRIES


def _resolve_retry_base(value: float | None) -> float:
    if value is not None:
        return max(0.0, float(value))
    env_value = os.environ.get(ENV_DRIVE_RETRY_BASE)
    if env_value:
        try:
            return max(0.0, float(env_value))
        except ValueError:
            pass
    return DEFAULT_DRIVE_RETRY_BASE


class DriveServiceGateway:
    """Owns Google API imports, service construction, raw Drive calls, and retry policy."""

    def __init__(
        self,
        *,
        auth_manager: Any,
        retries: int,
        retry_base: float,
    ) -> None:
        self._auth_manager = auth_manager
        self._retries = retries
        self._retry_base = retry_base
        self._service = None

    def call_with_retry(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        from tenacity import Retrying

        retryer = Retrying(
            stop=stop_after_attempt(max(self._retries, 0) + 1),
            wait=wait_exponential(multiplier=self._retry_base, min=self._retry_base, max=10),
            retry=retry_if_exception_type(Exception)
            & retry_if_not_exception_type((DriveAuthError, DriveNotFoundError)),
            reraise=True,
        )
        return retryer(func, *args, **kwargs)

    def _service_handle(self) -> Any:
        if self._service is not None:
            creds = getattr(self._service, "_http", None)
            if creds is not None:
                http_creds = getattr(creds, "credentials", None)
                if http_creds is not None and getattr(http_creds, "expired", False):
                    logger.info("Cached service credentials expired, re-authenticating")
                    self._service = None
                    return self._service_handle()
            return self._service
        build = _import_module("googleapiclient.discovery").build
        creds = self._auth_manager.load_credentials()
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._service

    def get_file(self, file_id: str, fields: str) -> dict[str, Any]:
        service = self._service_handle()
        return self.call_with_retry(lambda: service.files().get(fileId=file_id, fields=fields).execute())

    def list_files(
        self,
        *,
        q: str,
        fields: str,
        page_token: str | None,
        page_size: int,
    ) -> dict[str, Any]:
        service = self._service_handle()
        return self.call_with_retry(
            lambda t=page_token: service.files().list(q=q, fields=fields, pageToken=t, pageSize=page_size).execute()
        )

    def _download_request(
        self,
        request: Any,
        handle: Any,
        downloader_cls: Any,
        *,
        file_id: str,
    ) -> None:
        downloader = downloader_cls(handle, request)
        done = False
        max_chunks = 10_000
        chunks = 0
        while not done:
            _, done = downloader.next_chunk()
            chunks += 1
            if chunks >= max_chunks:
                raise DriveError(f"Download exceeded {max_chunks} chunks for file {file_id}")

    def download_file(self, file_id: str, handle: BinaryIO) -> None:
        """Download file content into handle."""
        media_io_base_download_cls = _import_module("googleapiclient.http").MediaIoBaseDownload
        service = self._service_handle()
        request = service.files().get_media(fileId=file_id)
        self._download_request(request, handle, media_io_base_download_cls, file_id=file_id)


__all__ = [
    "DEFAULT_DRIVE_RETRIES",
    "DEFAULT_DRIVE_RETRY_BASE",
    "DriveServiceGateway",
    "_import_module",
    "_resolve_retries",
    "_resolve_retry_base",
]

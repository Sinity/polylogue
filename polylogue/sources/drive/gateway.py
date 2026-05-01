from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import ParamSpec, Protocol, TypeAlias, TypeVar, Unpack, runtime_checkable

from tenacity import (
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from typing_extensions import TypedDict

from polylogue.core.json import JSONDocument, JSONDocumentList
from polylogue.logging import get_logger

from .types import (
    DriveAuthError,
    DriveConfigLike,
    DriveCredentialLike,
    DriveNotFoundError,
    DriveRetryPolicy,
)
from .types import DriveError as DriveServiceError

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
DrivePayloadRecord: TypeAlias = JSONDocument


class DriveListFilesResponse(TypedDict, total=False):
    files: JSONDocumentList
    nextPageToken: str


class _DriveGetKwargs(TypedDict):
    fileId: str
    fields: str


class _DriveListKwargs(TypedDict):
    q: str
    fields: str
    pageToken: str | None
    pageSize: int


class _DriveGetMediaKwargs(TypedDict):
    fileId: str


DEFAULT_DRIVE_RETRIES = 3
DEFAULT_DRIVE_RETRY_BASE = 0.5


class _DriveAuthManagerLike(Protocol):
    def load_credentials(self) -> DriveCredentialLike: ...


class _ExecutableRequest(Protocol[T_co]):
    def execute(self) -> T_co: ...


class _BinaryWritable(Protocol):
    def write(self, data: bytes) -> object: ...


class _DriveFilesResource(Protocol):
    def get(self, **kwargs: Unpack[_DriveGetKwargs]) -> _ExecutableRequest[DrivePayloadRecord]: ...

    def list(self, **kwargs: Unpack[_DriveListKwargs]) -> _ExecutableRequest[DriveListFilesResponse]: ...

    def get_media(self, **kwargs: Unpack[_DriveGetMediaKwargs]) -> object: ...


@runtime_checkable
class _DriveService(Protocol):
    _http: object | None

    def files(self) -> _DriveFilesResource: ...


class _DriveServiceBuilder(Protocol):
    def __call__(
        self,
        api_name: str,
        api_version: str,
        *,
        credentials: DriveCredentialLike,
        cache_discovery: bool,
    ) -> _DriveService: ...


class _MediaIoBaseDownload(Protocol):
    def next_chunk(self) -> tuple[object | None, bool]: ...


MediaDownloadFactory = Callable[[_BinaryWritable, object], _MediaIoBaseDownload]


def _import_module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise DriveAuthError(
            "Drive dependencies are not available. "
            "Install google-api-python-client + google-auth-oauthlib "
            "or run Polylogue from a Nix build/dev shell."
        ) from exc


def _resolve_retries(value: int | None, config: DriveConfigLike | None = None) -> int:
    """Resolve retry count from explicit value, config, or default."""
    if value is not None:
        return max(0, int(value))

    configured = config.retry_count if config is not None else None
    if configured is not None:
        return max(0, int(configured))

    return DEFAULT_DRIVE_RETRIES


def _resolve_retry_base(value: float | None) -> float:
    if value is not None:
        return max(0.0, float(value))
    return DEFAULT_DRIVE_RETRY_BASE


def resolve_drive_retry_policy(
    *,
    retries: int | None,
    retry_base: float | None,
    config: DriveConfigLike | None = None,
) -> DriveRetryPolicy:
    return DriveRetryPolicy(
        retries=_resolve_retries(retries, config),
        retry_base=_resolve_retry_base(retry_base),
    )


class DriveServiceGateway:
    """Owns Google API imports, service construction, raw Drive calls, and retry policy."""

    def __init__(
        self,
        *,
        auth_manager: _DriveAuthManagerLike,
        retry_policy: DriveRetryPolicy,
    ) -> None:
        self._auth_manager = auth_manager
        self._retry_policy = retry_policy
        self._service: _DriveService | None = None

    def call_with_retry(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        from tenacity import Retrying

        retryer = Retrying(
            stop=stop_after_attempt(max(self._retry_policy.retries, 0) + 1),
            wait=wait_exponential(
                multiplier=self._retry_policy.retry_base,
                min=self._retry_policy.retry_base,
                max=10,
            ),
            retry=retry_if_exception_type(Exception)
            & retry_if_not_exception_type((DriveAuthError, DriveNotFoundError)),
            reraise=True,
        )
        return retryer(func, *args, **kwargs)

    @staticmethod
    def _credentials_expired(service: object) -> bool:
        http = getattr(service, "_http", None)
        if http is None:
            return False
        credentials = getattr(http, "credentials", None)
        return bool(getattr(credentials, "expired", False))

    def _service_handle(self) -> _DriveService:
        if self._service is not None:
            if self._credentials_expired(self._service):
                logger.info("Cached service credentials expired, re-authenticating")
                self._service = None
                return self._service_handle()
            return self._service

        discovery = _import_module("googleapiclient.discovery")
        build: _DriveServiceBuilder = discovery.build
        creds = self._auth_manager.load_credentials()
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return self._service

    def get_file(self, file_id: str, fields: str) -> DrivePayloadRecord:
        service = self._service_handle()
        return self.call_with_retry(lambda: service.files().get(fileId=file_id, fields=fields).execute())

    def list_files(
        self,
        *,
        q: str,
        fields: str,
        page_token: str | None,
        page_size: int,
    ) -> DriveListFilesResponse:
        service = self._service_handle()

        def _load_page() -> DriveListFilesResponse:
            return service.files().list(q=q, fields=fields, pageToken=page_token, pageSize=page_size).execute()

        return self.call_with_retry(_load_page)

    def _download_request(
        self,
        request: object,
        handle: _BinaryWritable,
        downloader_cls: MediaDownloadFactory,
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
                raise DriveServiceError(f"Download exceeded {max_chunks} chunks for file {file_id}")

    def download_file(self, file_id: str, handle: _BinaryWritable) -> None:
        """Download file content into a writable binary handle."""
        http_module = _import_module("googleapiclient.http")
        downloader_cls: MediaDownloadFactory = http_module.MediaIoBaseDownload
        service = self._service_handle()
        request = service.files().get_media(fileId=file_id)
        self._download_request(request, handle, downloader_cls, file_id=file_id)


__all__ = [
    "DEFAULT_DRIVE_RETRIES",
    "DEFAULT_DRIVE_RETRY_BASE",
    "DriveServiceGateway",
    "_import_module",
    "_resolve_retries",
    "_resolve_retry_base",
    "resolve_drive_retry_policy",
]

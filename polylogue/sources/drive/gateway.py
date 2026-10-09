from __future__ import annotations

import importlib
import json
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

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.json import JSONDocument, JSONDocumentList
from polylogue.logging import get_logger

from .types import (
    DriveAccessDeniedError,
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

#: Drive error reasons (``error.errors[].reason``, ``error.details[].reason``
#: and ``error.status``, compared case-folded) that report a quota or rate
#: window. Drive sends them with HTTP 403 as well as 429, so the status alone
#: cannot tell a throttled request from a refused file.
_DRIVE_RATE_LIMIT_REASONS = frozenset(
    {
        "dailylimitexceeded",
        "downloadquotaexceeded",
        "quotaexceeded",
        "rate_limit_exceeded",
        "ratelimitexceeded",
        "resource_exhausted",
        "sharingratelimitexceeded",
        "userratelimitexceeded",
    }
)
#: ``error.errors[].domain`` for every Drive usage-limit reason.
_DRIVE_USAGE_LIMIT_DOMAIN = "usagelimits"


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


def _http_error_type() -> type[BaseException] | None:
    try:
        http_error = _import_module("googleapiclient.errors").HttpError
    except DriveAuthError:
        return None
    return http_error if isinstance(http_error, type) and issubclass(http_error, BaseException) else None


def _http_error_status(exc: BaseException) -> int | None:
    status = getattr(getattr(exc, "resp", None), "status", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _http_error_reasons(exc: BaseException) -> frozenset[str]:
    """Every reason, usage domain, and status token the Drive error body names."""
    content = getattr(exc, "content", None)
    if not isinstance(content, bytes | str):
        return frozenset()
    try:
        document = json.loads(content)
    except (UnicodeDecodeError, ValueError):
        return frozenset()
    error = document.get("error") if isinstance(document, dict) else None
    if not isinstance(error, dict):
        return frozenset()
    reasons: set[str] = set()
    status = error.get("status")
    if isinstance(status, str) and status:
        reasons.add(status.casefold())
    for key in ("errors", "details"):
        entries = error.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            for field in ("reason", "domain"):
                value = entry.get(field)
                if isinstance(value, str) and value:
                    reasons.add(value.casefold())
    return frozenset(reasons)


def drive_http_failure(exc: BaseException) -> DriveNotFoundError | DriveAccessDeniedError | None:
    """Translate a permanent provider HTTP answer into its typed Drive error.

    Only an explicit answer about the file is permanent: 404, or a 403 whose
    body names a reason and none of them is a quota or rate window. A
    throttled 403, a 403 with no readable reason, and every other status stay
    the provider's own exception, which the retry policy and later
    convergence passes treat as retryable.
    """
    http_error = _http_error_type()
    if http_error is None or not isinstance(exc, http_error):
        return None
    status = _http_error_status(exc)
    if status == 404:
        return DriveNotFoundError(str(exc))
    if status != 403:
        return None
    reasons = _http_error_reasons(exc)
    if not reasons or reasons & _DRIVE_RATE_LIMIT_REASONS or _DRIVE_USAGE_LIMIT_DOMAIN in reasons:
        return None
    return DriveAccessDeniedError(str(exc))


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

        def attempt() -> T:
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                permanent = drive_http_failure(exc)
                if permanent is None:
                    raise
                raise permanent from exc

        retryer = Retrying(
            stop=stop_after_attempt(max(self._retry_policy.retries, 0) + 1),
            wait=wait_exponential(
                multiplier=self._retry_policy.retry_base,
                min=self._retry_policy.retry_base,
                max=10,
            ),
            retry=retry_if_exception_type(Exception)
            & retry_if_not_exception_type(
                (DriveAuthError, DriveNotFoundError, DriveAccessDeniedError, DaemonOperationCancelled)
            ),
            reraise=True,
        )
        return retryer(attempt)

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
        # A chunk count bounds neither bytes nor elapsed work. A valid large
        # download can take more than 10,000 chunks, including one that
        # completes exactly on that boundary. Detect a stuck stream instead.
        teller = getattr(handle, "tell", None)

        def position() -> int | None:
            if not callable(teller):
                return None
            try:
                observed = teller()
            except (OSError, TypeError, ValueError):
                return None
            return observed if isinstance(observed, int) and observed >= 0 else None

        prior_position = position()
        stalled_chunks = 0
        while True:
            check_compute_cancelled()
            _, done = downloader.next_chunk()
            if done:
                return
            next_position = position()
            if prior_position is not None and next_position is not None:
                stalled_chunks = 0 if next_position > prior_position else stalled_chunks + 1
                if stalled_chunks >= 100:
                    raise DriveServiceError(f"Download made no byte progress for file {file_id}")
            prior_position = next_position

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
    "drive_http_failure",
    "resolve_drive_retry_policy",
]

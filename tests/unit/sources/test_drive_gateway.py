"""Contract tests for DriveServiceGateway — retry, service lifecycle, and transport."""

from __future__ import annotations

from types import ModuleType
from unittest.mock import MagicMock

import pytest

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.sources.drive.gateway import (
    DEFAULT_DRIVE_RETRIES,
    DEFAULT_DRIVE_RETRY_BASE,
    DriveServiceGateway,
    _BinaryWritable,
    _DriveService,
    _import_module,
    _resolve_retries,
    _resolve_retry_base,
    resolve_drive_retry_policy,
)
from polylogue.sources.drive.types import (
    DriveAccessDeniedError,
    DriveAuthError,
    DriveError,
    DriveNotFoundError,
    DriveRetryPolicy,
)
from tests.infra.drive_mocks import MockDriveService, MockMediaIoBaseDownload, drive_http_error


def _as_drive_service(value: object) -> _DriveService:
    if not isinstance(value, _DriveService):
        raise TypeError(f"expected _DriveService, got {type(value).__name__}")
    return value


def _as_mock(value: object) -> MagicMock:
    if not isinstance(value, MagicMock):
        raise TypeError(f"expected MagicMock, got {type(value).__name__}")
    return value


def _fake_module(**attrs: object) -> ModuleType:
    module = ModuleType("fake_google_module")
    for name, value in attrs.items():
        setattr(module, name, value)
    return module


def _gateway(*, retries: int = 0, retry_base: float = 0.0) -> DriveServiceGateway:
    """Build a gateway with a mock auth manager."""
    auth_manager = MagicMock()
    auth_manager.load_credentials.return_value = object()
    return DriveServiceGateway(
        auth_manager=auth_manager,
        retry_policy=DriveRetryPolicy(retries=retries, retry_base=retry_base),
    )


# ---------------------------------------------------------------------------
# _import_module
# ---------------------------------------------------------------------------


def test_import_module_wraps_missing_drive_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "polylogue.sources.drive.gateway.importlib.import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )
    with pytest.raises(DriveAuthError, match="Drive dependencies are not available"):
        _import_module("googleapiclient.discovery")


# ---------------------------------------------------------------------------
# _resolve_retries / _resolve_retry_base
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("explicit", "config_value", "expected"),
    [
        (5, None, 5),
        (0, None, 0),
        (-5, None, 0),
        (None, 7, 7),
        (None, -2, 0),
        (None, None, DEFAULT_DRIVE_RETRIES),
        (10, 5, 10),
        (None, 5, 5),
    ],
)
def test_resolve_retries_precedence_contract(
    explicit: int | None,
    config_value: int | None,
    expected: int,
) -> None:
    config = None if config_value is None else MagicMock(retry_count=config_value)
    assert _resolve_retries(value=explicit, config=config) == expected


@pytest.mark.parametrize(
    ("explicit", "expected"),
    [
        (1.5, 1.5),
        (0.1, 0.1),
        (-0.5, 0.0),
        (None, DEFAULT_DRIVE_RETRY_BASE),
    ],
)
def test_resolve_retry_base_contract(
    explicit: float | None,
    expected: float,
) -> None:
    assert _resolve_retry_base(explicit) == expected


def test_resolve_drive_retry_policy_contract() -> None:
    config = MagicMock(retry_count=7)
    assert resolve_drive_retry_policy(retries=None, retry_base=None, config=config) == DriveRetryPolicy(
        retries=7,
        retry_base=DEFAULT_DRIVE_RETRY_BASE,
    )


# ---------------------------------------------------------------------------
# call_with_retry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("retries", "failure_count", "terminal_error", "succeeds"),
    [
        (2, 2, None, True),
        (5, 1, DriveAuthError, False),
        (5, 1, DriveNotFoundError, False),
        (5, 1, DaemonOperationCancelled, False),
        (2, 3, None, False),
    ],
    ids=["transient-recovers", "auth-terminal", "notfound-terminal", "cancelled", "exhausted"],
)
def test_call_with_retry_contract(
    retries: int,
    failure_count: int,
    terminal_error: type[Exception] | None,
    succeeds: bool,
) -> None:
    gw = _gateway(retries=retries, retry_base=0.0)
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if terminal_error is not None:
            raise terminal_error("stop")
        if attempts["count"] <= failure_count:
            raise RuntimeError("temporary failure")
        return "ok"

    if succeeds:
        assert gw.call_with_retry(flaky) == "ok"
    elif terminal_error is not None:
        with pytest.raises(terminal_error, match="stop"):
            gw.call_with_retry(flaky)
    else:
        with pytest.raises(RuntimeError, match="temporary failure"):
            gw.call_with_retry(flaky)

    expected_attempts = 1 if terminal_error is not None else min(failure_count + 1, retries + 1)
    assert attempts["count"] == expected_attempts


@pytest.mark.parametrize(
    ("status", "errors", "raised", "retried"),
    [
        (403, (("usageLimits", "userRateLimitExceeded"),), None, True),
        (403, (("usageLimits", "rateLimitExceeded"),), None, True),
        (403, (("global", "downloadQuotaExceeded"),), None, True),
        (403, (), None, True),
        (403, (("global", "insufficientFilePermissions"),), DriveAccessDeniedError, False),
        (404, (("global", "notFound"),), DriveNotFoundError, False),
    ],
    ids=["user-rate-limit", "rate-limit", "download-quota", "unreadable-403", "denied", "not-found"],
)
def test_call_with_retry_tells_a_throttled_403_from_a_denied_file(
    status: int,
    errors: tuple[tuple[str, str], ...],
    raised: type[Exception] | None,
    retried: bool,
) -> None:
    """07.F002: a 403 is permanent only when Drive says it is about the file.

    Anti-vacuity: classify on status alone (``status in {403, 404}``) and the
    three throttled cases become ``DriveAccessDeniedError`` after one
    attempt; drop the translation and the denied and not-found cases are
    retried and escape as the raw ``HttpError``.
    """
    retries = 2
    gw = _gateway(retries=retries, retry_base=0.0)
    failure = drive_http_error(status, *errors)
    attempts = {"count": 0}

    def fail() -> None:
        attempts["count"] += 1
        raise failure

    if raised is None:
        with pytest.raises(type(failure)) as caught:
            gw.call_with_retry(fail)
        assert caught.value is failure
    else:
        with pytest.raises(raised) as caught_typed:
            gw.call_with_retry(fail)
        assert caught_typed.value.__cause__ is failure
    assert attempts["count"] == (retries + 1 if retried else 1)


# ---------------------------------------------------------------------------
# _service_handle — cache and rebuild
# ---------------------------------------------------------------------------


def test_service_handle_returns_cached_service_when_not_expired(monkeypatch: pytest.MonkeyPatch) -> None:
    gw = _gateway()
    service = MockDriveService()
    service._http = MagicMock()
    service._http.credentials.expired = False
    cached_service = _as_drive_service(service)
    gw._service = cached_service
    assert gw._service_handle() is cached_service


def test_service_handle_rebuilds_on_expired_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    gw = _gateway()
    expired = MockDriveService()
    expired._http = MagicMock()
    expired._http.credentials.expired = True
    gw._service = _as_drive_service(expired)
    creds = object()
    rebuilt = MockDriveService()
    load_credentials = _as_mock(gw._auth_manager.load_credentials)
    load_credentials.return_value = creds
    rebuilt_service = _as_drive_service(rebuilt)
    build = MagicMock(return_value=rebuilt_service)

    def fake_import(name: str) -> ModuleType:
        if name == "googleapiclient.discovery":
            return _fake_module(build=build)
        raise AssertionError(name)

    monkeypatch.setattr("polylogue.sources.drive.gateway._import_module", fake_import)
    assert gw._service_handle() is rebuilt_service
    load_credentials.assert_called_once_with()
    build.assert_called_once_with("drive", "v3", credentials=creds, cache_discovery=False)


def test_service_handle_builds_and_caches_service(monkeypatch: pytest.MonkeyPatch) -> None:
    gw = _gateway()
    creds = object()
    built = object()
    load_credentials = _as_mock(gw._auth_manager.load_credentials)
    load_credentials.return_value = creds
    build = MagicMock(return_value=built)

    def fake_import(name: str) -> ModuleType:
        if name == "googleapiclient.discovery":
            return _fake_module(build=build)
        raise AssertionError(name)

    monkeypatch.setattr("polylogue.sources.drive.gateway._import_module", fake_import)

    first = gw._service_handle()
    second = gw._service_handle()

    assert first is built
    assert second is built
    load_credentials.assert_called_once_with()
    build.assert_called_once_with("drive", "v3", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------------------
# download_file — chunk loop via _download_request
# ---------------------------------------------------------------------------


def test_download_file_writes_content(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    gw = _gateway()
    gw._service = _as_drive_service(MockDriveService(file_content={"file-1": b"hello-bytes"}))

    monkeypatch.setattr(
        "polylogue.sources.drive.gateway._import_module",
        lambda name: (
            MagicMock(MediaIoBaseDownload=MockMediaIoBaseDownload)
            if name == "googleapiclient.http"
            else (_ for _ in ()).throw(AssertionError(name))
        ),
    )

    buf = io.BytesIO()
    gw.download_file("file-1", buf)
    assert buf.getvalue() == b"hello-bytes"


@pytest.mark.parametrize("chunk_count", [10_000, 10_001])
def test_download_accepts_completion_at_and_beyond_old_chunk_ceiling(chunk_count: int) -> None:
    import io

    class ChunkedDownload:
        def __init__(self, handle: _BinaryWritable, total: object) -> None:
            if not isinstance(total, int):
                raise TypeError("fixture chunk count must be an integer")
            self.handle = handle
            self.total = total
            self.count = 0

        def next_chunk(self) -> tuple[None, bool]:
            self.handle.write(b"x")
            self.count += 1
            return None, self.count == self.total

    output = io.BytesIO()
    _gateway()._download_request(chunk_count, output, ChunkedDownload, file_id="fixture")
    assert output.getvalue() == b"x" * chunk_count


def test_download_refuses_consecutive_chunks_without_byte_progress() -> None:
    import io

    class StalledDownload:
        calls = 0

        def __init__(self, _handle: _BinaryWritable, _request: object) -> None:
            pass

        def next_chunk(self) -> tuple[None, bool]:
            StalledDownload.calls += 1
            return None, False

    output = io.BytesIO()
    with pytest.raises(DriveError, match="no byte progress"):
        _gateway()._download_request(object(), output, StalledDownload, file_id="fixture")
    assert StalledDownload.calls == 100

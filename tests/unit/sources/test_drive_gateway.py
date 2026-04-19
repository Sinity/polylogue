"""Contract tests for DriveServiceGateway — retry, service lifecycle, and transport."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from polylogue.sources.drive_gateway import (
    DEFAULT_DRIVE_RETRIES,
    DEFAULT_DRIVE_RETRY_BASE,
    DriveServiceGateway,
    _import_module,
    _resolve_retries,
    _resolve_retry_base,
)
from polylogue.sources.drive_types import DriveAuthError, DriveNotFoundError
from tests.infra.drive_mocks import MockDriveService, MockMediaIoBaseDownload


def _gateway(*, retries: int = 0, retry_base: float = 0.0) -> DriveServiceGateway:
    """Build a gateway with a mock auth manager."""
    auth_manager = MagicMock()
    auth_manager.load_credentials.return_value = object()
    return DriveServiceGateway(auth_manager=auth_manager, retries=retries, retry_base=retry_base)


# ---------------------------------------------------------------------------
# _import_module
# ---------------------------------------------------------------------------


def test_import_module_wraps_missing_drive_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "polylogue.sources.drive_gateway.importlib.import_module",
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


# ---------------------------------------------------------------------------
# call_with_retry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("retries", "failure_count", "terminal_error", "succeeds"),
    [
        (2, 2, None, True),
        (5, 1, DriveAuthError, False),
        (5, 1, DriveNotFoundError, False),
        (2, 3, None, False),
    ],
    ids=["transient-recovers", "auth-terminal", "notfound-terminal", "exhausted"],
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


# ---------------------------------------------------------------------------
# _service_handle — cache and rebuild
# ---------------------------------------------------------------------------


def test_service_handle_returns_cached_service_when_not_expired(monkeypatch: pytest.MonkeyPatch) -> None:
    gw = _gateway()
    service = MagicMock()
    service._http.credentials.expired = False
    cast(Any, gw)._service = service
    assert gw._service_handle() is service


def test_service_handle_rebuilds_on_expired_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    gw = _gateway()
    expired = MagicMock()
    expired._http.credentials.expired = True
    cast(Any, gw)._service = expired
    creds = object()
    rebuilt = object()
    load_credentials = cast(MagicMock, gw._auth_manager.load_credentials)
    load_credentials.return_value = creds
    build = MagicMock(return_value=rebuilt)

    def fake_import(name: str) -> Any:
        if name == "googleapiclient.discovery":
            return MagicMock(build=build)
        raise AssertionError(name)

    monkeypatch.setattr("polylogue.sources.drive_gateway._import_module", fake_import)
    assert gw._service_handle() is rebuilt
    load_credentials.assert_called_once_with()
    build.assert_called_once_with("drive", "v3", credentials=creds, cache_discovery=False)


def test_service_handle_builds_and_caches_service(monkeypatch: pytest.MonkeyPatch) -> None:
    gw = _gateway()
    creds = object()
    built = object()
    load_credentials = cast(MagicMock, gw._auth_manager.load_credentials)
    load_credentials.return_value = creds
    build = MagicMock(return_value=built)

    def fake_import(name: str) -> Any:
        if name == "googleapiclient.discovery":
            return MagicMock(build=build)
        raise AssertionError(name)

    monkeypatch.setattr("polylogue.sources.drive_gateway._import_module", fake_import)

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
    cast(Any, gw)._service = MockDriveService(file_content={"file-1": b"hello-bytes"})

    monkeypatch.setattr(
        "polylogue.sources.drive_gateway._import_module",
        lambda name: (
            MagicMock(MediaIoBaseDownload=MockMediaIoBaseDownload)
            if name == "googleapiclient.http"
            else (_ for _ in ()).throw(AssertionError(name))
        ),
    )

    buf = io.BytesIO()
    gw.download_file("file-1", buf)
    assert buf.getvalue() == b"hello-bytes"

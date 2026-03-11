"""Law-based contracts for Drive client payload parsing and file filtering."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polylogue.sources.drive_client import (
    GEMINI_PROMPT_MIME_TYPE,
    DriveAuthError,
    DriveClient,
    DriveError,
    DriveFile,
    DriveNotFoundError,
    _import_module,
    _is_supported_drive_payload,
    _needs_download,
    _parse_downloaded_json_payload,
)
from tests.infra.drive_mocks import MockDriveService, MockMediaIoBaseDownload, mock_drive_file
from tests.infra.strategies import json_document_strategy


@given(
    st.lists(json_document_strategy(), min_size=0, max_size=8),
    st.sampled_from(("session.jsonl", "session.jsonl.txt", "session.ndjson", "SESSION.JSONL")),
)
@settings(max_examples=35)
def test_parse_downloaded_json_payload_preserves_newline_delimited_documents(
    documents: list[dict[str, object]],
    name: str,
) -> None:
    raw = b"\n\n".join(json.dumps(document).encode("utf-8") for document in documents)
    assert _parse_downloaded_json_payload(raw, name=name) == documents


@given(
    st.one_of(
        json_document_strategy(),
        st.lists(json_document_strategy(), min_size=0, max_size=6),
    )
)
@settings(max_examples=35)
def test_parse_downloaded_json_payload_round_trips_standard_json(payload: object) -> None:
    raw = json.dumps(payload).encode("utf-8")
    assert _parse_downloaded_json_payload(raw, name="payload.json") == payload


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=10_000),
            st.sampled_from((".json", ".jsonl", ".jsonl.txt", ".ndjson", ".txt", ".md", "")),
            st.sampled_from((
                "application/json",
                "text/plain",
                "application/octet-stream",
                GEMINI_PROMPT_MIME_TYPE,
            )),
            st.booleans(),
        ),
        min_size=1,
        max_size=20,
        unique_by=lambda item: item[0],
    )
)
@settings(max_examples=30, deadline=None)
def test_iter_json_files_filters_supported_entries(
    file_specs: list[tuple[int, str, str, bool]],
) -> None:
    folder_id = "folder-law"
    client = DriveClient()
    service = MockDriveService()
    client._service = service
    service._files_resource.files.clear()

    expected_ids: list[str] = []
    for file_num, suffix, mime_type, in_folder in file_specs:
        file_id = f"file-{file_num}"
        name = f"payload-{file_num}{suffix}"
        parents = [folder_id] if in_folder else ["other-folder"]
        service._files_resource.files[file_id] = mock_drive_file(
            file_id=file_id,
            name=name,
            mime_type=mime_type,
            parents=parents,
        )
        if in_folder and (
            name.lower().endswith((".json", ".jsonl", ".jsonl.txt", ".ndjson"))
            or mime_type == GEMINI_PROMPT_MIME_TYPE
        ):
            expected_ids.append(file_id)

    files = list(client.iter_json_files(folder_id))

    assert [file.file_id for file in files] == expected_ids
    assert list(client._meta_cache) == expected_ids


class _StubFlow:
    def __init__(self, *, credentials: object, fetch_error: Exception | None = None) -> None:
        self.credentials = credentials
        self.fetch_error = fetch_error
        self.fetch_codes: list[str] = []
        self.authorization_calls: list[tuple[str, str]] = []

    def authorization_url(self, *, prompt: str, access_type: str) -> tuple[str, None]:
        self.authorization_calls.append((prompt, access_type))
        return ("https://accounts.example/authorize", None)

    def fetch_token(self, *, code: str) -> None:
        self.fetch_codes.append(code)
        if self.fetch_error is not None:
            raise self.fetch_error


def _make_drive_ui(code: str | None) -> MagicMock:
    ui = MagicMock()
    ui.console = MagicMock()
    ui.input = MagicMock(return_value=code)
    return ui


def test_run_manual_auth_flow_success_contract() -> None:
    creds = object()
    flow = _StubFlow(credentials=creds)
    ui = _make_drive_ui("auth-code-123")
    client = DriveClient(ui=ui)

    result = client._run_manual_auth_flow(flow)

    assert result is creds
    assert flow.authorization_calls == [("consent", "offline")]
    assert flow.fetch_codes == ["auth-code-123"]
    ui.console.print.assert_any_call("Open this URL in your browser to authorize Drive access:")
    ui.console.print.assert_any_call("https://accounts.example/authorize")
    ui.input.assert_called_once_with("Paste the authorization code")


def test_run_manual_auth_flow_cancel_contract() -> None:
    flow = _StubFlow(credentials=object())
    ui = _make_drive_ui("")
    client = DriveClient(ui=ui)

    with pytest.raises(DriveAuthError, match="Drive authorization cancelled"):
        client._run_manual_auth_flow(flow)

    assert flow.fetch_codes == []


def test_run_manual_auth_flow_wraps_fetch_errors() -> None:
    error = RuntimeError("bad authorization code")
    flow = _StubFlow(credentials=object(), fetch_error=error)
    ui = _make_drive_ui("bad-code")
    client = DriveClient(ui=ui)

    with pytest.raises(DriveAuthError, match="Drive authorization failed") as exc_info:
        client._run_manual_auth_flow(flow)

    assert exc_info.value.__cause__ is error
    assert flow.fetch_codes == ["bad-code"]


def test_call_with_retry_retries_generic_errors_until_success() -> None:
    client = DriveClient(retries=2, retry_base=0.0)
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("temporary failure")
        return "ok"

    assert client._call_with_retry(flaky) == "ok"
    assert attempts["count"] == 3


@pytest.mark.parametrize("error_type", [DriveAuthError, DriveNotFoundError])
def test_call_with_retry_does_not_retry_terminal_errors(error_type: type[Exception]) -> None:
    client = DriveClient(retries=5, retry_base=0.0)
    attempts = {"count": 0}

    def fail() -> None:
        attempts["count"] += 1
        raise error_type("stop")

    with pytest.raises(error_type, match="stop"):
        client._call_with_retry(fail)

    assert attempts["count"] == 1


def test_call_with_retry_stops_after_retry_budget() -> None:
    client = DriveClient(retries=2, retry_base=0.0)
    attempts = {"count": 0}

    def fail() -> None:
        attempts["count"] += 1
        raise RuntimeError("still failing")

    with pytest.raises(RuntimeError, match="still failing"):
        client._call_with_retry(fail)

    assert attempts["count"] == 3


def test_service_handle_reuses_cached_service_when_not_expired() -> None:
    client = DriveClient()
    service = MagicMock()
    service._http.credentials.expired = False
    client._service = service

    assert client._service_handle() is service


def test_service_handle_rebuilds_expired_cached_service(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DriveClient()
    expired = MagicMock()
    expired._http.credentials.expired = True
    client._service = expired

    creds = object()
    rebuilt = object()
    client._load_credentials = MagicMock(return_value=creds)
    build = MagicMock(return_value=rebuilt)

    def fake_import(name: str):
        if name == "googleapiclient.discovery":
            return MagicMock(build=build)
        raise AssertionError(name)

    monkeypatch.setattr("polylogue.sources.drive_client._import_module", fake_import)

    assert client._service_handle() is rebuilt
    client._load_credentials.assert_called_once_with()
    build.assert_called_once_with("drive", "v3", credentials=creds, cache_discovery=False)


def test_import_module_wraps_missing_drive_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("polylogue.sources.drive_client.importlib.import_module", fake_import)

    with pytest.raises(DriveAuthError, match="Drive dependencies are not available"):
        _import_module("googleapiclient.discovery")


def test_download_to_path_skips_unchanged_redownload(tmp_path: Path) -> None:
    client = DriveClient()
    client.get_metadata = MagicMock(
        return_value=DriveFile(
            file_id="file-1",
            name="payload.json",
            mime_type="application/json",
            modified_time="2025-01-01T00:00:00Z",
            size_bytes=5,
        ),
    )
    client._service_handle = MagicMock(side_effect=AssertionError("download should not happen"))

    dest = tmp_path / "payload.json"
    dest.write_bytes(b"12345")
    ts = 1735689600.0
    import os

    os.utime(dest, (ts, ts))

    result = client.download_to_path("file-1", dest)

    assert result.name == "payload.json"
    client.get_metadata.assert_called_once_with("file-1")


@pytest.mark.parametrize(
    ("name", "mime_type", "expected"),
    [
        ("chat.json", "application/json", True),
        ("chat.jsonl", "text/plain", True),
        ("chat.jsonl.txt", "application/octet-stream", True),
        ("chat.ndjson", "application/octet-stream", True),
        ("prompt.bin", GEMINI_PROMPT_MIME_TYPE, True),
        ("notes.md", "text/markdown", False),
    ],
)
def test_supported_drive_payload_contract(name: str, mime_type: str, expected: bool) -> None:
    assert _is_supported_drive_payload(name, mime_type) is expected


def test_needs_download_contract(tmp_path: Path) -> None:
    dest = tmp_path / "payload.json"
    dest.write_text("12345", encoding="utf-8")
    ts = 1735689600.0
    import os

    os.utime(dest, (ts, ts))
    base = DriveFile(
        file_id="file-1",
        name="payload.json",
        mime_type="application/json",
        modified_time="2025-01-01T00:00:00Z",
        size_bytes=5,
    )

    assert _needs_download(base, dest) is False
    assert _needs_download(replace(base, size_bytes=6), dest) is True
    assert _needs_download(replace(base, modified_time="2025-01-01T00:00:05Z"), dest) is True
    assert _needs_download(base, tmp_path / "missing.json") is True


def test_resolve_folder_id_prefers_existing_folder_id() -> None:
    client = DriveClient()
    service = MockDriveService()
    folder = mock_drive_file(
        file_id="folder-1",
        name="Target Folder",
        mime_type="application/vnd.google-apps.folder",
        parents=[],
    )
    service._files_resource.files[folder.file_id] = folder
    client._service = service

    resolved = client.resolve_folder_id("folder-1")

    assert resolved == "folder-1"
    assert service._files_resource._list_filters == {}


def test_load_credentials_prefers_token_store_before_token_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    token_file = tmp_path / "token.json"
    token_file.write_text('{"token":"stale"}', encoding="utf-8")

    valid_creds = MagicMock()
    valid_creds.valid = True
    valid_creds.expired = False
    valid_creds.refresh_token = "refresh-token"
    valid_creds.to_json.return_value = '{"token":"fresh"}'
    credentials_cls = MagicMock()
    credentials_cls.from_authorized_user_info.return_value = valid_creds
    credentials_cls.from_authorized_user_file.side_effect = AssertionError("token file should not be used")

    def fake_import(name: str):
        if name == "google.oauth2.credentials":
            return MagicMock(Credentials=credentials_cls)
        if name == "google_auth_oauthlib.flow":
            return MagicMock(InstalledAppFlow=MagicMock())
        raise AssertionError(name)

    client = DriveClient(ui=None, token_path=token_file)
    client._token_store = MagicMock()
    client._token_store.load.return_value = '{"token":"from-store"}'
    monkeypatch.setattr("polylogue.sources.drive_client._import_module", fake_import)

    result = client._load_credentials()

    assert result is valid_creds
    credentials_cls.from_authorized_user_info.assert_called_once()
    client._token_store.save.assert_called_once_with("drive_token", '{"token":"fresh"}')


def test_load_credentials_falls_back_from_invalid_token_store_to_token_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    token_file = tmp_path / "token.json"
    token_file.write_text('{"token":"from-file"}', encoding="utf-8")

    file_creds = MagicMock()
    file_creds.valid = True
    file_creds.expired = False
    file_creds.refresh_token = "refresh-token"
    file_creds.to_json.return_value = '{"token":"from-file"}'
    credentials_cls = MagicMock()
    credentials_cls.from_authorized_user_info.side_effect = json.JSONDecodeError("bad", "x", 0)
    credentials_cls.from_authorized_user_file.return_value = file_creds

    def fake_import(name: str):
        if name == "google.oauth2.credentials":
            return MagicMock(Credentials=credentials_cls)
        if name == "google_auth_oauthlib.flow":
            return MagicMock(InstalledAppFlow=MagicMock())
        raise AssertionError(name)

    client = DriveClient(ui=None, token_path=token_file)
    client._token_store = MagicMock()
    client._token_store.load.return_value = "not-json"
    monkeypatch.setattr("polylogue.sources.drive_client._import_module", fake_import)

    result = client._load_credentials()

    assert result is file_creds
    credentials_cls.from_authorized_user_file.assert_called_once_with(str(token_file), ["https://www.googleapis.com/auth/drive.readonly"])


def test_load_credentials_refreshes_expired_token_and_persists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    token_file = tmp_path / "token.json"
    token_file.write_text('{"token":"stale"}', encoding="utf-8")
    refreshed_creds = MagicMock()
    refreshed_creds.valid = False
    refreshed_creds.expired = True
    refreshed_creds.refresh_token = "refresh-token"
    refreshed_creds.to_json.return_value = '{"token":"fresh"}'

    def refresh(request) -> None:
        refreshed_creds.valid = True
        refreshed_creds.expired = False

    refreshed_creds.refresh.side_effect = refresh
    credentials_cls = MagicMock()
    credentials_cls.from_authorized_user_info.return_value = refreshed_creds
    request_cls = MagicMock(return_value=SimpleNamespace(session=SimpleNamespace(timeout=None)))

    def fake_import(name: str):
        if name == "google.oauth2.credentials":
            return MagicMock(Credentials=credentials_cls)
        if name == "google_auth_oauthlib.flow":
            return MagicMock(InstalledAppFlow=MagicMock())
        if name == "google.auth.transport.requests":
            return SimpleNamespace(Request=request_cls)
        raise AssertionError(name)

    client = DriveClient(ui=None, token_path=token_file)
    client._token_store = MagicMock()
    client._token_store.load.return_value = '{"token":"stale"}'
    monkeypatch.setattr("polylogue.sources.drive_client._import_module", fake_import)

    result = client._load_credentials()

    assert result is refreshed_creds
    refreshed_creds.refresh.assert_called_once()
    client._token_store.save.assert_called_once_with("drive_token", '{"token":"fresh"}')


def test_load_credentials_returns_valid_cached_token_without_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    token_file = tmp_path / "token.json"
    token_file.write_text('{"token":"cached"}', encoding="utf-8")

    cached_creds = MagicMock()
    cached_creds.valid = True
    cached_creds.expired = False
    cached_creds.refresh_token = "refresh-token"
    cached_creds.to_json.return_value = '{"token":"cached"}'
    credentials_cls = MagicMock()
    credentials_cls.from_authorized_user_file.return_value = cached_creds

    def fake_import(name: str):
        if name == "google.oauth2.credentials":
            return MagicMock(Credentials=credentials_cls)
        if name == "google_auth_oauthlib.flow":
            return MagicMock(InstalledAppFlow=MagicMock())
        raise AssertionError(name)

    client = DriveClient(ui=None, token_path=token_file)
    client._token_store = MagicMock()
    client._token_store.load.return_value = None
    monkeypatch.setattr("polylogue.sources.drive_client._import_module", fake_import)

    result = client._load_credentials()

    assert result is cached_creds
    assert not cached_creds.refresh.called
    client._token_store.save.assert_called_once_with("drive_token", '{"token":"cached"}')


def test_load_credentials_raises_for_invalid_non_refreshable_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    token_file = tmp_path / "token.json"
    token_file.write_text('{"token":"stale"}', encoding="utf-8")

    invalid_creds = MagicMock()
    invalid_creds.valid = False
    invalid_creds.expired = True
    invalid_creds.refresh_token = None
    credentials_cls = MagicMock()
    credentials_cls.from_authorized_user_file.return_value = invalid_creds

    def fake_import(name: str):
        if name == "google.oauth2.credentials":
            return MagicMock(Credentials=credentials_cls)
        if name == "google_auth_oauthlib.flow":
            return MagicMock(InstalledAppFlow=MagicMock())
        raise AssertionError(name)

    client = DriveClient(ui=None, token_path=token_file)
    client._token_store = MagicMock()
    client._token_store.load.return_value = None
    monkeypatch.setattr("polylogue.sources.drive_client._import_module", fake_import)

    with pytest.raises(DriveAuthError, match="cannot be refreshed"):
        client._load_credentials()


def test_load_credentials_rejects_corrupt_plain_token_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    token_file = tmp_path / "token.json"
    token_file.write_text("not json", encoding="utf-8")
    credentials_cls = MagicMock()
    credentials_cls.from_authorized_user_file.side_effect = ValueError("bad token json")

    def fake_import(name: str):
        if name == "google.oauth2.credentials":
            return MagicMock(Credentials=credentials_cls)
        if name == "google_auth_oauthlib.flow":
            return MagicMock(InstalledAppFlow=MagicMock())
        raise AssertionError(name)

    client = DriveClient(ui=None, token_path=token_file)
    client._token_store = MagicMock()
    client._token_store.load.return_value = None
    monkeypatch.setattr("polylogue.sources.drive_client._import_module", fake_import)

    with pytest.raises(DriveAuthError, match="invalid or expired"):
        client._load_credentials()


def test_load_credentials_uses_manual_flow_when_local_server_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    credentials_path = tmp_path / "client.json"
    credentials_path.write_text('{"installed":{}}', encoding="utf-8")
    token_path = tmp_path / "token.json"
    issued_creds = MagicMock()
    issued_creds.to_json.return_value = '{"token":"issued"}'
    flow = MagicMock()
    flow.run_local_server.side_effect = OSError("port unavailable")
    flow.credentials = issued_creds
    installed_app_flow_cls = MagicMock()
    installed_app_flow_cls.from_client_secrets_file.return_value = flow
    manual_creds = MagicMock()
    manual_creds.to_json.return_value = '{"token":"manual"}'

    def fake_import(name: str):
        if name == "google.oauth2.credentials":
            return MagicMock(Credentials=MagicMock())
        if name == "google_auth_oauthlib.flow":
            return MagicMock(InstalledAppFlow=installed_app_flow_cls)
        raise AssertionError(name)

    client = DriveClient(ui=MagicMock(plain=False), credentials_path=credentials_path, token_path=token_path)
    client._token_store = MagicMock()
    client._token_store.load.return_value = None
    client._run_manual_auth_flow = MagicMock(return_value=manual_creds)
    monkeypatch.setattr("polylogue.sources.drive_client._import_module", fake_import)

    result = client._load_credentials()

    assert result is manual_creds
    client._run_manual_auth_flow.assert_called_once_with(flow)
    client._token_store.save.assert_called_once_with("drive_token", '{"token":"manual"}')


def test_resolve_folder_id_falls_back_to_name_on_lookup_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeHttpError(Exception):
        def __init__(self, status: int) -> None:
            super().__init__(f"http {status}")
            self.resp = SimpleNamespace(status=status)

    client = DriveClient(retries=0, retry_base=0.0)
    service = MockDriveService()
    resource = service._files_resource
    resource.get = MagicMock(side_effect=_FakeHttpError(404))
    resource.list = MagicMock(return_value=SimpleNamespace(execute=lambda: {"files": [{"id": "folder-by-name", "name": "Target"}]}))
    service._http = None
    client._service = service
    monkeypatch.setattr("polylogue.sources.drive_client._import_module", lambda name: SimpleNamespace(HttpError=_FakeHttpError))

    resolved = client.resolve_folder_id("Target")

    assert resolved == "folder-by-name"
    resource.list.assert_called_once()


def test_resolve_folder_id_raises_not_found_when_lookup_and_search_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeHttpError(Exception):
        def __init__(self, status: int) -> None:
            super().__init__(f"http {status}")
            self.resp = SimpleNamespace(status=status)

    client = DriveClient(retries=0, retry_base=0.0)
    service = MockDriveService()
    resource = service._files_resource
    resource.get = MagicMock(side_effect=_FakeHttpError(404))
    resource.list = MagicMock(return_value=SimpleNamespace(execute=lambda: {"files": []}))
    service._http = None
    client._service = service
    monkeypatch.setattr("polylogue.sources.drive_client._import_module", lambda name: SimpleNamespace(HttpError=_FakeHttpError))

    with pytest.raises(DriveNotFoundError, match="Folder not found: Missing"):
        client.resolve_folder_id("Missing")


def test_resolve_folder_id_escapes_names_and_skips_id_lookup_for_spaced_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeHttpError(Exception):
        def __init__(self, status: int) -> None:
            super().__init__(f"http {status}")
            self.resp = SimpleNamespace(status=status)

    client = DriveClient(retries=0, retry_base=0.0)
    service = MockDriveService()
    resource = service._files_resource
    resource.get = MagicMock(side_effect=AssertionError("id lookup should not run"))
    resource.list = MagicMock(return_value=SimpleNamespace(execute=lambda: {"files": [{"id": "folder-id", "name": "O'Hare"}]}))
    service._http = None
    client._service = service
    monkeypatch.setattr("polylogue.sources.drive_client._import_module", lambda name: SimpleNamespace(HttpError=_FakeHttpError))

    resolved = client.resolve_folder_id("O'Hare Folder")

    assert resolved == "folder-id"
    assert resource.list.call_args.kwargs["q"] == (
        "name = 'O\\'Hare Folder' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )


def test_get_metadata_uses_cache_and_falls_back_to_file_id_name() -> None:
    client = DriveClient()
    service = MockDriveService(
        files_data={
            "file-1": mock_drive_file(file_id="file-1", name="", mime_type="application/json", size=0),
        }
    )
    service._files_resource.get = MagicMock(wraps=service._files_resource.get)
    service._http = None
    client._service = service

    first = client.get_metadata("file-1")
    second = client.get_metadata("file-1")

    assert first.name == "file-1"
    assert second is first
    service._files_resource.get.assert_called_once_with(
        fileId="file-1",
        fields="id,name,mimeType,modifiedTime,size",
    )


def test_iter_json_files_paginates_and_caches_all_supported_entries() -> None:
    client = DriveClient(retries=0, retry_base=0.0)
    service = MockDriveService()
    resource = service._files_resource
    first_page = {
        "nextPageToken": "page-2",
        "files": [
            {"id": "f1", "name": "one.json", "mimeType": "application/json", "modifiedTime": None, "size": "1"},
            {"id": "skip", "name": "readme.txt", "mimeType": "text/plain", "modifiedTime": None, "size": "1"},
        ],
    }
    second_page = {
        "files": [
            {"id": "f2", "name": "two.ndjson", "mimeType": "application/octet-stream", "modifiedTime": None, "size": "2"},
            {"id": "f3", "name": "prompt.bin", "mimeType": GEMINI_PROMPT_MIME_TYPE, "modifiedTime": None, "size": "3"},
        ],
    }
    resource.list = MagicMock(
        side_effect=[
            SimpleNamespace(execute=lambda: first_page),
            SimpleNamespace(execute=lambda: second_page),
        ]
    )
    service._http = None
    client._service = service

    files = list(client.iter_json_files("folder-1"))

    assert [file.file_id for file in files] == ["f1", "f2", "f3"]
    assert list(client._meta_cache) == ["f1", "f2", "f3"]
    assert resource.list.call_args_list[1].kwargs["pageToken"] == "page-2"


def test_download_to_path_writes_content_and_sets_mtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = DriveClient(retries=0, retry_base=0.0)
    service = MockDriveService(
        files_data={
            "file-1": mock_drive_file(
                file_id="file-1",
                name="payload.json",
                mime_type="application/json",
                modified_time="2025-01-01T00:00:00Z",
                size=5,
            )
        },
        file_content={"file-1": b"12345"},
    )
    client._service = service

    def fake_import(name: str):
        if name == "googleapiclient.http":
            return SimpleNamespace(MediaIoBaseDownload=MockMediaIoBaseDownload)
        raise AssertionError(name)

    monkeypatch.setattr("polylogue.sources.drive_client._import_module", fake_import)
    dest = tmp_path / "payload.json"

    result = client.download_to_path("file-1", dest)

    assert result.file_id == "file-1"
    assert dest.read_bytes() == b"12345"
    assert abs(dest.stat().st_mtime - 1735689600.0) <= 1


def test_download_to_path_cleans_up_temp_file_when_download_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = DriveClient(retries=0, retry_base=0.0)
    client.get_metadata = MagicMock(
        return_value=DriveFile(
            file_id="file-1",
            name="payload.json",
            mime_type="application/json",
            modified_time=None,
            size_bytes=5,
        ),
    )
    service = MockDriveService(file_content={"file-1": b"12345"})
    client._service = service

    class _BrokenDownloader(MockMediaIoBaseDownload):
        def next_chunk(self):
            raise OSError("download blew up")

    def fake_import(name: str):
        if name == "googleapiclient.http":
            return SimpleNamespace(MediaIoBaseDownload=_BrokenDownloader)
        raise AssertionError(name)

    monkeypatch.setattr("polylogue.sources.drive_client._import_module", fake_import)
    dest = tmp_path / "payload.json"

    with pytest.raises(OSError, match="download blew up"):
        client.download_to_path("file-1", dest)

    assert not list(tmp_path.glob("tmp*"))
    assert not dest.exists()


def test_download_bytes_returns_media_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DriveClient(retries=0, retry_base=0.0)
    service = MockDriveService(file_content={"file-1": b'{"hello":"world"}'})
    client._service = service

    def fake_import(name: str):
        if name == "googleapiclient.http":
            return SimpleNamespace(MediaIoBaseDownload=MockMediaIoBaseDownload)
        raise AssertionError(name)

    monkeypatch.setattr("polylogue.sources.drive_client._import_module", fake_import)

    assert client.download_bytes("file-1") == b'{"hello":"world"}'


def test_download_json_payload_delegates_to_download_bytes() -> None:
    client = DriveClient()
    client.download_bytes = MagicMock(return_value=b'{"id":"payload"}')

    result = client.download_json_payload("file-1", name="payload.json")

    assert result == {"id": "payload"}
    client.download_bytes.assert_called_once_with("file-1")


def test_persist_token_saves_store_and_writes_private_file(tmp_path: Path) -> None:
    client = DriveClient(token_path=tmp_path / "token.json")
    client._token_store = MagicMock()
    creds = MagicMock()
    creds.to_json.return_value = '{"token":"persisted"}'
    token_path = tmp_path / "token.json"

    client._persist_token(creds, token_path)

    client._token_store.save.assert_called_once_with("drive_token", '{"token":"persisted"}')
    assert token_path.read_text(encoding="utf-8") == '{"token":"persisted"}'
    assert oct(token_path.stat().st_mode & 0o777) == "0o600"


def test_download_request_stops_after_chunk_limit() -> None:
    client = DriveClient()

    class _NeverEndingDownloader:
        def __init__(self, handle, request) -> None:
            self.calls = 0

        def next_chunk(self):
            self.calls += 1
            return None, False

    with pytest.raises(DriveError, match="Download exceeded 10000 chunks"):
        client._download_request(object(), object(), _NeverEndingDownloader, file_id="file-1")

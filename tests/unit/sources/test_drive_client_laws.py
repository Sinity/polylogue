"""Law-based contracts for Drive client auth, filtering, and downloads."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
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
    DriveFile,
    DriveNotFoundError,
    _import_module,
    _is_supported_drive_payload,
    _needs_download,
    _parse_downloaded_json_payload,
)
from tests.infra.drive_mocks import MockDriveService, MockMediaIoBaseDownload, mock_drive_file
from tests.infra.strategies import json_document_strategy


@dataclass(frozen=True)
class AuthLoadCase:
    name: str
    token_store_value: str | None
    token_file_value: str | None
    info_side_effect: object | None = None
    file_side_effect: object | None = None
    creds_factory: callable | None = None
    expect_message: str | None = None
    expect_manual_flow: bool = False
    refreshes: bool = False


@dataclass(frozen=True)
class FolderResolutionCase:
    folder_ref: str
    get_result: object
    list_result: list[dict[str, str]]
    expected: str | None
    expected_query: str | None = None
    expect_get: bool = True
    expect_error: type[Exception] | None = None


@dataclass(frozen=True)
class DownloadCase:
    name: str
    metadata: DriveFile
    existing_bytes: bytes | None = None
    existing_mtime: float | None = None
    downloader: type[MockMediaIoBaseDownload] = MockMediaIoBaseDownload
    service_bytes: bytes = b"12345"
    expect_write: bool = True
    expect_path_exists: bool = True
    expect_error: type[Exception] | None = None


@dataclass(frozen=True)
class RetryCase:
    retries: int
    failure_count: int
    terminal_error: type[Exception] | None = None
    succeeds: bool = True


def _creds(*, valid: bool, expired: bool, refresh_token: str | None = "refresh-token", token_json: str = '{"token":"fresh"}'):
    creds = MagicMock()
    creds.valid = valid
    creds.expired = expired
    creds.refresh_token = refresh_token
    creds.to_json.return_value = token_json
    return creds


@given(
    st.lists(json_document_strategy(), min_size=0, max_size=8),
    st.sampled_from(("session.jsonl", "session.jsonl.txt", "session.ndjson", "SESSION.JSONL")),
)
@settings(max_examples=35)
def test_parse_downloaded_json_payload_preserves_newline_delimited_documents(documents: list[dict[str, object]], name: str) -> None:
    raw = b"\n\n".join(json.dumps(document).encode("utf-8") for document in documents)
    assert _parse_downloaded_json_payload(raw, name=name) == documents


@given(st.one_of(json_document_strategy(), st.lists(json_document_strategy(), min_size=0, max_size=6)))
@settings(max_examples=35)
def test_parse_downloaded_json_payload_round_trips_standard_json(payload: object) -> None:
    raw = json.dumps(payload).encode("utf-8")
    assert _parse_downloaded_json_payload(raw, name="payload.json") == payload


@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=10_000),
            st.sampled_from((".json", ".jsonl", ".jsonl.txt", ".ndjson", ".txt", ".md", "")),
            st.sampled_from(("application/json", "text/plain", "application/octet-stream", GEMINI_PROMPT_MIME_TYPE)),
            st.booleans(),
        ),
        min_size=1,
        max_size=20,
        unique_by=lambda item: item[0],
    )
)
@settings(max_examples=30, deadline=None)
def test_iter_json_files_filters_supported_entries(file_specs: list[tuple[int, str, str, bool]]) -> None:
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
        service._files_resource.files[file_id] = mock_drive_file(file_id=file_id, name=name, mime_type=mime_type, parents=parents)
        if in_folder and (name.lower().endswith((".json", ".jsonl", ".jsonl.txt", ".ndjson")) or mime_type == GEMINI_PROMPT_MIME_TYPE):
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


@pytest.mark.parametrize(
    ("code", "fetch_error", "expected_message"),
    [
        ("auth-code-123", None, None),
        ("", None, "Drive authorization cancelled"),
        ("bad-code", RuntimeError("bad authorization code"), "Drive authorization failed"),
    ],
)
def test_run_manual_auth_flow_contract(code: str, fetch_error: Exception | None, expected_message: str | None) -> None:
    creds = object()
    flow = _StubFlow(credentials=creds, fetch_error=fetch_error)
    ui = _make_drive_ui(code)
    client = DriveClient(ui=ui)

    if expected_message is None:
        result = client._run_manual_auth_flow(flow)
        assert result is creds
        assert flow.fetch_codes == ["auth-code-123"]
    else:
        with pytest.raises(DriveAuthError, match=expected_message):
            client._run_manual_auth_flow(flow)

    assert flow.authorization_calls == [("consent", "offline")]
    ui.console.print.assert_any_call("Open this URL in your browser to authorize Drive access:")
    ui.console.print.assert_any_call("https://accounts.example/authorize")


@pytest.mark.parametrize(
    "case",
    [
        RetryCase(retries=2, failure_count=2, succeeds=True),
        RetryCase(retries=5, failure_count=1, terminal_error=DriveAuthError, succeeds=False),
        RetryCase(retries=5, failure_count=1, terminal_error=DriveNotFoundError, succeeds=False),
        RetryCase(retries=2, failure_count=3, succeeds=False),
    ],
    ids=lambda case: f"retries={case.retries}-failures={case.failure_count}-terminal={getattr(case.terminal_error, '__name__', 'none')}",
)
def test_call_with_retry_contract(case: RetryCase) -> None:
    client = DriveClient(retries=case.retries, retry_base=0.0)
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if case.terminal_error is not None:
            raise case.terminal_error("stop")
        if attempts["count"] <= case.failure_count:
            raise RuntimeError("temporary failure")
        return "ok"

    if case.succeeds:
        assert client._call_with_retry(flaky) == "ok"
    elif case.terminal_error is not None:
        with pytest.raises(case.terminal_error, match="stop"):
            client._call_with_retry(flaky)
    else:
        with pytest.raises(RuntimeError, match="temporary failure"):
            client._call_with_retry(flaky)

    expected_attempts = 1 if case.terminal_error is not None else min(case.failure_count + 1, case.retries + 1)
    assert attempts["count"] == expected_attempts


def test_service_handle_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DriveClient()
    service = MagicMock()
    service._http.credentials.expired = False
    client._service = service
    assert client._service_handle() is service

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
    monkeypatch.setattr("polylogue.sources.drive_client.importlib.import_module", lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)))
    with pytest.raises(DriveAuthError, match="Drive dependencies are not available"):
        _import_module("googleapiclient.discovery")


AUTH_LOAD_CASES = [
    AuthLoadCase(
        name="token-store preferred",
        token_store_value='{"token":"from-store"}',
        token_file_value='{"token":"stale"}',
        creds_factory=lambda: _creds(valid=True, expired=False),
    ),
    AuthLoadCase(
        name="invalid store falls back to file",
        token_store_value="not-json",
        token_file_value='{"token":"from-file"}',
        info_side_effect=json.JSONDecodeError("bad", "x", 0),
        creds_factory=lambda: _creds(valid=True, expired=False, token_json='{"token":"from-file"}'),
    ),
    AuthLoadCase(
        name="refresh expired token",
        token_store_value='{"token":"stale"}',
        token_file_value='{"token":"stale"}',
        creds_factory=lambda: _creds(valid=False, expired=True),
        refreshes=True,
    ),
    AuthLoadCase(
        name="invalid non-refreshable token fails",
        token_store_value=None,
        token_file_value='{"token":"stale"}',
        creds_factory=lambda: _creds(valid=False, expired=True, refresh_token=None),
        expect_message="cannot be refreshed",
    ),
    AuthLoadCase(
        name="corrupt token file fails in plain mode",
        token_store_value=None,
        token_file_value="not json",
        file_side_effect=ValueError("bad token json"),
        expect_message="invalid or expired",
    ),
]


@pytest.mark.parametrize("case", AUTH_LOAD_CASES, ids=lambda case: case.name)
def test_load_credentials_state_machine(case: AuthLoadCase, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    token_path = tmp_path / "token.json"
    if case.token_file_value is not None:
        token_path.write_text(case.token_file_value, encoding="utf-8")

    credentials_cls = MagicMock()
    creds = case.creds_factory() if case.creds_factory is not None else None
    if case.info_side_effect is not None:
        credentials_cls.from_authorized_user_info.side_effect = case.info_side_effect
    elif creds is not None:
        credentials_cls.from_authorized_user_info.return_value = creds
    if case.file_side_effect is not None:
        credentials_cls.from_authorized_user_file.side_effect = case.file_side_effect
    elif creds is not None:
        credentials_cls.from_authorized_user_file.return_value = creds

    request_cls = MagicMock(return_value=SimpleNamespace(session=SimpleNamespace(timeout=None)))

    def fake_import(name: str):
        if name == "google.oauth2.credentials":
            return MagicMock(Credentials=credentials_cls)
        if name == "google_auth_oauthlib.flow":
            return MagicMock(InstalledAppFlow=MagicMock())
        if name == "google.auth.transport.requests":
            return SimpleNamespace(Request=request_cls)
        raise AssertionError(name)

    client = DriveClient(ui=None, token_path=token_path)
    client._token_store = MagicMock()
    client._token_store.load.return_value = case.token_store_value
    monkeypatch.setattr("polylogue.sources.drive_client._import_module", fake_import)

    if case.refreshes and creds is not None:
        def refresh(_request) -> None:
            creds.valid = True
            creds.expired = False
        creds.refresh.side_effect = refresh

    if case.expect_message is not None:
        with pytest.raises(DriveAuthError, match=case.expect_message):
            client._load_credentials()
        return

    result = client._load_credentials()
    assert result is creds
    client._token_store.save.assert_called_once_with("drive_token", creds.to_json())
    if case.refreshes and creds is not None:
        creds.refresh.assert_called_once()


def test_load_credentials_uses_manual_flow_when_local_server_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    credentials_path = tmp_path / "client.json"
    credentials_path.write_text('{"installed":{}}', encoding="utf-8")
    token_path = tmp_path / "token.json"
    flow = MagicMock()
    flow.run_local_server.side_effect = OSError("port unavailable")
    installed_app_flow_cls = MagicMock()
    installed_app_flow_cls.from_client_secrets_file.return_value = flow
    manual_creds = _creds(valid=True, expired=False, token_json='{"token":"manual"}')

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


@pytest.mark.parametrize(
    "case",
    [
        FolderResolutionCase(
            folder_ref="folder-1",
            get_result={"id": "folder-1", "mimeType": "application/vnd.google-apps.folder"},
            list_result=[],
            expected="folder-1",
        ),
        FolderResolutionCase(
            folder_ref="Target",
            get_result=Exception("skip id lookup"),
            list_result=[{"id": "folder-by-name", "name": "Target"}],
            expected="folder-by-name",
            expect_get=False,
        ),
        FolderResolutionCase(
            folder_ref="O'Hare Folder",
            get_result=AssertionError("id lookup should not run"),
            list_result=[{"id": "folder-id", "name": "O'Hare"}],
            expected="folder-id",
            expected_query="name = 'O\\'Hare Folder' and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
            expect_get=False,
        ),
        FolderResolutionCase(
            folder_ref="Missing",
            get_result=Exception("404"),
            list_result=[],
            expected=None,
            expect_get=False,
            expect_error=DriveNotFoundError,
        ),
    ],
    ids=lambda case: case.folder_ref,
)
def test_resolve_folder_id_contract(case: FolderResolutionCase, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeHttpError(Exception):
        def __init__(self, status: int) -> None:
            super().__init__(f"http {status}")
            self.resp = SimpleNamespace(status=status)

    client = DriveClient(retries=0, retry_base=0.0)
    service = MockDriveService()
    resource = service._files_resource
    if case.expect_get:
        resource.get = MagicMock(return_value=SimpleNamespace(execute=lambda: case.get_result))
    else:
        resource.get = MagicMock(side_effect=AssertionError("id lookup should not run"))
    resource.list = MagicMock(return_value=SimpleNamespace(execute=lambda: {"files": case.list_result}))
    service._http = None
    client._service = service
    monkeypatch.setattr("polylogue.sources.drive_client._import_module", lambda name: SimpleNamespace(HttpError=_FakeHttpError))

    if case.folder_ref in {"Missing", "Target"}:
        resource.get = MagicMock(side_effect=_FakeHttpError(404))

    if case.expect_error is not None:
        with pytest.raises(case.expect_error, match="Folder not found"):
            client.resolve_folder_id(case.folder_ref)
    else:
        assert client.resolve_folder_id(case.folder_ref) == case.expected

    if case.expected_query is not None:
        assert resource.list.call_args.kwargs["q"] == case.expected_query


@pytest.mark.parametrize(
    "case",
    [
        DownloadCase(
            name="skip unchanged",
            metadata=DriveFile(file_id="file-1", name="payload.json", mime_type="application/json", modified_time="2025-01-01T00:00:00Z", size_bytes=5),
            existing_bytes=b"12345",
            existing_mtime=1735689600.0,
            expect_write=False,
        ),
        DownloadCase(
            name="write and set mtime",
            metadata=DriveFile(file_id="file-1", name="payload.json", mime_type="application/json", modified_time="2025-01-01T00:00:00Z", size_bytes=5),
        ),
    ],
    ids=lambda case: case.name,
)
def test_download_to_path_contract(case: DownloadCase, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = DriveClient(retries=0, retry_base=0.0)
    client.get_metadata = MagicMock(return_value=case.metadata)
    service = MockDriveService(file_content={case.metadata.file_id: case.service_bytes})
    client._service = service
    dest = tmp_path / case.metadata.name

    if case.existing_bytes is not None:
        dest.write_bytes(case.existing_bytes)
    if case.existing_mtime is not None:
        import os
        os.utime(dest, (case.existing_mtime, case.existing_mtime))

    monkeypatch.setattr("polylogue.sources.drive_client._import_module", lambda name: SimpleNamespace(MediaIoBaseDownload=case.downloader) if name == "googleapiclient.http" else (_ for _ in ()).throw(AssertionError(name)))

    result = client.download_to_path(case.metadata.file_id, dest)
    assert result.file_id == case.metadata.file_id
    if case.expect_write:
        assert dest.read_bytes() == case.service_bytes
    else:
        client.get_metadata.assert_called_once_with(case.metadata.file_id)


def test_download_to_path_cleans_up_temp_file_when_download_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = DriveClient(retries=0, retry_base=0.0)
    client.get_metadata = MagicMock(return_value=DriveFile(file_id="file-1", name="payload.json", mime_type="application/json", modified_time=None, size_bytes=5))
    client._service = MockDriveService(file_content={"file-1": b"12345"})

    class _BrokenDownloader(MockMediaIoBaseDownload):
        def next_chunk(self):
            raise OSError("download blew up")

    monkeypatch.setattr("polylogue.sources.drive_client._import_module", lambda name: SimpleNamespace(MediaIoBaseDownload=_BrokenDownloader) if name == "googleapiclient.http" else (_ for _ in ()).throw(AssertionError(name)))
    dest = tmp_path / "payload.json"

    with pytest.raises(OSError, match="download blew up"):
        client.download_to_path("file-1", dest)

    assert not list(tmp_path.glob("tmp*"))
    assert not dest.exists()


def test_download_bytes_and_json_payload_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    client = DriveClient(retries=0, retry_base=0.0)
    client._service = MockDriveService(file_content={"file-1": b'{"hello":"world"}', "file-2": b'{"a":1}\n{"b":2}\n'})
    monkeypatch.setattr("polylogue.sources.drive_client._import_module", lambda name: SimpleNamespace(MediaIoBaseDownload=MockMediaIoBaseDownload) if name == "googleapiclient.http" else (_ for _ in ()).throw(AssertionError(name)))

    assert client.download_bytes("file-1") == b'{"hello":"world"}'
    assert client.download_json_payload("file-1", name="payload.json") == {"hello": "world"}
    assert client.download_json_payload("file-2", name="payload.jsonl") == [{"a": 1}, {"b": 2}]


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
    import os
    os.utime(dest, (1735689600.0, 1735689600.0))
    base = DriveFile(file_id="file-1", name="payload.json", mime_type="application/json", modified_time="2025-01-01T00:00:00Z", size_bytes=5)

    assert _needs_download(base, dest) is False
    assert _needs_download(replace(base, size_bytes=6), dest) is True
    assert _needs_download(replace(base, modified_time="2025-01-01T00:00:05Z"), dest) is True
    assert _needs_download(base, tmp_path / "missing.json") is True


def test_get_metadata_and_iteration_cache_contract() -> None:
    client = DriveClient(retries=0, retry_base=0.0)
    service = MockDriveService(
        files_data={"file-1": mock_drive_file(file_id="file-1", name="", mime_type="application/json", size=0)}
    )
    service._files_resource.get = MagicMock(wraps=service._files_resource.get)
    service._files_resource.list = MagicMock(
        side_effect=[
            SimpleNamespace(execute=lambda: {"nextPageToken": "page-2", "files": [{"id": "f1", "name": "one.json", "mimeType": "application/json", "modifiedTime": None, "size": "1"}, {"id": "skip", "name": "readme.txt", "mimeType": "text/plain", "modifiedTime": None, "size": "1"}]}),
            SimpleNamespace(execute=lambda: {"files": [{"id": "f2", "name": "two.ndjson", "mimeType": "application/octet-stream", "modifiedTime": None, "size": "2"}, {"id": "f3", "name": "prompt.bin", "mimeType": GEMINI_PROMPT_MIME_TYPE, "modifiedTime": None, "size": "3"}]}),
        ]
    )
    service._http = None
    client._service = service

    first = client.get_metadata("file-1")
    second = client.get_metadata("file-1")
    files = list(client.iter_json_files("folder-1"))

    assert first.name == "file-1"
    assert second is first
    assert [file.file_id for file in files] == ["f1", "f2", "f3"]
    assert list(client._meta_cache) == ["file-1", "f1", "f2", "f3"]

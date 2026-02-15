"""Drive resilience tests — backoff, retries, error types, client init, token refresh, failure tracking."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import polylogue.sources.drive_client as drive_client
from polylogue.config import Source
from polylogue.errors import PolylogueError
from polylogue.sources import (
    DriveAuthError,
    DriveClient,
    DriveError,
    DriveFile,
    DriveNotFoundError,
    iter_drive_conversations,
)
from polylogue.sources.drive_client import _is_retryable_error

# ============================================================================
# Test Data Tables (Module-level constants for parametrization)
# ============================================================================

BACKOFF_TEST_CASES = [
    ((0.5, 0.5, 10), [1, 2, 3], "increases_exponentially"),
    ((1, 1, 5), [10], "respects_max"),
]

EXCEPTION_TYPE_CASES = [
    (DriveAuthError, "Auth failed", [DriveError, PolylogueError], "auth_error_is_drive_error"),
    (DriveNotFoundError, "File not found", [DriveError, PolylogueError], "not_found_error_is_drive_error"),
    (DriveError, "Test message", [PolylogueError], "drive_error_is_polylogue_error"),
]

NETWORK_ERROR_CASES = [
    (ConnectionRefusedError("Connection refused"), True, "connection_refused"),
    (ConnectionResetError("Connection reset by peer"), True, "connection_reset"),
    (TimeoutError("Operation timed out"), True, "timeout"),
    (BrokenPipeError("Broken pipe"), True, "broken_pipe"),
]

RETRY_SCENARIOS = [
    (429, True, "rate_limit_retries"),
    (500, True, "server_error_retries"),
    (502, True, "bad_gateway_retries"),
    (503, True, "service_unavailable_retries"),
    (504, True, "gateway_timeout_retries"),
]

NO_RETRY_SCENARIOS = [
    (401, False, "auth_error_no_retry"),
    (403, False, "forbidden_no_retry"),
    (404, False, "not_found_no_retry"),
]


# ============================================================================
# Tests for Exponential Backoff Behavior
# ============================================================================


class TestExponentialBackoff:
    """Tests for exponential backoff behavior."""

    @pytest.mark.parametrize("config,attempts,test_id", BACKOFF_TEST_CASES)
    def test_backoff_behavior(self, config, attempts, test_id):
        """Test exponential backoff with various configurations (parametrized)."""
        from tenacity import wait_exponential

        multiplier, min_wait, max_wait = config
        wait = wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait)

        class MockRetryState:
            def __init__(self, attempt):
                self.attempt_number = attempt

        if test_id == "increases_exponentially":
            waits = [wait(MockRetryState(a)) for a in attempts]
            assert waits[1] >= waits[0]
            assert waits[2] >= waits[1]
        elif test_id == "respects_max":
            wait_val = wait(MockRetryState(attempts[0]))
            assert wait_val <= max_wait


# ============================================================================
# Parametrized Tests for HTTP Error Retry Classification
# ============================================================================


class TestHttpErrorRetries:
    """Tests for HTTP error handling and retry classification."""

    @pytest.mark.parametrize("status_code,should_retry,description", RETRY_SCENARIOS)
    def test_retryable_http_errors(self, status_code: int, should_retry: bool, description: str):
        """HTTP errors that should trigger retry."""
        class MockHTTPError(Exception):
            def __init__(self, status):
                self.status = status
                super().__init__(f"HTTP {status}")

        exc = MockHTTPError(status_code)
        assert _is_retryable_error(exc) is should_retry


# ============================================================================
# Tests for Drive Exception Types
# ============================================================================


class TestDriveExceptionTypes:
    """Tests for Drive exception hierarchy."""

    @pytest.mark.parametrize("exc_class,message,base_classes,test_id", EXCEPTION_TYPE_CASES)
    def test_exception_hierarchy(self, exc_class, message, base_classes, test_id):
        """Test Drive exception hierarchy (parametrized)."""
        exc = exc_class(message)
        assert str(exc) == message
        for base_class in base_classes:
            assert isinstance(exc, base_class), f"Failed for {test_id}"

    def test_auth_error_chaining(self):
        """DriveAuthError can chain from other exceptions."""
        original = ValueError("Original error")
        exc = DriveAuthError("Wrapped error")
        exc.__cause__ = original

        assert exc.__cause__ is original


# ============================================================================
# Parametrized Tests for Network Error Handling
# ============================================================================


class TestNetworkErrors:
    """Tests for network error handling and retry classification."""

    @pytest.mark.parametrize("exc,expected,desc", NETWORK_ERROR_CASES)
    def test_network_errors(self, exc, expected, desc):
        """Test network error retry classification."""
        assert _is_retryable_error(exc) is expected, f"Failed for {desc}"


# ============================================================================
# Tests for Client Initialization and Mocked Operations
# ============================================================================


class TestDriveClientInit:
    """Tests for DriveClient initialization."""

    def test_init_with_mock_credentials(self, mock_drive_credentials):
        """Initialize Drive client with mock credentials."""
        client = DriveClient(
            credentials_path=mock_drive_credentials["credentials_path"],
            token_path=mock_drive_credentials["token_path"],
        )
        assert client is not None

    def test_init_with_retries(self, mock_drive_credentials):
        """Initialize with custom retry settings."""
        client = DriveClient(
            credentials_path=mock_drive_credentials["credentials_path"],
            token_path=mock_drive_credentials["token_path"],
            retries=5,
            retry_base=1.0,
        )
        assert client._retries == 5
        assert client._retry_base == 1.0


class TestDriveClientMocked:
    """Tests for Drive client operations with mocked service."""

    @pytest.fixture
    def drive_client(self, mock_drive_credentials, mock_drive_service):
        """Drive client with mocked service."""
        client = DriveClient(
            credentials_path=mock_drive_credentials["credentials_path"],
            token_path=mock_drive_credentials["token_path"],
        )
        client._service = mock_drive_service["service"]
        return client

    def test_list_files_in_folder(self, drive_client, mock_drive_service):
        """List files in a specific folder."""
        service = mock_drive_service["service"]

        response = service.files().list(q="'folder1' in parents").execute()
        assert "files" in response
        assert len(response["files"]) == 1
        assert response["files"][0]["name"] == "Test Prompt"

    def test_get_file_metadata(self, drive_client, mock_drive_service):
        """Get metadata for a specific file."""
        service = mock_drive_service["service"]
        response = service.files().get(fileId="prompt1").execute()
        assert response["name"] == "Test Prompt"
        assert response["mimeType"] == "application/vnd.google-makersuite.prompt"

    def test_get_file_content(self, drive_client, mock_drive_service):
        """Download file content."""
        service = mock_drive_service["service"]
        content = service.files().get_media(fileId="prompt1").execute()
        assert b"Test Prompt" in content

    def test_file_not_found_raises(self, drive_client, mock_drive_service):
        """Raise error when file not found."""
        service = mock_drive_service["service"]
        with pytest.raises(Exception, match="File not found"):
            service.files().get(fileId="nonexistent").execute()


class TestRetryLogic:
    """Tests for retry logic and configuration."""

    def test_retry_decorator_exists(self, mock_drive_credentials):
        """Verify retry logic is configured."""
        client = DriveClient(
            credentials_path=mock_drive_credentials["credentials_path"],
            token_path=mock_drive_credentials["token_path"],
            retries=3,
        )
        assert hasattr(client, "_call_with_retry")
        assert client._retries == 3

    def test_auth_error_not_retried(self):
        """Auth errors should not be retried."""
        auth_error = DriveAuthError("Invalid credentials")
        assert not _is_retryable_error(auth_error)

    def test_generic_error_is_retryable(self):
        """Generic errors should be retried."""
        generic_error = RuntimeError("Network timeout")
        assert _is_retryable_error(generic_error)


class TestDriveEdgeCases:
    """Tests for edge cases and pagination support."""

    def test_pagination_support(self, mock_drive_service):
        """Mock service supports pagination."""
        service = mock_drive_service["service"]

        from tests.infra.drive_mocks import mock_drive_file

        many_files = {f"file{i}": mock_drive_file(file_id=f"file{i}", name=f"File {i}") for i in range(150)}
        service._files_resource.files.update(many_files)

        response = service.files().list(pageSize=50).execute()
        assert len(response["files"]) <= 50

        if len(many_files) > 50:
            assert response.get("nextPageToken") is not None


class TestAPIOperations:
    """Tests for public API methods (resolve_folder_id, iter_json_files, etc.)."""

    @pytest.fixture
    def drive_client_with_service(self, mock_drive_credentials, mock_drive_service):
        """Drive client with mocked service."""
        client = DriveClient(
            credentials_path=mock_drive_credentials["credentials_path"],
            token_path=mock_drive_credentials["token_path"],
        )
        client._service = mock_drive_service["service"]
        return client, mock_drive_service

    def test_resolve_folder_id_by_name(self, drive_client_with_service):
        """Resolve folder ID by name."""
        client, mock_service = drive_client_with_service

        folder_id = client.resolve_folder_id("Google AI Studio")
        assert folder_id == "folder1"

    def test_resolve_folder_id_not_found(self, drive_client_with_service):
        """Raise error when folder not found."""
        client, mock_service = drive_client_with_service

        with pytest.raises(DriveNotFoundError, match="Folder.*not found"):
            client.resolve_folder_id("NonexistentFolder")

    def test_iter_json_files(self, drive_client_with_service):
        """Iterate through JSON files in a folder."""
        client, mock_service = drive_client_with_service

        from tests.infra.drive_mocks import mock_drive_file

        json_file1 = mock_drive_file(
            file_id="json1",
            name="test.json",
            mime_type="application/json",
            parents=["folder1"],
        )
        json_file2 = mock_drive_file(
            file_id="json2",
            name="data.json",
            mime_type="application/json",
            parents=["folder1"],
        )

        mock_service["service"]._files_resource.files.update({
            "json1": json_file1,
            "json2": json_file2,
        })
        mock_service["file_content"]["json1"] = b'{"test": "content"}'
        mock_service["file_content"]["json2"] = b'{"data": "content"}'

        files = list(client.iter_json_files("folder1"))
        assert len(files) >= 1
        file_names = [f.name for f in files]
        assert "test.json" in file_names or "data.json" in file_names

    def test_get_metadata(self, drive_client_with_service):
        """Get file metadata."""
        client, mock_service = drive_client_with_service

        metadata = client.get_metadata("prompt1")
        assert metadata.name == "Test Prompt"
        assert metadata.mime_type == "application/vnd.google-makersuite.prompt"

    def test_download_bytes(self, drive_client_with_service, mock_media_downloader):
        """Download file as bytes."""
        client, mock_service = drive_client_with_service

        content = client.download_bytes("prompt1")
        assert isinstance(content, bytes)
        assert b"Test Prompt" in content

    def test_download_json_payload(self, drive_client_with_service, mock_media_downloader):
        """Download and parse JSON payload."""
        client, mock_service = drive_client_with_service

        payload = client.download_json_payload("prompt1", name="prompt1.json")
        assert isinstance(payload, dict)
        assert payload["title"] == "Test Prompt"
        assert payload["content"] == "Test content"

    def test_download_to_path(self, drive_client_with_service, mock_media_downloader, tmp_path):
        """Download file to local path."""
        client, mock_service = drive_client_with_service

        output_path = tmp_path / "downloaded.json"
        result = client.download_to_path("prompt1", output_path)

        assert result.file_id == "prompt1"
        assert output_path.exists()
        assert b"Test Prompt" in output_path.read_bytes()

    def test_download_with_encoding_fallback(self, drive_client_with_service, mock_media_downloader):
        """Download should handle encoding issues gracefully."""
        client, mock_service = drive_client_with_service

        from tests.infra.drive_mocks import mock_drive_file

        binary_file = mock_drive_file(
            file_id="binary1",
            name="binary.dat",
            mime_type="application/octet-stream",
        )
        mock_service["service"]._files_resource.files["binary1"] = binary_file
        mock_service["file_content"]["binary1"] = b"\xff\xfe Invalid UTF-8"

        content = client.download_bytes("binary1")
        assert isinstance(content, bytes)
        assert content == b"\xff\xfe Invalid UTF-8"

    def test_resolve_folder_id_with_multiple_matches(self, drive_client_with_service):
        """Handle multiple folders with same name."""
        client, mock_service = drive_client_with_service

        from tests.infra.drive_mocks import mock_drive_file

        folder2 = mock_drive_file(
            file_id="folder2",
            name="Google AI Studio",
            mime_type="application/vnd.google-apps.folder",
        )
        mock_service["service"]._files_resource.files["folder2"] = folder2

        folder_id = client.resolve_folder_id("Google AI Studio")
        assert folder_id in ["folder1", "folder2"]

    def test_iter_json_files_empty_folder(self, drive_client_with_service):
        """Iterate through empty folder."""
        client, mock_service = drive_client_with_service

        from tests.infra.drive_mocks import mock_drive_file

        empty_folder = mock_drive_file(
            file_id="empty",
            name="Empty Folder",
            mime_type="application/vnd.google-apps.folder",
        )
        mock_service["service"]._files_resource.files["empty"] = empty_folder

        files = list(client.iter_json_files("empty"))
        assert len(files) == 0


# ============================================================================
# Tests for Token Refresh and Error Handling
# ============================================================================


class TestTokenRefreshErrorHandling:
    """Tests for OAuth token refresh error handling."""

    def test_missing_dependency_raises_auth_error(self, monkeypatch):
        """Missing googleapiclient dependency should raise DriveAuthError."""
        real_import = importlib.import_module

        def fake_import(name: str):
            if name.startswith("googleapiclient"):
                raise ModuleNotFoundError(name)
            return real_import(name)

        client = DriveClient(ui=None)
        monkeypatch.setattr(client, "_load_credentials", lambda: object())
        monkeypatch.setattr(drive_client.importlib, "import_module", fake_import)

        with pytest.raises(DriveAuthError, match="Drive dependencies"):
            client._service_handle()

    def test_refresh_failure_raises_specific_error(self, monkeypatch):
        """Token refresh failure should raise DriveAuthError."""
        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "valid_refresh_token"
        mock_creds.refresh.side_effect = Exception("Network error during refresh")

        mock_request = MagicMock()

        def mock_import(name: str):
            if name == "google.auth.transport.requests":
                return MagicMock(Request=MagicMock(return_value=mock_request))
            if name == "google.oauth2.credentials":
                return MagicMock(Credentials=MagicMock(from_authorized_user_file=MagicMock(return_value=mock_creds)))
            return importlib.import_module(name)

        monkeypatch.setattr(drive_client, "_import_module", mock_import)

        client = DriveClient(ui=None)
        with pytest.raises(DriveAuthError, match="refresh|token"):
            client._load_credentials()

    def test_refresh_failure_includes_original_error(self, monkeypatch, tmp_path):
        """DriveAuthError should include the original exception details."""
        token_file = tmp_path / "token.json"
        token_file.write_text('{"valid": false, "expired": true}')

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "valid_refresh_token"
        mock_creds.refresh.side_effect = ConnectionError("Server unreachable")

        mock_request = MagicMock()

        def mock_import(name: str):
            if name == "google.auth.transport.requests":
                return MagicMock(Request=MagicMock(return_value=mock_request))
            if name == "google.oauth2.credentials":
                mock_cls = MagicMock()
                mock_cls.from_authorized_user_file = MagicMock(return_value=mock_creds)
                return MagicMock(Credentials=mock_cls)
            return importlib.import_module(name)

        monkeypatch.setattr(drive_client, "_import_module", mock_import)

        client = DriveClient(ui=None, token_path=token_file)
        with pytest.raises(DriveAuthError) as exc_info:
            client._load_credentials()

        error_str = str(exc_info.value)
        assert "Server unreachable" in error_str or "refresh" in error_str.lower()

    def test_invalid_credentials_raises_auth_error(self, monkeypatch, tmp_path):
        """Invalid credentials without refresh capability should raise clear auth error."""
        token_file = tmp_path / "token.json"
        token_file.write_text('{"valid": false}')

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = None

        def mock_import(name: str):
            if name == "google.oauth2.credentials":
                mock_cls = MagicMock()
                mock_cls.from_authorized_user_file = MagicMock(return_value=mock_creds)
                return MagicMock(Credentials=mock_cls)
            return importlib.import_module(name)

        monkeypatch.setattr(drive_client, "_import_module", mock_import)

        client = DriveClient(ui=None, token_path=token_file)
        with pytest.raises(DriveAuthError, match="invalid|expired|re-run"):
            client._load_credentials()

    def test_successful_token_refresh(self, monkeypatch, tmp_path):
        """Successful token refresh should return valid credentials."""
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "old_token"}')

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "valid_refresh_token"
        mock_creds.to_json.return_value = '{"token": "new_token", "refresh_token": "valid_refresh_token"}'

        def mock_refresh(request):
            mock_creds.valid = True
            mock_creds.expired = False
            mock_creds.token = "new_token"

        mock_creds.refresh = mock_refresh
        mock_request = MagicMock()

        def mock_import(name: str):
            if name == "google.auth.transport.requests":
                return MagicMock(Request=MagicMock(return_value=mock_request))
            if name == "google.oauth2.credentials":
                mock_cls = MagicMock()
                mock_cls.from_authorized_user_file = MagicMock(return_value=mock_creds)
                return MagicMock(Credentials=mock_cls)
            return importlib.import_module(name)

        monkeypatch.setattr(drive_client, "_import_module", mock_import)

        client = DriveClient(ui=None, token_path=token_file)
        result = client._load_credentials()

        assert result is not None
        assert result.valid is True
        assert result.token == "new_token"

    def test_valid_cached_credentials(self, monkeypatch, tmp_path):
        """Valid cached credentials should be returned without refresh."""
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "current_token"}')

        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.expired = False
        mock_creds.refresh_token = "refresh_token"
        mock_creds.token = "current_token"
        mock_creds.to_json.return_value = '{"token": "current_token", "refresh_token": "refresh_token"}'

        def mock_import(name: str):
            if name == "google.oauth2.credentials":
                mock_cls = MagicMock()
                mock_cls.from_authorized_user_file = MagicMock(return_value=mock_creds)
                return MagicMock(Credentials=mock_cls)
            return importlib.import_module(name)

        monkeypatch.setattr(drive_client, "_import_module", mock_import)

        client = DriveClient(ui=None, token_path=token_file)
        result = client._load_credentials()

        assert result is not None
        assert result.valid is True
        assert result.token == "current_token"
        assert not mock_creds.refresh.called

    def test_corrupt_token_file_handling(self, monkeypatch, tmp_path):
        """Corrupt token file should be handled gracefully."""
        token_file = tmp_path / "token.json"
        token_file.write_text('invalid json {{{')

        def mock_import(name: str):
            if name == "google.oauth2.credentials":
                mock_cls = MagicMock()
                mock_cls.from_authorized_user_file = MagicMock(side_effect=ValueError("Invalid JSON"))
                return MagicMock(Credentials=mock_cls)
            return importlib.import_module(name)

        monkeypatch.setattr(drive_client, "_import_module", mock_import)

        client = DriveClient(ui=None, token_path=token_file)

        with pytest.raises(DriveAuthError, match="invalid|expired"):
            client._load_credentials()


# ============================================================================
# Tests for Drive Ingestion
# ============================================================================


@dataclass
class StubDriveClient:
    payload: dict

    def resolve_folder_id(self, folder_ref: str) -> str:
        return "folder-1"

    def iter_json_files(self, folder_id: str):
        yield DriveFile(
            file_id="file-1",
            name="chat.json",
            mime_type="application/json",
            modified_time=None,
            size_bytes=None,
        )

    def download_json_payload(self, file_id: str, *, name: str):
        return self.payload

    def download_to_path(self, file_id, dest):
        raise AssertionError("download_to_path should not be called when download_assets=False")


def test_drive_ingest_chunked_prompt_no_download(tmp_path):
    payload = {
        "title": "Drive Chat",
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "Hi"},
                {
                    "role": "model",
                    "text": "Hello",
                    "driveDocument": {"id": "att-1", "name": "doc.txt"},
                },
            ]
        },
    }
    source = Source(name="gemini", folder="Google AI Studio")
    client = StubDriveClient(payload=payload)

    conversations = list(
        iter_drive_conversations(
            source=source,
            archive_root=tmp_path,
            client=client,
            download_assets=False,
        )
    )
    assert len(conversations) == 1
    convo = conversations[0]
    assert [msg.role for msg in convo.messages] == ["user", "assistant"]
    assert len(convo.attachments) == 1
    attachment = convo.attachments[0]
    assert attachment.provider_attachment_id == "att-1"
    assert attachment.message_provider_id == "chunk-2"
    assert attachment.path is None


class TestDriveDownloadFailureTracking:
    """Tests for tracking Drive download failures."""

    def test_download_failure_tracked_in_result(self):
        """Failed downloads should be tracked in the result."""
        from polylogue.sources import download_drive_files
        from polylogue.sources.drive_client import DriveFile

        mock_client = MagicMock()
        mock_client.iter_json_files.return_value = [
            DriveFile(file_id="file1", name="good.json", mime_type="application/json", modified_time=None, size_bytes=100),
            DriveFile(file_id="file2", name="bad.json", mime_type="application/json", modified_time=None, size_bytes=100),
            DriveFile(file_id="file3", name="also_good.json", mime_type="application/json", modified_time=None, size_bytes=100),
        ]

        def mock_download(file_id, dest):
            if file_id == "file2":
                raise OSError("Download failed")
            dest.write_text('{"test": true}')

        mock_client.download_to_path.side_effect = mock_download

        result = download_drive_files(mock_client, "folder123", Path("/tmp/test"))

        assert hasattr(result, "failed_files") or "failed" in result
        assert len(result.failed_files) >= 1
        assert any("bad.json" in str(f) for f in result.failed_files)

    def test_download_continues_after_single_failure(self):
        """Download should continue processing other files after one fails."""
        from polylogue.sources import download_drive_files
        from polylogue.sources.drive_client import DriveFile

        mock_client = MagicMock()
        mock_client.iter_json_files.return_value = [
            DriveFile(file_id="f1", name="first.json", mime_type="application/json", modified_time=None, size_bytes=100),
            DriveFile(file_id="f2", name="fails.json", mime_type="application/json", modified_time=None, size_bytes=100),
            DriveFile(file_id="f3", name="third.json", mime_type="application/json", modified_time=None, size_bytes=100),
        ]

        download_count = [0]

        def mock_download(file_id, dest):
            if file_id == "f2":
                raise OSError("Failed")
            download_count[0] += 1
            dest.write_text('{}')

        mock_client.download_to_path.side_effect = mock_download

        download_drive_files(mock_client, "folder", Path("/tmp/test"))

        assert download_count[0] == 2

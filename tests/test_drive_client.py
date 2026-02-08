from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest

import polylogue.sources.drive_client as drive_client
from polylogue.sources import DriveAuthError, DriveClient


def test_drive_client_reports_missing_dependency(monkeypatch):
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


class TestTokenRefreshErrorHandling:
    """Tests for OAuth token refresh error handling."""

    def test_refresh_failure_raises_specific_error(self, monkeypatch):
        """Token refresh failure should raise DriveAuthError, not be swallowed.

        This test verifies that real refresh errors (network, auth failures) are
        properly exposed to the user instead of being silently logged.
        """
        # Mock credentials that fail to refresh
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
        # Create a temporary token file
        token_file = tmp_path / "token.json"
        token_file.write_text('{"valid": false, "expired": true}')

        # Mock credentials that fail to refresh
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

        # Original error should be preserved in exception chain or message
        error_str = str(exc_info.value)
        assert "Server unreachable" in error_str or "refresh" in error_str.lower()

    def test_invalid_credentials_raises_auth_error(self, monkeypatch, tmp_path):
        """Invalid credentials without refresh capability should raise clear auth error."""
        # Create a temporary token file
        token_file = tmp_path / "token.json"
        token_file.write_text('{"valid": false}')

        # Mock invalid credentials (expired but no refresh token)
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
        # Create token file
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "old_token"}')

        # Mock expired credentials that successfully refresh
        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "valid_refresh_token"
        mock_creds.to_json.return_value = '{"token": "new_token", "refresh_token": "valid_refresh_token"}'

        # After refresh, credentials become valid
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
        # Create token file
        token_file = tmp_path / "token.json"
        token_file.write_text('{"token": "current_token"}')

        # Mock valid credentials that don't need refresh
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
        # Verify refresh was never called
        assert not mock_creds.refresh.called

    def test_corrupt_token_file_handling(self, monkeypatch, tmp_path):
        """Corrupt token file should be handled gracefully."""
        # Create corrupt token file
        token_file = tmp_path / "token.json"
        token_file.write_text('invalid json {{{')

        # Mock that from_authorized_user_file raises ValueError
        def mock_import(name: str):
            if name == "google.oauth2.credentials":
                mock_cls = MagicMock()
                mock_cls.from_authorized_user_file = MagicMock(side_effect=ValueError("Invalid JSON"))
                return MagicMock(Credentials=mock_cls)
            return importlib.import_module(name)

        monkeypatch.setattr(drive_client, "_import_module", mock_import)

        client = DriveClient(ui=None, token_path=token_file)

        # Should raise DriveAuthError about invalid token
        with pytest.raises(DriveAuthError, match="invalid|expired"):
            client._load_credentials()


class TestCredentialsResolution:
    """Tests for credential path resolution."""

    def test_default_credentials_path(self):
        """Get default credentials path."""
        from polylogue.sources.drive_client import default_credentials_path

        path = default_credentials_path()
        assert "credentials.json" in str(path)

    def test_default_token_path(self):
        """Get default token path."""
        from polylogue.sources.drive_client import default_token_path

        path = default_token_path()
        assert "token.json" in str(path)

    def test_resolve_credentials_from_env(self, tmp_path, monkeypatch):
        """Resolve credentials path from environment variable."""
        from polylogue.sources.drive_client import _resolve_credentials_path

        creds_path = tmp_path / "my_creds.json"
        creds_path.write_text('{"installed": {}}')
        monkeypatch.setenv("POLYLOGUE_CREDENTIAL_PATH", str(creds_path))

        result = _resolve_credentials_path(ui=None, config=None)
        assert result == creds_path

    def test_resolve_credentials_missing_raises(self, monkeypatch, tmp_path):
        """Raise error when credentials not found."""
        from polylogue.sources.drive_client import _resolve_credentials_path

        # Ensure no env var is set
        monkeypatch.delenv("POLYLOGUE_CREDENTIAL_PATH", raising=False)

        # Mock default path to non-existent location
        nonexistent_path = tmp_path / "nonexistent" / "credentials.json"
        monkeypatch.setattr(
            "polylogue.sources.drive_client.default_credentials_path",
            lambda config: nonexistent_path
        )

        with pytest.raises(DriveAuthError, match="credentials not found"):
            _resolve_credentials_path(ui=None, config=None)

    def test_resolve_token_from_env(self, tmp_path, monkeypatch):
        """Resolve token path from environment variable."""
        from polylogue.sources.drive_client import _resolve_token_path

        token_path = tmp_path / "my_token.json"
        monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", str(token_path))

        result = _resolve_token_path(config=None)
        assert result == token_path

    def test_resolve_retries_from_env(self, monkeypatch):
        """Resolve retry count from environment."""
        from polylogue.sources.drive_client import _resolve_retries

        monkeypatch.setenv("POLYLOGUE_DRIVE_RETRIES", "5")
        result = _resolve_retries(value=None, config=None)
        assert result == 5

    def test_resolve_retries_default(self):
        """Use default retry count when not specified."""
        from polylogue.sources.drive_client import _resolve_retries

        result = _resolve_retries(value=None, config=None)
        assert result >= 0  # Should be a reasonable default

    def test_resolve_retries_explicit_value(self):
        """Use explicit retry value when provided."""
        from polylogue.sources.drive_client import _resolve_retries

        result = _resolve_retries(value=10, config=None)
        assert result == 10


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
        # Inject mock service (bypass actual OAuth)
        client._service = mock_drive_service["service"]
        return client

    def test_list_files_in_folder(self, drive_client, mock_drive_service):
        """List files in a specific folder."""
        # Mock service has "folder1" with "prompt1" inside
        service = mock_drive_service["service"]

        # List files in folder1
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
    """Tests for retry logic."""

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
        from polylogue.sources.drive_client import _is_retryable_error

        auth_error = DriveAuthError("Invalid credentials")
        assert not _is_retryable_error(auth_error)

    def test_generic_error_is_retryable(self):
        """Generic errors should be retried."""
        from polylogue.sources.drive_client import _is_retryable_error

        generic_error = RuntimeError("Network timeout")
        assert _is_retryable_error(generic_error)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_pagination_support(self, mock_drive_service):
        """Mock service supports pagination."""
        service = mock_drive_service["service"]

        # Add enough files to trigger pagination
        from tests.mocks.drive_mocks import mock_drive_file

        many_files = {f"file{i}": mock_drive_file(file_id=f"file{i}", name=f"File {i}") for i in range(150)}
        service._files_resource.files.update(many_files)

        # Request with small page size
        response = service.files().list(pageSize=50).execute()
        assert len(response["files"]) <= 50

        # Check if next page token exists when there are more files
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
        # Inject mock service
        client._service = mock_drive_service["service"]
        return client, mock_drive_service

    def test_resolve_folder_id_by_name(self, drive_client_with_service):
        """Resolve folder ID by name."""
        client, mock_service = drive_client_with_service

        # Mock service already has "folder1" with name "Google AI Studio"
        folder_id = client.resolve_folder_id("Google AI Studio")
        assert folder_id == "folder1"

    def test_resolve_folder_id_not_found(self, drive_client_with_service):
        """Raise error when folder not found."""
        from polylogue.sources import DriveNotFoundError

        client, mock_service = drive_client_with_service

        with pytest.raises(DriveNotFoundError, match="Folder.*not found"):
            client.resolve_folder_id("NonexistentFolder")

    def test_iter_json_files(self, drive_client_with_service):
        """Iterate through JSON files in a folder."""
        client, mock_service = drive_client_with_service

        # Add JSON files to mock service
        from tests.mocks.drive_mocks import mock_drive_file

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
        # Add file content for the JSON files
        mock_service["file_content"]["json1"] = b'{"test": "content"}'
        mock_service["file_content"]["json2"] = b'{"data": "content"}'

        # Iterate through JSON files - returns DriveFile objects
        files = list(client.iter_json_files("folder1"))
        assert len(files) >= 1  # At least some JSON files from fixture
        # DriveFile objects use attribute access, not dict access
        file_names = [f.name for f in files]
        assert "test.json" in file_names or "data.json" in file_names

    def test_get_metadata(self, drive_client_with_service):
        """Get file metadata."""
        client, mock_service = drive_client_with_service

        # get_metadata returns DriveFile object, not dict
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

        # Result is DriveFile metadata, not path
        assert result.file_id == "prompt1"
        assert output_path.exists()
        assert b"Test Prompt" in output_path.read_bytes()

    def test_download_with_encoding_fallback(self, drive_client_with_service, mock_media_downloader):
        """Download should handle encoding issues gracefully."""
        client, mock_service = drive_client_with_service

        # Add file with non-UTF8 content
        from tests.mocks.drive_mocks import mock_drive_file

        binary_file = mock_drive_file(
            file_id="binary1",
            name="binary.dat",
            mime_type="application/octet-stream",
        )
        mock_service["service"]._files_resource.files["binary1"] = binary_file
        mock_service["file_content"]["binary1"] = b"\xff\xfe Invalid UTF-8"

        # Should download as bytes without error
        content = client.download_bytes("binary1")
        assert isinstance(content, bytes)
        assert content == b"\xff\xfe Invalid UTF-8"

    def test_resolve_folder_id_with_multiple_matches(self, drive_client_with_service):
        """Handle multiple folders with same name."""
        client, mock_service = drive_client_with_service

        # Add duplicate folder names
        from tests.mocks.drive_mocks import mock_drive_file

        folder2 = mock_drive_file(
            file_id="folder2",
            name="Google AI Studio",
            mime_type="application/vnd.google-apps.folder",
        )
        mock_service["service"]._files_resource.files["folder2"] = folder2

        # Should return first match
        folder_id = client.resolve_folder_id("Google AI Studio")
        assert folder_id in ["folder1", "folder2"]

    def test_iter_json_files_empty_folder(self, drive_client_with_service):
        """Iterate through empty folder."""
        client, mock_service = drive_client_with_service

        # Add empty folder
        from tests.mocks.drive_mocks import mock_drive_file

        empty_folder = mock_drive_file(
            file_id="empty",
            name="Empty Folder",
            mime_type="application/vnd.google-apps.folder",
        )
        mock_service["service"]._files_resource.files["empty"] = empty_folder

        # Should return empty iterator
        files = list(client.iter_json_files("empty"))
        assert len(files) == 0

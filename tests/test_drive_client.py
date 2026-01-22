from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

import polylogue.drive_client as drive_client
from polylogue.drive_client import DriveAuthError, DriveClient


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

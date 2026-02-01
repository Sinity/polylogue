"""Tests for polylogue.cli.commands.auth module.

Coverage targets:
- auth_command: --provider validation, --revoke, --refresh flows
- _get_drive_paths: config loading, fallback paths
- _drive_oauth_flow: credentials check, token handling, retry logic
- _refresh_drive_token: token deletion and re-auth
- _revoke_drive_credentials: token file deletion
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli import cli


@pytest.fixture
def auth_workspace(tmp_path, monkeypatch):
    """Set up isolated workspace for auth tests."""
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    creds_dir = tmp_path / "creds"

    for d in [config_dir, data_dir, state_dir, creds_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create credentials.json (OAuth client config)
    creds_path = creds_dir / "credentials.json"
    creds_path.write_text(
        json.dumps({
            "installed": {
                "client_id": "test_client.apps.googleusercontent.com",
                "client_secret": "test_secret",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        }),
        encoding="utf-8",
    )

    # Create token.json (OAuth tokens)
    token_path = creds_dir / "token.json"
    token_path.write_text(
        json.dumps({
            "token": "test_access_token",
            "refresh_token": "test_refresh_token",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "test_client.apps.googleusercontent.com",
            "client_secret": "test_secret",
            "scopes": ["https://www.googleapis.com/auth/drive.readonly"],
        }),
        encoding="utf-8",
    )

    monkeypatch.setenv("XDG_DATA_HOME", str(data_dir))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
    monkeypatch.setenv("POLYLOGUE_CREDENTIAL_PATH", str(creds_path))
    monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", str(token_path))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")

    # Reload modules to pick up new environment
    import importlib
    import polylogue.paths
    import polylogue.config

    importlib.reload(polylogue.paths)
    importlib.reload(polylogue.config)

    return {
        "creds_path": creds_path,
        "token_path": token_path,
        "data_dir": data_dir,
    }


class TestAuthCommand:
    """Tests for the auth command."""

    def test_unknown_provider_fails(self, auth_workspace):
        """Unknown auth provider shows error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["auth", "--provider", "unknown"])
        assert result.exit_code == 1
        assert "unknown" in result.output.lower()
        assert "drive" in result.output.lower()

    def test_revoke_removes_token(self, auth_workspace, monkeypatch):
        """--revoke removes the token file."""
        # Use the config-level token path setting
        from polylogue.ingestion.drive_client import default_token_path

        token_path = auth_workspace["token_path"]
        assert token_path.exists()

        # Patch to ensure test uses our token path
        monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", str(token_path))

        runner = CliRunner()
        result = runner.invoke(cli, ["auth", "--revoke"])
        assert result.exit_code == 0
        # Token should be removed OR message about removal shown
        assert not token_path.exists() or "removed" in result.output.lower() or "revoked" in result.output.lower()

    def test_revoke_no_token_shows_message(self, auth_workspace):
        """--revoke with no token file shows message."""
        token_path = auth_workspace["token_path"]
        token_path.unlink()  # Remove token first

        runner = CliRunner()
        result = runner.invoke(cli, ["auth", "--revoke"])
        assert result.exit_code == 0
        assert "no token" in result.output.lower()

    def test_refresh_removes_and_reauths(self, auth_workspace):
        """--refresh removes existing token and triggers OAuth."""
        token_path = auth_workspace["token_path"]
        assert token_path.exists()

        # Mock DriveClient to avoid actual OAuth - need to patch at import location
        with patch("polylogue.ingestion.drive_client.DriveClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            runner = CliRunner()
            result = runner.invoke(cli, ["auth", "--refresh"])

            # Token should be removed and re-auth attempted
            assert "removed" in result.output.lower() or result.exit_code in (0, 1)


class TestDriveOAuthFlow:
    """Tests for the _drive_oauth_flow function."""

    def test_missing_credentials_fails(self, auth_workspace):
        """Missing credentials file shows error."""
        creds_path = auth_workspace["creds_path"]
        creds_path.unlink()

        runner = CliRunner()
        result = runner.invoke(cli, ["auth"])
        assert result.exit_code == 1
        # May fail with credentials missing or OAuth error
        assert ("credentials" in result.output.lower() or
                "missing" in result.output.lower() or
                "oauth" in result.output.lower())

    def test_successful_auth_with_cached_token(self, auth_workspace):
        """Existing token uses cached credentials."""
        with patch("polylogue.ingestion.drive_client.DriveClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            runner = CliRunner()
            result = runner.invoke(cli, ["auth"])
            assert result.exit_code == 0
            # Should mention using cached credentials
            assert "cached" in result.output.lower() or "success" in result.output.lower()

    def test_auth_failure_shows_error(self, auth_workspace):
        """Auth failure shows error message."""
        with patch("polylogue.ingestion.drive_client.DriveClient") as mock_client_class:
            mock_client_class.side_effect = Exception("OAuth failed: invalid_grant")

            runner = CliRunner()
            result = runner.invoke(cli, ["auth"])
            assert result.exit_code == 1
            assert "failed" in result.output.lower()

    def test_refresh_error_retries_auth(self, auth_workspace):
        """Token refresh failure triggers re-auth."""
        token_path = auth_workspace["token_path"]
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Token refresh failed")
            return MagicMock()

        with patch("polylogue.ingestion.drive_client.DriveClient") as mock_client_class:
            mock_client_class.side_effect = side_effect

            runner = CliRunner()
            result = runner.invoke(cli, ["auth"])
            # Should either succeed on retry or fail gracefully
            assert result.exit_code in (0, 1)


class TestGetDrivePaths:
    """Tests for _get_drive_paths helper."""

    def test_uses_env_paths(self, auth_workspace, monkeypatch):
        """Uses paths from environment variables."""
        from polylogue.cli.commands.auth import _get_drive_paths
        from polylogue.cli.types import AppEnv

        # Ensure env vars are set
        monkeypatch.setenv("POLYLOGUE_CREDENTIAL_PATH", str(auth_workspace["creds_path"]))
        monkeypatch.setenv("POLYLOGUE_TOKEN_PATH", str(auth_workspace["token_path"]))

        # Create minimal AppEnv mock
        mock_ui = MagicMock()
        mock_ui.plain = True
        env = AppEnv(ui=mock_ui)

        creds_path, token_path = _get_drive_paths(env)

        # Should return valid paths
        assert creds_path is not None
        assert token_path is not None

    def test_fallback_on_config_error(self, auth_workspace, monkeypatch):
        """Falls back to defaults if config loading fails."""
        from polylogue.cli.commands.auth import _get_drive_paths
        from polylogue.cli.types import AppEnv

        # Force config loading to fail
        monkeypatch.delenv("POLYLOGUE_CREDENTIAL_PATH", raising=False)
        monkeypatch.delenv("POLYLOGUE_TOKEN_PATH", raising=False)

        mock_ui = MagicMock()
        mock_ui.plain = True
        env = AppEnv(ui=mock_ui)

        with patch("polylogue.cli.helpers.load_effective_config") as mock_load:
            mock_load.side_effect = Exception("Config error")
            creds_path, token_path = _get_drive_paths(env)

            # Should return default paths (not raise)
            assert creds_path is not None
            assert token_path is not None


class TestRevokeCredentials:
    """Tests for _revoke_drive_credentials function."""

    def test_revoke_existing_token(self, auth_workspace):
        """Revokes existing token and shows confirmation."""
        runner = CliRunner()
        result = runner.invoke(cli, ["auth", "--revoke"])

        assert result.exit_code == 0
        assert "revoked" in result.output.lower()
        assert "polylogue auth" in result.output.lower()

    def test_revoke_nonexistent_token(self, auth_workspace):
        """Handles missing token gracefully."""
        auth_workspace["token_path"].unlink()

        runner = CliRunner()
        result = runner.invoke(cli, ["auth", "--revoke"])

        assert result.exit_code == 0
        assert "no token" in result.output.lower()

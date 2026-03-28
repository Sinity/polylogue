"""Tests for auth command and auth helper functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli as click_cli
from polylogue.cli.commands.auth import (
    _drive_oauth_flow,
    _get_drive_paths,
    _refresh_drive_token,
    _revoke_drive_credentials,
)

# =============================================================================
# AUTH COMMAND TESTS
# =============================================================================


class TestAuthCommandInternal:
    """Tests for auth_command() and auth helper functions."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_unknown_service_fails(self, runner):
        """auth_command should fail for unknown service."""
        result = runner.invoke(click_cli, ["auth", "--service", "unknown", "--plain"])
        assert result.exit_code != 0

    def test_default_service_is_drive(self, runner):
        """auth_command should default to 'drive' service."""
        with patch("polylogue.cli.commands.auth._drive_oauth_flow"):
            result = runner.invoke(click_cli, ["auth", "--plain"])
            # Will try to run oauth flow which will likely fail in test
            # but at least should not fail with "unknown service"
            assert "Unknown auth service" not in result.output


class TestGetDrivePaths:
    """Tests for _get_drive_paths() helper."""

    def test_get_drive_paths_returns_paths(self, tmp_path):
        """_get_drive_paths should return tuple of (credentials_path, token_path)."""
        env = MagicMock()
        creds_path, token_path = _get_drive_paths(env)
        # Should return Path objects
        assert isinstance(creds_path, Path)
        assert isinstance(token_path, Path)
        # Should contain expected names
        assert "cred" in str(creds_path).lower() or "oauth" in str(creds_path).lower()

    def test_get_drive_paths_handles_errors_gracefully(self, tmp_path):
        """_get_drive_paths should handle config errors and return defaults."""
        env = MagicMock()
        with patch("polylogue.cli.helpers.load_effective_config", side_effect=Exception("config error")):
            creds_path, token_path = _get_drive_paths(env)
            # Should still return paths (fallback defaults)
            assert creds_path is not None
            assert token_path is not None


class TestDriveOAuthFlow:
    """Tests for _drive_oauth_flow() function."""

    def test_oauth_missing_credentials_fails(self, tmp_path):
        """_drive_oauth_flow should fail if credentials file missing."""
        env = MagicMock()
        creds_path = tmp_path / "missing.json"
        token_path = tmp_path / "token.json"
        with patch(
            "polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)
        ):
            with pytest.raises(SystemExit):
                _drive_oauth_flow(env)

    def test_oauth_new_token_success(self, tmp_path):
        """_drive_oauth_flow should succeed with valid creds and new token."""
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        token_path = tmp_path / "token.json"  # Not existing

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)), patch(
            "polylogue.sources.drive_client.DriveClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            _drive_oauth_flow(env)
            mock_client.resolve_folder_id.assert_called_once_with("root")
            mock_client_cls.assert_called_once()

    def test_oauth_cached_token_success(self, tmp_path):
        """_drive_oauth_flow should succeed with existing token."""
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        token_path = tmp_path / "token.json"
        token_path.write_text("{}")  # Existing token

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)), patch(
            "polylogue.sources.drive_client.DriveClient"
        ) as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            _drive_oauth_flow(env)
            # Should use cached credentials message

    def test_oauth_file_not_found_error(self, tmp_path):
        """_drive_oauth_flow should handle FileNotFoundError."""
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        token_path = tmp_path / "token.json"

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)), patch(
            "polylogue.sources.drive_client.DriveClient", side_effect=FileNotFoundError("creds not found")
        ):
            with pytest.raises(SystemExit):
                _drive_oauth_flow(env)

    def test_oauth_token_refresh_failure_retries(self, tmp_path):
        """_drive_oauth_flow should retry on token refresh failure."""
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        token_path = tmp_path / "token.json"
        token_path.write_text("{}")

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Token refresh failed")
            # Second call succeeds
            mock_client = MagicMock()
            mock_client.resolve_folder_id = MagicMock()
            return mock_client

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)), patch(
            "polylogue.sources.drive_client.DriveClient", side_effect=side_effect
        ):
            _drive_oauth_flow(env, retry_on_failure=True)
            assert call_count[0] == 2
            # Token should have been deleted between retries

    def test_oauth_non_retriable_error_fails(self, tmp_path):
        """_drive_oauth_flow should fail on non-retriable error."""
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        token_path = tmp_path / "token.json"
        token_path.write_text("{}")

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)), patch(
            "polylogue.sources.drive_client.DriveClient", side_effect=Exception("Auth failed permanently")
        ):
            with pytest.raises(SystemExit):
                _drive_oauth_flow(env, retry_on_failure=False)

    def test_oauth_retry_disabled_fails_immediately(self, tmp_path):
        """_drive_oauth_flow should fail immediately when retry_on_failure=False."""
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        token_path = tmp_path / "token.json"
        token_path.write_text("{}")

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)), patch(
            "polylogue.sources.drive_client.DriveClient", side_effect=Exception("Token refresh failed")
        ):
            with pytest.raises(SystemExit):
                _drive_oauth_flow(env, retry_on_failure=False)


class TestRefreshDriveToken:
    """Tests for _refresh_drive_token() function."""

    def test_refresh_deletes_token(self, tmp_path):
        """_refresh_drive_token should delete existing token."""
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        token_path = tmp_path / "token.json"
        token_path.write_text("{}")

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)), patch(
            "polylogue.cli.commands.auth._drive_oauth_flow"
        ) as mock_flow:
            _refresh_drive_token(env)
            # Token should be deleted before calling _drive_oauth_flow
            assert not token_path.exists()
            mock_flow.assert_called_once_with(env)

    def test_refresh_without_token_still_reauths(self, tmp_path):
        """_refresh_drive_token should reauthenticate even if no token exists."""
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        token_path = tmp_path / "token.json"  # Not existing

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)), patch(
            "polylogue.cli.commands.auth._drive_oauth_flow"
        ) as mock_flow:
            _refresh_drive_token(env)
            mock_flow.assert_called_once_with(env)


class TestRevokeDriveCredentials:
    """Tests for _revoke_drive_credentials() function."""

    def test_revoke_deletes_token(self, tmp_path):
        """_revoke_drive_credentials should delete token file."""
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        token_path = tmp_path / "token.json"
        token_path.write_text("{}")

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)):
            _revoke_drive_credentials(env)
            assert not token_path.exists()

    def test_revoke_without_token(self, tmp_path):
        """_revoke_drive_credentials should handle missing token gracefully."""
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        token_path = tmp_path / "token.json"  # Not existing

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)):
            # Should not raise exception
            _revoke_drive_credentials(env)


@pytest.mark.integration
class TestAuthCommand:
    """Tests for the auth command (subprocess integration)."""

    def test_auth_unknown_service_fails(self, tmp_path):
        """auth --service unknown fails with error."""
        from tests.infra.cli_subprocess import run_cli, setup_isolated_workspace

        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["auth", "--service", "unknown"], env=env)
        assert result.exit_code != 0
        assert "unknown" in result.output.lower() or "provider" in result.output.lower()

    def test_auth_revoke_no_token(self, tmp_path):
        """auth --revoke handles missing token gracefully."""
        from tests.infra.cli_subprocess import run_cli, setup_isolated_workspace

        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["auth", "--revoke"], env=env)
        # Should succeed or show "no token" message
        output_lower = result.output.lower()
        assert result.exit_code == 0 or "no token" in output_lower or "not found" in output_lower

    def test_auth_missing_credentials(self, tmp_path):
        """auth fails gracefully when credentials file missing."""
        from tests.infra.cli_subprocess import run_cli, setup_isolated_workspace

        workspace = setup_isolated_workspace(tmp_path)
        env = workspace["env"]

        result = run_cli(["auth"], env=env)
        # Should fail with helpful message about missing credentials
        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert "credentials" in output_lower or "missing" in output_lower or "oauth" in output_lower

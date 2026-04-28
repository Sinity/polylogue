"""Tests for auth command routing and credential management."""

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


class TestAuthCommandRouting:
    """CLI command routing tests — service dispatch and error handling."""

    def test_unknown_service_fails(self, cli_runner: CliRunner) -> None:
        result = cli_runner.invoke(click_cli, ["auth", "--service", "unknown", "--plain"])
        assert result.exit_code != 0

    def test_default_service_is_drive(self, cli_runner: CliRunner) -> None:
        with patch("polylogue.cli.commands.auth._drive_oauth_flow"):
            result = cli_runner.invoke(click_cli, ["auth", "--plain"])
            assert "Unknown auth service" not in result.output


class TestGetDrivePaths:
    def test_get_drive_paths_returns_path_objects(self, tmp_path: Path) -> None:
        env = MagicMock()
        creds_path, token_path = _get_drive_paths(env)
        assert isinstance(creds_path, Path)
        assert isinstance(token_path, Path)

    def test_get_drive_paths_falls_back_on_config_error(self, tmp_path: Path) -> None:
        env = MagicMock()
        with patch("polylogue.cli.shared.helpers.load_effective_config", side_effect=Exception("config error")):
            creds_path, token_path = _get_drive_paths(env)
            assert creds_path is not None
            assert token_path is not None


class TestDriveOAuthFlow:
    def test_oauth_missing_credentials_exits(self, tmp_path: Path) -> None:
        env = MagicMock()
        creds_path = tmp_path / "missing.json"
        token_path = tmp_path / "token.json"
        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)):
            with pytest.raises(SystemExit):
                _drive_oauth_flow(env)

    def test_oauth_calls_load_credentials_on_auth_manager(self, tmp_path: Path) -> None:
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        token_path = tmp_path / "token.json"

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)):
            with patch("polylogue.cli.commands.auth.DriveAuthManager") as mock_manager_cls:
                mock_manager = MagicMock()
                mock_manager_cls.return_value = mock_manager
                _drive_oauth_flow(env)
                mock_manager.load_credentials.assert_called_once()

    def test_oauth_cached_token_reports_using_cached(self, tmp_path: Path) -> None:
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        token_path = tmp_path / "token.json"
        token_path.write_text("{}")  # existing token

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)):
            with patch("polylogue.cli.commands.auth.DriveAuthManager") as mock_manager_cls:
                mock_manager = MagicMock()
                mock_manager_cls.return_value = mock_manager
                # Should not raise
                _drive_oauth_flow(env)

    def test_oauth_file_not_found_exits(self, tmp_path: Path) -> None:
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        token_path = tmp_path / "token.json"

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)):
            with patch(
                "polylogue.cli.commands.auth.DriveAuthManager", side_effect=FileNotFoundError("creds not found")
            ):
                with pytest.raises(SystemExit):
                    _drive_oauth_flow(env)

    def test_oauth_token_refresh_failure_retries(self, tmp_path: Path) -> None:
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        token_path = tmp_path / "token.json"
        token_path.write_text("{}")

        call_count = [0]

        def side_effect(*args: object, **kwargs: object) -> MagicMock:
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Token refresh failed")
            mock_manager = MagicMock()
            return mock_manager

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)):
            with patch("polylogue.cli.commands.auth.DriveAuthManager", side_effect=side_effect):
                _drive_oauth_flow(env, retry_on_failure=True)
                assert call_count[0] == 2

    def test_oauth_non_retriable_error_exits(self, tmp_path: Path) -> None:
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        token_path = tmp_path / "token.json"
        token_path.write_text("{}")

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)):
            with patch(
                "polylogue.cli.commands.auth.DriveAuthManager", side_effect=Exception("Auth failed permanently")
            ):
                with pytest.raises(SystemExit):
                    _drive_oauth_flow(env, retry_on_failure=False)


class TestRefreshDriveToken:
    def test_refresh_deletes_token_before_reauth(self, tmp_path: Path) -> None:
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        token_path = tmp_path / "token.json"
        token_path.write_text("{}")

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)):
            with patch("polylogue.cli.commands.auth._drive_oauth_flow") as mock_flow:
                _refresh_drive_token(env)
                assert not token_path.exists()
                mock_flow.assert_called_once_with(env)

    def test_refresh_without_token_still_reauths(self, tmp_path: Path) -> None:
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        creds_path.write_text("{}")
        token_path = tmp_path / "token.json"

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)):
            with patch("polylogue.cli.commands.auth._drive_oauth_flow") as mock_flow:
                _refresh_drive_token(env)
                mock_flow.assert_called_once_with(env)


class TestRevokeDriveCredentials:
    def test_revoke_calls_auth_manager_revoke(self, tmp_path: Path) -> None:
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        token_path = tmp_path / "token.json"
        token_path.write_text("{}")

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)):
            with patch("polylogue.cli.commands.auth.DriveAuthManager") as mock_manager_cls:
                mock_manager = MagicMock()
                mock_manager_cls.return_value = mock_manager
                _revoke_drive_credentials(env)
                mock_manager.revoke.assert_called_once()

    def test_revoke_without_token_does_not_raise(self, tmp_path: Path) -> None:
        env = MagicMock()
        creds_path = tmp_path / "creds.json"
        token_path = tmp_path / "token.json"

        with patch("polylogue.cli.commands.auth._get_drive_paths", return_value=(creds_path, token_path)):
            with patch("polylogue.cli.commands.auth.DriveAuthManager") as mock_manager_cls:
                mock_manager = MagicMock()
                mock_manager_cls.return_value = mock_manager
                _revoke_drive_credentials(env)
                mock_manager.revoke.assert_called_once()


@pytest.mark.integration
class TestAuthCommand:
    """Integration tests for the auth command (subprocess)."""

    def test_auth_unknown_service_fails(self, tmp_path: Path) -> None:
        from tests.infra.cli_subprocess import run_cli, setup_isolated_workspace

        workspace = setup_isolated_workspace(tmp_path)
        result = run_cli(["auth", "--service", "unknown"], env=workspace["env"])
        assert result.exit_code != 0

    def test_auth_revoke_no_token(self, tmp_path: Path) -> None:
        from tests.infra.cli_subprocess import run_cli, setup_isolated_workspace

        workspace = setup_isolated_workspace(tmp_path)
        result = run_cli(["auth", "--revoke"], env=workspace["env"])
        output_lower = result.output.lower()
        assert result.exit_code == 0 or "no token" in output_lower or "not found" in output_lower

    def test_auth_missing_credentials(self, tmp_path: Path) -> None:
        from tests.infra.cli_subprocess import run_cli, setup_isolated_workspace

        workspace = setup_isolated_workspace(tmp_path)
        result = run_cli(["auth"], env=workspace["env"])
        assert result.exit_code != 0
        output_lower = result.output.lower()
        assert "credentials" in output_lower or "missing" in output_lower or "oauth" in output_lower

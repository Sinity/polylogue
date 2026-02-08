"""Auth command for OAuth flows (Google Drive)."""

from __future__ import annotations

from pathlib import Path

import click

from polylogue.cli.types import AppEnv


@click.command("auth")
@click.option("--service", default="drive", help="Auth service (default: drive)")
@click.option("--refresh", is_flag=True, help="Force token refresh")
@click.option("--revoke", is_flag=True, help="Revoke existing credentials")
@click.pass_obj
def auth_command(env: AppEnv, service: str, refresh: bool, revoke: bool) -> None:
    """Authenticate with external services (Google Drive for Gemini).

    \b
    Examples:
        polylogue auth           # Interactive OAuth for Google Drive
        polylogue auth --refresh # Force token refresh
        polylogue auth --revoke  # Revoke stored credentials
    """
    ui = env.ui

    if service != "drive":
        click.echo(f"Unknown auth service: {service}", err=True)
        click.echo("Available services: drive", err=True)
        raise SystemExit(1)

    if revoke:
        _revoke_drive_credentials(env)
        return

    if refresh:
        _refresh_drive_token(env)
        return

    # Interactive OAuth flow
    _drive_oauth_flow(env)


def _get_drive_paths(env: AppEnv) -> tuple[Path, Path]:
    """Get credentials and token paths from config or defaults."""
    from polylogue.cli.helpers import load_effective_config
    from polylogue.sources.drive_client import default_credentials_path, default_token_path

    try:
        config = load_effective_config(env)
        drive_config = config.drive_config
        credentials_path = default_credentials_path(drive_config)
        token_path = default_token_path(drive_config)
        return credentials_path, token_path
    except Exception:
        # Fallback to defaults if config loading fails
        return default_credentials_path(), default_token_path()


def _drive_oauth_flow(env: AppEnv, retry_on_failure: bool = True) -> None:
    """Run interactive OAuth flow for Google Drive."""
    credentials_path, token_path = _get_drive_paths(env)

    if not credentials_path.exists():
        click.echo(f"Missing credentials file: {credentials_path}", err=True)
        click.echo("Download OAuth credentials from Google Cloud Console.", err=True)
        raise SystemExit(1)

    # Check if we have a valid token already
    has_token = token_path.exists()

    env.ui.console.print("Starting Google Drive OAuth flow...")
    if not has_token:
        env.ui.console.print("A browser window will open for authentication.")

    try:
        from polylogue.sources.drive_client import DriveClient

        client = DriveClient(
            credentials_path=credentials_path,
            token_path=token_path,
            ui=env.ui,
        )
        # Trigger auth by resolving root folder (forces credential load)
        client.resolve_folder_id("root")

        if has_token:
            click.echo("✓ Using cached credentials")
        else:
            click.echo("Authentication successful!")
    except FileNotFoundError as exc:
        click.echo(f"Missing credentials file: {exc}", err=True)
        click.echo("Download OAuth credentials from Google Cloud Console.", err=True)
        raise SystemExit(1) from exc
    except Exception as exc:
        # If token refresh failed and we haven't retried yet, delete token and retry
        if retry_on_failure and token_path.exists() and "refresh" in str(exc).lower():
            click.echo("Token expired or revoked. Removing and re-authenticating...")
            token_path.unlink()
            _drive_oauth_flow(env, retry_on_failure=False)
            return
        click.echo(f"OAuth failed: {exc}", err=True)
        raise SystemExit(1) from exc


def _refresh_drive_token(env: AppEnv) -> None:
    """Force refresh of Drive OAuth token by re-authenticating."""
    credentials_path, token_path = _get_drive_paths(env)

    # Delete existing token to force re-auth
    if token_path.exists():
        token_path.unlink()
        env.ui.console.print(f"Removed existing token: {token_path}")

    # Run OAuth flow again
    _drive_oauth_flow(env)


def _revoke_drive_credentials(env: AppEnv) -> None:
    """Revoke stored Drive credentials."""
    _, token_path = _get_drive_paths(env)

    if token_path.exists():
        token_path.unlink()
        click.echo(f"Removed token file: {token_path}")
    else:
        click.echo("No token file found.")

    click.echo("Credentials revoked.")
    click.echo("Run `polylogue auth` to re-authenticate.")


__all__ = ["auth_command"]

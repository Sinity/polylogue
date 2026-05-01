# mypy: disable-error-code="assignment,arg-type,comparison-overlap"

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from polylogue.sources.drive.source_factory import build_drive_source_client


def test_build_drive_source_client_wires_auth_gateway_and_client(tmp_path: Path) -> None:
    ui = object()
    config = object()

    with (
        patch(
            "polylogue.sources.drive.source_factory.resolve_drive_retry_policy", return_value="retry-policy"
        ) as mock_policy,
        patch("polylogue.sources.drive.source_factory.DriveAuthManager", return_value="auth-manager") as mock_auth,
        patch("polylogue.sources.drive.source_factory.DriveServiceGateway", return_value="gateway") as mock_gateway,
        patch("polylogue.sources.drive.source_factory.DriveSourceClient", return_value="client") as mock_client,
    ):
        client = build_drive_source_client(
            ui=ui,
            credentials_path=tmp_path / "credentials.json",
            token_path=tmp_path / "token.json",
            retries=5,
            retry_base=2.5,
            config=config,
        )

    assert client == "client"
    mock_policy.assert_called_once_with(retries=5, retry_base=2.5, config=config)
    mock_auth.assert_called_once_with(
        ui=ui,
        credentials_path=tmp_path / "credentials.json",
        token_path=tmp_path / "token.json",
        config=config,
    )
    mock_gateway.assert_called_once_with(auth_manager="auth-manager", retry_policy="retry-policy")
    mock_client.assert_called_once_with(gateway="gateway")

from __future__ import annotations

from pathlib import Path

from .drive_auth import DriveAuthManager
from .drive_gateway import (
    DriveServiceGateway,
    _resolve_retries,
    _resolve_retry_base,
)
from .drive_source_client import DriveSourceClient


def build_drive_source_client(
    *,
    ui: object | None = None,
    credentials_path: Path | None = None,
    token_path: Path | None = None,
    retries: int | None = None,
    retry_base: float | None = None,
    config: object | None = None,
) -> DriveSourceClient:
    """Build the canonical Drive source client from auth and gateway primitives."""
    auth_manager = DriveAuthManager(
        ui=ui,
        credentials_path=credentials_path,
        token_path=token_path,
        config=config,
    )
    gateway = DriveServiceGateway(
        auth_manager=auth_manager,
        retries=_resolve_retries(retries, config),
        retry_base=_resolve_retry_base(retry_base),
    )
    return DriveSourceClient(gateway=gateway)

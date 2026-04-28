from __future__ import annotations

from pathlib import Path

from .drive_auth import DriveAuthManager
from .drive_gateway import (
    DriveServiceGateway,
    resolve_drive_retry_policy,
)
from .drive_source_client import DriveSourceClient
from .drive_types import DriveConfigLike, DriveUILike


def build_drive_source_client(
    *,
    ui: DriveUILike | None = None,
    credentials_path: Path | None = None,
    token_path: Path | None = None,
    retries: int | None = None,
    retry_base: float | None = None,
    config: DriveConfigLike | None = None,
) -> DriveSourceClient:
    """Build the canonical Drive source client from auth and gateway primitives."""
    retry_policy = resolve_drive_retry_policy(
        retries=retries,
        retry_base=retry_base,
        config=config,
    )
    auth_manager = DriveAuthManager(
        ui=ui,
        credentials_path=credentials_path,
        token_path=token_path,
        config=config,
    )
    gateway = DriveServiceGateway(
        auth_manager=auth_manager,
        retry_policy=retry_policy,
    )
    return DriveSourceClient(gateway=gateway)

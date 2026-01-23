from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..assets import asset_path
from ..config import Source
from .drive_client import DriveClient
from .source import ParsedConversation, parse_drive_payload

LOGGER = logging.getLogger(__name__)


@dataclass
class DriveDownloadResult:
    """Result of Drive file download operation."""

    downloaded_files: list[Path]
    failed_files: list[dict]
    total_files: int


def download_drive_files(
    client: DriveClient,
    folder_id: str,
    dest_dir: Path,
) -> DriveDownloadResult:
    """Download files from Drive folder with failure tracking.

    Args:
        client: DriveClient instance
        folder_id: Drive folder ID to download from
        dest_dir: Destination directory for downloaded files

    Returns:
        DriveDownloadResult with lists of downloaded/failed files and counts
    """
    files = client.list_files(folder_id)
    downloaded = []
    failed = []

    for file_info in files:
        file_id = file_info["id"]
        name = file_info["name"]
        dest_path = dest_dir / name

        try:
            client.download_file(file_id, dest_path)
            downloaded.append(dest_path)
        except Exception as exc:
            LOGGER.warning("Failed to download %s (%s): %s", name, file_id, exc)
            failed.append(
                {
                    "file_id": file_id,
                    "name": name,
                    "error": str(exc),
                }
            )
            continue

    return DriveDownloadResult(
        downloaded_files=downloaded,
        failed_files=failed,
        total_files=len(files),
    )


def _apply_drive_attachments(
    *,
    convo: ParsedConversation,
    client: DriveClient,
    archive_root: Path,
    download_assets: bool,
) -> None:
    if not download_assets:
        return
    for attachment in convo.attachments:
        if not attachment.provider_attachment_id:
            continue
        dest = asset_path(archive_root, attachment.provider_attachment_id)
        meta = client.download_to_path(attachment.provider_attachment_id, dest)
        attachment.path = str(dest)
        attachment.name = attachment.name or meta.name
        attachment.mime_type = attachment.mime_type or meta.mime_type
        attachment.size_bytes = attachment.size_bytes or meta.size_bytes
        meta_payload = dict(attachment.provider_meta or {})
        meta_payload.update(
            {
                "drive_id": attachment.provider_attachment_id,
                "name": attachment.name,
                "mime_type": attachment.mime_type,
            }
        )
        attachment.provider_meta = meta_payload


def iter_drive_conversations(
    *,
    source: Source,
    archive_root: Path,
    ui: object | None = None,
    client: DriveClient | None = None,
    download_assets: bool = True,
    cursor_state: dict | None = None,
) -> Iterable[ParsedConversation]:
    if not source.folder:
        return
    drive_client = client or DriveClient(ui=ui)
    folder_id = drive_client.resolve_folder_id(source.folder)
    if cursor_state is not None:
        cursor_state.setdefault("file_count", 0)
    for file_meta in drive_client.iter_json_files(folder_id):
        if cursor_state is not None:
            cursor_state["file_count"] = cursor_state.get("file_count", 0) + 1
            if file_meta.modified_time:
                ts = _parse_modified_time(file_meta.modified_time)
                last_ts = cursor_state.get("latest_mtime")
                if ts is not None and (last_ts is None or ts > last_ts):
                    cursor_state["latest_mtime"] = ts
                    cursor_state["latest_file_id"] = file_meta.file_id
                    cursor_state["latest_file_name"] = file_meta.name
        try:
            payload = drive_client.download_json_payload(file_meta.file_id, name=file_meta.name)
        except Exception as exc:
            if cursor_state is not None:
                cursor_state["error_count"] = cursor_state.get("error_count", 0) + 1
                cursor_state["latest_error"] = str(exc)
                cursor_state["latest_error_file"] = file_meta.name
            LOGGER.warning(
                "Failed to download Drive payload for %s (%s): %s",
                file_meta.name,
                file_meta.file_id,
                exc,
            )
            continue
        conversations = parse_drive_payload(source.name, payload, file_meta.file_id)
        for convo in conversations:
            _apply_drive_attachments(
                convo=convo,
                client=drive_client,
                archive_root=archive_root,
                download_assets=download_assets,
            )
            yield convo


def _parse_modified_time(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return None


__all__ = ["iter_drive_conversations", "download_drive_files", "DriveDownloadResult"]

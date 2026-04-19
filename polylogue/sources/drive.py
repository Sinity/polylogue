from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from polylogue.logging import get_logger
from polylogue.types import Provider

from ..assets import asset_path
from ..config import Source
from ..paths import safe_path_component
from .dispatch import parse_drive_payload
from .drive_source import DriveSourceAPI, _parse_modified_time, build_drive_source_client
from .drive_types import DriveConfigLike, DriveUILike
from .parsers.base import ParsedConversation, RawConversationData

logger = get_logger(__name__)


@dataclass
class DriveDownloadResult:
    """Result of Drive file download operation."""

    downloaded_files: list[Path]
    failed_files: list[dict[str, str | int]]
    total_files: int


def drive_cache_file_path(dest_dir: Path, name: str) -> Path:
    """Return the canonical local cache path for a Drive JSON payload."""
    safe_name = safe_path_component(name, fallback="drive_file")
    if not any(safe_name.lower().endswith(ext) for ext in (".json", ".jsonl", ".ndjson")):
        safe_name += ".json"
    return dest_dir / safe_name


def download_drive_files(
    client: DriveSourceAPI,
    folder_id: str,
    dest_dir: Path,
) -> DriveDownloadResult:
    """Download files from Drive folder with failure tracking.

    Args:
        client: Drive source client
        folder_id: Drive folder ID to download from
        dest_dir: Destination directory for downloaded files

    Returns:
        DriveDownloadResult with lists of downloaded/failed files and counts
    """
    downloaded: list[Path] = []
    failed: list[dict[str, str | int]] = []

    for file_info in client.iter_json_files(folder_id):
        file_id = file_info.file_id
        name = file_info.name
        dest_path = drive_cache_file_path(dest_dir, name)

        try:
            client.download_to_path(file_id, dest_path)
            downloaded.append(dest_path)
        except Exception as exc:
            logger.warning("Failed to download %s (%s): %s", name, file_id, exc)
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
        total_files=len(downloaded) + len(failed),
    )


def _apply_drive_attachments(
    *,
    convo: ParsedConversation,
    client: DriveSourceAPI,
    archive_root: Path,
    download_assets: bool,
) -> None:
    if not download_assets:
        return
    for attachment in convo.attachments:
        if not attachment.provider_attachment_id:
            continue
        attachment_kind = None
        if isinstance(attachment.provider_meta, dict):
            raw_kind = attachment.provider_meta.get("attachment_kind")
            attachment_kind = raw_kind if isinstance(raw_kind, str) else None
        if attachment_kind in {"inline_file", "youtube_video"}:
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
    ui: DriveUILike | None = None,
    client: DriveSourceAPI | None = None,
    download_assets: bool = True,
    cursor_state: dict[str, Any] | None = None,
    drive_config: DriveConfigLike | None = None,
) -> Iterable[ParsedConversation]:
    if not source.folder:
        return
    drive_client = client or build_drive_source_client(ui=ui, config=drive_config)
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
            logger.warning(
                "Failed to download Drive payload for %s (%s): %s",
                file_meta.name,
                file_meta.file_id,
                exc,
            )
            continue
        fallback_id = drive_cache_file_path(source.path or Path(source.name), file_meta.name).stem
        conversations = parse_drive_payload(source.name, payload, fallback_id)
        for convo in conversations:
            _apply_drive_attachments(
                convo=convo,
                client=drive_client,
                archive_root=archive_root,
                download_assets=download_assets,
            )
            yield convo


def iter_drive_raw_data(
    *,
    source: Source,
    ui: DriveUILike | None = None,
    client: DriveSourceAPI | None = None,
    cursor_state: dict[str, Any] | None = None,
    drive_config: DriveConfigLike | None = None,
    known_mtimes: dict[str, str] | None = None,
) -> Iterable[RawConversationData]:
    """Iterate Drive payloads as raw bytes without writing a local cache.

    Note: googleapiclient / httplib2 are not thread-safe — a single service
    object cannot be shared across threads. Downloads therefore remain
    sequential at the Drive runtime boundary.
    """
    if not source.folder:
        return

    drive_client = client or build_drive_source_client(ui=ui, config=drive_config)
    folder_id = drive_client.resolve_folder_id(source.folder)
    if cursor_state is not None:
        cursor_state.setdefault("file_count", 0)

    for file_meta in drive_client.iter_json_files(folder_id):
        dest_path = drive_cache_file_path(source.path or Path(source.name), file_meta.name)
        source_path = str(dest_path)
        if cursor_state is not None:
            cursor_state["file_count"] = cursor_state.get("file_count", 0) + 1
            if file_meta.modified_time:
                ts = _parse_modified_time(file_meta.modified_time)
                last_ts = cursor_state.get("latest_mtime")
                if ts is not None and (last_ts is None or ts > last_ts):
                    cursor_state["latest_mtime"] = ts
                    cursor_state["latest_file_id"] = file_meta.file_id
                    cursor_state["latest_file_name"] = file_meta.name

        if (
            known_mtimes is not None
            and file_meta.modified_time is not None
            and known_mtimes.get(source_path) == file_meta.modified_time
        ):
            continue

        from polylogue.storage.blob_store import get_blob_store

        blob_store = get_blob_store()

        # Check if a local cache file exists (drive-cache or legacy path).
        # If so, use it instead of re-downloading from Drive.
        cache_path = drive_cache_file_path(source.path or Path(source.name), file_meta.name)
        blob_hash: str | None = None
        blob_size: int = 0

        if cache_path.exists():
            blob_hash, blob_size = blob_store.write_from_path(cache_path)
        else:
            try:
                raw_bytes = drive_client.download_bytes(file_meta.file_id)
            except Exception as exc:
                if cursor_state is not None:
                    cursor_state["error_count"] = cursor_state.get("error_count", 0) + 1
                    cursor_state["latest_error"] = str(exc)
                    cursor_state["latest_error_file"] = file_meta.name
                logger.warning(
                    "Failed to download Drive payload for %s (%s): %s",
                    file_meta.name,
                    file_meta.file_id,
                    exc,
                )
                continue
            blob_hash, blob_size = blob_store.write_from_bytes(raw_bytes)
            # Write to cache for future runs
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(raw_bytes)
            del raw_bytes

        provider_hint = Provider.from_string(source.name)
        yield RawConversationData(
            raw_bytes=b"",
            source_path=source_path,
            source_index=None,
            file_mtime=file_meta.modified_time,
            provider_hint=provider_hint,
            blob_hash=blob_hash,
            blob_size=blob_size,
        )


__all__ = [
    "DriveDownloadResult",
    "download_drive_files",
    "drive_cache_file_path",
    "iter_drive_conversations",
    "iter_drive_raw_data",
]

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..assets import asset_path
from ..config import Source
from polylogue.logging import get_logger
from ..paths import safe_path_component
from .drive_client import DriveClient
from .drive_source import DriveSourceAPI, _parse_modified_time
from .parsers.base import RawConversationData
from .source import ParsedConversation, detect_provider, parse_drive_payload

logger = get_logger(__name__)

# Maximum concurrent Drive download threads.  Each download is an independent
# HTTP request to the Drive API; 4 workers reduce wall time ~4× compared to
# sequential downloads on a typical connection without overwhelming the API.
_DRIVE_DOWNLOAD_CONCURRENCY = 4


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
        client: DriveClient instance
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
    client: DriveSourceAPI | None = None,
    download_assets: bool = True,
    cursor_state: dict[str, Any] | None = None,
    drive_config: object | None = None,
) -> Iterable[ParsedConversation]:
    if not source.folder:
        return
    drive_client = client or DriveClient(ui=ui, config=drive_config)
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
        conversations = parse_drive_payload(source.name, payload, file_meta.file_id)
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
    ui: object | None = None,
    client: DriveSourceAPI | None = None,
    cursor_state: dict[str, Any] | None = None,
    drive_config: object | None = None,
    known_mtimes: dict[str, str] | None = None,
) -> Iterable[RawConversationData]:
    """Iterate Drive payloads as raw bytes without writing a local cache.

    Downloads run in a thread pool (up to _DRIVE_DOWNLOAD_CONCURRENCY
    concurrent workers) so independent HTTP requests overlap.  Results are
    yielded in the original listing order.

    Two-phase approach:
    1. List all files (sequential paginated API call) — cheap, updates cursor.
    2. Download needed files in parallel — the expensive network phase.
    """
    if not source.folder:
        return

    drive_client = client or DriveClient(ui=ui, config=drive_config)
    folder_id = drive_client.resolve_folder_id(source.folder)

    # Phase 1: collect metadata and update cursor (sequential API listing).
    all_files: list[tuple[Any, str]] = []  # (DriveFile, source_path)
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
        all_files.append((file_meta, source_path))

    # Phase 2: filter known-mtime files, download the rest in parallel.
    to_download: list[tuple[Any, str]] = []
    for file_meta, source_path in all_files:
        if (
            known_mtimes is not None
            and file_meta.modified_time is not None
            and known_mtimes.get(source_path) == file_meta.modified_time
        ):
            continue
        to_download.append((file_meta, source_path))

    if not to_download:
        return

    worker_count = min(len(to_download), _DRIVE_DOWNLOAD_CONCURRENCY)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            (file_meta, source_path, executor.submit(drive_client.download_bytes, file_meta.file_id))
            for file_meta, source_path in to_download
        ]

    # Yield results in original listing order; handle per-file failures.
    for file_meta, source_path, future in futures:
        try:
            raw_bytes = future.result()
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

        provider_hint = detect_provider(None, Path(file_meta.name)) or source.name
        yield RawConversationData(
            raw_bytes=raw_bytes,
            source_path=source_path,
            source_index=None,
            file_mtime=file_meta.modified_time,
            provider_hint=provider_hint,
        )


__all__ = [
    "DriveDownloadResult",
    "download_drive_files",
    "drive_cache_file_path",
    "iter_drive_conversations",
    "iter_drive_raw_data",
]

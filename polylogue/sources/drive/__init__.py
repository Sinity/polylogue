from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.core.json import JSONValue
from polylogue.logging import get_logger
from polylogue.storage.blob_publication import publication_receipt_id
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload

from ...config import Source
from ...paths.sanitize import safe_path_component
from ..parsers.base import RawSessionData
from ..source_acquisition_components import (
    ObservationCallback,
    StatusCallback,
    make_status_heartbeat,
    observe_acquisition,
)
from .attachment_fetch import fetch_live_drive_attachment_bytes
from .source import DriveSourceAPI, _parse_modified_time, build_drive_source_client
from .types import DriveConfigLike, DriveFile, DriveUILike

logger = get_logger(__name__)


@dataclass
class DriveDownloadResult:
    """Result of Drive file download operation."""

    downloaded_files: list[Path]
    failed_files: list[dict[str, str | int]]
    total_files: int


@dataclass(slots=True)
class _DriveCursorTracker:
    cursor_state: CursorStatePayload | None

    def _record_latest_file(self, file_meta: DriveFile) -> None:
        if self.cursor_state is None or not file_meta.modified_time:
            return
        modified_timestamp = _parse_modified_time(file_meta.modified_time)
        last_timestamp = self.cursor_state.get("latest_mtime")
        if modified_timestamp is None or (last_timestamp is not None and modified_timestamp <= last_timestamp):
            return
        self.cursor_state["latest_mtime"] = modified_timestamp
        self.cursor_state["latest_file_id"] = file_meta.file_id
        self.cursor_state["latest_file_name"] = file_meta.name

    def observe_file(self, file_meta: DriveFile) -> None:
        if self.cursor_state is None:
            return
        self.cursor_state["file_count"] = self.cursor_state.get("file_count", 0) + 1
        self._record_latest_file(file_meta)

    def record_failure(self, *, file_name: str, error: Exception) -> None:
        if self.cursor_state is None:
            return
        self.cursor_state["error_count"] = self.cursor_state.get("error_count", 0) + 1
        self.cursor_state["latest_error"] = str(error)
        self.cursor_state["latest_error_file"] = file_name


def _cursor_tracker(cursor_state: CursorStatePayload | None) -> _DriveCursorTracker:
    if cursor_state is not None:
        cursor_state.setdefault("file_count", 0)
    return _DriveCursorTracker(cursor_state)


def _resolved_drive_client(
    *,
    ui: DriveUILike | None,
    client: DriveSourceAPI | None,
    drive_config: DriveConfigLike | None,
) -> DriveSourceAPI:
    return client or build_drive_source_client(ui=ui, config=drive_config)


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


def _inject_live_drive_attachment_bytes(
    raw_bytes: bytes,
    drive_client: DriveSourceAPI,
    file_meta: DriveFile,
) -> bytes:
    """Fetch live Drive-hosted attachment bytes into the raw session payload.

    Must run here, before this function's caller's live-client scope closes:
    googleapiclient/httplib2 are not thread-safe, and acquire (this generator,
    with a live client) and parse (a separate subprocess, no client) are
    deliberately decoupled for memory-bounded streaming. This is the one place
    both the live client and the raw JSON are available together.

    Returns ``raw_bytes`` unchanged when nothing was fetched (no Drive-hosted
    references found, or all fetches failed/were oversize), so an ordinary
    session's raw bytes are never needlessly re-serialized.
    """
    try:
        payload: JSONValue = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw_bytes
    resolved, stats = fetch_live_drive_attachment_bytes(payload, drive_client.download_bytes)
    if stats.fetched_count == 0:
        return raw_bytes
    logger.info(
        "Resolved %d live Drive attachment(s) for %s (%d bytes fetched, %d failed, %d oversize)",
        stats.fetched_count,
        file_meta.name,
        stats.fetched_bytes,
        stats.failed_count,
        stats.skipped_too_large_count,
    )
    return json.dumps(resolved, ensure_ascii=False).encode("utf-8")


def iter_drive_raw_data(
    *,
    source: Source,
    ui: DriveUILike | None = None,
    client: DriveSourceAPI | None = None,
    cursor_state: CursorStatePayload | None = None,
    drive_config: DriveConfigLike | None = None,
    known_mtimes: dict[str, str] | None = None,
    observation_callback: ObservationCallback | None = None,
    status_callback: StatusCallback | None = None,
    blob_store: BlobStore | None = None,
) -> Iterable[RawSessionData]:
    """Iterate Drive payloads as raw bytes without writing a local cache.

    Note: googleapiclient / httplib2 are not thread-safe — a single service
    object cannot be shared across threads. Downloads therefore remain
    sequential at the Drive runtime boundary.
    """
    if not source.folder:
        return

    drive_client = _resolved_drive_client(ui=ui, client=client, drive_config=drive_config)
    folder_id = drive_client.resolve_folder_id(source.folder)
    tracker = _cursor_tracker(cursor_state)

    for file_meta in drive_client.iter_json_files(folder_id):
        dest_path = drive_cache_file_path(source.path or Path(source.name), file_meta.name)
        source_path = str(dest_path)
        tracker.observe_file(file_meta)
        heartbeat = make_status_heartbeat(
            status_callback,
            source_name=source.name,
            source_path=source_path,
        )
        if heartbeat is not None:
            heartbeat()

        if (
            known_mtimes is not None
            and file_meta.modified_time is not None
            and known_mtimes.get(source_path) == file_meta.modified_time
        ):
            continue

        if blob_store is None:
            from polylogue.paths import blob_store_root

            blob_store = BlobStore(blob_store_root())

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
                tracker.record_failure(file_name=file_meta.name, error=exc)
                logger.warning(
                    "Failed to download Drive payload for %s (%s): %s",
                    file_meta.name,
                    file_meta.file_id,
                    exc,
                )
                continue
            raw_bytes = _inject_live_drive_attachment_bytes(raw_bytes, drive_client, file_meta)
            blob_hash, blob_size = blob_store.write_from_bytes(raw_bytes)
            # Write to cache for future runs
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(raw_bytes)
            del raw_bytes

        provider_hint = Provider.from_string(source.name)
        observe_acquisition(
            observation_callback,
            phase="drive-file-streamed",
            source_path=source_path,
            provider_hint=provider_hint,
            blob_size=blob_size,
            drive_file_id=file_meta.file_id,
            drive_file_name=file_meta.name,
            drive_modified_time=file_meta.modified_time,
            drive_size_bytes=file_meta.size_bytes,
        )
        yield RawSessionData(
            raw_bytes=b"",
            source_path=source_path,
            source_index=None,
            file_mtime=file_meta.modified_time,
            provider_hint=provider_hint,
            blob_hash=blob_hash,
            blob_size=blob_size,
            blob_publication_receipt_id=publication_receipt_id(blob_store, blob_hash),
        )


__all__ = [
    "DriveDownloadResult",
    "download_drive_files",
    "drive_cache_file_path",
    "iter_drive_raw_data",
]

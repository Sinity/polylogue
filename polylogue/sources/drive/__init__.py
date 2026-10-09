from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import ijson

from polylogue.core.compute import DaemonOperationCancelled
from polylogue.core.compute_cancel import check_compute_cancelled
from polylogue.core.enums import Provider
from polylogue.logging import get_logger
from polylogue.storage.blob_publication import publication_receipt_id
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload

from ...config import Source
from ..parsers.base import RawSessionData
from ..source_acquisition_components import (
    ObservationCallback,
    StatusCallback,
    make_status_heartbeat,
    observe_acquisition,
)
from .source import DriveSourceAPI, _parse_modified_time, build_drive_source_client
from .types import DriveConfigLike, DriveFile, DriveUILike
from .witness import DriveListingWitness, drive_cache_directory, drive_source_coordinate

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


def drive_cache_file_path(dest_dir: Path, file_id: str) -> Path:
    """Return the canonical local cache path for a Drive JSON payload."""
    return dest_dir / f"{hashlib.sha256(file_id.encode()).hexdigest()}.json"


def _cache_revision_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.revision")


def _cache_document_is_readable(path: Path) -> bool:
    """Validate every JSON event in a cache without retaining its contents."""
    try:
        with path.open("rb") as handle:
            # EOF is part of the proof; a prefix cannot prove a cache complete.
            events = 0
            for _event in ijson.parse(handle, multiple_values=True):
                check_compute_cancelled()
                events += 1
            return events > 0
    except (OSError, UnicodeDecodeError, ValueError, ijson.JSONError):
        return False


def _replace_atomically(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _cache_revision_matches(path: Path, revision: str | None) -> bool:
    """Only a recorded provider revision can select a cached payload."""
    if revision is None:
        return False
    try:
        if _cache_revision_path(path).read_text(encoding="utf-8") != revision:
            return False
    except OSError:
        return False
    return True


def _cache_holds_readable_revision(path: Path, revision: str | None) -> bool:
    """Whether the recorded revision still has a complete readable cache."""
    return _cache_revision_matches(path, revision) and _cache_document_is_readable(path)


def _write_cache_atomically(path: Path, source: Path, revision: str | None) -> None:
    """Replace the cache, then record the revision it holds.

    The document is written first: a crash between the two writes leaves the
    previous revision recorded against new bytes, which reads as a miss and
    re-downloads, never as a stale hit.
    """
    _cache_revision_path(path).unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            with source.open("rb") as reader:
                while chunk := reader.read(1024 * 1024):
                    check_compute_cancelled()
                    handle.write(chunk)
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if revision is not None:
        _replace_atomically(_cache_revision_path(path), revision.encode("utf-8"))


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
        dest_path = drive_cache_file_path(drive_cache_directory(dest_dir, folder_id), file_id)

        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_to_path(file_id, dest_path)
            downloaded.append(dest_path)
        except DaemonOperationCancelled:
            raise
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
    witness: DriveListingWitness | None = None,
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

    own_witness = witness is None
    witness = witness or DriveListingWitness(source.name, source.folder)
    try:
        witness.enumerate(drive_client, folder_id)
        for file_meta in witness.files():
            cache_dir = drive_cache_directory(source.path or Path(source.name), folder_id)
            cache_path = drive_cache_file_path(cache_dir, file_meta.file_id)
            source_path = drive_source_coordinate(source.name, folder_id, file_meta.file_id)
            tracker.observe_file(file_meta)
            heartbeat = make_status_heartbeat(
                status_callback,
                source_name=source.name,
                source_path=file_meta.name,
            )
            if heartbeat is not None:
                heartbeat()

            if blob_store is None:
                from polylogue.paths import blob_store_root

                blob_store = BlobStore(blob_store_root())

            # Cache identity follows the resolved folder and native file ID. Names
            # are presentation metadata and cannot select another file's bytes.
            blob_hash: str | None = None
            blob_size: int = 0
            cache_exists = cache_path.exists()
            if (
                known_mtimes is not None
                and file_meta.modified_time is not None
                and _parse_modified_time(file_meta.modified_time) is not None
                and _parse_modified_time(known_mtimes.get(source_path)) == _parse_modified_time(file_meta.modified_time)
                and (not cache_exists or _cache_holds_readable_revision(cache_path, file_meta.modified_time))
            ):
                # Unchanged revision with a still-decodable cache: nothing here
                # needs the payload, so nothing here reads it.
                continue

            def checkpoint(status: Callable[[], None] | None = heartbeat) -> None:
                check_compute_cancelled()
                if status is not None:
                    status()

            prepared = None
            try:
                if cache_exists and _cache_revision_matches(cache_path, file_meta.modified_time):
                    try:
                        prepared = blob_store.prepare_from_path(cache_path, heartbeat=checkpoint)
                        # Validate the exact retained copy, not a separately
                        # reopened cache name that a replacement could change.
                        if not _cache_document_is_readable(prepared.temporary_path):
                            blob_store.discard_prepared(prepared)
                            prepared = None
                    except DaemonOperationCancelled:
                        raise
                    except Exception as exc:
                        tracker.record_failure(file_name=file_meta.name, error=exc)
                        witness.record_failure(source_path, "blob", exc)
                        continue
                if prepared is None:
                    try:
                        before = drive_client.get_metadata(file_meta.file_id, refresh=True)
                        prepared = blob_store.prepare_from_writer(
                            partial(drive_client.download_into, file_meta.file_id), heartbeat=checkpoint
                        )
                        after = drive_client.get_metadata(file_meta.file_id, refresh=True)
                        if any(
                            (observed.file_id, observed.mime_type, observed.modified_time, observed.size_bytes)
                            != (file_meta.file_id, file_meta.mime_type, file_meta.modified_time, file_meta.size_bytes)
                            for observed in (before, after)
                        ):
                            witness.changed = True
                            continue
                    except DaemonOperationCancelled:
                        raise
                    except Exception as exc:
                        tracker.record_failure(file_name=file_meta.name, error=exc)
                        witness.record_failure(source_path, "download", exc)
                        logger.warning(
                            "Failed to download Drive payload for %s (%s): %s",
                            file_meta.name,
                            file_meta.file_id,
                            exc,
                        )
                        continue
                    try:
                        _write_cache_atomically(cache_path, prepared.temporary_path, file_meta.modified_time)
                    except DaemonOperationCancelled:
                        raise
                    except Exception as exc:
                        tracker.record_failure(file_name=file_meta.name, error=exc)
                        witness.record_failure(source_path, "cache", exc)
                        continue
                try:
                    blob_hash, blob_size = blob_store.publish_prepared(prepared)
                except DaemonOperationCancelled:
                    raise
                except Exception as exc:
                    tracker.record_failure(file_name=file_meta.name, error=exc)
                    witness.record_failure(source_path, "blob", exc)
                    continue
            finally:
                if prepared is not None:
                    blob_store.discard_prepared(prepared)

            witness.record_acquired_revision(source_path, file_meta.modified_time)
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
                # The native source coordinate is independent of local cache and name.
                canonical_source_path=source_path,
                source_index=None,
                file_mtime=file_meta.modified_time,
                provider_hint=provider_hint,
                blob_hash=blob_hash,
                blob_size=blob_size,
                blob_publication_receipt_id=publication_receipt_id(blob_store, blob_hash),
            )

        witness.reobserve(drive_client)
    finally:
        if own_witness:
            witness.close()


__all__ = [
    "DriveDownloadResult",
    "download_drive_files",
    "drive_cache_file_path",
    "iter_drive_raw_data",
]

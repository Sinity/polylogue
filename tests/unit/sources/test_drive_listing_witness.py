"""Drive enumeration completion requires EOF on every provider page."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from polylogue.config import Source
from polylogue.operations.drive_readiness import DriveCatchupState, inspect_drive_readiness
from polylogue.sources.drive.source_client import DriveSourceClient
from polylogue.sources.drive.types import GEMINI_PROMPT_MIME_TYPE, DriveFile
from polylogue.sources.drive.witness import DriveListingWitness
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root


@pytest.mark.parametrize("count", [0, 3])
def test_listing_witness_pages_the_production_drive_client_before_declaring_complete(
    tmp_path: Path, count: int
) -> None:
    class Gateway:
        calls: list[str | None] = []

        def list_files(self, *, page_token: str | None, **kwargs: Any) -> dict[str, Any]:
            if kwargs.get("fields") == "files(id,name)":
                return {"files": [{"id": "resolved-folder", "name": "configured-folder"}]}
            self.calls.append(page_token)
            ordinal = int(page_token or "0")
            files = (
                []
                if count == 0
                else [
                    {
                        "id": f"id-{ordinal}",
                        "name": "same.json",
                        "mimeType": GEMINI_PROMPT_MIME_TYPE,
                        "modifiedTime": "2026-01-01T00:00:00Z",
                        "size": "1",
                    }
                ]
            )
            result: dict[str, Any] = {"files": files}
            if ordinal + 1 < count:
                result["nextPageToken"] = str(ordinal + 1)
            return result

        def get_file(self, file_id: str, fields: str) -> dict[str, Any]:
            return {
                "id": "resolved-folder",
                "name": "configured-folder",
                "mimeType": "application/vnd.google-apps.folder",
            }

    gateway = Gateway()
    client = DriveSourceClient(gateway=gateway)  # type: ignore[arg-type]
    witness = DriveListingWitness("aistudio", "configured-folder")
    try:
        witness.enumerate(client, "resolved-folder")
        witness.reobserve(client)
        summary = witness.summary()
        assert summary["listed_count"] == count
        assert summary["listing_complete"] is summary["postlisting_complete"] is True
        assert summary["listing_digest"]
        assert summary["postlisting_digest"] == summary["listing_digest"]
        assert summary["postlisted_count"] == count
        assert gateway.calls == ([None] if count == 0 else [None, "1", "2"]) * 2
        assert not client._meta_cache  # Metadata stays on disk, not in a folder-sized Python cache.
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            report = inspect_drive_readiness(
                [Source("aistudio", folder="configured-folder")],
                archive.source_connection,
                archive.index_connection,
                {"aistudio": witness},
            )
        assert report.state is (DriveCatchupState.COMPLETE if count == 0 else DriveCatchupState.PENDING)
        assert report.enumerated_count == count
        assert report.acquired_count == 0
        assert report.materialization_pending == count
    finally:
        witness.close()


def test_interrupted_listing_never_publishes_its_observed_prefix_as_the_denominator(tmp_path: Path) -> None:
    class Client:
        def iter_json_files(self, folder: str) -> Iterator[DriveFile]:
            yield DriveFile("seen", "one", GEMINI_PROMPT_MIME_TYPE, "2026-01-01T00:00:00Z", 1)
            raise OSError("second page unavailable")

    witness = DriveListingWitness("aistudio", "folder")
    try:
        with pytest.raises(OSError):
            witness.enumerate(Client(), "folder")  # type: ignore[arg-type]
        bootstrap_archive_root(tmp_path)
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            report = inspect_drive_readiness(
                [Source("aistudio", folder="folder")],
                archive.source_connection,
                archive.index_connection,
                {"aistudio": witness},
            )
        assert report.state is DriveCatchupState.UNKNOWN
        assert report.enumerated_count is report.acquired_count is report.materialization_pending is None
        assert report.witness[0]["listing_digest"] is None
        assert "drive_listing_witness_unfinished" in report.gaps
    finally:
        witness.close()


@pytest.mark.parametrize("change", ["added", "removed", "revision"])
def test_postlisting_reobserves_native_membership_changes(tmp_path: Path, change: str) -> None:
    class Client:
        files = [DriveFile("one", "name", GEMINI_PROMPT_MIME_TYPE, "2026-01-01T00:00:00Z", 1)]

        def resolve_folder_id(self, folder: str) -> str:
            return folder

        def iter_json_files(self, folder: str) -> Iterator[DriveFile]:
            yield from self.files

    client = Client()
    witness = DriveListingWitness("aistudio", "folder")
    try:
        witness.enumerate(client, "folder")  # type: ignore[arg-type]
        if change == "added":
            client.files.append(DriveFile("two", "name", GEMINI_PROMPT_MIME_TYPE, "2026-01-01T00:00:00Z", 1))
        elif change == "removed":
            client.files.clear()
        else:
            client.files[0].modified_time = "2026-01-02T00:00:00Z"
        witness.reobserve(client)  # type: ignore[arg-type]
        assert witness.changed
        assert witness.postlisting_complete
    finally:
        witness.close()


def test_postlisting_detects_named_folder_resolution_change() -> None:
    class Client:
        resolved = "first-folder"

        def resolve_folder_id(self, folder: str) -> str:
            return self.resolved

        def iter_json_files(self, folder: str) -> Iterator[DriveFile]:
            return iter(())

    client = Client()
    witness = DriveListingWitness("aistudio", "named-folder")
    try:
        witness.enumerate(client, client.resolved)  # type: ignore[arg-type]
        client.resolved = "second-folder"
        witness.reobserve(client)  # type: ignore[arg-type]
        assert witness.changed
        assert witness.postlisting_complete
    finally:
        witness.close()

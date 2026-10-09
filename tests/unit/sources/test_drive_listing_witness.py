"""Drive enumeration completion requires EOF on every provider page."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from polylogue.config import Source
from polylogue.core.compute import DaemonOperationCancelled
from polylogue.operations.drive_readiness import DriveCatchupState
from polylogue.operations.drive_readiness import inspect_drive_readiness as _inspect_drive_readiness
from polylogue.sources.drive.source_client import DriveSourceClient
from polylogue.sources.drive.types import (
    GEMINI_PROMPT_MIME_TYPE,
    DriveAccessDeniedError,
    DriveAuthError,
    DriveFile,
    DriveIncompleteSearchError,
    DriveNotFolderError,
)
from polylogue.sources.drive.witness import DriveListingWitness
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from tests.infra.archive_templates import bootstrap_archive_root


def inspect_drive_readiness(*args: Any, **kwargs: Any) -> Any:
    # These acquisition laws supply settled Raw currency independently. The
    # retained production route exercises the actual classifier in daemon tests.
    return _inspect_drive_readiness(*args, inspect_raw=lambda key: "valid", **kwargs)


@pytest.mark.parametrize("count", [0, 3])
def test_listing_witness_pages_the_production_drive_client_before_declaring_complete(
    tmp_path: Path, count: int
) -> None:
    class Gateway:
        calls: list[str | None] = []

        def list_files(self, *, page_token: str | None, **kwargs: Any) -> dict[str, Any]:
            if kwargs.get("fields") == "incompleteSearch,files(id,name)":
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


@pytest.mark.parametrize("incomplete_call", [1, 2])
def test_provider_incomplete_search_never_certifies_an_empty_folder(tmp_path: Path, incomplete_call: int) -> None:
    """Removing the provider completeness check makes the unknown report complete."""

    class Gateway:
        calls = 0

        def get_file(self, file_id: str, fields: str) -> dict[str, Any]:
            return {"id": file_id, "mimeType": "application/vnd.google-apps.folder"}

        def list_files(self, **kwargs: Any) -> dict[str, Any]:
            assert "incompleteSearch" in kwargs["fields"]
            self.calls += 1
            return {"files": [], "incompleteSearch": self.calls == incomplete_call}

    bootstrap_archive_root(tmp_path)
    client = DriveSourceClient(gateway=Gateway())  # type: ignore[arg-type]
    configured = Source("aistudio", folder="native-folder")
    witness = DriveListingWitness(configured.name, configured.folder or "")
    try:
        with pytest.raises(DriveIncompleteSearchError):
            witness.enumerate(client, client.resolve_folder_id(configured.folder or ""))
            witness.reobserve(client)
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            report = inspect_drive_readiness(
                [configured], archive.source_connection, archive.index_connection, {configured.name: witness}
            )
        assert report.state is DriveCatchupState.UNKNOWN
        assert report.enumerated_count is report.acquired_count is report.materialization_pending is None
        assert "drive_listing_witness_unfinished" in report.gaps
    finally:
        witness.close()


def test_incomplete_named_folder_search_does_not_select_its_first_match() -> None:
    class Gateway:
        def list_files(self, **kwargs: Any) -> dict[str, Any]:
            assert "incompleteSearch" in kwargs["fields"]
            return {"files": [{"id": "unproved-folder", "name": "Named Folder"}], "incompleteSearch": True}

    client = DriveSourceClient(gateway=Gateway())  # type: ignore[arg-type]
    with pytest.raises(DriveIncompleteSearchError):
        client.resolve_folder_id("Named Folder")


def test_restart_discards_failed_listing_and_certifies_a_new_complete_empty_listing(tmp_path: Path) -> None:
    class Gateway:
        incomplete = True

        def get_file(self, file_id: str, fields: str) -> dict[str, Any]:
            return {"id": file_id, "mimeType": "application/vnd.google-apps.folder"}

        def list_files(self, **kwargs: Any) -> dict[str, Any]:
            return {"files": [], "incompleteSearch": self.incomplete}

    bootstrap_archive_root(tmp_path)
    gateway = Gateway()
    client = DriveSourceClient(gateway=gateway)  # type: ignore[arg-type]
    configured = Source("aistudio", folder="native-folder")
    prior = DriveListingWitness(configured.name, configured.folder or "")
    try:
        with pytest.raises(DriveIncompleteSearchError):
            prior.enumerate(client, "native-folder")
    finally:
        prior.close()
    with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
        missing = inspect_drive_readiness([configured], archive.source_connection, archive.index_connection, {})
    assert missing.state is DriveCatchupState.UNKNOWN
    assert "drive_listing_witness_missing" in missing.gaps
    gateway.incomplete = False
    current = DriveListingWitness(configured.name, configured.folder or "")
    try:
        current.enumerate(client, client.resolve_folder_id(configured.folder or ""))
        current.reobserve(client)
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            report = inspect_drive_readiness(
                [configured], archive.source_connection, archive.index_connection, {configured.name: current}
            )
        assert report.state is DriveCatchupState.COMPLETE
        assert report.enumerated_count == report.acquired_count == report.materialization_pending == 0
        assert report.witness[0]["resolved_folder"] == "native-folder"
    finally:
        current.close()


@pytest.mark.parametrize("failure_call", [1, 2])
@pytest.mark.parametrize(
    "failure_type", [OSError, DriveAuthError, DriveAccessDeniedError, DaemonOperationCancelled, DriveNotFolderError]
)
def test_native_lookup_failure_never_substitutes_an_exact_named_empty_folder(
    tmp_path: Path, failure_call: int, failure_type: type[Exception]
) -> None:
    """Restoring failed-ID name fallback certifies the unrelated empty folder."""

    class Gateway:
        gets = 0
        name_lookups = 0

        def get_file(self, file_id: str, fields: str) -> dict[str, Any]:
            self.gets += 1
            if self.gets == failure_call:
                if failure_type is DriveNotFolderError:
                    return {"id": file_id, "mimeType": "application/json"}
                raise failure_type("synthetic native lookup failure")
            return {"id": file_id, "mimeType": "application/vnd.google-apps.folder"}

        def list_files(self, **kwargs: Any) -> dict[str, Any]:
            if "name =" in kwargs["q"]:
                self.name_lookups += 1
                return {"files": [{"id": "unrelated-empty-folder", "name": "native-folder"}]}
            return {"files": [], "incompleteSearch": False}

    bootstrap_archive_root(tmp_path)
    gateway = Gateway()
    client = DriveSourceClient(gateway=gateway)  # type: ignore[arg-type]
    configured = Source("aistudio", folder="native-folder")
    witness = DriveListingWitness(configured.name, configured.folder or "")
    try:
        with pytest.raises(failure_type):
            witness.enumerate(client, client.resolve_folder_id(configured.folder or ""))
            witness.reobserve(client)
        assert gateway.name_lookups == 0
        with ArchiveStore.open_existing(tmp_path, read_only=True) as archive:
            report = inspect_drive_readiness(
                [configured], archive.source_connection, archive.index_connection, {configured.name: witness}
            )
        assert report.state is DriveCatchupState.UNKNOWN
        assert report.enumerated_count is report.acquired_count is report.materialization_pending is None
        assert witness.folder_id in {None, "native-folder"}
    finally:
        witness.close()

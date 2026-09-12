"""Route-neutral provider attachment convergence (polylogue-ck5v)."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from polylogue.core.enums import Provider, Role
from polylogue.operations.attachment_convergence import converge_drive_attachments
from polylogue.sources.drive.types import DriveNotFoundError
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _session(session_id: str, *, upload_origin: str = "drive", file_id: str | None = None) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.GEMINI,
        provider_session_id=session_id,
        messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text="legacy zip")],
        attachments=[
            ParsedAttachment(
                provider_attachment_id=file_id or f"file-{session_id}",
                provider_file_id=file_id or f"file-{session_id}",
                message_provider_id="m0",
                name="legacy.txt",
                mime_type="text/plain",
                upload_origin=upload_origin,
            )
        ],
    )


def _open_index(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def test_legacy_route_attachment_is_backfilled_and_bounded(tmp_path: Path) -> None:
    """A ZIP-restored index row is fetched without Drive iterator enumeration."""
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    session = _session("legacy-zip", file_id="drive-file-1")
    write_parsed_session_to_archive(index, session, raw_id="legacy-zip-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)

    payload = b"bytes restored from Drive after ZIP import"
    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        return payload

    result = converge_drive_attachments(
        index,
        source,
        archive_root=tmp_path,
        download_bytes=fetch,
        limit=1,
    )

    row = index.execute("SELECT blob_hash, byte_count, acquisition_status FROM attachments").fetchone()
    source_ref = source.execute(
        "SELECT blob_hash, ref_id, ref_type, size_bytes FROM blob_refs WHERE ref_type = 'attachment'"
    ).fetchone()
    assert result.acquired == 1
    assert result.complete
    assert calls == ["drive-file-1"]
    assert row["acquisition_status"] == "acquired"
    assert row["byte_count"] == len(payload)
    assert bytes(row["blob_hash"]) == hashlib.sha256(payload).digest()
    assert bytes(source_ref["blob_hash"]) == hashlib.sha256(payload).digest()
    assert source_ref["ref_id"] == "legacy-zip-raw"
    assert source_ref["size_bytes"] == len(payload)
    index.close()
    source.close()


def test_attachment_convergence_records_debt_for_the_next_bounded_window(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    write_parsed_session_to_archive(index, _session("legacy-one", file_id="drive-file-1"), raw_id="raw-1")
    write_parsed_session_to_archive(index, _session("legacy-two", file_id="drive-file-2"), raw_id="raw-2")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)

    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        return file_id.encode()

    result = converge_drive_attachments(
        index,
        source,
        archive_root=tmp_path,
        download_bytes=fetch,
        limit=1,
    )

    statuses = index.execute(
        "SELECT acquisition_status, COUNT(*) FROM attachments GROUP BY acquisition_status"
    ).fetchall()
    assert result.inspected == 1
    assert result.acquired == 1
    assert result.deferred == 1
    assert not result.complete
    assert len(calls) == 1
    assert dict(statuses) == {"acquired": 1, "unfetched": 1}
    index.close()
    source.close()


def test_attachment_convergence_terminal_failure_does_not_fabricate_bytes(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    session = _session("gone", file_id="deleted-file")
    write_parsed_session_to_archive(index, session, raw_id="gone-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)

    result = converge_drive_attachments(
        index,
        source,
        archive_root=tmp_path,
        download_bytes=lambda _file_id: (_ for _ in ()).throw(DriveNotFoundError("deleted")),
    )

    row = index.execute("SELECT blob_hash, byte_count, acquisition_status FROM attachments").fetchone()
    assert result.terminal == 1
    assert result.complete
    assert row["acquisition_status"] == "unavailable"
    assert row["blob_hash"] is None
    assert row["byte_count"] == 0
    index.close()
    source.close()

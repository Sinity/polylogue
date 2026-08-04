"""Focused contracts for Drive ingestion helpers and live attachment acquisition.

`_apply_drive_attachments`/`iter_drive_sessions` (the decoupled, dead
local-path-writing attachment path) were removed as part of polylogue-83u.2:
they had zero live callers and wrote to `attachment.path` instead of
`inline_bytes`, so acquired Drive attachment bytes never reached the blob
store. The live path is `iter_drive_raw_data`, which now resolves
Drive-hosted attachment references (`driveDocument`/`driveImage`/etc.) via the
same live client used to download the session document, injecting fetched
bytes into the raw payload before it is cached/blob-stored. The tests below
exercise that live path end to end through the ordinary parse+write pipeline,
proving `acquisition_status='acquired'` with a blob at the attachment's true
SHA-256 (AC#1, Drive sub-case), and that a fetch failure leaves the attachment
honestly `unfetched` rather than fabricating a hash.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
from polylogue.config import Source
from polylogue.core.enums import Provider
from polylogue.core.json import JSONValue
from polylogue.pipeline.ids import session_content_hash
from polylogue.pipeline.ids import session_id as make_session_id
from polylogue.pipeline.services.ingest_worker import SessionWritePayload
from polylogue.sources import DriveFile, download_drive_files
from polylogue.sources.dispatch import parse_payload
from polylogue.sources.drive import iter_drive_raw_data
from polylogue.sources.parsers.base import ParsedSession
from polylogue.storage.blob_publication import ArchiveBlobPublisher
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.cursor_state import CursorStatePayload
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.connection import open_connection


@dataclass
class _DriveSessionClient:
    """Minimal `DriveSourceAPI` stub covering the live raw-acquisition path."""

    files: list[DriveFile]
    payload_bytes: dict[str, bytes]
    attachment_bytes: dict[str, bytes] = field(default_factory=dict)
    attachment_failures: dict[str, Exception] = field(default_factory=dict)
    download_bytes_calls: list[str] = field(default_factory=list)

    def resolve_folder_id(self, folder_ref: str) -> str:
        return f"folder:{folder_ref}"

    def iter_json_files(self, folder_id: str) -> Iterable[DriveFile]:
        yield from self.files

    def download_json_payload(self, file_id: str, *, name: str) -> JSONValue:
        raise NotImplementedError("iter_drive_raw_data uses download_bytes, not download_json_payload")

    def download_to_path(self, file_id: str, dest: Path) -> DriveFile:
        raise NotImplementedError("not used by the live raw-acquisition path")

    def download_bytes(self, file_id: str) -> bytes:
        self.download_bytes_calls.append(file_id)
        if file_id in self.attachment_failures:
            raise self.attachment_failures[file_id]
        if file_id in self.attachment_bytes:
            return self.attachment_bytes[file_id]
        return self.payload_bytes[file_id]


def _empty_cursor_state() -> CursorStatePayload:
    return {}


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _preacquired(store: BlobStore, session: ParsedSession) -> dict[int, tuple[bytes | None, int, str]]:
    acquired: dict[int, tuple[bytes | None, int, str]] = {}
    for attachment in session.attachments:
        if attachment.inline_bytes is None:
            continue
        blob_hash, size = store.write_from_bytes(attachment.inline_bytes)
        acquired[id(attachment)] = (bytes.fromhex(blob_hash), size, "acquired")
    return acquired


def test_download_drive_files_contract(tmp_path: Path) -> None:
    from unittest.mock import MagicMock

    client = MagicMock()
    client.iter_json_files.return_value = [
        DriveFile("good", "session", "application/json", None, None),
        DriveFile("bad", "broken.jsonl", "application/json", None, None),
    ]

    def download(file_id: str, dest: Path) -> None:
        if file_id == "bad":
            raise PermissionError("denied")
        dest.write_bytes(b'{"id":"good"}')

    client.download_to_path.side_effect = download

    result = download_drive_files(client, "folder-1", tmp_path)

    assert result.total_files == 2
    assert [path.name for path in result.downloaded_files] == ["session.json"]
    assert result.downloaded_files[0].read_bytes() == b'{"id":"good"}'
    assert result.failed_files == [{"file_id": "bad", "name": "broken.jsonl", "error": "denied"}]


def test_iter_drive_raw_data_injects_live_attachment_bytes_into_raw_payload(tmp_path: Path) -> None:
    """Drive-hosted attachment bytes are fetched INSIDE the live client's
    iterator scope and land in the stored raw payload as a base64 sidecar."""
    payload = {
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "Hi"},
                {
                    "role": "model",
                    "text": "Here is the file",
                    "driveDocument": {"id": "att-1", "name": "doc.txt", "mimeType": "text/plain"},
                },
            ]
        }
    }
    client = _DriveSessionClient(
        files=[DriveFile("file-1", "chat.json", "application/json", "2025-01-01T00:00:00Z", 10)],
        payload_bytes={"file-1": json.dumps(payload).encode("utf-8")},
        attachment_bytes={"att-1": b"the actual drive attachment bytes"},
    )
    blob_store = BlobStore(tmp_path / "blob")
    cursor_state = _empty_cursor_state()

    records = list(
        iter_drive_raw_data(
            source=Source(name="gemini", folder="Google AI Studio", path=tmp_path),
            client=client,
            cursor_state=cursor_state,
            blob_store=blob_store,
        )
    )

    assert len(records) == 1
    assert client.download_bytes_calls == ["file-1", "att-1"]
    assert records[0].blob_hash is not None
    stored_bytes = blob_store.read_all(records[0].blob_hash)
    stored_payload = json.loads(stored_bytes)
    stored_doc = stored_payload["chunkedPrompt"]["chunks"][1]["driveDocument"]
    assert stored_doc["id"] == "att-1"
    assert stored_doc["name"] == "doc.txt"

    # The cache file written for future runs carries the same injected bytes.
    cache_path = Path(records[0].source_path)
    assert json.loads(cache_path.read_bytes()) == stored_payload


def test_iter_drive_raw_data_backfills_a_preexisting_cache_file(tmp_path: Path) -> None:
    """CodeRabbit/Codex P2: a local cache file predating live attachment
    acquisition (or written by an older polylogue version) must not be
    permanently skipped just because the top-level document doesn't need
    re-download. The injector must run on cache hits too."""
    payload = {
        "chunkedPrompt": {
            "chunks": [
                {
                    "role": "model",
                    "driveDocument": {"id": "att-1", "name": "doc.txt", "mimeType": "text/plain"},
                },
            ]
        }
    }
    client = _DriveSessionClient(
        files=[DriveFile("file-1", "chat.json", "application/json", "2025-01-01T00:00:00Z", 10)],
        payload_bytes={"file-1": json.dumps(payload).encode("utf-8")},
        attachment_bytes={"att-1": b"backfilled attachment bytes"},
    )
    blob_store = BlobStore(tmp_path / "blob")

    # Pre-populate the cache file exactly as an older run (with no live
    # attachment acquisition) would have left it -- the raw payload, no
    # sidecar, and the top-level document is never re-downloaded because it
    # already exists on disk.
    cache_path = Path(
        list(
            iter_drive_raw_data(
                source=Source(name="gemini", folder="Google AI Studio", path=tmp_path),
                client=_DriveSessionClient(
                    files=client.files,
                    payload_bytes=client.payload_bytes,
                    attachment_failures={"att-1": RuntimeError("attachment not fetchable yet")},
                ),
                blob_store=BlobStore(tmp_path / "throwaway-blob"),
            )
        )[0].source_path
    )
    stale_cache_bytes = cache_path.read_bytes()
    assert b"att-1" in stale_cache_bytes
    assert b"backfilled" not in stale_cache_bytes  # sidecar was never injected

    records = list(
        iter_drive_raw_data(
            source=Source(name="gemini", folder="Google AI Studio", path=tmp_path),
            client=client,
            blob_store=blob_store,
        )
    )

    assert len(records) == 1
    # The top-level document was NOT re-downloaded (cache hit) -- only the
    # attachment was fetched.
    assert client.download_bytes_calls == ["att-1"]
    assert records[0].blob_hash is not None
    resolved_payload = json.loads(blob_store.read_all(records[0].blob_hash))
    resolved_doc = resolved_payload["chunkedPrompt"]["chunks"][0]["driveDocument"]
    assert resolved_doc["id"] == "att-1"
    # The cache file itself is refreshed with the backfilled sidecar so the
    # NEXT run also sees it without re-fetching.
    assert json.loads(cache_path.read_bytes()) == resolved_payload


def test_drive_live_attachment_bytes_reach_acquired_blob_with_true_hash(tmp_path: Path) -> None:
    """End-to-end: live client -> raw injection -> ordinary parse -> write.

    Proves polylogue-83u.2 AC#1 for the Drive sub-case: a seeded fixture with
    a live Drive-hosted attachment reference produces
    acquisition_status='acquired' with a blob at the attachment's true
    SHA-256, using the same generic dispatch (`parse_payload`) and write
    (`write_parsed_session_to_archive`) path the daemon uses in production.
    """
    attachment_bytes = b"the actual drive attachment bytes"
    payload = {
        "title": "Drive Chat",
        "chunkedPrompt": {
            "chunks": [
                {"role": "user", "text": "Hi"},
                {
                    "role": "model",
                    "text": "Here is the file",
                    "driveDocument": {"id": "att-1", "name": "doc.txt", "mimeType": "text/plain"},
                },
            ]
        },
    }
    client = _DriveSessionClient(
        files=[DriveFile("file-1", "chat.json", "application/json", "2025-01-01T00:00:00Z", 10)],
        payload_bytes={"file-1": json.dumps(payload).encode("utf-8")},
        attachment_bytes={"att-1": attachment_bytes},
    )
    blob_store = BlobStore(tmp_path / "blob")

    records = list(
        iter_drive_raw_data(
            source=Source(name="gemini", folder="Google AI Studio", path=tmp_path),
            client=client,
            blob_store=blob_store,
        )
    )
    assert len(records) == 1
    assert records[0].blob_hash is not None
    resolved_payload = json.loads(blob_store.read_all(records[0].blob_hash))

    # The subprocess parse stage has no live client — it only ever sees the
    # already-resolved raw payload, dispatched generically like production.
    sessions = parse_payload("gemini", resolved_payload, "fallback-id")
    assert len(sessions) == 1
    session = sessions[0]
    assert len(session.attachments) == 1
    attachment = session.attachments[0]
    assert attachment.upload_origin == "drive"
    assert attachment.inline_bytes == attachment_bytes

    conn = _connect(tmp_path / "index.db")
    write_parsed_session_to_archive(conn, session, preacquired_attachment_blobs=_preacquired(blob_store, session))

    row = conn.execute("SELECT blob_hash, byte_count, acquisition_status FROM attachments").fetchone()
    assert row["acquisition_status"] == "acquired"
    assert row["byte_count"] == len(attachment_bytes)
    assert bytes(row["blob_hash"]) == hashlib.sha256(attachment_bytes).digest()
    assert blob_store.read_all(hashlib.sha256(attachment_bytes).hexdigest()) == attachment_bytes


def test_drive_attachment_fetch_failure_stays_honestly_unfetched(tmp_path: Path) -> None:
    """A Drive attachment whose live fetch fails is NOT acquired and carries
    no synthetic hash — it stays `unfetched`, same as a genuinely-unfetchable
    handle (source_url-only). This is the negative-path complement to the
    acquired-blob test above."""
    payload = {
        "chunkedPrompt": {
            "chunks": [
                {
                    "role": "model",
                    "driveDocument": {"id": "att-dead", "name": "gone.bin", "mimeType": "application/octet-stream"},
                },
            ]
        }
    }
    client = _DriveSessionClient(
        files=[DriveFile("file-1", "chat.json", "application/json", None, 10)],
        payload_bytes={"file-1": json.dumps(payload).encode("utf-8")},
        attachment_failures={"att-dead": RuntimeError("file no longer accessible")},
    )
    blob_store = BlobStore(tmp_path / "blob")

    records = list(
        iter_drive_raw_data(
            source=Source(name="gemini", folder="Google AI Studio", path=tmp_path),
            client=client,
            blob_store=blob_store,
        )
    )
    assert records[0].blob_hash is not None
    resolved_payload = json.loads(blob_store.read_all(records[0].blob_hash))
    sessions = parse_payload("gemini", resolved_payload, "fallback-id")
    session = sessions[0]
    attachment = session.attachments[0]
    assert attachment.inline_bytes is None

    conn = _connect(tmp_path / "index.db")
    write_parsed_session_to_archive(conn, session)

    row = conn.execute("SELECT blob_hash, acquisition_status FROM attachments").fetchone()
    assert row["acquisition_status"] == "unfetched"
    assert row["blob_hash"] is None


def _write_via_ingest_batch(
    *,
    conn: sqlite3.Connection,
    source_conn: sqlite3.Connection,
    blob_publisher: ArchiveBlobPublisher,
    session: ParsedSession,
    raw_id: str,
) -> None:
    payload = SessionWritePayload(
        session_id=str(make_session_id(session.source_name, session.provider_session_id)),
        content_hash=session_content_hash(session),
        parsed_session=session,
        message_count=len(session.messages),
        attachment_count=len(session.attachments),
        raw_id=raw_id,
    )
    changed, _ = ingest_batch_core._write_session(conn, payload, blob_publisher=blob_publisher, source_conn=source_conn)
    assert changed is True
    conn.commit()


def test_iter_drive_raw_data_two_revision_fixture_gets_real_lineage(tmp_path: Path) -> None:
    """polylogue-sp72 AC1: thread two REAL ``iter_drive_raw_data`` acquisition
    passes (the exact live-attachment-backfill shape, not a hand-crafted
    byte-append fixture) all the way through the actual production write
    path (``ingest_batch._core._write_session``) and inspect what
    ``raw_sessions`` lineage they end up with.

    First pass: the Drive-hosted attachment reference fails to fetch, so the
    cache file is written with the plain downloaded JSON. Second pass: the
    same file, now with the attachment fetchable -- ``iter_drive_raw_data``
    backfills the resolved bytes into the cached JSON and re-serializes the
    whole document (``_inject_live_drive_attachment_bytes``), producing a
    brand-new raw for the SAME logical session, exactly the polylogue-sp72
    problem shape.

    Both raws MUST carry a real, non-NULL ``logical_source_key`` (the
    concrete, durable fix from #3656: they are no longer invisible to
    ``raw_revision_heads``/future arbitration) and ``revision_kind='full'``
    (never the pre-fix ``'unknown'``).

    A full JSON re-serialization that injects the resolved attachment bytes
    into the MIDDLE of the document (not appended at the very end) is NOT a
    byte-prefix superset of the original, so the byte-prefix classifier
    (``classify_historical_full_revision_streams``,
    ``archive/revision_authority.py``) alone lands
    ``revision_authority='quarantined'`` with no predecessor link for this
    exact shape (that residual gap was #3656's documented scope boundary,
    deferred to polylogue-1fijp). polylogue-1fijp AC (b) closes it: this
    fixture's second raw is exactly the shape
    ``sources.drive.structural_diff.classify_drive_structural_relation``
    exists to recognize -- the first raw's ``driveDocument`` dict is a
    structural subset of the second's (same keys/values, plus the injected
    fetch-data field) -- so ``_bind_drive_revision_lineage``'s new
    structural-growth pre-check now binds the second raw directly as a
    ``FULL``/``ASSERTED`` revision with a real ``predecessor_raw_id``
    pointing at the first raw, instead of falling through to the
    byte-prefix quarantine path.
    """
    payload = {
        "chunkedPrompt": {
            "chunks": [
                {
                    "role": "model",
                    "driveDocument": {"id": "att-1", "name": "doc.txt", "mimeType": "text/plain"},
                },
            ]
        }
    }
    client = _DriveSessionClient(
        files=[DriveFile("file-1", "chat.json", "application/json", "2025-01-01T00:00:00Z", 10)],
        payload_bytes={"file-1": json.dumps(payload).encode("utf-8")},
        attachment_failures={"att-1": RuntimeError("not yet fetchable")},
    )
    acquire_blob_store = BlobStore(tmp_path / "acquire-blob")

    first_records = list(
        iter_drive_raw_data(
            source=Source(name="gemini", folder="Google AI Studio", path=tmp_path),
            client=client,
            blob_store=acquire_blob_store,
        )
    )
    assert len(first_records) == 1
    first_bytes = acquire_blob_store.read_all(first_records[0].blob_hash)  # type: ignore[arg-type]

    fetchable_client = _DriveSessionClient(
        files=client.files,
        payload_bytes=client.payload_bytes,
        attachment_bytes={"att-1": b"the actual drive attachment bytes"},
    )
    second_records = list(
        iter_drive_raw_data(
            source=Source(name="gemini", folder="Google AI Studio", path=tmp_path),
            client=fetchable_client,
            blob_store=acquire_blob_store,
        )
    )
    assert len(second_records) == 1
    second_bytes = acquire_blob_store.read_all(second_records[0].blob_hash)  # type: ignore[arg-type]
    assert second_bytes != first_bytes
    # Confirms the real mechanism's shape: the backfill re-serializes the
    # WHOLE document (injecting resolved bytes into the middle), it does not
    # append at the end -- so the second raw is not a literal byte-prefix
    # continuation of the first.
    assert not second_bytes.startswith(first_bytes)

    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)
    source_db_path = archive_root / "source.db"
    blob_publisher = ArchiveBlobPublisher(source_db_path, archive_root / "blob")

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        first_raw_id = archive.write_raw_payload(
            provider=Provider.GEMINI,
            payload=first_bytes,
            source_path=first_records[0].source_path,
            acquired_at_ms=1_767_000_000_000,
        )
        second_raw_id = archive.write_raw_payload(
            provider=Provider.GEMINI,
            payload=second_bytes,
            source_path=second_records[0].source_path,
            acquired_at_ms=1_767_000_000_500,
        )
    assert first_raw_id != second_raw_id

    first_sessions = parse_payload("gemini", json.loads(first_bytes), "fallback-id")
    second_sessions = parse_payload("gemini", json.loads(second_bytes), "fallback-id")
    assert len(first_sessions) == 1
    assert len(second_sessions) == 1

    with (
        open_connection(archive_root / "index.db") as conn,
        sqlite3.connect(str(source_db_path)) as source_conn,
    ):
        _write_via_ingest_batch(
            conn=conn,
            source_conn=source_conn,
            blob_publisher=blob_publisher,
            session=first_sessions[0],
            raw_id=first_raw_id,
        )
        _write_via_ingest_batch(
            conn=conn,
            source_conn=source_conn,
            blob_publisher=blob_publisher,
            session=second_sessions[0],
            raw_id=second_raw_id,
        )

    with sqlite3.connect(str(source_db_path)) as verify_conn:
        verify_conn.row_factory = sqlite3.Row
        first_row = verify_conn.execute(
            "SELECT logical_source_key, revision_kind FROM raw_sessions WHERE raw_id = ?",
            (first_raw_id,),
        ).fetchone()
        second_row = verify_conn.execute(
            """
            SELECT logical_source_key, revision_kind, predecessor_raw_id, revision_authority
            FROM raw_sessions WHERE raw_id = ?
            """,
            (second_raw_id,),
        ).fetchone()

    provider_session_id = first_sessions[0].provider_session_id
    expected_logical_source_key = f"{Provider.GEMINI.value}:{provider_session_id}"
    # The durable fix: both raws now carry the real identity key and a typed
    # revision_kind, no longer NULL/'unknown'.
    assert first_row["logical_source_key"] == expected_logical_source_key
    assert second_row["logical_source_key"] == expected_logical_source_key
    assert first_row["revision_kind"] == "full"
    assert second_row["revision_kind"] == "full"
    # polylogue-1fijp AC (b): the JSON-structural-diff classifier proves the
    # second raw is a genuine structural extension of the first (the
    # attachment-injected chunk's dict grew a key, nothing existing changed
    # or was removed), so it gets typed, predecessor-linked lineage --
    # never the pre-fix quarantined-unknown outcome, and no longer even the
    # #3656 interim outcome (typed identity, but no predecessor) for this
    # exact shape.
    assert second_row["predecessor_raw_id"] == first_raw_id
    assert second_row["revision_authority"] == "asserted"

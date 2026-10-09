"""Attachment reacquisition across capture revisions (polylogue-4zqh3).

An upload-only claude.ai ``files`` reference records a name and a size but no
bytes, so its first ingest is honestly ``unfetched``. When a later revision of
the same capture carries the payload as ``extracted_content``, the bytes must
land as an ``acquired`` payload version under the same message reference.
The unfetched descriptor row is swept, so no original reference is stranded.

Anti-vacuity: drop ``extracted_content`` from the ``files`` branch of
``attachment_from_meta`` and the second revision stays ``unfetched``; fold
payload-version publication without relinking and sweeping the metadata row
would grow the row count or strand the original reference.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider, Role
from polylogue.sources.parsers.base import ParsedSession
from polylogue.sources.parsers.claude import parse_ai
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from tests.infra.index_writer import write_fixture_index_session

SESSION_UUID = "reacquisition-session"
FILE_UUID = "upload-only-file"
PAYLOAD = "restored attachment payload\n"
PAYLOAD_BYTES = PAYLOAD.encode("utf-8")


def _capture(*, extracted_content: str | None) -> dict[str, object]:
    """One claude.ai capture revision carrying a single upload-only reference."""
    file_record: dict[str, object] = {
        "file_uuid": FILE_UUID,
        "uuid": FILE_UUID,
        "file_kind": "blob",
        "file_name": "restored.md",
        "size_bytes": len(PAYLOAD_BYTES),
        "path": "/mnt/user-data/uploads/restored.md",
        "success": True,
    }
    if extracted_content is not None:
        file_record["extracted_content"] = extracted_content
    return {
        "uuid": SESSION_UUID,
        "name": "Attachment reacquisition",
        "chat_messages": [
            {
                "uuid": "m0",
                "sender": "human",
                "text": "Please read this.",
                "files": [file_record],
            }
        ],
    }


def _connect(path: Path) -> sqlite3.Connection:
    conn = connect_measured(path)
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


def _attachment_state(conn: sqlite3.Connection) -> sqlite3.Row:
    row: sqlite3.Row | None = conn.execute(
        "SELECT attachment_id, display_name, byte_count, blob_hash, acquisition_status FROM attachments"
    ).fetchone()
    assert row is not None
    return row


def _ref_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM attachment_refs").fetchone()[0])


def test_upload_only_reference_gains_bytes_at_a_stable_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BlobStore(tmp_path / "blob")
    monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: store)
    conn = _connect(tmp_path / "index.db")

    before = parse_ai(_capture(extracted_content=None), "fallback")
    write_fixture_index_session(conn, before, preacquired_attachment_blobs=_preacquired(store, before))

    unfetched = _attachment_state(conn)
    assert unfetched["acquisition_status"] == "unfetched"
    assert unfetched["blob_hash"] is None
    assert unfetched["byte_count"] == len(PAYLOAD_BYTES)
    identity = str(unfetched["attachment_id"])
    reference = conn.execute("SELECT ref_id FROM attachment_refs").fetchone()[0]
    assert _ref_count(conn) == 1

    after = parse_ai(_capture(extracted_content=PAYLOAD), "fallback")
    write_fixture_index_session(conn, after, preacquired_attachment_blobs=_preacquired(store, after))

    acquired = _attachment_state(conn)
    assert str(acquired["attachment_id"]) != identity
    assert conn.execute("SELECT ref_id FROM attachment_refs").fetchone()[0] == reference
    assert conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0] == 1
    assert acquired["acquisition_status"] == "acquired"
    assert bytes(acquired["blob_hash"]) == hashlib.sha256(PAYLOAD_BYTES).digest()
    assert acquired["byte_count"] == len(PAYLOAD_BYTES)
    assert store.read_all(hashlib.sha256(PAYLOAD_BYTES).hexdigest()) == PAYLOAD_BYTES
    assert _ref_count(conn) == 1


def test_replaying_the_acquired_revision_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = BlobStore(tmp_path / "blob")
    monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: store)
    conn = _connect(tmp_path / "index.db")

    for _ in range(2):
        session = parse_ai(_capture(extracted_content=PAYLOAD), "fallback")
        write_fixture_index_session(conn, session, preacquired_attachment_blobs=_preacquired(store, session))

    acquired = _attachment_state(conn)
    assert acquired["acquisition_status"] == "acquired"
    assert bytes(acquired["blob_hash"]) == hashlib.sha256(PAYLOAD_BYTES).digest()
    assert _ref_count(conn) == 1


@pytest.mark.parametrize("metadata_first", [False, True])
def test_metadata_replay_does_not_borrow_another_captures_acquired_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata_first: bool
) -> None:
    store = BlobStore(tmp_path / "blob")
    monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: store)
    conn = _connect(tmp_path / "index.db")
    acquired_capture = _capture(extracted_content=PAYLOAD)
    metadata_capture = _capture(extracted_content=None)
    for capture in (acquired_capture, metadata_capture):
        messages = capture["chat_messages"]
        assert isinstance(messages, list)
        messages[0]["files"][0]["size_bytes"] = 10_000_000
    metadata_capture["uuid"] = "metadata-session"
    metadata_messages = metadata_capture["chat_messages"]
    assert isinstance(metadata_messages, list)
    metadata_messages[0]["uuid"] = "metadata-message"
    acquired = parse_ai(acquired_capture, "fallback")
    metadata = parse_ai(metadata_capture, "fallback")
    first, second = (metadata, acquired) if metadata_first else (acquired, metadata)
    try:
        for session in (first, second, metadata, metadata):
            write_fixture_index_session(conn, session, preacquired_attachment_blobs=_preacquired(store, session))
        retained = conn.execute(
            "SELECT a.byte_count,a.blob_hash,a.acquisition_status FROM attachments a "
            "JOIN attachment_refs r ON r.attachment_id=a.attachment_id WHERE r.session_id='claude-ai-export:reacquisition-session'"
        ).fetchone()
        unmeasured = conn.execute(
            "SELECT a.byte_count,a.blob_hash,a.acquisition_status FROM attachments a "
            "JOIN attachment_refs r ON r.attachment_id=a.attachment_id WHERE r.session_id='claude-ai-export:metadata-session'"
        ).fetchone()
        digest = hashlib.sha256(PAYLOAD_BYTES).digest()
        assert conn.execute("SELECT count(*) FROM attachments").fetchone()[0] == 2
        assert unmeasured is not None
        assert tuple(unmeasured) == (10_000_000, None, "unfetched")
        assert retained["byte_count"] == len(PAYLOAD_BYTES)
        assert bytes(retained["blob_hash"]) == digest
        assert retained["acquisition_status"] == "acquired"
        assert store.read_all(digest.hex()) == PAYLOAD_BYTES
        assert store.blob_path(digest.hex()).stat().st_size == len(PAYLOAD_BYTES)
        assert _ref_count(conn) == 2
    finally:
        conn.close()


def test_colliding_native_attachment_insertion_keeps_the_existing_reference(tmp_path: Path) -> None:
    from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage
    from polylogue.storage.sqlite.archive_tiers.write import _attachment_id

    attachments = [
        ParsedAttachment(provider_attachment_id=native_id, message_provider_id="m", name=native_id)
        for native_id in ("cert-collision-50449", "cert-collision-111329")
    ]
    inserted, original = sorted(attachments, key=lambda item: _attachment_id("", item))
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="stable-collision",
        messages=[ParsedMessage(provider_message_id="m", role=Role.USER, text="files")],
        attachments=[original],
    )
    conn = _connect(tmp_path / "index.db")
    try:
        write_fixture_index_session(conn, session, raw_id="raw-original")

        def reference(native_id: str) -> tuple[str, str]:
            return tuple(
                conn.execute(
                    "SELECT r.ref_id, r.supplying_raw_id FROM attachment_refs r "
                    "JOIN attachment_native_ids n ON n.ref_id=r.ref_id "
                    "WHERE n.id_kind='attachment' AND n.native_id=?",
                    (native_id,),
                ).fetchone()
            )

        before = reference(original.provider_attachment_id)
        write_fixture_index_session(
            conn,
            session.model_copy(update={"attachments": [original, inserted]}),
            raw_id="raw-expanded",
            force_replace=True,
        )
        after = reference(original.provider_attachment_id)
        assert after[0] == before[0]
        assert reference(inserted.provider_attachment_id)[0] != before[0]
        assert after[1] == "raw-expanded"
        write_fixture_index_session(
            conn,
            session.model_copy(update={"attachments": [inserted, original]}),
            raw_id="raw-reordered",
            force_replace=True,
        )
        assert reference(original.provider_attachment_id)[0] == before[0]
    finally:
        conn.close()


@pytest.mark.parametrize(
    "native_ids",
    [
        (" ", "  "),
        ("opaque:a", "opaque:b"),
        ("\ud83d\ude00", "😀"),
        ("\ud800", "\ufffd"),
    ],
)
def test_source_native_identity_is_injective_in_the_actual_writer(tmp_path: Path, native_ids: tuple[str, str]) -> None:
    from polylogue.core.identity_law import attachment_native_identity
    from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage

    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="exact-native",
        messages=[ParsedMessage(provider_message_id="m", role=Role.USER, text="files")],
        attachments=[
            ParsedAttachment(provider_attachment_id=native_id, message_provider_id="m", name="file")
            for native_id in native_ids
        ],
    )
    conn = _connect(tmp_path / "index.db")
    try:
        write_fixture_index_session(conn, session, raw_id="raw-exact")
        rows = conn.execute(
            "SELECT ref_id,native_identity,supplying_raw_id FROM attachment_refs ORDER BY native_identity"
        ).fetchall()
        assert len(rows) == 2
        assert len({row["ref_id"] for row in rows}) == 2
        assert {row["native_identity"] for row in rows} == {
            attachment_native_identity(native_id) for native_id in native_ids
        }
        assert {bytes.fromhex(row["native_identity"]).decode("utf-8", errors="surrogatepass") for row in rows} == set(
            native_ids
        )
        assert {row["supplying_raw_id"] for row in rows} == {"raw-exact"}
    finally:
        conn.close()


def test_optional_native_enrichment_does_not_rename_the_attachment_reference(tmp_path: Path) -> None:
    from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage

    original = ParsedAttachment(provider_attachment_id="opaque-native", message_provider_id="m", name="before")
    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="native-enrichment",
        messages=[ParsedMessage(provider_message_id="m", role=Role.USER, text="file")],
        attachments=[original],
    )
    conn = _connect(tmp_path / "index.db")
    try:
        write_fixture_index_session(conn, session, raw_id="raw-before")
        ref_id = conn.execute("SELECT ref_id FROM attachment_refs").fetchone()[0]
        enriched = original.model_copy(
            update={"provider_file_id": "file", "provider_drive_id": "drive", "name": "renamed"}
        )
        write_fixture_index_session(
            conn, session.model_copy(update={"attachments": [enriched]}), raw_id="raw-enriched", force_replace=True
        )
        rows = conn.execute("SELECT ref_id,supplying_raw_id FROM attachment_refs").fetchall()
        assert [tuple(row) for row in rows] == [(ref_id, "raw-enriched")]
        assert conn.execute("SELECT display_name FROM attachments").fetchone()[0] == "renamed"
    finally:
        conn.close()


def test_missing_native_attachment_identity_is_a_visible_refusal(tmp_path: Path) -> None:
    from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage

    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="missing-native",
        messages=[ParsedMessage(provider_message_id="m", role=Role.USER, text="file")],
        attachments=[ParsedAttachment(provider_attachment_id="", message_provider_id="m")],
    )
    conn = _connect(tmp_path / "index.db")
    try:
        with pytest.raises(ValueError, match="declared native identity"):
            write_fixture_index_session(conn, session, raw_id="raw-missing")
        assert conn.execute("SELECT COUNT(*) FROM attachment_refs").fetchone()[0] == 0
    finally:
        conn.close()


@pytest.mark.parametrize("payload", [None, b"AA"])
@pytest.mark.parametrize("native_ids", [("cert-collision-50449", "cert-collision-111329"), ("\ud83d\ude00", "😀")])
def test_actual_attachment_publication_and_index_share_exact_native_references(
    tmp_path: Path, native_ids: tuple[str, str], payload: bytes | None
) -> None:
    import json

    from polylogue.material_protocol.v1 import RevisionManifest, decode_session_revision, verify_revision
    from polylogue.sinex.material_adapter import encode_parsed_session_publication
    from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage

    session = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="s1",
        messages=[ParsedMessage(provider_message_id="m", role=Role.USER, text="files")],
        attachments=[
            ParsedAttachment(
                provider_attachment_id=native_id,
                message_provider_id="m",
                name="file",
                inline_bytes=payload,
                provider_file_id="file",
                provider_drive_id="drive",
                size_bytes=999,
            )
            for native_id in native_ids
        ],
    )
    conn = _connect(tmp_path / "index.db")
    try:
        session_id = write_fixture_index_session(
            conn,
            session,
            raw_id="raw-wire",
            preacquired_attachment_blobs=_preacquired(BlobStore(tmp_path / "blob"), session),
        )
        publication = encode_parsed_session_publication(session, session_id=session_id)
        manifest = RevisionManifest.from_dict(json.loads(publication.manifest_bytes))
        names = dict(publication.segments)
        segments = {
            descriptor.index: names[descriptor.filename] for descriptor in (*manifest.segments, manifest.head_segment)
        }
        verify_revision(manifest, segments)
        decoded = decode_session_revision(manifest, segments)
        wire_refs = {
            (attachment["record_id"], attachment["native_identity"], attachment["attachment_id"])
            for message in decoded.messages
            for attachment in message.attachments
        }
        index_refs = {
            tuple(row) for row in conn.execute("SELECT ref_id,native_identity,attachment_id FROM attachment_refs")
        }
        assert len(wire_refs) == 2
        assert wire_refs == index_refs
        assert manifest.semantics_version == 8
        assert {gap.record_id for gap in manifest.fidelity_gaps if gap.scope == "attachment"} == {
            row[0] for row in index_refs
        }
        assert {attachment["byte_count"] for message in decoded.messages for attachment in message.attachments} == {
            len(payload) if payload is not None else 999
        }
    finally:
        conn.close()


def test_retained_source_union_preserves_an_omitted_colliding_native_reference(tmp_path: Path) -> None:
    import json

    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.write_lease import write_lease
    from tests.infra.archive_templates import bootstrap_archive_root
    from tests.infra.index_writer import write_fixture_retained_session

    with write_lease("test.native-attachment-capture", archive_root=tmp_path):
        bootstrap_archive_root(tmp_path)
    conn = _connect(tmp_path / "index.db")
    original_id, inserted_id = "cert-collision-50449", "cert-collision-111329"

    def acquire(native_id: str, revision: int) -> tuple[ParsedSession, str]:
        capture = _capture(extracted_content=None)
        messages = capture["chat_messages"]
        assert isinstance(messages, list)
        messages[0]["files"][0].update({"file_uuid": native_id, "uuid": native_id, "file_name": native_id})
        with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
            raw_id = archive.write_raw_payload(
                provider=Provider.CLAUDE_AI,
                payload=json.dumps(capture).encode(),
                source_path=f"native-capture-{revision}.json",
                canonical_source_path=f"native-capture-{revision}.json",
                acquired_at_ms=revision,
            )
            archive.commit()
        return parse_ai(capture, "native-capture"), raw_id

    try:
        original, raw_original = acquire(original_id, 1)
        write_fixture_retained_session(conn, original, raw_id=raw_original)
        before = tuple(conn.execute("SELECT ref_id,supplying_raw_id FROM attachment_refs").fetchone())
        inserted, raw_inserted = acquire(inserted_id, 2)
        write_fixture_retained_session(conn, inserted, raw_id=raw_inserted)
        rows = conn.execute("SELECT ref_id,native_identity,supplying_raw_id FROM attachment_refs").fetchall()
        assert len(rows) == 2
        old = next(row for row in rows if bytes.fromhex(row["native_identity"]).decode() == original_id)
        new = next(row for row in rows if bytes.fromhex(row["native_identity"]).decode() == inserted_id)
        assert (old["ref_id"], old["supplying_raw_id"]) == before
        assert new["ref_id"] != before[0]
        assert new["supplying_raw_id"] == raw_inserted
    finally:
        conn.close()


def test_one_native_reference_cannot_choose_between_competing_objects(tmp_path: Path) -> None:
    from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage
    from polylogue.storage.sqlite.archive_tiers.write import AttachmentReferenceAmbiguityError

    session = ParsedSession(
        source_name=Provider.CHATGPT,
        provider_session_id="competing-native",
        messages=[ParsedMessage(provider_message_id="m", role=Role.USER, text="files")],
        attachments=[
            ParsedAttachment(provider_attachment_id="one-native", message_provider_id="m", provider_file_id=file_id)
            for file_id in ("file-a", "file-b")
        ],
    )
    conn = _connect(tmp_path / "index.db")
    try:
        with pytest.raises(AttachmentReferenceAmbiguityError, match="competing objects"):
            write_fixture_index_session(conn, session, raw_id="raw-contested")
        assert conn.execute("SELECT COUNT(*) FROM attachment_refs").fetchone()[0] == 0
    finally:
        conn.close()

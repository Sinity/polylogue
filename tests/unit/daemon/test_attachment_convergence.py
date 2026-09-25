"""Route-neutral provider attachment convergence (polylogue-ck5v)."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from polylogue.core.enums import Provider, Role
from polylogue.core.types import AttachmentUploadOrigin
from polylogue.operations.attachment_convergence import converge_drive_attachments
from polylogue.pipeline.ids import session_content_hash, session_revision_projection
from polylogue.sources.drive.types import DriveNotFoundError
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive


def _session(
    session_id: str, *, upload_origin: AttachmentUploadOrigin = "drive", file_id: str | None = None
) -> ParsedSession:
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


def test_polylogue_ck5v_legacy_route_attachment_is_backfilled_and_bounded(tmp_path: Path) -> None:
    """A ZIP-restored row is fetched without Drive iterator enumeration.

    Anti-vacuity: this asserts durable attachment rows and source blob refs,
    not a fetch-helper call. Removing the ``upload_origin='drive'`` production
    guard (or restoring the iterator-only route) leaves the motivating row
    unfetched and makes the public durable-state assertions fail.
    """
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    session = _session("legacy-zip", file_id="drive-file-1")
    write_parsed_session_to_archive(index, session, raw_id="legacy-zip-raw")
    negative = _session("negative-paste", upload_origin="paste", file_id="paste-file-1")
    write_parsed_session_to_archive(index, negative, raw_id="negative-paste-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)

    payload = b"bytes restored from Drive after ZIP import"
    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        return payload

    before = {
        str(row["attachment_id"]): (row["blob_hash"], row["acquisition_status"])
        for row in index.execute("SELECT attachment_id, blob_hash, acquisition_status FROM attachments")
    }

    result = converge_drive_attachments(
        index,
        source,
        archive_root=tmp_path,
        download_bytes=fetch,
        limit=1,
    )

    rows = {
        str(row["attachment_id"]): row
        for row in index.execute("SELECT attachment_id, blob_hash, byte_count, acquisition_status FROM attachments")
    }
    source_ref = source.execute(
        "SELECT blob_hash, ref_id, ref_type, size_bytes FROM blob_refs WHERE ref_type = 'attachment'"
    ).fetchone()
    assert result.acquired == 1
    assert result.complete
    assert calls == ["drive-file-1"]
    assert set(rows) == set(before)
    assert len(rows) == 2
    acquired = next(row for row in rows.values() if row["acquisition_status"] == "acquired")
    assert before[str(acquired["attachment_id"])] == (None, "unfetched")
    assert acquired["acquisition_status"] == "acquired"
    assert acquired["byte_count"] == len(payload)
    assert bytes(acquired["blob_hash"]) == hashlib.sha256(payload).digest()
    negative_row = next(row for row in rows.values() if row["acquisition_status"] == "unfetched")
    assert before[str(negative_row["attachment_id"])] == (None, "unfetched")
    assert negative_row["blob_hash"] is None
    assert negative_row["byte_count"] == 0
    assert bytes(source_ref["blob_hash"]) == hashlib.sha256(payload).digest()
    assert source_ref["ref_id"] == "legacy-zip-raw"
    assert source_ref["size_bytes"] == len(payload)

    # The production backfill only upgrades durable acquisition evidence.  A
    # later raw replay would legitimately produce a new full session hash once
    # the bytes are present, but the comparison identity must remain stable so
    # that this fidelity upgrade is not mistaken for a different attachment.
    acquired_session = session.model_copy(
        update={
            "attachments": [
                session.attachments[0].model_copy(update={"inline_bytes": payload, "size_bytes": len(payload)})
            ]
        }
    )
    before_projection = session_revision_projection(session)
    after_projection = session_revision_projection(acquired_session)
    assert before_projection.attachment_identities == after_projection.attachment_identities
    assert before_projection.attachment_contents != after_projection.attachment_contents
    assert session_content_hash(session) != session_content_hash(acquired_session)

    # Negative control: non-Drive/paste references are not selected by the
    # route-neutral stage, while their identity/hash partition has the same
    # acquisition semantics.
    paste_session = _session("negative-paste", upload_origin="paste", file_id="paste-file-1")
    paste_acquired = paste_session.model_copy(
        update={
            "attachments": [
                paste_session.attachments[0].model_copy(update={"inline_bytes": payload, "size_bytes": len(payload)})
            ]
        }
    )
    paste_before = session_revision_projection(paste_session)
    paste_after = session_revision_projection(paste_acquired)
    assert paste_before.attachment_identities == paste_after.attachment_identities
    assert paste_before.attachment_contents != paste_after.attachment_contents
    assert session_content_hash(paste_session) != session_content_hash(paste_acquired)
    index.close()
    source.close()


def test_attachment_convergence_keeps_retryable_provider_failure_as_debt(tmp_path: Path) -> None:
    """Transient provider failures remain unfetched and retryable."""
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    write_parsed_session_to_archive(index, _session("retry", file_id="temporarily-busy"), raw_id="retry-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)

    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        raise TimeoutError("provider timeout")

    result = converge_drive_attachments(
        index,
        source,
        archive_root=tmp_path,
        download_bytes=fetch,
    )

    row = index.execute("SELECT blob_hash, byte_count, acquisition_status FROM attachments").fetchone()
    assert result.inspected == 1
    assert result.acquired == 0
    assert result.terminal == 0
    # One unit records the provider failure itself; another records that the
    # canonical row remains for the scheduler's next debt pass.
    assert result.deferred >= 1
    assert not result.complete
    assert calls == ["temporarily-busy"]
    assert row["acquisition_status"] == "unfetched"
    assert row["blob_hash"] is None
    assert row["byte_count"] == 0
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


def test_shared_attachment_fetches_once_but_records_each_raw_ref(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    write_parsed_session_to_archive(index, _session("shared-one", file_id="shared-file"), raw_id="raw-1")
    write_parsed_session_to_archive(index, _session("shared-two", file_id="shared-file"), raw_id="raw-2")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)

    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        return b"shared attachment bytes"

    result = converge_drive_attachments(
        index,
        source,
        archive_root=tmp_path,
        download_bytes=fetch,
        limit=10,
    )

    refs = source.execute("SELECT ref_id FROM blob_refs WHERE ref_type = 'attachment' ORDER BY ref_id").fetchall()
    assert result.acquired == 2
    assert calls == ["shared-file"]
    assert [row[0] for row in refs] == ["raw-1", "raw-2"]
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

    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        raise DriveNotFoundError("deleted")

    result = converge_drive_attachments(
        index,
        source,
        archive_root=tmp_path,
        download_bytes=fetch,
    )

    row = index.execute("SELECT blob_hash, byte_count, acquisition_status FROM attachments").fetchone()
    assert result.terminal == 1
    assert result.complete
    assert row["acquisition_status"] == "unavailable"
    assert row["blob_hash"] is None
    assert row["byte_count"] == 0
    retry = converge_drive_attachments(
        index,
        source,
        archive_root=tmp_path,
        download_bytes=fetch,
    )
    assert retry.inspected == 0
    assert calls == ["deleted-file"]
    index.close()
    source.close()


def test_surviving_blob_is_rebound_without_a_provider_request(tmp_path: Path) -> None:
    """A rebuilt attachment row re-binds bytes the blob store still holds.

    Anti-vacuity: without the ``blob_refs`` consultation the second pass
    reaches the downloader, which fails this test outright, and the row would
    only be restored by re-fetching content the archive never lost.
    """
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    write_parsed_session_to_archive(index, _session("survivor", file_id="drive-file-1"), raw_id="survivor-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)

    payload = b"bytes that outlive the derived tier"
    first = converge_drive_attachments(
        index,
        source,
        archive_root=tmp_path,
        download_bytes=lambda file_id: payload,
    )
    assert first.acquired == 1
    blob_hash = hashlib.sha256(payload).digest()
    assert (tmp_path / "blob").exists()
    ref_count = source.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_type = 'attachment'").fetchone()[0]
    assert ref_count == 1

    # Rebuild the derived tier: the index row loses its binding while the
    # durable source ledger and the blob bytes survive untouched.
    with index:
        index.execute("UPDATE attachments SET blob_hash = NULL, byte_count = 0, acquisition_status = 'unfetched'")

    attempted: list[str] = []

    def refuse(file_id: str) -> bytes:
        # The production route classifies exceptions, so record the attempt
        # too: a swallowed refusal must still be visible to the assertions.
        attempted.append(file_id)
        raise AssertionError(f"surviving blob must not be re-downloaded: {file_id}")

    second = converge_drive_attachments(
        index,
        source,
        archive_root=tmp_path,
        download_bytes=refuse,
    )

    row = index.execute("SELECT blob_hash, byte_count, acquisition_status FROM attachments").fetchone()
    assert attempted == []
    assert second.acquired == 1
    assert second.complete
    assert bytes(row["blob_hash"]) == blob_hash
    assert row["byte_count"] == len(payload)
    assert row["acquisition_status"] == "acquired"
    # The re-bind adds no duplicate durable ref.
    assert source.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_type = 'attachment'").fetchone()[0] == 1
    index.close()
    source.close()


def test_contradicted_survivor_is_not_rebound(tmp_path: Path) -> None:
    """A stored object that no longer hashes right is not a survivor.

    polylogue-o0uw5: the survival probe used ``exists`` -- "a file sits at
    that path" -- and the re-bind it gated then wrote
    ``acquisition_status = 'acquired'``, a positive claim that these exact
    bytes were fetched and stored. Corrupting the published object in place
    made the archive assert acquisition of content it no longer held, with no
    provider request and no signal. The claim now carries the evidence it
    asserts: the object is re-hashed, a contradicted one falls through to the
    provider, and a dead provider leaves the row explicitly ``unavailable``
    rather than falsely ``acquired``.

    Anti-vacuity: restore ``blob_store.exists`` in ``_surviving_blob_ref`` and
    the row comes back ``acquired`` with the stale hash, ``attempted`` stays
    empty, and ``result.acquired`` is 1 -- every assertion below flips.
    ``test_surviving_blob_is_rebound_without_a_provider_request`` keeps an
    intact object re-binding without a download, so "never re-bind" fails.
    """
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    write_parsed_session_to_archive(index, _session("decayed", file_id="drive-file-1"), raw_id="decayed-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)

    payload = b"bytes the archive published once"
    first = converge_drive_attachments(
        index,
        source,
        archive_root=tmp_path,
        download_bytes=lambda file_id: payload,
    )
    assert first.acquired == 1
    store = BlobStore(tmp_path / "blob")
    blob_hash = hashlib.sha256(payload).hexdigest()
    assert store.verify(blob_hash) is True

    # Rebuild the derived tier, then decay the published object in place: the
    # path survives, the content no longer matches the recorded identity.
    with index:
        index.execute("UPDATE attachments SET blob_hash = NULL, byte_count = 0, acquisition_status = 'unfetched'")
    store.blob_path(blob_hash).write_bytes(b"not the bytes that were fetched")
    assert store.exists(blob_hash) is True
    assert store.verify(blob_hash) is False

    attempted: list[str] = []

    def gone(file_id: str) -> bytes:
        attempted.append(file_id)
        raise DriveNotFoundError(file_id)

    result = converge_drive_attachments(
        index,
        source,
        archive_root=tmp_path,
        download_bytes=gone,
    )

    row = index.execute("SELECT blob_hash, byte_count, acquisition_status FROM attachments").fetchone()
    assert attempted == ["drive-file-1"]
    assert result.acquired == 0
    assert result.terminal == 1
    assert row["acquisition_status"] == "unavailable"
    assert row["blob_hash"] is None
    assert row["byte_count"] == 0
    index.close()
    source.close()


def test_a_contradicted_destination_blocks_the_acquired_outcome(tmp_path: Path) -> None:
    """Republished original bytes must not be reported acquired over a bad file.

    ``BlobStore.publish_prepared`` returns early whenever the destination
    already exists -- content-addressed deduplication -- and discards the
    staged payload. So when the canonical object under a hash has decayed and
    Drive still serves the *original* payload, the pass staged the good bytes,
    threw them away at ``flush()``, left the corrupted file in place, and wrote
    ``acquisition_status = 'acquired'`` with the original hash anyway.
    Convergence reported a recovery it had not performed, and the only signal
    the archive had lost those bytes was erased.

    Anti-vacuity: delete the ``publisher.exists(...) and not
    publisher.verify(...)`` refusal in ``converge_drive_attachments`` and this
    goes red on every assertion below -- ``result.acquired`` becomes 1, the row
    reads ``acquired`` with the original hash, and the object on disk still
    hashes to nothing.
    ``test_polylogue_ck5v_legacy_route_attachment_is_backfilled_and_bounded``
    pins the other direction, so refusing every publication cannot pass.
    """
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    write_parsed_session_to_archive(index, _session("decayed", file_id="drive-file-1"), raw_id="decayed-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)

    payload = b"bytes the archive published once"
    assert (
        converge_drive_attachments(index, source, archive_root=tmp_path, download_bytes=lambda _f: payload).acquired
        == 1
    )

    store = BlobStore(tmp_path / "blob")
    blob_hash = hashlib.sha256(payload).hexdigest()
    with index:
        index.execute("UPDATE attachments SET blob_hash = NULL, byte_count = 0, acquisition_status = 'unfetched'")
    store.blob_path(blob_hash).write_bytes(b"not the bytes that were fetched")
    assert store.verify(blob_hash) is False

    # Drive still serves the original payload: the republished bytes hash to
    # the contradicted destination, which is the collision that made the
    # dedupe silently discard them.
    result = converge_drive_attachments(index, source, archive_root=tmp_path, download_bytes=lambda _f: payload)

    row = index.execute("SELECT blob_hash, byte_count, acquisition_status FROM attachments").fetchone()
    assert result.acquired == 0
    assert result.terminal == 0
    assert result.contradicted == 1
    assert result.deferred >= 1
    assert result.complete is False
    assert row["acquisition_status"] == "unfetched"
    assert row["blob_hash"] is None
    # The refusal must not have "repaired" the object either: the contradiction
    # is still on disk and still visible, which is what an operator acts on.
    assert store.verify(blob_hash) is False
    index.close()
    source.close()


def _multi_attachment_session(session_id: str, file_ids: tuple[str, ...]) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.GEMINI,
        provider_session_id=session_id,
        messages=[ParsedMessage(provider_message_id="m0", role=Role.USER, text="two drive docs")],
        attachments=[
            ParsedAttachment(
                provider_attachment_id=file_id,
                provider_file_id=file_id,
                message_provider_id="m0",
                name=f"{file_id}.txt",
                mime_type="text/plain",
                upload_origin="drive",
            )
            for file_id in file_ids
        ],
    )


def test_polylogue_ck5v_every_retained_attachment_of_one_raw_is_rebound(tmp_path: Path) -> None:
    """Retained bytes stay reachable when a raw carries several attachments.

    A Drive session with two live-fetched attachments writes two durable
    ``blob_refs`` rows under the same ``ref_id``. After the derived tier is
    rebuilt both index rows are ``unfetched`` while both payloads are still in
    the blob store, so both must re-bind from retained evidence without a
    provider request -- and must do so even when the provider files are gone.

    Anti-vacuity: with a raw-wide acquisition coordinate the two refs are
    indistinguishable, the survival probe refuses as ambiguous, both rows fall
    through to the downloader, the deleted-file refusal marks them terminally
    ``unavailable``, and the retained bytes become unreachable from the index.
    Reverting the per-attachment coordinate therefore fails these assertions.
    """
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    write_parsed_session_to_archive(
        index,
        _multi_attachment_session("two-docs", ("drive-file-a", "drive-file-b")),
        raw_id="two-docs-raw",
    )
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)

    payloads = {
        "drive-file-a": b"first retained document",
        "drive-file-b": b"second retained document",
    }
    first = converge_drive_attachments(
        index,
        source,
        archive_root=tmp_path,
        download_bytes=lambda file_id: payloads[file_id],
    )
    assert first.acquired == 2
    assert source.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_type = 'attachment'").fetchone()[0] == 2

    # Rebuild the derived tier; the durable source ledger and blob bytes survive.
    with index:
        index.execute("UPDATE attachments SET blob_hash = NULL, byte_count = 0, acquisition_status = 'unfetched'")

    attempted: list[str] = []

    def gone(file_id: str) -> bytes:
        attempted.append(file_id)
        raise DriveNotFoundError(file_id)

    second = converge_drive_attachments(
        index,
        source,
        archive_root=tmp_path,
        download_bytes=gone,
    )

    rows = {
        bytes(row["blob_hash"]).hex() if row["blob_hash"] is not None else None: row["acquisition_status"]
        for row in index.execute("SELECT blob_hash, acquisition_status FROM attachments")
    }
    assert attempted == []
    assert second.acquired == 2
    assert second.terminal == 0
    assert second.complete
    assert rows == {
        hashlib.sha256(payloads["drive-file-a"]).digest().hex(): "acquired",
        hashlib.sha256(payloads["drive-file-b"]).digest().hex(): "acquired",
    }
    # Re-binding adds no duplicate durable ref.
    assert source.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_type = 'attachment'").fetchone()[0] == 2
    index.close()
    source.close()


def test_contested_provider_identity_is_refused_not_downloaded_under_a_lexical_winner(
    tmp_path: Path,
) -> None:
    """Two 'file' ids for one reference leave it unresolved, not guessed (polylogue-vnx8v).

    ``attachment_native_ids`` is keyed ``(ref_id, id_kind, native_id)``, so an
    archive can hold two provider file ids for one attachment reference.
    ``ORDER BY native_id LIMIT 1`` answered that by lexical order and handed
    the winner to ``download_bytes`` -- binding whatever bytes came back to an
    attachment whose identity the archive never resolved. The candidate scan
    now selects only a unique observation, so a contested reference is
    reported and left ``unfetched``: an explicit unresolved download target
    rather than a lexical guess or a false terminal ``unavailable``.

    Anti-vacuity (both verified by mutation): dropping the
    ``AND NOT contested_native_id_predicate()`` clause from ``_candidate_rows``
    makes ``calls`` ``["drive-file-contested-a", "drive-file-resolvable"]`` and
    flips the contested row to ``acquired``; making the refusal terminal
    instead of leaving the row owed makes the ``unfetched`` assertion red.
    Both stored aliases are asserted to survive, so "resolve the ambiguity by
    deleting one observation" also fails.
    """
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    contested = _session("contested", file_id="drive-file-contested-a")
    write_parsed_session_to_archive(index, contested, raw_id="contested-raw")
    resolvable = _session("resolvable", file_id="drive-file-resolvable")
    write_parsed_session_to_archive(index, resolvable, raw_id="resolvable-raw")

    ref_id = index.execute("SELECT r.ref_id FROM attachment_refs AS r WHERE r.session_id LIKE '%contested'").fetchone()[
        "ref_id"
    ]
    # A second observation of the same id kind for the same reference: the
    # state the table's primary key admits and the readers had to answer.
    index.execute(
        "INSERT INTO attachment_native_ids (ref_id, id_kind, native_id) VALUES (?, 'file', ?)",
        (ref_id, "drive-file-contested-z"),
    )
    index.commit()

    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)

    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        return b"bytes for %s" % file_id.encode()

    result = converge_drive_attachments(index, source, archive_root=tmp_path, download_bytes=fetch)

    assert calls == ["drive-file-resolvable"]
    assert result.unresolved_identity == 1
    assert result.inspected == 1
    assert result.terminal == 0

    statuses = {
        str(row["ref_id"]): str(row["acquisition_status"])
        for row in index.execute(
            "SELECT r.ref_id, a.acquisition_status FROM attachments AS a "
            "JOIN attachment_refs AS r ON r.attachment_id = a.attachment_id"
        )
    }
    assert statuses[ref_id] == "unfetched"
    assert sorted(statuses.values()) == ["acquired", "unfetched"]

    # Refusing the ambiguity must not resolve it by discarding an observation.
    surviving = sorted(
        str(row[0])
        for row in index.execute(
            "SELECT native_id FROM attachment_native_ids WHERE ref_id = ? AND id_kind = 'file'",
            (ref_id,),
        )
    )
    assert surviving == ["drive-file-contested-a", "drive-file-contested-z"]
    index.close()
    source.close()

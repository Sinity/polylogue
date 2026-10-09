"""Route-neutral provider attachment convergence (polylogue-ck5v)."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path
from typing import IO, Any, cast
from unittest.mock import MagicMock

import pytest

from polylogue.core.enums import Provider, Role
from polylogue.core.stage_admission import admit_stage_write
from polylogue.core.types import AttachmentUploadOrigin
from polylogue.operations.attachment_convergence import AttachmentConvergenceResult, converge_drive_attachments
from polylogue.pipeline.ids import session_content_hash, session_revision_projection
from polylogue.sources.drive.gateway import DriveServiceGateway
from polylogue.sources.drive.source_client import DriveSourceClient
from polylogue.sources.drive.types import DriveNotFoundError, DriveRetryPolicy
from polylogue.sources.parsers.base import ParsedAttachment, ParsedMessage, ParsedSession
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.io_phase_metrics import connect_measured
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.write_lease import write_lease
from tests.infra.drive_mocks import drive_http_error
from tests.infra.index_writer import write_fixture_index_session


def _converge(index: sqlite3.Connection, source: sqlite3.Connection, **kwargs: Any) -> AttachmentConvergenceResult:
    """A standalone convergence pass: its write sections run under the caller's lease."""
    with write_lease("test.attachment-convergence", archive_root=kwargs["archive_root"]):
        return converge_drive_attachments(index, source, **kwargs)


def _into(fetch: Callable[[str], bytes]) -> Callable[[str, IO[bytes]], None]:
    """Adapt a bytes-returning fake to the streaming download contract."""

    def download_into(file_id: str, handle: IO[bytes]) -> None:
        handle.write(fetch(file_id))

    return download_into


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
    # Index capture binds its seal to the connection's original measured creator.
    conn = connect_measured(path, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    return conn


def _retain_raws(source: sqlite3.Connection, *raw_ids: str) -> None:
    """Record durable acquisitions the index's references can name as their supplier."""
    with source:
        source.executemany(
            "INSERT INTO raw_sessions (raw_id, origin, source_path, blob_hash, blob_size, acquired_at_ms) "
            "VALUES (?, 'aistudio-drive', ?, ?, 0, 0)",
            ((raw_id, f"/exports/{raw_id}.json", hashlib.sha256(raw_id.encode()).digest()) for raw_id in raw_ids),
        )


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
    write_fixture_index_session(index, session, raw_id="legacy-zip-raw")
    negative = _session("negative-paste", upload_origin="paste", file_id="paste-file-1")
    write_fixture_index_session(index, negative, raw_id="negative-paste-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "legacy-zip-raw", "negative-paste-raw")

    payload = b"bytes restored from Drive after ZIP import"
    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        return payload

    before = {
        str(row["attachment_id"]): (row["blob_hash"], row["acquisition_status"])
        for row in index.execute("SELECT attachment_id, blob_hash, acquisition_status FROM attachments")
    }

    result = _converge(
        index,
        source,
        archive_root=tmp_path,
        download_into=_into(fetch),
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
    assert len(set(rows) - set(before)) == 1
    assert len(rows) == 2
    acquired = next(row for row in rows.values() if row["acquisition_status"] == "acquired")
    assert str(acquired["attachment_id"]) not in before
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
    write_fixture_index_session(index, _session("retry", file_id="temporarily-busy"), raw_id="retry-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "retry-raw")

    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        raise TimeoutError("provider timeout")

    result = _converge(
        index,
        source,
        archive_root=tmp_path,
        download_into=_into(fetch),
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


def test_attachment_download_streams_to_disk_and_has_no_size_cap(tmp_path: Path) -> None:
    """An attachment of any size is acquired through a real file, never a buffer.

    Anti-vacuity: stage the download through an in-memory buffer and
    ``fileno()`` raises; reinstate a size cap below the payload and the row
    becomes ``unavailable`` instead of ``acquired``.
    """
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    write_fixture_index_session(index, _session("large", file_id="large-file"), raw_id="large-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "large-raw")

    chunk = bytes(range(256)) * 4096  # 1 MiB
    chunk_count = 3
    digest = hashlib.sha256()

    def stream(file_id: str, handle: IO[bytes]) -> None:
        assert file_id == "large-file"
        handle.fileno()
        for _ in range(chunk_count):
            handle.write(chunk)
            digest.update(chunk)

    result = _converge(index, source, archive_root=tmp_path, download_into=stream)

    row = index.execute("SELECT blob_hash, byte_count, acquisition_status FROM attachments").fetchone()
    assert result.acquired == 1
    assert row["acquisition_status"] == "acquired"
    assert row["byte_count"] == len(chunk) * chunk_count
    assert bytes(row["blob_hash"]) == digest.digest()
    assert BlobStore(tmp_path / "blob").read_all(digest.hexdigest()) == chunk * chunk_count
    assert not any((tmp_path / "blob" / ".staging").iterdir())
    index.close()
    source.close()


def test_attachment_convergence_records_debt_for_the_next_bounded_window(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    write_fixture_index_session(index, _session("legacy-one", file_id="drive-file-1"), raw_id="raw-1")
    write_fixture_index_session(index, _session("legacy-two", file_id="drive-file-2"), raw_id="raw-2")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "raw-1", "raw-2")

    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        return file_id.encode()

    result = _converge(
        index,
        source,
        archive_root=tmp_path,
        download_into=_into(fetch),
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
    write_fixture_index_session(index, _session("shared-one", file_id="shared-file"), raw_id="raw-1")
    write_fixture_index_session(index, _session("shared-two", file_id="shared-file"), raw_id="raw-2")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "raw-1", "raw-2")

    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        return b"shared attachment bytes"

    result = _converge(
        index,
        source,
        archive_root=tmp_path,
        download_into=_into(fetch),
        limit=10,
    )

    refs = source.execute("SELECT ref_id FROM blob_refs WHERE ref_type = 'attachment' ORDER BY ref_id").fetchall()
    assert result.acquired == 2
    assert calls == ["shared-file"]
    assert [row[0] for row in refs] == ["raw-1", "raw-2"]
    index.close()
    source.close()


def test_shared_attachment_attribution_survives_window_restart_and_new_reference(tmp_path: Path) -> None:
    """The shared byte status cannot discharge a supplier outside the window.

    Mutation: select only unfetched global attachment rows. The first window
    hides raw 26 and the later new raw, despite neither having a Source ref.
    """
    from polylogue.operations.attachment_convergence import inspect_attachment_readiness

    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    raw_ids = [f"shared-raw-{i:02d}" for i in range(26)]
    for i, raw_id in enumerate(raw_ids):
        write_fixture_index_session(index, _session(f"shared-session-{i:02d}", file_id="shared-file"), raw_id=raw_id)
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, *raw_ids)
    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        return b"shared attachment bytes"

    first = _converge(index, source, archive_root=tmp_path, download_into=_into(fetch))
    assert first.inspected == 25
    assert first.transport_pending and not first.complete
    assert source.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_type='attachment'").fetchone()[0] == 25
    assert inspect_attachment_readiness(index, source)["allowed_unfetched"] == 1
    index.close()
    source.close()

    # No common provider observation survives for the unattempted supplier.
    # It must measure the current file rather than borrow older captured bytes.
    index = _open_index(tmp_path / "index.db")
    source = sqlite3.connect(tmp_path / "source.db")
    second = _converge(index, source, archive_root=tmp_path, download_into=_into(fetch))
    assert second.inspected == 1 and second.acquired == 1 and second.complete
    assert {row[0] for row in source.execute("SELECT ref_id FROM blob_refs WHERE ref_type='attachment'")} == set(
        raw_ids
    )
    assert inspect_attachment_readiness(index, source)["allowed_unfetched"] == 0
    assert calls == ["shared-file", "shared-file"]

    write_fixture_index_session(index, _session("late-session", file_id="shared-file"), raw_id="late-raw")
    index.commit()
    _retain_raws(source, "late-raw")
    assert inspect_attachment_readiness(index, source)["allowed_unfetched"] == 1
    late = _converge(index, source, archive_root=tmp_path, download_into=_into(fetch))
    assert late.inspected == 1 and late.acquired == 1 and late.complete
    assert (
        source.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_type='attachment' AND ref_id='late-raw'").fetchone()[0]
        == 1
    )
    assert calls == ["shared-file", "shared-file", "shared-file"]
    index.close()
    source.close()


def test_shared_attribution_cancellation_retries_without_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancel the publication of a new supplier and retry the same owed ref."""
    import asyncio

    import polylogue.operations.attachment_convergence as convergence

    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    write_fixture_index_session(index, _session("first", file_id="shared-file"), raw_id="first-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "first-raw")
    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        return b"shared attachment bytes"

    assert _converge(index, source, archive_root=tmp_path, download_into=_into(fetch)).complete
    later_session = _session("later", file_id="shared-file")
    later_session.attachments[0].inline_bytes = b"shared attachment bytes"
    write_fixture_index_session(
        index,
        later_session,
        raw_id="later-raw",
        preacquired_attachment_blobs={
            later_session.attachments[0].acquisition_key: (
                hashlib.sha256(b"shared attachment bytes").digest(),
                len(b"shared attachment bytes"),
                "acquired",
            )
        },
    )
    index.commit()
    _retain_raws(source, "later-raw")
    admission = admit_stage_write

    def cancel(*args: object) -> None:
        raise asyncio.CancelledError

    with monkeypatch.context() as patch:
        patch.setattr(convergence, "admit_stage_write", cancel)
        with pytest.raises(asyncio.CancelledError):
            _converge(index, source, archive_root=tmp_path, download_into=_into(fetch))
    assert vars(convergence)["admit_stage_write"] is admission
    assert (
        source.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_type='attachment' AND ref_id='later-raw'").fetchone()[
            0
        ]
        == 0
    )
    assert convergence.inspect_attachment_readiness(index, source)["allowed_unfetched"] == 1
    retry = _converge(index, source, archive_root=tmp_path, download_into=_into(fetch))
    assert retry.complete and retry.acquired == 1
    assert (
        source.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_type='attachment' AND ref_id='later-raw'").fetchone()[
            0
        ]
        == 1
    )
    assert calls == ["shared-file"]
    index.close()
    source.close()


def test_retryable_prefix_reaches_later_work_across_restart_and_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing the persisted keyset position starves the healthy 26th row."""
    import asyncio

    import polylogue.operations.attachment_convergence as convergence

    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    source = sqlite3.connect(tmp_path / "source.db")
    for i in range(26):
        write_fixture_index_session(index, _session(f"sweep-{i}"), raw_id=f"sweep-raw-{i}")
    _retain_raws(source, *(f"sweep-raw-{i}" for i in range(26)))
    ordered = index.execute(
        "SELECT a.attachment_id,r.ref_id,ani.native_id FROM attachments a "
        "JOIN attachment_refs r ON r.attachment_id=a.attachment_id "
        "JOIN attachment_native_ids ani ON ani.ref_id=r.ref_id AND ani.id_kind='file' "
        "ORDER BY a.attachment_id,r.ref_id"
    ).fetchall()
    healthy = str(ordered[-1][2])
    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        if file_id != healthy:
            raise OSError("synthetic retryable per-file fault")
        return b"healthy payload"

    first = _converge(index, source, archive_root=tmp_path, download_into=_into(fetch))
    assert first.inspected == 25 and first.acquired == 0 and first.transport_pending
    assert healthy not in calls
    with closing(sqlite3.connect(tmp_path / "ops.db")) as ops:
        position = ops.execute("SELECT attachment_id,ref_id FROM attachment_convergence_cursor").fetchone()
    assert position == tuple(ordered[24][:2])
    index.close()
    source.close()

    index = _open_index(tmp_path / "index.db")
    source = sqlite3.connect(tmp_path / "source.db")

    def cancel(*args: object) -> None:
        raise asyncio.CancelledError

    with monkeypatch.context() as patch:
        patch.setattr(convergence, "admit_stage_write", cancel)
        with pytest.raises(asyncio.CancelledError):
            _converge(index, source, archive_root=tmp_path, download_into=_into(fetch))
    with closing(sqlite3.connect(tmp_path / "ops.db")) as ops:
        assert ops.execute("SELECT attachment_id,ref_id FROM attachment_convergence_cursor").fetchone() == position
    assert convergence.inspect_attachment_readiness(index, source)["allowed_unfetched"] == 26
    resumed = _converge(index, source, archive_root=tmp_path, download_into=_into(fetch))
    assert resumed.inspected == 1 and resumed.acquired == 1 and resumed.transport_pending
    assert calls.count(healthy) == 2
    assert convergence.inspect_attachment_readiness(index, source)["allowed_unfetched"] == 25
    retry = _converge(index, source, archive_root=tmp_path, download_into=_into(fetch))
    assert retry.inspected == 25 and retry.acquired == 0 and retry.transport_pending
    assert source.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_type='attachment'").fetchone()[0] == 1
    index.close()
    source.close()


def test_new_supplier_observes_new_bytes_and_renamed_old_supplier_rebinds_its_own_revision(tmp_path: Path) -> None:
    """A global descriptor shortcut replaces raw-A's AA with raw-B's BB."""
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    source = sqlite3.connect(tmp_path / "source.db")
    _retain_raws(source, "raw-a", "raw-b")
    old = _session("capture-a", file_id="revision-file")
    old.attachments[0].inline_bytes = b"AA"
    old.attachments[0].size_bytes = 2
    digest, size = BlobStore(tmp_path / "blob").write_from_bytes(b"AA")
    write_fixture_index_session(
        index,
        old,
        raw_id="raw-a",
        preacquired_attachment_blobs={old.attachments[0].acquisition_key: (bytes.fromhex(digest), size, "acquired")},
    )

    def no_download(_file: str) -> bytes:
        raise AssertionError("captured bytes require no provider download")

    assert _converge(index, source, archive_root=tmp_path, download_into=_into(no_download)).complete
    original = tuple(source.execute("SELECT ref_id,source_path,blob_hash FROM blob_refs WHERE ref_type='attachment'"))
    new = _session("capture-b", file_id="revision-file")
    new.attachments[0].size_bytes = 2
    write_fixture_index_session(index, new, raw_id="raw-b")
    calls: list[str] = []

    def current(file_id: str) -> bytes:
        calls.append(file_id)
        return b"BB"

    assert _converge(index, source, archive_root=tmp_path, download_into=_into(current)).complete
    assert calls == ["revision-file"]
    rows = index.execute(
        "SELECT r.supplying_raw_id,a.blob_hash FROM attachments a JOIN attachment_refs r "
        "ON r.attachment_id=a.attachment_id ORDER BY r.supplying_raw_id"
    ).fetchall()
    assert [(row[0], bytes(row[1])) for row in rows] == [
        ("raw-a", hashlib.sha256(b"AA").digest()),
        ("raw-b", hashlib.sha256(b"BB").digest()),
    ]
    assert (
        tuple(
            source.execute(
                "SELECT ref_id,source_path,blob_hash FROM blob_refs WHERE ref_type='attachment' AND ref_id='raw-a'"
            )
        )
        == original
    )
    # A renamed descriptor does not change the exact retained raw/file
    # coordinate. Reparse binds AA from Source, rather than current BB.
    renamed = _session("capture-a", file_id="revision-file")
    renamed.attachments[0].name = "renamed.txt"
    renamed.attachments[0].size_bytes = 2
    write_fixture_index_session(index, renamed, raw_id="raw-a", force_replace=True)
    rebound = _converge(index, source, archive_root=tmp_path, download_into=_into(no_download))
    assert rebound.acquired == 1 and rebound.complete
    assert (
        bytes(
            index.execute(
                "SELECT a.blob_hash FROM attachments a JOIN attachment_refs r ON r.attachment_id=a.attachment_id "
                "WHERE r.supplying_raw_id='raw-a'"
            ).fetchone()[0]
        )
        == hashlib.sha256(b"AA").digest()
    )
    assert (
        tuple(
            source.execute(
                "SELECT ref_id,source_path,blob_hash FROM blob_refs WHERE ref_type='attachment' AND ref_id='raw-a'"
            )
        )
        == original
    )
    index.close()
    source.close()


def test_attachment_convergence_terminal_failure_does_not_fabricate_bytes(tmp_path: Path) -> None:
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    session = _session("gone", file_id="deleted-file")
    write_fixture_index_session(index, session, raw_id="gone-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "gone-raw")

    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        raise DriveNotFoundError("deleted")

    result = _converge(
        index,
        source,
        archive_root=tmp_path,
        download_into=_into(fetch),
    )

    row = index.execute("SELECT blob_hash, byte_count, acquisition_status FROM attachments").fetchone()
    assert result.terminal == 1
    assert result.complete
    assert row["acquisition_status"] == "unavailable"
    assert row["blob_hash"] is None
    assert row["byte_count"] == 0
    retry = _converge(
        index,
        source,
        archive_root=tmp_path,
        download_into=_into(fetch),
    )
    assert retry.inspected == 0
    assert calls == ["deleted-file"]
    index.close()
    source.close()


def _drive_client(
    monkeypatch: pytest.MonkeyPatch, download: Callable[[str, IO[bytes]], None], *, retries: int
) -> DriveSourceClient:
    """The production Drive client and gateway, with only the media request replaced."""
    gateway = DriveServiceGateway(
        auth_manager=MagicMock(),
        retry_policy=DriveRetryPolicy(retries=retries, retry_base=0.0),
    )
    monkeypatch.setattr(gateway, "download_file", download)
    return DriveSourceClient(gateway=gateway)


def test_rate_limited_403_keeps_the_attachment_owed_until_the_quota_resets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """07.F002: a throttled 403 is retryable debt, a denied 403 is terminal.

    Drive answers ``userRateLimitExceeded`` with HTTP 403. The gateway retries
    it and re-raises the provider error, and the attachment must stay
    ``unfetched`` so a later pass acquires it once the quota resets.

    Anti-vacuity: restore the status-only rule (``status in {403, 404}``) in
    ``_permanent_failure`` and the throttled row turns ``unavailable`` on the
    first pass, so the second pass inspects nothing and acquires nothing.
    Classify every 403 as retryable and the denied row stays ``unfetched``.
    """
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    write_fixture_index_session(index, _session("throttled", file_id="drive-throttled"), raw_id="throttled-raw")
    write_fixture_index_session(index, _session("denied", file_id="drive-denied"), raw_id="denied-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "throttled-raw", "denied-raw")

    retries = 2
    attempts: list[str] = []
    throttled = drive_http_error(403, ("usageLimits", "userRateLimitExceeded"))
    denied = drive_http_error(403, ("global", "insufficientFilePermissions"))

    def quota_exhausted(file_id: str, _handle: IO[bytes]) -> None:
        attempts.append(file_id)
        raise throttled if file_id == "drive-throttled" else denied

    first = _converge(
        index,
        source,
        archive_root=tmp_path,
        download_into=_drive_client(monkeypatch, quota_exhausted, retries=retries).download_into,
        limit=10,
    )

    def status(file_id: str) -> str:
        return str(
            index.execute(
                "SELECT a.acquisition_status FROM attachments a "
                "JOIN attachment_refs r ON r.attachment_id = a.attachment_id "
                "JOIN attachment_native_ids n ON n.ref_id = r.ref_id AND n.id_kind = 'file' "
                "WHERE n.native_id = ?",
                (file_id,),
            ).fetchone()[0]
        )

    assert attempts.count("drive-throttled") == retries + 1
    assert attempts.count("drive-denied") == 1
    assert first.terminal == 1
    assert first.transport_pending
    assert status("drive-throttled") == "unfetched"
    assert status("drive-denied") == "unavailable"

    payload = b"bytes served once the quota window reset"

    def quota_reset(file_id: str, handle: IO[bytes]) -> None:
        attempts.append(file_id)
        handle.write(payload)

    second = _converge(
        index,
        source,
        archive_root=tmp_path,
        download_into=_drive_client(monkeypatch, quota_reset, retries=retries).download_into,
        limit=10,
    )
    assert second.inspected == 1
    assert second.acquired == 1
    assert second.complete
    assert status("drive-throttled") == "acquired"
    assert status("drive-denied") == "unavailable"
    index.close()
    source.close()


def _carried_forward_attachment(index: sqlite3.Connection) -> None:
    """Raw A holds a Drive attachment; a later raw B of the session omits it.

    B keeps the owning message, so projection carry-forward restores the
    reference while ``sessions.raw_id`` moves to B.
    """
    with_attachment = _session("carried", file_id="drive-carried")
    write_fixture_index_session(index, with_attachment, raw_id="raw-a")
    write_fixture_index_session(index, with_attachment.model_copy(update={"attachments": []}), raw_id="raw-b")
    index.commit()
    assert index.execute("SELECT raw_id FROM sessions").fetchone()[0] == "raw-b"
    assert [tuple(row) for row in index.execute("SELECT supplying_raw_id FROM attachment_refs")] == [("raw-a",)]


def test_carried_forward_attachment_bytes_are_attributed_to_the_raw_that_held_it(tmp_path: Path) -> None:
    """07.F004: the durable ref names the acquisition that supplied the reference.

    Anti-vacuity: take ``raw_id`` from ``sessions`` again (the previous
    candidate query) and the blob ref is written under ``raw-b``; drop
    ``supplying_raw_id`` from the carry-forward capture and the restored
    reference has no supplier, so nothing is acquired.
    """
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    _carried_forward_attachment(index)
    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "raw-a", "raw-b")

    payload = b"attachment bytes raw A referenced"
    result = _converge(index, source, archive_root=tmp_path, download_into=_into(lambda _file_id: payload))

    assert result.acquired == 1
    assert result.complete
    refs = source.execute("SELECT ref_id, blob_hash FROM blob_refs WHERE ref_type = 'attachment'").fetchall()
    assert [(row[0], bytes(row[1])) for row in refs] == [("raw-a", hashlib.sha256(payload).digest())]
    index.close()
    source.close()


def test_an_unretained_supplier_is_never_replaced_by_the_sessions_current_raw(tmp_path: Path) -> None:
    """07.F004: the supplier is checked against durable ``raw_sessions``.

    The index is rebuildable, so its ``supplying_raw_id`` is only a claim.
    When ``source.db`` no longer retains raw A the reference cannot be
    attributed: nothing is downloaded, no blob ref is written for raw B, and
    the obligation stays open without being retried as transport work.

    Anti-vacuity: fall back to ``sessions.raw_id`` for an unretained supplier
    and the pass downloads the file and writes ``blob_refs(ref_id='raw-b')``;
    skip the ``raw_sessions`` probe and it writes a durable ref naming a raw
    ``source.db`` does not hold.
    """
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    _carried_forward_attachment(index)
    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "raw-b")

    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        return b"never fetched"

    result = _converge(index, source, archive_root=tmp_path, download_into=_into(fetch))

    assert calls == []
    assert result.inspected == 0
    assert result.unattributed == 1
    assert not result.transport_pending
    assert not result.complete
    assert source.execute("SELECT COUNT(*) FROM blob_refs").fetchone()[0] == 0
    assert index.execute("SELECT acquisition_status FROM attachments").fetchone()[0] == "unfetched"
    index.close()
    source.close()


def test_a_supplier_retired_during_the_download_gets_no_durable_ref(tmp_path: Path) -> None:
    """07.F004: the supplier is re-checked under the writer, before publication.

    Anti-vacuity: drop the publish-time probe and the pass writes a
    ``blob_refs`` row for the retired raw and reserves its bytes.
    """
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    _carried_forward_attachment(index)
    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "raw-a", "raw-b")

    def retire_then_open() -> tuple[sqlite3.Connection, sqlite3.Connection]:
        with source:
            source.execute("DELETE FROM raw_sessions WHERE raw_id = 'raw-a'")
        write_index = _open_index(tmp_path / "index.db")
        write_source = sqlite3.connect(tmp_path / "source.db")
        return write_index, write_source

    payload = b"downloaded while raw A was retired"
    result = _converge(
        index,
        source,
        archive_root=tmp_path,
        download_into=_into(lambda _file_id: payload),
        open_write_connections=retire_then_open,
    )

    digest = hashlib.sha256(payload).digest()
    assert result.acquired == 0
    assert result.unattributed == 1
    assert source.execute("SELECT COUNT(*) FROM blob_refs").fetchone()[0] == 0
    reservations = source.execute(
        "SELECT COUNT(*) FROM blob_publication_reservations WHERE blob_hash = ?", (digest,)
    ).fetchone()[0]
    assert reservations == 0
    assert not BlobStore(tmp_path / "blob").exists(digest.hex())
    assert index.execute("SELECT acquisition_status FROM attachments").fetchone()[0] == "unfetched"
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
    write_fixture_index_session(index, _session("survivor", file_id="drive-file-1"), raw_id="survivor-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "survivor-raw")

    payload = b"bytes that outlive the derived tier"
    first = _converge(
        index,
        source,
        archive_root=tmp_path,
        download_into=_into(lambda file_id: payload),
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

    second = _converge(
        index,
        source,
        archive_root=tmp_path,
        download_into=_into(refuse),
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
    write_fixture_index_session(index, _session("decayed", file_id="drive-file-1"), raw_id="decayed-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "decayed-raw")

    payload = b"bytes the archive published once"
    first = _converge(
        index,
        source,
        archive_root=tmp_path,
        download_into=_into(lambda file_id: payload),
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

    result = _converge(
        index,
        source,
        archive_root=tmp_path,
        download_into=_into(gone),
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
    write_fixture_index_session(index, _session("decayed", file_id="drive-file-1"), raw_id="decayed-raw")
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "decayed-raw")

    payload = b"bytes the archive published once"
    assert _converge(index, source, archive_root=tmp_path, download_into=_into(lambda _f: payload)).acquired == 1

    store = BlobStore(tmp_path / "blob")
    blob_hash = hashlib.sha256(payload).hexdigest()
    with index:
        index.execute("UPDATE attachments SET blob_hash = NULL, byte_count = 0, acquisition_status = 'unfetched'")
    store.blob_path(blob_hash).write_bytes(b"not the bytes that were fetched")
    assert store.verify(blob_hash) is False

    # Drive still serves the original payload: the republished bytes hash to
    # the contradicted destination, which is the collision that made the
    # dedupe silently discard them.
    result = _converge(index, source, archive_root=tmp_path, download_into=_into(lambda _f: payload))

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


@pytest.mark.parametrize("equal_content", (False, True))
def test_polylogue_ck5v_every_retained_attachment_of_one_raw_is_rebound(tmp_path: Path, equal_content: bool) -> None:
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
    write_fixture_index_session(
        index,
        _multi_attachment_session("two-docs", ("drive-file-a", "drive-file-b")),
        raw_id="two-docs-raw",
    )
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "two-docs-raw")

    payloads = {
        "drive-file-a": b"first retained document",
        "drive-file-b": b"first retained document" if equal_content else b"second retained document",
    }
    first = _converge(
        index,
        source,
        archive_root=tmp_path,
        download_into=_into(lambda file_id: payloads[file_id]),
    )
    assert first.acquired == 2
    assert source.execute("SELECT COUNT(*) FROM blob_refs WHERE ref_type = 'attachment'").fetchone()[0] == 2

    # Destroy and recreate the derived database, keeping only durable bytes/evidence.
    index.close()
    (tmp_path / "index.db").unlink()
    index = _open_index(tmp_path / "index.db")
    write_fixture_index_session(
        index, _multi_attachment_session("two-docs", ("drive-file-a", "drive-file-b")), raw_id="two-docs-raw"
    )
    index.commit()

    attempted: list[str] = []

    def gone(file_id: str) -> bytes:
        attempted.append(file_id)
        raise DriveNotFoundError(file_id)

    second = _converge(
        index,
        source,
        archive_root=tmp_path,
        download_into=_into(gone),
    )

    rows = {
        bytes(row["blob_hash"]).hex() if row["blob_hash"] is not None else None: row["acquisition_status"]
        for row in index.execute("SELECT blob_hash, acquisition_status FROM attachments")
    }
    assert [
        tuple(row)
        for row in source.execute("SELECT source_path FROM blob_refs WHERE ref_type='attachment' ORDER BY source_path")
    ] == [("attachment:drive-file-a",), ("attachment:drive-file-b",)]
    assert index.execute("SELECT COUNT(*) FROM attachments WHERE acquisition_status='acquired'").fetchone()[0] == 2
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
    write_fixture_index_session(index, contested, raw_id="contested-raw")
    resolvable = _session("resolvable", file_id="drive-file-resolvable")
    write_fixture_index_session(index, resolvable, raw_id="resolvable-raw")

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
    _retain_raws(source, "contested-raw", "resolvable-raw")

    calls: list[str] = []

    def fetch(file_id: str) -> bytes:
        calls.append(file_id)
        return b"bytes for %s" % file_id.encode()

    result = _converge(index, source, archive_root=tmp_path, download_into=_into(fetch))

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


def _seed_contested(tmp_path: Path, *, with_resolvable: bool) -> tuple[sqlite3.Connection, str]:
    """One reference with two 'file' ids of one kind, optionally beside a resolvable one."""
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    write_fixture_index_session(index, _session("contested", file_id="drive-file-contested-a"), raw_id="c-raw")
    if with_resolvable:
        write_fixture_index_session(index, _session("resolvable", file_id="drive-file-resolvable"), raw_id="r-raw")
    ref_id = str(
        index.execute("SELECT r.ref_id FROM attachment_refs AS r WHERE r.session_id LIKE '%contested'").fetchone()[
            "ref_id"
        ]
    )
    index.execute(
        "INSERT INTO attachment_native_ids (ref_id, id_kind, native_id) VALUES (?, 'file', 'drive-file-contested-z')",
        (ref_id,),
    )
    index.commit()
    source = sqlite3.connect(tmp_path / "source.db")
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "c-raw", "r-raw")
    source.close()
    return index, ref_id


class _CountingDriveClient:
    constructed = 0
    downloads: list[str] = []

    def __init__(self) -> None:
        type(self).constructed += 1

    def download_into(self, file_id: str, handle: IO[bytes]) -> None:
        type(self).downloads.append(file_id)
        handle.write(b"bytes for %s" % file_id.encode())


def _run_passes(tmp_path: Path, passes: int) -> type[_CountingDriveClient]:
    from polylogue.core.stage_admission import stage_write_admission
    from polylogue.daemon.convergence import DaemonConverger
    from polylogue.operations.attachment_convergence import make_attachment_convergence_stage

    client = type("_Client", (_CountingDriveClient,), {"constructed": 0, "downloads": []})
    stage = make_attachment_convergence_stage(
        tmp_path / "index.db", archive_root=tmp_path, client_factory=client, limit=10
    )
    converger = DaemonConverger(stages=(stage,))

    def admit(actor: str, work: Callable[[], object]) -> object:
        # As the daemon's admission does: the stage's write section runs under the lease.
        with write_lease(actor, archive_root=tmp_path):
            return work()

    with stage_write_admission(admit):
        for _ in range(passes):
            converger.converge_batch((tmp_path / "source-batch.jsonl",))
    return client


def _ordinary_attachment_status(root: Path) -> dict[str, object]:
    from polylogue.operations.daemon_reads import DaemonReadDependencies, execute_read_operation
    from polylogue.operations.operation_context import open_operation_read

    with open_operation_read(root) as pinned:
        payload = execute_read_operation(
            "status",
            {},
            archive=pinned.archive,
            serving_identity="daemon",
            dependencies=DaemonReadDependencies(status_now_ms=1_700_000_000_000),
            read_view=pinned.read_view,
        )
    components = cast(dict[str, object], payload["component_readiness"])
    component = cast(dict[str, object], components["attachments"])
    if component["state"] == "degraded":
        assert payload["ok"] is False
        guard = cast(dict[str, dict[str, object]], payload["claim_guard"])
        assert guard["converged"]["value"] is False
        assert guard["converged"]["reason"] == component["summary"]
    return component


def test_contested_only_identity_is_not_complete_and_never_re_executes(tmp_path: Path) -> None:
    """polylogue-xfw1t: a contested reference is reported, not retried forever.

    Anti-vacuity: counting every unfetched Drive ref in the stage check (the
    previous predicate) executes the stage on each of the three passes, so
    ``constructed`` is 3; folding ``unresolved_identity`` back out of
    ``complete`` certifies this archive complete.
    """
    index, ref_id = _seed_contested(tmp_path, with_resolvable=False)
    client = _run_passes(tmp_path, passes=3)
    assert client.constructed == 0
    assert client.downloads == []
    component = _ordinary_attachment_status(tmp_path)
    assert component["state"] == "degraded"
    assert component["scope"] == "owed_drive_references"
    assert cast(dict[str, int], component["counts"])["unresolved_identity"] == 1

    source = sqlite3.connect(tmp_path / "source.db")
    result = _converge(index, source, archive_root=tmp_path, download_into=_into(lambda _id: b""))
    assert result.unresolved_identity == 1
    assert not result.transport_pending
    assert not result.complete
    status = index.execute(
        "SELECT a.acquisition_status FROM attachments a JOIN attachment_refs r ON r.attachment_id = a.attachment_id "
        "WHERE r.ref_id = ?",
        (ref_id,),
    ).fetchone()[0]
    assert status == "unfetched"
    index.close()
    source.close()


def test_mixed_set_fetches_resolvable_work_once_and_stays_incomplete(tmp_path: Path) -> None:
    """Resolvable work proceeds; the contested remainder bounds no further execution.

    Anti-vacuity: with contested identity counted as deferred transport work
    the stage returns pending after acquiring the resolvable row and every
    later pass re-executes it, so ``constructed`` becomes 3.
    """
    index, _ref_id = _seed_contested(tmp_path, with_resolvable=True)
    client = _run_passes(tmp_path, passes=3)
    assert client.downloads == ["drive-file-resolvable"]
    assert client.constructed == 1
    component = _ordinary_attachment_status(tmp_path)
    assert component["state"] == "degraded"
    assert cast(dict[str, int], component["counts"])["unresolved_identity"] == 1

    source = sqlite3.connect(tmp_path / "source.db")
    result = _converge(index, source, archive_root=tmp_path, download_into=_into(lambda _id: b""))
    assert result.unresolved_identity == 1
    assert result.inspected == 0
    assert not result.complete
    index.close()
    source.close()


def test_terminal_absence_stays_distinct_from_contested_identity(tmp_path: Path) -> None:
    """A provider 404 is terminal ``unavailable`` and completes the obligation; contested identity does not."""
    initialize_active_archive_root(tmp_path)
    index = _open_index(tmp_path / "index.db")
    write_fixture_index_session(index, _session("gone", file_id="drive-file-gone"), raw_id="g-raw")
    source = sqlite3.connect(tmp_path / "source.db")
    source.row_factory = sqlite3.Row
    initialize_archive_tier(source, ArchiveTier.SOURCE)
    _retain_raws(source, "g-raw")

    def missing(file_id: str, _handle: IO[bytes]) -> None:
        raise DriveNotFoundError(file_id)

    result = _converge(index, source, archive_root=tmp_path, download_into=missing)
    assert result.terminal == 1
    assert result.unresolved_identity == 0
    assert result.complete
    assert index.execute("SELECT acquisition_status FROM attachments").fetchone()[0] == "unavailable"
    index.close()
    source.close()


@pytest.mark.parametrize("disposition", ["resolved", "terminal", "empty", "unavailable"])
def test_ordinary_attachment_status_distinguishes_zero_from_unavailable(
    tmp_path: Path,
    disposition: str,
) -> None:
    """Mutation: hide contested identity, or swallow a failed count as zero."""
    index, ref_id = _seed_contested(tmp_path, with_resolvable=False)
    if disposition == "resolved":
        index.execute(
            "DELETE FROM attachment_native_ids WHERE ref_id = ? AND native_id = 'drive-file-contested-z'", (ref_id,)
        )
    elif disposition == "terminal":
        index.execute("UPDATE attachments SET acquisition_status = 'unavailable'")
    elif disposition == "empty":
        index.execute("DELETE FROM attachment_refs")
    else:
        index.execute("DROP TABLE attachment_native_ids")
    index.commit()
    if disposition == "unavailable":
        from polylogue.operations.daemon_status import _attachment_component

        component = cast(dict[str, object], _attachment_component(index, None).to_dict())
        index.close()
        from polylogue.core.errors import SchemaVersionMismatchError

        with pytest.raises(SchemaVersionMismatchError):
            _ordinary_attachment_status(tmp_path)
    else:
        index.close()
        component = _ordinary_attachment_status(tmp_path)
    assert component["state"] == (
        "unknown" if disposition == "unavailable" else "degraded" if disposition == "resolved" else "ready"
    )
    assert (
        component["counts"] == {}
        if disposition == "unavailable"
        else cast(dict[str, int], component["counts"])["unresolved_identity"] == 0
    )
    if disposition == "resolved":
        assert cast(dict[str, int], component["counts"])["allowed_unfetched"] == 1
    if disposition == "terminal":
        assert cast(dict[str, int], component["counts"])["terminal_unavailable"] == 1

"""No publication route can put excised bytes back on disk.

The excision ledger is read in the reservation's own write transaction, so
every ``ArchiveBlobPublisher`` caller -- raw acquisition, parse-time
snapshots, attachment convergence -- inherits the refusal (polylogue-u6jyu).

Anti-vacuity: drop the ``_excised_hashes`` filter in ``reserve_many`` and the
excised payload is reserved and published under its content hash.
"""

from __future__ import annotations

import hashlib
from contextlib import closing
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.storage.blob_publication import ArchiveBlobPublisher, ConnectionBlobPublicationRead
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.source_write import record_excised_blob_hash
from polylogue.storage.sqlite.connection_profile import open_source_tier_write_connection
from polylogue.storage.sqlite.write_lease import write_lease


def test_completed_claim_retirement_preserves_other_same_hash_capture(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.blob-publication", archive_root=root):
        with ArchiveStore(root, initialize=True, read_only=False):
            pass
        publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
        prepared = publisher.prepare_from_bytes(b"same captured bytes")
        claim = publisher.prepare_claim(prepared)
        publisher.queue_prepared(prepared, claim=claim)
        with pytest.raises(RuntimeError):
            publisher.forget_completed_claim(claim)
        assert publisher.flush() == (claim.receipt,)

        blob_hash, _size = publisher.write_from_bytes(b"same captured bytes")
        other_receipt = publisher.receipt_id(blob_hash)
        assert other_receipt is not None and other_receipt != claim.receipt.publication_id
        publisher.forget_completed_claim(claim)
        assert publisher.receipt_id(blob_hash) == other_receipt
        assert publisher.has_pending
        assert tuple(receipt.publication_id for receipt in publisher.flush()) == (other_receipt,)
        publisher.queue_prepared(prepared, claim=claim)
        assert not publisher.has_pending


def test_completed_excised_claim_releases_local_refusal_but_preserves_ledger(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.blob-publication", archive_root=root):
        with ArchiveStore(root, initialize=True, read_only=False):
            pass
        publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
        prepared = publisher.prepare_from_bytes(b"excised prepared capture")
        claim = publisher.prepare_claim(prepared)
        with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source, source:
            record_excised_blob_hash(
                source,
                blob_hash=bytes.fromhex(prepared.hash_hex),
                reason="synthetic excision",
                actor="test",
                excised_at_ms=1,
            )
        publisher.queue_prepared(prepared, claim=claim)
        assert publisher.flush() == ()
        assert publisher.refused_as_excised(prepared.hash_hex)
        publisher.forget_completed_claim(claim)
        assert not publisher.refused_as_excised(prepared.hash_hex)
        assert publisher.excised_now(prepared.hash_hex)
        assert not publisher.exists(prepared.hash_hex)


def test_an_excised_payload_is_never_published(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.blob-publication", archive_root=root):
        with ArchiveStore(root, initialize=True, read_only=False):
            pass
        excised = b"bytes the operator excised"
        kept = b"bytes that stay"
        with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source, source:
            record_excised_blob_hash(
                source,
                blob_hash=hashlib.sha256(excised).digest(),
                reason="synthetic excision",
                actor="test",
                excised_at_ms=1,
            )

        publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
        excised_hex, _ = publisher.write_from_bytes(excised)
        kept_hex, _ = publisher.write_from_bytes(kept)
        receipts = publisher.flush()

        assert [receipt.blob_hash for receipt in receipts] == [kept_hex]
        assert not publisher.exists(excised_hex)
        assert publisher.read_all(kept_hex) == kept
        assert publisher.receipt_id(excised_hex) is None
        assert not any((root / "blob" / ".staging").iterdir())
        with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source, source:
            reserved = {
                bytes(row[0]).hex() for row in source.execute("SELECT blob_hash FROM blob_publication_reservations")
            }
        assert reserved == {kept_hex}


def test_repeated_sealed_claim_reuses_exact_reservation_after_private_file_publication(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.blob-publication", archive_root=root):
        with ArchiveStore(root, initialize=True, read_only=False):
            pass
        publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
        prepared = publisher.prepare_from_bytes(b"neutral sealed attachment")
        claim = publisher.prepare_claim(prepared)
        publisher.queue_prepared(prepared, claim=claim)
        first = publisher.flush()
        assert tuple(receipt.publication_id for receipt in first) == (claim.receipt.publication_id,)
        assert not prepared.temporary_path.exists()
        publisher.queue_prepared(prepared, claim=claim)
        assert not publisher.has_pending
        assert publisher.flush() == ()
        with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source, source:
            rows = source.execute(
                "SELECT publication_id, blob_hash, size_bytes, publisher_id FROM blob_publication_reservations"
            ).fetchall()
            assert rows == [
                (
                    claim.receipt.publication_id,
                    bytes.fromhex(claim.receipt.blob_hash),
                    claim.receipt.size_bytes,
                    publisher.publisher_id,
                )
            ]
            publisher.validate_published_claim(ConnectionBlobPublicationRead(source), claim, source_path="neutral.txt")


def test_an_excised_sqlite_snapshot_is_a_typed_excision_not_a_parse_failure(tmp_path: Path) -> None:
    """A parse route that reads its snapshot back after flushing stops typed.

    Anti-vacuity: drop ``require_published`` after the Hermes state-db flush
    and ``parse_state_db`` opens the discarded snapshot path, so the call
    raises ``FileNotFoundError`` and the walk records a cursor failure.
    """
    import pytest

    from polylogue.sources.source_parsing import parse_one_source_path
    from polylogue.sources.sqlite_snapshot import snapshot_sqlite_to_blob
    from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError
    from tests.unit.sources.test_hermes_import_explain import _write_state_db

    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.blob-publication", archive_root=root):
        with ArchiveStore(root, initialize=True, read_only=False):
            pass
        state_db = tmp_path / "state.db"
        _write_state_db(state_db)
        probe = ArchiveBlobPublisher(root / "source.db", root / "blob")
        snapshot_hash = snapshot_sqlite_to_blob(state_db, probe).blob_hash
        probe.discard_pending()
        with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source, source:
            record_excised_blob_hash(
                source,
                blob_hash=bytes.fromhex(snapshot_hash),
                reason="synthetic excision",
                actor="test",
                excised_at_ms=1,
            )

        publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
        with pytest.raises(ContentExcisedError) as refused:
            list(
                parse_one_source_path(
                    str(state_db),
                    file_mtime=None,
                    source_name="hermes",
                    sidecar_data={},
                    capture_raw=True,
                    blob_store=publisher,
                )
            )
        assert refused.value.blob_hash.hex() == snapshot_hash
        assert not publisher.exists(snapshot_hash)


def test_reading_a_refused_hash_is_a_typed_excision(tmp_path: Path) -> None:
    """Anti-vacuity: let ``blob_path`` fall through for a refused hash and a
    reader gets the staging path that flush deleted (``FileNotFoundError``)."""
    import pytest

    from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError

    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.blob-publication", archive_root=root):
        with ArchiveStore(root, initialize=True, read_only=False):
            pass
        excised = b"bytes a later reader asks for"
        with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source, source:
            record_excised_blob_hash(
                source,
                blob_hash=hashlib.sha256(excised).digest(),
                reason="synthetic excision",
                actor="test",
                excised_at_ms=1,
            )
        publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
        excised_hex, _ = publisher.write_from_bytes(excised)
        publisher.flush()

        assert publisher.exists(excised_hex) is False
        with pytest.raises(ContentExcisedError):
            publisher.read_all(excised_hex)


def _claude_code_line(session_id: str, text: str) -> bytes:
    import json

    record = {
        "type": "user",
        "uuid": f"{session_id}-message-1",
        "parentUuid": None,
        "sessionId": session_id,
        "timestamp": "2026-07-10T00:00:00Z",
        "message": {"role": "user", "content": text},
    }
    return (json.dumps(record) + "\n").encode("utf-8")


def test_an_excised_grouped_raw_capture_is_refused_and_a_zip_keeps_its_other_members(tmp_path: Path) -> None:
    """Raw capture of refused bytes stops typed, per file and per ZIP member.

    Anti-vacuity: drop ``require_published`` after the grouped flush and the
    plain file yields a ``RawSessionData`` naming discarded bytes; drop it in
    the ZIP member route and the excised member is still emitted.
    """
    import zipfile

    import pytest

    from polylogue.sources.source_parsing import parse_one_source_path
    from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError

    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.blob-publication", archive_root=root):
        with ArchiveStore(root, initialize=True, read_only=False):
            pass
        excised = _claude_code_line("excised-session", "bytes the operator excised")
        kept = _claude_code_line("kept-session", "bytes that stay")
        with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source, source:
            record_excised_blob_hash(
                source,
                blob_hash=hashlib.sha256(excised).digest(),
                reason="synthetic excision",
                actor="test",
                excised_at_ms=1,
            )
        plain = tmp_path / "excised.jsonl"
        plain.write_bytes(excised)
        publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
        with pytest.raises(ContentExcisedError):
            list(
                parse_one_source_path(
                    str(plain),
                    file_mtime=None,
                    source_name="claude-code",
                    sidecar_data={},
                    capture_raw=True,
                    blob_store=publisher,
                )
            )

        bundle = tmp_path / "bundle.zip"
        with zipfile.ZipFile(bundle, "w") as zf:
            zf.writestr("projects/p/excised.jsonl", excised)
            zf.writestr("projects/p/kept.jsonl", kept)
        pairs = list(
            parse_one_source_path(
                str(bundle),
                file_mtime=None,
                source_name="claude-code",
                sidecar_data={},
                capture_raw=True,
                blob_store=publisher,
            )
        )
        captured = {raw.blob_hash for raw, _session in pairs if raw is not None}
        assert hashlib.sha256(kept).hexdigest() in captured
        assert hashlib.sha256(excised).hexdigest() not in captured


def test_archive_store_records_an_excised_inline_attachment_unavailable(tmp_path: Path) -> None:
    """The direct replay and membership routes neither republish nor reference excised bytes.

    Anti-vacuity: drop the ledger check from
    ``ArchiveStore._preacquire_attachment_blobs`` and it returns an
    ``acquired`` attachment and an ``ArchiveSourceBlobRef`` for the excised
    hash, which ``write_source_blob_refs`` then refuses for the whole session.
    """
    from types import SimpleNamespace
    from typing import Any, cast

    from polylogue.sources.parsers.base import ParsedAttachment

    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.blob-publication", archive_root=root):
        excised = b"attachment bytes excised from another session"
        kept = b"attachment bytes that stay"
        with ArchiveStore(root, initialize=True, read_only=False) as store:
            record_excised_blob_hash(
                store._ensure_source_conn(),
                blob_hash=hashlib.sha256(excised).digest(),
                reason="synthetic excision",
                actor="test",
                excised_at_ms=1,
            )
            store._ensure_source_conn().commit()
            excised_attachment = ParsedAttachment(provider_attachment_id="excised", inline_bytes=excised)
            kept_attachment = ParsedAttachment(provider_attachment_id="kept", inline_bytes=kept)
            acquired, refs = store._preacquire_attachment_blobs(
                cast(Any, SimpleNamespace(attachments=[excised_attachment, kept_attachment])),
                source_path="synthetic",
                acquired_at_ms=1,
            )
            if store._blob_publisher is not None:
                store._blob_publisher.discard_pending()

        assert acquired[excised_attachment.acquisition_key] == (None, len(excised), "unavailable")
        assert acquired[kept_attachment.acquisition_key][2] == "acquired"
        assert [ref.blob_hash for ref in refs] == [hashlib.sha256(kept).digest()]


def test_attachments_refused_between_check_and_flush_are_reconciled(tmp_path: Path) -> None:
    """An excision landing after the ledger check still ends in ``unavailable``.

    Anti-vacuity: skip ``reconcile_refused_attachments`` after a replay flush
    and the refused attachment keeps its ``acquired`` entry and blob
    reference, which ``write_source_blob_refs`` refuses for the whole session.
    """
    from types import SimpleNamespace

    from polylogue.storage.blob_publication import reconcile_refused_attachments

    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.blob-publication", archive_root=root):
        with ArchiveStore(root, initialize=True, read_only=False):
            pass
        excised = b"bytes excised after the caller's check"
        kept = b"bytes that stay"
        publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
        excised_hex, _ = publisher.write_from_bytes(excised)
        kept_hex, _ = publisher.write_from_bytes(kept)
        # The excision commits after the caller queued both attachments.
        with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source, source:
            record_excised_blob_hash(
                source,
                blob_hash=bytes.fromhex(excised_hex),
                reason="synthetic excision",
                actor="test",
                excised_at_ms=1,
            )
        publisher.flush()
        acquired: dict[object, tuple[bytes | None, int, str]] = {
            "excised": (bytes.fromhex(excised_hex), len(excised), "acquired"),
            "kept": (bytes.fromhex(kept_hex), len(kept), "acquired"),
        }
        refs = (
            SimpleNamespace(blob_hash=bytes.fromhex(excised_hex)),
            SimpleNamespace(blob_hash=bytes.fromhex(kept_hex)),
        )

        reconciled, kept_refs = reconcile_refused_attachments(acquired, refs, publisher)

        assert reconciled["excised"] == (None, len(excised), "unavailable")
        assert reconciled["kept"] == acquired["kept"]
        assert [ref.blob_hash.hex() for ref in kept_refs] == [kept_hex]


def test_a_refused_publication_does_not_hide_bytes_still_retained(tmp_path: Path) -> None:
    """A hash excised for one reingest stays readable where its bytes are still on disk.

    Anti-vacuity (Codex P2, #5696): consult the refusal set before the final
    path and every later read of the still-retained blob raises
    ``ContentExcisedError``.
    """
    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.blob-publication", archive_root=root):
        with ArchiveStore(root, initialize=True, read_only=False):
            pass
        payload = b"bytes another retained session still references"
        publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
        retained_hex, _ = publisher.write_from_bytes(payload)
        publisher.flush()
        with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source, source:
            record_excised_blob_hash(
                source,
                blob_hash=bytes.fromhex(retained_hex),
                reason="synthetic excision",
                actor="test",
                excised_at_ms=1,
            )
        publisher.write_from_bytes(payload)
        publisher.flush()
        assert publisher.refused_as_excised(retained_hex)

        assert publisher.exists(retained_hex) is True
        assert publisher.read_all(retained_hex) == payload


def test_an_excision_after_a_successful_flush_is_reconciled_from_the_ledger(tmp_path: Path) -> None:
    """An excision landing between a clean flush and the reference write ends ``unavailable``.

    Anti-vacuity (Codex P2, #5696): reconcile only flush-local refusals and
    the attachment stays ``acquired`` with its reference, which the reference
    write then refuses for the whole replay.
    """
    from types import SimpleNamespace

    from polylogue.storage.blob_publication import reconcile_refused_attachments

    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.blob-publication", archive_root=root):
        with ArchiveStore(root, initialize=True, read_only=False):
            pass
        payload = b"bytes excised after the flush"
        publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
        blob_hex, _ = publisher.write_from_bytes(payload)
        publisher.flush()
        with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source, source:
            record_excised_blob_hash(
                source, blob_hash=bytes.fromhex(blob_hex), reason="synthetic excision", actor="test", excised_at_ms=1
            )
        acquired: dict[object, tuple[bytes | None, int, str]] = {
            "a": (bytes.fromhex(blob_hex), len(payload), "acquired")
        }
        refs = (SimpleNamespace(blob_hash=bytes.fromhex(blob_hex)),)

        with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source, source:
            reconciled, kept_refs = reconcile_refused_attachments(acquired, refs, publisher, source_conn=source)

        assert reconciled["a"] == (None, len(payload), "unavailable")
        assert kept_refs == ()


def test_require_published_refuses_a_hash_excised_after_the_flush(tmp_path: Path) -> None:
    """An excision committed after the flush is still the typed refusal.

    Anti-vacuity (Codex P2, #5696): consult only the flush's own refusals and
    a snapshot whose bytes were excised a moment later is accepted.
    """
    import pytest

    from polylogue.storage.blob_publication import require_published
    from polylogue.storage.sqlite.archive_tiers.source_write import ContentExcisedError

    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.blob-publication", archive_root=root):
        with ArchiveStore(root, initialize=True, read_only=False):
            pass
        publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
        payload = b"raw capture excised after its flush"
        blob_hash, _size = publisher.write_from_bytes(payload)
        publisher.flush()
        require_published(publisher, blob_hash, source_path="capture.jsonl")

        with closing(open_source_tier_write_connection(root / "source.db", archive_root=root)) as source, source:
            record_excised_blob_hash(
                source,
                blob_hash=hashlib.sha256(payload).digest(),
                reason="synthetic excision",
                actor="test",
                excised_at_ms=1,
            )

        with pytest.raises(ContentExcisedError):
            require_published(publisher, blob_hash, source_path="capture.jsonl")


def test_retained_replay_writes_hold_the_publisher_slot_through_their_commit(tmp_path: Path) -> None:
    """An excision cannot take its exclusion while a retained write is uncommitted.

    Anti-vacuity (Codex P1, #5696): release the shared publisher slot before
    the replay's commit and an excision can remove the session in between,
    which the replay then recreates.
    """
    import fcntl

    import pytest

    import polylogue.storage.sqlite.archive_tiers.archive as archive_module
    from polylogue.storage.blob_publication import _writer_lock_path

    root = tmp_path / "archive"
    root.mkdir()
    with write_lease("test.blob-publication", archive_root=root):
        with ArchiveStore(root, initialize=True, read_only=False):
            pass
        lock_path = _writer_lock_path(root / "source.db")

        def exclusion_available() -> bool:
            with lock_path.open("a+b") as handle:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return False
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                return True

        seen: list[bool] = []

        def observed_write(*_args: object, **_kwargs: object) -> object:
            seen.append(exclusion_available())
            return object()

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(archive_module, "apply_raw_revision_replay", observed_write)
        plan = cast(Any, object())
        outcome = cast(Any, object())
        try:
            with ArchiveStore(root, read_only=False) as store:
                store.apply_raw_revision_replay(plan, {}, prepared_outcome=outcome, acquired_at_ms=1)
                assert exclusion_available()
                store.apply_raw_revision_replay(
                    plan, {}, prepared_outcome=outcome, acquired_at_ms=1, manage_transaction=False
                )
                # A batched write keeps the slot until its batch commits.
                assert not exclusion_available()
                store.commit()
                assert exclusion_available()
        finally:
            monkeypatch.undo()

        assert seen == [False, False]

"""Logical paths of machine-ingest inputs frozen from a staged import."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from polylogue.operations.ingest_inputs import discover_ingest_input_spool, retain_input_page, unlink_spool
from polylogue.sources.source_staging import stage_source_input
from polylogue.storage.blob_publication import ArchiveBlobPublisher


@pytest.mark.parametrize("read_only", [False, True])
def test_spool_failed_close_retains_exact_artifact_until_creator_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, read_only: bool
) -> None:
    import sqlite3

    from polylogue.operations.ingest_inputs import spool_connection
    from polylogue.storage.sqlite import connection_profile

    path = tmp_path / "private-spool.sqlite"
    with spool_connection(path) as conn:
        conn.execute("CREATE TABLE evidence(value INTEGER)").close()
        conn.execute("INSERT INTO evidence VALUES (1)").close()
    actual_connect = sqlite3.connect
    fail_close = [True]

    # Spool connections are measured connections (``connect_measured``): their
    # owner settles cursors through the measured class before closing, so the
    # fault is injected into that same class rather than replacing it.
    from polylogue.storage.io_phase_metrics import _MeasuredConnection

    class FailingClose(_MeasuredConnection):
        def close(self) -> None:
            if fail_close[0]:
                raise sqlite3.OperationalError("synthetic private spool close failure")
            super().close()

    def connect(database: str, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        kwargs["factory"] = FailingClose
        connection: sqlite3.Connection = actual_connect(database, *args, **kwargs)
        return connection

    monkeypatch.setattr(sqlite3, "connect", connect)
    try:
        with pytest.raises(connection_profile.NativeConnectionSettlementError):
            with spool_connection(path, read_only=read_only) as conn:
                assert conn.execute("PRAGMA temp_store").fetchone() == (1,)
                assert conn.execute("PRAGMA journal_mode").fetchone() == ("delete",)
                assert conn.execute("SELECT value FROM evidence").fetchone() == (1,)
        owners = connection_profile.retained_native_sql_owners_for_lifetime(path)
        assert owners
        with pytest.raises(connection_profile.NativeConnectionSettlementError):
            unlink_spool(path)
        assert path.is_file()
    finally:
        fail_close[0] = False
        for owner in connection_profile.retained_native_sql_owners_for_lifetime(path):
            owner.close()
        unlink_spool(path)
    assert not path.exists()


def test_spool_cancelled_transaction_rolls_back_and_settles_before_cleanup(tmp_path: Path) -> None:
    from polylogue.operations.ingest_inputs import spool_connection
    from polylogue.storage.sqlite.connection_profile import retained_native_sql_owners_for_lifetime

    path = tmp_path / "cancelled-spool.sqlite"
    with spool_connection(path) as conn:
        conn.execute("CREATE TABLE evidence(value INTEGER)").close()
    with pytest.raises(InterruptedError):
        with spool_connection(path) as conn:
            conn.execute("INSERT INTO evidence VALUES (1)").close()
            raise InterruptedError("synthetic cancellation")
    assert not retained_native_sql_owners_for_lifetime(path)
    with spool_connection(path, read_only=True) as conn:
        assert conn.execute("SELECT COUNT(*) FROM evidence").fetchone() == (0,)
    unlink_spool(path)


def _retain(path: Path, source_path: str | None, tmp_path: Path) -> set[tuple[str, str]]:
    spool = discover_ingest_input_spool(path, source_path=source_path, check_stop=lambda: None)
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    try:
        page = retain_input_page(spool, after_coordinate=None, publisher=publisher, check_stop=lambda: None)
    finally:
        publisher.discard_pending()
        unlink_spool(spool)
    return {(item.coordinate, item.source_path) for item in page}


@pytest.mark.skipif(os.geteuid() == 0, reason="requires an unprivileged directory reader")
def test_ordinary_directory_intake_refuses_a_denied_hidden_subtree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recursive glob silently publishes only the public part of the input."""
    original = tmp_path / "exports"
    hidden = original / "hidden"
    hidden.mkdir(parents=True)
    (original / "public.jsonl").write_bytes(b"{}\n")
    (hidden / "session.jsonl").write_bytes(b"{}\n")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    hidden.chmod(0)
    try:
        with pytest.raises(PermissionError):
            discover_ingest_input_spool(original, source_path=None, check_stop=lambda: None)
    finally:
        hidden.chmod(0o700)
    assert list(scratch.glob("polylogue-ingest-paths-*.sqlite")) == []


def test_staged_directory_members_are_keyed_under_the_callers_path(tmp_path: Path) -> None:
    """03.F045: a staged directory import is acquired where the caller's export lives.

    ``polylogue import <dir>`` stages outside the watched inbox, so the ingest
    operation is its only route; each member keeps its relative path under the
    submitted ``source_path``.

    Anti-vacuity: refuse a directory with a ``source_path`` again and this
    raises; key members on the staged physical path and the logical paths
    name ``import-staging`` instead of ``/exports/account``.
    """
    original = tmp_path / "exports" / "account"
    (original / "nested").mkdir(parents=True)
    (original / "conversations.json").write_text("[]")
    (original / "nested" / "chat.jsonl").write_text("{}\n")
    staged = stage_source_input(original, tmp_path / "import-staging", check_stop=lambda: None)

    assert _retain(staged, str(original), tmp_path) == {
        ("conversations.json", str(original / "conversations.json")),
        ("nested/chat.jsonl", str(original / "nested" / "chat.jsonl")),
    }


@pytest.mark.parametrize("member_count", [1, 32])
def test_staged_intake_reads_its_outside_receipt_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, member_count: int
) -> None:
    """Rehashing the whole directory receipt per member makes metadata work quadratic."""
    from polylogue.sources import source_staging, sqlite_export

    original = tmp_path / "exports"
    original.mkdir()
    for ordinal in range(member_count):
        (original / f"member-{ordinal:03d}.jsonl").write_bytes(b"{}\n")
    staged = stage_source_input(original, tmp_path / "staging", check_stop=lambda: None)
    metadata = source_staging.staging_metadata_path(staged)
    accounting = tmp_path / "receipt-reads.txt"
    fixture = Path(__file__).parents[2] / "fixtures/sqlite_source/staging_read_accounting.py"
    monkeypatch.setenv("POLYLOGUE_TEST_RECEIPT_NAME", metadata.name)
    monkeypatch.setenv("POLYLOGUE_TEST_RECEIPT_ACCOUNTING", str(accounting))
    monkeypatch.setattr(
        sqlite_export, "_WORKER_COMMAND", f"exec(compile(open({str(fixture)!r}).read(), {str(fixture)!r}, 'exec'))"
    )

    retained = _retain(staged, str(original), tmp_path)
    assert len(retained) == member_count
    assert sum(int(line) for line in accounting.read_text().splitlines()) == metadata.stat().st_size


@pytest.mark.parametrize("mutation", ["receipt-rewrite", "receipt-replace", "slot-replace", "late-member"])
def test_captured_staging_receipt_does_not_admit_changed_members_or_custody(tmp_path: Path, mutation: str) -> None:
    """The fully read receipt stays authority only while its captured custody is current."""
    from polylogue.sources.source_staging import staging_metadata_path

    original = tmp_path / "exports"
    original.mkdir()
    for ordinal in range(3):
        (original / f"member-{ordinal}.jsonl").write_bytes(b"{}\n")
    staged = stage_source_input(original, tmp_path / "staging", check_stop=lambda: None)
    spool = discover_ingest_input_spool(staged, source_path=str(original), check_stop=lambda: None)
    publisher = ArchiveBlobPublisher(tmp_path / "source.db", tmp_path / "blob")
    try:
        metadata = staging_metadata_path(staged)
        if mutation == "receipt-rewrite":
            metadata.write_bytes(metadata.read_bytes())
        elif mutation == "receipt-replace":
            replacement = tmp_path / "replacement-receipt"
            replacement.write_bytes(metadata.read_bytes())
            replacement.replace(metadata)
        elif mutation == "slot-replace":
            moved = staged.with_name("moved-slot")
            staged.rename(moved)
            staged.symlink_to(moved, target_is_directory=True)
        else:
            (staged / "member-2.jsonl").write_bytes(b"[]\n")
        with pytest.raises(OSError):
            retain_input_page(spool, after_coordinate=None, publisher=publisher, check_stop=lambda: None)
    finally:
        publisher.discard_pending()
        unlink_spool(spool)


def test_file_intake_requires_the_captured_original_declaration(tmp_path: Path) -> None:
    import pytest

    original = tmp_path / "exports" / "session.jsonl"
    original.parent.mkdir()
    original.write_text("{}\n")
    staged = stage_source_input(original, tmp_path / "import-staging", check_stop=lambda: None)
    assert _retain(staged, str(original), tmp_path) == {("input:0", str(original))}
    with pytest.raises(ValueError):
        _retain(staged, "/unproved/session.jsonl", tmp_path)
    assert _retain(original, None, tmp_path) == {("input:0", str(original))}


def test_machine_zip_enumeration_preserves_the_accepted_decoder_identity(tmp_path: Path) -> None:
    """Machine preparation carries its accepted decoder into coordinate and raw ID."""
    import json
    import zipfile

    from polylogue.core.raw_coordinates import captured_zip_member_raw_id
    from polylogue.operations.ingest_inputs import PreparedSourceRecord, enumerate_ingest_input
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    root = tmp_path / "archive"
    initialize_active_archive_root(root)
    source = tmp_path / "export.zip"
    payload = {
        "id": "synthetic-conversation",
        "title": "Synthetic",
        "mapping": {
            "message": {
                "id": "message",
                "parent": None,
                "children": [],
                "message": {
                    "id": "message",
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["hello"]},
                },
            }
        },
    }
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("conversations.json", json.dumps([payload]))
    spool = discover_ingest_input_spool(source, source_path=None, check_stop=lambda: None)
    publisher = ArchiveBlobPublisher(root / "source.db", root / "blob")
    try:
        (item,) = retain_input_page(spool, after_coordinate=None, publisher=publisher, check_stop=lambda: None)
        identities: list[str] = []
        for fingerprint in ("b" * 64, "c" * 64):
            records = [
                record
                for record in enumerate_ingest_input(
                    item,
                    source_generation_id="accepted-machine",
                    enumeration_fingerprint=fingerprint,
                    publisher=publisher,
                    acquired_at_ms=1,
                    check_stop=lambda: None,
                )
                if isinstance(record, PreparedSourceRecord)
            ]
            assert len(records) == 1
            record = records[0]
            coordinate = record.admission.request.captured_zip_coordinate
            assert coordinate is not None
            assert coordinate.decoder_fingerprint == fingerprint
            assert coordinate.container_blob_hash == item.blob_hash
            assert item.captured_identity is not None
            assert coordinate.canonical_container == item.captured_identity.canonical_source_path
            assert record.admission.raw_id == captured_zip_member_raw_id(
                coordinate, record.admission.request.blob_hash.hex()
            )
            identities.append(record.admission.raw_id)
        assert identities[0] != identities[1]
    finally:
        publisher.discard_pending()
        unlink_spool(spool)


@pytest.mark.parametrize("member_count", [1, 16, pytest.param(256, marks=pytest.mark.timeout(0))])
def test_actual_retained_input_page_uses_one_settled_byte_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, member_count: int
) -> None:
    import subprocess

    source = tmp_path / "exports"
    source.mkdir()
    for ordinal in range(member_count):
        (source / f"member-{ordinal:03}.jsonl").write_bytes(b"{}\n")
    from polylogue.sources import sqlite_export

    original = subprocess.Popen
    exchange = sqlite_export._exchange_source_worker
    children: list[subprocess.Popen[bytes]] = []
    binding_children: list[subprocess.Popen[bytes]] = []
    binding = False

    def observe_exchange(request: dict[str, Any], handle: Any = None) -> dict[str, Any]:
        nonlocal binding
        previous = binding
        binding = request["operation"] == "binding"
        try:
            return exchange(request, handle)
        finally:
            binding = previous

    def launch(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        child = original(*args, **kwargs)
        (binding_children if binding else children).append(child)
        return child

    monkeypatch.setattr(sqlite_export, "_exchange_source_worker", observe_exchange)
    monkeypatch.setattr(subprocess, "Popen", launch)
    assert len(_retain(source, None, tmp_path)) == member_count
    # The page's one reader also proves every binding (5525e77e71); a fresh
    # binding process per input made a 10,241-input ingest spawn 10,241.
    assert binding_children == []
    assert len(children) == 1 and children[0].poll() == 0
    for child in [*binding_children, *children]:
        assert child.poll() == 0
        assert child.stdin is not None and child.stdin.closed
        assert child.stdout is not None and child.stdout.closed

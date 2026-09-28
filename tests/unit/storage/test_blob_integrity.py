"""Read-only blob integrity scanner contracts (#1231)."""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path

import pytest

from polylogue.archive import zip_admission
from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedAttachment, ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage import blob_integrity
from polylogue.storage.blob_integrity import (
    classify_blob_reference_debt,
    referenced_blob_hashes,
    scan_attachment_coverage,
    scan_blob_integrity,
    scan_blob_reference_debt,
)
from polylogue.storage.blob_store import BlobStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import (
    initialize_archive_database,
    initialize_archive_tier,
)
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.write import write_parsed_session_to_archive
from polylogue.storage.sqlite.connection_profile import open_readonly_connection


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE raw_sessions (
            raw_id TEXT PRIMARY KEY,
            source_name TEXT NOT NULL DEFAULT '',
            source_path TEXT NOT NULL DEFAULT '',
            blob_hash BLOB,
            blob_size INTEGER NOT NULL DEFAULT 0,
            acquired_at TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE blob_refs (
            blob_hash BLOB NOT NULL,
            raw_id TEXT NOT NULL,
            ref_type TEXT NOT NULL DEFAULT 'raw_payload'
        );
        """
    )
    conn.commit()
    return conn


def test_scan_blob_integrity_classifies_missing_orphan_and_hash_mismatch(tmp_path: Path) -> None:
    """Covers the three findings ``scan_blob_integrity`` still classifies.

    A prior revision of this test also covered ``stale_leases``/leased-blob
    classification via ``pending_blob_refs``; that table and the lease
    mechanism it backed were removed as unreachable dead code
    (polylogue-v7e0 — no production ingest caller ever populated the lease
    payload keys), so the leased-blob assertions were removed with it.
    """
    db_path = tmp_path / "archive.db"
    store = BlobStore(tmp_path / "blob")
    conn = _make_db(db_path)

    referenced_ok, ok_size = store.write_from_bytes(b"referenced")
    orphan_hash, _ = store.write_from_bytes(b"orphan")
    corrupt_hash, corrupt_size = store.write_from_bytes(b"original")
    store.blob_path(corrupt_hash).write_bytes(b"corrupted")
    missing_hash = "0" * 64

    for blob_hash, size in ((referenced_ok, ok_size), (missing_hash, 128), (corrupt_hash, corrupt_size)):
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, blob_hash, blob_size, acquired_at) VALUES (?, ?, ?, ?)",
            (f"raw-{blob_hash[:8]}", bytes.fromhex(blob_hash), size, "2026-05-24T00:00:00+00:00"),
        )
    conn.commit()
    conn.close()

    report = scan_blob_integrity(db_path, store=store, full=True)

    by_kind = {finding.kind: finding for finding in report.findings}
    assert by_kind["missing_referenced_blobs"].sample == (missing_hash,)
    assert by_kind["orphan_blobs"].sample == (orphan_hash,)
    assert by_kind["hash_mismatch"].sample == (corrupt_hash,)


def test_scan_blob_integrity_reports_invalid_namespace_entries(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    store = BlobStore(tmp_path / "blob")
    _make_db(db_path).close()
    blob_hash, _ = store.write_from_bytes(b"valid")
    sidecar = store.root / f"{blob_hash}-wal"
    sidecar.write_bytes(b"not a blob")

    report = scan_blob_integrity(db_path, store=store, full=True)

    finding = next(finding for finding in report.findings if finding.kind == "invalid_namespace_entries")
    assert finding.severity == "critical"
    assert finding.count == 1
    assert finding.sample == (sidecar.name,)


def test_scan_blob_integrity_reports_file_backed_root(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    root = tmp_path / "blob"
    root.write_bytes(b"not a directory")
    _make_db(db_path).close()

    report = scan_blob_integrity(db_path, store=BlobStore(root), full=True)

    finding = next(finding for finding in report.findings if finding.kind == "invalid_namespace_entries")
    assert finding.severity == "critical"
    assert finding.count == 1
    assert finding.sample == (".",)


def test_scan_blob_integrity_bounds_default_probe_but_full_scans_everything(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    store = BlobStore(tmp_path / "blob")
    conn = _make_db(db_path)
    hashes = [store.write_from_bytes(f"payload-{idx}".encode())[0] for idx in range(3)]
    for blob_hash in hashes:
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, blob_hash, blob_size, acquired_at) VALUES (?, ?, ?, ?)",
            (f"raw-{blob_hash[:8]}", bytes.fromhex(blob_hash), 9, "2026-05-24T00:00:00+00:00"),
        )
    conn.commit()
    conn.close()

    sampled = scan_blob_integrity(db_path, store=store, full=False, sample_size=1)
    full = scan_blob_integrity(db_path, store=store, full=True, sample_size=1)

    assert sampled.scanned_blobs == 1
    assert sampled.scanned_references == 1
    assert full.scanned_blobs == 3
    assert full.scanned_references == 3


def test_scan_blob_integrity_reads_source_tier_blob_refs(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    store = BlobStore(tmp_path / "blob")
    raw_hash, raw_size = store.write_from_bytes(b"raw payload")
    attachment_hash, attachment_size = store.write_from_bytes(b"attachment")
    orphan_hash, _ = store.write_from_bytes(b"orphan")
    with sqlite3.connect(source_db) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                blob_hash BLOB NOT NULL,
                blob_size INTEGER NOT NULL
            );
            CREATE TABLE blob_refs (
                blob_hash BLOB NOT NULL,
                raw_id TEXT NOT NULL,
                ref_type TEXT NOT NULL,
                source_path TEXT,
                size_bytes INTEGER NOT NULL,
                acquired_at_ms INTEGER NOT NULL,
                PRIMARY KEY(blob_hash, raw_id, ref_type)
            );
            """
        )
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, blob_hash, blob_size) VALUES (?, ?, ?)",
            ("raw-v1", bytes.fromhex(raw_hash), raw_size),
        )
        conn.execute(
            """
            INSERT INTO blob_refs (
                blob_hash, raw_id, ref_type, source_path, size_bytes, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (bytes.fromhex(attachment_hash), "raw-v1", "attachment", "/tmp/a.bin", attachment_size, 1),
        )

    report = scan_blob_integrity(source_db, store=store, full=True)

    by_kind = {finding.kind: finding for finding in report.findings}
    assert report.total_references_seen == 1
    assert set(by_kind["orphan_blobs"].sample) == {attachment_hash, orphan_hash}
    assert raw_hash not in by_kind["orphan_blobs"].sample
    assert attachment_hash in by_kind["orphan_blobs"].sample


def test_scan_blob_reference_debt_counts_all_missing_refs_with_bounded_sample(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    store = BlobStore(tmp_path / "blob")
    present_hash, present_size = store.write_from_bytes(b"present")
    missing_hashes = [f"{idx:064x}" for idx in range(5)]
    with sqlite3.connect(source_db) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                blob_hash BLOB NOT NULL,
                blob_size INTEGER NOT NULL
            );
            CREATE TABLE blob_refs (
                blob_hash BLOB NOT NULL,
                raw_id TEXT NOT NULL,
                ref_type TEXT NOT NULL,
                PRIMARY KEY(blob_hash, raw_id, ref_type)
            );
            """
        )
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, blob_hash, blob_size) VALUES (?, ?, ?)",
            ("raw-present", bytes.fromhex(present_hash), present_size),
        )
        for idx, blob_hash in enumerate(missing_hashes):
            conn.execute(
                "INSERT INTO blob_refs (blob_hash, raw_id, ref_type) VALUES (?, ?, ?)",
                (bytes.fromhex(blob_hash), f"raw-{idx}", "attachment"),
            )

    report = scan_blob_reference_debt(source_db, store=store, sample_size=2)

    assert report.ok is True
    assert report.total_references_seen == 1
    assert report.missing_referenced_blobs == 0
    assert report.sample == ()
    assert report.reference_sources == {"source.db.raw_sessions": 1}
    assert referenced_blob_hashes(source_db) == [present_hash]


def test_source_capabilities_choose_current_shape_over_zero_user_version(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        conn.execute("PRAGMA user_version = 0")
        capabilities = blob_integrity._source_schema_capabilities(conn)

    assert capabilities.kind == "current_unversioned"
    assert capabilities.current_authority is True
    with pytest.raises(RuntimeError, match="canonical blob liveness projection blocked"):
        referenced_blob_hashes(source_db)


def test_source_capabilities_keep_versioned_schema_fail_closed(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        capabilities = blob_integrity._source_schema_capabilities(conn)

    assert capabilities.kind == "current_versioned"
    assert capabilities.current_authority is True


def test_source_capabilities_project_legacy_raw_only_carrier(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    expected_hash = "a" * 64
    with sqlite3.connect(source_db) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY, blob_hash BLOB NOT NULL);
            """
        )
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, blob_hash) VALUES (?, ?)", ("raw-1", bytes.fromhex(expected_hash))
        )

        capabilities = blob_integrity._source_schema_capabilities(conn)

    assert capabilities.kind == "legacy_raw_only"
    assert capabilities.legacy_carriers == ("raw_sessions",)
    assert referenced_blob_hashes(source_db, require_index=False) == [expected_hash]


def test_source_capabilities_conserve_mixed_legacy_and_typed_references(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    raw_hash = "b" * 64
    sidecar_hash = "c" * 64
    ledger_hash = "d" * 64
    with sqlite3.connect(source_db) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY, blob_hash BLOB NOT NULL);
            CREATE TABLE raw_hook_events (hook_event_id TEXT PRIMARY KEY);
            CREATE TABLE history_sidecars (sidecar_id TEXT PRIMARY KEY, blob_hash BLOB NOT NULL);
            CREATE TABLE blob_refs (
                blob_hash BLOB NOT NULL,
                ref_id TEXT NOT NULL,
                ref_type TEXT NOT NULL,
                PRIMARY KEY (blob_hash, ref_type, ref_id)
            );
            """
        )
        conn.execute("INSERT INTO raw_sessions (raw_id, blob_hash) VALUES (?, ?)", ("raw-1", bytes.fromhex(raw_hash)))
        conn.execute(
            "INSERT INTO history_sidecars (sidecar_id, blob_hash) VALUES (?, ?)",
            ("sidecar-1", bytes.fromhex(sidecar_hash)),
        )
        conn.execute(
            "INSERT INTO blob_refs (blob_hash, ref_id, ref_type) VALUES (?, ?, ?)",
            (bytes.fromhex(raw_hash), "raw-1", "raw_payload"),
        )
        conn.execute(
            "INSERT INTO blob_refs (blob_hash, ref_id, ref_type) VALUES (?, ?, ?)",
            (bytes.fromhex(ledger_hash), "raw-gone", "attachment"),
        )
        capabilities = blob_integrity._source_schema_capabilities(conn)

    assert capabilities.kind == "mixed_transitional"
    assert capabilities.legacy_carriers == ("raw_sessions", "history_sidecars", "blob_refs")
    assert referenced_blob_hashes(source_db, require_index=False) == sorted((raw_hash, sidecar_hash, ledger_hash))


def test_source_catalog_failure_is_not_an_empty_reference_projection(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    source_db.write_bytes(b"not a sqlite database")

    with pytest.raises(RuntimeError, match="source tier referenced-hash query failed"):
        referenced_blob_hashes(source_db, require_index=False)


def test_scan_blob_reference_debt_reads_initialized_source_tier(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    store = BlobStore(tmp_path / "blob")
    present_hash, present_size = store.write_from_bytes(b"present")
    initialize_archive_database(source_db, ArchiveTier.SOURCE)
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "raw-present",
                "codex-session",
                "native-1",
                "/tmp/session.jsonl",
                bytes.fromhex(present_hash),
                present_size,
                1,
            ),
        )

    # A current source tier cannot silently classify attachment-only bytes as
    # unreferenced when the index authority is absent.
    with pytest.raises(RuntimeError, match="index tier is unavailable"):
        scan_blob_reference_debt(source_db, store=store)

    initialize_archive_database(index_db, ArchiveTier.INDEX)

    report = scan_blob_reference_debt(source_db, store=store)

    assert report.ok is True
    assert report.total_references_seen == 1
    assert report.reference_sources == {"source.db.raw_sessions": 1}


def _session_with_attachment(attachment: ParsedAttachment) -> ParsedSession:
    return ParsedSession(
        source_name=Provider.GEMINI,
        provider_session_id="s1",
        title="s1",
        messages=[
            ParsedMessage(
                provider_message_id="m0",
                role=Role.USER,
                text="here is a file",
                position=0,
                variant_index=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="here is a file")],
            )
        ],
        attachments=[attachment],
    )


def test_scan_attachment_coverage_never_counts_unfetched_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """83u.4: unfetched (blob_hash NULL) attachments are an honest floor, never debt."""

    index_db = tmp_path / "index.db"
    store = BlobStore(tmp_path / "blob")
    monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: store)
    conn = sqlite3.connect(index_db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    write_parsed_session_to_archive(
        conn,
        _session_with_attachment(
            ParsedAttachment(
                provider_attachment_id="att-remote",
                message_provider_id="m0",
                name="remote-file.txt",
                mime_type="text/plain",
                source_url="https://example.invalid/remote-file.txt",
            )
        ),
    )
    conn.close()

    report = scan_attachment_coverage(index_db, store=store)

    assert report.total_attachments == 1
    assert report.unfetched_count == 1
    assert report.acquired_count == 0
    assert report.acquired_missing_blob_count == 0
    assert report.acquired_unreachable_count == 0
    assert report.acquired_reachable_count == 0
    assert report.ok is True


def test_scan_attachment_coverage_uses_read_profile_that_rejects_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production scan opener permits reads and refuses every write class."""

    index_db = tmp_path / "index.db"
    conn = sqlite3.connect(index_db)
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    conn.close()

    real_open = open_readonly_connection
    attempted = (
        "INSERT INTO attachments DEFAULT VALUES",
        "UPDATE attachments SET acquisition_status = 'acquired'",
        "DELETE FROM attachments",
        "CREATE TABLE profile_probe (value INTEGER)",
        "PRAGMA user_version = 99",
        "ATTACH DATABASE ':memory:' AS profile_probe",
    )
    rejected: list[str] = []

    def open_and_probe(*args: object, **kwargs: object) -> sqlite3.Connection:
        conn = real_open(*args, **kwargs)  # type: ignore[arg-type]
        assert conn.execute("SELECT COUNT(*) FROM attachments").fetchone()[0] == 0
        for statement in attempted:
            with pytest.raises(sqlite3.DatabaseError):
                conn.execute(statement)
            rejected.append(statement)
        return conn

    monkeypatch.setattr(blob_integrity, "open_readonly_connection", open_and_probe)

    report = scan_attachment_coverage(index_db, store=BlobStore(tmp_path / "blob"))

    assert report.total_attachments == 0
    assert rejected == list(attempted)


def test_scan_attachment_coverage_flags_acquired_row_with_missing_blob_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An acquired attachment whose blob file vanished from the store is genuine debt."""

    index_db = tmp_path / "index.db"
    store = BlobStore(tmp_path / "blob")
    monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: store)
    payload = b"attachment bytes that will be deleted from disk"
    conn = sqlite3.connect(index_db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    session = _session_with_attachment(
        ParsedAttachment(
            provider_attachment_id="att-acquired",
            message_provider_id="m0",
            name="note.txt",
            mime_type="text/plain",
            inline_bytes=payload,
        )
    )
    blob_hash, blob_size = store.write_from_bytes(payload)
    write_parsed_session_to_archive(
        conn,
        session,
        preacquired_attachment_blobs={id(session.attachments[0]): (bytes.fromhex(blob_hash), blob_size, "acquired")},
    )
    expected_attachment_id = conn.execute("SELECT attachment_id FROM attachments").fetchone()["attachment_id"]
    conn.close()

    store.blob_path(blob_hash).unlink()

    report = scan_attachment_coverage(index_db, store=store, sample_size=5)

    assert report.total_attachments == 1
    assert report.acquired_count == 1
    assert report.acquired_missing_blob_count == 1
    assert report.acquired_missing_blob_sample == (expected_attachment_id,)
    assert report.unfetched_count == 0
    # Missing blob bytes is a distinct dimension from reachability: this
    # attachment still has its attachment_refs row, it's just missing bytes.
    assert report.acquired_unreachable_count == 0
    assert report.acquired_reachable_count == 1
    assert report.ok is False


def test_scan_attachment_coverage_flags_acquired_row_with_no_attachment_ref(tmp_path: Path) -> None:
    """polylogue-w06b: an acquired attachment row with zero attachment_refs
    rows is unreachable from every session/message read path even though its
    bytes are genuinely on disk -- ``acquisition_status='acquired'`` alone
    overstates real coverage. The writer path no longer produces such rows
    (see test_archive_tiers_write.py's
    test_full_replace_sweeps_removed_attachment_that_loses_its_last_ref), but
    the scanner must still classify one correctly if it's ever found (e.g. in
    an archive ingested before that fix landed).
    """

    index_db = tmp_path / "index.db"
    store = BlobStore(tmp_path / "blob")
    payload = b"orphaned but genuinely fetched bytes"
    blob_hash, blob_size = store.write_from_bytes(payload)

    conn = sqlite3.connect(index_db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    conn.execute(
        """
        INSERT INTO attachments (
            attachment_id, display_name, media_type, byte_count, blob_hash, acquisition_status, ref_count
        ) VALUES (?, ?, ?, ?, ?, 'acquired', 1)
        """,
        ("orphan-att-1", "orphan.txt", "text/plain", blob_size, bytes.fromhex(blob_hash)),
    )
    conn.commit()
    conn.close()

    report = scan_attachment_coverage(index_db, store=store, sample_size=5)

    assert report.total_attachments == 1
    assert report.acquired_count == 1
    assert report.acquired_missing_blob_count == 0  # bytes ARE on disk
    assert report.acquired_unreachable_count == 1
    assert report.acquired_unreachable_sample == ("orphan-att-1",)
    assert report.acquired_reachable_count == 0
    # The stale non-zero ref_count is what makes this debt: refs existed when
    # the sweep last ran and then went away without it running again. A row
    # inserted with ref_count 0 is the writer's typed unowned retention and is
    # reported as `unowned_count` instead.
    assert report.acquired_unowned_count == 0
    # `ok` only tracks missing blob bytes (a distinct dimension) -- bytes ARE
    # present on disk here, so `ok` stays True even though the row is
    # unreachable. Reachability debt is its own signal
    # (acquired_unreachable_count), deliberately not folded into `ok`.
    assert report.ok is True


def test_acquired_coverage_partitioned_by_stored_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """polylogue-o0uw5: ``acquired`` is a claim; the store decides it.

    The 2026-09-14 pre-wipe census found 1240 of 1446 ``acquired`` attachment
    hashes with no bytes behind them, and this report answered "1446
    acquired". Three rows that all claim ``acquired`` must land in three
    different terms: corroborated, contradicted (a hash the store does not
    hold), and unverifiable (``acquired`` with no hash to check at all).

    Anti-vacuity: the unverifiable row is the one the scanner used to skip
    outright. Restore ``if blob_hash is None: continue`` and this test goes
    red on ``acquired_unverifiable_count`` and on ``ok`` -- an unchecked claim
    counted as a verified one. Dropping the ``with_bytes`` tally, or folding
    contradicted rows back into it, fails the first assertion. The
    corroborated row is asserted to stay ``acquired`` with its bytes, so
    "call every acquired row unverifiable" also fails.
    """

    index_db = tmp_path / "index.db"
    store = BlobStore(tmp_path / "blob")
    monkeypatch.setattr("polylogue.storage.blob_store.get_blob_store", lambda: store)
    payloads = {
        "att-corroborated": b"bytes that stay in the store",
        "att-contradicted": b"bytes the store will lose",
        "att-unverifiable": b"bytes whose identity is erased",
    }
    attachments = [
        ParsedAttachment(
            provider_attachment_id=name,
            message_provider_id="m0",
            name=f"{name}.txt",
            mime_type="text/plain",
            inline_bytes=payload,
        )
        for name, payload in payloads.items()
    ]
    session = ParsedSession(
        source_name=Provider.GEMINI,
        provider_session_id="coverage-partition",
        title="coverage-partition",
        messages=[
            ParsedMessage(
                provider_message_id="m0",
                role=Role.USER,
                text="three files",
                position=0,
                variant_index=0,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="three files")],
            )
        ],
        attachments=attachments,
    )
    preacquired: dict[int, tuple[bytes | None, int, str]] = {}
    hashes: dict[str, str] = {}
    for attachment in session.attachments:
        assert attachment.inline_bytes is not None
        blob_hash, size = store.write_from_bytes(attachment.inline_bytes)
        preacquired[id(attachment)] = (bytes.fromhex(blob_hash), size, "acquired")
        hashes[attachment.provider_attachment_id] = blob_hash

    conn = sqlite3.connect(index_db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    write_parsed_session_to_archive(conn, session, preacquired_attachment_blobs=preacquired)
    ids = {
        str(row["display_name"]): str(row["attachment_id"])
        for row in conn.execute("SELECT attachment_id, display_name FROM attachments")
    }
    # The census shape: the row keeps its hash, the store loses the bytes.
    store.blob_path(hashes["att-contradicted"]).unlink()
    # The unverifiable shape: an acquisition claim with no identity to check.
    conn.execute("UPDATE attachments SET blob_hash = NULL WHERE attachment_id = ?", (ids["att-unverifiable.txt"],))
    store.blob_path(hashes["att-unverifiable"]).unlink()
    conn.commit()
    assert [row[0] for row in conn.execute("SELECT DISTINCT acquisition_status FROM attachments")] == ["acquired"]
    conn.close()

    report = scan_attachment_coverage(index_db, store=store, sample_size=5)

    assert report.acquired_count == 3
    assert report.acquired_with_bytes_count == 1
    assert report.acquired_missing_blob_count == 1
    assert report.acquired_missing_blob_sample == (ids["att-contradicted.txt"],)
    assert report.acquired_unverifiable_count == 1
    assert report.acquired_unverifiable_sample == (ids["att-unverifiable.txt"],)
    assert report.ok is False
    # Criterion 3: the row whose bytes are present is untouched by any of this.
    assert store.verify(hashes["att-corroborated"]) is True
    payload = report.to_dict()
    assert payload["acquired_with_bytes_count"] == 1
    assert payload["acquired_unverifiable_count"] == 1


def test_unverifiable_acquired_row_alone_is_not_ok(tmp_path: Path) -> None:
    """A lone ``acquired`` row with no blob hash cannot certify coverage.

    Anti-vacuity: before this, the scanner skipped hashless acquired rows, so
    this archive reported ``acquired_count=1``, ``acquired_missing_blob_count=0``
    and ``ok is True`` -- a fully green coverage answer for an archive that
    checked nothing. Restore the skip and the last three assertions go red.
    """

    index_db = tmp_path / "index.db"
    store = BlobStore(tmp_path / "blob")
    conn = sqlite3.connect(index_db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    initialize_archive_tier(conn, ArchiveTier.INDEX)
    conn.execute(
        """
        INSERT INTO attachments (
            attachment_id, display_name, media_type, byte_count, blob_hash, acquisition_status, ref_count
        ) VALUES ('att-no-identity', 'ghost.txt', 'text/plain', 0, NULL, 'acquired', 0)
        """
    )
    conn.commit()
    conn.close()

    report = scan_attachment_coverage(index_db, store=store, sample_size=5)

    assert report.acquired_count == 1
    assert report.acquired_with_bytes_count == 0
    assert report.acquired_missing_blob_count == 0
    assert report.acquired_unverifiable_count == 1
    assert report.acquired_unverifiable_sample == ("att-no-identity",)
    assert report.ok is False


def test_classify_blob_reference_debt_groups_recovery_evidence(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    store = BlobStore(tmp_path / "blob")
    source_file = tmp_path / "exports" / "chatgpt.json"
    source_file.parent.mkdir()
    source_file.write_text("{}", encoding="utf-8")
    present_hash, present_size = store.write_from_bytes(b"present")
    missing_raw_hash = "1" * 64
    missing_attachment_hash = "2" * 64
    orphan_ref_hash = "3" * 64

    with sqlite3.connect(source_db) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                native_id TEXT,
                source_path TEXT,
                blob_hash BLOB,
                blob_size INTEGER NOT NULL,
                parse_error TEXT,
                validation_status TEXT
            );
            CREATE TABLE blob_refs (
                blob_hash BLOB NOT NULL,
                raw_id TEXT NOT NULL,
                ref_type TEXT NOT NULL,
                source_path TEXT,
                size_bytes INTEGER NOT NULL,
                PRIMARY KEY(blob_hash, raw_id, ref_type)
            );
            """
        )
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, validation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "raw-present",
                "codex-session",
                "native-present",
                str(source_file),
                bytes.fromhex(present_hash),
                present_size,
                "passed",
            ),
        )
        conn.execute(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size, validation_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "raw-missing",
                "chatgpt-export",
                "native-missing",
                str(source_file),
                bytes.fromhex(missing_raw_hash),
                123,
                "passed",
            ),
        )
        conn.execute(
            "INSERT INTO blob_refs (blob_hash, raw_id, ref_type, source_path, size_bytes) VALUES (?, ?, ?, ?, ?)",
            (bytes.fromhex(missing_attachment_hash), "raw-missing", "attachment", str(source_file), 456),
        )
        conn.execute(
            "INSERT INTO blob_refs (blob_hash, raw_id, ref_type, source_path, size_bytes) VALUES (?, ?, ?, ?, ?)",
            (bytes.fromhex(orphan_ref_hash), "raw-gone", "raw_payload", str(tmp_path / "missing.json"), 789),
        )

    report = classify_blob_reference_debt(source_db, store=store, sample_size=2, group_limit=3)

    assert report.ok is False
    assert report.distinct_referenced_blobs == 4
    assert report.reference_rows == 4
    assert report.missing_distinct_blobs == 3
    assert report.missing_by_table == {"blob_refs": 2, "raw_sessions": 1}
    assert report.missing_by_ref_type == {"attachment": 1, "raw_payload": 2}
    assert report.missing_by_origin == {"(none)": 1, "chatgpt-export": 2}
    assert report.missing_ref_id_join == {"ref_id_has_raw_session": 2, "ref_id_without_raw_session": 1}
    assert report.missing_source_path_presence == {
        "recoverable_source_path_exists": 2,
        "source_path_missing": 1,
    }
    assert report.missing_validation_status == {"(none)": 1, "passed": 2}
    assert report.missing_parse_error == {"no_parse_error": 3}
    assert len(report.samples) == 2
    payload = report.to_dict()
    samples = payload["samples"]
    assert isinstance(samples, list)
    first_sample = samples[0]
    assert isinstance(first_sample, dict)
    assert first_sample["sample_source_available"] is True


def test_blob_recovery_rejects_duplicate_container_member_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_source = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(zip_source, "w") as archive:
        archive.writestr("conversations.json", b'{"first": true}')
        archive.writestr("conversations.json", b'{"second": true}')

    def fail_open(*args: object, **kwargs: object) -> object:
        raise AssertionError("rejected duplicate member must not be opened")

    monkeypatch.setattr(zipfile.ZipFile, "open", fail_open)
    payload, reason = blob_integrity._current_raw_payload_bytes(
        f"{zip_source}:conversations.json",
        0,
    )

    assert payload is None
    assert reason == "ambiguous_container_member"


def test_blob_recovery_admits_declared_non_json_sidecar(tmp_path: Path) -> None:
    zip_source = tmp_path / "claude.zip"
    payload = b"opaque tool result"
    with zipfile.ZipFile(zip_source, "w") as archive:
        archive.writestr("session/tool-results/toolu.txt", payload)

    recovered, reason = blob_integrity._current_raw_payload_bytes(
        f"{zip_source}:session/tool-results/toolu.txt",
        0,
        provider_hint="claude-code",
    )

    assert reason is None
    assert recovered == payload


def test_blob_recovery_rejects_oversized_container_member_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zip_source = tmp_path / "oversized.zip"
    with zipfile.ZipFile(zip_source, "w") as archive:
        archive.writestr("conversations.json", b"{}")

    monkeypatch.setattr(zip_admission, "MAX_UNCOMPRESSED_SIZE", 1)
    monkeypatch.setattr(blob_integrity, "MAX_UNCOMPRESSED_SIZE", 1)

    def fail_open(*args: object, **kwargs: object) -> object:
        raise AssertionError("rejected oversized member must not be opened")

    monkeypatch.setattr(zipfile.ZipFile, "open", fail_open)
    payload, reason = blob_integrity._current_raw_payload_bytes(
        f"{zip_source}:conversations.json",
        0,
    )

    assert payload is None
    assert reason == "container_member_rejected"


def test_scan_blob_integrity_uses_sibling_archive_source_from_index_db(tmp_path: Path) -> None:
    index_db = tmp_path / "index.db"
    source_db = tmp_path / "source.db"
    store = BlobStore(tmp_path / "blob")
    raw_hash, raw_size = store.write_from_bytes(b"raw payload")
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
    with sqlite3.connect(source_db) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                blob_hash BLOB NOT NULL,
                blob_size INTEGER NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, blob_hash, blob_size) VALUES (?, ?, ?)",
            ("raw-v1", bytes.fromhex(raw_hash), raw_size),
        )

    report = scan_blob_integrity(index_db, store=store, full=True)

    assert report.ok is True
    assert report.total_references_seen == 1
    assert report.scanned_references == 1


class TestSourcePathSurvivesAnArchiveRootMove:
    """A recorded absolute path outlives the root it was written under.

    Acquisition records absolute paths. When the archive root moves, those
    paths stop resolving while the material itself is present under the new
    root, and reporting it missing is false loss — the number that decides
    whether a prune was safe.
    """

    @staticmethod
    def _recorded_under(previous_root: Path, tail: str) -> str:
        return str(previous_root / tail)

    def test_a_path_under_a_previous_root_resolves_against_the_current_one(self, tmp_path: Path) -> None:
        """Anti-vacuity: drop the re-anchoring and this reports False."""
        archive_root = tmp_path / "state" / "polylogue"
        (archive_root / "inbox").mkdir(parents=True)
        (archive_root / "inbox" / "export.zip").write_bytes(b"payload")
        recorded = self._recorded_under(tmp_path / "db" / "polylogue", "inbox/export.zip")

        available, resolved = blob_integrity._source_path_availability(recorded, archive_root)

        assert available is True
        assert resolved == str(archive_root / "inbox" / "export.zip")

    def test_a_zip_member_under_a_previous_root_resolves_through_its_container(self, tmp_path: Path) -> None:
        archive_root = tmp_path / "state" / "polylogue"
        (archive_root / "inbox").mkdir(parents=True)
        (archive_root / "inbox" / "export.zip").write_bytes(b"payload")
        recorded = self._recorded_under(tmp_path / "db" / "polylogue", "inbox/export.zip") + ":conversations.json"

        available, resolved = blob_integrity._source_path_availability(recorded, archive_root)

        assert available is True
        assert resolved == str(archive_root / "inbox" / "export.zip")

    def test_material_absent_under_both_roots_stays_missing(self, tmp_path: Path) -> None:
        """Re-anchoring must not invent evidence: a real absence stays absent."""
        archive_root = tmp_path / "state" / "polylogue"
        (archive_root / "inbox").mkdir(parents=True)
        recorded = self._recorded_under(tmp_path / "db" / "polylogue", "inbox/never-acquired.zip")

        available, _resolved = blob_integrity._source_path_availability(recorded, archive_root)

        assert available is False

    def test_a_path_outside_every_archive_owned_directory_is_not_re_anchored(self, tmp_path: Path) -> None:
        """A harness path is not archive-owned; re-anchoring it would be a guess."""
        archive_root = tmp_path / "state" / "polylogue"
        archive_root.mkdir(parents=True)
        (archive_root / "projects").mkdir()
        (archive_root / "projects" / "session.jsonl").write_bytes(b"payload")
        recorded = str(tmp_path / "home" / "projects" / "session.jsonl")

        available, _resolved = blob_integrity._source_path_availability(recorded, archive_root)

        assert available is False


def test_oversized_non_container_source_is_refused_by_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A grown source is refused, never read whole or hashed as a prefix.

    Anti-vacuity: with the unbounded ``path.read_bytes()`` restored, the call
    returns the full payload and ``reason is None``, so both assertions fail.
    The fixture must exceed the ceiling, or the bounded and unbounded reads
    agree and the test proves nothing.
    """
    monkeypatch.setattr(blob_integrity, "MAX_UNCOMPRESSED_SIZE", 64)
    source = tmp_path / "grown.jsonl"
    source.write_bytes(b"x" * 65)

    payload, reason = blob_integrity._current_raw_payload_bytes(str(source), None)

    assert payload is None
    assert reason == "source_too_large"


def test_source_at_the_ceiling_is_still_returned_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal is a ceiling, not an off-by-one that drops valid sources."""
    monkeypatch.setattr(blob_integrity, "MAX_UNCOMPRESSED_SIZE", 64)
    source = tmp_path / "exact.jsonl"
    source.write_bytes(b"y" * 64)

    payload, reason = blob_integrity._current_raw_payload_bytes(str(source), None)

    assert reason is None
    assert payload == b"y" * 64


def test_generation_resolved_index_scans_the_configured_blob_root(tmp_path: Path) -> None:
    """Blob debt is scanned under ``configured_root``, not the index generation.

    Anti-vacuity: with the store derived from ``db_path.parent`` again, the
    scan looks under ``.index-generations/<gen>/blob``, which does not exist,
    so the present blob counts missing and ``missing_referenced_blobs`` is 1
    instead of 0. The fixture must place ``index.db`` in a generation
    directory and reference a blob that really exists at the configured root,
    or the two roots coincide and the test proves nothing.
    """
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    store = BlobStore(archive_root / "blob")
    present_hash, present_size = store.write_from_bytes(b"durable blob bytes")

    source_db = archive_root / "source.db"
    with sqlite3.connect(source_db) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                blob_hash BLOB NOT NULL,
                blob_size INTEGER NOT NULL
            );
            CREATE TABLE blob_refs (
                blob_hash BLOB NOT NULL,
                raw_id TEXT NOT NULL,
                ref_type TEXT NOT NULL,
                PRIMARY KEY(blob_hash, raw_id, ref_type)
            );
            """
        )
        conn.execute(
            "INSERT INTO raw_sessions (raw_id, blob_hash, blob_size) VALUES (?, ?, ?)",
            ("raw-present", bytes.fromhex(present_hash), present_size),
        )

    generation_dir = archive_root / ".index-generations" / "gen-1"
    generation_dir.mkdir(parents=True, exist_ok=True)
    index_db = generation_dir / "index.db"
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")

    report = scan_blob_reference_debt(index_db, configured_root=archive_root)

    assert report.total_references_seen == 1
    assert report.missing_referenced_blobs == 0

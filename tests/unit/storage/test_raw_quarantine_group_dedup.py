"""polylogue-zm4w8: fully-quarantined byte-identical group dedup classifier.

Proves the read-only classifier scopes strictly to (source_path, blob_hash)
groups among quarantined raw_sessions rows where every member is quarantined
and NONE of them (nor any other raw sharing that blob_hash anywhere) already
has a materialized index.db session or a non-quarantined revision_authority
-- the residual population raw-byte-duplicate-supersession (which requires
an already-indexed twin) cannot see by construction.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from polylogue.core.enums import Provider
from polylogue.storage.raw_quarantine_group_dedup import plan_raw_quarantine_group_dedup
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _write_quarantined_raw(archive: ArchiveStore, *, raw_id: str, payload: bytes, source_path: str) -> None:
    """Write a raw the way real acquisition does: no revision envelope, default quarantined authority."""
    archive.write_raw_payload(
        provider=Provider.CODEX,
        payload=payload,
        source_path=source_path,
        source_index=-1,
        acquired_at_ms=1_700_000_000_000,
        raw_id=raw_id,
    )


def _index_session(conn: sqlite3.Connection, *, raw_id: str, native_id: str) -> None:
    conn.execute(
        "INSERT INTO sessions (origin, native_id, content_hash, raw_id, created_at_ms, updated_at_ms) "
        "VALUES ('codex-session', ?, ?, ?, 0, 0)",
        (native_id, b"\x22" * 32, raw_id),
    )


def test_flags_fully_quarantined_same_source_path_group(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    repeated_payload = b'{"messages":["repeated"]}\n'
    unique_payload = b'{"messages":["unique"]}\n'

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        # Three separate acquisitions of the SAME source file, byte-identical
        # -- the target population. Deliberately written raw-c, raw-a, raw-b
        # out of id order to prove representative selection is by raw_id
        # value, not insertion order.
        for raw_id in ("raw-c", "raw-a", "raw-b"):
            _write_quarantined_raw(
                archive,
                raw_id=raw_id,
                payload=repeated_payload,
                source_path=str(tmp_path / "repeated.jsonl"),
            )
        # A single, never-repeated acquisition -- not a group (count == 1).
        _write_quarantined_raw(
            archive,
            raw_id="raw-singleton",
            payload=unique_payload,
            source_path=str(tmp_path / "singleton.jsonl"),
        )
        archive.commit()

    source_conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    index_conn = sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)
    try:
        plan = plan_raw_quarantine_group_dedup(source_conn, index_conn)
    finally:
        source_conn.close()
        index_conn.close()

    assert plan.scanned_count == 4
    assert len(plan.groups) == 1
    group = plan.groups[0]
    assert group.source_path == str(tmp_path / "repeated.jsonl")
    assert group.raw_ids == ("raw-a", "raw-b", "raw-c")
    assert group.representative_raw_id == "raw-a"
    assert group.duplicate_raw_ids == ("raw-b", "raw-c")
    assert group.blob_size == len(repeated_payload)
    assert plan.duplicate_count == 2
    assert plan.duplicate_bytes == len(repeated_payload) * 2
    assert plan.already_resolved_group_count == 0


def test_different_source_paths_never_group_even_if_byte_identical(tmp_path: Path) -> None:
    """Same bytes, different source_path -- not this classifier's group key
    (that would be raw-byte-duplicate-supersession's territory if one side
    were indexed; here neither is, so it stays untouched by both)."""
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    shared_payload = b'{"messages":["shared-bytes-different-files"]}\n'

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        _write_quarantined_raw(archive, raw_id="raw-x", payload=shared_payload, source_path=str(tmp_path / "x.jsonl"))
        _write_quarantined_raw(archive, raw_id="raw-y", payload=shared_payload, source_path=str(tmp_path / "y.jsonl"))
        archive.commit()

    source_conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    index_conn = sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)
    try:
        plan = plan_raw_quarantine_group_dedup(source_conn, index_conn)
    finally:
        source_conn.close()
        index_conn.close()

    assert plan.scanned_count == 2
    assert plan.groups == ()
    assert plan.already_resolved_group_count == 0


def test_group_with_indexed_twin_elsewhere_is_already_resolved_not_flagged(tmp_path: Path) -> None:
    """raw-byte-duplicate-supersession-apply's own territory: if ANY raw
    sharing this blob_hash (any source_path) already has a materialized
    session, this classifier must defer to that actuator, not double-act.
    """
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    payload = b'{"messages":["already-has-an-indexed-twin"]}\n'

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for raw_id in ("raw-dup-1", "raw-dup-2"):
            _write_quarantined_raw(archive, raw_id=raw_id, payload=payload, source_path=str(tmp_path / "twinned.jsonl"))
        # A THIRD raw, different source_path, same bytes -- and this one is
        # indexed. The group above must not be flagged.
        _write_quarantined_raw(
            archive, raw_id="raw-indexed-elsewhere", payload=payload, source_path=str(tmp_path / "elsewhere.jsonl")
        )
        archive.commit()
        _index_session(archive._conn, raw_id="raw-indexed-elsewhere", native_id="native-indexed-elsewhere")
        archive.commit()

    source_conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    index_conn = sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)
    try:
        plan = plan_raw_quarantine_group_dedup(source_conn, index_conn)
    finally:
        source_conn.close()
        index_conn.close()

    assert plan.scanned_count == 3
    assert plan.groups == ()
    assert plan.already_resolved_group_count == 1


def test_limit_caps_number_of_groups_returned(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for group_index in range(3):
            payload = f'{{"n":{group_index}}}\n'.encode()
            for member_index in range(2):
                _write_quarantined_raw(
                    archive,
                    raw_id=f"raw-{group_index}-{member_index}",
                    payload=payload,
                    source_path=str(tmp_path / f"group-{group_index}.jsonl"),
                )
        archive.commit()

    source_conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    index_conn = sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)
    try:
        plan = plan_raw_quarantine_group_dedup(source_conn, index_conn, limit=2)
    finally:
        source_conn.close()
        index_conn.close()

    assert len(plan.groups) == 2


def test_limit_zero_returns_no_groups(tmp_path: Path) -> None:
    """Regression (CodeRabbit PR #3697): the cap must be checked BEFORE a
    group is appended, not after -- an after-the-fact check silently
    appended exactly one group even when the caller explicitly asked for
    zero via limit=0. The apply path iterates plan.groups directly, so this
    bug would have promoted and marked one duplicate group despite a
    limit=0 dry-run/apply call asking for none."""
    archive_root = tmp_path / "archive"
    initialize_active_archive_root(archive_root)

    with ArchiveStore.open_existing(archive_root, read_only=False) as archive:
        for raw_id in ("raw-a", "raw-b"):
            _write_quarantined_raw(
                archive, raw_id=raw_id, payload=b'{"n":0}\n', source_path=str(tmp_path / "group.jsonl")
            )
        archive.commit()

    source_conn = sqlite3.connect(f"file:{archive_root / 'source.db'}?mode=ro", uri=True)
    index_conn = sqlite3.connect(f"file:{archive_root / 'index.db'}?mode=ro", uri=True)
    try:
        plan = plan_raw_quarantine_group_dedup(source_conn, index_conn, limit=0)
    finally:
        source_conn.close()
        index_conn.close()

    assert plan.groups == ()

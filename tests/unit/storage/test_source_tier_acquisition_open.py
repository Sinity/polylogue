"""Source-tier acquisition open mode (polylogue-gbs02).

Acquire-only degraded mode must be able to admit raw evidence while the
derived index tier is at an older schema version, without ever opening —
let alone writing — index.db. These tests exercise the production
``ArchiveStore`` open modes against a real archive whose index tier is then
aged, plus the durable-tier refusal that keeps the mode from masking real
corruption risk.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.core.errors import SchemaSkew
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.bootstrap import archive_tier_spec
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _set_user_version(db: Path, version: int) -> None:
    conn = sqlite3.connect(db)
    try:
        conn.execute(f"PRAGMA user_version = {int(version)}")
        conn.commit()
    finally:
        conn.close()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def stale_index_root(workspace_env: dict[str, Path]) -> Iterator[Path]:
    root = workspace_env["archive_root"]
    conn = sqlite3.connect(root / "index.db")
    try:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        changed = conn.execute(
            "UPDATE schema_identity SET identity = ? WHERE tier = 'index'", ("synthetic-stale-index",)
        )
        assert changed.rowcount == 1
        conn.commit()
        assert (
            conn.execute("SELECT identity FROM schema_identity WHERE tier='index'").fetchone()[0]
            == "synthetic-stale-index"
        )
        yield root
    finally:
        conn.close()


def test_ordinary_writer_open_refuses_stale_index(stale_index_root: Path) -> None:
    files = (stale_index_root / "index.db", stale_index_root / "index.db-wal")
    before = {str(path): path.read_bytes() for path in files if path.exists()}
    with pytest.raises(SchemaSkew) as refused:
        ArchiveStore.open_existing(stale_index_root, read_only=False)
    assert refused.value.tier == "index"
    assert {str(path): path.read_bytes() for path in files if path.exists()} == before


def test_session_delete_refuses_identity_changed_after_store_open(workspace_env: dict[str, Path]) -> None:
    """The delete connection must admit its own current identity before DDL."""
    root = workspace_env["archive_root"]
    with ArchiveStore.open_existing(root, read_only=False) as archive:
        conn = sqlite3.connect(root / "index.db")
        try:
            conn.execute(
                "INSERT INTO sessions(native_id, origin, content_hash) "
                "VALUES ('retained', 'codex-session', zeroblob(32))"
            )
            changed = conn.execute(
                "UPDATE schema_identity SET identity=? WHERE tier='index'", ("synthetic-stale-index",)
            )
            assert changed.rowcount == 1
            conn.commit()
            assert (
                conn.execute("SELECT identity FROM schema_identity WHERE tier='index'").fetchone()[0]
                == "synthetic-stale-index"
            )
            files = (root / "index.db", root / "index.db-wal")
            before = {str(path): path.read_bytes() for path in files if path.exists()}
            with pytest.raises(SchemaSkew):
                archive.delete_sessions(("codex-session:retained",))
            assert conn.execute("SELECT COUNT(*) FROM sessions WHERE native_id='retained'").fetchone()[0] == 1
            assert {str(path): path.read_bytes() for path in files if path.exists()} == before
        finally:
            conn.close()


def test_source_tier_acquisition_opens_and_admits_raw(stale_index_root: Path) -> None:
    index_db = stale_index_root / "index.db"
    index_digest_before = _file_digest(index_db)

    with ArchiveStore.open_source_tier_acquisition(stale_index_root) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=b'{"type":"summary","summary":"acquired in degraded mode"}\n',
            source_path=str(stale_index_root / "inbox" / "session.jsonl"),
            canonical_source_path=str(stale_index_root / "inbox" / "session.jsonl"),
            acquired_at_ms=1_754_200_000_000,
        )
    assert raw_id

    source_conn = sqlite3.connect(f"file:{stale_index_root / 'source.db'}?mode=ro", uri=True)
    try:
        row = source_conn.execute(
            "SELECT parsed_at_ms FROM raw_sessions WHERE raw_id = ?",
            (raw_id,),
        ).fetchone()
    finally:
        source_conn.close()
    assert row is not None, "raw admission must land a raw_sessions row"
    assert row[0] is None, "acquire-only admission must leave the raw unparsed (convergence backlog)"

    # The stale derived tier must be byte-identical: no handle was ever opened.
    assert _file_digest(index_db) == index_digest_before


def test_source_tier_acquisition_index_access_raises(stale_index_root: Path) -> None:
    store = ArchiveStore.open_source_tier_acquisition(stale_index_root)
    try:
        with pytest.raises(RuntimeError, match="index tier is unavailable"):
            store.begin_read_snapshot()
        with pytest.raises(RuntimeError, match="index tier is unavailable"):
            store._conn.execute("SELECT 1")
    finally:
        store.close()


def test_source_tier_acquisition_refuses_stale_durable_tier(workspace_env: dict[str, Path]) -> None:
    root = workspace_env["archive_root"]
    _set_user_version(root / "source.db", archive_tier_spec(ArchiveTier.SOURCE).version + 1)
    with pytest.raises(SchemaSkew) as refusal:
        ArchiveStore.open_source_tier_acquisition(root)
    assert refusal.value.tier == "source"

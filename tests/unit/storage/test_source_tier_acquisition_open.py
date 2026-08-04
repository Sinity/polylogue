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
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
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
def stale_index_root(workspace_env: dict[str, Path]) -> Path:
    root = workspace_env["archive_root"]
    # A one-version-old index can hit a declared in-place fast-forward delta;
    # this mode exists for the SEMANTIC_REPARSE distance (rebuild required),
    # so age the index far enough that no fast-forward chain covers it —
    # v46 is the live pre-818fy generation, a known rebuild-only distance.
    _set_user_version(root / "index.db", 46)
    return root


def test_ordinary_writer_open_refuses_stale_index(stale_index_root: Path) -> None:
    with pytest.raises(RuntimeError, match="schema version"):
        ArchiveStore.open_existing(stale_index_root, read_only=False)


def test_source_tier_acquisition_opens_and_admits_raw(stale_index_root: Path) -> None:
    index_db = stale_index_root / "index.db"
    index_digest_before = _file_digest(index_db)

    with ArchiveStore.open_source_tier_acquisition(stale_index_root) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CLAUDE_CODE,
            payload=b'{"type":"summary","summary":"acquired in degraded mode"}\n',
            source_path=str(stale_index_root / "inbox" / "session.jsonl"),
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
    with pytest.raises(RuntimeError, match="durable tier source.db"):
        ArchiveStore.open_source_tier_acquisition(root)

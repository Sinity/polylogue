"""Collection decisions retain current Source custody through Index replacement."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing

import pytest

from polylogue.storage.blob_gc import _inspect_gc_protection
from polylogue.storage.blob_liveness import LivenessState


@pytest.fixture
def source() -> Iterator[sqlite3.Connection]:
    """Neutral durable reference relations; no archive or filesystem collector."""
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.executescript(
            "CREATE TABLE raw_sessions(raw_id TEXT,blob_hash BLOB);"
            "CREATE TABLE raw_hook_events(hook_event_id TEXT,blob_hash BLOB);"
            "CREATE TABLE material_observations(blob_hash BLOB);"
            "CREATE TABLE source_items(blob_hash BLOB);"
            "CREATE TABLE source_attachments(blob_hash BLOB);"
            "CREATE TABLE history_sidecars(sidecar_id TEXT);"
            "CREATE TABLE blob_refs(blob_hash BLOB,ref_type TEXT,ref_id TEXT);"
            "CREATE TABLE blob_publication_reservations(publication_id TEXT,blob_hash BLOB);"
        )
        yield connection


def _index() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE attachments(attachment_id TEXT,blob_hash BLOB)")
    return connection


@pytest.mark.parametrize("final_recheck", [False, True])
def test_complete_smaller_index_preserves_source_bytes_and_allows_unreferenced_candidate(
    source: sqlite3.Connection, final_recheck: bool
) -> None:
    """Source custody survives replacement while unrelated bytes remain eligible."""
    retained, retired, orphan = b"a" * 32, b"b" * 32, b"c" * 32
    source.execute("INSERT INTO raw_sessions VALUES ('raw',?)", (b"r" * 32,))
    source.execute("INSERT INTO blob_refs VALUES (?,'attachment','raw')", (retained,))
    source.execute("INSERT INTO source_attachments VALUES (?)", (retired,))
    with closing(_index()) as original, closing(_index()) as successor:
        original.executemany("INSERT INTO attachments VALUES (?,?)", [("kept", retained), ("retired", retired)])
        successor.execute("INSERT INTO attachments VALUES ('kept',?)", (retained,))
        for index in (original, successor):
            for blob_hash in (retained, retired):
                protection = _inspect_gc_protection(source, index, blob_hash.hex(), final_recheck=final_recheck)
                assert protection.is_live
                assert protection.blockers == ()
            protection = _inspect_gc_protection(source, index, orphan.hex(), final_recheck=final_recheck)
            assert protection.liveness.state is LivenessState.UNREFERENCED
            assert protection.reservation.state is LivenessState.UNREFERENCED
            assert not protection.is_live
            assert protection.blockers == ()


@pytest.mark.parametrize("final_recheck", [False, True])
@pytest.mark.parametrize("missing_tier", ["source", "index"])
def test_missing_current_reference_evidence_refuses_collection(
    source: sqlite3.Connection, final_recheck: bool, missing_tier: str
) -> None:
    """A smaller valid population never makes missing owner evidence equivalent to absence."""
    with closing(_index()) as index:
        (source if missing_tier == "source" else index).execute(
            "DROP TABLE source_attachments" if missing_tier == "source" else "DROP TABLE attachments"
        )
        protection = _inspect_gc_protection(source, index, (b"a" * 32).hex(), final_recheck=final_recheck)
        assert protection.liveness.state is LivenessState.BLOCKED
        assert protection.blockers


def test_final_recheck_retains_republication_reserved_after_planning(source: sqlite3.Connection) -> None:
    """A reservation committed after eligibility selection withholds final unlink authority."""
    blob_hash = b"a" * 32
    with closing(_index()) as index:
        planned = _inspect_gc_protection(source, index, blob_hash.hex(), final_recheck=False)
        assert not planned.is_live
        assert planned.blockers == ()
        source.execute("INSERT INTO blob_publication_reservations VALUES ('new-publication',?)", (blob_hash,))
        source.commit()
        final = _inspect_gc_protection(source, index, blob_hash.hex(), final_recheck=True)
        assert final.liveness.state is LivenessState.UNREFERENCED
        assert final.reservation.state is LivenessState.LIVE
        assert final.is_live
        assert final.blockers == ()


def test_final_recheck_retains_source_attachment_published_after_planning(source: sqlite3.Connection) -> None:
    """A durable acquisition committed after selection is checked by the final seam."""
    blob_hash = b"a" * 32
    with closing(_index()) as index:
        assert not _inspect_gc_protection(source, index, blob_hash.hex(), final_recheck=False).is_live
        source.execute("INSERT INTO source_attachments VALUES (?)", (blob_hash,))
        source.commit()
        final = _inspect_gc_protection(source, index, blob_hash.hex(), final_recheck=True)
        assert final.liveness.state is LivenessState.LIVE
        assert final.is_live
        assert final.blockers == ()

"""Finite Source reads preserve the canonical session classifier and cursor custody."""

from __future__ import annotations

import os
import sqlite3
from builtins import BaseExceptionGroup
from contextlib import closing
from pathlib import Path
from typing import Any, Self

import pytest

from polylogue.storage.blob_liveness import (
    BLOB_OWNERS,
    BlobLiveness,
    BlobOwner,
    ConnectionSessionBlobLivenessRead,
    LivenessState,
    SessionBlobLivenessSourceRead,
    inspect_session_blob_references,
    session_blob_direct_query,
    session_blob_ledger_query,
)
from polylogue.storage.io_phase_metrics import connect_measured, connection_cursor, live_connection_cursors
from polylogue.storage.sqlite import connection_profile
from polylogue.storage.sqlite.connection_profile import (
    NativeConnectionSettlementError,
    retained_native_sql_owners_for_lifetime,
)
from tests.infra.sqlite_cursor_settlement import ControlledConnection, ControlledCursor


def _relations(source: sqlite3.Connection, index: sqlite3.Connection) -> None:
    """Minimal neutral relations with actual canonical owner/referent columns."""
    source.executescript(
        "CREATE TABLE raw_sessions(raw_id TEXT,blob_hash BLOB);"
        "CREATE TABLE raw_hook_events(hook_event_id TEXT,blob_hash BLOB);"
        "CREATE TABLE material_observations(blob_hash BLOB);"
        "CREATE TABLE source_items(blob_hash BLOB);"
        "CREATE TABLE source_attachments(blob_hash BLOB);"
        "CREATE TABLE history_sidecars(sidecar_id TEXT);"
        "CREATE TABLE blob_refs(blob_hash BLOB,ref_type TEXT,ref_id TEXT);"
    )
    index.executescript(
        "CREATE TABLE attachments(attachment_id TEXT,blob_hash BLOB);"
        "CREATE TABLE attachment_refs(attachment_id TEXT,session_id TEXT);"
    )


class FiniteSourceRead:
    """Capture only named finite query results; no classification or mutation API."""

    def __init__(self, connection: sqlite3.Connection, hashes: tuple[bytes, ...]) -> None:
        ordinary = ConnectionSessionBlobLivenessRead(connection)
        self.blockers = ordinary.session_blob_global_blockers()
        self.available = {
            owner: ordinary.session_blob_owner_available(owner)
            for owner in BLOB_OWNERS
            if owner.tier == "source" and owner.ref_type is None
        }
        self.direct: dict[BlobOwner, tuple[bytes, ...]] = {}
        self.ledger: dict[BlobOwner, tuple[bytes, ...]] = {}
        self.chunks: list[tuple[bytes, ...]] = []
        self.served: list[tuple[bytes, ...]] = []
        if not self.blockers:
            for owner in BLOB_OWNERS:
                if owner.tier != "source":
                    continue
                if owner.ref_type is None and not self.available[owner]:
                    continue
                builder = session_blob_direct_query if owner.ref_type is None else session_blob_ledger_query
                with connection_cursor(connection, *builder(owner, hashes)) as cursor:
                    values = tuple(bytes(row[0]) for row in cursor)
                (self.direct if owner.ref_type is None else self.ledger)[owner] = values

    def session_blob_global_blockers(self) -> tuple[str, ...]:
        return self.blockers

    def session_blob_owner_available(self, owner: BlobOwner) -> bool:
        return self.available[owner]

    def session_blob_direct_hashes(self, owner: BlobOwner, hashes: tuple[bytes, ...]) -> tuple[bytes, ...]:
        self.chunks.append(hashes)
        values = tuple(value for value in self.direct[owner] if value in hashes)
        self.served.append(values)
        return values

    def session_blob_ledger_hashes(self, owner: BlobOwner, hashes: tuple[bytes, ...]) -> tuple[bytes, ...]:
        self.chunks.append(hashes)
        values = tuple(value for value in self.ledger[owner] if value in hashes)
        self.served.append(values)
        return values


def test_finite_source_read_matches_actual_owner_ledger_and_index_relations() -> None:
    """Wrong owner/typed referent, census inclusion, exclusion or chunking turns red."""
    hashes = tuple(value.to_bytes(32, "big") for value in range(1, 503))
    with closing(sqlite3.connect(":memory:")) as source, closing(sqlite3.connect(":memory:")) as index:
        _relations(source, index)
        source.execute("INSERT INTO raw_sessions VALUES ('raw',?)", (hashes[0],))
        source.execute("INSERT INTO raw_hook_events VALUES ('hook',?)", (hashes[1],))
        source.execute("INSERT INTO material_observations VALUES (?)", (hashes[2],))
        source.executemany("INSERT INTO source_items VALUES (?)", [(hashes[3],), (hashes[500],)])
        source.execute("INSERT INTO source_attachments VALUES (?)", (hashes[4],))
        source.execute("INSERT INTO history_sidecars VALUES ('sidecar')")
        source.executemany(
            "INSERT INTO blob_refs VALUES (?,?,?)",
            [
                (hashes[5], "raw_payload", "raw"),
                (hashes[6], "raw_payload", "missing"),
                (hashes[7], "hook_payload", "hook"),
                (hashes[8], "sidecar", "sidecar"),
                (hashes[9], "attachment", "raw"),
                (hashes[0], "attachment", "raw"),
            ],
        )
        index.executemany("INSERT INTO attachments VALUES (?,?)", [("outside", hashes[500]), ("inside", hashes[501])])
        index.executemany("INSERT INTO attachment_refs VALUES (?,?)", [("outside", "kept"), ("inside", "excised")])
        finite = FiniteSourceRead(source, hashes)

        def classify(reader: SessionBlobLivenessSourceRead) -> dict[bytes, BlobLiveness]:
            return inspect_session_blob_references(
                reader,
                (*hashes, hashes[0]),
                index_conn=index,
                excluding_session_ids=frozenset({"excised"}),
            )

        actual = classify(ConnectionSessionBlobLivenessRead(source))
        assert classify(finite) == actual
        assert source.in_transaction and index.in_transaction
        assert len(actual) == len(hashes)
        expected = {
            0: ("source.db.raw_sessions", "source.db.blob_refs"),
            1: ("source.db.raw_hook_events",),
            2: ("source.db.material_observations",),
            3: ("source.db.source_items",),
            5: ("source.db.blob_refs",),
            7: ("source.db.blob_refs",),
            8: ("source.db.blob_refs",),
            9: ("source.db.blob_refs",),
            500: ("source.db.source_items", "index.db.attachment_refs"),
        }
        for ordinal, blob_hash in enumerate(hashes):
            if ordinal in expected:
                assert actual[blob_hash] == BlobLiveness(LivenessState.LIVE, expected[ordinal])
            else:
                assert actual[blob_hash] == BlobLiveness(LivenessState.UNREFERENCED)
        assert {len(chunk) for chunk in finite.chunks} == {500, 2}
        assert any(hashes[500] in chunk for chunk in finite.chunks)
        assert any(hashes[500] in values for values in finite.served)
        assert all(len(chunk) == len(set(chunk)) for chunk in finite.chunks)


@pytest.mark.parametrize("damage", ["missing-table", "missing-column", "unknown-ref-type"])
def test_finite_source_read_preserves_schema_and_unknown_type_blockers(damage: str) -> None:
    with closing(sqlite3.connect(":memory:")) as source, closing(sqlite3.connect(":memory:")) as index:
        _relations(source, index)
        if damage == "missing-table":
            source.execute("DROP TABLE raw_sessions")
        elif damage == "missing-column":
            source.execute("ALTER TABLE raw_hook_events DROP COLUMN blob_hash")
        else:
            source.execute("INSERT INTO blob_refs VALUES (?, 'unrecognized', 'raw')", (b"b" * 32,))
        hashes = (b"a" * 32,)
        ordinary = ConnectionSessionBlobLivenessRead(source)
        if damage == "missing-column":
            with pytest.raises(sqlite3.OperationalError):
                inspect_session_blob_references(ordinary, hashes, index_conn=index, excluding_session_ids=frozenset())
            with pytest.raises(sqlite3.OperationalError):
                FiniteSourceRead(source, hashes)
            return
        actual = inspect_session_blob_references(ordinary, hashes, index_conn=index, excluding_session_ids=frozenset())
        assert (
            inspect_session_blob_references(
                FiniteSourceRead(source, hashes),
                hashes,
                index_conn=index,
                excluding_session_ids=frozenset(),
            )
            == actual
        )
        assert actual[hashes[0]].state is LivenessState.BLOCKED
        expected = {
            "missing-table": "source.raw_sessions is missing",
            "unknown-ref-type": "unknown blob_refs ref_type(s): unrecognized",
        }[damage]
        assert actual[hashes[0]].blockers == (expected,)
        assert expected in ordinary.session_blob_global_blockers()


def test_query_builders_reject_foreign_owner_before_sql_and_preserve_empty_selection() -> None:
    with closing(sqlite3.connect(":memory:")) as source, closing(sqlite3.connect(":memory:")) as index:
        _relations(source, index)
        reader = ConnectionSessionBlobLivenessRead(source)
        direct = BLOB_OWNERS[0]
        ledger = next(owner for owner in BLOB_OWNERS if owner.ref_type is not None)
        assert reader.session_blob_direct_hashes(direct, ()) == ()
        assert reader.session_blob_ledger_hashes(ledger, ()) == ()
        assert inspect_session_blob_references(reader, (), index_conn=index, excluding_session_ids=frozenset()) == {}
        for owner in (
            BlobOwner("source", "raw_sessions; DROP TABLE raw_sessions", blob_column="blob_hash"),
            BLOB_OWNERS[5],
            ledger,
        ):
            with pytest.raises(ValueError):
                reader.session_blob_direct_hashes(owner, (b"a" * 32,))
            with pytest.raises(ValueError):
                reader.session_blob_owner_available(owner)
        with pytest.raises(ValueError):
            reader.session_blob_ledger_hashes(direct, (b"a" * 32,))
        assert source.execute("SELECT count(*) FROM raw_sessions").fetchone()[0] == 0


def _contains(error: BaseException, target: BaseException) -> bool:
    return (
        error is target
        or isinstance(error, BaseExceptionGroup)
        and any(_contains(child, target) for child in error.exceptions)
    )


def _fds(path: Path) -> set[str]:
    found = set()
    for fd in Path("/proc/self/fd").iterdir():
        try:
            if os.readlink(fd) == str(path):
                found.add(fd.name)
        except FileNotFoundError:
            pass
    return found


@pytest.mark.parametrize("target", ["direct", "ledger", "global", "global-close", "schema", "index"])
def test_actual_liveness_cursor_read_failure_retains_owner_callback_until_creator_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    """Execute+blocked-close failure cannot release the original SQL owner or FD."""
    source_path, index_path = tmp_path / "source-read.sqlite", tmp_path / "index-read.sqlite"
    with closing(sqlite3.connect(source_path)) as source, closing(sqlite3.connect(index_path)) as index:
        _relations(source, index)
        source.execute("INSERT INTO raw_sessions VALUES ('raw',?)", (b"a" * 32,))
        source.execute("INSERT INTO blob_refs VALUES (?, 'raw_payload', 'raw')", (b"a" * 32,))
        source.commit()
        index.execute("INSERT INTO attachments VALUES ('attachment',?)", (b"a" * 32,))
        index.execute("INSERT INTO attachment_refs VALUES ('attachment','kept')")
        index.commit()
    real_connect = connect_measured

    def controlled(database: str | Path, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if str(source_path) in str(database) or str(index_path) in str(database):
            return sqlite3.connect(database, *args, factory=ControlledConnection, **kwargs)
        return real_connect(database, *args, **kwargs)

    monkeypatch.setattr(connection_profile, "connect_measured", controlled)
    dependency = object()
    path = index_path if target == "index" else source_path
    owner = connection_profile._open_readonly_owner(path, validate_schema=False, lifetime_dependencies=(dependency,))
    connection = owner.require_connection()
    assert isinstance(connection, ControlledConnection)
    primary = sqlite3.OperationalError("synthetic read-execute failure")
    retained: list[ControlledCursor] = []
    completed: list[bool] = []
    owner.retain_settlement_callback(lambda: completed.append(True))
    prefix = {
        "direct": "SELECT DISTINCT blob_hash FROM raw_sessions",
        "ledger": "SELECT DISTINCT ref.blob_hash",
        "global": "SELECT DISTINCT ref_type",
        "global-close": "SELECT DISTINCT ref_type",
        "schema": "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        "index": "SELECT DISTINCT a.blob_hash",
    }[target]

    class FaultCursor(ControlledCursor):
        def execute(self, sql: str, parameters: Any = (), /) -> Self:
            result = super().execute(sql, parameters)
            if sql.startswith(prefix) and (target != "schema" or parameters == ("blob_refs",)):
                retained.append(self)
                self.allow_cleanup.clear()
                if target == "global-close":
                    self.cleanup_failure = sqlite3.OperationalError("synthetic successful-read close failure")
                else:
                    raise primary
            return result

    real_cursor = connection.cursor
    monkeypatch.setattr(
        connection, "cursor", lambda factory=None: real_cursor(factory=FaultCursor if factory is None else factory)
    )
    other = sqlite3.connect(source_path if target == "index" else index_path)
    try:
        with pytest.raises(sqlite3.OperationalError if target == "global-close" else BaseExceptionGroup) as caught:
            inspect_session_blob_references(
                ConnectionSessionBlobLivenessRead(other if target == "index" else connection),
                (b"a" * 32,),
                index_conn=connection if target == "index" else other,
                excluding_session_ids=frozenset(),
            )
        if target != "global-close":
            assert _contains(caught.value, primary)
        assert len(retained) == 1
        assert _contains(caught.value, retained[0].cleanup_failure)
        assert retained[0] in live_connection_cursors(connection)
        assert _fds(path)
        with pytest.raises(NativeConnectionSettlementError):
            owner.close()
        assert owner in retained_native_sql_owners_for_lifetime(dependency)
        assert completed == []
        retained[0].allow_cleanup.set()
        owner.close()
        assert live_connection_cursors(connection) == ()
        assert retained_native_sql_owners_for_lifetime(dependency) == ()
        assert completed == [True]
        assert not _fds(path)
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
        assert other.execute("SELECT 1").fetchone()[0] == 1
    finally:
        for cursor in retained:
            cursor.allow_cleanup.set()
        owner.close()
        other.close()

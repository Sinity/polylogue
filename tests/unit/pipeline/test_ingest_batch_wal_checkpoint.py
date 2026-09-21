"""Ingest-batch upkeep: planner statistics, and no checkpointing of its own."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import polylogue.pipeline.services.ingest_batch._core as ingest_batch_core
from polylogue.core.enums import Provider
from polylogue.core.sources import origin_from_provider
from polylogue.pipeline.services.ingest_batch import _process_ingest_batch_sync
from polylogue.pipeline.services.ingest_worker import IngestRecordResult
from polylogue.storage.runtime import RawSessionRecord
from polylogue.storage.sqlite.connection import open_connection
from polylogue.storage.sqlite.wal_checkpoint import (
    WalCheckpointObservation,
    checkpoint_archive_wals,
    checkpoint_wal,
)


def test_format_foreign_key_violations_renders_tuple_rows() -> None:
    rendered = ingest_batch_core._format_foreign_key_violations(
        [
            ("messages", 42, "sessions", 0),
            ("blocks", 7, "messages", 1),
        ]
    )

    assert "sqlite3.Row object" not in rendered
    assert "'table': 'messages'" in rendered
    assert "'rowid': 42" in rendered
    assert "'parent': 'sessions'" in rendered
    assert "'fkid': 0" in rendered


def test_format_foreign_key_violations_renders_sqlite_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "fk.db"
    with open_connection(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("CREATE TABLE parent (id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE child (id TEXT PRIMARY KEY, parent_id TEXT REFERENCES parent(id))")
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("INSERT INTO child (id, parent_id) VALUES ('child-1', 'missing-parent')")
        rows = conn.execute("PRAGMA foreign_key_check").fetchall()

    rendered = ingest_batch_core._format_foreign_key_violations(rows)

    assert "sqlite3.Row object" not in rendered
    assert "'table': 'child'" in rendered
    assert "'parent': 'parent'" in rendered
    assert "'fkid': 0" in rendered


def test_scoped_foreign_key_check_ignores_preexisting_orphans(tmp_path: Path) -> None:
    db_path = tmp_path / "scoped-fk.db"
    with open_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (origin, native_id, content_hash) VALUES ('codex-session', 'old', zeroblob(32))"
        )
        conn.execute(
            "INSERT INTO sessions (origin, native_id, content_hash) VALUES ('codex-session', 'new', zeroblob(32))"
        )
        conn.commit()
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            """
            INSERT INTO blocks (message_id, session_id, position, block_type, text)
            VALUES ('codex-session:old:old-missing-message', 'codex-session:old', 0, 'text', 'old orphan')
            """
        )
        conn.commit()
        conn.execute("PRAGMA foreign_keys = ON")

        violations = ingest_batch_core._foreign_key_violations_for_sessions(conn, ("codex-session:new",))

    assert violations == []


def test_scoped_foreign_key_check_reports_current_session_orphans(tmp_path: Path) -> None:
    db_path = tmp_path / "scoped-fk.db"
    with open_connection(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions (origin, native_id, content_hash) VALUES ('codex-session', 'new', zeroblob(32))"
        )
        conn.commit()
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            """
            INSERT INTO blocks (message_id, session_id, position, block_type, text)
            VALUES ('codex-session:new:new-missing-message', 'codex-session:new', 0, 'text', 'new orphan')
            """
        )
        conn.commit()
        conn.execute("PRAGMA foreign_keys = ON")

        violations = ingest_batch_core._foreign_key_violations_for_sessions(conn, ("codex-session:new",))

    assert violations == [
        {
            "table": "blocks",
            "rowid": 1,
            "parent": "messages",
            "fkid": 0,
            "session_id": "codex-session:new",
            "child_key": {
                "message_id": "codex-session:new:new-missing-message",
                "session_id": "codex-session:new",
            },
        }
    ]


def _seed_two_sessions_and_one_message(conn: sqlite3.Connection) -> tuple[str, str]:
    """Return (message_id, block_id), both owned by ``codex-session:a``."""
    for native_id in ("a", "b"):
        conn.execute(
            "INSERT INTO sessions (origin, native_id, content_hash) VALUES ('codex-session', ?, zeroblob(32))",
            (native_id,),
        )
    conn.execute(
        """
        INSERT INTO messages
            (session_id, native_id, position, content_occurrence, role, content_identity, content_hash)
        VALUES ('codex-session:a', 'm1', 0, 0, 'user', 'identity', zeroblob(32))
        """
    )
    message_id = str(conn.execute("SELECT message_id FROM messages").fetchone()[0])
    conn.execute(
        """
        INSERT INTO blocks (message_id, session_id, position, block_type, text)
        VALUES (?, 'codex-session:a', 0, 'text', 'owned by a')
        """,
        (message_id,),
    )
    block_id = str(conn.execute("SELECT block_id FROM blocks").fetchone()[0])
    conn.commit()
    return message_id, block_id


_CONTRADICTORY_ROWS: dict[str, str] = {
    "blocks": """
        INSERT INTO blocks (message_id, session_id, position, block_type, text)
        VALUES (:message_id, 'codex-session:b', 1, 'text', 'two owners disagree')
        """,
    "web_content_constructs": """
        INSERT INTO web_content_constructs
            (session_id, message_id, block_id, position, provider, construct_type)
        VALUES ('codex-session:b', :message_id, :block_id, 0, 'openai', 'search_query')
        """,
    "action_pairs": """
        INSERT INTO action_pairs (tool_use_block_id, session_id, message_id)
        VALUES (:block_id, 'codex-session:b', :message_id)
        """,
}


@pytest.mark.parametrize("table", sorted(_CONTRADICTORY_ROWS))
def test_bulk_precommit_check_reports_a_compound_owner_disagreement(tmp_path: Path, table: str) -> None:
    """The bulk path's ``foreign_keys=OFF`` window must not commit two owners that disagree.

    ``message_id`` names a message of session ``a`` while ``session_id`` says
    ``b``. Each column alone resolves, so the pair of independent
    single-column ``NOT EXISTS`` probes this check used to run returned ``[]``
    while ``PRAGMA foreign_key_check`` returned the violation -- measured, and
    the reason a contradictory row committed silently on a large raw batch.

    Anti-vacuity, executed: narrow the compound existence check to its first
    column pair (``child_columns[:1]`` in ``_scoped_foreign_key_sql``) and all
    three parameters fail with ``assert [] == [(table, 'messages')]`` while
    ``PRAGMA foreign_key_check`` still reports the row -- the pre-fix
    behaviour, measured. Running this with ``foreign_keys=ON`` would not
    qualify: the INSERT is refused inline there and the pre-commit check is
    never consulted, which is asserted below.
    """
    db_path = tmp_path / f"compound-fk-{table}.db"
    with open_connection(db_path) as conn:
        message_id, block_id = _seed_two_sessions_and_one_message(conn)
        parameters = {"message_id": message_id, "block_id": block_id}

        # The constraint IS enforced inline when SQLite is checking it, so the
        # bulk window is the only route that can admit this row at all.
        conn.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(_CONTRADICTORY_ROWS[table], parameters)
        # The refused INSERT left an open transaction, and ``PRAGMA
        # foreign_keys`` is a silent no-op inside one.
        conn.rollback()

        conn.execute("PRAGMA foreign_keys = OFF")
        assert int(conn.execute("PRAGMA foreign_keys").fetchone()[0]) == 0
        conn.execute(_CONTRADICTORY_ROWS[table], parameters)

        violations = ingest_batch_core._foreign_key_violations_for_sessions(
            conn, ("codex-session:a", "codex-session:b")
        )
        pragma_rows = conn.execute("PRAGMA foreign_key_check").fetchall()
        conn.rollback()

    assert [(str(row[0]), str(row[2])) for row in pragma_rows] == [(table, "messages")]
    assert [(item["table"], item["parent"]) for item in violations] == [(table, "messages")]
    assert violations[0]["session_id"] == "codex-session:b"
    assert violations[0]["child_key"] == {"message_id": message_id, "session_id": "codex-session:b"}


def test_foreign_key_check_plan_is_total_over_the_schema(tmp_path: Path) -> None:
    """Every foreign key the schema declares is checked by one route or the other.

    Derivation, not enumeration, is the fix: the list this replaced named
    three tables (and the wrong parent for two of them) while the built schema
    declares foreign keys on far more, and ``#5333`` had already found one
    affected table that no written list mentioned.

    Anti-vacuity, executed: re-introduce any written-down table set and a
    table absent from it lands in neither partition, so ``uncovered`` is
    non-empty; drop the ``unscoped`` fallback and the four tables that name no
    session go unchecked.
    """
    db_path = tmp_path / "plan.db"
    with open_connection(db_path) as conn:
        scoped, unscoped = ingest_batch_core._foreign_key_check_plan(conn)
        declared = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")
            if conn.execute(f'PRAGMA foreign_key_list("{row[0]}")').fetchall()
        }
        declared_keys = {
            (table, int(fk[0]))
            for table in declared
            for fk in conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
        }

    covered = {check.table for check in scoped} | set(unscoped)
    assert declared - covered == set()
    assert {(check.table, check.fkid) for check in scoped} == {key for key in declared_keys if key[0] not in unscoped}
    # The tables that carry no session column at all, checked whole instead.
    assert set(unscoped) == {
        "attachment_native_ids",
        "repo_checkouts",
        "work_evidence_edges",
        "work_evidence_nodes",
    }


def test_compound_foreign_keys_are_probed_as_one_key(tmp_path: Path) -> None:
    """A compound key is one existence check over both columns, never two.

    Anti-vacuity, executed: generate one probe per column and ``child_columns``
    becomes length 1 for every entry below, which is exactly the shape that
    could not see the disagreement the test above reproduces.
    """
    db_path = tmp_path / "compound.db"
    with open_connection(db_path) as conn:
        scoped, _ = ingest_batch_core._foreign_key_check_plan(conn)
        compound = {check.table: check for check in scoped if len(check.child_columns) > 1}
        placeholders = "?"
        plans = {
            table: [
                str(row[3])
                for row in conn.execute(
                    "EXPLAIN QUERY PLAN " + ingest_batch_core._scoped_foreign_key_sql(check, placeholders),
                    ("codex-session:a",),
                ).fetchall()
            ]
            for table, check in compound.items()
        }

    assert set(compound) == {
        "action_pairs",
        "attachment_refs",
        "blocks",
        "file_edits",
        "paste_spans",
        "web_content_constructs",
    }
    for check in compound.values():
        assert check.parent == "messages"
        assert check.child_columns == ("message_id", "session_id")
        assert check.parent_columns == ("message_id", "session_id")
    # A per-batch pre-commit check must not itself scan the archive.
    for table, detail_rows in plans.items():
        assert not any(detail.upper().startswith("SCAN ") for detail in detail_rows), (table, detail_rows)


def test_process_ingest_batch_sync_does_not_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ingest publishes; the recurring coordinator checkpoints.

    A checkpoint reintroduced here would run inside the batch's own writer
    hold, where it is charged to publication and invisible to the checkpoint
    budget -- which is exactly what this asserts cannot happen.
    """
    db_path = tmp_path / "index.db"
    archive_root = tmp_path / "archive"
    blob_root = tmp_path / "blob"
    source_path = tmp_path / "raw.jsonl"
    source_path.write_text("{}", encoding="utf-8")
    raw_record = RawSessionRecord(
        raw_id="raw-wal",
        source_name="codex",
        source_path=str(source_path),
        blob_size=source_path.stat().st_size,
        acquired_at="2026-04-02T00:00:00Z",
    )

    origin = origin_from_provider(Provider.from_string(raw_record.source_name)).value
    with open_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions
                (raw_id, origin, native_id, source_path, source_index,
                 blob_hash, blob_size, acquired_at_ms)
            VALUES (?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (
                raw_record.raw_id,
                origin,
                raw_record.raw_id,
                raw_record.source_path,
                b"\x00" * 32,
                raw_record.blob_size,
                1_775_433_600_000,
            ),
        )
        conn.commit()

    def fake_ingest_record(
        record: RawSessionRecord,
        archive_root_str: str,
        validation_mode: str,
        measure_ingest_result_size: bool,
        *,
        blob_root_str: str | None,
    ) -> IngestRecordResult:
        del record, archive_root_str, validation_mode, measure_ingest_result_size, blob_root_str
        return IngestRecordResult(raw_id=raw_record.raw_id, sessions=[])

    def refuse_checkpoint(*_args: object, **_kwargs: object) -> WalCheckpointObservation:
        raise AssertionError("the ingest path must not checkpoint")

    optimize_calls: list[str] = []

    def fake_optimize(conn: object, *, reason: str) -> object:
        del conn
        optimize_calls.append(reason)
        return type("OptimizeObservation", (), {"error": None})()

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest_record)

    def fake_drain(*_: object, **kwargs: object) -> None:
        ensure_transaction = kwargs.get("ensure_index_transaction")
        assert callable(ensure_transaction)
        ensure_transaction()

    monkeypatch.setattr(ingest_batch_core, "_drain_ingest_result", fake_drain)
    monkeypatch.setattr("polylogue.storage.sqlite.wal_checkpoint.checkpoint_wal", refuse_checkpoint)
    monkeypatch.setattr("polylogue.storage.sqlite.wal_checkpoint.checkpoint_archive_wals", refuse_checkpoint)
    monkeypatch.setattr("polylogue.storage.sqlite.maintenance.maybe_optimize_sqlite", fake_optimize)

    summary = _process_ingest_batch_sync(
        [raw_record],
        db_path=db_path,
        archive_root_str=str(archive_root),
        blob_root_str=str(blob_root),
        validation_mode="off",
        ingest_workers=1,
        measure_ingest_result_size=False,
    )

    assert summary.raw_record_count == 1
    assert optimize_calls == ["ingest_batch_commit"]


def test_maybe_optimize_sqlite_runs_bounded_pragma(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.maintenance import maybe_optimize_sqlite

    db_path = tmp_path / "optimize.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE sample (id INTEGER PRIMARY KEY, value TEXT)")
        conn.executemany("INSERT INTO sample(value) VALUES (?)", [("a",), ("b",)])
        observation = maybe_optimize_sqlite(conn, reason="test", analysis_limit=17)

    assert observation.ran is True
    assert observation.reason == "test"
    assert observation.analysis_limit == 17
    assert observation.error is None


def test_maybe_optimize_archive_tiers_covers_existing_split_tiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from polylogue.storage.sqlite.maintenance import maybe_optimize_archive_tiers

    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    for filename in ("source.db", "index.db", "ops.db"):
        (archive_root / filename).write_bytes(b"sqlite placeholder")

    class FakeConnection:
        def __init__(self, path: Path) -> None:
            self.path = path
            self.closed = False

        def execute(self, _sql: str) -> object:
            return object()

        def close(self) -> None:
            self.closed = True

    opened: list[FakeConnection] = []

    def fake_open(path: Path, *, timeout: float) -> FakeConnection:
        assert timeout == 11.0
        conn = FakeConnection(path)
        opened.append(conn)
        return conn

    monkeypatch.setattr("polylogue.storage.sqlite.connection_profile.open_daemon_connection", fake_open)

    observations = maybe_optimize_archive_tiers(archive_root, reason="test", analysis_limit=19, timeout_s=11.0)

    assert [conn.path.name for conn in opened] == ["source.db", "index.db", "ops.db"]
    assert [observation.ran for observation in observations] == [True, True, True]
    assert {observation.analysis_limit for observation in observations} == {19}
    assert all(conn.closed for conn in opened)


def test_checkpoint_archive_wals_covers_existing_split_tiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    for filename in ("source.db", "index.db", "user.db"):
        (archive_root / filename).write_bytes(b"sqlite placeholder")

    calls: list[tuple[Path, str, str]] = []

    def fake_checkpoint(
        db: Path,
        *,
        reason: str,
        escalation: str = "recurring",
        **_: object,
    ) -> WalCheckpointObservation:
        calls.append((db, reason, escalation))
        return WalCheckpointObservation(
            reason=reason,
            mode="passive",
            escalation=escalation,  # type: ignore[arg-type]
            wal_bytes_before=100,
            wal_bytes_after=0,
        )

    monkeypatch.setattr("polylogue.storage.sqlite.wal_checkpoint.checkpoint_wal", fake_checkpoint)

    observations = checkpoint_archive_wals(archive_root, reason="periodic")

    assert [path.name for path, _reason, _escalation in calls] == ["source.db", "index.db", "user.db"]
    assert {reason for _path, reason, _escalation in calls} == {"periodic"}
    assert {escalation for _path, _reason, escalation in calls} == {"recurring"}
    assert [observation.mode for observation in observations] == ["passive", "passive", "passive"]


def test_process_ingest_batch_sync_does_not_force_memory_release_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"
    archive_root = tmp_path / "archive"
    blob_root = tmp_path / "blob"
    source_path = tmp_path / "large-raw.jsonl"
    source_path.write_text("{}", encoding="utf-8")
    raw_record = RawSessionRecord(
        raw_id="raw-large",
        source_name="codex",
        source_path=str(source_path),
        blob_size=2 * 1024 * 1024 * 1024,
        acquired_at="2026-04-02T00:00:00Z",
    )

    origin = origin_from_provider(Provider.from_string(raw_record.source_name)).value
    with open_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO raw_sessions
                (raw_id, origin, native_id, source_path, source_index,
                 blob_hash, blob_size, acquired_at_ms)
            VALUES (?, ?, ?, ?, 0, ?, ?, ?)
            """,
            (
                raw_record.raw_id,
                origin,
                raw_record.raw_id,
                raw_record.source_path,
                b"\x00" * 32,
                raw_record.blob_size,
                1_775_433_600_000,
            ),
        )
        conn.commit()

    def fake_ingest_record(
        record: RawSessionRecord,
        archive_root_str: str,
        validation_mode: str,
        measure_ingest_result_size: bool,
        *,
        blob_root_str: str | None,
    ) -> IngestRecordResult:
        del record, archive_root_str, validation_mode, measure_ingest_result_size, blob_root_str
        return IngestRecordResult(raw_id=raw_record.raw_id, sessions=[])

    def fail_if_sync_releases_memory() -> None:
        raise AssertionError("sync ingest finalization must not run memory release before returning")

    monkeypatch.setattr(ingest_batch_core, "ingest_record", fake_ingest_record)
    monkeypatch.setattr(ingest_batch_core, "release_process_memory", fail_if_sync_releases_memory)

    summary = _process_ingest_batch_sync(
        [raw_record],
        db_path=db_path,
        archive_root_str=str(archive_root),
        blob_root_str=str(blob_root),
        validation_mode="off",
        ingest_workers=1,
        measure_ingest_result_size=False,
    )

    assert summary.raw_record_count == 1
    assert summary.total_blob_mb >= 1024.0


def test_checkpoint_wal_reports_blocking_processes_when_the_route_asks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "index.db"

    class FakeConnection:
        def execute(self, sql: str) -> object:
            assert sql == "PRAGMA wal_checkpoint(PASSIVE)"
            return self

        def fetchone(self) -> tuple[int, int, int]:
            return (1, 25, 12)

        def close(self) -> None:
            return None

    monkeypatch.setattr("polylogue.storage.sqlite.wal_checkpoint._wal_size", lambda db: 1024)
    monkeypatch.setattr(
        "polylogue.storage.sqlite.wal_checkpoint.open_daemon_connection", lambda *_args, **_kwargs: FakeConnection()
    )
    monkeypatch.setattr(
        "polylogue.storage.sqlite.wal_checkpoint._sqlite_file_holders",
        lambda db: ("1234:polylogue-mcp",) if db == db_path else (),
    )

    observation = checkpoint_wal(db_path, reason="test", warn_bytes=0, escalation_bytes=0, collect_blockers=True)

    assert observation.busy_pages == 1
    assert observation.blocking_processes == ("1234:polylogue-mcp",)

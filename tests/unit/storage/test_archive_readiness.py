from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from polylogue.archive.revision_authority import BYTE_AUTHORITY_CENSUS_DETAIL
from polylogue.core.errors import SchemaSkew
from polylogue.storage.archive_readiness import (
    CLAUDE_WORKFLOW_STAGE_NAME,
    RAW_ALIAS_BLOB_MISSING_CATEGORY,
    archive_readiness_status,
    claude_workflow_materialization_status,
    probe_archive_tier,
    raw_materialization_readiness_snapshot,
    raw_materialization_ready,
)
from polylogue.storage.raw_authority import (
    RAW_AUTHORITY_PARSER_FINGERPRINT,
)
from polylogue.storage.sqlite import connection_profile
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


def _category_counts(snapshot: Mapping[str, object]) -> Mapping[str, object]:
    return cast(Mapping[str, object], snapshot["category_counts"])


def _write_blob(root: Path, hex_hash: str, payload: bytes = b"{}") -> None:
    """Materialize the content-addressed blob a raw row points at."""
    blob = root / "blob" / hex_hash[:2] / hex_hash[2:]
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(payload)


def _stamp_index_as_current_schema(index_db: Path) -> None:
    """Declare a hand-built stub index to be at this runtime's schema.

    These fixtures assert raw-gap classification, not schema compatibility,
    and the readiness probe reads through the schema-enforcing open. Stamping
    states the precondition explicitly instead of leaving the probe to accept
    an index whose shape it never checked.
    """
    from polylogue.storage.sqlite.schema_bootstrap import stamp_derived_schema_identity

    with sqlite3.connect(index_db) as conn:
        conn.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]}")
        stamp_derived_schema_identity(conn, "index")


def test_probe_archive_tier_reports_schema_skew_without_opening_a_usable_reader(tmp_path: Path) -> None:
    """The version probe remains readable when the normal reader rejects stale schemas."""
    db_path = tmp_path / "source.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA user_version = {ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE] - 1}")

    with pytest.raises(SchemaSkew):
        connection_profile.open_readonly_connection(db_path)

    probe = probe_archive_tier(ArchiveTier.SOURCE, db_path)

    assert probe.user_version == ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE] - 1
    assert probe.version_status == "mismatch"


def test_raw_materialization_readiness_requires_the_parser_census() -> None:
    """The durable parser census, not a per-pass census row, gates the claim.

    Anti-vacuity: the last assertion is the only True, and it differs from the
    one above it solely by ``raw_authority_parser_census.available`` being the
    boolean the projection writes rather than a truthy string. Re-adding a
    retired census precondition would make it False.
    """
    counters_green: dict[str, object] = {"available": True}

    assert raw_materialization_ready(counters_green) is False
    assert raw_materialization_ready({**counters_green, "raw_authority_parser_census": {"available": "yes"}}) is False
    assert raw_materialization_ready({**counters_green, "raw_authority_parser_census": {"available": True}}) is True


def test_raw_materialization_snapshot_rejects_malformed_parser_receipt(tmp_path: Path) -> None:
    """Readiness shares the promotion gate's parser-receipt validation."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"malformed-census"}}\n',
            source_path="codex/malformed-census.jsonl",
            acquired_at_ms=1,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_authority_parser_census (
                raw_id, parser_fingerprint, status, logical_keys_json, detail, censused_at_ms
            ) VALUES (?, ?, 'complete', '["codex-session:duplicate", "codex-session:duplicate"]', '', 1)
            """,
            (raw_id, RAW_AUTHORITY_PARSER_FINGERPRINT),
        )
        conn.commit()

    snapshot = raw_materialization_readiness_snapshot(tmp_path)
    parser_census = cast(Mapping[str, object], snapshot["raw_authority_parser_census"])

    assert parser_census["complete_count"] == 0
    assert parser_census["incomplete_count"] == 1
    assert parser_census["non_complete_receipt_count"] == 1


def test_raw_materialization_snapshot_rejects_receipt_key_drift_from_durable_binding(tmp_path: Path) -> None:
    """Readiness and frozen promotion reject the same mismatched receipt evidence."""
    from polylogue.archive.revision_authority import RawRevisionEnvelope, RawRevisionKind
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"durable-binding"}}\n',
            source_path="codex/durable-binding.jsonl",
            acquired_at_ms=1,
            revision=RawRevisionEnvelope("codex:durable-binding", RawRevisionKind.FULL, "v1", 0),
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_authority_parser_census (
                raw_id, parser_fingerprint, status, logical_keys_json, detail, censused_at_ms
            ) VALUES (?, ?, 'complete', '["codex-session:wrong-binding"]', '', 1)
            """,
            (raw_id, RAW_AUTHORITY_PARSER_FINGERPRINT),
        )
        conn.commit()

    parser_census = cast(
        Mapping[str, object], raw_materialization_readiness_snapshot(tmp_path)["raw_authority_parser_census"]
    )

    assert parser_census["complete_count"] == 0
    assert parser_census["incomplete_count"] == 1


def test_raw_materialization_snapshot_audits_validation_skipped_raws(tmp_path: Path) -> None:
    """Skipped schema validation does not exempt a raw from parser authority."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"skipped-census"}}\n',
            source_path="codex/skipped-census.jsonl",
            acquired_at_ms=1,
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute("UPDATE raw_sessions SET validation_status = 'skipped' WHERE raw_id = ?", (raw_id,))
        conn.execute("DELETE FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,))
        conn.commit()

    parser_census = cast(
        Mapping[str, object], raw_materialization_readiness_snapshot(tmp_path)["raw_authority_parser_census"]
    )

    assert parser_census["complete_count"] == 0
    assert parser_census["incomplete_count"] == 1
    assert parser_census["missing_receipt_count"] == 1


def test_raw_materialization_snapshot_accepts_parser_confirmed_empty_non_session(tmp_path: Path) -> None:
    """A real parser census may authoritatively establish no sessions."""
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"empty-census"}}\n',
            source_path="codex/empty-census.jsonl",
            acquired_at_ms=1,
            post_parse=True,
        )
        archive.replace_raw_membership_census(
            raw_id,
            [],
            parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
            censused_at_ms=1,
        )

    parser_census = cast(
        Mapping[str, object], raw_materialization_readiness_snapshot(tmp_path)["raw_authority_parser_census"]
    )

    assert parser_census["complete_count"] == 1
    assert parser_census["incomplete_count"] == 0


def test_raw_materialization_snapshot_streams_parser_census_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bounded status projection iterates the real SQLite census cursor."""
    from polylogue.archive.revision_authority import RawRevisionEnvelope, RawRevisionKind
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"stream-census"}}\n',
            source_path="codex/stream-census.jsonl",
            acquired_at_ms=1,
            revision=RawRevisionEnvelope("codex:stream-census", RawRevisionKind.FULL, "v1", 0),
        )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            INSERT INTO raw_authority_parser_census (
                raw_id, parser_fingerprint, status, logical_keys_json, detail, censused_at_ms
            ) VALUES (?, ?, 'complete', '["codex:stream-census"]', '', 1)
            """,
            (raw_id, RAW_AUTHORITY_PARSER_FINGERPRINT),
        )
        conn.commit()

    import polylogue.storage.archive_readiness as readiness_mod

    readiness_sqlite = cast(Any, readiness_mod).sqlite3
    original_connect = readiness_sqlite.connect

    class GuardedCursor:
        def __init__(self, cursor: sqlite3.Cursor, *, census_query: bool) -> None:
            self._cursor = cursor
            self._census_query = census_query

        def fetchall(self) -> list[tuple[object, ...]]:
            if self._census_query:
                raise AssertionError("parser census readiness must not materialize its cursor")
            return self._cursor.fetchall()

        def __iter__(self) -> GuardedCursor:
            return self

        def __next__(self) -> tuple[object, ...]:
            return next(self._cursor)

        def __getattr__(self, name: str) -> object:
            return getattr(self._cursor, name)

    class GuardedConnection:
        _connection: sqlite3.Connection

        def __init__(self, connection: sqlite3.Connection) -> None:
            object.__setattr__(self, "_connection", connection)

        def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor | GuardedCursor:
            cursor = self._connection.execute(sql, cast(Any, parameters))
            if "raw_authority_parser_census AS c" in sql:
                return GuardedCursor(cursor, census_query=True)
            return cursor

        def __getattr__(self, name: str) -> object:
            return getattr(self._connection, name)

        def __setattr__(self, name: str, value: object) -> None:
            setattr(self._connection, name, value)

    def guarded_connect(*args: object, **kwargs: object) -> GuardedConnection:
        return GuardedConnection(original_connect(*args, **kwargs))

    monkeypatch.setattr(readiness_sqlite, "connect", guarded_connect)

    snapshot = raw_materialization_readiness_snapshot(tmp_path, classify_gaps=False)

    assert cast(Mapping[str, object], snapshot["raw_authority_parser_census"])["complete_count"] == 1


def test_exact_archive_readiness_blocks_parser_census_debt(tmp_path: Path) -> None:
    """Exact readiness consumes source parser debt from its real SQLite projection."""
    from polylogue.core.enums import Provider
    from polylogue.sources.revision_backfill import census_historical_revision_evidence
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=(
                b'{"type":"session_meta","payload":{"id":"exact-readiness"}}\n'
                b'{"type":"response_item","payload":{"type":"message","id":"m1","role":"user",'
                b'"content":[{"type":"input_text","text":"exact readiness"}]}}\n'
            ),
            source_path="codex/exact-readiness.jsonl",
            acquired_at_ms=1,
        )

    blocked = archive_readiness_status(tmp_path)
    assert blocked["surfaces"]["raw_artifacts"]["ready"] is False
    assert "parser_census_incomplete" in blocked["surfaces"]["raw_artifacts"]["blockers"]

    census = census_historical_revision_evidence(tmp_path, selected_raw_ids=[raw_id])
    assert census.scanned == 1
    with sqlite3.connect(tmp_path / "source.db") as conn:
        receipt = conn.execute(
            "SELECT status, logical_keys_json, detail FROM raw_authority_parser_census WHERE raw_id = ?", (raw_id,)
        ).fetchone()
    assert receipt is not None
    assert receipt[0] == "complete"
    assert receipt[1] == '["codex-session:exact-readiness"]'
    assert str(receipt[2]).startswith("parser-observed:")

    ready = archive_readiness_status(tmp_path)
    assert ready["surfaces"]["raw_artifacts"]["ready"] is True


def test_raw_materialization_readiness_rejects_unresolved_authority_blockers() -> None:
    readiness = {
        "available": True,
        "raw_authority_parser_census": {"available": True},
        "raw_authority_blocker_count": 1,
    }

    assert raw_materialization_ready(readiness) is False
    assert raw_materialization_ready({**readiness, "raw_authority_blocker_count": 0}) is True


def test_raw_materialization_snapshot_classifies_durable_authority_gaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        conn.executescript(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY, origin TEXT, native_id TEXT, source_path TEXT,
                blob_hash BLOB, source_index INTEGER, revision_authority TEXT,
                validation_status TEXT, parse_error TEXT, parsed_at_ms INTEGER
            );
            CREATE TABLE raw_membership_census (
                raw_id TEXT PRIMARY KEY, status TEXT, member_count INTEGER, detail TEXT
            );
            CREATE TABLE raw_session_memberships (raw_id TEXT, decision TEXT);
            """
        )
        conn.executemany(
            """
            INSERT INTO raw_sessions VALUES (?, 'codex-session', NULL, '', NULL, ?, ?, 'valid', NULL, NULL)
            """,
            [
                ("append-quarantine", -1, "quarantined"),
                ("membership-quarantine", 0, "quarantined"),
                ("terminal-application", 0, "quarantined"),
                ("terminal-application-error", 0, "quarantined"),
                ("authority-pending", -1, "quarantined"),
                ("append-proven", -1, "byte_proven"),
                ("membership-settled", 0, "quarantined"),
                ("membership-incomplete", 0, "quarantined"),
                ("membership-null", 0, "quarantined"),
                ("application-deferred", 0, "quarantined"),
            ],
        )
        conn.execute(
            "UPDATE raw_sessions SET parse_error = 'database locked' WHERE raw_id = 'terminal-application-error'"
        )
        conn.executemany(
            "INSERT INTO raw_membership_census VALUES (?, ?, ?, ?)",
            [
                (
                    "append-quarantine",
                    "failed",
                    0,
                    BYTE_AUTHORITY_CENSUS_DETAIL,
                ),
                ("membership-quarantine", "complete", 1, None),
                ("membership-settled", "complete", 2, None),
                ("membership-incomplete", "complete", 2, None),
                ("membership-null", "complete", 1, None),
            ],
        )
        conn.executemany(
            "INSERT INTO raw_session_memberships VALUES (?, ?)",
            [
                ("membership-quarantine", "ambiguous"),
                ("membership-settled", "applied"),
                ("membership-settled", "superseded_equivalent"),
                ("membership-incomplete", "applied"),
                ("membership-null", None),
            ],
        )
    with sqlite3.connect(index_db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (session_id TEXT PRIMARY KEY, raw_id TEXT);
            CREATE TABLE raw_revision_applications (raw_id TEXT, decision TEXT, detail TEXT);
            INSERT INTO raw_revision_applications VALUES ('terminal-application', 'superseded', 'test');
            INSERT INTO raw_revision_applications VALUES ('terminal-application-error', 'superseded', 'test');
            INSERT INTO raw_revision_applications VALUES (
                'application-deferred', 'deferred', 'ordinary_replay:incomparable_existing_index_state'
            );
            """
        )

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot.get("available") is True, snapshot
    assert snapshot["classified"] == 5
    assert snapshot["critical"] == 1
    assert snapshot["actionable"] == 1
    assert snapshot["affected_actionable"] == 1
    assert snapshot["blocked"] == 1
    assert snapshot["unchecked"] == 3
    assert snapshot["affected_unchecked"] == 3
    assert _category_counts(snapshot) == {
        "raw_id_join_gap": 3,
        "skipped": 0,
        "parse_failed": 1,
        "raw_parse_failed": 1,
        "parsed_without_index_session": 0,
        "append-authority-quarantined": 1,
        "append-authority-proven": 1,
        "membership-authority-classified": 2,
        "revision-application-terminal": 1,
        "adoption_deferred": 1,
    }

    def fail_if_classified(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("bounded readiness must not classify individual raw gaps")

    monkeypatch.setattr("polylogue.storage.archive_readiness._classify_raw_gap_rows", fail_if_classified)
    aggregate = raw_materialization_readiness_snapshot(tmp_path, classify_gaps=False)

    assert aggregate["classification"] == "not_run"
    assert aggregate["classified"] == 0
    assert aggregate["unchecked"] == 9
    assert aggregate["actionable"] == 0


def test_raw_materialization_snapshot_reads_append_census_writer_contract(tmp_path: Path) -> None:
    from polylogue.core.enums import Provider
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root

    initialize_active_archive_root(tmp_path)
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"append":true}\n',
            source_path="session.jsonl",
            source_index=-1,
            acquired_at_ms=1,
        )
        archive.replace_raw_membership_census(
            raw_id,
            None,
            parser_fingerprint=RAW_AUTHORITY_PARSER_FINGERPRINT,
            censused_at_ms=0,
            detail=BYTE_AUTHORITY_CENSUS_DETAIL,
        )

    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["classified"] == 1
    assert snapshot["unchecked"] == 0
    assert _category_counts(snapshot)["append-authority-quarantined"] == 1
    parser_census = cast(Mapping[str, object], snapshot["raw_authority_parser_census"])
    assert parser_census["complete_count"] == 1
    assert parser_census["incomplete_count"] == 0


def test_raw_materialization_snapshot_ignores_skipped_raw_rows(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                validation_status TEXT,
                parse_error TEXT,
                parsed_at_ms INTEGER
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO raw_sessions(raw_id, origin, validation_status, parse_error, parsed_at_ms)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("raw-materializable", "chatgpt-export", "valid", None, 123),
                ("raw-skipped", "aistudio-drive", "skipped", None, None),
            ],
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, raw_id TEXT)")

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["available"] is True
    assert snapshot["raw_artifact_count"] == 1
    assert snapshot["materialized_raw_artifact_count"] == 0
    assert snapshot["archive_session_count"] == 0
    assert snapshot["join_gap_count"] == 1
    assert snapshot["total"] == 1
    assert snapshot["unchecked"] == 1
    assert snapshot["affected_unchecked"] == 1
    assert snapshot["category_counts"] == {
        "raw_id_join_gap": 1,
        "skipped": 0,
        "parse_failed": 0,
        "raw_parse_failed": 0,
        "parsed_without_index_session": 1,
    }
    assert snapshot["source_family_counts"] == {"chatgpt-export": 1}


def test_raw_materialization_snapshot_counts_raw_artifacts_once(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                validation_status TEXT,
                parse_error TEXT,
                parsed_at_ms INTEGER
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO raw_sessions(raw_id, origin, validation_status, parse_error, parsed_at_ms)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("raw-shared", "claude-code-session", "valid", None, 123),
                ("raw-gap", "codex-session", "valid", None, 124),
            ],
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, raw_id TEXT)")
        conn.executemany(
            "INSERT INTO sessions(session_id, raw_id) VALUES (?, ?)",
            [
                ("session-one", "raw-shared"),
                ("session-two", "raw-shared"),
            ],
        )

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["raw_artifact_count"] == 2
    assert snapshot["materialized_raw_artifact_count"] == 1
    assert snapshot["archive_session_count"] == 2
    assert snapshot["join_gap_count"] == 1
    assert snapshot["total"] == 1
    assert snapshot["source_family_counts"] == {"codex-session": 1}


def test_raw_materialization_snapshot_marks_parse_failures_actionable(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                validation_status TEXT,
                parse_error TEXT,
                parsed_at_ms INTEGER
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO raw_sessions(raw_id, origin, validation_status, parse_error, parsed_at_ms)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                ("raw-failed-one", "codex-session", "failed", "bad json", None),
                ("raw-failed-two", "aistudio-drive", "failed", "bad json", None),
            ],
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, raw_id TEXT)")

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["total"] == 2
    assert snapshot["critical"] == 2
    assert snapshot["actionable"] == 2
    assert snapshot["affected_actionable"] == 2
    assert snapshot["unchecked"] == 0
    assert snapshot["affected_unchecked"] == 0
    assert snapshot["category_counts"] == {
        "raw_id_join_gap": 0,
        "skipped": 0,
        "parse_failed": 2,
        "raw_parse_failed": 2,
        "parsed_without_index_session": 0,
    }


def test_raw_materialization_snapshot_classifies_native_aliases(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                native_id TEXT,
                source_path TEXT,
                blob_hash BLOB,
                validation_status TEXT,
                parse_error TEXT,
                parsed_at_ms INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, validation_status, parse_error, parsed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("raw-alias", "chatgpt-export", "conv-1", "capture.json", bytes.fromhex("11" * 32), "passed", None, 123),
        )
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, validation_status, parse_error, parsed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("older-raw", "chatgpt-export", "conv-1", "older.json", bytes.fromhex("12" * 32), "passed", None, 122),
        )
    _write_blob(tmp_path, "11" * 32)
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, native_id TEXT, raw_id TEXT)")
        conn.execute(
            "INSERT INTO sessions(session_id, origin, native_id, raw_id) VALUES (?, ?, ?, ?)",
            ("chatgpt-export:conv-1", "chatgpt-export", "conv-1", "older-raw"),
        )

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["classification"] == "cheap_projection"
    assert snapshot["total"] == 1
    assert snapshot["classified"] == 1
    assert snapshot["affected_classified"] == 1
    assert snapshot["unchecked"] == 0
    assert snapshot["affected_unchecked"] == 0
    counts = _category_counts(snapshot)
    assert counts["materialized-alias"] == 1
    assert counts["raw_id_join_gap"] == 0


def test_raw_materialization_snapshot_reports_an_alias_whose_own_blob_is_gone(tmp_path: Path) -> None:
    """An alias reconciles identity, not bytes: a missing blob stays owed work.

    Anti-vacuity: drop the blob-existence check ahead of the alias test and the
    row is classified ``materialized-alias``, so ``affected_actionable`` falls
    back to 0 and a genuinely lost newer snapshot reads as a clean archive.
    """
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                native_id TEXT,
                source_path TEXT,
                blob_hash BLOB,
                validation_status TEXT,
                parse_error TEXT,
                parsed_at_ms INTEGER
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, validation_status, parse_error, parsed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                # The newer snapshot: same provider session, its own blob gone.
                ("raw-newer", "chatgpt-export", "conv-1", "newer.json", bytes.fromhex("11" * 32), "passed", None, 123),
                ("older-raw", "chatgpt-export", "conv-1", "older.json", bytes.fromhex("12" * 32), "passed", None, 122),
            ],
        )
    _write_blob(tmp_path, "12" * 32)
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, native_id TEXT, raw_id TEXT)")
        conn.execute(
            "INSERT INTO sessions(session_id, origin, native_id, raw_id) VALUES (?, ?, ?, ?)",
            ("chatgpt-export:conv-1", "chatgpt-export", "conv-1", "older-raw"),
        )

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    counts = _category_counts(snapshot)
    assert counts[RAW_ALIAS_BLOB_MISSING_CATEGORY] == 1
    assert counts.get("materialized-alias", 0) == 0
    assert snapshot["classified"] == 0
    assert snapshot["affected_actionable"] == 1
    assert snapshot["unchecked"] == 0


def test_raw_materialization_snapshot_classifies_stale_decode_aliases(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                native_id TEXT,
                source_path TEXT,
                blob_hash BLOB,
                validation_status TEXT,
                parse_error TEXT,
                parsed_at_ms INTEGER
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, validation_status, parse_error, parsed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "raw-stale-error",
                    "codex-session",
                    "session-1",
                    "session-1.jsonl",
                    bytes.fromhex("11" * 32),
                    "failed",
                    "decode: [Errno 2] No such file or directory: '/tmp/archive/blob/11/1111'",
                    None,
                ),
                (
                    "raw-current",
                    "codex-session",
                    "session-1",
                    "current.jsonl",
                    bytes.fromhex("12" * 32),
                    "passed",
                    None,
                    123,
                ),
            ],
        )
    # The decode error is stale precisely because the blob is present again.
    _write_blob(tmp_path, "11" * 32)
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, native_id TEXT, raw_id TEXT)")
        conn.execute(
            "INSERT INTO sessions(session_id, origin, native_id, raw_id) VALUES (?, ?, ?, ?)",
            ("codex-session:session-1", "codex-session", "session-1", "raw-current"),
        )

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["actionable"] == 0
    assert snapshot["affected_actionable"] == 0
    assert snapshot["classified"] == 1
    counts = _category_counts(snapshot)
    assert counts["materialized-alias"] == 1
    assert counts["parse_failed"] == 0
    assert counts["raw_parse_failed"] == 1


def test_raw_materialization_snapshot_classifies_dangling_index_raw_link_as_lost_source_evidence(
    tmp_path: Path,
) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                native_id TEXT,
                source_path TEXT,
                blob_hash BLOB,
                validation_status TEXT,
                parse_error TEXT,
                parsed_at_ms INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, validation_status, parse_error, parsed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("raw-new", "chatgpt-export", "conv-1", "capture.json", bytes.fromhex("11" * 32), "passed", None, 123),
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, native_id TEXT, raw_id TEXT)")
        conn.execute(
            "INSERT INTO sessions(session_id, origin, native_id, raw_id) VALUES (?, ?, ?, ?)",
            ("chatgpt-export:conv-1", "chatgpt-export", "conv-1", "older-missing-raw"),
        )

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["lost_source_evidence_count"] == 1
    assert snapshot["classified"] == 1
    assert snapshot["unchecked"] == 0
    assert raw_materialization_ready(snapshot) is False
    counts = _category_counts(snapshot)
    assert counts.get("materialized-alias", 0) == 0
    assert counts["lost-source-evidence-alias"] == 1
    assert counts["raw_id_join_gap"] == 0


def test_lost_source_evidence_samples_include_generated_session_identity(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute(
            """
            INSERT INTO sessions (native_id, origin, raw_id, title, content_hash)
            VALUES ('missing', 'codex-session', 'raw-missing', 'missing raw', ?)
            """,
            (bytes(32),),
        )
        conn.commit()

    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["lost_source_evidence_count"] == 1
    samples = cast(list[dict[str, object]], snapshot["lost_source_evidence_samples"])
    assert samples[0]["session_id"] == "codex-session:missing"
    assert samples[0]["missing_raw_id"] == "raw-missing"


def test_raw_materialization_snapshot_marks_reverse_authority_query_failure_unavailable(
    tmp_path: Path,
) -> None:
    """A failed lost-source count cannot become a healthy zero."""

    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                validation_status TEXT,
                parse_error TEXT,
                parsed_at_ms INTEGER
            )
            """
        )
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.executescript(
            """
            CREATE TABLE session_rows (raw_value INTEGER NOT NULL);
            INSERT INTO session_rows VALUES (-9223372036854775808);
            CREATE VIEW sessions AS
            SELECT abs(raw_value) AS raw_id
            FROM session_rows;
            """
        )

    _stamp_index_as_current_schema(tmp_path / "index.db")
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["available"] is False
    assert "integer overflow" in str(snapshot["error"])
    assert raw_materialization_ready(snapshot) is False


def test_raw_materialization_snapshot_classifies_source_path_aliases(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    source_path = tmp_path / "cache" / "native-alias_1.jsonl.txt.json"
    source_path.parent.mkdir()
    source_path.write_text("{}", encoding="utf-8")
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                native_id TEXT,
                source_path TEXT,
                blob_hash BLOB,
                validation_status TEXT,
                parse_error TEXT,
                parsed_at_ms INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, validation_status, parse_error, parsed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "raw-source-alias",
                "claude-code-session",
                None,
                str(source_path),
                bytes.fromhex("12" * 32),
                "passed",
                None,
                123,
            ),
        )
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, validation_status, parse_error, parsed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "older-raw",
                "claude-code-session",
                "native-alias",
                "older.json",
                bytes.fromhex("13" * 32),
                "passed",
                None,
                122,
            ),
        )
    _write_blob(tmp_path, "12" * 32)
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, native_id TEXT, raw_id TEXT)")
        conn.execute(
            "INSERT INTO sessions(session_id, origin, native_id, raw_id) VALUES (?, ?, ?, ?)",
            ("claude-code-session:native-alias", "claude-code-session", "native-alias", "older-raw"),
        )

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["classified"] == 1
    assert snapshot["unchecked"] == 0
    assert _category_counts(snapshot)["materialized-alias"] == 1


def test_raw_materialization_snapshot_classifies_parsed_non_session_artifacts(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    blob = tmp_path / "blob" / "dd" / ("dd" * 31)
    blob.parent.mkdir(parents=True)
    blob.write_text('{"type":"file-history-snapshot","messageId":"m1"}\n', encoding="utf-8")
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                native_id TEXT,
                source_path TEXT,
                blob_hash BLOB,
                validation_status TEXT,
                parse_error TEXT,
                parsed_at_ms INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, validation_status, parse_error, parsed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "raw-sidecar",
                "claude-code-session",
                "sidecar-native",
                str(tmp_path / "sidecar.jsonl"),
                bytes.fromhex("dd" * 32),
                "passed",
                None,
                123,
            ),
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, native_id TEXT, raw_id TEXT)")

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["classification"] == "cheap_projection"
    assert snapshot["classified"] == 1
    assert snapshot["unchecked"] == 0
    counts = _category_counts(snapshot)
    assert counts["parsed-non-session-artifact"] == 1
    assert counts["raw_id_join_gap"] == 0


def test_raw_materialization_snapshot_keeps_unexplained_gaps_unchecked(tmp_path: Path) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    blob = tmp_path / "blob" / "dd" / ("dd" * 31)
    blob.parent.mkdir(parents=True)
    blob.write_text('{"type":"file-history-snapshot","messageId":"m1"}\n', encoding="utf-8")
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                native_id TEXT,
                source_path TEXT,
                blob_hash BLOB,
                validation_status TEXT,
                parse_error TEXT,
                parsed_at_ms INTEGER
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, validation_status, parse_error, parsed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "raw-sidecar",
                    "claude-code-session",
                    "sidecar-native",
                    str(tmp_path / "sidecar.jsonl"),
                    bytes.fromhex("dd" * 32),
                    "passed",
                    None,
                    123,
                ),
                (
                    "raw-session-shaped",
                    "codex-session",
                    "codex-native",
                    str(tmp_path / "session.jsonl"),
                    bytes.fromhex("ee" * 32),
                    "passed",
                    None,
                    123,
                ),
                (
                    "raw-skipped",
                    "chatgpt-export",
                    "skipped-native",
                    str(tmp_path / "skipped.json"),
                    bytes.fromhex("ff" * 32),
                    "skipped",
                    None,
                    123,
                ),
            ],
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, native_id TEXT, raw_id TEXT)")

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["raw_artifact_count"] == 2
    assert snapshot["total"] == 2
    assert snapshot["classified"] == 1
    assert snapshot["unchecked"] == 1
    assert snapshot["affected_unchecked"] == 1
    counts = _category_counts(snapshot)
    assert counts["parsed-non-session-artifact"] == 1
    assert counts["raw_id_join_gap"] == 1


def test_raw_materialization_snapshot_classifies_same_native_lost_source_evidence(
    tmp_path: Path,
) -> None:
    source_db = tmp_path / "source.db"
    index_db = tmp_path / "index.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT,
                native_id TEXT,
                source_path TEXT,
                blob_hash BLOB,
                validation_status TEXT,
                parse_error TEXT,
                parsed_at_ms INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO raw_sessions(
                raw_id, origin, native_id, source_path, blob_hash, validation_status, parse_error, parsed_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "newer-raw",
                "claude-code-session",
                "session-native",
                str(tmp_path / "session-native.jsonl"),
                bytes.fromhex("aa" * 32),
                "passed",
                None,
                123,
            ),
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, native_id TEXT, raw_id TEXT)")
        conn.execute(
            """
            INSERT INTO sessions(session_id, origin, native_id, raw_id)
            VALUES (?, ?, ?, ?)
            """,
            (
                "claude-code-session:session-native",
                "claude-code-session",
                "session-native",
                "missing-older-raw",
            ),
        )

    _stamp_index_as_current_schema(index_db)
    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["lost_source_evidence_count"] == 1
    assert snapshot["classified"] == 1
    assert snapshot["unchecked"] == 0
    assert snapshot["affected_unchecked"] == 0
    assert raw_materialization_ready(snapshot) is False
    counts = _category_counts(snapshot)
    assert counts["lost-source-evidence-alias"] == 1
    assert counts["raw_id_join_gap"] == 0


def test_raw_materialization_ready_rejects_failed_debt_classifier() -> None:
    """A readiness dict carrying debt_classifier_error must not read as ready.

    paths._merge_raw_materialization_debt records classifier failures under
    this key; the composed readiness contract requires the classifier, so a
    recorded failure blocks the ready claim even when every structural count
    is clean (removing the predicate's debt_classifier_error check fails this).
    """
    clean = {
        "available": True,
        "raw_authority_parser_census": {"available": True},
        "critical": 0,
        "warning": 0,
        "actionable": 0,
        "blocked": 0,
        "affected_actionable": 0,
        "affected_blocked": 0,
        "affected_open": 0,
        "lost_source_evidence_count": 0,
        "unchecked": 0,
        "affected_unchecked": 0,
    }
    assert raw_materialization_ready(clean) is True
    assert raw_materialization_ready({**clean, "debt_classifier_error": "RuntimeError: ops.db locked"}) is False


def test_claude_workflow_materialization_status_missing_ops_db_returns_none(tmp_path: Path) -> None:
    assert claude_workflow_materialization_status(tmp_path / "ops.db") is None


def test_claude_workflow_materialization_status_reads_latest_stage_event(tmp_path: Path) -> None:
    """Reads back exactly what daemon/convergence_stages.py's claude_workflow
    stage persists via record_daemon_stage_event -- the wiring this bead adds
    so a materialization gap count survives past one log line (bd polylogue-uh9l).
    """
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
    from polylogue.storage.sqlite.archive_tiers.ops_write import record_daemon_stage_event
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    ops_db = tmp_path / "ops.db"
    conn = sqlite3.connect(ops_db)
    try:
        initialize_archive_tier(conn, ArchiveTier.OPS)
        record_daemon_stage_event(
            conn,
            stage=CLAUDE_WORKFLOW_STAGE_NAME,
            status="gaps",
            observed_at_ms=1_700_000_000_000,
            payload={"gap_count": 2, "gaps": ["missing agent metadata for transcript agent-a", "unresolved call x"]},
        )
        conn.commit()
    finally:
        conn.close()

    status = claude_workflow_materialization_status(ops_db)
    assert status is not None
    assert status["status"] == "gaps"
    assert status["gap_count"] == 2
    assert status["gaps"] == ["missing agent metadata for transcript agent-a", "unresolved call x"]
    assert status["observed_at_ms"] == 1_700_000_000_000


def _pinned_index_over(tmp_path: Path) -> sqlite3.Connection:
    """An index reader with the source tier attached, as an operation pins it."""
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database

    initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
    initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
    conn = sqlite3.connect(tmp_path / "index.db")
    conn.execute("ATTACH DATABASE ? AS source_tier", (str(tmp_path / "source.db"),))
    return conn


def test_pinned_materialization_readiness_degrades_like_its_path_twin(tmp_path: Path) -> None:
    """A degraded pinned tier must report unavailable, not raise.

    ``raw_materialization_readiness_snapshot`` wraps its reads and returns
    ``{"available": False, "error": ...}``. Its pinned twin, added alongside it,
    had no handler, so a busy reader or a column the snapshot predates raised
    out of ``_raw_materialization_status`` -- which runs unconditionally on
    every status poll.

    Anti-vacuity: remove the ``except (OSError, sqlite3.Error)`` clause from
    ``raw_materialization_readiness_from_pinned_index`` and this goes red with
    ``sqlite3.OperationalError: no such column: s.raw_id``.
    """
    from polylogue.storage.archive_readiness import raw_materialization_readiness_from_pinned_index

    conn = _pinned_index_over(tmp_path)
    try:
        # Drop a column the projection selects: readable tier, unreadable query.
        conn.execute("DROP INDEX IF EXISTS idx_sessions_raw_id")
        conn.execute("ALTER TABLE sessions DROP COLUMN raw_id")
        conn.commit()
        result = raw_materialization_readiness_from_pinned_index(conn, archive_root=tmp_path)
    finally:
        conn.close()

    assert result["available"] is False
    assert "raw_id" in str(result["error"])


def test_pinned_readiness_status_degrades_like_its_path_twin(tmp_path: Path) -> None:
    """The same missing failure contract on the readiness-status twin.

    ``archive_readiness_status`` returns ``{"checked": False, "reason": ...}``
    on ``sqlite3.Error``; the pinned variant raised instead.

    Anti-vacuity: remove the ``except (OSError, sqlite3.Error)`` clause from
    ``archive_readiness_status_from_connections`` and this goes red.
    """
    from polylogue.storage.archive_readiness import archive_readiness_status_from_connections

    conn = _pinned_index_over(tmp_path)
    try:
        conn.execute("DROP VIEW IF EXISTS threads")
        conn.execute("ALTER TABLE sessions DROP COLUMN message_count")
        conn.commit()
        result = archive_readiness_status_from_connections(conn, None, raw_materialization_readiness=None)
    finally:
        conn.close()

    assert result["checked"] is False
    assert "message_count" in str(result["reason"])


def test_pinned_materialization_readiness_defaults_to_the_bounded_contract(tmp_path: Path) -> None:
    """Exact gap classification is a diagnostic read, not a status poll.

    The path twin's caller passes ``classify_gaps=False`` because exact
    classification opens a blob and reads JSONL per unmaterialized raw. The
    pinned twin defaulted it to True and its only caller omits the kwarg, so
    every status poll walked the whole unmaterialized corpus -- which, during a
    from-scratch rebuild, is nearly every raw.

    Anti-vacuity: flip the default back to True and this goes red.
    """
    import inspect

    from polylogue.storage.archive_readiness import raw_materialization_readiness_from_pinned_index

    signature = inspect.signature(raw_materialization_readiness_from_pinned_index)
    assert signature.parameters["classify_gaps"].default is False


def test_converged_archive_reports_its_real_materialization_counts(tmp_path: Path) -> None:
    """A fully materialized archive is not an empty one.

    The counters query drove its join from the *gap* rows. A converged archive
    has none, and a join with an empty side leaves every non-aggregated total
    NULL, which the ``int(row[...] or 0)`` coercions turned into zero -- so an
    archive holding one raw artifact and one session reported
    ``raw_artifact_count == 0`` and ``archive_session_count == 0``, the
    unmeasured-state-as-a-positive-result shape.

    Anti-vacuity: restore ``FROM gaps CROSS JOIN materialization CROSS JOIN
    session_count`` and every assertion below goes red with ``0``.
    """
    from polylogue.archive.message.roles import Role
    from polylogue.core.enums import BlockType, Provider
    from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
    from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
    from tests.infra.live_ingest import write_index_session

    initialize_active_archive_root(tmp_path)
    session = ParsedSession(
        source_name=Provider.CODEX,
        provider_session_id="converged-counts",
        updated_at="2026-01-02T00:00:00Z",
        messages=[
            ParsedMessage(
                provider_message_id="m1",
                role=Role.USER,
                blocks=[ParsedContentBlock(type=BlockType.TEXT, text="converged prose")],
            )
        ],
    )
    with ArchiveStore.open_existing(tmp_path, read_only=False) as archive:
        raw_id = archive.write_raw_payload(
            provider=Provider.CODEX,
            payload=b'{"type":"session_meta","payload":{"id":"converged-counts"}}\n',
            source_path="codex/converged-counts.jsonl",
            acquired_at_ms=1,
            post_parse=True,
        )
        session_id = write_index_session(archive, session)

    # Bind the index session to its raw artifact: the index tier is
    # rebuildable, and this is the join the counters read.
    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("UPDATE sessions SET raw_id = ? WHERE session_id = ?", (raw_id, session_id))
        conn.commit()

    snapshot = raw_materialization_readiness_snapshot(tmp_path, classify_gaps=False)

    assert snapshot["available"] is True
    assert snapshot["raw_artifact_count"] == 1
    assert snapshot["materialized_raw_artifact_count"] == 1
    assert snapshot["archive_session_count"] == 1
    assert snapshot["join_gap_count"] == 0
    assert snapshot["total"] == 0


def test_archive_sessions_surface_is_computed_not_asserted(tmp_path: Path) -> None:
    """polylogue-bu47u: the sessions surface published ``ready=True`` unconditionally.

    Every sibling surface derives ``ready`` from its blockers; this one was a
    literal, so an index tier whose ``messages`` relation is gone still
    certified the surface while reporting a fabricated ``message_count`` of 0.

    Anti-vacuity: restore ``surface(ready=True, blockers=[])`` for
    ``archive_sessions`` (or drop the relation-presence counts that feed it)
    and this fails; the healthy assertion below fails if the blockers are
    raised unconditionally instead.
    """

    initialize_active_archive_root(tmp_path)

    healthy = archive_readiness_status(tmp_path)
    assert healthy["surfaces"]["archive_sessions"]["ready"] is True
    assert healthy["surfaces"]["archive_sessions"]["blockers"] == []

    with sqlite3.connect(tmp_path / "index.db") as conn:
        conn.execute("DROP TABLE IF EXISTS messages_fts")
        conn.execute("DROP TABLE messages")

    blocked = archive_readiness_status(tmp_path)
    sessions = blocked["surfaces"]["archive_sessions"]
    assert sessions["ready"] is False
    assert sessions["blockers"] == ["messages_relation_missing"]
    assert sessions["evidence"]["messages_table_present"] is False

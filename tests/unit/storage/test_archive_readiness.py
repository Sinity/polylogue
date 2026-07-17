from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from polylogue.archive.revision_authority import BYTE_AUTHORITY_CENSUS_DETAIL
from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot, raw_materialization_ready
from polylogue.storage.raw_authority import (
    RawReplayPlan,
    RawReplayPlanOutcome,
    RawReplayPlanStatus,
    finalize_raw_authority_census,
    record_raw_authority_census,
    record_raw_replay_outcome,
)
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_active_archive_root


def _category_counts(snapshot: Mapping[str, object]) -> Mapping[str, object]:
    return cast(Mapping[str, object], snapshot["category_counts"])


def test_raw_materialization_readiness_requires_completed_frontier_census() -> None:
    counters_green: dict[str, object] = {"available": True}

    assert raw_materialization_ready(counters_green) is False
    assert (
        raw_materialization_ready({**counters_green, "raw_authority_frontier": {"lifecycle_status": "interrupted"}})
        is False
    )
    assert (
        raw_materialization_ready({**counters_green, "raw_authority_frontier": {"lifecycle_status": "completed"}})
        is True
    )


def test_readiness_uses_frontier_postflight_not_preapply_scope(tmp_path: Path) -> None:
    """An applied repair must not remain blocked by its immutable preflight."""
    initialize_active_archive_root(tmp_path)
    plan = RawReplayPlan(
        plan_id="raw-authority-frontier:" + "a" * 64,
        input_digest="b" * 64,
        input_raw_ids=("raw-1",),
        logical_keys=("chatgpt-export:conversation-1",),
        authority_witness={"schema": "polylogue.raw-authority-frontier-plan.v1"},
        source_preconditions={},
        index_preconditions={},
    )
    preview = record_raw_authority_census(
        tmp_path,
        (plan,),
        selected_plan_ids=set(),
        executable_plan_ids={plan.plan_id},
        mode="dry_run",
        quiescent=True,
        scope={
            "schema": "polylogue.raw-authority-frontier-scope.v1",
            "state_counts": {"missing_source_bytes": 1},
        },
        residual={
            "schema": "polylogue.raw-authority-frontier-residual.v1",
            "state_counts": {"missing_source_bytes": 1},
            "frontier_state_counts": {"missing_source_bytes": 1, "proven_current": 2},
        },
    )
    dry_run_snapshot = raw_materialization_readiness_snapshot(tmp_path)
    dry_run_frontier = cast(Mapping[str, object], dry_run_snapshot["raw_authority_frontier"])
    assert dry_run_frontier["census_id"] == preview.census_id
    assert dry_run_frontier["state_counts"] == {"missing_source_bytes": 1, "proven_current": 2}
    assert dry_run_frontier["blocking_count"] == 1

    receipt = record_raw_authority_census(
        tmp_path,
        (plan,),
        selected_plan_ids={plan.plan_id},
        executable_plan_ids={plan.plan_id},
        mode="apply",
        quiescent=True,
        scope={
            "schema": "polylogue.raw-authority-frontier-scope.v1",
            "state_counts": {"missing_source_bytes": 1},
        },
        residual={
            "schema": "polylogue.raw-authority-frontier-residual.v1",
            "state_counts": {"missing_source_bytes": 1},
            "frontier_state_counts": {"missing_source_bytes": 1, "proven_current": 2},
        },
    )
    record_raw_replay_outcome(
        tmp_path,
        receipt.census_id,
        RawReplayPlanOutcome(
            plan_id=plan.plan_id,
            input_raw_ids=plan.input_raw_ids,
            status=RawReplayPlanStatus.EXECUTED,
            reason="fixture repaired the exact plan",
            next_action="none",
        ),
    )
    finalize_raw_authority_census(
        tmp_path,
        receipt.census_id,
        post_plans=(),
        post_residual={
            "schema": "polylogue.raw-authority-frontier-residual.v1",
            "state_counts": {},
            "frontier_state_counts": {"proven_current": 3},
        },
    )

    snapshot = raw_materialization_readiness_snapshot(tmp_path)
    frontier = cast(Mapping[str, object], snapshot["raw_authority_frontier"])

    assert frontier["state_counts"] == {"proven_current": 3}
    assert snapshot["raw_authority_frontier_blocking_count"] == 0
    assert raw_materialization_ready(snapshot) is True


def test_raw_materialization_snapshot_classifies_durable_authority_gaps(tmp_path: Path) -> None:
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
            parser_fingerprint="revision-membership-v1",
            censused_at_ms=0,
            detail=BYTE_AUTHORITY_CENSUS_DETAIL,
        )

    snapshot = raw_materialization_readiness_snapshot(tmp_path)

    assert snapshot["classified"] == 1
    assert snapshot["unchecked"] == 0
    assert _category_counts(snapshot)["append-authority-quarantined"] == 1


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
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, native_id TEXT, raw_id TEXT)")
        conn.execute(
            "INSERT INTO sessions(session_id, origin, native_id, raw_id) VALUES (?, ?, ?, ?)",
            ("chatgpt-export:conv-1", "chatgpt-export", "conv-1", "older-raw"),
        )

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
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, native_id TEXT, raw_id TEXT)")
        conn.execute(
            "INSERT INTO sessions(session_id, origin, native_id, raw_id) VALUES (?, ?, ?, ?)",
            ("codex-session:session-1", "codex-session", "session-1", "raw-current"),
        )

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
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT, native_id TEXT, raw_id TEXT)")
        conn.execute(
            "INSERT INTO sessions(session_id, origin, native_id, raw_id) VALUES (?, ?, ?, ?)",
            ("claude-code-session:native-alias", "claude-code-session", "native-alias", "older-raw"),
        )

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
        "raw_authority_frontier": {"lifecycle_status": "completed"},
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

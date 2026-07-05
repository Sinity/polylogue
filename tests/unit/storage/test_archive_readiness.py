from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from polylogue.storage.archive_readiness import raw_materialization_readiness_snapshot, raw_materialization_ready


def _category_counts(snapshot: Mapping[str, object]) -> Mapping[str, object]:
    return cast(Mapping[str, object], snapshot["category_counts"])


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

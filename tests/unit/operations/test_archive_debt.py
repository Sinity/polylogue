"""Tests for the unified archive debt projection."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from polylogue.operations import archive_debt as module
from polylogue.operations.archive_debt import archive_debt_list
from polylogue.storage.sqlite.archive_tiers.bootstrap import ARCHIVE_TIER_SPECS, initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion


def _write_tier_version(path: Path, version: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()
    finally:
        conn.close()


def _write_current_tier_files(root: Path) -> None:
    for spec in ARCHIVE_TIER_SPECS.values():
        _write_tier_version(root / spec.filename, spec.version)


def test_archive_debt_reports_missing_required_tiers(tmp_path: Path) -> None:
    payload = archive_debt_list(archive_root=tmp_path, kinds=("archive-tier",))

    refs = {row.debt_ref for row in payload.rows}
    assert "debt:archive-tier:source:missing" in refs
    assert "debt:archive-tier:user:missing" in refs
    assert payload.totals.critical >= 2
    assert all(row.kind == "archive-tier" for row in payload.rows)


def test_archive_debt_reports_candidate_assertions_as_actionable(tmp_path: Path) -> None:
    _write_current_tier_files(tmp_path)
    user_db = tmp_path / "user.db"
    conn = sqlite3.connect(user_db)
    try:
        initialize_archive_tier(conn, ArchiveTier.USER)
        upsert_assertion(
            conn,
            assertion_id="cand-1",
            target_ref="session:sess-1",
            kind=AssertionKind.TRANSFORM_CANDIDATE,
            value={"candidate_kind": "summary", "source": "transform"},
            body_text="Candidate summary",
            evidence_refs=("message:msg-1",),
            status="candidate",
            context_policy={"inject": False, "promotion_required": True},
            now_ms=1_765_584_000_000,
        )
        conn.commit()
    finally:
        conn.close()

    payload = archive_debt_list(archive_root=tmp_path, kinds=("assertion-candidate",))

    assert payload.totals.total == 1
    row = payload.rows[0]
    assert row.debt_ref == "debt:assertion-candidate:cand-1"
    assert row.kind == "assertion-candidate"
    assert row.stage == "candidate-judgment"
    assert row.subject_ref == "assertion:cand-1"
    assert row.severity == "info"
    assert row.status == "actionable"
    assert row.owner == "user"
    assert row.details == "Candidate summary"
    assert row.evidence_refs == ("message:msg-1", "assertion:cand-1")
    assert row.actions[0].command == ("polylogue", "mark", "candidates", "list", "--format", "json")


def test_archive_debt_reports_convergence_failures(tmp_path: Path) -> None:
    _write_current_tier_files(tmp_path)
    ops_db = tmp_path / "ops.db"
    conn = sqlite3.connect(ops_db)
    try:
        conn.execute(
            """
            CREATE TABLE convergence_debt (
                debt_id INTEGER PRIMARY KEY,
                stage TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                updated_at_ms INTEGER NOT NULL,
                last_error TEXT,
                next_retry_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO convergence_debt (
                stage, target_type, target_id, status, attempts, priority, updated_at_ms, last_error, next_retry_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "fts",
                "session",
                "sess-1",
                "failed",
                2,
                10,
                int(datetime(2026, 6, 19, tzinfo=UTC).timestamp() * 1000),
                "boom",
                None,
            ),
        )
        conn.execute(
            """
            INSERT INTO convergence_debt (
                stage, target_type, target_id, status, attempts, priority, updated_at_ms, last_error, next_retry_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "convergence",
                "session",
                "sess-2",
                "failed",
                1,
                5,
                int(datetime(2026, 6, 19, tzinfo=UTC).timestamp() * 1000) - 1,
                "generic failure",
                None,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    payload = archive_debt_list(archive_root=tmp_path, kinds=("convergence",))

    assert payload.totals.total == 2
    rows_by_stage = {row.stage: row for row in payload.rows}
    fts_row = rows_by_stage["fts"]
    assert fts_row.kind == "convergence"
    assert fts_row.subject_ref == "session:sess-1"
    assert fts_row.status == "actionable"
    assert fts_row.details == "boom"
    assert tuple(fts_row.actions[0].command) == (
        "polylogue",
        "ops",
        "maintenance",
        "run",
        "--target",
        "dangling_fts",
    )
    generic_row = rows_by_stage["convergence"]
    assert generic_row.subject_ref == "session:sess-2"
    assert generic_row.actions == ()


def test_archive_debt_converts_embedding_and_fts_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_current_tier_files(tmp_path)

    monkeypatch.setattr(
        module,
        "embedding_readiness_info",
        lambda _path, detail=False: {
            "embedding_config_enabled": True,
            "embedding_enabled": True,
            "embedding_has_voyage_key": True,
            "embedding_pending_count": 3,
            "embedding_pending_message_count": 30,
            "embedding_stale_count": 1,
            "embedding_failure_count": 2,
        },
    )
    monkeypatch.setattr(
        module,
        "fts_readiness_info",
        lambda _path, exact=False: {
            "invariant_ready": False,
            "surfaces": {
                "messages_fts": {
                    "source_exists": True,
                    "exists": True,
                    "triggers_present": False,
                    "ready": False,
                    "missing_rows": 7,
                    "excess_rows": 0,
                    "duplicate_rows": 0,
                }
            },
        },
    )

    payload = archive_debt_list(archive_root=tmp_path, kinds=("embedding", "fts"))

    refs = {row.debt_ref for row in payload.rows}
    assert "debt:embedding:catchup:failures" in refs
    assert "debt:embedding:catchup:backlog" in refs
    assert "debt:fts:messages_fts" in refs
    assert payload.totals.total == 3
    assert payload.totals.critical == 2


def _init_raw_materialization_fixture(root: Path) -> tuple[Path, Path, Path]:
    source_db = root / "source.db"
    index_db = root / "index.db"
    blob_root = root / "blob"
    blob_root.mkdir()
    source_file = root / "source.json"
    source_file.write_text("{}", encoding="utf-8")

    with sqlite3.connect(source_db) as conn:
        conn.execute(
            """
            CREATE TABLE raw_sessions (
                raw_id TEXT PRIMARY KEY,
                origin TEXT NOT NULL,
                native_id TEXT,
                source_path TEXT NOT NULL,
                source_index INTEGER NOT NULL DEFAULT 0,
                blob_hash BLOB NOT NULL,
                blob_size INTEGER NOT NULL,
                acquired_at_ms INTEGER NOT NULL,
                file_mtime_ms INTEGER,
                parsed_at_ms INTEGER,
                parse_error TEXT,
                validated_at_ms INTEGER,
                validation_status TEXT,
                validation_error TEXT,
                validation_drift_count INTEGER NOT NULL DEFAULT 0,
                validation_mode TEXT,
                detection_warnings_json TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO raw_sessions (
                raw_id, origin, native_id, source_path, blob_hash, blob_size,
                acquired_at_ms, parsed_at_ms, parse_error, validated_at_ms,
                validation_status
            ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
            """,
            (
                (
                    "raw-missing-blob",
                    "codex-session",
                    "codex-native",
                    str(source_file),
                    bytes.fromhex("aa" * 32),
                    2048,
                    None,
                    None,
                    None,
                    None,
                ),
                (
                    "raw-parsed-no-session",
                    "aistudio-drive",
                    None,
                    str(source_file),
                    bytes.fromhex("bb" * 32),
                    4096,
                    123,
                    None,
                    123,
                    "passed",
                ),
                (
                    "raw-skipped-non-session",
                    "aistudio-drive",
                    None,
                    str(source_file),
                    bytes.fromhex("cc" * 32),
                    1024,
                    123,
                    None,
                    123,
                    "skipped",
                ),
                (
                    "raw-claude-sidecar",
                    "claude-code-session",
                    "sidecar-native",
                    str(root / "sidecar.jsonl"),
                    bytes.fromhex("dd" * 32),
                    2048,
                    123,
                    None,
                    123,
                    "passed",
                ),
                (
                    "raw-claude-file-history-progress",
                    "claude-code-session",
                    "sidecar-native-progress",
                    str(root / "sidecar-progress.jsonl"),
                    bytes.fromhex("de" * 32),
                    2048,
                    123,
                    None,
                    123,
                    "passed",
                ),
                (
                    "raw-claude-metadata-descriptor",
                    "claude-code-session",
                    "metadata-native",
                    str(root / "metadata.jsonl"),
                    bytes.fromhex("df" * 32),
                    2048,
                    123,
                    None,
                    123,
                    "passed",
                ),
                (
                    "raw-codex-metadata-only",
                    "codex-session",
                    "codex-metadata-only",
                    str(root / "rollout.jsonl"),
                    bytes.fromhex("ee" * 32),
                    2048,
                    123,
                    None,
                    123,
                    "passed",
                ),
            ),
        )
    (blob_root / "bb").mkdir()
    (blob_root / "bb" / ("bb" * 31)).write_bytes(b"payload")
    (blob_root / "dd").mkdir()
    (blob_root / "dd" / ("dd" * 31)).write_text(
        '{"type":"file-history-snapshot","messageId":"m1"}\n',
        encoding="utf-8",
    )
    (blob_root / "de").mkdir()
    (blob_root / "de" / ("de" * 31)).write_text(
        '{"type":"file-history-snapshot","messageId":"m1"}\n{"type":"progress","messageId":"m1"}\n',
        encoding="utf-8",
    )
    (blob_root / "df").mkdir()
    (blob_root / "df" / ("df" * 31)).write_text(
        '{"sessionId":"s1","projectHash":"p","startTime":"2026-06-30T00:00:00Z",'
        '"lastUpdated":"2026-06-30T00:00:00Z","kind":"metadata"}\n',
        encoding="utf-8",
    )
    (blob_root / "ee").mkdir()
    (blob_root / "ee" / ("ee" * 31)).write_text(
        '{"type":"session_meta","timestamp":"2026-06-30T00:00:00Z"}\n',
        encoding="utf-8",
    )

    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT, origin TEXT, native_id TEXT, raw_id TEXT)")

    return source_db, index_db, source_file


def test_archive_debt_reports_raw_materialization_debt(tmp_path: Path) -> None:
    _init_raw_materialization_fixture(tmp_path)

    payload = archive_debt_list(archive_root=tmp_path, kinds=("raw-materialization",))

    by_ref = {row.debt_ref: row for row in payload.rows}
    missing_blob = by_ref["debt:raw-materialization:codex-session:missing-blob"]
    assert missing_blob.severity == "critical"
    assert missing_blob.status == "actionable"
    assert missing_blob.kind == "raw-materialization"
    assert "missing blob payloads" in missing_blob.summary
    assert any(ref.startswith("blob:") for ref in missing_blob.evidence_refs)

    parsed_gap = by_ref["debt:raw-materialization:aistudio-drive:parsed-without-session"]
    assert parsed_gap.severity == "warning"
    assert "parsed but have no materialized session" in parsed_gap.summary
    assert "passed=1" in (parsed_gap.details or "")

    sidecars = by_ref["debt:raw-materialization:claude-code-session:parsed-non-session-artifact"]
    assert sidecars.severity == "info"
    assert sidecars.status == "open"
    assert "parsed as non-session artifacts" in sidecars.summary
    assert "passed=3" in (sidecars.details or "")
    assert sidecars.actions == ()

    metadata_only = by_ref["debt:raw-materialization:codex-session:parsed-non-session-artifact"]
    assert metadata_only.severity == "info"
    assert "metadata-only" in (metadata_only.details or "") or "non-session artifacts" in metadata_only.summary


def test_archive_debt_reports_codex_zero_token_projection_debt(tmp_path: Path) -> None:
    _write_current_tier_files(tmp_path)
    index_db = tmp_path / "index.db"
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT NOT NULL)")
        conn.execute(
            """
            CREATE TABLE session_model_usage (
                session_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                cache_write_tokens INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute("INSERT INTO sessions (session_id, origin) VALUES ('codex-session:s1', 'codex-session')")
        conn.execute(
            """
            INSERT INTO session_model_usage (
                session_id, model_name, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
            ) VALUES ('codex-session:s1', 'gpt-5-codex', 0, 0, 0, 0)
            """
        )

    payload = archive_debt_list(archive_root=tmp_path, kinds=("provider-usage",))

    assert payload.totals.total == 1
    row = payload.rows[0]
    assert row.debt_ref == "debt:provider-usage:codex-session:zero-token-projection"
    assert row.kind == "provider-usage"
    assert row.stage == "usage-projection"
    assert row.severity == "warning"
    assert row.status == "open"
    assert "no projected token usage" in row.summary
    assert "not evidence that the sessions consumed no tokens" in row.caveats[0]


def test_archive_debt_ignores_codex_usage_rows_with_nonzero_tokens(tmp_path: Path) -> None:
    _write_current_tier_files(tmp_path)
    index_db = tmp_path / "index.db"
    with sqlite3.connect(index_db) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY, origin TEXT NOT NULL)")
        conn.execute(
            """
            CREATE TABLE session_model_usage (
                session_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                cache_write_tokens INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute("INSERT INTO sessions (session_id, origin) VALUES ('codex-session:s1', 'codex-session')")
        conn.execute(
            """
            INSERT INTO session_model_usage (
                session_id, model_name, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens
            ) VALUES ('codex-session:s1', 'gpt-5-codex', 10, 0, 0, 0)
            """
        )

    payload = archive_debt_list(archive_root=tmp_path, kinds=("provider-usage",))

    assert payload.rows == ()


def test_archive_debt_raw_materialization_ignores_materialized_rows(tmp_path: Path) -> None:
    source_db, index_db, _source_file = _init_raw_materialization_fixture(tmp_path)
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, origin, native_id, raw_id) VALUES ('s1', 'aistudio-drive', NULL, 'raw-parsed-no-session')"
        )

    payload = archive_debt_list(archive_root=tmp_path, kinds=("raw-materialization",))

    refs = {row.debt_ref for row in payload.rows}
    assert "debt:raw-materialization:aistudio-drive:parsed-without-session" not in refs
    assert "debt:raw-materialization:codex-session:missing-blob" in refs
    assert source_db.exists()


def test_archive_debt_raw_materialization_reports_native_id_aliases(tmp_path: Path) -> None:
    _source_db, index_db, _source_file = _init_raw_materialization_fixture(tmp_path)
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET native_id = ? WHERE raw_id = ?",
            ("aistudio-native", "raw-parsed-no-session"),
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, origin, native_id, raw_id) VALUES (?, ?, ?, ?)",
            ("aistudio-drive:aistudio-native", "aistudio-drive", "aistudio-native", "older-raw"),
        )

    payload = archive_debt_list(archive_root=tmp_path, kinds=("raw-materialization",))

    refs = {row.debt_ref: row for row in payload.rows}
    assert "debt:raw-materialization:aistudio-drive:parsed-without-session" not in refs
    alias = refs["debt:raw-materialization:aistudio-drive:materialized-alias"]
    assert alias.severity == "info"
    assert alias.status == "open"
    assert "materialized through native/source aliases" in alias.summary
    assert alias.actions == ()
    assert "debt:raw-materialization:codex-session:missing-blob" in refs


def test_archive_debt_raw_materialization_reports_source_path_native_aliases(tmp_path: Path) -> None:
    _source_db, index_db, _source_file = _init_raw_materialization_fixture(tmp_path)
    source_path = tmp_path / "drive-cache" / "gemini" / "native-alias_1.jsonl.txt.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("{}", encoding="utf-8")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET origin = ?, source_path = ? WHERE raw_id = ?",
            ("claude-code-session", str(source_path), "raw-parsed-no-session"),
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, origin, native_id, raw_id) VALUES (?, ?, ?, ?)",
            ("claude-code-session:native-alias", "claude-code-session", "native-alias", "older-raw"),
        )

    payload = archive_debt_list(archive_root=tmp_path, kinds=("raw-materialization",))

    refs = {row.debt_ref: row for row in payload.rows}
    assert "debt:raw-materialization:claude-code-session:parsed-without-session" not in refs
    alias = refs["debt:raw-materialization:claude-code-session:materialized-alias"]
    assert alias.severity == "info"
    assert alias.status == "open"
    assert "should not be replayed blindly" in (alias.details or "")
    assert "debt:raw-materialization:codex-session:missing-blob" in refs


def test_archive_debt_reports_partial_embedded_claude_code_aggregates(tmp_path: Path) -> None:
    _source_db, index_db, _source_file = _init_raw_materialization_fixture(tmp_path)
    source_path = tmp_path / "drive-cache" / "gemini" / "aggregate.jsonl.txt.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "\n".join(
            (
                '{"type":"user","sessionId":"materialized-session","uuid":"u1"}',
                '{"type":"assistant","sessionId":"missing-session","uuid":"u2"}',
            )
        ),
        encoding="utf-8",
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET origin = ?, source_path = ? WHERE raw_id = ?",
            ("claude-code-session", str(source_path), "raw-parsed-no-session"),
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, origin, native_id, raw_id) VALUES (?, ?, ?, ?)",
            (
                "claude-code-session:materialized-session",
                "claude-code-session",
                "materialized-session",
                "older-raw",
            ),
        )

    payload = archive_debt_list(archive_root=tmp_path, kinds=("raw-materialization",))

    refs = {row.debt_ref: row for row in payload.rows}
    debt = refs["debt:raw-materialization:claude-code-session:aggregate-partial-materialization"]
    assert "partially materialized" in debt.summary
    assert "1/2 embedded session id(s) materialized" in (debt.details or "")
    assert "debt:raw-materialization:claude-code-session:parsed-without-session" not in refs


def test_archive_debt_ignores_fully_materialized_embedded_claude_code_aggregates(tmp_path: Path) -> None:
    _source_db, index_db, _source_file = _init_raw_materialization_fixture(tmp_path)
    source_path = tmp_path / "drive-cache" / "gemini" / "aggregate.jsonl.txt.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "\n".join(
            (
                '{"type":"user","sessionId":"first-session","uuid":"u1"}',
                '{"type":"assistant","sessionId":"second-session","uuid":"u2"}',
            )
        ),
        encoding="utf-8",
    )
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            "UPDATE raw_sessions SET origin = ?, source_path = ? WHERE raw_id = ?",
            ("claude-code-session", str(source_path), "raw-parsed-no-session"),
        )
    with sqlite3.connect(index_db) as conn:
        conn.executemany(
            "INSERT INTO sessions (session_id, origin, native_id, raw_id) VALUES (?, ?, ?, ?)",
            (
                ("claude-code-session:first-session", "claude-code-session", "first-session", "raw-1"),
                ("claude-code-session:second-session", "claude-code-session", "second-session", "raw-2"),
            ),
        )

    payload = archive_debt_list(archive_root=tmp_path, kinds=("raw-materialization",))

    refs = {row.debt_ref for row in payload.rows}
    assert "debt:raw-materialization:claude-code-session:aggregate-partial-materialization" not in refs
    assert "debt:raw-materialization:claude-code-session:parsed-without-session" not in refs
    assert "debt:raw-materialization:claude-code-session:materialized-alias" not in refs
    assert "debt:raw-materialization:codex-session:missing-blob" in refs


def test_archive_debt_source_path_aliases_do_not_hide_parse_failures(tmp_path: Path) -> None:
    _source_db, index_db, _source_file = _init_raw_materialization_fixture(tmp_path)
    source_path = tmp_path / "drive-cache" / "codex-native.jsonl.txt.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("{}", encoding="utf-8")
    with sqlite3.connect(tmp_path / "source.db") as conn:
        conn.execute(
            """
            UPDATE raw_sessions
            SET native_id = NULL, source_path = ?, blob_hash = ?, parsed_at_ms = ?, parse_error = ?
            WHERE raw_id = ?
            """,
            (str(source_path), bytes.fromhex("bb" * 32), 123, "parser failed", "raw-missing-blob"),
        )
    with sqlite3.connect(index_db) as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, origin, native_id, raw_id) VALUES (?, ?, ?, ?)",
            ("codex-session:codex-native", "codex-session", "codex-native", "older-raw"),
        )

    payload = archive_debt_list(archive_root=tmp_path, kinds=("raw-materialization",))

    refs = {row.debt_ref for row in payload.rows}
    assert "debt:raw-materialization:codex-session:parse-failed" in refs

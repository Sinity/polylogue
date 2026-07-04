"""Fast-path coverage for ``polylogue ops embed status``."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.cli.commands.embed import embed_command
from polylogue.storage.embeddings import status_payload as status_payload_mod
from polylogue.storage.embeddings.progress import (
    CatchupRunStart,
    finish_embedding_catchup_run,
    start_embedding_catchup_run,
)


class _Cfg:
    def __init__(
        self,
        *,
        embedding_enabled: bool,
        voyage_api_key: str | None,
    ) -> None:
        self.embedding_enabled = embedding_enabled
        self.voyage_api_key = voyage_api_key
        self.embedding_model = "voyage-4"
        self.embedding_dimension = 1024
        self.embedding_max_cost_usd = 5.0


def _env(db_path: Path) -> Any:
    env = MagicMock()
    env.config.db_path = db_path
    return env


def _seed_archive_without_embedding_ledgers(db_path: Path, *, vec_table: bool = False) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE sessions (session_id TEXT PRIMARY KEY)")
        conn.execute(
            """
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT,
                role TEXT NOT NULL DEFAULT 'user',
                message_type TEXT NOT NULL DEFAULT 'message',
                material_origin TEXT NOT NULL DEFAULT 'human_authored',
                word_count INTEGER NOT NULL DEFAULT 8,
                content_hash TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO sessions (session_id) VALUES (?)",
            [("conv-1",), ("conv-2",)],
        )
        conn.executemany(
            "INSERT INTO messages (message_id, session_id, content_hash) VALUES (?, ?, ?)",
            [("msg-1", "conv-1", "h1"), ("msg-2", "conv-2", "h2")],
        )
        if vec_table:
            conn.execute("CREATE TABLE message_embeddings (message_id TEXT PRIMARY KEY)")
        conn.commit()


def _seed_archive_file_set_from_archive_tiers(index_db: Path) -> None:
    embeddings_db = index_db.with_name("embeddings.db")
    with sqlite3.connect(index_db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                message_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                message_type TEXT NOT NULL DEFAULT 'message',
                material_origin TEXT NOT NULL DEFAULT 'human_authored',
                word_count INTEGER NOT NULL DEFAULT 8,
                content_hash BLOB NOT NULL
            );
            INSERT INTO sessions VALUES ('codex-session:complete', 1);
            INSERT INTO sessions VALUES ('codex-session:pending', 2);
            INSERT INTO messages (message_id, session_id, content_hash)
            VALUES ('codex-session:complete:m1', 'codex-session:complete', x'01');
            INSERT INTO messages (message_id, session_id, content_hash)
            VALUES ('codex-session:pending:m1', 'codex-session:pending', x'02');
            INSERT INTO messages (message_id, session_id, content_hash)
            VALUES ('codex-session:pending:m2', 'codex-session:pending', x'03');
            """
        )
        conn.commit()
    with sqlite3.connect(embeddings_db) as conn:
        conn.executescript(
            """
            CREATE TABLE message_embeddings (
                message_id TEXT PRIMARY KEY
            );
            CREATE TABLE message_embeddings_meta (
                message_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                embedded_at_ms INTEGER NOT NULL,
                content_hash BLOB,
                needs_reindex INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE embedding_status (
                session_id TEXT PRIMARY KEY,
                origin TEXT NOT NULL,
                message_count_embedded INTEGER NOT NULL DEFAULT 0,
                needs_reindex INTEGER NOT NULL DEFAULT 0,
                error_message TEXT
            );
            INSERT INTO message_embeddings VALUES ('codex-session:complete:m1');
            INSERT INTO message_embeddings_meta VALUES (
                'codex-session:complete:m1', 'voyage-4', 1024, 1767225700000, x'01', 0
            );
            INSERT INTO embedding_status VALUES ('codex-session:complete', 'codex-session', 1, 0, NULL);
            """
        )
        conn.commit()


def _payload(result_output: str) -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(result_output))


def _run_status(db_path: Path, *args: str, cfg: _Cfg | None = None) -> dict[str, Any]:
    runner = CliRunner(env={"POLYLOGUE_FORCE_PLAIN": "1"})
    with patch(
        "polylogue.config.load_polylogue_config",
        return_value=cfg or _Cfg(embedding_enabled=False, voyage_api_key=None),
    ):
        result = runner.invoke(
            embed_command,
            ["status", "--format", "json", *args],
            obj=_env(db_path),
            catch_exceptions=False,
        )
    assert result.exit_code == 0
    return _payload(result.output)


def _run_status_text(db_path: Path, *, cfg: _Cfg | None = None) -> str:
    runner = CliRunner(env={"POLYLOGUE_FORCE_PLAIN": "1"})
    with patch(
        "polylogue.config.load_polylogue_config",
        return_value=cfg or _Cfg(embedding_enabled=False, voyage_api_key=None),
    ):
        result = runner.invoke(embed_command, ["status"], obj=_env(db_path), catch_exceptions=False)
    assert result.exit_code == 0
    return str(result.output)


def test_status_json_fast_path_handles_absent_embedding_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)

    payload = _run_status(db_path)

    assert payload["status"] == "none"
    assert payload["total_sessions"] == 2
    assert payload["embedded_sessions"] == 0
    assert payload["pending_sessions"] == 2
    assert payload["pending_messages"] is None
    assert payload["pending_messages_exact"] is False
    assert payload["retrieval_bands"] == {}


def test_status_json_reads_archive_file_set_from_archive_index(tmp_path: Path) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")

    payload = _run_status(db_anchor, "--detail", cfg=_Cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["status"] == "partial"
    assert payload["total_sessions"] == 2
    assert payload["embedded_sessions"] == 1
    assert payload["pending_sessions"] == 1
    assert payload["embedded_messages"] == 1
    assert payload["pending_messages"] == 2
    assert payload["pending_messages_exact"] is True
    assert payload["candidate_prose_messages"] == 3
    assert payload["candidate_prose_messages_exact"] is True
    assert payload["stale_messages"] == 0
    assert payload["retrieval_ready"] is True
    assert payload["freshness_status"] == "partial"
    assert payload["embedding_coverage_percent"] == 50.0
    assert payload["embedding_coverage_basis"] == "sessions"
    assert payload["message_coverage_percent"] == 33.3
    assert payload["embedding_models"] == {"voyage-4": 1}
    assert payload["embedding_dimensions"] == {"1024": 1} or payload["embedding_dimensions"] == {1024: 1}


def test_status_json_reports_archive_embedding_metadata_without_detail(tmp_path: Path) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")

    payload = _run_status(db_anchor, cfg=_Cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["pending_messages"] is None
    assert payload["pending_messages_exact"] is False
    assert payload["embedding_models"] == {"voyage-4": 1}
    assert payload["embedding_dimensions"] == {"1024": 1} or payload["embedding_dimensions"] == {1024: 1}
    assert payload["oldest_embedded_at"] == "2026-01-01T00:01:40+00:00"
    assert payload["newest_embedded_at"] == "2026-01-01T00:01:40+00:00"


def test_status_json_detail_uses_analyzed_prose_index_for_candidate_count(tmp_path: Path) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    index_db = tmp_path / "index.db"
    _seed_archive_file_set_from_archive_tiers(index_db)
    with sqlite3.connect(index_db) as conn:
        conn.executescript(
            """
            CREATE INDEX idx_messages_embedding_prose
            ON messages(session_id, message_id)
            WHERE message_type = 'message'
              AND role IN ('user', 'assistant')
              AND material_origin IN ('human_authored', 'assistant_authored')
              AND word_count > 0;
            ANALYZE idx_messages_embedding_prose;
            """
        )

    payload = _run_status(db_anchor, "--detail", cfg=_Cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["candidate_prose_messages"] == 3
    assert payload["candidate_prose_messages_exact"] is False
    assert payload["message_coverage_percent"] == 33.3


def test_status_json_default_does_not_exact_count_archive_session_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")

    def fail_exact_session_state(*args: object, **kwargs: object) -> object:
        raise AssertionError("default status must not scan exact archive embedding session state")

    monkeypatch.setattr(status_payload_mod, "count_archive_embedding_session_state", fail_exact_session_state)

    payload = _run_status(db_anchor, cfg=_Cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["status"] == "partial"
    assert payload["embedded_sessions"] == 1
    assert payload["pending_sessions"] == 1
    assert payload["pending_messages"] is None
    assert payload["pending_messages_exact"] is False


def test_status_json_default_skips_embedding_metadata_summary_scans(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")
    real_rows_with_timeout = status_payload_mod._rows_with_timeout

    def reject_metadata_summary_rows(
        conn: sqlite3.Connection,
        sql: str,
        *,
        timeout_ms: int,
        params: tuple[object, ...] = (),
    ) -> list[sqlite3.Row | tuple[object, ...]] | None:
        if "message_embeddings_meta" in sql:
            raise AssertionError("default embedding status must not scan metadata summaries")
        return real_rows_with_timeout(conn, sql, timeout_ms=timeout_ms, params=params)

    monkeypatch.setattr(status_payload_mod, "_rows_with_timeout", reject_metadata_summary_rows)

    payload = _run_status(db_anchor, cfg=_Cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["status"] == "partial"
    assert payload["embedding_models"] == {}
    assert payload["embedding_dimensions"] == {}


def test_status_json_detail_uses_uniform_metadata_probe_when_grouping_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")
    real_rows_with_timeout = status_payload_mod._rows_with_timeout

    def fake_rows_with_timeout(
        conn: sqlite3.Connection,
        sql: str,
        *,
        timeout_ms: int,
        params: tuple[object, ...] = (),
    ) -> list[sqlite3.Row | tuple[object, ...]] | None:
        if "GROUP BY" in sql and "message_embeddings_meta" in sql:
            return None
        return real_rows_with_timeout(conn, sql, timeout_ms=timeout_ms, params=params)

    monkeypatch.setattr(status_payload_mod, "_rows_with_timeout", fake_rows_with_timeout)

    payload = _run_status(db_anchor, "--detail", cfg=_Cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["embedding_models"] == {"voyage-4": 1}
    assert payload["embedding_dimensions"] == {"1024": 1} or payload["embedding_dimensions"] == {1024: 1}


def test_status_json_uses_status_ledger_for_archive_embedded_sessions(tmp_path: Path) -> None:
    index_db = tmp_path / "index.db"
    embeddings_db = tmp_path / "embeddings.db"
    with sqlite3.connect(index_db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                authored_user_message_count INTEGER NOT NULL DEFAULT 0,
                assistant_message_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                message_type TEXT NOT NULL DEFAULT 'message',
                material_origin TEXT NOT NULL DEFAULT 'human_authored',
                word_count INTEGER NOT NULL DEFAULT 8,
                content_hash BLOB NOT NULL
            );
            INSERT INTO sessions VALUES ('codex-session:complete', 20, 20);
            INSERT INTO sessions VALUES ('codex-session:pending', 3, 1);
            INSERT INTO messages (message_id, session_id, content_hash)
            VALUES ('codex-session:complete:m1', 'codex-session:complete', x'01');
            INSERT INTO messages (message_id, session_id, content_hash)
            VALUES ('codex-session:pending:m1', 'codex-session:pending', x'02');
            """
        )
    with sqlite3.connect(embeddings_db) as conn:
        conn.executescript(
            """
            CREATE TABLE message_embeddings_meta (
                message_id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                embedded_at_ms INTEGER NOT NULL,
                content_hash BLOB,
                needs_reindex INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE embedding_status (
                session_id TEXT PRIMARY KEY,
                origin TEXT NOT NULL,
                message_count_embedded INTEGER NOT NULL DEFAULT 0,
                needs_reindex INTEGER NOT NULL DEFAULT 0,
                error_message TEXT
            );
            INSERT INTO embedding_status VALUES ('codex-session:complete', 'codex-session', 1, 0, NULL);
            """
        )

    payload = _run_status(tmp_path / "custom.sqlite", cfg=_Cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["embedded_sessions"] == 1
    assert payload["pending_sessions"] == 1
    assert payload["embedding_coverage_percent"] == 50.0


def test_status_json_detail_falls_back_when_exact_pending_count_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")
    original_scalar = status_payload_mod._scalar_int_with_timeout

    def fake_scalar_int_with_timeout(conn: sqlite3.Connection, sql: str, *, timeout_ms: int) -> int | None:
        if "LEFT JOIN embeddings.message_embeddings_meta" in sql and "LEFT JOIN embeddings.embedding_status" in sql:
            return None
        return original_scalar(conn, sql, timeout_ms=timeout_ms)

    monkeypatch.setattr(status_payload_mod, "_scalar_int_with_timeout", fake_scalar_int_with_timeout)

    payload = _run_status(db_anchor, "--detail", cfg=_Cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["status"] == "partial"
    assert payload["pending_sessions"] == 1
    assert payload["pending_messages"] is None
    assert payload["pending_messages_exact"] is False
    assert payload["candidate_prose_messages"] == 3
    assert payload["candidate_prose_messages_exact"] is True
    assert payload["message_coverage_percent"] == 33.3
    assert payload["total_estimated_cost_usd"] is None
    assert payload["retrieval_ready"] is True


def test_status_json_detail_falls_back_when_exact_session_state_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")

    def interrupted_session_state(*args: object, **kwargs: object) -> object:
        raise sqlite3.OperationalError("interrupted")

    monkeypatch.setattr(status_payload_mod, "count_archive_embedding_session_state", interrupted_session_state)

    payload = _run_status(db_anchor, "--detail", cfg=_Cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["status"] == "partial"
    assert payload["embedded_sessions"] == 1
    assert payload["pending_sessions"] == 1
    assert payload["pending_messages"] is None
    assert payload["pending_messages_exact"] is False
    assert payload["candidate_prose_messages"] == 3
    assert payload["candidate_prose_messages_exact"] is True
    assert payload["message_coverage_percent"] == 33.3
    assert payload["total_estimated_cost_usd"] is None


def test_status_text_detail_does_not_claim_zero_cost_when_exact_pending_count_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")
    original_scalar = status_payload_mod._scalar_int_with_timeout

    def fake_scalar_int_with_timeout(conn: sqlite3.Connection, sql: str, *, timeout_ms: int) -> int | None:
        if "LEFT JOIN embeddings.message_embeddings_meta" in sql and "LEFT JOIN embeddings.embedding_status" in sql:
            return None
        return original_scalar(conn, sql, timeout_ms=timeout_ms)

    monkeypatch.setattr(status_payload_mod, "_scalar_int_with_timeout", fake_scalar_int_with_timeout)

    runner = CliRunner(env={"POLYLOGUE_FORCE_PLAIN": "1"})
    with patch(
        "polylogue.config.load_polylogue_config",
        return_value=_Cfg(embedding_enabled=True, voyage_api_key="vk-live"),
    ):
        result = runner.invoke(
            embed_command,
            ["status", "--detail"],
            obj=_env(db_anchor),
            catch_exceptions=False,
        )

    assert result.exit_code == 0
    assert "Pending:              1 convs, msgs not calculated" in result.output
    assert "Session coverage:     50.0%" in result.output
    assert "Message coverage:     33.3% of 3 candidate prose msgs" in result.output
    assert "Estimated total cost: unknown" in result.output
    assert "use --detail" not in result.output


def test_status_json_reports_manual_backfill_when_config_disabled_but_partial(tmp_path: Path) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")

    payload = _run_status(db_anchor, cfg=_Cfg(embedding_enabled=False, voyage_api_key="vk-live"))

    assert payload["status"] == "partial"
    assert payload["retrieval_ready"] is True
    assert payload["config_enabled"] is False
    assert payload["daemon_stage_enabled"] is False
    assert payload["next_action"] == {
        "code": "continue_backfill",
        "command": "polylogue ops embed backfill --yes --max-sessions 10",
        "reason": (
            "Manual embedding coverage exists, but daemon convergence is disabled; "
            "continue bounded backfill or enable daemon catch-up."
        ),
    }


def test_status_json_reads_latest_catchup_from_ops_db(tmp_path: Path) -> None:
    db_anchor = tmp_path / "index.db"
    archive_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
    _seed_archive_file_set_from_archive_tiers(archive_db)
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.ops_write import upsert_embedding_catchup_run
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(ops_db) as conn:
        upsert_embedding_catchup_run(
            conn,
            run_id="v1-run",
            status="completed",
            started_at_ms=1_767_225_700_000,
            finished_at_ms=1_767_225_705_000,
            scanned_sessions=2,
            embedded_sessions=2,
            error_count=0,
            embedded_messages=4,
            estimated_cost_usd=0.001,
        )

    payload = _run_status(db_anchor, cfg=_Cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    latest = payload["latest_catchup_run"]
    assert latest["run_id"] == "v1-run"
    assert latest["status"] == "completed"
    assert latest["processed_sessions"] == 2
    assert latest["embedded_sessions"] == 2
    assert latest["error_count"] == 0
    assert latest["embedded_messages"] == 4
    assert latest["estimated_cost_usd"] == 0.001
    assert payload["latest_material_catchup_run"] == latest


def test_status_json_distinguishes_latest_material_archive_catchup(tmp_path: Path) -> None:
    db_anchor = tmp_path / "index.db"
    archive_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
    _seed_archive_file_set_from_archive_tiers(archive_db)
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.ops_write import upsert_embedding_catchup_run
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(ops_db) as conn:
        upsert_embedding_catchup_run(
            conn,
            run_id="material-run",
            status="completed",
            started_at_ms=1_767_225_700_000,
            finished_at_ms=1_767_225_705_000,
            scanned_sessions=63,
            embedded_sessions=63,
            error_count=0,
            embedded_messages=2_818,
            estimated_cost_usd=0.1409,
        )
        upsert_embedding_catchup_run(
            conn,
            run_id="zero-progress-run",
            status="completed",
            started_at_ms=1_767_225_800_000,
            finished_at_ms=1_767_225_801_000,
            scanned_sessions=25,
            embedded_sessions=0,
            error_count=0,
            embedded_messages=0,
            estimated_cost_usd=0.0,
        )

    payload = _run_status(db_anchor, cfg=_Cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    latest = payload["latest_catchup_run"]
    material = payload["latest_material_catchup_run"]
    assert latest["run_id"] == "zero-progress-run"
    assert latest["processed_sessions"] == 25
    assert latest["embedded_messages"] == 0
    assert material["run_id"] == "material-run"
    assert material["embedded_sessions"] == 63
    assert material["embedded_messages"] == 2_818
    assert material["estimated_cost_usd"] == 0.1409


def test_status_json_treats_skipped_archive_catchup_as_material(tmp_path: Path) -> None:
    db_anchor = tmp_path / "index.db"
    archive_db = tmp_path / "index.db"
    ops_db = tmp_path / "ops.db"
    _seed_archive_file_set_from_archive_tiers(archive_db)
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.ops_write import upsert_embedding_catchup_run
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_archive_database(ops_db, ArchiveTier.OPS)
    with sqlite3.connect(ops_db) as conn:
        upsert_embedding_catchup_run(
            conn,
            run_id="skipped-run",
            status="completed",
            started_at_ms=1_767_225_900_000,
            finished_at_ms=1_767_225_901_000,
            scanned_sessions=25,
            embedded_sessions=0,
            skipped_sessions=25,
            error_count=0,
            embedded_messages=0,
            estimated_cost_usd=0.0,
        )

    payload = _run_status(db_anchor, cfg=_Cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    latest = payload["latest_catchup_run"]
    material = payload["latest_material_catchup_run"]
    assert latest["run_id"] == "skipped-run"
    assert latest["skipped_sessions"] == 25
    assert material == latest


def test_status_json_reads_index_when_db_anchor_exists(tmp_path: Path) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_without_embedding_ledgers(db_anchor)
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")

    payload = _run_status(
        db_anchor,
        cfg=_Cfg(embedding_enabled=True, voyage_api_key="vk-live"),
    )

    assert payload["status"] == "partial"
    assert payload["total_sessions"] == 2
    assert payload["embedded_sessions"] == 1
    assert payload["pending_sessions"] == 1


def test_status_json_bypasses_schema_version_gate_for_operator_readiness(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA user_version = 9")

    payload = _run_status(db_path, cfg=_Cfg(embedding_enabled=False, voyage_api_key="vk-live"))

    assert payload["status"] == "none"
    assert payload["config_enabled"] is False
    assert payload["has_voyage_api_key"] is True
    assert payload["configured_model"] == "voyage-4"
    assert payload["configured_dimension"] == 1024
    assert payload["monthly_cost_cap_usd"] == 5.0
    assert payload["pending_sessions"] == 2
    assert payload["next_action"] == {
        "code": "enable_embeddings",
        "command": "polylogue ops embed enable --yes",
        "reason": "A Voyage key is available, but embedding convergence is disabled in config.",
    }


def test_status_json_counts_empty_vec0_rowids_as_zero_embeddings(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path, vec_table=True)

    payload = _run_status(db_path)

    assert payload["embedded_messages"] == 0
    assert payload["retrieval_ready"] is False
    assert payload["freshness_status"] == "none"


@pytest.mark.parametrize(
    ("cfg", "config_enabled", "has_key", "stage_enabled"),
    [
        (_Cfg(embedding_enabled=False, voyage_api_key="vk-live"), False, True, False),
        (_Cfg(embedding_enabled=True, voyage_api_key=None), True, False, False),
    ],
)
def test_status_json_reports_config_gate_combinations(
    tmp_path: Path,
    cfg: _Cfg,
    config_enabled: bool,
    has_key: bool,
    stage_enabled: bool,
) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)

    payload = _run_status(db_path, cfg=cfg)

    assert payload["config_enabled"] is config_enabled
    assert payload["has_voyage_api_key"] is has_key
    assert payload["daemon_stage_enabled"] is stage_enabled
    assert payload["next_action"]["code"] == ("set_voyage_key" if not has_key else "enable_embeddings")


def test_status_json_detail_mode_stays_embedding_scoped(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)

    payload = _run_status(db_path, "--detail")

    assert payload["pending_sessions"] == 2
    assert payload["pending_messages"] == 2
    assert payload["pending_messages_exact"] is True
    assert payload["retrieval_bands"] == {}


def test_status_json_detail_matches_archive_embedding_text_floor(tmp_path: Path) -> None:
    index_db = tmp_path / "index.db"
    with sqlite3.connect(index_db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (session_id TEXT PRIMARY KEY);
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                text TEXT,
                role TEXT NOT NULL DEFAULT 'user',
                message_type TEXT NOT NULL DEFAULT 'message',
                material_origin TEXT NOT NULL DEFAULT 'human_authored',
                word_count INTEGER NOT NULL DEFAULT 1,
                content_hash TEXT
            );
            INSERT INTO sessions VALUES ('conv-1');
            INSERT INTO messages (
                message_id, session_id, text, role, message_type, material_origin, word_count, content_hash
            ) VALUES
                ('msg-long', 'conv-1', 'authored prose long enough', 'user', 'message', 'human_authored', 4, 'h1'),
                ('msg-short', 'conv-1', 'tiny', 'user', 'message', 'human_authored', 1, 'h2');
            """
        )
        conn.commit()

    payload = _run_status(index_db, "--detail")

    assert payload["pending_messages"] == 1
    assert payload["pending_messages_exact"] is True
    assert payload["candidate_prose_messages"] == 2
    assert payload["candidate_prose_messages_exact"] is True
    assert payload["message_coverage_percent"] == 0.0


def test_status_json_includes_latest_catchup_run(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)

    run_id = start_embedding_catchup_run(
        db_path,
        CatchupRunStart(
            rebuild=True,
            max_sessions=2,
            max_messages=10,
            stop_after_seconds=None,
            max_errors=None,
            planned_sessions=2,
            planned_messages=2,
        ),
    )
    finish_embedding_catchup_run(db_path, run_id, status="interrupted", stop_reason="keyboard interrupt")

    payload = _run_status(db_path)

    latest = payload["latest_catchup_run"]
    assert latest["run_id"] == run_id
    assert latest["status"] == "interrupted"
    assert latest["stop_reason"] == "keyboard interrupt"
    assert latest["rebuild"] is True
    assert latest["planned_sessions"] == 2


def test_status_text_prints_machine_readable_next_action(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)

    output = _run_status_text(db_path, cfg=_Cfg(embedding_enabled=False, voyage_api_key="vk-live"))

    assert "Configured model:     voyage-4 (1024d)" in output
    assert "Monthly cost cap:     $5.00" in output
    assert "Total sessions:       2" in output
    assert "Embedded sessions:    0" in output
    assert "Embedded messages:    0" in output
    assert "Next action:          enable_embeddings" in output
    assert "Command:              polylogue ops embed enable --yes" in output


def test_status_text_prints_manual_backfill_when_config_disabled_but_partial(tmp_path: Path) -> None:
    db_anchor = tmp_path / "custom.sqlite"
    _seed_archive_file_set_from_archive_tiers(tmp_path / "index.db")

    output = _run_status_text(db_anchor, cfg=_Cfg(embedding_enabled=False, voyage_api_key="vk-live"))

    assert "Next action:          continue_backfill" in output
    assert "Command:              polylogue ops embed backfill --yes --max-sessions 10" in output


def test_status_text_prints_daemon_catchup_when_enabled(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)

    output = _run_status_text(db_path, cfg=_Cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert "Next action:          drain_backlog" in output
    assert "Command:              polylogue ops embed backfill --yes --max-sessions 10" in output


def test_status_json_reports_ready_next_action(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    _seed_archive_without_embedding_ledgers(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE embedding_status (
                session_id TEXT PRIMARY KEY,
                embedded_message_count INTEGER,
                needs_reindex INTEGER DEFAULT 0,
                error_message TEXT
            )
            """
        )
        conn.execute("CREATE TABLE message_embeddings (message_id TEXT PRIMARY KEY)")
        conn.executemany(
            "INSERT INTO embedding_status (session_id, embedded_message_count, needs_reindex) VALUES (?, ?, 0)",
            [("conv-1", 1), ("conv-2", 1)],
        )
        conn.executemany("INSERT INTO message_embeddings (message_id) VALUES (?)", [("msg-1",), ("msg-2",)])
        conn.commit()

    payload = _run_status(db_path, cfg=_Cfg(embedding_enabled=True, voyage_api_key="vk-live"))

    assert payload["status"] == "complete"
    assert payload["retrieval_ready"] is True
    assert payload["next_action"] == {
        "code": "ready",
        "command": "polylogue --semantic <query>",
        "reason": "Embeddings are retrieval-ready.",
    }

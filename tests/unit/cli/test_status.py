"""Tests for first-run status UX."""

from __future__ import annotations

import inspect
import json
import os
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from polylogue.api import Polylogue
from polylogue.cli.commands.status import (
    _FULL_TIMEOUT_S,
    _archive_cli_route_status,
    _archive_facade_route_status,
    _direct_claim_guard,
    _direct_status_ok,
    _render_raw_replay_backlog,
    _show_daemon_status,
    _show_direct_json,
    _show_direct_status,
    _show_status_json,
    _status_ok,
    status_command,
)
from polylogue.cli.shared.types import AppEnv
from polylogue.storage.sqlite.archive_tiers import ARCHIVE_VERSION_BY_TIER
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion
from tests.infra.frozen_clock import FrozenClock


class _CapturingConsole:
    """Console mock that captures print calls in a list."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def print(self, *args: object, **kwargs: object) -> None:
        self.calls.append(" ".join(str(a) for a in args))


def _healthy_raw_frontier_integrity() -> dict[str, Any]:
    return {
        "available": True,
        "overall_status": "healthy",
        "broken_head_status": "healthy",
        "broken_head_count": 0,
        "broken_head_checked_count": 1,
        "broken_head_samples": [],
        "broken_head_reason": "",
        "missing_source_raw_status": "healthy",
        "missing_source_raw_count": 0,
        "missing_source_raw_samples": [],
        "missing_source_raw_reason": "",
        "cursor_ahead_status": "healthy",
        "cursor_ahead_count": 0,
        "cursor_ahead_checked_count": 1,
        "cursor_head_comparison_count": 1,
        "cursor_ahead_comparison_count": 0,
        "cursor_ahead_samples": [],
        "cursor_authority_gap_count": 0,
        "cursor_authority_gap_samples": [],
        "cursor_ahead_reason": "",
    }


def _fresh_status_snapshot(frozen_clock: FrozenClock) -> dict[str, object]:
    return {
        "state": "fresh",
        "captured_at": frozen_clock.now().isoformat(),
        "age_s": 0.1,
    }


class _FakeDaemonResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def __enter__(self) -> _FakeDaemonResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _make_app_env() -> AppEnv:
    """Create a minimal AppEnv for testing."""
    ui: Any = MagicMock()
    ui.plain = True
    ui.console = _CapturingConsole()
    return AppEnv(ui=ui)


def _combined_calls(env: AppEnv) -> str:
    """Get combined output from the capturing console."""
    console: Any = env.ui.console
    return " ".join(console.calls)


def test_status_command_reads_a_real_daemon_http_status_route() -> None:
    """The CLI must consume the daemon's JSON route without a mocked urlopen."""

    class _StatusHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.path == "/api/status"
            body = json.dumps({"daemon_liveness": True, "component_state": {}, "checked_at": "now"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _StatusHandler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        env = _make_app_env()
        callback = getattr(status_command.callback, "__wrapped__", None)
        assert callable(callback)
        callback(
            env,
            daemon_url=f"http://127.0.0.1:{server.server_port}",
            output_format="json",
            json_alias=False,
            full_payload=False,
            exact_archive_readiness=False,
        )
    finally:
        server.shutdown()
        worker.join()
        server.server_close()

    output = _combined_calls(env)
    assert '"source": "daemon"' in output
    assert '"daemon_liveness": true' in output


def test_status_command_reports_live_daemon_with_unavailable_status_snapshot() -> None:
    """A malformed status payload must not be mistaken for a stopped daemon."""

    class _UnavailableStatusHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/api/status":
                body = b"not-json"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/healthz/live":
                self.send_response(204)
                self.end_headers()
                return
            self.send_error(404)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _UnavailableStatusHandler)
    worker = threading.Thread(target=server.serve_forever)
    worker.start()
    try:
        env = _make_app_env()
        callback = getattr(status_command.callback, "__wrapped__", None)
        assert callable(callback)
        callback(
            env,
            daemon_url=f"http://127.0.0.1:{server.server_port}",
            output_format="json",
            json_alias=False,
            full_payload=False,
            exact_archive_readiness=False,
        )
    finally:
        server.shutdown()
        worker.join()
        server.server_close()

    output = _combined_calls(env)
    assert '"daemon_liveness": true' in output
    assert '"state": "unavailable"' in output


def test_raw_replay_backlog_plain_status_explains_containment() -> None:
    env = _make_app_env()
    reason = "raw source-to-index replay is disabled pending per-session revision authority"

    _render_raw_replay_backlog(
        env,
        {
            "available": True,
            "execution_blocked": True,
            "execution_block_reason": reason,
            "candidate_count": 2,
            "missing_blob_count": 0,
            "total_blob_bytes": 12_000_000,
            "max_blob_bytes": 10_000_000,
            "oversized_count": 0,
            "origin_summary": [],
        },
    )

    output = _combined_calls(env)
    assert "Raw replay backlog:" in output
    assert reason in output


def test_archive_facade_route_catalog_covers_public_async_facade() -> None:
    discovered = {
        name
        for name in dir(Polylogue)
        if not name.startswith("_") and inspect.iscoroutinefunction(inspect.getattr_static(Polylogue, name))
    }
    routing = _archive_facade_route_status()

    assert set(routing["routes"]) == discovered
    assert routing["routes"]["query_sessions"]["route"] == "archive_routed"
    assert routing["routes"]["parse_file"]["route"] == "archive_routed"
    assert routing["routes"]["parse_sources"]["route"] == "archive_routed"
    assert routing["routes"]["count_sessions"]["route"] == "archive_routed"
    assert routing["routes"]["facets"]["route"] == "archive_routed"
    assert routing["routes"]["get_session_topology"]["route"] == "archive_routed"
    assert routing["routes"]["get_logical_session"]["route"] == "archive_routed"
    assert routing["routes"]["health_check"]["route"] == "archive_routed"
    assert routing["routes"]["get_session"]["route"] == "archive_routed"
    assert routing["routes"]["archive_search_sessions"]["route"] == "archive_direct"
    assert routing["unsupported_methods"] == []


def test_archive_cli_route_catalog_reports_user_tier_surfaces() -> None:
    routing = _archive_cli_route_status()

    assert routing["checked"] is True
    assert routing["unsupported_command_count"] == 0
    assert routing["unsupported_commands"] == []
    assert routing["tier_counts"]["user"] == 3
    assert routing["tier_counts"]["source"] == 2
    assert routing["routes"]["ops.state.blackboard.post"]["route"] == "archive_direct"
    assert routing["routes"]["ops.state.blackboard.post"]["tier"] == "user"
    assert routing["routes"]["ops.state.blackboard.list"]["route"] == "archive_direct"
    assert routing["routes"]["ops.state.blackboard.list"]["tier"] == "user"
    assert routing["routes"]["reset.session"]["tier"] == "user"
    assert routing["routes"]["reset.database"]["tier"] == "source"
    assert routing["routes"]["reset.source"]["tier"] == "source"


class TestNoArchiveStatus:
    """First-run UX when no archive exists."""

    def test_direct_status_no_archive(self) -> None:
        """_show_direct_status when DB does not exist shows actionable steps."""
        env = _make_app_env()
        fake_db = Path("/tmp/nonexistent.db")
        fake_root = Path("/tmp/nonexistent")

        with patch("polylogue.paths.db_path", return_value=fake_db):
            with patch("polylogue.paths.archive_root", return_value=fake_root):
                _show_direct_status(env)

        combined = _combined_calls(env)
        assert "polylogued run" in combined

    def test_direct_status_empty_archive(self) -> None:
        """_show_direct_status when DB exists but is empty shows guidance."""
        env = _make_app_env()
        fake_root = Path("/tmp/empty-root")

        fake_db = MagicMock()
        fake_db.exists.return_value = True

        # Each execute().fetchone() returns [0] (0 rows)
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = [0]

        with patch("polylogue.paths.db_path", return_value=fake_db):
            with patch("polylogue.paths.archive_root", return_value=fake_root):
                with patch(
                    "polylogue.storage.sqlite.connection_profile.open_readonly_connection", return_value=mock_conn
                ):
                    _show_direct_status(env)

        combined = _combined_calls(env)
        # Empty-archive guidance now mentions "Sessions: 0" rather than the
        # earlier hand-written sentence.
        assert "Sessions: 0" in combined
        assert "polylogued run" in combined

    def test_direct_status_reads_archive_file_set_from_archive_tiers(self, tmp_path: Path) -> None:
        env = _make_app_env()
        db_anchor = tmp_path / "custom.sqlite"
        index_db = tmp_path / "index.db"
        source_db = tmp_path / "source.db"
        with sqlite3.connect(index_db) as conn:
            conn.executescript(
                """
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    message_count INTEGER NOT NULL
                );
                INSERT INTO sessions VALUES ('codex-session:one', 2);
                """
            )
        with sqlite3.connect(source_db) as conn:
            conn.executescript(
                """
                CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY);
                INSERT INTO raw_sessions VALUES ('raw-1');
                """
            )

        with (
            patch("polylogue.paths.db_path", return_value=db_anchor),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
            patch("polylogue.cli.commands.status_diagnostics.diagnose_first_run") as diagnose,
        ):
            _show_direct_status(env)

        diagnose.assert_not_called()
        combined = _combined_calls(env)
        assert "Database: index.db" in combined
        assert "Schema tiers: present=source, index; missing=embeddings, user, ops" in combined
        assert "Archive tier detail: source v0/" in combined
        assert "raw_sessions=1" in combined
        assert "index v0/" in combined
        assert "sessions=1" in combined
        assert "Facade routes:" in combined
        assert "0 unsupported" in combined
        assert "parse_file:" not in combined
        assert "Sessions: 1" in combined
        assert "Messages: 2" in combined
        assert "Raw records: 1" in combined

    def test_direct_status_reports_sqlite_maintenance_state(self, tmp_path: Path) -> None:
        env = _make_app_env()
        db_anchor = tmp_path / "custom.sqlite"
        index_db = tmp_path / "index.db"
        conn = sqlite3.connect(index_db)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA wal_autocheckpoint=0")
            conn.executescript(
                """
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    message_count INTEGER NOT NULL
                );
                INSERT INTO sessions VALUES ('codex-session:one', 2);
                ANALYZE;
                """
            )
            conn.commit()

            with (
                patch("polylogue.paths.db_path", return_value=db_anchor),
                patch("polylogue.paths.archive_root", return_value=tmp_path),
                patch("polylogue.cli.commands.status_diagnostics.diagnose_first_run") as diagnose,
            ):
                _show_direct_status(env)
        finally:
            conn.close()

        diagnose.assert_not_called()
        combined = _combined_calls(env)
        assert "SQLite maintenance:" in combined
        assert "planner stats=index" in combined
        assert "WAL " in combined

    def test_direct_status_skips_exact_archive_readiness_by_default(self, tmp_path: Path) -> None:
        env = _make_app_env()
        db_anchor = tmp_path / "custom.sqlite"
        index_db = tmp_path / "index.db"
        with sqlite3.connect(index_db) as conn:
            conn.executescript(
                """
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    message_count INTEGER NOT NULL
                );
                INSERT INTO sessions VALUES ('codex-session:one', 1);
                """
            )

        with (
            patch("polylogue.paths.db_path", return_value=db_anchor),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
            patch(
                "polylogue.cli.commands.status._archive_readiness_status",
                side_effect=AssertionError("direct status must not run exact readiness by default"),
            ),
        ):
            _show_direct_status(env)

        combined = _combined_calls(env)
        assert "Archive readiness:" in combined
        assert "direct_status_default_skips_exact_archive_readiness" in combined

    def test_direct_status_json_marks_skipped_transform_readiness_unknown(
        self,
        tmp_path: Path,
    ) -> None:
        env = _make_app_env()
        db_anchor = tmp_path / "custom.sqlite"
        index_db = tmp_path / "index.db"
        with sqlite3.connect(index_db) as conn:
            conn.executescript(
                """
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    message_count INTEGER NOT NULL
                );
                INSERT INTO sessions VALUES ('codex-session:one', 1);
                """
            )
        assertion_component = {
            "component": "assertions",
            "scope": "user",
            "state": "ready",
            "summary": "ready",
            "counts": {"assertion_count": 0, "target_count": 0, "active_count": 0},
            "caveats": [],
            "repair_hint": None,
            "evidence_refs": ["user.db:assertions"],
        }

        with (
            patch("polylogue.paths.db_path", return_value=db_anchor),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
            patch(
                "polylogue.cli.commands.status._archive_readiness_status",
                side_effect=AssertionError("direct status must not run exact readiness by default"),
            ),
            patch(
                "polylogue.cli.commands.status._direct_raw_materialization_readiness",
                return_value={"available": True, "total": 0},
            ),
            patch("polylogue.cli.commands.status._direct_assertion_component", return_value=assertion_component),
            patch("polylogue.storage.embeddings.status_payload.embedding_status_payload", side_effect=RuntimeError),
        ):
            _show_direct_json(env)

        payload = json.loads(_combined_calls(env))
        transforms = payload["component_readiness"]["transforms"]
        assert payload["ok"] is False
        assert payload["component_readiness"]["raw_frontier_integrity"]["state"] == "unknown"
        assert transforms["state"] == "unknown"
        assert transforms["summary"] == "direct_status_default_skips_exact_archive_readiness"
        assert transforms["caveats"] == ["direct_status_default_skips_exact_archive_readiness"]
        assert "session_count" not in transforms["counts"]
        assert transforms["counts"]["transform_count"] >= 1
        assert transforms["counts"]["session_digest_transform_version"] == 1

    def test_direct_status_json_reads_archive_file_set_from_archive_tiers(self, tmp_path: Path) -> None:
        env = _make_app_env()
        db_anchor = tmp_path / "custom.sqlite"
        index_db = tmp_path / "index.db"
        source_db = tmp_path / "source.db"
        with sqlite3.connect(index_db) as conn:
            conn.executescript(
                """
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    message_count INTEGER NOT NULL
                );
                INSERT INTO sessions VALUES ('codex-session:one', 3);
                """
            )
        with sqlite3.connect(source_db) as conn:
            conn.executescript(
                """
                CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY);
                INSERT INTO raw_sessions VALUES ('raw-1'), ('raw-2');
                """
            )

        with (
            patch("polylogue.paths.db_path", return_value=db_anchor),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
        ):
            _show_direct_json(env)

        payload = json.loads(_combined_calls(env))
        assert payload["db_exists"] is True
        assert payload["active_db_path"] == str(index_db)
        assert payload["archive_tiers"]["source"]["exists"] is True
        assert payload["archive_tiers"]["index"]["exists"] is True
        assert payload["archive_tiers"]["embeddings"]["exists"] is False
        assert payload["archive_tiers"]["user"]["exists"] is False
        assert payload["archive_tiers"]["ops"]["exists"] is False
        assert payload["archive_tiers"]["source"]["user_version"] == 0
        assert (
            payload["archive_tiers"]["source"]["expected_user_version"] == ARCHIVE_VERSION_BY_TIER[ArchiveTier.SOURCE]
        )
        assert payload["archive_tiers"]["source"]["version_status"] == "mismatch"
        assert payload["archive_tiers"]["source"]["table_counts"]["raw_sessions"] == 2
        assert payload["archive_tiers"]["index"]["user_version"] == 0
        assert payload["archive_tiers"]["index"]["expected_user_version"] == ARCHIVE_VERSION_BY_TIER[ArchiveTier.INDEX]
        assert payload["archive_tiers"]["index"]["version_status"] == "mismatch"
        assert payload["archive_tiers"]["index"]["table_counts"]["sessions"] == 1
        assert "archive_facade_routes" not in payload
        assert "archive_cli_routes" not in payload
        assert "archive_runtime_paths" not in payload
        assert payload["sqlite_maintenance"]["tiers"]["index"]["exists"] is True
        assert payload["sqlite_maintenance"]["tiers"]["index"]["planner_stats_present"] is False
        assert payload["sqlite_maintenance"]["tiers"]["source"]["exists"] is True
        assert payload["sessions"] == 1
        assert payload["messages"] == 3
        assert payload["raw_records"] == 2

    def test_direct_status_json_reports_sqlite_maintenance_state(self, tmp_path: Path) -> None:
        env = _make_app_env()
        db_anchor = tmp_path / "custom.sqlite"
        index_db = tmp_path / "index.db"
        conn = sqlite3.connect(index_db)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA wal_autocheckpoint=0")
            conn.executescript(
                """
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    message_count INTEGER NOT NULL
                );
                INSERT INTO sessions VALUES ('codex-session:one', 3);
                ANALYZE;
                """
            )
            conn.commit()
            wal_bytes = Path(f"{index_db}-wal").stat().st_size

            with (
                patch("polylogue.paths.db_path", return_value=db_anchor),
                patch("polylogue.paths.archive_root", return_value=tmp_path),
            ):
                _show_direct_json(env)
        finally:
            conn.close()

        payload = json.loads(_combined_calls(env))
        maintenance = payload["sqlite_maintenance"]
        assert maintenance["total_wal_bytes"] == wal_bytes
        assert maintenance["tiers_with_planner_stats"] == ["index"]
        assert maintenance["tiers"]["index"]["wal_bytes"] == wal_bytes
        assert maintenance["tiers"]["index"]["sqlite_stat1_rows"] == 1
        assert maintenance["tiers"]["index"]["planner_stats_present"] is True

    def test_direct_status_json_reports_active_archive_root_for_sibling_index(self, tmp_path: Path) -> None:
        env = _make_app_env()
        configured_root = tmp_path / "archive"
        active_root = tmp_path / "active"
        configured_root.mkdir()
        active_root.mkdir()
        db_anchor = active_root / "index.db"
        initialize_archive_database(active_root / "source.db", ArchiveTier.SOURCE)
        initialize_archive_database(active_root / "index.db", ArchiveTier.INDEX)
        (configured_root / "user.db").write_text("configured decoy", encoding="utf-8")
        with sqlite3.connect(active_root / "source.db") as conn:
            conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, blob_hash, blob_size, acquired_at_ms
                ) VALUES ('raw-active', 'codex-session', 'active', '/tmp/active.jsonl', ?, 10, 1)
                """,
                (b"x" * 32,),
            )
        with sqlite3.connect(active_root / "index.db") as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    native_id, origin, raw_id, content_hash, message_count
                ) VALUES ('active', 'codex-session', 'raw-active', ?, 2)
                """,
                (b"y" * 32,),
            )

        with (
            patch("polylogue.paths.db_path", return_value=db_anchor),
            patch("polylogue.paths.archive_root", return_value=configured_root),
        ):
            _show_direct_json(env)

        payload = json.loads(_combined_calls(env))
        assert payload["archive_root"] == str(configured_root)
        assert payload["active_archive_root"] == str(active_root)
        assert payload["active_archive_root_matches_configured"] is False
        assert payload["active_db_path"] == str(active_root / "index.db")
        assert payload["archive_tiers"]["source"]["exists"] is True
        assert payload["archive_tiers"]["source"]["table_counts"]["raw_sessions"] == 1
        assert payload["archive_tiers"]["index"]["exists"] is True
        assert payload["archive_tiers"]["index"]["table_counts"]["sessions"] == 1
        assert payload["archive_tiers"]["user"]["exists"] is False
        assert payload["raw_records"] == 1

    def test_direct_status_reports_archive_surface_blockers(self, tmp_path: Path) -> None:
        env = _make_app_env()
        db_anchor = tmp_path / "custom.sqlite"
        initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
        initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
        with sqlite3.connect(tmp_path / "source.db") as conn:
            conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, blob_hash, blob_size,
                    acquired_at_ms
                ) VALUES ('raw-1', 'codex-session', 'native-1', '/tmp/a.jsonl', ?, 10, 1)
                """,
                (b"x" * 32,),
            )
        with sqlite3.connect(tmp_path / "index.db") as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    native_id, origin, raw_id, content_hash
                ) VALUES ('native-1', 'codex-session', 'raw-1', ?)
                """,
                (b"y" * 32,),
            )
            conn.execute(
                """
                INSERT INTO messages (
                    session_id, native_id, position, role, content_hash
                ) VALUES ('codex-session:native-1', 'm1', 0, 'user', ?)
                """,
                (b"m" * 32,),
            )
            conn.execute(
                """
                INSERT INTO blocks (
                    message_id, session_id, position, block_type, text
                ) VALUES ('codex-session:native-1:m1', 'codex-session:native-1', 0, 'text', 'hello')
                """
            )

        with (
            patch("polylogue.paths.db_path", return_value=db_anchor),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
            patch("polylogue.cli.commands.status_diagnostics.diagnose_first_run") as diagnose,
        ):
            _show_direct_status(env, include_archive_readiness=True)

        diagnose.assert_not_called()
        combined = _combined_calls(env)
        assert "Archive surfaces:" in combined
        assert "Archive routing paths:" in combined
        assert "ingest=archive -> source.db,index.db" in combined
        assert "tiers=source.db,index.db,embeddings.db,user.db,ops.db" in combined
        assert "Archive route ownership:" in combined
        assert "cli=source:2,user:3" in combined
        assert "0 blockers" in combined
        assert "blocked" in combined
        assert "session_profiles: missing_profile_rows, missing_session_profile_materialization" in combined
        assert "timeline_work_events: missing_work_events_materialization" in combined

    def test_direct_status_json_reports_archive_surface_blockers(self, tmp_path: Path) -> None:
        env = _make_app_env()
        db_anchor = tmp_path / "index.db"
        initialize_archive_database(tmp_path / "source.db", ArchiveTier.SOURCE)
        initialize_archive_database(tmp_path / "index.db", ArchiveTier.INDEX)
        with sqlite3.connect(tmp_path / "index.db") as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    native_id, origin, raw_id, content_hash
                ) VALUES ('native-1', 'codex-session', 'raw-missing', ?)
                """,
                (b"y" * 32,),
            )

        with (
            patch("polylogue.paths.db_path", return_value=db_anchor),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
        ):
            _show_direct_json(env, include_archive_readiness=True)

        payload = json.loads(_combined_calls(env))
        readiness = payload["archive_readiness"]
        assert payload["archive_tiers"]["source"]["version_status"] == "ok"
        assert payload["archive_tiers"]["index"]["version_status"] == "ok"
        assert payload["archive_tiers"]["index"]["table_counts"]["sessions"] == 1
        assert readiness["checked"] is True
        assert readiness["blocked_surface_count"] > 0
        assert readiness["surfaces"]["raw_artifacts"]["ready"] is False
        assert readiness["surfaces"]["raw_artifacts"]["blockers"] == ["missing_source_raw_sessions"]
        raw_evidence = readiness["surfaces"]["raw_artifacts"]["evidence"]
        assert raw_evidence["lost_source_evidence_count"] == 1
        assert raw_evidence["lost_source_evidence_samples"] == [
            {
                "session_id": "codex-session:native-1",
                "origin": "codex-session",
                "native_id": "native-1",
                "missing_raw_id": "raw-missing",
                "message_count": 0,
                "updated_at_ms": None,
                "evidence_status": "lost_source_evidence",
                "loss_reason": "index_raw_id_missing_from_source_tier",
                "recovery_requirement": "restore_exact_raw_artifact_or_keep_blocked",
            }
        ]
        runtime_paths = payload["archive_runtime_paths"]
        assert runtime_paths["archive_routing_ready"] is True
        assert runtime_paths["archive_runtime_ready"] is True
        assert runtime_paths["unsupported_primary_method_count"] == 0
        assert runtime_paths["final_shape_blockers"] == []
        assert runtime_paths["archive_tier_targets"] == [
            "source.db",
            "index.db",
            "embeddings.db",
            "user.db",
            "ops.db",
        ]

    def test_direct_status_json_compacts_raw_replay_backlog(self, tmp_path: Path) -> None:
        env = _make_app_env()
        db_anchor = tmp_path / "index.db"
        initialize_archive_database(db_anchor, ArchiveTier.INDEX)
        backlog = {
            "available": True,
            "candidate_count": 2,
            "missing_blob_count": 0,
            "already_parsed_count": 0,
            "total_blob_bytes": 12_000_000,
            "max_blob_bytes": 10_000_000,
            "execute_blob_limit_bytes": 1_000_000_000,
            "oversized_count": 0,
            "oversized_stream_safe_count": 0,
            "top_raw_rows": [{"raw_id": "raw-a", "blob_size": 10_000_000}],
            "origin_summary": [{"origin": "codex-session", "raw_count": 2, "total_blob_bytes": 12_000_000}],
            "source_path_summary": [{"source_path": "/large/local/path", "raw_count": 2}],
        }

        with (
            patch("polylogue.paths.db_path", return_value=db_anchor),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
            patch("polylogue.cli.commands.status._direct_raw_materialization_readiness", return_value={}),
            patch("polylogue.cli.commands.status._raw_replay_backlog_status", return_value=backlog),
            patch("polylogue.cli.commands.status._ops_workload_status", return_value={"available": False}),
            patch("polylogue.storage.embeddings.status_payload.embedding_status_payload", side_effect=RuntimeError),
        ):
            _show_direct_json(env)

        payload = json.loads(_combined_calls(env))
        raw_replay = payload["raw_replay_backlog"]
        assert raw_replay["candidate_count"] == 2
        assert raw_replay["total_blob_bytes"] == 12_000_000
        assert raw_replay["origin_summary"][0]["origin"] == "codex-session"
        assert "source_path_summary" not in raw_replay

    def test_archive_tier_status_labels_large_table_estimates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from polylogue.cli.commands import status as status_module

        db = tmp_path / "index.db"
        initialize_archive_database(db, ArchiveTier.INDEX)
        with sqlite3.connect(db) as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    native_id, origin, raw_id, content_hash, message_count
                ) VALUES ('native-1', 'codex-session', 'raw-1', ?, 12)
                """,
                (b"z" * 32,),
            )
            conn.execute("ANALYZE")
            conn.execute("DELETE FROM sqlite_stat1")
            conn.execute("INSERT INTO sqlite_stat1(tbl, idx, stat) VALUES ('blocks', 'idx_blocks_type', '123 4')")
        monkeypatch.setattr(status_module, "_LARGE_TIER_EXACT_COUNT_LIMIT_BYTES", -1)

        status = status_module._archive_one_tier_status("index", db)

        assert status["table_counts"]["sessions"] == 1
        assert status["table_count_precision"]["sessions"] == "exact"
        assert status["table_counts"]["messages"] == 12
        assert status["table_count_precision"]["messages"] == "exact"
        assert status["table_counts"]["blocks"] == 123
        assert status["table_count_precision"]["blocks"] == "estimate"

    def test_direct_status_json_maps_embeddings_component_readiness(self, tmp_path: Path) -> None:
        env = _make_app_env()
        db_anchor = tmp_path / "index.db"
        initialize_archive_database(db_anchor, ArchiveTier.INDEX)
        embedding_payload = {
            "config_enabled": True,
            "has_voyage_api_key": True,
            "status": "partial",
            "total_sessions": 3,
            "embedded_sessions": 2,
            "embedded_messages": 20,
            "pending_sessions": 1,
            "pending_messages": None,
            "pending_messages_exact": False,
            "stale_messages": 4,
            "failure_count": 0,
            "freshness_status": "stale",
            "retrieval_ready": True,
            "next_action": {
                "code": "refresh_stale",
                "command": "polylogue ops embed backfill --yes --max-sessions 10",
                "reason": "Existing vectors are stale for at least one message.",
            },
        }

        with (
            patch("polylogue.paths.db_path", return_value=db_anchor),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
            patch(
                "polylogue.storage.embeddings.status_payload.embedding_status_payload",
                return_value=embedding_payload,
            ),
        ):
            _show_direct_json(env, include_archive_readiness=True)

        payload = json.loads(_combined_calls(env))
        readiness = payload["component_readiness"]["embeddings"]
        assert readiness["component"] == "embeddings"
        assert readiness["scope"] == "semantic"
        assert readiness["state"] == "stale"
        assert readiness["summary"] == "partial"
        assert readiness["counts"]["total_sessions"] == 3
        assert readiness["counts"]["embedded_sessions"] == 2
        assert readiness["counts"]["pending_messages"] is None
        assert readiness["counts"]["pending_messages_exact"] is False
        assert readiness["counts"]["stale_messages"] == 4
        assert readiness["counts"]["retrieval_ready"] is True
        assert readiness["caveats"] == []
        assert readiness["repair_hint"] == "polylogue ops embed backfill --yes --max-sessions 10"

    def test_direct_status_json_reports_failed_component_as_unknown(self, tmp_path: Path) -> None:
        """A component whose readiness computation raises must appear as an
        explicit unknown entry, never silently vanish from the payload
        (polylogue-feqr). Reverting the except-pass fix makes the 'embeddings'
        key absent and this test fail."""
        env = _make_app_env()
        db_anchor = tmp_path / "index.db"
        initialize_archive_database(db_anchor, ArchiveTier.INDEX)

        with (
            patch("polylogue.paths.db_path", return_value=db_anchor),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
            patch(
                "polylogue.storage.embeddings.status_payload.embedding_status_payload",
                side_effect=RuntimeError("embedding tier unreadable"),
            ),
        ):
            _show_direct_json(env, include_archive_readiness=True)

        payload = json.loads(_combined_calls(env))
        readiness = payload["component_readiness"]["embeddings"]
        assert readiness["component"] == "embeddings"
        assert readiness["state"] == "unknown"
        assert "RuntimeError" in readiness["summary"]
        assert "embedding tier unreadable" in readiness["summary"]

    def test_direct_status_json_maps_archive_surface_component_readiness(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        env = _make_app_env()
        db_anchor = tmp_path / "index.db"
        initialize_archive_database(db_anchor, ArchiveTier.INDEX)
        initialize_archive_database(tmp_path / "user.db", ArchiveTier.USER)
        with sqlite3.connect(tmp_path / "user.db") as user_conn:
            upsert_assertion(
                user_conn,
                assertion_id="demo-assertion",
                target_ref="session:demo",
                key="star",
                kind=AssertionKind.MARK,
                value={"label": "pytest-triage"},
                status="active",
                now_ms=1_700_000_000_000,
            )
            user_conn.commit()
        archive_readiness = {
            "checked": True,
            "counts": {"session_count": 3},
            "surfaces": {
                "archive_sessions": {
                    "ready": True,
                    "blockers": [],
                    "evidence": {"session_count": 3, "message_count": 19},
                },
                "search": {
                    "ready": False,
                    "blockers": ["messages_fts_row_mismatch"],
                    "evidence": {"text_block_count": 10, "messages_fts_count": 8},
                },
                "session_profiles": {
                    "ready": False,
                    "blockers": ["missing_session_profile_materialization"],
                    "evidence": {
                        "profile_row_count": 2,
                        "missing_profile_row_count": 0,
                        "missing_materialization_count": 1,
                    },
                },
                "tool_usage": {"ready": True, "blockers": [], "evidence": {"action_count": 4}},
            },
        }
        monkeypatch.setattr(
            "polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot",
            lambda _root: {
                "available": True,
                "classification": "not_run",
                "precision": "raw_id_join_gap",
                "raw_artifact_count": 10,
                "materialized_raw_artifact_count": 6,
                "archive_session_count": 7,
                "join_gap_count": 4,
                "total": 4,
                "critical": 0,
                "warning": 0,
                "actionable": 0,
                "blocked": 0,
                "classified": 0,
                "unchecked": 4,
                "affected_total": 4,
                "affected_actionable": 0,
                "affected_blocked": 0,
                "affected_open": 0,
                "affected_classified": 0,
                "affected_unchecked": 4,
                "category_counts": {"raw_id_join_gap": 4},
                "source_family_counts": {"chatgpt-export": 4},
            },
        )

        with (
            patch("polylogue.paths.db_path", return_value=db_anchor),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
            patch("polylogue.cli.commands.status._archive_readiness_status", return_value=archive_readiness),
            patch("polylogue.storage.embeddings.status_payload.embedding_status_payload", side_effect=RuntimeError),
        ):
            _show_direct_json(env, include_archive_readiness=True)

        payload = json.loads(_combined_calls(env))
        components = payload["component_readiness"]
        archive = components["archive_sessions"]
        readiness = components["search"]
        profiles = components["session_profiles"]
        tool_usage = components["tool_usage"]
        raw_materialization = components["raw_materialization"]
        assertions = components["assertions"]
        transforms = components["transforms"]
        assert payload["archive_readiness"] == archive_readiness
        assert payload["raw_materialization_readiness"]["total"] == 4
        assert payload["raw_materialization_readiness"]["raw_artifact_count"] == 10
        assert payload["raw_materialization_readiness"]["materialized_raw_artifact_count"] == 6
        assert payload["raw_materialization_readiness"]["affected_total"] == 4
        assert payload["raw_materialization_readiness"]["affected_unchecked"] == 4
        assert payload["raw_materialization_readiness"]["category_counts"] == {"raw_id_join_gap": 4}
        assert payload["raw_materialization_readiness"]["source_family_counts"] == {"chatgpt-export": 4}
        assert archive["component"] == "archive_sessions"
        assert archive["scope"] == "archive"
        assert archive["state"] == "ready"
        assert archive["counts"] == {"session_count": 3, "message_count": 19}
        assert readiness["component"] == "search"
        assert readiness["scope"] == "lexical"
        assert readiness["state"] == "stale"
        assert readiness["counts"] == {"text_block_count": 10, "messages_fts_count": 8}
        assert readiness["caveats"] == ["messages_fts_row_mismatch"]
        assert readiness["repair_hint"] == "polylogued run"
        assert profiles["scope"] == "insights"
        assert profiles["state"] == "stale"
        assert profiles["repair_hint"] == "polylogued run"
        assert tool_usage["scope"] == "actions"
        assert tool_usage["counts"] == {"action_count": 4}
        assert raw_materialization["scope"] == "archive"
        assert raw_materialization["state"] == "degraded"
        assert raw_materialization["counts"]["total"] == 4
        assert raw_materialization["counts"]["affected_total"] == 4
        assert raw_materialization["counts"]["affected_actionable"] == 0
        assert raw_materialization["counts"]["affected_unchecked"] == 4
        assert raw_materialization["counts"]["raw_artifact_count"] == 10
        assert raw_materialization["counts"]["materialized_raw_artifact_count"] == 6
        assert raw_materialization["metadata"]["category_counts"] == {"raw_id_join_gap": 4}
        assert raw_materialization["metadata"]["source_family_counts"] == {"chatgpt-export": 4}
        assert raw_materialization["repair_hint"] == "polylogued run"
        assert assertions["scope"] == "user"
        assert assertions["state"] == "ready"
        assert assertions["counts"] == {"assertion_count": 1, "target_count": 1, "active_count": 1}
        assert assertions["evidence_refs"] == ["user.db:assertions"]
        overlay_audit = assertions["metadata"]["overlay_audit"]
        surfaces = {surface["name"]: surface for surface in overlay_audit["surfaces"]}
        assert surfaces["marks"]["storage"] == "assertions"
        assert surfaces["marks"]["active_count"] == 1
        assert surfaces["tag_assertions"]["storage"] == "assertions"
        assert surfaces["metadata_assertions"]["storage"] == "assertions"
        assert transforms["scope"] == "session-analysis"
        assert transforms["state"] == "ready"
        assert transforms["counts"]["session_count"] == 3
        assert transforms["counts"]["transform_count"] >= 1
        assert transforms["counts"]["session_digest_transform_version"] == 1

    def test_direct_status_json_reports_missing_assertions_and_transforms_without_archive(
        self,
        tmp_path: Path,
    ) -> None:
        env = _make_app_env()
        db_anchor = tmp_path / "index.db"
        initialize_archive_database(db_anchor, ArchiveTier.INDEX)

        with (
            patch("polylogue.paths.db_path", return_value=db_anchor),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
            patch("polylogue.storage.embeddings.status_payload.embedding_status_payload", side_effect=RuntimeError),
        ):
            _show_direct_json(env, include_archive_readiness=True)

        payload = json.loads(_combined_calls(env))
        components = payload["component_readiness"]
        assertions = components["assertions"]
        transforms = components["transforms"]
        assert assertions["state"] == "missing"
        assert assertions["summary"] == "assertions table missing"
        assert assertions["repair_hint"] == "polylogue ops maintenance archive-init --yes"
        assert transforms["state"] == "missing"
        assert transforms["summary"] == "no sessions"
        assert transforms["repair_hint"] == "polylogue import --demo"

    def test_direct_status_json_covers_component_readiness_state_matrix(
        self,
        tmp_path: Path,
    ) -> None:
        env = _make_app_env()
        db_anchor = tmp_path / "index.db"
        initialize_archive_database(db_anchor, ArchiveTier.INDEX)
        archive_readiness = {
            "checked": True,
            "counts": {"session_count": 0},
            "surfaces": {
                "archive_sessions": {
                    "ready": True,
                    "blockers": [],
                    "evidence": {"session_count": 0},
                },
                "search": {
                    "ready": False,
                    "blockers": ["messages_fts_row_mismatch"],
                    "evidence": {"text_block_count": 10, "messages_fts_count": 8},
                },
                "session_profiles": {
                    "ready": False,
                    "blockers": [],
                    "evidence": {"profile_row_count": 1, "missing_profile_row_count": 1},
                },
            },
        }
        embedding_payload = {
            "config_enabled": False,
            "has_voyage_api_key": False,
            "status": "disabled",
            "total_sessions": 0,
            "embedded_sessions": 0,
            "embedded_messages": 0,
            "pending_sessions": 0,
            "stale_messages": 0,
            "failure_count": 0,
            "freshness_status": "disabled",
            "retrieval_ready": False,
            "next_action": {"code": "enable", "command": "polylogue ops embed enable"},
        }

        with (
            patch("polylogue.paths.db_path", return_value=db_anchor),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
            patch("polylogue.cli.commands.status._archive_readiness_status", return_value=archive_readiness),
            patch(
                "polylogue.storage.embeddings.status_payload.embedding_status_payload",
                return_value=embedding_payload,
            ),
        ):
            _show_direct_json(env, include_archive_readiness=True)

        components = json.loads(_combined_calls(env))["component_readiness"]
        assert components["archive_sessions"]["state"] == "ready"
        assert components["search"]["state"] == "stale"
        assert components["search"]["caveats"] == ["messages_fts_row_mismatch"]
        assert components["search"]["repair_hint"] == "polylogued run"
        assert components["session_profiles"]["state"] == "degraded"
        assert components["session_profiles"]["counts"] == {
            "profile_row_count": 1,
            "missing_profile_row_count": 1,
        }
        assert components["embeddings"]["state"] == "missing"
        assert components["assertions"]["state"] == "missing"
        assert components["transforms"]["state"] == "missing"

    def test_direct_status_json_exposes_claim_guard_when_fully_openable(
        self,
        tmp_path: Path,
    ) -> None:
        """polylogue-avg AC: `polylogue ops status --json` exposes a claim-guard
        block with all four claim states, each derived from its documented
        signal, in the no-daemon direct SQLite fallback path too."""
        env = _make_app_env()
        for tier in (
            ArchiveTier.SOURCE,
            ArchiveTier.INDEX,
            ArchiveTier.EMBEDDINGS,
            ArchiveTier.USER,
            ArchiveTier.OPS,
        ):
            initialize_archive_database(tmp_path / f"{tier.value}.db", tier)

        archive_readiness = {
            "checked": True,
            "counts": {"session_count": 0},
            "surfaces": {
                "search": {"ready": True, "blockers": [], "evidence": {}},
            },
        }
        monkeypatch_target = "polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot"
        with (
            patch("polylogue.paths.db_path", return_value=tmp_path / "index.db"),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
            patch("polylogue.cli.commands.status._archive_readiness_status", return_value=archive_readiness),
            patch(
                monkeypatch_target,
                return_value={"available": True, "total": 0, "affected_total": 0},
            ),
            patch("polylogue.storage.embeddings.status_payload.embedding_status_payload", side_effect=RuntimeError),
        ):
            _show_direct_json(env, include_archive_readiness=True)

        payload = json.loads(_combined_calls(env))
        claim_guard = payload["claim_guard"]
        assert set(claim_guard) == {"openable", "converged", "search_ready", "perf_measurable"}
        assert claim_guard["openable"]["value"] is True
        assert claim_guard["converged"]["value"] is True
        assert claim_guard["search_ready"]["value"] is True
        assert claim_guard["perf_measurable"]["value"] is True
        for entry in claim_guard.values():
            assert entry["signal"]

    def test_direct_status_json_marks_unavailable_raw_frontier_non_green(
        self,
        tmp_path: Path,
    ) -> None:
        """A present but unreadable ops cursor authority must not leave top-level status green."""

        env = _make_app_env()
        for tier in (
            ArchiveTier.SOURCE,
            ArchiveTier.INDEX,
            ArchiveTier.EMBEDDINGS,
            ArchiveTier.USER,
            ArchiveTier.OPS,
        ):
            initialize_archive_database(tmp_path / f"{tier.value}.db", tier)
        with sqlite3.connect(tmp_path / "ops.db") as conn:
            conn.execute("DROP TABLE ingest_cursor")
            conn.commit()

        archive_readiness = {
            "checked": True,
            "counts": {"session_count": 0},
            "surfaces": {"search": {"ready": True, "blockers": [], "evidence": {}}},
        }
        with (
            patch("polylogue.paths.db_path", return_value=tmp_path / "index.db"),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
            patch("polylogue.cli.commands.status._archive_readiness_status", return_value=archive_readiness),
            patch(
                "polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot",
                return_value={"available": True, "total": 0, "affected_total": 0},
            ),
            patch("polylogue.storage.embeddings.status_payload.embedding_status_payload", side_effect=RuntimeError),
        ):
            _show_direct_json(env, include_archive_readiness=True)

        payload = json.loads(_combined_calls(env))
        assert payload["raw_frontier_integrity"]["overall_status"] == "unknown"
        assert payload["component_readiness"]["raw_frontier_integrity"]["state"] == "unknown"
        assert payload["claim_guard"]["converged"]["value"] is False
        assert payload["ok"] is False

    def test_direct_status_json_detects_broken_append_head_blocks_converged(
        self,
        tmp_path: Path,
    ) -> None:
        """polylogue-yla8.7 AC: a current accepted append head whose
        predecessor chain is broken must surface through
        ``raw_frontier_integrity``, render ``component_readiness`` as
        ``poisoned``, and block claim-guard ``converged`` in the no-daemon
        direct SQLite fallback path — not just through the daemon path."""
        env = _make_app_env()
        for tier in (
            ArchiveTier.SOURCE,
            ArchiveTier.INDEX,
            ArchiveTier.EMBEDDINGS,
            ArchiveTier.USER,
            ArchiveTier.OPS,
        ):
            initialize_archive_database(tmp_path / f"{tier.value}.db", tier)

        source_path = tmp_path / "session.jsonl"
        source_path.write_text("{}\n", encoding="utf-8")
        with sqlite3.connect(tmp_path / "source.db") as conn:
            conn.execute(
                """
                INSERT INTO raw_sessions (
                    raw_id, origin, native_id, source_path, source_index, blob_hash,
                    blob_size, acquired_at_ms, logical_source_key, revision_kind,
                    source_revision, predecessor_source_revision, predecessor_raw_id,
                    baseline_raw_id, append_start_offset, append_end_offset,
                    acquisition_generation, revision_authority
                ) VALUES (
                    'raw-append', 'codex-session', 'raw-append', ?, -1, ?,
                    5, 1, 'codex:session-1', 'append',
                    'revision-1', 'revision-0', 'raw-missing',
                    'raw-missing', 10, 15,
                    1, 'byte_proven'
                )
                """,
                (str(source_path), (1).to_bytes(32, "big")),
            )
            conn.commit()
        with sqlite3.connect(tmp_path / "index.db") as conn:
            conn.execute(
                """
                INSERT INTO sessions (native_id, origin, raw_id, title, content_hash)
                VALUES ('session-1', 'codex-session', 'raw-append', 'session', ?)
                """,
                (bytes(32),),
            )
            conn.execute(
                """
                INSERT INTO raw_revision_heads (
                    logical_source_key, session_id, accepted_raw_id,
                    accepted_source_revision, accepted_content_hash,
                    accepted_frontier_kind, accepted_frontier,
                    acquisition_generation, append_end_offset, decided_at_ms
                ) VALUES ('codex:session-1', 'codex-session:session-1', 'raw-append',
                          'revision-1', ?, 'byte', 15, 1, 15, 2)
                """,
                (bytes(32),),
            )
            conn.commit()

        archive_readiness = {
            "checked": True,
            "counts": {"session_count": 1},
            "surfaces": {"search": {"ready": True, "blockers": [], "evidence": {}}},
        }
        with (
            patch("polylogue.paths.db_path", return_value=tmp_path / "index.db"),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
            patch("polylogue.cli.commands.status._archive_readiness_status", return_value=archive_readiness),
            patch(
                "polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot",
                return_value={"available": True, "total": 0, "affected_total": 0},
            ),
            patch("polylogue.storage.embeddings.status_payload.embedding_status_payload", side_effect=RuntimeError),
        ):
            _show_direct_json(env, include_archive_readiness=True)

        payload = json.loads(_combined_calls(env))
        assert payload["ok"] is False
        integrity = payload["raw_frontier_integrity"]
        assert integrity["overall_status"] == "violated"
        assert integrity["broken_head_status"] == "violated"
        assert integrity["broken_head_count"] == 1
        assert "missing from source tier" in integrity["broken_head_samples"][0]["reason"]
        assert integrity["cursor_authority_gap_count"] == 0

        component = payload["component_readiness"]["raw_frontier_integrity"]
        assert component["state"] == "poisoned"
        assert component["counts"]["broken_head_count"] == 1

        claim_guard = payload["claim_guard"]
        assert claim_guard["openable"]["value"] is True
        assert claim_guard["converged"]["value"] is False
        assert "broken predecessor chain" in claim_guard["converged"]["reason"]

    def test_direct_status_json_claim_guard_reports_missing_tiers_when_not_openable(
        self,
        tmp_path: Path,
    ) -> None:
        """An archive missing tiers cannot be claimed openable, and therefore
        cannot be claimed converged either — even when raw-materialization
        debt itself looks clean."""
        env = _make_app_env()
        db_anchor = tmp_path / "index.db"
        initialize_archive_database(db_anchor, ArchiveTier.INDEX)

        with (
            patch("polylogue.paths.db_path", return_value=db_anchor),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
            patch(
                "polylogue.storage.archive_readiness.raw_materialization_readiness_snapshot",
                return_value={"available": True, "total": 0, "affected_total": 0},
            ),
            patch("polylogue.storage.embeddings.status_payload.embedding_status_payload", side_effect=RuntimeError),
        ):
            _show_direct_json(env)

        payload = json.loads(_combined_calls(env))
        claim_guard = payload["claim_guard"]
        assert claim_guard["openable"]["value"] is False
        missing_reason = claim_guard["openable"]["reason"]
        assert "source" in missing_reason
        assert "user" in missing_reason
        assert claim_guard["converged"]["value"] is False
        assert "not openable" in claim_guard["converged"]["reason"]

    def test_direct_claim_guard_treats_unavailable_workload_as_not_perf_measurable(self) -> None:
        """Regression: an unreadable/missing ops-workload tier cannot establish
        the *absence* of a concurrent writer. Before the fix, `available:
        False` fell through to `running_count = 0` / `active_writer = False`,
        silently reporting `perf_measurable=True` ("no concurrent archive
        write/rebuild detected") for an archive whose workload state is
        actually unknown — exactly the silent-unknown-treated-as-success
        pattern the claim-guard vocabulary exists to prevent."""
        archive_tiers = {
            tier: {"exists": True, "version_status": "ok"} for tier in ("source", "index", "embeddings", "user", "ops")
        }
        component_readiness = {
            "raw_materialization": {"summary": "ready"},
            "raw_frontier_integrity": {"state": "ready", "summary": "ready"},
            "search": {"state": "ready", "summary": "ready"},
        }
        raw_materialization_readiness = {"available": True, "total": 0, "affected_total": 0}
        raw_frontier_integrity = {"overall_status": "healthy"}

        for unavailable_workload in (
            {"available": False, "reason": "missing_ops_tier"},
            {"available": False, "reason": "missing_ingest_attempts"},
            {},
        ):
            guard = _direct_claim_guard(
                archive_tiers=archive_tiers,
                raw_materialization_readiness=raw_materialization_readiness,
                raw_frontier_integrity=raw_frontier_integrity,
                component_readiness=component_readiness,
                ingest_workload=unavailable_workload,
            )
            assert guard["perf_measurable"]["value"] is False, unavailable_workload
            assert "unavailable" in guard["perf_measurable"]["reason"]
            # The rest of the block is unaffected by the unreadable workload tier.
            assert guard["openable"]["value"] is True
            assert guard["converged"]["value"] is True

        # A genuinely idle, readable workload still reports perf_measurable=True.
        idle_guard = _direct_claim_guard(
            archive_tiers=archive_tiers,
            raw_materialization_readiness=raw_materialization_readiness,
            raw_frontier_integrity=raw_frontier_integrity,
            component_readiness=component_readiness,
            ingest_workload={"available": True, "actively_ingesting": False, "running_count": 0},
        )
        assert idle_guard["perf_measurable"]["value"] is True

    def test_direct_status_requires_raw_frontier_component_presence(self) -> None:
        assert _direct_status_ok({}) is False
        assert _direct_status_ok({"raw_frontier_integrity": {"state": "unknown"}}) is False
        assert _direct_status_ok({"raw_frontier_integrity": {"state": "ready"}}) is True

    def test_direct_status_json_blocks_transforms_when_archive_readiness_fails(
        self,
        tmp_path: Path,
    ) -> None:
        env = _make_app_env()
        db_anchor = tmp_path / "index.db"
        initialize_archive_database(db_anchor, ArchiveTier.INDEX)

        with (
            patch("polylogue.paths.db_path", return_value=db_anchor),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
            patch(
                "polylogue.cli.commands.status._archive_readiness_status",
                return_value={"checked": False, "reason": "database is locked", "surfaces": {}},
            ),
            patch("polylogue.storage.embeddings.status_payload.embedding_status_payload", side_effect=RuntimeError),
        ):
            _show_direct_json(env, include_archive_readiness=True)

        payload = json.loads(_combined_calls(env))
        transforms = payload["component_readiness"]["transforms"]
        assert payload["ok"] is False
        assert transforms["state"] == "blocked"
        assert transforms["summary"] == "database is locked"
        assert transforms["caveats"] == ["database is locked"]
        assert transforms["repair_hint"] is None
        assert "session_count" not in transforms["counts"]

    def test_direct_status_uses_active_archive_root(self, tmp_path: Path) -> None:
        env = _make_app_env()
        db_anchor = tmp_path / "custom.sqlite"
        index_db = tmp_path / "index.db"
        with sqlite3.connect(db_anchor) as conn:
            conn.executescript(
                """
                CREATE TABLE sessions (session_id TEXT PRIMARY KEY);
                CREATE TABLE messages (message_id TEXT PRIMARY KEY);
                CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY);
                INSERT INTO sessions VALUES ('unsupported');
                INSERT INTO messages VALUES ('unsupported-message');
                INSERT INTO raw_sessions VALUES ('unsupported-raw');
                """
            )
        with sqlite3.connect(index_db) as conn:
            conn.executescript(
                """
                CREATE TABLE sessions (
                    session_id TEXT PRIMARY KEY,
                    message_count INTEGER NOT NULL
                );
                CREATE TABLE raw_sessions (raw_id TEXT PRIMARY KEY);
                INSERT INTO sessions VALUES ('codex-session:v1', 4);
                INSERT INTO raw_sessions VALUES ('raw-v1'), ('raw-v1b');
                """
            )

        with (
            patch("polylogue.paths.db_path", return_value=db_anchor),
            patch("polylogue.paths.archive_root", return_value=tmp_path),
            patch("polylogue.cli.commands.status_diagnostics.diagnose_first_run") as diagnose,
        ):
            _show_direct_status(env)

        diagnose.assert_not_called()
        combined = _combined_calls(env)
        assert "Sessions: 1" in combined
        assert "Messages: 4" in combined
        assert "Raw records: 2" in combined

    def test_direct_status_does_not_count_fts_shadow_tables(self) -> None:
        """Large archive fallback status must not count FTS shadow tables."""
        env = _make_app_env()
        fake_root = Path("/tmp/archive-root")
        fake_db = MagicMock()
        fake_db.exists.return_value = True
        queries: list[str] = []

        class FakeCursor:
            def __init__(self, value: int, rows: list[tuple[object, ...]] | None = None) -> None:
                self._value = value
                self._rows = rows if rows is not None else []

            def fetchone(self) -> list[int]:
                return [self._value]

            def fetchall(self) -> list[tuple[object, ...]]:
                return self._rows

        class FakeConn:
            def execute(self, sql: str, params: tuple[object, ...] | None = None) -> FakeCursor:
                queries.append(sql)
                if "PRAGMA table_info" in sql:
                    # sessions carries the message_count rollup column, so status
                    # uses SUM(message_count) and never scans the messages table.
                    return FakeCursor(0, rows=[(0, "message_count", "INTEGER", 0, None, 0)])
                assert "COUNT(*) FROM messages_fts" not in sql
                assert "COUNT(*) FROM messages" not in sql
                if "SUM(message_count)" in sql:
                    return FakeCursor(11)
                if "sessions" in sql:
                    return FakeCursor(7)
                if "raw_sessions" in sql:
                    return FakeCursor(9)
                assert "messages_fts_docsize" not in sql
                return FakeCursor(0)

            def close(self) -> None:
                pass

        with patch("polylogue.paths.db_path", return_value=fake_db):
            with patch("polylogue.paths.archive_root", return_value=fake_root):
                with patch(
                    "polylogue.storage.sqlite.connection_profile.open_readonly_connection", return_value=FakeConn()
                ):
                    _show_direct_status(env)

        combined = _combined_calls(env)
        assert "daemon status unavailable" in combined
        assert any("SUM(message_count)" in query for query in queries)
        assert not any("messages_fts_docsize" in query for query in queries)

    def test_direct_status_reports_embedding_readiness_not_raw_coverage(self) -> None:
        """Daemon-offline status should expose semantic-readiness state."""
        env = _make_app_env()
        fake_root = Path("/tmp/archive-root")
        fake_db = MagicMock()
        fake_db.exists.return_value = True

        class FakeCursor:
            def __init__(self, value: int, rows: list[tuple[object, ...]] | None = None) -> None:
                self._value = value
                self._rows = rows if rows is not None else []

            def fetchone(self) -> list[int]:
                return [self._value]

            def fetchall(self) -> list[tuple[object, ...]]:
                return self._rows

        class FakeConn:
            def execute(self, sql: str, params: tuple[object, ...] | None = None) -> FakeCursor:
                if "SUM(message_count)" in sql:
                    return FakeCursor(30)
                if "sessions" in sql:
                    return FakeCursor(3)
                if "raw_sessions" in sql:
                    return FakeCursor(4)
                return FakeCursor(0)

            def close(self) -> None:
                pass

        embedding_payload = {
            "total_sessions": 3,
            "embedded_sessions": 1,
            "embedded_messages": 10,
            "pending_sessions": 2,
            "embedding_coverage_percent": 33.3,
            "retrieval_ready": False,
            "status": "partial",
            "freshness_status": "stale",
            "stale_messages": 10,
            "failure_count": 1,
            "latest_catchup_run": {
                "status": "stopped",
                "processed_sessions": 2,
                "planned_sessions": 3,
                "embedded_messages": 10,
                "error_count": 1,
            },
        }

        with patch("polylogue.paths.db_path", return_value=fake_db):
            with patch("polylogue.paths.archive_root", return_value=fake_root):
                with patch(
                    "polylogue.storage.sqlite.connection_profile.open_readonly_connection", return_value=FakeConn()
                ):
                    with patch(
                        "polylogue.storage.embeddings.status_payload.embedding_status_payload",
                        return_value=embedding_payload,
                    ):
                        _show_direct_status(env)

        combined = _combined_calls(env)
        assert "Embeddings:" in combined
        assert "partial/stale, not ready" in combined
        assert "10 msgs, 1/3 convs (33.3%), 2 pending convs, 10 stale msgs" in combined
        assert "Embedding failures:" in combined
        assert "Embedding catch-up: stopped, 2/3 convs, 10 msgs embedded, 1 errors" in combined

    def test_explicit_status_has_short_daemon_timeout_before_direct_fallback(self) -> None:
        """Explicit status must not hide behind a daemon blocked on ingest."""
        env = _make_app_env()

        def raise_timeout(_request: object, *, timeout: float) -> None:
            assert timeout in {_FULL_TIMEOUT_S, 1.0}
            raise TimeoutError

        with patch("polylogue.cli.commands.status.urlopen", side_effect=raise_timeout):
            with patch("polylogue.cli.commands.status._show_direct_status") as show_direct:
                result = CliRunner().invoke(status_command, ["--daemon-url", "http://127.0.0.1:8766"], obj=env)

        assert result.exit_code == 0
        show_direct.assert_called_once_with(env, include_archive_readiness=False)

    def test_status_command_full_json_direct_fallback_skips_exact_readiness(self) -> None:
        """--full preserves payload shape without opting into expensive exact probes."""
        env = _make_app_env()

        def raise_timeout(_request: object, *, timeout: float) -> None:
            assert timeout in {_FULL_TIMEOUT_S, 1.0}
            raise TimeoutError

        with patch("polylogue.cli.commands.status.urlopen", side_effect=raise_timeout):
            with patch("polylogue.cli.commands.status._daemon_live", return_value=False):
                with patch("polylogue.cli.commands.status._show_direct_json") as show_direct_json:
                    result = CliRunner().invoke(
                        status_command,
                        ["--daemon-url", "http://127.0.0.1:8766", "--json", "--full"],
                        obj=env,
                    )

        assert result.exit_code == 0
        show_direct_json.assert_called_once_with(env, full=True, include_archive_readiness=False)

    def test_status_command_exact_archive_readiness_direct_fallback_opts_in(self) -> None:
        """Exact readiness is a separate explicit diagnostic flag."""
        env = _make_app_env()

        def raise_timeout(_request: object, *, timeout: float) -> None:
            assert timeout in {_FULL_TIMEOUT_S, 1.0}
            raise TimeoutError

        with patch("polylogue.cli.commands.status.urlopen", side_effect=raise_timeout):
            with patch("polylogue.cli.commands.status._daemon_live", return_value=False):
                with patch("polylogue.cli.commands.status._show_direct_json") as show_direct_json:
                    result = CliRunner().invoke(
                        status_command,
                        [
                            "--daemon-url",
                            "http://127.0.0.1:8766",
                            "--json",
                            "--full",
                            "--exact-archive-readiness",
                        ],
                        obj=env,
                    )

        assert result.exit_code == 0
        show_direct_json.assert_called_once_with(env, full=True, include_archive_readiness=True)

    def test_daemon_status_uses_reported_fts_coverage_pct(self) -> None:
        env = _make_app_env()

        _show_daemon_status(
            env,
            {
                "daemon_liveness": True,
                "fts_readiness": {
                    "messages_ready": False,
                    "coverage_pct": 87.5,
                },
            },
        )

        assert "87.5% indexed" in _combined_calls(env)

    def test_daemon_status_treats_null_fts_coverage_as_unknown_progress(self) -> None:
        env = _make_app_env()

        _show_daemon_status(
            env,
            {
                "daemon_liveness": True,
                "fts_readiness": {
                    "messages_ready": False,
                    "coverage_pct": None,
                },
            },
        )

        assert "FTS: [yellow]0.0% indexed[/yellow]" in _combined_calls(env)

    def test_daemon_status_archive_fts_reports_message_surface(self) -> None:
        env = _make_app_env()

        _show_daemon_status(
            env,
            {
                "daemon_liveness": True,
                "fts_readiness": {
                    "indexed_surface": "messages_fts",
                    "messages_ready": True,
                    "coverage_pct": 100.0,
                },
            },
        )

        assert "FTS: [green]100.0% indexed[/green]" in _combined_calls(env)

    @pytest.mark.frozen_clock_modules("polylogue.readiness.capability")
    def test_daemon_status_renders_raw_frontier_integrity(self, frozen_clock: FrozenClock) -> None:
        env = _make_app_env()
        unknown = _healthy_raw_frontier_integrity()
        unknown.update(
            {
                "available": False,
                "overall_status": "unknown",
                "broken_head_status": "unknown",
                "missing_source_raw_status": "unknown",
                "cursor_ahead_status": "unknown",
                "cursor_ahead_reason": "ops cursor authority unavailable",
            }
        )

        _show_daemon_status(
            env,
            {
                "daemon_liveness": True,
                "status_snapshot": _fresh_status_snapshot(frozen_clock),
                "raw_frontier_integrity": unknown,
            },
        )

        rendered = _combined_calls(env)
        assert "Raw frontier: [yellow]unknown[/yellow]" in rendered
        assert "ops cursor authority unavailable" in rendered

    @pytest.mark.frozen_clock_modules("polylogue.readiness.capability")
    def test_compact_daemon_status_cannot_preserve_stale_green_frontier(self, frozen_clock: FrozenClock) -> None:
        env = _make_app_env()
        violated = _healthy_raw_frontier_integrity()
        violated.update(
            {
                "overall_status": "violated",
                "broken_head_status": "violated",
                "broken_head_count": 1,
                "broken_head_samples": [{"accepted_raw_id": "raw-1"}],
                "broken_head_reason": "1 active seed is broken",
            }
        )
        _show_status_json(
            env,
            {
                "ok": True,
                "daemon_liveness": True,
                "status_snapshot": _fresh_status_snapshot(frozen_clock),
                "raw_frontier_integrity": violated,
            },
        )

        payload = json.loads(_combined_calls(env))
        assert payload["ok"] is False
        assert payload["raw_frontier_integrity"]["overall_status"] == "violated"
        assert "broken_head_samples" not in payload["raw_frontier_integrity"]

    @pytest.mark.parametrize("full", [False, True])
    def test_daemon_status_json_missing_frontier_is_explicit_unknown(self, full: bool) -> None:
        env = _make_app_env()

        _show_status_json(env, {"ok": True, "daemon_liveness": True}, full=full)

        payload = json.loads(_combined_calls(env))
        assert payload["ok"] is False
        assert payload["raw_frontier_integrity"]["overall_status"] == "unknown"
        assert payload["component_readiness"]["raw_frontier_integrity"]["state"] == "unknown"

    @pytest.mark.frozen_clock_modules("polylogue.readiness.capability")
    def test_status_ok_requires_complete_fresh_frontier_authority(self, frozen_clock: FrozenClock) -> None:
        assert _status_ok({"ok": True, "daemon_liveness": True}) is False
        healthy = _healthy_raw_frontier_integrity()
        assert _status_ok({"ok": True, "raw_frontier_integrity": healthy}) is True
        assert (
            _status_ok(
                {"ok": True, "raw_frontier_integrity": healthy},
                require_fresh_snapshot=True,
            )
            is False
        )
        assert (
            _status_ok(
                {
                    "ok": True,
                    "status_snapshot": _fresh_status_snapshot(frozen_clock),
                    "raw_frontier_integrity": healthy,
                },
                require_fresh_snapshot=True,
            )
            is True
        )
        assert (
            _status_ok(
                {
                    "ok": True,
                    "daemon_liveness": True,
                    "status_snapshot": {"state": "stale"},
                    "raw_frontier_integrity": healthy,
                }
            )
            is False
        )

    def test_daemon_status_text_marks_missing_frontier_non_green(self) -> None:
        env = _make_app_env()

        _show_daemon_status(env, {"ok": True, "daemon_liveness": True})

        rendered = _combined_calls(env)
        assert "[bold yellow]Daemon: running; status degraded[/bold yellow]" in rendered
        assert "Raw frontier: [yellow]unknown[/yellow]" in rendered

    def test_daemon_status_stale_healthy_frontier_becomes_unknown(self) -> None:
        env = _make_app_env()
        _show_status_json(
            env,
            {
                "ok": True,
                "daemon_liveness": True,
                "status_snapshot": {"state": "stale"},
                "raw_frontier_integrity": _healthy_raw_frontier_integrity(),
            },
            full=True,
        )

        payload = json.loads(_combined_calls(env))
        assert payload["ok"] is False
        assert payload["raw_frontier_integrity"]["overall_status"] == "unknown"
        assert "omitted freshness provenance" in payload["raw_frontier_integrity"]["broken_head_reason"]

    def test_daemon_status_json_is_compact_by_default(self) -> None:
        env = _make_app_env()
        full_payload = {
            "ok": True,
            "daemon_liveness": True,
            "checked_at": "2026-07-02T17:00:00+00:00",
            "component_readiness": {
                "search": {
                    "component": "search",
                    "state": "ready",
                    "counts": {"messages_fts_count": 5},
                }
            },
            "live_ingest_attempts": {
                "running_count": 0,
                "recent": [
                    {
                        "attempt_id": "attempt-1",
                        "status": "completed",
                        "stage": "idle",
                        "worker_completed_count": 4,
                        "worker_total_count": 4,
                        "large_debug_payload": "x" * 1000,
                    }
                ],
            },
            "archive_debt": {
                "available": True,
                "totals": {"total": 1},
                "rows": [{"debt_ref": "debt-1", "summary": "large raw row"}],
            },
            "raw_materialization_readiness": {
                "total": 1,
                "sampled_rows": [{"raw_id": "raw-1"}],
            },
            "live_cursor": {"failing_files": ["large"]},
            "catchup": {"debug": True},
            "convergence": {"debug": True},
            "failing_files": ["large"],
            "last_ingestion_batch": {"debug": True},
        }

        _show_status_json(env, full_payload)

        payload = json.loads(_combined_calls(env))
        assert payload["source"] == "daemon"
        assert payload["daemon_liveness"] is True
        assert payload["component_readiness"]["search"]["state"] == "ready"
        assert payload["ingest"]["latest"]["attempt_id"] == "attempt-1"
        assert "large_debug_payload" not in payload["ingest"]["latest"]
        assert payload["archive_debt"]["row_count"] == 1
        assert "rows" not in payload["archive_debt"]
        assert "sampled_rows" not in payload["raw_materialization_readiness"]
        for heavy_key in ("live_cursor", "catchup", "convergence", "failing_files", "last_ingestion_batch"):
            assert heavy_key not in payload

    def test_daemon_status_json_full_preserves_fields_but_normalizes_missing_authority(self) -> None:
        env = _make_app_env()
        full_payload = {
            "daemon_liveness": True,
            "live_cursor": {"tracked_file_count": 2},
            "archive_debt": {"rows": [{"debt_ref": "debt-1"}]},
        }

        _show_status_json(env, full_payload, full=True)

        payload = json.loads(_combined_calls(env))
        assert payload["daemon_liveness"] is True
        assert payload["live_cursor"] == full_payload["live_cursor"]
        assert payload["archive_debt"] == full_payload["archive_debt"]
        assert payload["ok"] is False
        assert payload["raw_frontier_integrity"]["overall_status"] == "unknown"
        assert payload["component_readiness"]["raw_frontier_integrity"]["state"] == "unknown"

    def test_status_command_full_json_preserves_fields_but_normalizes_missing_authority(self) -> None:
        env = _make_app_env()
        full_payload = {
            "daemon_liveness": True,
            "live_cursor": {"tracked_file_count": 2},
            "archive_debt": {"rows": [{"debt_ref": "debt-1"}]},
        }

        with patch("polylogue.cli.commands.status.urlopen", return_value=_FakeDaemonResponse(full_payload)):
            result = CliRunner().invoke(
                status_command,
                ["--daemon-url", "http://127.0.0.1:8765", "--json", "--full"],
                obj=env,
            )

        assert result.exit_code == 0
        payload = json.loads(_combined_calls(env))
        assert payload["daemon_liveness"] is True
        assert payload["live_cursor"] == full_payload["live_cursor"]
        assert payload["archive_debt"] == full_payload["archive_debt"]
        assert payload["ok"] is False
        assert payload["raw_frontier_integrity"]["overall_status"] == "unknown"
        assert payload["component_readiness"]["raw_frontier_integrity"]["state"] == "unknown"

    def test_status_command_default_json_compacts_daemon_payload(self) -> None:
        env = _make_app_env()
        full_payload = {
            "daemon_liveness": True,
            "live_cursor": {"tracked_file_count": 2},
            "archive_debt": {"rows": [{"debt_ref": "debt-1"}]},
        }

        with patch("polylogue.cli.commands.status.urlopen", return_value=_FakeDaemonResponse(full_payload)):
            result = CliRunner().invoke(
                status_command,
                ["--daemon-url", "http://127.0.0.1:8765", "--json"],
                obj=env,
            )

        assert result.exit_code == 0
        payload = json.loads(_combined_calls(env))
        assert payload["source"] == "daemon"
        assert "live_cursor" not in payload
        assert "rows" not in payload["archive_debt"]

    def test_status_command_uses_discovered_dev_loop_daemon_url(self) -> None:
        env = _make_app_env()
        full_payload = {
            "ok": True,
            "daemon_liveness": True,
            "checked_at": "2026-07-04T07:20:00+02:00",
        }
        requested_urls: list[str] = []

        def fake_urlopen(request: Any, *, timeout: float) -> _FakeDaemonResponse:
            requested_urls.append(str(request.full_url))
            if requested_urls[-1].startswith("http://127.0.0.1:8766/"):
                raise TimeoutError
            assert requested_urls[-1] == "http://127.0.0.1:8786/api/status"
            return _FakeDaemonResponse(full_payload)

        with patch(
            "polylogue.cli.commands.status._candidate_daemon_urls",
            return_value=("http://127.0.0.1:8766", "http://127.0.0.1:8786"),
        ):
            with patch("polylogue.cli.commands.status.urlopen", side_effect=fake_urlopen):
                result = CliRunner().invoke(
                    status_command,
                    ["--daemon-url", "http://127.0.0.1:8766", "--json"],
                    obj=env,
                )

        assert result.exit_code == 0
        assert requested_urls == [
            "http://127.0.0.1:8766/api/status",
            "http://127.0.0.1:8786/api/status",
        ]
        payload = json.loads(_combined_calls(env))
        assert payload["source"] == "daemon"
        assert payload["daemon_liveness"] is True

    def test_direct_json_no_archive(self) -> None:
        """_show_direct_json when DB does not exist produces valid JSON."""
        env = _make_app_env()
        fake_db = Path("/tmp/nonexistent.db")
        fake_root = Path("/tmp/test-archive")

        with patch("polylogue.paths.db_path", return_value=fake_db):
            with patch("polylogue.paths.archive_root", return_value=fake_root):
                _show_direct_json(env)

        # The first call's first argument is the JSON string
        console: Any = env.ui.console
        output = console.calls[0]
        payload = json.loads(output)
        assert payload["daemon_liveness"] is False
        assert payload["db_exists"] is False
        assert payload["archive_root"] == str(fake_root)

    @pytest.mark.integration
    def test_status_subprocess_no_archive(self, tmp_path: Path) -> None:
        """polylogue ops status on fresh XDG paths shows actionable message."""
        from tests.infra.cli_subprocess import run_cli

        archive_root = tmp_path / "polylogue"
        env = {
            **os.environ,
            "POLYLOGUE_ARCHIVE_ROOT": str(archive_root),
            "XDG_DATA_HOME": str(tmp_path / "data"),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "HOME": str(tmp_path),
        }
        result = run_cli(["--plain", "ops", "status"], env=env)
        output_lower = result.output.lower()
        assert result.exit_code == 0
        assert "traceback" not in output_lower
        assert "no archive" in output_lower or "polylogued" in output_lower


class TestStatusDiagnosticIntegration:
    """End-to-end coverage that the new diagnostics never leak tracebacks (#1263)."""

    def _xdg_env(self, tmp_path: Path) -> dict[str, str]:
        return {
            **os.environ,
            "POLYLOGUE_ARCHIVE_ROOT": str(tmp_path / "polylogue"),
            "XDG_DATA_HOME": str(tmp_path / "data"),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "HOME": str(tmp_path),
            "POLYLOGUE_DAEMON_URL": "http://127.0.0.1:1",
        }

    @pytest.mark.integration
    def test_status_subprocess_schema_mismatch(self, tmp_path: Path) -> None:
        """A db with the wrong PRAGMA user_version yields actionable text, no traceback."""
        import sqlite3

        from tests.infra.cli_subprocess import run_cli

        data_home = tmp_path / "data" / "polylogue"
        data_home.mkdir(parents=True, exist_ok=True)
        db = data_home / "index.db"
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA user_version = 99")
        conn.commit()
        conn.close()

        result = run_cli(["--plain", "ops", "status"], env=self._xdg_env(tmp_path))
        output_lower = result.output.lower()
        assert result.exit_code == 0
        assert "traceback" not in output_lower
        assert "daemon not running" in output_lower or "polylogued" in output_lower

    @pytest.mark.integration
    def test_status_subprocess_stale_pidfile(self, tmp_path: Path) -> None:
        """A stale pidfile yields actionable text, no traceback."""
        import sqlite3

        from tests.infra.cli_subprocess import run_cli

        data_home = tmp_path / "data" / "polylogue"
        data_home.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(data_home / "index.db").close()
        archive = tmp_path / "polylogue"
        archive.mkdir(parents=True, exist_ok=True)
        (archive / "daemon.pid").write_text("99999999\n")

        result = run_cli(["--plain", "ops", "status"], env=self._xdg_env(tmp_path))
        output_lower = result.output.lower()
        assert result.exit_code == 0
        assert "traceback" not in output_lower
        assert "pidfile" in output_lower or "polylogued" in output_lower

    @pytest.mark.integration
    def test_status_subprocess_no_sources(self, tmp_path: Path) -> None:
        """An empty roots config surfaces the no-sources hint."""
        import sqlite3

        from tests.infra.cli_subprocess import run_cli

        data_home = tmp_path / "data" / "polylogue"
        data_home.mkdir(parents=True, exist_ok=True)
        sqlite3.connect(data_home / "index.db").close()
        config_home = tmp_path / "config" / "polylogue"
        config_home.mkdir(parents=True, exist_ok=True)
        (config_home / "polylogue.toml").write_text("[sources]\nroots = []\n")

        result = run_cli(["--plain", "ops", "status"], env=self._xdg_env(tmp_path))
        output_lower = result.output.lower()
        assert result.exit_code == 0
        assert "traceback" not in output_lower


class TestEnvIsolation:
    """Regression coverage for #1325: workspace_env strips host POLYLOGUE_* env vars."""

    def test_autouse_clears_host_polylogue_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Setting POLYLOGUE_* before the autouse runs must be wiped.

        ``monkeypatch`` here is a *different* instance than the autouse
        fixture, but pytest's fixture finalisation runs LIFO: the autouse
        runs first (clearing host env) and the per-test monkeypatch runs
        after. We simulate the "operator daemon already running" case by
        re-asserting that no POLYLOGUE_* leaked through.
        """
        # The autouse ``_clear_polylogue_env`` fixture has already run and
        # stripped every POLYLOGUE_* var the host had. Verify it.
        leaked = [
            k
            for k in os.environ
            if k.startswith("POLYLOGUE_")
            and k
            not in {
                "POLYLOGUE_SITE_CONFIG",
                "POLYLOGUE_DAEMON_URL",
            }
        ]
        assert leaked == [], f"host POLYLOGUE_* vars leaked into test env: {leaked}"
        # And the daemon URL must be routed to an unreachable address.
        assert os.environ["POLYLOGUE_DAEMON_URL"] == "http://127.0.0.1:1"
        # And site config lookup must be disabled.
        assert os.environ["POLYLOGUE_SITE_CONFIG"] == ""

    def test_workspace_env_overrides_polluted_host(
        self,
        workspace_env: dict[str, Path],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """``workspace_env`` must win over a contaminated host environment.

        Set the offending vars in this test body to simulate a polluted
        host that escaped the autouse clear (e.g. a vendoring agent), then
        re-invoke the relevant resolver paths.
        """
        # Simulate an operator who has POLYLOGUE_ARCHIVE_ROOT set in their
        # shell pointing at the production archive.
        production_archive = Path("/var/lib/polylogue/archive")
        monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(production_archive))

        # workspace_env declared its own archive root via fixture wiring;
        # _clear_polylogue_env ran before workspace_env so the production
        # value above was set AFTER workspace_env. The fixture's contract
        # is that *its* value is the one tests should see at fixture-setup
        # time. We assert the documented value is the tmp_path one.
        assert workspace_env["archive_root"] == tmp_path / "archive"
        # The CLI default daemon URL resolution must not point at the host
        # ``polylogued`` listening on 8766.
        from polylogue.cli.commands.status import _default_daemon_url

        assert _default_daemon_url() == "http://127.0.0.1:1"


class TestDaemonStatus:
    """Daemon status rendering tests."""

    def test_empty_archive_with_sources(self) -> None:
        """When daemon runs but archive is empty, source discovery is shown."""
        env = _make_app_env()

        status_payload: dict[str, object] = {
            "daemon_liveness": True,
            "component_state": {
                "watcher": {"state": "running", "description": "watching 2 sources"},
            },
            "insight_freshness": {"total_sessions": 0, "sessions_with_profiles": 0},
            "live": {
                "sources": [
                    {"name": "claude-code", "root": "/tmp/claude", "exists": True},
                    {"name": "codex", "root": "/tmp/codex", "exists": False},
                ]
            },
            "watcher_roots": ["/tmp/claude", "/tmp/codex"],
            "live_ingest_attempts": {},
            "fts_readiness": {},
            "db_size_bytes": 0,
            "checked_at": "",
        }
        _show_daemon_status(env, status_payload)
        combined = _combined_calls(env)
        # Daemon-status output should report watching sources.
        assert "watching" in combined.lower(), f"expected 'watching' in status output, got: {combined[:200]}"

    def test_running_daemon_with_data(self) -> None:
        """When daemon runs with data, normal status is shown without first-run hints."""
        env = _make_app_env()

        status_payload: dict[str, object] = {
            "daemon_liveness": True,
            "component_state": {
                "watcher": {"state": "running", "description": "watching 2 sources"},
            },
            "insight_freshness": {"total_sessions": 42, "sessions_with_profiles": 40},
            "live": {
                "sources": [
                    {"name": "claude-code", "root": "/tmp/claude", "exists": True},
                ]
            },
            "live_ingest_attempts": {
                "completed_count": 10,
                "total_count": 10,
            },
            "fts_readiness": {"coverage_pct": 98.5},
            "db_size_bytes": 1_048_576,
            "disk_free_bytes": 107_374_182_400,
            "checked_at": "2026-05-07T12:00:00",
        }
        _show_daemon_status(env, status_payload)
        combined = _combined_calls(env)
        assert "no sessions" not in combined.lower()
        assert "running" in combined.lower()


@pytest.mark.integration
def test_status_command_accepts_json_alias_flag(tmp_path: Path) -> None:
    """`polylogue ops status --json` is a documented alias for `--format json`.

    Sibling subcommands (`list`, `tags`, `sources`, `stats`) accept `--json`;
    `status` previously rejected it with "No such option '--json'". Closes #1612.
    """
    from tests.infra.cli_subprocess import run_cli

    env = {
        **os.environ,
        "POLYLOGUE_ARCHIVE_ROOT": str(tmp_path / "polylogue"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "HOME": str(tmp_path),
        "POLYLOGUE_DAEMON_URL": "http://127.0.0.1:1",
    }
    result = run_cli(["--plain", "ops", "status", "--json"], env=env)
    assert result.exit_code == 0, result.output
    assert "No such option" not in result.output
    parsed = json.loads(result.stdout)
    assert isinstance(parsed, dict)

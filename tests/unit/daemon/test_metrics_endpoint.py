"""Contract tests for the daemon ``/metrics`` Prometheus endpoint (#1321).

Pins:

1. **Exposition format** — every metric carries ``# HELP`` and ``# TYPE``
   directives before its samples, and the response uses the documented
   ``text/plain; version=0.0.4`` content type that Prometheus expects.
2. **Series inventory** — the operator-facing series names are stable;
   adding or removing one is a contract change visible in this file.
3. **Graceful degradation** — when a backing table is missing (fresh
   archive, schema bump mid-rollout) the endpoint still emits the
   discovery skeleton with zero values rather than 5xx-ing.
4. **Unauthenticated** — same posture as ``/healthz/*``; scrapers do
   not carry credentials.
5. **State observation** — counts derived from ``live_ingest_attempt``,
   ``live_convergence_debt``, ``pending_blob_refs``, and FTS triggers
   round-trip through the exposition format.
"""

from __future__ import annotations

import re
import sqlite3
from email.message import Message
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

from polylogue.daemon.metrics import (
    PROMETHEUS_CONTENT_TYPE,
    format_metrics,
)

if TYPE_CHECKING:
    from polylogue.daemon.http import DaemonAPIHandler, DaemonAPIHTTPServer


# ---------------------------------------------------------------------------
# Series inventory — pinned contract
# ---------------------------------------------------------------------------

EXPECTED_SERIES: frozenset[str] = frozenset(
    {
        "polylogue_daemon_uptime_seconds",
        "polylogue_daemon_build_info",
        "polylogue_live_ingest_attempts_total",
        "polylogue_live_ingest_attempts_in_flight",
        "polylogue_live_ingest_attempt_duration_seconds",
        "polylogue_convergence_debt_count",
        "polylogue_blob_lease_pending_count",
        "polylogue_blob_lease_distinct_operations",
        "polylogue_fts_trigger_present",
        "polylogue_fts_triggers_all_present",
        "polylogue_fts_freshness_ready",
        "polylogue_live_ingest_memory_mebibytes",
        "polylogue_stale_cursor_writes_total",
        "polylogue_embedding_conversations",
        "polylogue_embedding_messages",
        "polylogue_embedding_coverage_percent",
        "polylogue_embedding_status_state",
        "polylogue_embedding_retrieval_ready",
        "polylogue_embedding_latest_catchup_run_info",
        "polylogue_embedding_latest_catchup_conversations",
        "polylogue_embedding_latest_catchup_messages",
        "polylogue_embedding_latest_catchup_estimated_cost_usd",
    }
)


_HELP_RE = re.compile(r"^# HELP (\S+) ")
_TYPE_RE = re.compile(r"^# TYPE (\S+) (counter|gauge|histogram|summary|untyped)$")


def _parse_exposition(body: str) -> dict[str, dict[str, object]]:
    """Parse Prometheus exposition body into ``{metric: {type, samples}}``."""
    metrics: dict[str, dict[str, object]] = {}
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if m := _HELP_RE.match(line):
            metrics.setdefault(m.group(1), {"type": None, "help": True, "samples": []})
        elif m := _TYPE_RE.match(line):
            entry = metrics.setdefault(m.group(1), {"type": None, "help": False, "samples": []})
            entry["type"] = m.group(2)
        elif line.startswith("#"):
            continue
        else:
            # sample line: "metric{labels} value"
            name = line.split("{", 1)[0].split(" ", 1)[0]
            entry = metrics.setdefault(name, {"type": None, "help": False, "samples": []})
            cast(list[str], entry["samples"]).append(line)
    return metrics


# ---------------------------------------------------------------------------
# format_metrics() unit tests
# ---------------------------------------------------------------------------


class TestFormatMetricsExpositionShape:
    """Hand-rolled exposition must satisfy the documented format."""

    def test_missing_db_emits_discovery_skeleton(self, tmp_path: Path) -> None:
        body = format_metrics(tmp_path / "missing.db")
        parsed = _parse_exposition(body)
        # Uptime and build_info always emit; remaining series emit as
        # zero-sample discovery placeholders when the DB is absent.
        for name in EXPECTED_SERIES:
            assert name in parsed, f"missing series for fresh archive: {name}"
            assert parsed[name]["type"] is not None, f"missing TYPE for {name}"

    def test_every_sample_line_has_preceding_help_and_type(self, tmp_path: Path) -> None:
        body = format_metrics(tmp_path / "missing.db")
        parsed = _parse_exposition(body)
        for name, entry in parsed.items():
            assert entry["help"], f"{name} missing # HELP"
            assert entry["type"], f"{name} missing # TYPE"

    def test_build_info_exposes_version_label(self, tmp_path: Path) -> None:
        body = format_metrics(tmp_path / "missing.db")
        assert "polylogue_daemon_build_info{version=" in body

    def test_no_unknown_label_escaping(self, tmp_path: Path) -> None:
        """Quotes/backslashes inside label values must be escaped."""
        from polylogue.daemon.metrics import _escape_label_value

        assert _escape_label_value('a"b') == 'a\\"b'
        assert _escape_label_value("c\\d") == "c\\\\d"
        assert _escape_label_value("e\nf") == "e\\nf"


class TestFormatMetricsReadsArchiveState:
    """When the DB exists, series reflect live state."""

    def _make_db(self, tmp_path: Path) -> Path:
        db = tmp_path / "archive.db"
        conn = sqlite3.connect(db)
        try:
            conn.executescript(
                """
                CREATE TABLE live_ingest_attempt (
                    attempt_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    started_at TEXT,
                    updated_at TEXT,
                    convergence_time_s REAL,
                    stale_cursor_write_count INTEGER DEFAULT 0,
                    rss_current_mb REAL,
                    cgroup_memory_current_mb REAL,
                    cgroup_memory_file_mb REAL,
                    cgroup_memory_inactive_file_mb REAL,
                    source_paths_json TEXT
                );
                CREATE TABLE fts_freshness_state (
                    surface TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    checked_at TEXT NOT NULL,
                    source_rows INTEGER NOT NULL DEFAULT 0,
                    indexed_rows INTEGER NOT NULL DEFAULT 0,
                    missing_rows INTEGER NOT NULL DEFAULT 0,
                    excess_rows INTEGER NOT NULL DEFAULT 0,
                    duplicate_rows INTEGER NOT NULL DEFAULT 0,
                    detail TEXT
                );
                CREATE TABLE live_convergence_debt (
                    debt_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE TABLE pending_blob_refs (
                    blob_hash TEXT,
                    operation_id TEXT,
                    acquired_at INTEGER
                );
                CREATE TABLE messages (message_id TEXT PRIMARY KEY);
                CREATE TABLE conversations (conversation_id TEXT PRIMARY KEY);
                CREATE TABLE embedding_status (
                    conversation_id TEXT PRIMARY KEY,
                    needs_reindex INTEGER NOT NULL,
                    error_message TEXT
                );
                CREATE TABLE message_embeddings_rowids (message_id TEXT PRIMARY KEY);
                CREATE TRIGGER messages_fts_ai AFTER INSERT ON messages
                    BEGIN SELECT 1; END;
                CREATE TRIGGER messages_fts_ad AFTER DELETE ON messages
                    BEGIN SELECT 1; END;
                CREATE TRIGGER messages_fts_au AFTER UPDATE ON messages
                    BEGIN SELECT 1; END;
                """
            )
            conn.executemany(
                """
                INSERT INTO live_ingest_attempt (
                    attempt_id, status, started_at, updated_at, convergence_time_s,
                    stale_cursor_write_count, rss_current_mb, cgroup_memory_current_mb,
                    cgroup_memory_file_mb, cgroup_memory_inactive_file_mb
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("a1", "completed", "2026-05-01T00:00:00Z", "2026-05-01T00:00:00Z", 1.5, 0, 40.0, 80.0, 20.0, 10.0),
                    ("a2", "completed", "2026-05-01T00:00:01Z", "2026-05-01T00:00:01Z", 2.5, 1, 42.0, 82.0, 21.0, 11.0),
                    ("a3", "failed", "2026-05-01T00:00:02Z", "2026-05-01T00:00:02Z", None, 0, 43.0, 83.0, 22.0, 12.0),
                    ("a4", "running", "2026-05-01T00:00:03Z", "2026-05-01T00:00:03Z", None, 0, 44.0, 84.0, 23.0, 13.0),
                ],
            )
            conn.executemany(
                "INSERT INTO fts_freshness_state (surface, state, checked_at) VALUES (?, ?, ?)",
                [
                    ("messages_fts", "ready", "2026-05-01T00:00:04Z"),
                    ("action_events_fts", "stale", "2026-05-01T00:00:04Z"),
                ],
            )
            conn.executemany(
                "INSERT INTO live_convergence_debt (debt_id, stage, status) VALUES (?, ?, ?)",
                [
                    ("d1", "parse", "failed"),
                    ("d2", "parse", "failed"),
                    ("d3", "convergence", "failed"),
                    ("d4", "convergence", "resolved"),  # filtered
                ],
            )
            conn.executemany(
                "INSERT INTO pending_blob_refs (blob_hash, operation_id, acquired_at) VALUES (?, ?, ?)",
                [
                    ("hash_a", "op1", 100),
                    ("hash_b", "op1", 101),
                    ("hash_c", "op2", 102),
                ],
            )
            conn.executemany(
                "INSERT INTO conversations (conversation_id) VALUES (?)",
                [("conv-embedded",), ("conv-pending",), ("conv-missing-status",)],
            )
            conn.executemany(
                "INSERT INTO embedding_status (conversation_id, needs_reindex, error_message) VALUES (?, ?, ?)",
                [
                    ("conv-embedded", 0, None),
                    ("conv-pending", 1, "provider timeout"),
                ],
            )
            conn.executemany(
                "INSERT INTO message_embeddings_rowids (message_id) VALUES (?)",
                [("msg-1",), ("msg-2",)],
            )
            conn.commit()
        finally:
            conn.close()
        from polylogue.storage.embeddings.progress import (
            CatchupRunDelta,
            CatchupRunStart,
            finish_embedding_catchup_run,
            record_embedding_catchup_progress,
            start_embedding_catchup_run,
        )

        run_id = start_embedding_catchup_run(
            db,
            CatchupRunStart(
                rebuild=True,
                max_conversations=3,
                max_messages=9,
                stop_after_seconds=None,
                max_errors=1,
                planned_conversations=3,
                planned_messages=9,
            ),
        )
        record_embedding_catchup_progress(
            db,
            run_id,
            CatchupRunDelta(
                conversation_id="conv-embedded",
                embedded=True,
                embedded_messages=2,
                estimated_cost_usd=0.003,
            ),
        )
        record_embedding_catchup_progress(
            db,
            run_id,
            CatchupRunDelta(conversation_id="conv-missing-status", skipped=True),
        )
        record_embedding_catchup_progress(
            db,
            run_id,
            CatchupRunDelta(conversation_id="conv-pending", errored=True),
        )
        finish_embedding_catchup_run(db, run_id, status="stopped", stop_reason="max errors reached (1)")
        return db

    def test_attempt_counts_round_trip(self, tmp_path: Path) -> None:
        body = format_metrics(self._make_db(tmp_path))
        assert 'polylogue_live_ingest_attempts_total{status="completed"} 2' in body
        assert 'polylogue_live_ingest_attempts_total{status="failed"} 1' in body
        assert 'polylogue_live_ingest_attempts_total{status="running"} 1' in body
        assert "polylogue_live_ingest_attempts_in_flight 1" in body
        assert "polylogue_stale_cursor_writes_total 1" in body

    def test_duration_quantiles(self, tmp_path: Path) -> None:
        body = format_metrics(self._make_db(tmp_path))
        # Two completed durations: 1.5 and 2.5.
        assert 'polylogue_live_ingest_attempt_duration_seconds{quantile="min"} 1.5' in body
        assert 'polylogue_live_ingest_attempt_duration_seconds{quantile="max"} 2.5' in body
        assert 'polylogue_live_ingest_attempt_duration_seconds{quantile="mean"} 2.0' in body

    def test_convergence_debt_grouped_by_stage(self, tmp_path: Path) -> None:
        body = format_metrics(self._make_db(tmp_path))
        assert 'polylogue_convergence_debt_count{stage="convergence"} 1' in body
        assert 'polylogue_convergence_debt_count{stage="parse"} 2' in body

    def test_blob_lease_state(self, tmp_path: Path) -> None:
        body = format_metrics(self._make_db(tmp_path))
        assert "polylogue_blob_lease_pending_count 3" in body
        assert "polylogue_blob_lease_distinct_operations 2" in body

    def test_fts_trigger_presence_partial(self, tmp_path: Path) -> None:
        body = format_metrics(self._make_db(tmp_path))
        # Three of six triggers present in the test DB.
        assert 'polylogue_fts_trigger_present{trigger="messages_fts_ai"} 1' in body
        assert 'polylogue_fts_trigger_present{trigger="action_events_fts_ai"} 0' in body
        assert "polylogue_fts_triggers_all_present 0" in body

    def test_fts_freshness_and_memory_state(self, tmp_path: Path) -> None:
        body = format_metrics(self._make_db(tmp_path))
        assert 'polylogue_fts_freshness_ready{surface="messages_fts"} 1' in body
        assert 'polylogue_fts_freshness_ready{surface="action_events_fts"} 0' in body
        assert 'polylogue_live_ingest_memory_mebibytes{kind="rss_current"} 44.0' in body
        assert 'polylogue_live_ingest_memory_mebibytes{kind="cgroup_file"} 23.0' in body

    def test_embedding_backlog_and_latest_catchup_state(self, tmp_path: Path) -> None:
        body = format_metrics(self._make_db(tmp_path))
        assert 'polylogue_embedding_conversations{state="total"} 3' in body
        assert 'polylogue_embedding_conversations{state="embedded"} 1' in body
        assert 'polylogue_embedding_conversations{state="pending"} 2' in body
        assert 'polylogue_embedding_conversations{state="failed"} 1' in body
        assert 'polylogue_embedding_messages{state="embedded"} 2' in body
        assert "polylogue_embedding_coverage_percent 33.33333333333333" in body
        assert 'polylogue_embedding_status_state{status="partial"} 1' in body
        assert 'polylogue_embedding_status_state{status="complete"} 0' in body
        assert "polylogue_embedding_retrieval_ready 1" in body
        assert 'polylogue_embedding_latest_catchup_run_info{rebuild="true",status="stopped"} 1' in body
        assert 'polylogue_embedding_latest_catchup_conversations{state="planned"} 3' in body
        assert 'polylogue_embedding_latest_catchup_conversations{state="processed"} 3' in body
        assert 'polylogue_embedding_latest_catchup_conversations{state="embedded"} 1' in body
        assert 'polylogue_embedding_latest_catchup_conversations{state="skipped"} 1' in body
        assert 'polylogue_embedding_latest_catchup_conversations{state="failed"} 1' in body
        assert 'polylogue_embedding_latest_catchup_messages{state="planned"} 9' in body
        assert 'polylogue_embedding_latest_catchup_messages{state="embedded"} 2' in body
        assert "polylogue_embedding_latest_catchup_estimated_cost_usd 0.003" in body

    def test_archive_messages_total_uses_conversation_stats(self, tmp_path: Path) -> None:
        """#1629: per-source message counts avoid the 3.7M-row GROUP BY scan.

        Reading from ``conversation_stats`` (one row per conversation) is
        two orders of magnitude cheaper than ``SELECT source_name, COUNT(*)
        FROM messages GROUP BY source_name`` on a steady-state archive.
        """
        db = tmp_path / "archive.db"
        with sqlite3.connect(db) as conn:
            conn.executescript(
                """
                CREATE TABLE conversations (
                    conversation_id TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL
                );
                CREATE TABLE conversation_stats (
                    conversation_id TEXT PRIMARY KEY,
                    message_count INTEGER NOT NULL
                );
                INSERT INTO conversations VALUES ('c1', 'claude-code'), ('c2', 'claude-code'), ('c3', 'codex');
                INSERT INTO conversation_stats VALUES ('c1', 100), ('c2', 50), ('c3', 30);
                """
            )

        body = format_metrics(db)

        assert 'polylogue_archive_conversations_total{source="claude-code"} 2' in body
        assert 'polylogue_archive_conversations_total{source="codex"} 1' in body
        assert 'polylogue_archive_messages_total{source="claude-code"} 150' in body
        assert 'polylogue_archive_messages_total{source="codex"} 30' in body

    def test_embedding_metrics_tolerate_partial_tables(self, tmp_path: Path) -> None:
        db = tmp_path / "archive.db"
        with sqlite3.connect(db) as conn:
            conn.executescript("""
                CREATE TABLE embedding_status (
                    conversation_id TEXT PRIMARY KEY,
                    needs_reindex INTEGER NOT NULL,
                    error_message TEXT
                );
                INSERT INTO embedding_status VALUES ('conv-1', 1, NULL);
            """)

        body = format_metrics(db)

        assert 'polylogue_embedding_conversations{state="pending"} 1' in body
        assert "polylogue_embedding_coverage_percent 0.0" in body
        assert 'polylogue_embedding_status_state{status="none"} 1' in body
        assert "polylogue_embedding_retrieval_ready 0" in body


# ---------------------------------------------------------------------------
# HTTP handler integration — mirrors test_health_contract.py harness
# ---------------------------------------------------------------------------


class _MockServer:
    auth_token = ""
    api_host = "127.0.0.1"


class _MockHeaders:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self._headers = headers or {}

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._headers.get(key, default)


def _make_handler(method: str, path: str) -> DaemonAPIHandler:
    from polylogue.daemon.http import DaemonAPIHandler

    handler = DaemonAPIHandler.__new__(DaemonAPIHandler)
    handler.server = cast("DaemonAPIHTTPServer", _MockServer())
    handler.client_address = ("127.0.0.1", 12345)
    handler.path = path
    handler.command = method
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.headers = cast("Message[str, str]", _MockHeaders({"Content-Length": "0"}))
    handler.rfile = BytesIO(b"")
    handler.wfile = BytesIO()
    return handler


def _capture_text(handler: DaemonAPIHandler) -> MagicMock:
    send_text = MagicMock()
    handler._send_text = send_text  # type: ignore[method-assign]
    return send_text


class TestMetricsEndpoint:
    def test_metrics_route_responds_200_with_prometheus_content_type(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        handler = _make_handler("GET", "/metrics")
        send_text = _capture_text(handler)
        handler.do_GET()
        send_text.assert_called_once()
        status = send_text.call_args.args[0]
        content_type = send_text.call_args.kwargs["content_type"]
        assert status == HTTPStatus.OK
        assert content_type == PROMETHEUS_CONTENT_TYPE

    def test_metrics_endpoint_is_unauthenticated(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        """Same posture as /healthz/* — scrapers don't carry credentials."""
        handler = _make_handler("GET", "/metrics")
        handler.server.auth_token = "secret-token"
        send_text = _capture_text(handler)
        handler.do_GET()
        send_text.assert_called_once()
        status = send_text.call_args.args[0]
        assert status == HTTPStatus.OK

    def test_metrics_body_contains_expected_series(
        self,
        workspace_env: dict[str, Path],
    ) -> None:
        handler = _make_handler("GET", "/metrics")
        send_text = _capture_text(handler)
        handler.do_GET()
        body = send_text.call_args.args[1]
        parsed = _parse_exposition(body)
        missing = EXPECTED_SERIES - set(parsed.keys())
        assert not missing, f"missing series in /metrics response: {sorted(missing)}"

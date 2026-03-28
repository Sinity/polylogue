"""Focused tests for persisted run-record helpers."""

from __future__ import annotations

import asyncio
import json
import time

from polylogue.pipeline.runner import _write_run_json, latest_run
from polylogue.storage.backends.connection import open_connection


class TestWriteRunJson:
    def test_creates_runs_directory(self, tmp_path):
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        payload = {"run_id": "test-run-1", "timestamp": int(time.time()), "counts": {"conversations": 1}}

        result = _write_run_json(archive_root, payload)
        assert (archive_root / "runs").exists()
        assert result.exists()

    def test_writes_correct_content(self, tmp_path):
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        timestamp = int(time.time())
        payload = {
            "run_id": "abc123",
            "timestamp": timestamp,
            "counts": {"conversations": 10, "messages": 50},
            "indexed": True,
        }

        result_path = _write_run_json(archive_root, payload)
        content = json.loads(result_path.read_text())
        assert content["run_id"] == "abc123"
        assert content["timestamp"] == timestamp
        assert content["counts"] == {"conversations": 10, "messages": 50}
        assert content["indexed"] is True

    def test_filename_contains_timestamp_and_id(self, tmp_path):
        archive_root = tmp_path / "archive"
        archive_root.mkdir()
        result = _write_run_json(archive_root, {"run_id": "myrun", "timestamp": 1704067200})
        assert result.name == "run-1704067200-myrun.json"


class TestLatestRun:
    def test_no_runs_returns_none(self, workspace_env):
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with open_connection(db_path):
            pass
        assert asyncio.run(latest_run()) is None

    def test_returns_most_recent(self, workspace_env):
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with open_connection(db_path) as conn:
            for index, timestamp in enumerate([1000, 3000, 2000]):
                conn.execute(
                    """
                    INSERT INTO runs
                    (run_id, timestamp, plan_snapshot, counts_json, drift_json, indexed, duration_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"run-{index}",
                        str(timestamp),
                        None,
                        json.dumps({"conversations": index}),
                        None,
                        1 if index == 1 else 0,
                        100 * index,
                    ),
                )
            conn.commit()

        result = asyncio.run(latest_run())
        assert result is not None
        assert result.run_id == "run-1"
        assert result.timestamp == "3000"

    def test_parses_json_columns(self, workspace_env):
        db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        plan = {"conversations": 5, "messages": 20}
        counts = {"conversations": 4, "messages": 18, "rendered": 4}
        drift = {"new": {"conversations": 1}, "removed": {"conversations": 2}}

        with open_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO runs
                (run_id, timestamp, plan_snapshot, counts_json, drift_json, indexed, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("run-json", "5000", json.dumps(plan), json.dumps(counts), json.dumps(drift), 1, 500),
            )
            conn.commit()

        result = asyncio.run(latest_run())
        assert result is not None
        assert result.plan_snapshot == plan
        assert result.counts == counts
        assert result.drift == drift
        assert result.indexed is True
        assert result.duration_ms == 500

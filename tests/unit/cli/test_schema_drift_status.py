"""Tests for the `polylogue ops status` format-drift sentinel section (polylogue-da1)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from polylogue.cli.commands.status import _render_schema_drift_status
from polylogue.cli.shared.types import AppEnv
from polylogue.insights.schema_drift import schema_drift_status
from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_tier
from polylogue.storage.sqlite.archive_tiers.ops_write import record_schema_drift_sample
from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier


class _CapturingConsole:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def print(self, *args: object, **kwargs: object) -> None:
        self.calls.append(" ".join(str(a) for a in args))


def _make_app_env() -> AppEnv:
    ui: Any = MagicMock()
    ui.plain = True
    ui.console = _CapturingConsole()
    return AppEnv(ui=ui)


def test_schema_drift_status_reports_missing_ops_tier(tmp_path: Path) -> None:
    result = schema_drift_status(tmp_path, now_ms=10_000)
    assert result["available"] is False
    assert result["reason"] == "missing_ops_tier"


def test_schema_drift_status_computes_windowed_risky_rate(tmp_path: Path) -> None:
    conn = sqlite3.connect(str(tmp_path / "ops.db"))
    initialize_archive_tier(conn, ArchiveTier.OPS)
    now_ms = 40 * 24 * 60 * 60 * 1000  # arbitrary but far enough to have room for the 30d window
    within_window = now_ms - (10 * 24 * 60 * 60 * 1000)
    outside_window = now_ms - (40 * 24 * 60 * 60 * 1000)
    # 1 risky + 4 benign in-window -> 20% risky rate (crosses the warn threshold).
    record_schema_drift_sample(
        conn,
        origin="claude-code-session",
        element_kind="session_record",
        classification="field_changed",
        unseen_key_signature="",
        native_id_example="raw-risky",
        raw_id="raw-risky",
        observed_at_ms=within_window,
    )
    for i in range(4):
        record_schema_drift_sample(
            conn,
            origin="claude-code-session",
            element_kind="session_record",
            classification="new_field",
            unseen_key_signature="x",
            native_id_example=f"raw-benign-{i}",
            raw_id=f"raw-benign-{i}",
            observed_at_ms=within_window,
        )
    # A sample outside the window must not dilute the rate.
    record_schema_drift_sample(
        conn,
        origin="claude-code-session",
        element_kind="session_record",
        classification="new_field",
        unseen_key_signature="x",
        native_id_example="raw-stale",
        raw_id="raw-stale",
        observed_at_ms=outside_window,
    )
    conn.close()

    result = schema_drift_status(tmp_path, now_ms=now_ms)
    assert result["available"] is True
    origins = result["origins"]
    assert len(origins) == 1
    origin = origins[0]
    assert origin["origin"] == "claude-code-session"
    assert origin["total"] == 5
    assert origin["risky"] == 1
    assert abs(origin["risky_rate"] - 0.2) < 1e-9
    assert origin["severity"] == "error"  # 20% risky rate hits the error threshold


def test_render_schema_drift_status_prints_percent_and_examples() -> None:
    env = _make_app_env()
    drift = {
        "available": True,
        "since_ms": 1_700_000_000_000,
        "origins": [
            {
                "origin": "chatgpt-export",
                "total": 20,
                "risky": 5,
                "benign": 15,
                "risky_rate": 0.25,
                "severity": "error",
                "example_native_ids": ["raw-a", "raw-b"],
            }
        ],
    }
    _render_schema_drift_status(env, drift)
    output = " ".join(env.ui.console.calls)  # type: ignore[attr-defined]
    assert "chatgpt-export" in output
    assert "25%" in output
    assert "20 records" in output
    assert "carry unseen shapes" in output
    assert "raw-a" in output
    assert "devtools lab schema generate/promote" in output


def test_render_schema_drift_status_stays_quiet_when_all_origins_ok() -> None:
    env = _make_app_env()
    drift = {
        "available": True,
        "since_ms": 1_700_000_000_000,
        "origins": [
            {
                "origin": "chatgpt-export",
                "total": 20,
                "risky": 0,
                "benign": 20,
                "risky_rate": 0.0,
                "severity": "ok",
                "example_native_ids": [],
            }
        ],
    }
    _render_schema_drift_status(env, drift)
    assert env.ui.console.calls == []  # type: ignore[attr-defined]

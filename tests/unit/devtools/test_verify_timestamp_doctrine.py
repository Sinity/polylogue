from __future__ import annotations

import json

import pytest

from devtools import verify_timestamp_doctrine


def test_scan_flags_text_timestamp_column() -> None:
    """cpf.1: a TEXT column whose name looks like a timestamp must be flagged."""
    fixture_ddl = """
    CREATE TABLE IF NOT EXISTS widgets (
        widget_id     TEXT PRIMARY KEY,
        created_at    TEXT NOT NULL,
        label         TEXT
    ) STRICT;
    """
    violations = verify_timestamp_doctrine.scan_ddl_for_text_timestamps(fixture_ddl, tier="fixture")
    assert [v.column for v in violations] == ["created_at"]
    assert violations[0].declared_type == "TEXT"


def test_scan_passes_integer_epoch_ms_columns() -> None:
    """Existing INTEGER epoch-ms columns must not be flagged."""
    fixture_ddl = """
    CREATE TABLE IF NOT EXISTS widgets (
        widget_id       TEXT PRIMARY KEY,
        created_at_ms   INTEGER NOT NULL,
        updated_at_ms   INTEGER,
        label           TEXT
    ) STRICT;
    """
    assert verify_timestamp_doctrine.scan_ddl_for_text_timestamps(fixture_ddl, tier="fixture") == []


def test_scan_ignores_non_timestamp_text_columns() -> None:
    """Ordinary TEXT columns (no timestamp-like name) must not be flagged --
    only the segments at/ms/time/date or a `timestamp` substring count."""
    fixture_ddl = """
    CREATE TABLE IF NOT EXISTS widgets (
        widget_id      TEXT PRIMARY KEY,
        native_id      TEXT NOT NULL,
        content_hash   BLOB NOT NULL,
        word_count     INTEGER NOT NULL,
        git_branch     TEXT
    ) STRICT;
    """
    assert verify_timestamp_doctrine.scan_ddl_for_text_timestamps(fixture_ddl, tier="fixture") == []


def test_scan_flags_timestamp_substring_and_time_date_segments() -> None:
    fixture_ddl = """
    CREATE TABLE IF NOT EXISTS widgets (
        event_timestamp   TEXT NOT NULL,
        occurred_time     TEXT,
        observed_date     TEXT
    ) STRICT;
    """
    violations = verify_timestamp_doctrine.scan_ddl_for_text_timestamps(fixture_ddl, tier="fixture")
    assert {v.column for v in violations} == {"event_timestamp", "occurred_time", "observed_date"}


def test_durable_tier_ddl_has_no_text_timestamps(capsys: pytest.CaptureFixture[str]) -> None:
    """The real source.db/user.db DDL must currently pass -- all existing
    timestamp columns are already INTEGER epoch-ms."""
    assert verify_timestamp_doctrine.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["violations"] == []


def test_main_reports_violation_and_nonzero_exit(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        verify_timestamp_doctrine,
        "_collect_durable_tier_violations",
        lambda: [verify_timestamp_doctrine.TimestampViolation(tier="source", column="parsed_at", declared_type="TEXT")],
    )
    assert verify_timestamp_doctrine.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["violations"] == [{"tier": "source", "column": "parsed_at", "declared_type": "TEXT"}]

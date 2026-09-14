"""A refused, degraded or unread state is reported as itself, not as a positive result.

Each case below had the shape this repository's review rules single out: a
count, a verdict or an operator-facing line that reported something other than
what the run actually did.

Anti-vacuity is named per test.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import Provider
from polylogue.operations.daemon_workload_probe import _ops_recent_attempts
from polylogue.sources.parsers.base_models import ParsedSession
from polylogue.sources.parsers.claude.code_parser import _safe_int
from polylogue.sources.parsers.hermes_tool_outcome import tool_result_outcome


def test_recent_attempts_report_the_file_count_not_the_session_count(tmp_path: Path) -> None:
    """``parsed_raw_count`` is files; ``materialized_count`` is sessions.

    Anti-vacuity: restore ``int(row[7] or 0)`` for ``succeeded_file_count`` and
    this reports 9 succeeded files for an attempt that succeeded on 2 -- the
    CLI renders it as ``9/2 files``, which reads as ingestion that never
    happened.
    """
    ops_db = tmp_path / "ops.db"
    conn = sqlite3.connect(ops_db)
    conn.execute("PRAGMA user_version = 1")
    conn.execute(
        """
        CREATE TABLE ingest_attempts (
            attempt_id TEXT PRIMARY KEY,
            started_at_ms INTEGER,
            heartbeat_at_ms INTEGER,
            finished_at_ms INTEGER,
            status TEXT,
            phase TEXT,
            parsed_raw_count INTEGER,
            materialized_count INTEGER,
            error_message TEXT,
            source_paths_json TEXT
        )
        """
    )
    conn.execute("INSERT INTO ingest_attempts VALUES ('a1', 1000, 2000, 2000, 'succeeded', 'done', 2, 9, NULL, '[]')")
    conn.commit()
    conn.close()

    attempts = _ops_recent_attempts(ops_db, limit=5)

    assert len(attempts) == 1
    assert attempts[0]["succeeded_file_count"] == 2
    assert attempts[0]["materialized_session_count"] == 9


def test_hermes_boolean_error_field_is_an_unread_verdict_not_a_known_error() -> None:
    """``error`` carries a failure message; a bare boolean is not one.

    Anti-vacuity: restore ``if payload.get("error") is not None: return True``
    and ``{"output": "ok", "error": false}`` reports a known ERROR for a tool
    call the source recorded as fine.
    """
    is_error, _exit_code, reason = tool_result_outcome({"output": "ok", "error": False})
    assert is_error is None
    assert reason is not None

    is_error, _exit_code, reason = tool_result_outcome({"output": "ok", "error": ""})
    assert is_error is None
    assert reason is not None

    # A real failure message is still a known error.
    is_error, _exit_code, reason = tool_result_outcome({"output": "", "error": "command not found"})
    assert is_error is True
    assert reason is None


@pytest.mark.parametrize("hostile", [float("inf"), float("1e309"), float("nan")])
def test_non_finite_reported_cost_is_refused_not_admitted(hostile: float) -> None:
    """``"Infinity"`` is a valid JSON string that ``float()`` accepts.

    Anti-vacuity: drop the ``math.isfinite`` check and an ``inf`` cost is
    stored as exact reported evidence, poisoning every total it is summed into
    with no reader able to tell it from a real figure.
    """
    assert not math.isfinite(hostile)

    with pytest.raises(ValueError, match="finite"):
        ParsedSession(
            source_name=Provider.CLAUDE_CODE,
            provider_session_id="s",
            messages=[],
            reported_cost_usd=hostile,
        )


def test_finite_reported_cost_is_still_admitted() -> None:
    admitted = ParsedSession(
        source_name=Provider.CLAUDE_CODE,
        provider_session_id="s",
        messages=[],
        reported_cost_usd=1.5,
    )

    assert admitted.reported_cost_usd == 1.5


def test_non_finite_integer_field_does_not_escape_as_an_overflow_error() -> None:
    """``int(float("inf"))`` raises ``OverflowError``, which is not a ``ValueError``.

    Anti-vacuity: restore ``return int(float(str(value)))`` inside the
    ``(TypeError, ValueError)`` guard and this raises ``OverflowError`` out of
    the helper, refusing the whole session over one hostile integer field
    instead of the field itself.
    """
    assert _safe_int("Infinity") == 0
    assert _safe_int("1e309") == 0
    assert _safe_int("12") == 12

import sqlite3
from pathlib import Path
from typing import Any, cast

from scripts import agent_forensics


def _build_failure_followup_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE messages (
            session_id TEXT NOT NULL,
            message_id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            position INTEGER NOT NULL,
            model_name TEXT
        );
        CREATE TABLE blocks (
            block_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            message_id TEXT NOT NULL,
            block_type TEXT NOT NULL,
            tool_id TEXT,
            tool_name TEXT,
            tool_input TEXT,
            tool_result_is_error INTEGER,
            tool_result_exit_code INTEGER,
            text TEXT
        );
        CREATE INDEX idx_blocks_tool_result_outcome
            ON blocks(block_type, tool_result_is_error, tool_result_exit_code)
            WHERE block_type = 'tool_result';
        CREATE INDEX idx_blocks_tool_id
            ON blocks(tool_id)
            WHERE tool_id IS NOT NULL;
        """
    )
    messages = [
        ("s1", "tool-ack", "tool", 1, "claude-opus"),
        ("s1", "next-ack", "assistant", 2, "claude-sonnet"),
        ("s1", "tool-silent", "tool", 3, "claude-opus"),
        ("s1", "next-silent", "assistant", 4, "claude-haiku"),
        ("s1", "tool-short", "tool", 5, "claude-opus"),
        ("s1", "next-short", "assistant", 6, "claude-haiku"),
        ("s2", "tool-missing-next", "tool", 1, "deepseek-v4"),
    ]
    conn.executemany(
        "INSERT INTO messages(session_id, message_id, role, position, model_name) VALUES (?, ?, ?, ?, ?)",
        messages,
    )
    blocks = [
        ("u1", "s1", "tool-ack", "tool_use", "t1", "Bash", "pytest", None, None, None),
        ("r1", "s1", "tool-ack", "tool_result", "t1", None, None, 1, None, "failed"),
        ("u2", "s1", "tool-silent", "tool_use", "t2", "Bash", "ls missing", None, None, None),
        ("r2", "s1", "tool-silent", "tool_result", "t2", None, None, 0, 2, "exit 2"),
        ("u3", "s1", "tool-short", "tool_use", "t3", "Edit", "patch", None, None, None),
        ("r3", "s1", "tool-short", "tool_result", "t3", None, None, 1, None, "failed"),
        ("u4", "s2", "tool-missing-next", "tool_use", "t4", "Read", "read file", None, None, None),
        ("r4", "s2", "tool-missing-next", "tool_result", "t4", None, None, 0, 1, "exit 1"),
        (
            "b-next-ack",
            "s1",
            "next-ack",
            "text",
            None,
            None,
            None,
            None,
            None,
            "The command failed with an error, so I will fix it.",
        ),
        (
            "b-next-silent",
            "s1",
            "next-silent",
            "text",
            None,
            None,
            None,
            None,
            None,
            "I will continue by inspecting the neighboring module now.",
        ),
        ("b-next-short", "s1", "next-short", "text", None, None, None, None, None, "ok"),
    ]
    conn.executemany(
        """
        INSERT INTO blocks(
            block_id, session_id, message_id, block_type, tool_id, tool_name, tool_input,
            tool_result_is_error, tool_result_exit_code, text
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        blocks,
    )
    return conn


def _build_usage_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            origin TEXT NOT NULL,
            sort_key_ms INTEGER NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE messages (
            message_id TEXT PRIMARY KEY
        );
        CREATE TABLE blocks (
            message_id TEXT NOT NULL,
            text TEXT
        );
        CREATE TABLE session_model_usage (
            session_id TEXT NOT NULL,
            model_name TEXT,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            cache_read_tokens INTEGER NOT NULL DEFAULT 0,
            cache_write_tokens INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL,
            cost_provenance TEXT NOT NULL
        );
        """
    )
    conn.executemany(
        "INSERT INTO sessions(session_id, origin, sort_key_ms, message_count) VALUES (?, ?, ?, ?)",
        [
            ("s1", "codex-session", 1_700_000_000_000, 2),
            ("s2", "claude-code-session", 1_700_000_100_000, 2),
            ("s3", "local", 1_700_000_200_000, 2),
        ],
    )
    conn.executemany("INSERT INTO messages(message_id) VALUES (?)", [("m1",), ("m2",)])
    conn.executemany(
        """
        INSERT INTO session_model_usage(
            session_id, model_name, input_tokens, output_tokens, cache_read_tokens,
            cache_write_tokens, cost_usd, cost_provenance
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("s1", "gpt-5-codex", 1_000_000, 100_000, 2_000_000, 0, None, "origin_reported"),
            ("s2", "claude-sonnet-4-6", 1_000_000, 100_000, 2_000_000, 0, 3.0, "priced"),
            ("s3", "unknown-local-model", 1_000, 10, 0, 0, None, "origin_reported"),
        ],
    )
    return conn


def test_classify_failed_followup_uses_only_explicit_acknowledgment_markers() -> None:
    assert agent_forensics._classify_failed_followup(None) == "ambiguous"
    assert agent_forensics._classify_failed_followup("ok") == "ambiguous"
    assert agent_forensics._classify_failed_followup("The command failed with exit code 2.") == "acknowledged"
    assert agent_forensics._classify_failed_followup_evidence("The command failed with exit code 2.") == {
        "classification": "acknowledged",
        "reason": "explicit_acknowledgment_marker",
        "matched_marker": "failed",
    }
    assert (
        agent_forensics._classify_failed_followup("The argument parsing isn't working. Let me debug the input.")
        == "acknowledged"
    )
    assert agent_forensics._classify_failed_followup("Docs are out of sync from the rename.") == "acknowledged"
    assert agent_forensics._classify_failed_followup("GitHub still sees it as conflicting.") == "acknowledged"
    assert agent_forensics._classify_failed_followup("xtask build doesn't support --features.") == "acknowledged"
    assert agent_forensics._classify_failed_followup("The files are being modified by a formatter.") == "acknowledged"
    assert (
        agent_forensics._classify_failed_followup("The files appear to have been modified since I read them.")
        == "acknowledged"
    )
    assert agent_forensics._classify_failed_followup("The files were modified since the read.") == "acknowledged"
    assert agent_forensics._classify_failed_followup("Still the same Nix escaping issue.") == "acknowledged"
    assert agent_forensics._classify_failed_followup("That's a pre-existing bug in the parser.") == "acknowledged"
    assert agent_forensics._classify_failed_followup("The tables don't exist.") == "acknowledged"
    assert agent_forensics._classify_failed_followup("I need to fix a couple of issues.") == "acknowledged"
    assert agent_forensics._classify_failed_followup("Property tests found some issues.") == "acknowledged"
    assert agent_forensics._classify_failed_followup("I need to fix some additional issues.") == "acknowledged"
    assert agent_forensics._classify_failed_followup("Let me fix the unclosed function body.") == "acknowledged"
    assert agent_forensics._classify_failed_followup("The tests need a clean database.") == "acknowledged"
    assert agent_forensics._classify_failed_followup("cargo nextest still fails.") == "acknowledged"
    assert agent_forensics._classify_failed_followup("The tests seem to be hanging.") == "acknowledged"
    assert agent_forensics._classify_failed_followup("Let me remove the duplicate updates.") == "acknowledged"
    assert agent_forensics._classify_failed_followup("The global hook blocks force pushes.") == "acknowledged"
    assert (
        agent_forensics._classify_failed_followup("Let me search for all instances and fix them properly.")
        == "silent_proceed"
    )
    assert agent_forensics._classify_failed_followup("cargo clippy now runs without issues.") == "silent_proceed"
    assert agent_forensics._classify_failed_followup("The shim now blocks Hyprland protocol announcements.") == (
        "silent_proceed"
    )
    assert agent_forensics._classify_failed_followup("Scratch files are gitignored; staging can proceed.") == (
        "silent_proceed"
    )
    assert (
        agent_forensics._classify_failed_followup("I will continue by inspecting the next module now.")
        == "silent_proceed"
    )


def test_structured_failure_followups_aggregates_ref_backed_samples() -> None:
    conn = _build_failure_followup_db()
    result = agent_forensics._structured_failure_followups(conn, sample_limit=3)
    by_tool = cast(list[dict[str, Any]], result["by_tool"])
    by_model = cast(list[dict[str, Any]], result["by_model"])
    samples = cast(list[dict[str, Any]], result["samples"])
    stratified = cast(dict[str, list[dict[str, Any]]], result["samples_by_classification"])

    assert result["totals"] == {
        "failed_outcomes": 4,
        "acknowledged": 1,
        "silent_proceed": 1,
        "ambiguous": 2,
        "classified_outcomes": 2,
    }
    assert by_tool[0] == {
        "name": "Bash",
        "failed_outcomes": 2,
        "acknowledged": 1,
        "silent_proceed": 1,
        "ambiguous": 0,
        "classified_outcomes": 2,
        "silent_rate": 0.5,
        "silent_rate_among_classified": 0.5,
    }
    assert by_model[0]["name"] == "claude-haiku"
    assert by_model[0]["failed_outcomes"] == 2
    assert by_model[0]["classified_outcomes"] == 1
    assert by_model[0]["silent_rate"] == 0.5
    assert by_model[0]["silent_rate_among_classified"] == 1.0
    assert len(samples) == 3
    assert samples[0]["tool_message_ref"] == "message:tool-ack"
    assert samples[0]["next_message_ref"] == "message:next-ack"
    assert [item["classification"] for item in stratified["acknowledged"]] == ["acknowledged"]
    assert [item["classification"] for item in stratified["silent_proceed"]] == ["silent_proceed"]
    assert [item["classification"] for item in stratified["ambiguous"]] == ["ambiguous", "ambiguous"]


def test_structured_failure_followups_can_bound_failed_outcomes() -> None:
    conn = _build_failure_followup_db()

    result = agent_forensics._structured_failure_followups(conn, failed_outcome_limit=2)

    assert result["failed_outcome_limit"] == 2
    totals = cast(dict[str, int], result["totals"])
    assert totals["failed_outcomes"] == 2
    assert totals["acknowledged"] + totals["silent_proceed"] + totals["ambiguous"] == 2
    assert totals["classified_outcomes"] == totals["acknowledged"] + totals["silent_proceed"]


def test_analyze_prices_origin_reported_rows_with_catalog_without_changing_provenance() -> None:
    conn = _build_usage_db()

    result = agent_forensics.analyze(conn)

    economy = cast(dict[str, dict[str, int | float]], result["economy_by_provenance"])
    origin = economy["origin_reported"]
    priced = economy["priced"]

    assert origin["stored_cost"] == 0.0
    assert float(origin["catalog_cost"]) > 0.0
    assert int(origin["catalog_priced_rows"]) == 1
    assert int(origin["catalog_unpriced_rows"]) == 1
    assert priced["stored_cost"] == 3.0
    assert float(result["catalog_api_equivalent_usd"]) > float(result["stored_cost_usd"])
    assert result["catalog_price_missing_reasons"] == {"missing_price": 1}


def test_report_labels_catalog_estimates_separately_from_stored_cost(tmp_path: Path) -> None:
    conn = _build_usage_db()
    findings = agent_forensics.analyze(conn)

    report = agent_forensics.build_report(findings, tmp_path, archive_label="synthetic")

    assert "Catalog API-equivalent cost" in report
    assert "Stored/provider-priced subset" in report
    assert "| provenance | input | output | cache read | cache write | stored cost | catalog API-equivalent |" in report
    assert "Provider-reported token rows remain `origin_reported`" in report
    assert "origin_reported** rows (other providers report token counts but no per-token cost)" not in report

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest

from devtools.claim_vs_evidence import build_report
from polylogue.demo import seed_demo_archive


def _report_args(
    *,
    archive_root: Path,
    out_dir: Path | None,
    limit: int,
    sample_limit: int,
    n_min: int = 1,
    calibration_size: int = 3,
    calibration_seed: int = 7,
    calibration_labels: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        archive_root=archive_root,
        out_dir=out_dir,
        limit=limit,
        sample_limit=sample_limit,
        n_min=n_min,
        calibration_size=calibration_size,
        calibration_seed=calibration_seed,
        calibration_labels=calibration_labels,
        json=False,
    )


def _seed_archive(root: Path) -> None:
    root.mkdir(parents=True)
    conn = sqlite3.connect(root / "index.db")
    conn.executescript(
        """
        PRAGMA user_version=22;
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            origin TEXT NOT NULL,
            title TEXT,
            created_at_ms INTEGER,
            updated_at_ms INTEGER
        );
        CREATE TABLE messages (
            session_id TEXT NOT NULL,
            message_id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            position INTEGER NOT NULL,
            model_name TEXT
        );
        CREATE TABLE blocks (
            block_id TEXT GENERATED ALWAYS AS (message_id || ':' || position) STORED UNIQUE,
            message_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            position INTEGER NOT NULL,
            block_type TEXT NOT NULL,
            text TEXT,
            tool_name TEXT,
            tool_id TEXT,
            tool_input TEXT,
            semantic_type TEXT,
            tool_result_is_error INTEGER,
            tool_result_exit_code INTEGER,
            tool_command TEXT GENERATED ALWAYS AS (json_extract(tool_input, '$.command')) VIRTUAL,
            tool_path TEXT GENERATED ALWAYS AS (
                COALESCE(json_extract(tool_input, '$.file_path'), json_extract(tool_input, '$.path'))
            ) VIRTUAL,
            PRIMARY KEY(message_id, position)
        );
        CREATE INDEX idx_blocks_type ON blocks(block_type);
        CREATE INDEX idx_blocks_tool_result_outcome
        ON blocks(block_type, tool_result_is_error, tool_result_exit_code, session_id, tool_id, message_id)
        WHERE block_type = 'tool_result';
        CREATE INDEX idx_blocks_tool_id ON blocks(tool_id) WHERE tool_id IS NOT NULL;
        CREATE INDEX idx_messages_session_position ON messages(session_id, position);
        CREATE VIEW actions AS
        SELECT
            u.session_id,
            u.message_id,
            u.block_id AS tool_use_block_id,
            u.tool_name,
            u.semantic_type,
            u.tool_command,
            u.tool_path,
            u.tool_input,
            r.text AS output_text,
            r.tool_result_is_error AS is_error,
            r.tool_result_exit_code AS exit_code,
            r.block_id AS tool_result_block_id
        FROM blocks u
        LEFT JOIN blocks r
            ON r.tool_id = u.tool_id
           AND r.session_id = u.session_id
           AND r.block_type = 'tool_result'
        WHERE u.block_type = 'tool_use';
        """
    )
    conn.executemany(
        "INSERT INTO sessions(session_id, origin, title, created_at_ms, updated_at_ms) VALUES (?, ?, ?, ?, ?)",
        [
            ("s1", "claude-code-session", "fixture one", 1, 4),
            ("s2", "codex-session", "fixture two", 1, 1),
        ],
    )
    conn.executemany(
        "INSERT INTO messages(session_id, message_id, role, position, model_name) VALUES (?, ?, ?, ?, ?)",
        [
            ("s1", "tool-ack", "tool", 1, "claude-opus"),
            ("s1", "next-ack", "assistant", 2, "claude-sonnet"),
            ("s1", "tool-silent", "tool", 3, "claude-opus"),
            ("s1", "next-silent", "assistant", 4, "claude-haiku"),
            ("s1", "next-silent-ack", "assistant", 5, "claude-haiku"),
            ("s2", "tool-missing-next", "tool", 1, "codex"),
            ("s2", "next-prose", "assistant", 2, "codex"),
            ("s2", "tool-wordless", "tool", 3, "codex"),
            ("s2", "next-wordless", "assistant", 4, "codex"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO blocks(
            message_id, session_id, position, block_type, text, tool_name, tool_id,
            tool_input, tool_result_is_error, tool_result_exit_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("tool-ack", "s1", 0, "tool_use", None, "Bash", "t1", '{"command":"pytest"}', None, None),
            ("tool-ack", "s1", 1, "tool_result", "failed", None, "t1", None, 1, None),
            (
                "next-ack",
                "s1",
                0,
                "text",
                "The command failed with exit code 2, so I will fix it.",
                None,
                None,
                None,
                None,
                None,
            ),
            ("tool-silent", "s1", 0, "tool_use", None, "Bash", "t2", '{"command":"ls missing"}', None, None),
            ("tool-silent", "s1", 1, "tool_result", "missing", None, "t2", None, 0, 2),
            (
                "next-silent",
                "s1",
                0,
                "text",
                "I will continue by inspecting the neighboring module now.",
                None,
                None,
                None,
                None,
                None,
            ),
            (
                "next-silent-ack",
                "s1",
                0,
                "text",
                "The ls command failed; I will switch to a different path.",
                None,
                None,
                None,
                None,
                None,
            ),
            ("tool-missing-next", "s2", 0, "tool_use", None, "Read", "t3", '{"path":"x"}', None, None),
            ("tool-missing-next", "s2", 1, "tool_result", "nope", None, "t3", None, 0, 1),
            (
                "next-prose",
                "s2",
                0,
                "text",
                "Ok.",
                None,
                None,
                None,
                None,
                None,
            ),
            ("tool-wordless", "s2", 0, "tool_use", None, "Read", "t4", '{"path":"y"}', None, None),
            ("tool-wordless", "s2", 1, "tool_result", "nope again", None, "t4", None, 0, 1),
            ("next-wordless", "s2", 0, "tool_use", None, "Read", "t5", '{"path":"z"}', None, None),
        ],
    )
    conn.commit()
    conn.close()


def test_claim_vs_evidence_builds_bounded_artifacts(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    out_dir = tmp_path / "out"
    _seed_archive(archive)
    out_dir.mkdir()
    (out_dir / "ack-marker-calibration.labels.csv").write_text(
        "\n".join(
            [
                "sample_id,human_label,classification,classification_reason,matched_marker,origin,model_name,"
                "tool_name,handler_class,session_ref,tool_result_message_ref,next_message_ref,next_text_preview,"
                "next3_classification,next3_matched_marker,next3_text_preview",
                "cal-001,acknowledged,acknowledged,explicit_acknowledgment_marker,failed,claude-code-session,"
                "claude-sonnet,Bash,consequential,session:s1,message:tool-ack,message:next-ack,"
                "The command failed,acknowledged,failed,The command failed",
                "cal-002,acknowledged,silent_proceed,no_acknowledgment_marker,,claude-code-session,"
                "claude-haiku,Bash,consequential,session:s1,message:tool-silent,message:next-silent,"
                "I will continue,acknowledged,failed,The ls command failed",
                "",
            ]
        ),
        encoding="utf-8",
    )

    report = build_report(
        _report_args(
            archive_root=archive,
            out_dir=out_dir,
            limit=4,
            sample_limit=2,
        )
    )

    assert report["index_schema_version"] == 22
    assert report["sample_frame"] == {
        "classification_scope": "immediately following assistant message only",
        "complete_failure_frame": True,
        "failure_predicate": "tool_result_is_error = 1 OR tool_result_exit_code != 0",
        "inspected_structured_failures": 4,
        "limit": 4,
        "n_min": 1,
        "time_window": "entire archive (no since/until filter)",
        "sampled_by_origin": [
            {
                "inspected_structured_failures": 2,
                "origin": "claude-code-session",
                "requested_limit": 2,
                "total_structured_failures": 2,
            },
            {
                "inspected_structured_failures": 2,
                "origin": "codex-session",
                "requested_limit": 2,
                "total_structured_failures": 2,
            },
        ],
        "selection_order": "origin, session_id, tool_id, tool_result_message_id",
        "selection_strategy": (
            "origin-stratified bounded sample; at least one row per origin when limit allows, "
            "then proportional fill by origin failure count; each origin candidate frame is bounded "
            "before pairing to tool-use rows"
        ),
        "sensitivity_scope": "next 3 assistant messages after the failed result, stopping before the next user message",
        "thin_cell_policy": (
            "Split cells below n_min are retained for coverage accounting but publish no rates: "
            "coverage_status=insufficient_n and publication_status=not_supported."
        ),
        "total_by_origin": [
            {"failed_outcomes": 2, "origin": "claude-code-session"},
            {"failed_outcomes": 2, "origin": "codex-session"},
        ],
        "total_structured_failures": 4,
        "unpaired_structured_failures": 0,
    }
    assert report["totals"] == {
        "failed_outcomes": 4,
        "acknowledged": 1,
        "silent_proceed": 1,
        "ambiguous": 2,
        "ambiguous_wordless_continuation": 1,
        "ambiguous_prose_no_marker": 1,
        "classified_outcomes": 2,
    }
    assert report["window3_totals"] == {
        "failed_outcomes": 4,
        "acknowledged": 2,
        "silent_proceed": 0,
        "ambiguous": 2,
        "classified_outcomes": 2,
    }
    assert report["rates"]["silent_rate_lower_bound"] == 1 / 4
    assert report["rates"]["ack_later_within_3"] == 1
    assert report["rates"]["window3_silent_rate_lower_bound"] == 0
    assert report["calibration"]["sample_size"] == 3
    assert report["calibration"]["sample_seed"] == 7
    assert report["calibration"]["metrics"]["labeled_rows"] == 2
    assert report["calibration"]["metrics"]["ack_marker_precision"] == 1.0
    assert report["calibration"]["metrics"]["ack_marker_recall"] == 0.5
    assert report["by_handler_class"] == [
        {
            "name": "benign_recovery",
            "failed_outcomes": 2,
            "acknowledged": 0,
            "silent_proceed": 0,
            "ambiguous": 2,
            "ambiguous_wordless_continuation": 1,
            "ambiguous_prose_no_marker": 1,
            "classified_outcomes": 0,
            "n_min": 1,
            "coverage_status": "supported",
            "publication_status": "supported",
            "silent_rate_lower_bound": 0.0,
            "silent_rate_among_classified": None,
        },
        {
            "name": "consequential",
            "failed_outcomes": 2,
            "acknowledged": 1,
            "silent_proceed": 1,
            "ambiguous": 0,
            "ambiguous_wordless_continuation": 0,
            "ambiguous_prose_no_marker": 0,
            "classified_outcomes": 2,
            "n_min": 1,
            "coverage_status": "supported",
            "publication_status": "supported",
            "silent_rate_lower_bound": 0.5,
            "silent_rate_among_classified": 0.5,
        },
    ]
    assert set(report["samples_by_origin_classification"]) == {"claude-code-session", "codex-session"}
    assert report["samples_by_origin_classification"]["codex-session"]["ambiguous"][0]["origin"] == "codex-session"
    codex_ambiguous = report["samples_by_origin_classification"]["codex-session"]["ambiguous"]
    assert {sample["classification_reason"] for sample in codex_ambiguous} == {
        "prose_no_marker",
        "wordless_tool_continuation",
    }
    assert {sample["handler_class"] for sample in codex_ambiguous} == {"benign_recovery"}
    assert any(sample["next_has_tool_use"] for sample in codex_ambiguous)
    assert (
        report["samples_by_origin_classification"]["claude-code-session"]["acknowledged"][0]["next_text_preview"]
        == "The command failed with exit code 2, so I will fix it."
    )
    assert (
        report["samples_by_origin_classification"]["claude-code-session"]["acknowledged"][0]["classification_reason"]
        == "explicit_acknowledgment_marker"
    )
    silent_sample = report["samples_by_origin_classification"]["claude-code-session"]["silent_proceed"][0]
    assert silent_sample["next3_classification"] == "acknowledged"
    assert silent_sample["next3_matched_marker"] == "failed"
    assert silent_sample["next3_message_refs"] == ["message:next-silent", "message:next-silent-ack"]
    assert report["samples_by_origin_classification"]["claude-code-session"]["acknowledged"][0]["matched_marker"] == (
        "failed"
    )
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["claim"]
    assert summary["non_claim"]
    assert summary["proof_report"]["failed_outcomes"] == 4
    assert summary["proof_report"]["complete_failure_frame"] is True
    assert summary["proof_report"]["ambiguous_wordless_continuation"] == 1
    assert summary["proof_report"]["ambiguous_prose_no_marker"] == 1
    assert summary["proof_report"]["acknowledged_within_3"] == 2
    assert summary["proof_report"]["silent_proceed_within_3"] == 0
    assert summary["proof_report"]["ack_later_within_3"] == 1
    assert summary["proof_report"]["window3_silent_rate_lower_bound"] == 0
    assert summary["proof_report"]["calibration"] == {
        "sample_size": 3,
        "sample_seed": 7,
        "labeled_rows": 2,
        "ack_marker_precision": 1.0,
        "ack_marker_recall": 0.5,
    }
    assert summary["proof_report"]["by_handler_class"][0]["name"] == "benign_recovery"
    assert summary["proof_report"]["by_handler_class"][1]["coverage_status"] == "supported"
    assert summary["proof_report"]["by_handler_class"][1]["publication_status"] == "supported"
    assert summary["proof_report"]["by_handler_class"][1]["silent_rate_lower_bound"] == 0.5
    assert summary["proof_report"]["time_window"] == "entire archive (no since/until filter)"
    assert summary["proof_report"]["sampled_by_origin"] == [
        {
            "inspected_structured_failures": 2,
            "origin": "claude-code-session",
            "requested_limit": 2,
            "total_structured_failures": 2,
        },
        {
            "inspected_structured_failures": 2,
            "origin": "codex-session",
            "requested_limit": 2,
            "total_structured_failures": 2,
        },
    ]
    assert (out_dir / "claim-vs-evidence.report.json").exists()
    calibration_sample = (out_dir / "ack-marker-calibration.sample.csv").read_text()
    assert "sample_id,human_label,classification" in calibration_sample
    assert "acknowledged" in calibration_sample
    public_summary = json.loads((out_dir / "public-summary.json").read_text())
    assert public_summary["claim"].startswith("Polylogue can ground")
    assert "private archive" in public_summary["non_claim"]
    assert public_summary["proofs"][0]["total_structured_failures"] == 4
    assert public_summary["proofs"][2]["ack_marker_precision"] == 1.0
    assert "samples_by_classification" not in public_summary
    assert "calibration_sample" not in public_summary
    assert "next_text_preview" not in json.dumps(public_summary)
    public_reproduction = (out_dir / "PUBLIC_REPRODUCTION.md").read_text()
    assert "polylogue demo seed" in public_reproduction
    assert "actions where is_error:true | group by followup_class | count" in public_reproduction
    assert "polylogue --plain --format json actions where is_error:true" in public_reproduction
    assert "devtools workspace claim-vs-evidence" in public_reproduction
    assert "reproduces the method and artifact shape" in public_reproduction
    cold_reader_gate = (out_dir / "COLD_READER_GATE.md").read_text()
    assert "Expected Passing Answer" in cold_reader_gate
    assert "no private transcript previews" in cold_reader_gate
    readme = (out_dir / "README.md").read_text()
    assert "Claim-vs-Evidence" in readme
    assert "- time window: entire archive (no since/until filter)" in readme
    assert "### Handler-Class Split" in readme
    assert "- consequential: failed 2; silent 1; ambiguous 0; silent lower bound 50.0%" in readme
    assert "- acknowledgments appearing only after the next turn: 1" in readme
    assert "- silent lower bound after next-3 sensitivity: 0.0%" in readme
    assert "### Marker Calibration" in readme
    assert "- calibration sample size: 3" in readme
    assert "- acknowledged-marker precision: 100.0%" in readme
    assert "- acknowledged-marker recall: 50.0%" in readme
    assert "`public-summary.json`" in readme
    assert "`PUBLIC_REPRODUCTION.md`" in readme
    assert "`COLD_READER_GATE.md`" in readme
    assert "- claude-code-session: inspected 2 / 2 structured failures (requested 2)" in readme
    assert "- codex-session: inspected 2 / 2 structured failures (requested 2)" in readme


def test_claim_vs_evidence_bounded_sample_is_origin_stratified(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    _seed_archive(archive)

    report = build_report(
        _report_args(
            archive_root=archive,
            out_dir=None,
            limit=2,
            sample_limit=2,
        )
    )

    assert report["sample_frame"]["complete_failure_frame"] is False
    assert report["sample_frame"]["sampled_by_origin"] == [
        {
            "inspected_structured_failures": 1,
            "origin": "claude-code-session",
            "requested_limit": 1,
            "total_structured_failures": 2,
        },
        {
            "inspected_structured_failures": 1,
            "origin": "codex-session",
            "requested_limit": 1,
            "total_structured_failures": 2,
        },
    ]
    assert {row["name"] for row in report["by_origin"]} == {"claude-code-session", "codex-session"}


def test_claim_vs_evidence_refuses_rates_for_cells_below_n_min(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    _seed_archive(archive)

    report = build_report(
        _report_args(
            archive_root=archive,
            out_dir=None,
            limit=4,
            sample_limit=2,
            n_min=3,
        )
    )

    assert report["sample_frame"]["n_min"] == 3
    assert "publication_status=not_supported" in report["sample_frame"]["thin_cell_policy"]
    by_model = {str(row["name"]): row for row in report["by_model"]}
    thin = by_model["claude-haiku"]
    assert thin["failed_outcomes"] == 1
    assert thin["coverage_status"] == "insufficient_n"
    assert thin["publication_status"] == "not_supported"
    assert thin["silent_rate_lower_bound"] is None
    assert thin["silent_rate_among_classified"] is None
    assert report["rates"]["coverage_status"] == "supported"

    at_threshold = build_report(
        _report_args(
            archive_root=archive,
            out_dir=None,
            limit=4,
            sample_limit=2,
            n_min=2,
        )
    )
    by_origin = {str(row["name"]): row for row in at_threshold["by_origin"]}
    supported = by_origin["claude-code-session"]
    assert supported["failed_outcomes"] == 2
    assert supported["coverage_status"] == "supported"
    assert supported["publication_status"] == "supported"
    assert supported["silent_rate_lower_bound"] == 0.5

    assert at_threshold["rates"]["coverage_status"] == "supported"
    assert at_threshold["rates"]["silent_rate_lower_bound"] == 0.25


def test_claim_vs_evidence_refuses_aggregate_rates_below_n_min(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    _seed_archive(archive)

    report = build_report(
        _report_args(
            archive_root=archive,
            out_dir=None,
            limit=4,
            sample_limit=2,
            n_min=5,
        )
    )

    assert report["rates"]["coverage_status"] == "insufficient_n"
    assert report["rates"]["publication_status"] == "not_supported"
    assert report["rates"]["silent_rate_lower_bound"] is None
    assert report["rates"]["window3_silent_rate_lower_bound"] is None


def test_claim_vs_evidence_public_reproduction_handles_unlabeled_sample(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    out_dir = tmp_path / "out"
    _seed_archive(archive)

    report = build_report(
        _report_args(
            archive_root=archive,
            out_dir=out_dir,
            limit=4,
            sample_limit=2,
        )
    )

    assert report["calibration"]["metrics"]["labeled_rows"] == 0
    public_reproduction = (out_dir / "PUBLIC_REPRODUCTION.md").read_text()
    assert "- acknowledged-marker precision: not enough labels" in public_reproduction
    assert "- acknowledged-marker recall: not enough labels" in public_reproduction


@pytest.mark.asyncio
async def test_claim_vs_evidence_seeded_demo_reproduces_method(tmp_path: Path) -> None:
    archive = tmp_path / "demo-archive"
    out_dir = tmp_path / "demo-report"

    await seed_demo_archive(archive, force=True, with_overlays=True)
    report = build_report(
        _report_args(
            archive_root=archive,
            out_dir=out_dir,
            limit=5000,
            sample_limit=10,
            calibration_size=10,
        )
    )

    assert report["sample_frame"]["total_structured_failures"] == 7
    assert report["sample_frame"]["inspected_structured_failures"] == 7
    assert report["totals"]["acknowledged"] == 3
    assert report["totals"]["silent_proceed"] == 4
    assert report["totals"]["ambiguous"] == 0
    public_summary = json.loads((out_dir / "public-summary.json").read_text())
    assert public_summary["proofs"][0]["total_structured_failures"] == 7
    public_reproduction = (out_dir / "PUBLIC_REPRODUCTION.md").read_text()
    assert "Counts will differ because the deterministic demo archive is synthetic." in public_reproduction


def test_claim_vs_evidence_keeps_same_message_tool_result_identities(tmp_path: Path) -> None:
    archive = tmp_path / "archive"
    _seed_archive(archive)
    conn = sqlite3.connect(archive / "index.db")
    conn.executemany(
        """
        INSERT INTO blocks(
            message_id, session_id, position, block_type, text, tool_name, tool_id,
            tool_input, tool_result_is_error, tool_result_exit_code
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("tool-missing-next", "s2", 2, "tool_use", None, "Read", "t6", '{"path":"y"}', None, None),
            ("tool-missing-next", "s2", 3, "tool_result", "also nope", None, "t6", None, 0, 1),
        ],
    )
    conn.commit()
    conn.close()

    report = build_report(
        _report_args(
            archive_root=archive,
            out_dir=None,
            limit=10,
            sample_limit=10,
        )
    )

    assert report["sample_frame"]["total_structured_failures"] == 5
    assert {row["origin"]: row["failed_outcomes"] for row in report["sample_frame"]["total_by_origin"]} == {
        "claude-code-session": 2,
        "codex-session": 3,
    }
    codex_samples = report["samples_by_origin_classification"]["codex-session"]["ambiguous"]
    assert len(codex_samples) == 3
    assert {sample["tool_result_tool_id"] for sample in codex_samples} == {"t3", "t4", "t6"}
    assert {sample["tool_result_message_ref"] for sample in codex_samples} == {
        "message:tool-missing-next",
        "message:tool-wordless",
    }

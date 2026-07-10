from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.demo.models import DemoSeedResult, DemoTourResult, DemoTourStep, DemoVerifyResult
from polylogue.scenarios import (
    DEMO_CLAUDE_CODE_SESSION_ID,
    DEMO_CODEX_LINEAGE_FORK_SESSION_ID,
    DEMO_CODEX_RECEIPTS_SESSION_ID,
    DEMO_SESSION_IDS,
)


def test_demo_seed_and_verify_json_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    runner = CliRunner()

    seed = runner.invoke(cli, ["demo", "seed", "--with-overlays", "--format", "json"])
    assert seed.exit_code == 0, seed.output
    seed_payload = json.loads(seed.output)
    assert seed_payload["session_count"] == len(DEMO_SESSION_IDS)
    assert seed_payload["message_count"] >= 35
    assert seed_payload["overlays_seeded"] is True
    assert seed_payload["construct_coverage"]
    assert all(row["ok"] for row in seed_payload["construct_coverage"])

    verify = runner.invoke(cli, ["demo", "verify", "--require-overlays", "--format", "json"])
    assert verify.exit_code == 0, verify.output
    verify_payload = json.loads(verify.output)
    assert verify_payload["ok"] is True
    assert DEMO_CLAUDE_CODE_SESSION_ID in verify_payload["query_hits"]
    assert verify_payload["absolute_path_leaks"] == []
    assert verify_payload["construct_coverage"]
    assert all(row["ok"] for row in verify_payload["construct_coverage"])


def test_demo_receipts_compares_claim_with_structural_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    runner = CliRunner()

    seed = runner.invoke(cli, ["demo", "seed", "--format", "json"])
    assert seed.exit_code == 0, seed.output

    result = runner.invoke(cli, ["demo", "receipts", "--format", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)

    assert payload["session_ref"] == f"session:{DEMO_CODEX_RECEIPTS_SESSION_ID}"
    assert payload["verdict"] == "contradicted_at_claim_time_then_repaired"
    assert payload["failed"]["tool_name"] == "exec_command"
    assert payload["failed"]["semantic_type"] == "shell"
    assert payload["failed"]["exit_code"] == 1
    assert payload["failed"]["is_error"] is True
    assert payload["recovery"]["exit_code"] == 0
    assert payload["recovery"]["is_error"] is False
    assert payload["anti_grep_control"]["text_hits_for_error"] >= 1
    assert payload["anti_grep_control"]["failed_actions"] == 0
    assert payload["source_material"]["raw_id"]
    assert len(payload["source_material"]["blob_sha256"]) == 64
    assert payload["claim_ref"].startswith("block:")
    assert payload["failed"]["block_ref"].startswith("block:")


def test_demo_receipts_is_self_contained_without_configured_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POLYLOGUE_ARCHIVE_ROOT", raising=False)
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(cli, ["demo", "receipts", "--format", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["seeded_for_command"] is True
    assert payload["verdict"] == "contradicted_at_claim_time_then_repaired"
    assert Path(payload["archive_root"]) == tmp_path / "polylogue-receipts-demo" / "archive"
    assert (tmp_path / "polylogue-receipts-demo" / "archive" / "index.db").exists()


def test_demo_receipts_uses_configured_archive_without_reseeding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(archive))
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    runner = CliRunner()
    seed = runner.invoke(cli, ["demo", "seed", "--format", "json"])
    assert seed.exit_code == 0, seed.output
    index_mtime = (archive / "index.db").stat().st_mtime_ns

    result = runner.invoke(cli, ["demo", "receipts", "--format", "json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["seeded_for_command"] is False
    assert (archive / "index.db").stat().st_mtime_ns == index_mtime


def test_demo_script_prints_copy_pastable_commands(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["demo", "script", "--root", str(tmp_path / "archive")])

    assert result.exit_code == 0, result.output
    assert "POLYLOGUE_ARCHIVE_ROOT" in result.output
    assert "polylogue demo tour" in result.output
    assert "polylogue demo seed" in result.output
    assert "polylogue demo verify" in result.output
    assert "--with-overlays --format json" in result.output
    assert "--require-overlays --format json" in result.output
    assert str(tmp_path / "archive") in result.output


def test_demo_script_seed_and_verify_commands_are_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_root = tmp_path / "archive"
    runner = CliRunner()
    script = runner.invoke(cli, ["demo", "script", "--root", str(archive_root)])
    assert script.exit_code == 0, script.output

    exports: dict[str, str] = {}
    demo_commands: list[list[str]] = []
    for line in script.output.splitlines():
        if line.startswith("export "):
            name, value = line.removeprefix("export ").split("=", maxsplit=1)
            exports[name] = shlex.split(value)[0]
            monkeypatch.setenv(name, exports[name])
        elif line.startswith("polylogue demo "):
            expanded = os.path.expandvars(line)
            demo_commands.append(shlex.split(expanded)[1:])

    assert [command[:2] for command in demo_commands] == [["demo", "tour"], ["demo", "seed"], ["demo", "verify"]]
    assert exports["POLYLOGUE_ARCHIVE_ROOT"] == str(archive_root)

    seed = runner.invoke(cli, demo_commands[1])
    assert seed.exit_code == 0, seed.output
    seed_payload = json.loads(seed.output)
    assert seed_payload["session_count"] == len(DEMO_SESSION_IDS)
    assert seed_payload["message_count"] >= 35
    assert seed_payload["overlays_seeded"] is True
    assert all(row["ok"] for row in seed_payload["construct_coverage"])

    verify = runner.invoke(cli, demo_commands[2])
    assert verify.exit_code == 0, verify.output
    verify_payload = json.loads(verify.output)
    assert verify_payload["ok"] is True
    assert verify_payload["overlays_present"] is True
    assert DEMO_CLAUDE_CODE_SESSION_ID in verify_payload["query_hits"]
    assert verify_payload["absolute_path_leaks"] == []
    assert all(row["ok"] for row in verify_payload["construct_coverage"])


def test_demo_tour_writes_report_transcript_and_recording(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLYLOGUE_FORCE_PLAIN", "1")
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "demo",
            "tour",
            "--out-dir",
            str(tmp_path / "tour"),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["first_result_s"] <= 30
    assert payload["total_duration_s"] <= 420
    assert payload["seed"]["session_count"] == len(DEMO_SESSION_IDS)
    assert payload["verify"]["ok"] is True
    assert [step["name"] for step in payload["steps"]] == [
        "claim versus receipt",
        "failed actions aggregate",
        "composed lineage",
        "archive facets",
    ]
    assert all(step["exit_code"] == 0 for step in payload["steps"])

    report = Path(str(payload["report_markdown_path"]))
    transcript = Path(str(payload["transcript_path"]))
    recording = Path(str(payload["recording_tape_path"]))
    assert report.exists()
    assert transcript.exists()
    assert recording.exists()
    report_text = report.read_text(encoding="utf-8")
    assert "Polylogue Demo Tour Report" in report_text
    assert "What this tour proves" in report_text
    assert "What this tour does not prove" in report_text
    transcript_text = transcript.read_text(encoding="utf-8")
    assert "$ polylogue demo receipts" in transcript_text
    assert "contradicted_at_claim_time_then_repaired" in transcript_text
    assert "$ polylogue 'actions where is_error:true | group by tool | count'" in transcript_text
    assert f"$ polylogue --id {DEMO_CODEX_LINEAGE_FORK_SESSION_ID} read --view chronicle" in transcript_text
    assert "$ polylogue analyze --facets" in transcript_text
    assert "construct_coverage" not in transcript_text
    assert str(tmp_path.resolve()) not in transcript_text
    assert "archive: <demo-archive>" in transcript_text
    recording_text = recording.read_text(encoding="utf-8")
    # The tape must reproduce the run it sits next to: commands reference the
    # actual out-dir basename, never a machine-local absolute path.
    assert "polylogue demo tour --out-dir tour" in recording_text
    assert "cat tour/transcript.txt" in recording_text
    assert "cat tour/report.md" in recording_text
    assert str(tmp_path) not in recording_text


def test_demo_tour_plain_output_reports_artifacts(tmp_path: Path) -> None:
    runner = CliRunner()
    seed = DemoSeedResult(
        archive_root=tmp_path / "tour" / "archive",
        source_root=tmp_path / "tour" / "source",
        session_count=11,
        message_count=43,
        session_ids=("session-a",),
        overlays_seeded=True,
        assertion_count=4,
    )
    verify = DemoVerifyResult(
        archive_root=tmp_path / "tour" / "archive",
        ok=True,
        session_count=11,
        message_count=43,
        query_hits=("session-a",),
        overlays_present=True,
        absolute_path_leaks=(),
    )
    tour_result = DemoTourResult(
        archive_root=tmp_path / "tour" / "archive",
        output_dir=tmp_path / "tour",
        ok=True,
        first_result_s=3.2,
        total_duration_s=6.4,
        report_json_path=tmp_path / "tour" / "report.json",
        report_markdown_path=tmp_path / "tour" / "report.md",
        transcript_path=tmp_path / "tour" / "transcript.txt",
        recording_tape_path=tmp_path / "tour" / "recording.tape",
        seed=seed,
        verify=verify,
        steps=(
            DemoTourStep(
                name="archive facets",
                command=("polylogue", "analyze", "--facets"),
                exit_code=0,
                duration_s=1.0,
                output_path=tmp_path / "tour" / "command-output" / "01.txt",
                bytes_written=10,
            ),
        ),
    )

    with patch("polylogue.cli.commands.demo.run_demo_tour", return_value=tour_result) as run:
        result = runner.invoke(cli, ["demo", "tour", "--out-dir", str(tmp_path / "tour")])

    assert result.exit_code == 0, result.output
    assert "Polylogue demo tour: passed" in result.output
    assert "First result: 3.200s" in result.output
    assert "Report:" in result.output
    run.assert_called_once()

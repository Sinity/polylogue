from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from polylogue.cli.click_app import cli
from polylogue.demo.models import DemoSeedResult, DemoTourResult, DemoTourStep, DemoVerifyResult


def _verify_result(tmp_path: Path, *, ok: bool) -> DemoVerifyResult:
    archive_root = tmp_path / "archive"
    return DemoVerifyResult(
        archive_root=archive_root,
        ok=ok,
        session_count=1,
        message_count=4,
        query_hits=(),
        overlays_present=False,
        absolute_path_leaks=(),
        problems=() if ok else ("planted verification failure",),
    )


def _tour_result(tmp_path: Path, *, ok: bool) -> DemoTourResult:
    output_dir = tmp_path / "tour"
    archive_root = output_dir / "archive"
    seed = DemoSeedResult(
        archive_root=archive_root,
        source_root=output_dir / "source",
        session_count=1,
        message_count=4,
        session_ids=("hermes-session:demo-00@profile-deadbeef1234",),
        overlays_seeded=True,
        assertion_count=4,
    )
    verify = _verify_result(tmp_path, ok=ok)
    step = DemoTourStep(
        name="archive facets",
        command=("polylogue", "analyze", "--facets"),
        exit_code=0 if ok else 1,
        duration_s=0.1,
        output_path=output_dir / "command-output" / "01.txt",
        bytes_written=10,
    )
    return DemoTourResult(
        archive_root=archive_root,
        output_dir=output_dir,
        ok=ok,
        first_result_s=0.1,
        total_duration_s=0.2,
        report_json_path=output_dir / "report.json",
        report_markdown_path=output_dir / "report.md",
        transcript_path=output_dir / "transcript.txt",
        recording_tape_path=output_dir / "recording.tape",
        seed=seed,
        verify=verify,
        steps=(step,),
        problems=() if ok else ("planted tour failure",),
    )


def test_demo_verify_json_failure_is_one_document_with_nonzero_status(tmp_path: Path) -> None:
    runner = CliRunner()
    failed = _verify_result(tmp_path, ok=False)

    with patch("polylogue.cli.commands.demo.verify_demo_archive", return_value=failed):
        result = runner.invoke(cli, ["demo", "verify", "--root", str(tmp_path / "archive"), "--format", "json"])

    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["problems"] == ["planted verification failure"]


def test_demo_verify_success_preserves_json_and_human_output(tmp_path: Path) -> None:
    runner = CliRunner()
    passed = _verify_result(tmp_path, ok=True)

    with patch("polylogue.cli.commands.demo.verify_demo_archive", return_value=passed):
        json_result = runner.invoke(
            cli,
            ["demo", "verify", "--root", str(tmp_path / "archive"), "--format", "json"],
        )
        plain_result = runner.invoke(cli, ["demo", "verify", "--root", str(tmp_path / "archive")])

    assert json_result.exit_code == 0, json_result.output
    assert json.loads(json_result.output)["ok"] is True
    assert plain_result.exit_code == 0, plain_result.output
    assert "Demo archive verification: ok" in plain_result.output


def test_demo_tour_json_failure_is_one_document_with_nonzero_status(tmp_path: Path) -> None:
    runner = CliRunner()
    failed = _tour_result(tmp_path, ok=False)

    with patch("polylogue.cli.commands.demo.run_demo_tour", return_value=failed):
        result = runner.invoke(
            cli,
            ["demo", "tour", "--out-dir", str(tmp_path / "tour"), "--format", "json"],
        )

    assert result.exit_code != 0
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["problems"] == ["planted tour failure"]


def test_demo_tour_success_preserves_json_and_human_output(tmp_path: Path) -> None:
    runner = CliRunner()
    passed = _tour_result(tmp_path, ok=True)

    with patch("polylogue.cli.commands.demo.run_demo_tour", return_value=passed):
        json_result = runner.invoke(
            cli,
            ["demo", "tour", "--out-dir", str(tmp_path / "tour"), "--format", "json"],
        )
        plain_result = runner.invoke(cli, ["demo", "tour", "--out-dir", str(tmp_path / "tour")])

    assert json_result.exit_code == 0, json_result.output
    assert json.loads(json_result.output)["ok"] is True
    assert plain_result.exit_code == 0, plain_result.output
    assert "Polylogue demo tour: passed" in plain_result.output

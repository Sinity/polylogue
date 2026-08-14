from __future__ import annotations

from pathlib import Path

from polylogue.demo.models import DemoSeedResult, DemoTourResult, DemoTourStep, DemoVerifyResult
from polylogue.demo.tour import _render_report_markdown, _tour_report_payload


def _tour_result(tmp_path: Path, *, ok: bool = True) -> DemoTourResult:
    archive = tmp_path / "tour" / "archive"
    seed = DemoSeedResult(
        archive_root=archive,
        source_root=archive / "source",
        session_count=11,
        message_count=43,
        session_ids=("session-a",),
        overlays_seeded=True,
        assertion_count=5,
    )
    verify = DemoVerifyResult(
        archive_root=archive,
        ok=ok,
        session_count=11,
        message_count=43,
        query_hits=("session-a",),
        overlays_present=True,
        absolute_path_leaks=(),
        problems=() if ok else ("planted construct missing",),
    )
    step = DemoTourStep(
        name="archive facets",
        command=("polylogue", "analyze", "--facets"),
        exit_code=0 if ok else 1,
        duration_s=1.0,
        output_path=tmp_path / "tour" / "command-output" / "01.txt",
        bytes_written=20,
    )
    return DemoTourResult(
        archive_root=archive,
        output_dir=tmp_path / "tour",
        ok=ok,
        first_result_s=2.0,
        total_duration_s=4.0,
        report_json_path=tmp_path / "tour" / "report.json",
        report_markdown_path=tmp_path / "tour" / "report.md",
        transcript_path=tmp_path / "tour" / "transcript.txt",
        recording_tape_path=tmp_path / "tour" / "recording.tape",
        seed=seed,
        verify=verify,
        steps=(step,),
        problems=() if ok else ("demo archive verification failed",),
    )


def test_tour_report_contains_measured_result_and_no_self_attestation(tmp_path: Path) -> None:
    payload = _tour_report_payload(_tour_result(tmp_path))

    assert payload["ok"] is True
    assert payload["verify"]["ok"] is True  # type: ignore[index]
    assert payload["steps"][0]["exit_code"] == 0  # type: ignore[index]
    assert payload["problems"] == []
    assert "claim" not in payload
    assert "oracle" not in payload


def test_tour_report_preserves_failed_run_evidence(tmp_path: Path) -> None:
    payload = _tour_report_payload(_tour_result(tmp_path, ok=False))

    assert payload["ok"] is False
    assert payload["verify"]["ok"] is False  # type: ignore[index]
    assert payload["steps"][0]["exit_code"] == 1  # type: ignore[index]
    assert payload["problems"] == ["demo archive verification failed"]


def test_tour_markdown_renders_measured_run_values(tmp_path: Path) -> None:
    report = _render_report_markdown(_tour_result(tmp_path))

    assert "Status: **passed**" in report
    assert "First evidence result: 2.000s" in report
    assert "Full tour: 4.000s" in report
    assert "Sessions: 11" in report
    assert "Messages: 43" in report
    assert "| archive facets | 0 | 1.000s | 20 |" in report

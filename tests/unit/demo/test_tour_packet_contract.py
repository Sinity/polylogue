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


def test_tour_report_carries_v2_epistemic_fields(tmp_path: Path) -> None:
    payload = _tour_report_payload(_tour_result(tmp_path))

    assert payload["demo_packet_contract_version"] == "2.0.0"
    assert payload["primary_construct"]["id"] == "demo.public-tour"  # type: ignore[index]
    assert payload["claim"]["declared_before_execution"] is True  # type: ignore[index]
    assert payload["oracle"]["independent"] is True  # type: ignore[index]
    assert payload["baseline"]
    assert payload["controls"]
    assert payload["falsifier"]["triggered"] is False  # type: ignore[index]
    assert payload["non_claims"]


def test_tour_report_marks_a_failed_run_as_refuted(tmp_path: Path) -> None:
    payload = _tour_report_payload(_tour_result(tmp_path, ok=False))

    assert payload["claim"]["status"] == "refuted"  # type: ignore[index]
    assert payload["falsifier"]["triggered"] is True  # type: ignore[index]
    assert payload["falsifier"]["result"] == "fail"  # type: ignore[index]


def test_tour_markdown_keeps_non_claims_visible(tmp_path: Path) -> None:
    report = _render_report_markdown(_tour_result(tmp_path))

    assert "## Claim" in report
    assert "## Oracle" in report
    assert "## Non-claims" in report
    assert "private-archive Receipts benchmark" in report

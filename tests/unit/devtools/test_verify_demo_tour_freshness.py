from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools import verify_demo_tour_freshness


def test_mask_volatile_only_touches_duration_numbers() -> None:
    text = "Seeded 13 sessions in 3.497s. exit=0 duration=0.5s bytes=44 raw_id: abc123"
    masked = verify_demo_tour_freshness.mask_volatile(text)
    assert masked == "Seeded 13 sessions in <duration>. exit=0 duration=<duration> bytes=44 raw_id: abc123"


def test_mask_volatile_json_masks_only_the_declared_duration_keys() -> None:
    """Anti-vacuity: report.json's timing lives in numeric fields with no
    trailing 's' (`first_result_s`, `total_duration_s`, `steps[*].duration_s`),
    which the text-based `mask_volatile` regex cannot see. A real drift in
    any other field -- construct-coverage counts, exit codes, claim text --
    must still survive masking and be visible to the caller's diff."""
    payload = {
        "first_result_s": 2.072,
        "total_duration_s": 7.678,
        "ok": True,
        "steps": [
            {"name": "step one", "duration_s": 1.649, "exit_code": 0},
            {"name": "step two", "duration_s": 1.925, "exit_code": 0},
        ],
        "seed": {"session_count": 15},
    }
    masked = verify_demo_tour_freshness.mask_volatile_json(json.dumps(payload))
    reparsed = json.loads(masked)
    assert reparsed["first_result_s"] == "<duration>"
    assert reparsed["total_duration_s"] == "<duration>"
    assert [s["duration_s"] for s in reparsed["steps"]] == ["<duration>", "<duration>"]
    # Non-duration fields -- including nested structure -- must survive verbatim.
    assert reparsed["ok"] is True
    assert reparsed["seed"]["session_count"] == 15
    assert [s["exit_code"] for s in reparsed["steps"]] == [0, 0]


def test_mask_volatile_json_flags_a_real_construct_coverage_drift() -> None:
    """Seeded failure: a real content change (not a duration) in report.json
    must still produce a masked-text difference, proving the JSON-aware mask
    doesn't accidentally launder unrelated drift."""
    committed = json.dumps({"first_result_s": 2.0, "seed": {"session_count": 15}})
    fresh = json.dumps({"first_result_s": 2.9, "seed": {"session_count": 16}})
    assert verify_demo_tour_freshness.mask_volatile_json(committed) != verify_demo_tour_freshness.mask_volatile_json(
        fresh
    )


@pytest.mark.scale_small
def test_tour_freshness_diff_matches_committed_evidence(tmp_path: Path) -> None:
    """Anti-vacuity: exercises the real `run_demo_tour` production entrypoint
    against the real committed docs/examples/demo-tour/ fixture, with only
    the declared duration mask applied. A corpus change that alters session
    counts, construct coverage, or observed tool calls without regenerating
    the committed fixture (exactly the failure this bead exists to catch)
    makes this assertion fail."""
    out_dir = tmp_path / "polylogue-demo-tour"
    mismatches = verify_demo_tour_freshness.tour_freshness_diff(out_dir=out_dir)
    assert mismatches == {}


def test_tour_freshness_diff_flags_a_seeded_content_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seeded failure: a tour run that emits different content than committed
    must be reported per-file, not silently accepted."""
    committed_dir = tmp_path / "committed"
    (committed_dir / "command-output").mkdir(parents=True)
    (committed_dir / "transcript.txt").write_text("committed transcript\n", encoding="utf-8")
    (committed_dir / "report.md").write_text("committed report\n", encoding="utf-8")
    (committed_dir / "report.json").write_text(json.dumps({"total_duration_s": 7.678, "ok": True}), encoding="utf-8")
    (committed_dir / "recording.tape").write_text("committed tape\n", encoding="utf-8")
    (committed_dir / "command-output" / "01-step.txt").write_text("committed step\n", encoding="utf-8")
    monkeypatch.setattr(verify_demo_tour_freshness, "COMMITTED_DIR", committed_dir)

    def _fake_run_demo_tour(*, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "transcript.txt").write_text("fresh transcript DIFFERS\n", encoding="utf-8")
        (output_dir / "report.md").write_text("committed report\n", encoding="utf-8")
        # Only the duration differs -- proves report.json participates in the
        # comparison loop (polylogue-3tl.17 gap) without the volatile-field
        # mask laundering a real mismatch elsewhere.
        (output_dir / "report.json").write_text(json.dumps({"total_duration_s": 8.111, "ok": True}), encoding="utf-8")
        (output_dir / "recording.tape").write_text("committed tape\n", encoding="utf-8")
        (output_dir / "command-output").mkdir(parents=True, exist_ok=True)
        (output_dir / "command-output" / "01-step.txt").write_text("committed step\n", encoding="utf-8")

    monkeypatch.setattr("polylogue.demo.tour.run_demo_tour", _fake_run_demo_tour)

    out_dir = tmp_path / "out"
    mismatches = verify_demo_tour_freshness.tour_freshness_diff(out_dir=out_dir)

    assert set(mismatches) == {"transcript.txt"}
    committed_masked, fresh_masked = mismatches["transcript.txt"]
    assert committed_masked == "committed transcript\n"
    assert fresh_masked == "fresh transcript DIFFERS\n"


def test_tour_freshness_diff_flags_a_missing_fresh_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seeded failure: a committed artifact with no fresh counterpart at all
    (e.g. a step that stopped running) is reported as missing, not skipped."""
    committed_dir = tmp_path / "committed"
    committed_dir.mkdir()
    (committed_dir / "transcript.txt").write_text("committed transcript\n", encoding="utf-8")
    monkeypatch.setattr(verify_demo_tour_freshness, "COMMITTED_DIR", committed_dir)

    def _fake_run_demo_tour(*, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        # deliberately does not write transcript.txt

    monkeypatch.setattr("polylogue.demo.tour.run_demo_tour", _fake_run_demo_tour)

    out_dir = tmp_path / "out"
    mismatches = verify_demo_tour_freshness.tour_freshness_diff(out_dir=out_dir)

    assert mismatches["transcript.txt"] == ("committed transcript\n", "<MISSING>")

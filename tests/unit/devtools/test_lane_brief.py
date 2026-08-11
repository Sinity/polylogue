from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from devtools import click_dispatch, lane_brief


def _bd_show_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": "polylogue-a",
        "priority": 1,
        "issue_type": "task",
        "title": "Example bead",
        "description": "Touch devtools/lane_brief.py.",
        "design": "",
        "acceptance_criteria": "",
        "notes": "",
        "dependencies": [],
    }
    record.update(overrides)
    return record


def test_fetch_bead_parses_show_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: MagicMock(returncode=0, stdout=json.dumps([_bd_show_record()]), stderr=""),
    )
    record = lane_brief._fetch_bead("polylogue-a")
    assert record.found
    assert record.id == "polylogue-a"
    assert record.title == "Example bead"


@pytest.mark.parametrize(("acceptance_criteria", "expected_rc"), [("Observable result.", 0), ("", 2)])
def test_main_requires_ordinary_acceptance_criteria(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, acceptance_criteria: str, expected_rc: int
) -> None:
    monkeypatch.setattr(lane_brief, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        lane_brief,
        "_fetch_bead",
        lambda bead_id: lane_brief.BeadRecord(
            id=bead_id,
            found=True,
            acceptance_criteria=acceptance_criteria,
        ),
    )
    monkeypatch.setattr(lane_brief, "_load_bd_export", lambda repo_root, tmpdir: [])
    monkeypatch.setattr(lane_brief, "_verify_footprint", lambda repo_root, paths: [])
    monkeypatch.setattr(lane_brief, "_find_prior_art", lambda records, paths, exclude_ids: [])
    monkeypatch.setattr(lane_brief, "_recent_master_commits", lambda repo_root, paths, days: [])

    assert (
        lane_brief.main(["polylogue-a", "--out", str(tmp_path / "brief.md"), "--tmpdir", str(tmp_path)]) == expected_rc
    )


def test_lane_brief_published_route_forwards_recent_days_and_blocks_missing_ac(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, int] = {}
    monkeypatch.setattr(lane_brief, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        lane_brief,
        "_fetch_bead",
        lambda bead_id: lane_brief.BeadRecord(id=bead_id, found=True, title="CLI lane"),
    )
    monkeypatch.setattr(lane_brief, "_load_bd_export", lambda repo_root, tmpdir: [])
    monkeypatch.setattr(lane_brief, "_verify_footprint", lambda repo_root, paths: [])
    monkeypatch.setattr(lane_brief, "_find_prior_art", lambda records, paths, exclude_ids: [])
    monkeypatch.setattr(lane_brief, "_recent_master_commits", lambda repo_root, paths, days: [])

    def recent(repo_root: Path, paths: list[str], days: int) -> list[str]:
        seen["days"] = days
        return []

    monkeypatch.setattr(lane_brief, "_recent_master_commits", recent)
    rc = click_dispatch.main(
        [
            "workspace",
            "lane-brief",
            "polylogue-a",
            "--out",
            str(tmp_path / "brief.md"),
            "--tmpdir",
            str(tmp_path),
            "--recent-days",
            "3",
        ]
    )

    assert rc == 2
    assert seen["days"] == 3
    assert "no acceptance criteria" in (tmp_path / "brief.md").read_text()


def test_fetch_bead_reports_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: MagicMock(returncode=1, stdout="", stderr="bd: no such bead"),
    )
    record = lane_brief._fetch_bead("polylogue-missing")
    assert not record.found
    assert "bd: no such bead" in record.error


def test_fetch_bead_rejects_a_wrong_id_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Production dependency: lane-brief -> bd show adapter; catches accepting the first unintended record."""
    wrong = _bd_show_record(id="polylogue-unintended")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: MagicMock(returncode=0, stdout=json.dumps([wrong]), stderr=""),
    )

    fetched = lane_brief._fetch_bead("polylogue-requested")

    assert not fetched.found
    assert "does not match requested id 'polylogue-requested'" in fetched.error


def test_dep_labels_prefers_depends_on_id() -> None:
    assert lane_brief._dep_labels({"dependencies": [{"depends_on_id": "polylogue-b", "type": "blocks"}]}) == [
        "polylogue-b(blocks)"
    ]


def test_extract_paths_finds_project_file_references() -> None:
    text = "Touch polylogue/mcp/server.py and tests/unit/devtools/test_lane_brief.py please."
    paths = lane_brief._extract_paths(text)
    assert "polylogue/mcp/server.py" in paths
    assert "tests/unit/devtools/test_lane_brief.py" in paths


def test_verify_footprint_flags_missing_paths(tmp_path: Path) -> None:
    (tmp_path / "devtools").mkdir()
    real_file = tmp_path / "devtools" / "real.py"
    real_file.write_text("line1\nline2\nline3\n")

    evidence = lane_brief._verify_footprint(tmp_path, ["devtools/real.py", "devtools/ghost.py"])
    by_path = {e.path: e for e in evidence}

    assert by_path["devtools/real.py"].exists
    assert by_path["devtools/real.py"].line_count == 3
    assert not by_path["devtools/ghost.py"].exists


def test_find_prior_art_matches_closed_beads_sharing_a_path() -> None:
    export_records = [
        {
            "_type": "issue",
            "id": "polylogue-old",
            "status": "closed",
            "title": "Old fix",
            "description": "Fixed devtools/real.py",
            "close_reason": "Merged in #1234",
        },
        {
            "_type": "issue",
            "id": "polylogue-open-one",
            "status": "open",
            "title": "Still open",
            "description": "Touches devtools/real.py",
        },
    ]
    hits = lane_brief._find_prior_art(export_records, ["devtools/real.py"], exclude_ids=set())
    assert len(hits) == 1
    assert hits[0].id == "polylogue-old"
    assert "Merged in #1234" in hits[0].close_reason_head


def test_find_prior_art_excludes_beads_in_the_current_lane() -> None:
    export_records = [
        {
            "_type": "issue",
            "id": "polylogue-a",
            "status": "closed",
            "title": "Self",
            "description": "devtools/real.py",
        },
    ]
    hits = lane_brief._find_prior_art(export_records, ["devtools/real.py"], exclude_ids={"polylogue-a"})
    assert hits == []


def test_render_markdown_includes_all_mandatory_sections() -> None:
    record = lane_brief.BeadRecord(id="polylogue-a", found=True, priority=1, issue_type="task", title="T")
    md = lane_brief._render_markdown(["polylogue-a"], [record], [], [])
    for heading in (
        "## Scope",
        "## Footprint (verified)",
        "## Prior art",
        "## Recently merged on master (footprint overlap, last 7 days)",
        "## Measured baseline",
        "## Non-goals",
        "## Anti-vacuity contract",
        "## Hazards (standing)",
        "## Verification tier",
    ):
        assert heading in md
    assert "<!-- DISPATCHER MUST FILL -->" in md


def test_empty_acceptance_criteria_blocks_dispatch() -> None:
    record = lane_brief.BeadRecord(
        id="polylogue-a",
        found=True,
        priority=1,
        issue_type="task",
        title="Unspecified task",
    )
    md = lane_brief._render_markdown(["polylogue-a"], [record], [], [])
    assert "DISPATCH BLOCKED" in md


def test_render_markdown_reports_not_found_bead() -> None:
    record = lane_brief.BeadRecord(id="polylogue-missing", found=False, error="not found")
    md = lane_brief._render_markdown(["polylogue-missing"], [record], [], [])
    assert "polylogue-missing -- NOT FOUND" in md
    assert "not found" in md


def test_render_markdown_prior_satisfaction_warning_lists_recent_commits() -> None:
    record = lane_brief.BeadRecord(id="polylogue-a", found=True, priority=1, issue_type="task", title="T")
    md = lane_brief._render_markdown(
        ["polylogue-a"],
        [record],
        [],
        [],
        recent_commits=["abc1234 2026-08-01 feat: already did the thing (#3480)"],
    )
    assert "PRIOR-SATISFACTION CHECK" in md
    assert "abc1234 2026-08-01 feat: already did the thing (#3480)" in md


def test_recent_master_commits_finds_footprint_churn(tmp_path: Path) -> None:
    def _git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
        )

    _git("init", "-b", "master")
    target = tmp_path / "devtools" / "lane_brief.py"
    target.parent.mkdir()
    target.write_text("x = 1\n")
    _git("add", "devtools/lane_brief.py")
    _git("commit", "-m", "touch footprint file", "--no-gpg-sign")

    hits = lane_brief._recent_master_commits(tmp_path, ["devtools/lane_brief.py"], days=3650)
    assert len(hits) == 1
    assert "touch footprint file" in hits[0]
    assert lane_brief._recent_master_commits(tmp_path, [], days=3650) == []


def test_hazards_forbid_background_waits_and_worktree_bd_writes() -> None:
    joined = " ".join(lane_brief._HAZARDS)
    assert "foreground" in joined
    assert "polylogue-2ara" in joined


def test_main_writes_brief_to_out_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lane_brief, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        lane_brief,
        "_fetch_bead",
        lambda bead_id: lane_brief.BeadRecord(id=bead_id, found=True, acceptance_criteria="Observable result."),
    )
    monkeypatch.setattr(lane_brief, "_load_bd_export", lambda repo_root, tmpdir: [])

    out_path = tmp_path / "brief.md"
    rc = lane_brief.main(["polylogue-a", "--out", str(out_path), "--tmpdir", str(tmp_path)])

    assert rc == 0
    assert out_path.exists()
    assert "## Scope" in out_path.read_text()

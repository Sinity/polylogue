from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from devtools import lane_brief


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


def test_fetch_bead_reports_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: MagicMock(returncode=1, stdout="", stderr="bd: no such bead"),
    )
    record = lane_brief._fetch_bead("polylogue-missing")
    assert not record.found
    assert "bd: no such bead" in record.error


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
        "## Measured baseline",
        "## Non-goals",
        "## Anti-vacuity contract",
        "## Hazards (standing)",
        "## Verification tier",
    ):
        assert heading in md
    assert "<!-- DISPATCHER MUST FILL -->" in md


def test_render_markdown_reports_not_found_bead() -> None:
    record = lane_brief.BeadRecord(id="polylogue-missing", found=False, error="not found")
    md = lane_brief._render_markdown(["polylogue-missing"], [record], [], [])
    assert "polylogue-missing -- NOT FOUND" in md
    assert "not found" in md


def test_main_writes_brief_to_out_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(lane_brief, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(lane_brief, "_fetch_bead", lambda bead_id: lane_brief.BeadRecord(id=bead_id, found=True))
    monkeypatch.setattr(lane_brief, "_load_bd_export", lambda repo_root, tmpdir: [])

    out_path = tmp_path / "brief.md"
    rc = lane_brief.main(["polylogue-a", "--out", str(out_path), "--tmpdir", str(tmp_path)])

    assert rc == 0
    assert out_path.exists()
    assert "## Scope" in out_path.read_text()

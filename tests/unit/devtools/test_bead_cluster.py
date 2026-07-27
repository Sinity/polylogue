from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from devtools import bead_cluster


def _bead(
    bead_id: str,
    *,
    title: str = "untitled",
    priority: int = 2,
    status: str = "open",
    dependency_count: int = 0,
    labels: list[str] | None = None,
    design: str = "",
    notes: str = "",
    acceptance_criteria: str = "",
) -> dict[str, Any]:
    return {
        "id": bead_id,
        "title": title,
        "priority": priority,
        "status": status,
        "dependency_count": dependency_count,
        "labels": labels or [],
        "design": design,
        "notes": notes,
        "acceptance_criteria": acceptance_criteria,
    }


# ---------------------------------------------------------------------------
# Footprint extraction
# ---------------------------------------------------------------------------


def test_extract_footprint_finds_file_paths_and_packages() -> None:
    bead = _bead(
        "polylogue-a",
        design="Touch polylogue/mcp/server.py and tests/unit/mcp/test_server.py.",
    )
    fp = bead_cluster._extract_footprint(bead)
    assert "polylogue/mcp/server.py" in fp.files
    assert "tests/unit/mcp/test_server.py" in fp.files
    assert "polylogue/mcp/" in fp.packages
    assert "tests/unit/" in fp.packages


def test_extract_footprint_falls_back_to_area_labels_when_no_files() -> None:
    bead = _bead("polylogue-a", design="no file paths here", labels=["area:mcp"])
    fp = bead_cluster._extract_footprint(bead)
    assert "polylogue/mcp/" in fp.packages
    assert fp.files == []


def test_extract_footprint_area_labels_ignored_once_files_present() -> None:
    # Area labels should NOT be added once concrete files are found -- otherwise
    # broad area->package roots would over-merge clusters.
    bead = _bead(
        "polylogue-a",
        design="Touch polylogue/mcp/server.py.",
        labels=["area:storage"],
    )
    fp = bead_cluster._extract_footprint(bead)
    assert "storage/sqlite/" not in fp.packages
    assert "polylogue/storage/" not in fp.packages


def test_extract_footprint_finds_migration_slots_and_generated_surfaces() -> None:
    bead = _bead(
        "polylogue-a",
        design="Add migration 008_add_col.sql; regenerate topology-status.md too.",
    )
    fp = bead_cluster._extract_footprint(bead)
    assert "008" in fp.migration_slots
    assert "generated:topology-status" in fp.generated_surfaces


def test_footprint_overlap_keys_exclude_broad_packages() -> None:
    fp = bead_cluster.Footprint(files=[], packages=["polylogue/storage/", "polylogue/mcp/"])
    assert "polylogue/storage/" not in fp.overlap_keys()
    assert "polylogue/mcp/" in fp.overlap_keys()


def test_footprint_is_empty_when_no_signal() -> None:
    assert bead_cluster.Footprint().is_empty
    assert not bead_cluster.Footprint(packages=["polylogue/mcp/"]).is_empty


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classify_in_progress_takes_priority() -> None:
    bead = _bead("polylogue-a", status="in_progress", labels=["horizon:frontier"])
    assert bead_cluster._classify(bead) == "IN-PROGRESS"


def test_classify_blocked_when_dependency_count_positive() -> None:
    bead = _bead("polylogue-a", dependency_count=2, labels=["horizon:frontier"])
    assert bead_cluster._classify(bead) == "BLOCKED"


def test_classify_frontier_ready_from_horizon_label() -> None:
    bead = _bead("polylogue-a", labels=["horizon:frontier"])
    assert bead_cluster._classify(bead) == "FRONTIER-READY"


def test_classify_design_horizon_from_mid_or_vision_label() -> None:
    assert bead_cluster._classify(_bead("polylogue-a", labels=["horizon:mid"])) == "DESIGN-HORIZON"
    assert bead_cluster._classify(_bead("polylogue-b", labels=["horizon:vision"])) == "DESIGN-HORIZON"


def test_classify_unlabeled_high_priority_defaults_to_frontier_ready() -> None:
    bead = _bead("polylogue-a", priority=1, labels=[])
    assert bead_cluster._classify(bead) == "FRONTIER-READY"


def test_classify_unlabeled_low_priority_defaults_to_design_horizon() -> None:
    bead = _bead("polylogue-a", priority=3, labels=[])
    assert bead_cluster._classify(bead) == "DESIGN-HORIZON"


# ---------------------------------------------------------------------------
# Overlap graph + clustering
# ---------------------------------------------------------------------------


def test_build_overlap_graph_links_beads_sharing_a_package() -> None:
    beads = [
        _bead("polylogue-a", design="Touch polylogue/mcp/server.py."),
        _bead("polylogue-b", design="Touch polylogue/mcp/dispatch.py."),
        _bead("polylogue-c", design="Touch polylogue/cli/click_app.py."),
    ]
    footprints = {b["id"]: bead_cluster._extract_footprint(b) for b in beads}
    adj = bead_cluster._build_overlap_graph(beads, footprints)
    assert "polylogue-b" in adj["polylogue-a"]
    assert "polylogue-c" not in adj["polylogue-a"]


def test_connected_components_groups_transitively_linked_beads() -> None:
    adj = {
        "a": {"b"},
        "b": {"a", "c"},
        "c": {"b"},
        "d": set(),
    }
    components = bead_cluster._connected_components(["a", "b", "c", "d"], adj)
    component_sets = {frozenset(c) for c in components}
    assert frozenset({"a", "b", "c"}) in component_sets
    assert frozenset({"d"}) in component_sets


def test_find_contention_flags_shared_migration_slot() -> None:
    beads = [
        _bead("polylogue-a", design="Add migration 008_add_col.sql"),
        _bead("polylogue-b", design="Add migration 008_other.sql"),
        _bead("polylogue-c", design="Add migration 009_third.sql"),
    ]
    footprints = {b["id"]: bead_cluster._extract_footprint(b) for b in beads}
    contention = bead_cluster._find_contention(beads, footprints)
    events = {ev["key"]: set(ev["beads"]) for ev in contention}
    assert events["migration:008"] == {"polylogue-a", "polylogue-b"}
    assert "migration:009" not in events


# ---------------------------------------------------------------------------
# Roster-collision validation (AC3)
# ---------------------------------------------------------------------------


def test_validate_roster_detects_known_collision(capsys: pytest.CaptureFixture[str]) -> None:
    beads = [
        _bead("polylogue-lane-a", design="touches migration 008_x.sql"),
        _bead("polylogue-lane-b", design="touches migration 008_y.sql"),
    ]
    footprints = {b["id"]: bead_cluster._extract_footprint(b) for b in beads}
    bead_cluster._validate_roster(beads, footprints)
    out = capsys.readouterr().out
    assert "PREDICTED" in out
    assert "polylogue-lane-a" in out
    assert "polylogue-lane-b" in out


def test_validate_roster_reports_not_detected_when_absent(capsys: pytest.CaptureFixture[str]) -> None:
    beads = [_bead("polylogue-a", design="no migrations mentioned")]
    footprints = {b["id"]: bead_cluster._extract_footprint(b) for b in beads}
    bead_cluster._validate_roster(beads, footprints)
    out = capsys.readouterr().out
    assert "NOT-DETECTED" in out


# ---------------------------------------------------------------------------
# JSON stream parsing (tolerates bd's trailing pagination notice)
# ---------------------------------------------------------------------------


def test_parse_json_stream_ignores_trailing_pagination_notice() -> None:
    text = '[{"id": "polylogue-a"}]\nShowing 1 of 2 ready issues. Use --limit 0 for all.\n'
    value = bead_cluster._parse_json_stream(text)
    assert value == [{"id": "polylogue-a"}]


def test_parse_json_stream_plain_json_still_works() -> None:
    text = '[{"id": "polylogue-a"}]'
    assert bead_cluster._parse_json_stream(text) == [{"id": "polylogue-a"}]


# ---------------------------------------------------------------------------
# _load_beads / subprocess invocation
# ---------------------------------------------------------------------------


def test_load_beads_from_input_file(tmp_path: Path) -> None:
    path = tmp_path / "ready.json"
    path.write_text(json.dumps([{"id": "polylogue-a"}]))
    args = argparse.Namespace(input=str(path), all_open=False)
    assert bead_cluster._load_beads(args) == [{"id": "polylogue-a"}]


def test_load_beads_calls_bd_ready_with_unbounded_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        seen["cmd"] = cmd
        return MagicMock(returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = argparse.Namespace(input=None, all_open=False)
    assert bead_cluster._load_beads(args) == []
    assert seen["cmd"] == ["bd", "ready", "--json", "--limit", "0"]


def test_load_beads_all_open_calls_bd_list(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> MagicMock:
        seen["cmd"] = cmd
        return MagicMock(returncode=0, stdout="[]", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = argparse.Namespace(input=None, all_open=True)
    assert bead_cluster._load_beads(args) == []
    assert seen["cmd"] == ["bd", "list", "--status=open", "--json", "--limit", "0"]


def test_load_beads_tolerates_trailing_pagination_notice(monkeypatch: pytest.MonkeyPatch) -> None:
    stdout = '[{"id": "polylogue-a"}]\nShowing 1 of 2 ready issues. Use --limit 0 for all.\n'
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: MagicMock(returncode=0, stdout=stdout, stderr=""))
    args = argparse.Namespace(input=None, all_open=False)
    assert bead_cluster._load_beads(args) == [{"id": "polylogue-a"}]


def test_load_beads_exits_nonzero_on_bd_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: MagicMock(returncode=1, stdout="", stderr="bd: no workspace")
    )
    args = argparse.Namespace(input=None, all_open=False)
    with pytest.raises(SystemExit):
        bead_cluster._load_beads(args)


# ---------------------------------------------------------------------------
# main() end-to-end over --input
# ---------------------------------------------------------------------------


def test_main_json_output_clusters_overlapping_beads(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    beads = [
        _bead(
            "polylogue-a",
            priority=1,
            labels=["horizon:frontier"],
            design="Touch polylogue/mcp/server.py.",
        ),
        _bead(
            "polylogue-b",
            priority=2,
            labels=["horizon:frontier"],
            design="Touch polylogue/mcp/dispatch.py.",
        ),
        _bead(
            "polylogue-c",
            priority=1,
            labels=["horizon:frontier"],
            design="Touch polylogue/cli/click_app.py.",
        ),
        _bead("polylogue-blocked", priority=1, dependency_count=1, labels=["horizon:frontier"]),
        _bead("polylogue-wip", status="in_progress"),
        _bead("polylogue-vision", labels=["horizon:vision"]),
    ]
    path = tmp_path / "ready.json"
    path.write_text(json.dumps(beads))

    rc = bead_cluster.main(["--input", str(path), "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    cluster_ids = {frozenset(m["id"] for m in c["beads"]) for c in payload["frontier_clusters"]}
    assert frozenset({"polylogue-a", "polylogue-b"}) in cluster_ids
    assert frozenset({"polylogue-c"}) in cluster_ids
    assert [b["id"] for b in payload["blocked"]] == ["polylogue-blocked"]
    assert [b["id"] for b in payload["in_progress"]] == ["polylogue-wip"]
    assert payload["design_horizon_count"] == 1


def test_main_human_output_renders_sections(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    beads = [
        _bead("polylogue-a", priority=1, labels=["horizon:frontier"], design="Touch polylogue/mcp/server.py."),
    ]
    path = tmp_path / "ready.json"
    path.write_text(json.dumps(beads))

    rc = bead_cluster.main(["--input", str(path)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "FRONTIER-READY CLUSTERS" in out
    assert "polylogue-a" in out


def test_main_max_priority_filters_lower_priority_beads(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    beads = [
        _bead("polylogue-hi", priority=1, labels=["horizon:frontier"]),
        _bead("polylogue-lo", priority=3, labels=["horizon:frontier"]),
    ]
    path = tmp_path / "ready.json"
    path.write_text(json.dumps(beads))

    rc = bead_cluster.main(["--input", str(path), "--json", "--max-priority", "1"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    all_ids = {m["id"] for c in payload["frontier_clusters"] for m in c["beads"]}
    assert all_ids == {"polylogue-hi"}

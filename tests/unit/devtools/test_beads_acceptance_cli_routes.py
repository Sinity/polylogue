from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from devtools import beads_acceptance_contracts, click_dispatch, lane_brief
from polylogue.core.json import loads as json_loads


def _canonical_record() -> dict[str, object]:
    wanted = next(
        row
        for row in (json_loads(line) for line in Path(".beads/issues.jsonl").read_text().splitlines() if line.strip())
        if isinstance(row, dict) and row.get("id") == "polylogue-fyyro"
    )
    assert isinstance(wanted, dict)
    result: dict[str, object] = {}
    for key, value in wanted.items():
        result[str(key)] = value
    return result


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_validator_published_cli_route_forwards_issue_and_manifest_paths(capsys: pytest.CaptureFixture[str]) -> None:
    rc = click_dispatch.main(
        [
            "lab",
            "policy",
            "acceptance-contracts",
            ".beads/issues.jsonl",
            "--manifest",
            "docs/plans/beads-acceptance-contracts-2026-08-07.txt",
            "--json",
        ]
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["validated"] == 218


def test_validator_cli_route_fails_closed_for_malformed_structured_route_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    issue = _canonical_record()
    bead_id = str(issue["id"])
    metadata = issue["metadata"]
    assert isinstance(metadata, dict)
    contract = metadata["acceptance_contract_v1"]
    assert isinstance(contract, dict)
    route_spec = contract["route_spec"]
    assert isinstance(route_spec, dict)
    route_spec["mode"] = []
    issues = tmp_path / "issues.jsonl"
    _write(issues, [issue])
    monkeypatch.setattr(beads_acceptance_contracts, "load_manifest", lambda path: (bead_id,))
    monkeypatch.setattr(beads_acceptance_contracts, "validate_route_registry", lambda ids: [])

    rc = click_dispatch.main(
        [
            "lab",
            "policy",
            "acceptance-contracts",
            str(issues),
            "--manifest",
            str(tmp_path / "manifest.txt"),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["failures"][bead_id]
    assert any("route_spec.mode" in error for error in payload["failures"][bead_id])


def test_validator_cli_route_fails_closed_for_registry_errors_without_bead_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    issue = _canonical_record()
    bead_id = str(issue["id"])
    issues = tmp_path / "issues.jsonl"
    _write(issues, [issue])
    monkeypatch.setattr(beads_acceptance_contracts, "load_manifest", lambda path: (bead_id,))
    monkeypatch.setattr(beads_acceptance_contracts, "validate_route_registry", lambda ids: ["unbound route entry"])

    rc = click_dispatch.main(
        [
            "lab",
            "policy",
            "acceptance-contracts",
            str(issues),
            "--manifest",
            str(tmp_path / "manifest.txt"),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["failures"] == {}
    assert payload["route_registry_errors"] == ["unbound route entry"]
    assert payload["dispatch_blocked"] is True


def test_reconciliation_and_applier_published_routes_forward_exact_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    canonical = _canonical_record()
    bead_id = str(canonical["id"])
    monkeypatch.setattr(beads_acceptance_contracts, "load_manifest", lambda path: (bead_id,))
    monkeypatch.setattr(beads_acceptance_contracts, "validate_route_registry", lambda ids: [])
    repository = tmp_path / "repository.jsonl"
    before = tmp_path / "before.jsonl"
    wave = tmp_path / "wave.jsonl"
    report = tmp_path / "report.json"
    output = tmp_path / "output.jsonl"
    live = copy.deepcopy(canonical)
    live["metadata"] = {}
    live["acceptance_criteria"] = None
    _write(repository, [canonical])
    _write(before, [live])

    rc = click_dispatch.main(
        [
            "lab",
            "policy",
            "acceptance-contract-reconcile",
            "--repository",
            str(repository),
            "--live",
            str(before),
            "--wave",
            str(wave),
            "--report",
            str(report),
            "--manifest",
            "ignored-by-test-manifest-patch.txt",
            "--json",
        ]
    )
    assert rc == 0
    assert wave.is_file() and report.is_file()
    assert json.loads(capsys.readouterr().out)["targeted_ids"] == [bead_id]

    rc = click_dispatch.main(
        [
            "lab",
            "policy",
            "acceptance-contract-apply",
            "--repository",
            str(repository),
            "--before",
            str(before),
            "--wave",
            str(wave),
            "--report",
            str(report),
            "--output",
            str(output),
            "--json",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["idempotent"] is False

    rc = click_dispatch.main(
        [
            "lab",
            "policy",
            "acceptance-contract-apply",
            "--repository",
            str(repository),
            "--before",
            str(before),
            "--wave",
            str(wave),
            "--report",
            str(report),
            "--output",
            str(output),
            "--json",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["idempotent"] is True


def test_lane_brief_published_route_forwards_recent_days_and_blocks_bad_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, int] = {}
    monkeypatch.setattr(lane_brief, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        lane_brief,
        "_fetch_bead",
        lambda bead_id: lane_brief.BeadRecord(
            id=bead_id,
            found=True,
            title="CLI lane",
            description="devtools/lane_brief.py",
            contract_confidence="high",
            contract_errors=["unknown route identifier"],
        ),
    )
    monkeypatch.setattr(lane_brief, "_load_bd_export", lambda repo_root, tmpdir: [])
    monkeypatch.setattr(lane_brief, "_verify_footprint", lambda repo_root, paths: [])
    monkeypatch.setattr(lane_brief, "_find_prior_art", lambda records, paths, exclude_ids: [])

    def _recent(repo_root: Path, paths: list[str], days: int) -> list[str]:
        seen["days"] = days
        return []

    monkeypatch.setattr(lane_brief, "_recent_master_commits", _recent)

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
    assert "unknown route identifier" in (tmp_path / "brief.md").read_text()


def test_lane_brief_published_route_allows_non_manifest_no_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(beads_acceptance_contracts, "load_manifest", lambda path: ("polylogue-manifest",))
    monkeypatch.setattr(lane_brief, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        lane_brief,
        "_fetch_bead",
        lambda bead_id: lane_brief.BeadRecord(id=bead_id, found=True, title="Non-manifest lane"),
    )
    monkeypatch.setattr(lane_brief, "_load_bd_export", lambda repo_root, tmpdir: [])
    monkeypatch.setattr(lane_brief, "_verify_footprint", lambda repo_root, paths: [])
    monkeypatch.setattr(lane_brief, "_find_prior_art", lambda records, paths, exclude_ids: [])
    monkeypatch.setattr(lane_brief, "_recent_master_commits", lambda repo_root, paths, days: [])

    rc = click_dispatch.main(
        [
            "workspace",
            "lane-brief",
            "polylogue-non-manifest",
            "--out",
            str(tmp_path / "brief.md"),
            "--tmpdir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert "DISPATCH BLOCKED" not in (tmp_path / "brief.md").read_text()

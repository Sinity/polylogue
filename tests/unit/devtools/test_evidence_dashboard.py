"""Tests for ``devtools evidence-dashboard``."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from devtools import evidence_dashboard

# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _write_evidence_artifact(
    root: Path,
    *,
    name: str,
    contract: str,
    surface: str,
    test_nodeid: str,
    age_days: int = 0,
    dirty: bool = False,
) -> Path:
    evidence_dir = root / evidence_dashboard.EVIDENCE_DIR_REL
    evidence_dir.mkdir(parents=True, exist_ok=True)
    artifact = evidence_dir / f"{name}.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": contract,
                "surface": surface,
                "test_nodeid": test_nodeid,
                "timestamp": "2026-05-16T00:00:00+00:00",
                "git_sha": "abc123",
                "dirty": dirty,
            }
        )
    )
    if age_days:
        ts = datetime.now(timezone.utc) - timedelta(days=age_days)
        epoch = ts.timestamp()
        import os

        os.utime(artifact, (epoch, epoch))
    return artifact


# ──────────────────────────────────────────────────────────────────────
# section: dashboard JSON shape
# ──────────────────────────────────────────────────────────────────────


def test_dashboard_runs_on_empty_tree(tmp_path: Path) -> None:
    dashboard = evidence_dashboard.build_dashboard(tmp_path)
    assert dashboard["schema_version"] == 1
    assert dashboard["root"] == str(tmp_path)
    # All artifact-backed sections report unavailable on a bare tree.
    for section in (
        "pytest",
        "contract_evidence",
        "coverage",
        "benchmark_campaigns",
        "slo_catalog",
        "mutation_campaigns",
    ):
        assert section in dashboard
        assert dashboard[section]["available"] is False
    # Static gates and witnesses are still present with empty content.
    assert "static_gates" in dashboard
    assert "witnesses" in dashboard
    assert dashboard["witnesses"]["committed"]["available"] is False
    assert dashboard["witnesses"]["local"]["available"] is False


def test_dashboard_top_level_keys_are_stable(tmp_path: Path) -> None:
    dashboard = evidence_dashboard.build_dashboard(tmp_path)
    assert set(dashboard.keys()) == {
        "schema_version",
        "generated_at",
        "root",
        "pytest",
        "contract_evidence",
        "coverage",
        "benchmark_campaigns",
        "slo_catalog",
        "static_gates",
        "witnesses",
        "mutation_campaigns",
    }


# ──────────────────────────────────────────────────────────────────────
# section: pytest health
# ──────────────────────────────────────────────────────────────────────


def test_pytest_health_reads_last_pytest_json(tmp_path: Path) -> None:
    _write_json(
        tmp_path / evidence_dashboard.PYTEST_REPORT_REL,
        {
            "duration": 12.34,
            "summary": {"passed": 100, "failed": 0, "xfailed": 2, "total": 102},
        },
    )
    dashboard = evidence_dashboard.build_dashboard(tmp_path)
    pytest_data = dashboard["pytest"]
    assert pytest_data["available"] is True
    assert pytest_data["status"] == "ok"
    assert pytest_data["counts"]["passed"] == 100
    assert pytest_data["counts"]["failed"] == 0
    assert pytest_data["duration_s"] == 12.34


def test_pytest_health_reports_failures(tmp_path: Path) -> None:
    _write_json(
        tmp_path / evidence_dashboard.PYTEST_REPORT_REL,
        {"duration": 5.0, "summary": {"passed": 10, "failed": 1, "total": 11}},
    )
    dashboard = evidence_dashboard.build_dashboard(tmp_path)
    assert dashboard["pytest"]["status"] == "fail"


# ──────────────────────────────────────────────────────────────────────
# section: contract evidence
# ──────────────────────────────────────────────────────────────────────


def test_contract_evidence_groups_by_surface(tmp_path: Path) -> None:
    _write_evidence_artifact(
        tmp_path,
        name="cli-1",
        contract="cli.help.root",
        surface="cli",
        test_nodeid="tests/unit/cli/test_help.py::test_root",
    )
    _write_evidence_artifact(
        tmp_path,
        name="cli-2",
        contract="cli.stats",
        surface="cli",
        test_nodeid="tests/unit/cli/test_stats.py::test_stats",
    )
    _write_evidence_artifact(
        tmp_path,
        name="mcp-1",
        contract="mcp.search",
        surface="mcp",
        test_nodeid="tests/unit/mcp/test_search.py::test_search",
    )
    dashboard = evidence_dashboard.build_dashboard(tmp_path)
    contract = dashboard["contract_evidence"]
    assert contract["available"] is True
    assert contract["total_artifacts"] == 3
    assert contract["unique_contracts"] == 3
    assert contract["by_surface"]["cli"]["total_artifacts"] == 2
    assert contract["by_surface"]["cli"]["unique_contracts"] == 2
    assert contract["by_surface"]["mcp"]["total_artifacts"] == 1


def test_contract_evidence_flags_stale_artifacts(tmp_path: Path) -> None:
    _write_evidence_artifact(
        tmp_path,
        name="fresh",
        contract="cli.fresh",
        surface="cli",
        test_nodeid="tests/unit/cli/test_a.py::test_fresh",
    )
    _write_evidence_artifact(
        tmp_path,
        name="old",
        contract="cli.old",
        surface="cli",
        test_nodeid="tests/unit/cli/test_a.py::test_old",
        age_days=30,
    )
    dashboard = evidence_dashboard.build_dashboard(tmp_path, stale_days=7)
    contract = dashboard["contract_evidence"]
    assert contract["stale_count"] == 1
    assert "cli.old" in contract["stale_contracts"]
    assert "cli.fresh" not in contract["stale_contracts"]


def test_contract_evidence_counts_dirty(tmp_path: Path) -> None:
    _write_evidence_artifact(
        tmp_path,
        name="dirty",
        contract="cli.dirty",
        surface="cli",
        test_nodeid="tests/unit/cli/test.py::test_dirty",
        dirty=True,
    )
    dashboard = evidence_dashboard.build_dashboard(tmp_path)
    assert dashboard["contract_evidence"]["dirty_count"] == 1


# ──────────────────────────────────────────────────────────────────────
# section: coverage
# ──────────────────────────────────────────────────────────────────────


def test_coverage_reads_xml(tmp_path: Path) -> None:
    (tmp_path / "coverage.xml").write_text(
        '<?xml version="1.0"?><coverage line-rate="0.9123" lines-covered="9123" lines-valid="10000"></coverage>'
    )
    dashboard = evidence_dashboard.build_dashboard(tmp_path)
    cov = dashboard["coverage"]
    assert cov["available"] is True
    assert cov["percent"] == 91.23
    assert cov["lines_covered"] == 9123


def test_coverage_handles_binary_only(tmp_path: Path) -> None:
    (tmp_path / ".coverage").write_bytes(b"SQLite binary stub")
    dashboard = evidence_dashboard.build_dashboard(tmp_path)
    cov = dashboard["coverage"]
    assert cov["available"] is True
    assert "binary" in cov["note"]


# ──────────────────────────────────────────────────────────────────────
# section: static gates
# ──────────────────────────────────────────────────────────────────────


def test_static_gates_from_last_verify_result(tmp_path: Path) -> None:
    _write_json(
        tmp_path / evidence_dashboard.LAST_VERIFY_RESULT_REL,
        {
            "result": {
                "timestamp": "2026-05-16T04:00:00+00:00",
                "steps": [
                    {"name": "ruff format", "exit": 0, "duration_s": 0.3},
                    {"name": "ruff check", "exit": 0, "duration_s": 0.1},
                    {"name": "mypy", "exit": 1, "duration_s": 5.0},
                ],
            }
        },
    )
    dashboard = evidence_dashboard.build_dashboard(tmp_path)
    gates = dashboard["static_gates"]
    assert gates["available"] is True
    assert "mypy" in gates["failing"]
    gate_names = {g["name"] for g in gates["gates"]}
    assert "ruff format" in gate_names
    # Gates with no observed run are still listed but unavailable.
    untracked = [g for g in gates["gates"] if g.get("available") is False]
    assert len(untracked) >= 1


# ──────────────────────────────────────────────────────────────────────
# section: change trace
# ──────────────────────────────────────────────────────────────────────


def test_build_trace_returns_stable_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Trace JSON has stable top-level keys regardless of changed-path content."""

    def _fake_impact(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "base_ref": "origin/master",
            "head_ref": "HEAD",
            "change_subjects": [
                {
                    "id": "c-1",
                    "path": "tests/unit/cli/test_help.py",
                    "kind": "test",
                    "reason": "test source changed",
                    "subject_ids": ["cli.help"],
                    "operation_names": [],
                    "surface_names": ["cli"],
                    "checks": [
                        {"command": ["pytest", "tests/unit/cli/test_help.py"], "reason": "directly changed"},
                    ],
                },
            ],
            "required_pr_gates": [{"command": ["devtools", "verify"], "reason": "default"}],
            "deployment_gates": [],
        }

    monkeypatch.setattr(evidence_dashboard, "build_verification_impact_report", _fake_impact)

    trace = evidence_dashboard.build_trace(tmp_path)
    assert set(trace.keys()) >= {
        "schema_version",
        "generated_at",
        "base_ref",
        "head_ref",
        "changed_path_count",
        "changes",
        "required_gates",
    }
    assert trace["changed_path_count"] == 1
    assert trace["changes"][0]["path"] == "tests/unit/cli/test_help.py"
    assert trace["changes"][0]["evidence_artifact_count"] == 0


def test_build_trace_links_evidence_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Trace links contract-evidence artifacts to changed paths via test_nodeid."""

    nodeid = "tests/unit/cli/test_help.py::TestRootHelp::test_root"
    _write_evidence_artifact(
        tmp_path,
        name="cli-help",
        contract="cli.help.root",
        surface="cli",
        test_nodeid=nodeid,
    )

    def _fake_impact(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "base_ref": "origin/master",
            "head_ref": "HEAD",
            "change_subjects": [
                {
                    "id": "c-1",
                    "path": "tests/unit/cli/test_help.py",
                    "kind": "test",
                    "reason": "test source changed",
                    "subject_ids": [],
                    "operation_names": [],
                    "surface_names": ["cli"],
                    "checks": [],
                }
            ],
            "required_pr_gates": [],
            "deployment_gates": [],
        }

    monkeypatch.setattr(evidence_dashboard, "build_verification_impact_report", _fake_impact)

    trace = evidence_dashboard.build_trace(tmp_path)
    row = trace["changes"][0]
    assert row["evidence_artifact_count"] == 1
    assert row["evidence_artifacts"][0]["contract"] == "cli.help.root"


# ──────────────────────────────────────────────────────────────────────
# section: cli entrypoint
# ──────────────────────────────────────────────────────────────────────


def test_cli_dashboard_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evidence_dashboard, "ROOT", tmp_path)
    rc = evidence_dashboard.main(["--json"])
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["schema_version"] == 1
    assert payload["pytest"]["available"] is False


def test_cli_dashboard_markdown(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evidence_dashboard, "ROOT", tmp_path)
    rc = evidence_dashboard.main(["--markdown"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# Evidence Dashboard" in out
    assert "## Pytest health" in out
    assert "## Contract evidence" in out
    assert "## Coverage" in out
    assert "## Static gates" in out


def test_cli_trace_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evidence_dashboard, "ROOT", tmp_path)

    def _fake_impact(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "base_ref": "origin/master",
            "head_ref": "HEAD",
            "change_subjects": [],
            "required_pr_gates": [],
            "deployment_gates": [],
        }

    monkeypatch.setattr(evidence_dashboard, "build_verification_impact_report", _fake_impact)
    rc = evidence_dashboard.main(["trace", "--base", "origin/master", "--head", "HEAD", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["changed_path_count"] == 0
    assert payload["base_ref"] == "origin/master"

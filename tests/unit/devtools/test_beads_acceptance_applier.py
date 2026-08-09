from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from devtools import beads_acceptance_applier as applier
from devtools import beads_acceptance_contracts as contracts
from devtools import beads_acceptance_reconciliation as reconciliation


@pytest.fixture(autouse=True)
def _test_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contracts, "load_manifest", lambda path: ("polylogue-test",))
    monkeypatch.setattr(contracts, "validate_route_registry", lambda ids: [])
    monkeypatch.setattr(
        contracts,
        "resolve_route",
        lambda identifier: (
            {
                "bead_id": "polylogue-test",
                "class": "ImplementationRoute",
                "contract_type": "implementation",
                "dispatch": "production",
                "targets": ["Test production route."],
            }
            if isinstance(identifier, str) and identifier == "test/production"
            else None
        ),
    )


def _issue() -> dict[str, Any]:
    issue: dict[str, Any] = {
        "id": "polylogue-test",
        "title": "A real contract source",
        "description": "A source description with observable scope.",
        "design": "The design is recorded.",
        "notes": "The notes are stable.",
        "status": "open",
        "priority": 1,
        "issue_type": "task",
        "updated_at": "2026-08-07T00:00:00Z",
        "dependencies": [],
        "metadata": {},
        "acceptance_criteria": None,
    }
    contract: dict[str, Any] = {
        "schema_version": 1,
        "bead_id": issue["id"],
        "contract_type": "implementation",
        "risk": "ordinary",
        "confidence": "high",
        "outcome": "The behavior is observable through the production route.",
        "routes": ["Test production route."],
        "evidence": ["A source description with observable scope."],
        "retained_scope": [],
        "verification": ["Run a focused regression."],
        "verification_route": {"manager": "devtools", "focused": "devtools test", "default": "devtools verify"},
        "anti_vacuity": ["Removing the guard makes the regression fail."],
        "safety": [],
        "route_spec": {
            "mode": "named",
            "identifier": "test/production",
            "class": "ImplementationRoute",
            "dispatch": "production",
        },
        "closure": {
            "rule": "Close only with final-head evidence.",
            "disposition": "whole-or-explicit-partial",
            "successor_required_for_partial": True,
        },
    }
    evidence = contract["evidence"][0]
    snapshot = issue["description"]
    digest = hashlib.sha256(snapshot.encode()).hexdigest()
    text_digest = hashlib.sha256(evidence.encode()).hexdigest()
    contract["evidence_spans"] = [
        {
            "source_field": "description",
            "snapshot": snapshot,
            "snapshot_digest": digest,
            "range": {"start": 0, "end": len(evidence.encode())},
            "text_digest": text_digest,
        }
    ]
    contract["source_digest"] = reconciliation.source_digest(issue)
    contract["dependency_digest"] = contracts.dependency_digest(issue)
    issue["metadata"]["acceptance_contract_v1"] = contract
    issue["acceptance_criteria"] = reconciliation.render(contract)
    return issue


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_guarded_applier_is_idempotent_and_report_bound(tmp_path: Path) -> None:
    repository = tmp_path / "repository.jsonl"
    before = tmp_path / "before.jsonl"
    wave_path = tmp_path / "wave.jsonl"
    report_path = tmp_path / "report.json"
    output = tmp_path / "output.jsonl"
    master = _issue()
    live = copy.deepcopy(master)
    live["metadata"] = {}
    live["acceptance_criteria"] = None
    _write(repository, [master])
    _write(before, [live])
    report, wave = reconciliation.reconcile(repository, before)
    _write(wave_path, wave)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    first = applier.apply_guarded_wave(
        repository=repository, before=before, wave=wave_path, report=report_path, output=output
    )
    second = applier.apply_guarded_wave(
        repository=repository, before=before, wave=wave_path, report=report_path, output=output
    )

    assert first["idempotent"] is False
    assert second["idempotent"] is True
    assert reconciliation.load_jsonl(output)["polylogue-test"]["acceptance_criteria"] == master["acceptance_criteria"]


def test_guarded_applier_refuses_modified_wave(tmp_path: Path) -> None:
    repository = tmp_path / "repository.jsonl"
    before = tmp_path / "before.jsonl"
    wave_path = tmp_path / "wave.jsonl"
    report_path = tmp_path / "report.json"
    output = tmp_path / "output.jsonl"
    master = _issue()
    live = copy.deepcopy(master)
    live["metadata"] = {}
    live["acceptance_criteria"] = None
    _write(repository, [master])
    _write(before, [live])
    report, wave = reconciliation.reconcile(repository, before)
    wave[0]["acceptance_criteria"] = "modified wave"
    _write(wave_path, wave)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(reconciliation.ReconciliationError, match="targeted wave digest"):
        applier.apply_guarded_wave(
            repository=repository, before=before, wave=wave_path, report=report_path, output=output
        )


@pytest.mark.parametrize(
    "field",
    ["ids", "counts", "contract_refused_reasons", "contract_deferred_reasons", "already_guarded_ids"],
)
def test_guarded_applier_refuses_tampered_complete_report(tmp_path: Path, field: str) -> None:
    repository = tmp_path / "repository.jsonl"
    before = tmp_path / "before.jsonl"
    wave_path = tmp_path / "wave.jsonl"
    report_path = tmp_path / "report.json"
    output = tmp_path / "output.jsonl"
    master = _issue()
    live = copy.deepcopy(master)
    live["metadata"] = {}
    live["acceptance_criteria"] = None
    _write(repository, [master])
    _write(before, [live])
    report, wave = reconciliation.reconcile(repository, before)
    _write(wave_path, wave)
    report[field] = {**report[field]} if isinstance(report[field], dict) else list(report[field])
    if field == "ids":
        report[field]["live_only"] = ["polylogue-unexpected"]
    elif field == "counts":
        report[field]["live_only"] = 1
    elif field == "contract_refused_reasons":
        report[field]["polylogue-test"] = ["tampered refusal"]
    elif field == "contract_deferred_reasons":
        report[field]["polylogue-test"] = "tampered deferral"
    else:
        report[field].append("polylogue-test")
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(reconciliation.ReconciliationError, match="canonical recomputation"):
        applier.apply_guarded_wave(
            repository=repository, before=before, wave=wave_path, report=report_path, output=output
        )

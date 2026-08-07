from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

MODULE_PATH = Path(__file__).parents[3] / "devtools" / "beads_acceptance_contracts.py"
spec = importlib.util.spec_from_file_location("beads_acceptance_contracts", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _issue(kind: str = "implementation", risk: str = "ordinary") -> dict[str, Any]:
    contract = {
        "schema_version": 1,
        "bead_id": "polylogue-test",
        "contract_type": kind,
        "risk": risk,
        "confidence": "high",
        "outcome": "The named behavior is observable through the production route.",
        "routes": ["Exercise the real production entry point."],
        "evidence": ["A red-before receipt records the defect."],
        "retained_scope": [],
        "verification": [
            "Run a focused production-route regression.",
            "Run `devtools verify` for the affected-test baseline.",
        ],
        "anti_vacuity": ["Removing the guard makes the test fail."],
        "safety": ["Dry-run and backup are required."]
        if risk == "durable-mutation" or kind == "live_operation"
        else [],
        "closure": {
            "rule": "Close only with final-head evidence.",
            "disposition": "whole-or-explicit-partial",
            "successor_required_for_partial": True,
        },
        "source_digest": "a" * 64,
    }
    return {
        "id": "polylogue-test",
        "title": "Test contract",
        "description": "A test contract source.",
        "design": "The test design is explicit.",
        "notes": "The source snapshot is stable.",
        "status": "open",
        "priority": 2,
        "issue_type": "task",
        "updated_at": "2026-08-07T00:00:00Z",
        "metadata": {"acceptance_contract_v1": contract},
        "acceptance_criteria": mod.render(contract),
    }


def test_valid_contract_round_trips() -> None:
    issue = _issue()
    issue["metadata"]["acceptance_contract_v1"]["source_digest"] = mod.source_digest(issue)
    issue["acceptance_criteria"] = mod.render(issue["metadata"]["acceptance_contract_v1"])
    assert mod.validate(issue) == []


def test_durable_mutation_requires_safety() -> None:
    issue = _issue(risk="durable-mutation")
    issue["metadata"]["acceptance_contract_v1"]["source_digest"] = mod.source_digest(issue)
    issue["metadata"]["acceptance_contract_v1"]["safety"] = []
    issue["acceptance_criteria"] = mod.render(issue["metadata"]["acceptance_contract_v1"])
    assert "durable-mutation requires safety clauses" in mod.validate(issue)


def test_live_operation_requires_typed_receipt_verification() -> None:
    issue = _issue(kind="live_operation")
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["source_digest"] = mod.source_digest(issue)
    issue["acceptance_criteria"] = mod.render(contract)
    assert "live_operation requires typed receipt verification" in mod.validate(issue)

    contract["verification"].append("Record the immutable typed apply receipt and result status.")
    issue["acceptance_criteria"] = mod.render(contract)
    assert mod.validate(issue) == []


def test_live_operation_malformed_verification_is_reported_without_crashing() -> None:
    issue = _issue(kind="live_operation")
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["verification"] = None
    contract["source_digest"] = mod.source_digest(issue)

    errors = mod.validate(issue)

    assert "verification must be a non-empty list of strings" in errors
    assert "live_operation requires typed receipt verification" in errors


def test_live_operation_requires_positive_receipt_clause() -> None:
    issue = _issue(kind="live_operation")
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["verification"] = ["No receipt is required for this route."]
    contract["source_digest"] = mod.source_digest(issue)

    assert "live_operation requires typed receipt verification" in mod.validate(issue)


def test_closure_disposition_is_typed_and_rendered() -> None:
    issue = _issue()
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["closure"]["disposition"] = "partial"
    contract["source_digest"] = mod.source_digest(issue)

    errors = mod.validate(issue)

    assert "closure.disposition must be whole-or-explicit-partial" in errors


def test_invalid_confidence_is_rejected() -> None:
    issue = _issue()
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["confidence"] = "planner_review"
    contract["source_digest"] = mod.source_digest(issue)
    issue["acceptance_criteria"] = mod.render(contract)
    assert "confidence must be high, medium, or planner-review" in mod.validate(issue)


def test_lifecycle_changes_do_not_invalidate_scope_digest() -> None:
    issue = _issue()
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["source_digest"] = mod.source_digest(issue)
    issue["acceptance_criteria"] = mod.render(contract)
    issue["status"] = "closed"
    issue["updated_at"] = "2026-08-07T13:00:00Z"
    assert mod.validate(issue) == []


def test_dependency_changes_invalidate_scope_digest() -> None:
    issue = _issue()
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["source_digest"] = mod.source_digest(issue)
    issue["acceptance_criteria"] = mod.render(contract)
    issue["dependencies"] = [{"depends_on_id": "polylogue-successor", "type": "blocks"}]

    assert "source_digest does not match the Bead source snapshot" in mod.validate(issue)


def test_read_only_audit_contract_does_not_require_mutation_safety() -> None:
    issue = _issue(kind="audit", risk="read-only")
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["source_digest"] = mod.source_digest(issue)
    issue["acceptance_criteria"] = mod.render(contract)

    assert mod.validate(issue) == []
    assert "Safety:" not in issue["acceptance_criteria"]


def test_placeholder_is_rejected() -> None:
    issue = _issue()
    issue["metadata"]["acceptance_contract_v1"]["source_digest"] = mod.source_digest(issue)
    issue["metadata"]["acceptance_contract_v1"]["outcome"] = "Figure out the route ..."
    issue["acceptance_criteria"] = mod.render(issue["metadata"]["acceptance_contract_v1"])
    assert any("placeholder" in error for error in mod.validate(issue))


def test_lowercase_evidence_fragment_is_rejected() -> None:
    issue = _issue()
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["evidence"] = ["ed and capped test runs remain red."]
    contract["source_digest"] = mod.source_digest(issue)

    assert "evidence contains a lowercase fragment" in mod.validate(issue)


def test_render_drift_is_rejected() -> None:
    issue = _issue()
    issue["metadata"]["acceptance_contract_v1"]["source_digest"] = mod.source_digest(issue)
    issue["acceptance_criteria"] = "weaker prose"
    assert "acceptance_criteria drifted from structured contract" in mod.validate(issue)


def test_scalar_clause_and_stale_digest_are_rejected() -> None:
    issue = _issue()
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["routes"] = "a route"
    assert "routes must be a non-empty list of strings" in mod.validate(issue)

    issue = _issue()
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["source_digest"] = "0" * 64
    issue["acceptance_criteria"] = mod.render(contract)
    assert "source_digest does not match the Bead source snapshot" in mod.validate(issue)


def test_manifest_is_required_and_cannot_shrink(tmp_path: Path) -> None:
    issues = tmp_path / "issues.jsonl"
    issues.write_text("{}\n", encoding="utf-8")
    missing = tmp_path / "missing.txt"

    try:
        mod.main([str(issues), "--manifest", str(missing)])
    except SystemExit as exc:
        assert "manifest is missing" in str(exc)
    else:
        raise AssertionError("missing manifest must fail closed")

    manifest = tmp_path / "manifest.txt"
    manifest.write_text("polylogue-test\n", encoding="utf-8")
    try:
        mod.main([str(issues), "--manifest", str(manifest)])
    except SystemExit as exc:
        assert "manifest inventory is invalid" in str(exc)
    else:
        raise AssertionError("shrunk manifest must fail closed")

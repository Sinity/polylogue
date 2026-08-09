from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import pytest

MODULE_PATH = Path(__file__).parents[3] / "devtools" / "beads_acceptance_contracts.py"
spec = importlib.util.spec_from_file_location("beads_acceptance_contracts", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _issue(kind: str = "implementation", risk: str = "ordinary") -> dict[str, Any]:
    route_dispatch = {
        "audit": "read-only",
        "decision": "decision",
        "documentation": "documentation",
    }.get(kind, "production")
    contract: dict[str, Any] = {
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
        "route_spec": {"mode": "named", "identifier": "production-route", "dispatch": route_dispatch},
        "verification_route": {"manager": "devtools", "focused": "devtools test", "default": "devtools verify"},
        "closure": {
            "rule": "Close only with final-head evidence.",
            "disposition": "whole-or-explicit-partial",
            "successor_required_for_partial": True,
        },
        "source_digest": "a" * 64,
    }
    if kind == "live_operation":
        contract["receipt"] = {
            "kind": "live-operation",
            "requirement": "required",
            "bindings": [
                "archive_identity",
                "operation",
                "target",
                "before_state",
                "after_state",
                "result_status",
            ],
        }
    contract["evidence_spans"] = [
        {
            "snapshot": contract["evidence"][0],
            "snapshot_digest": hashlib.sha256(contract["evidence"][0].encode("utf-8")).hexdigest(),
            "range": {"start": 0, "end": len(contract["evidence"][0])},
            "text_digest": hashlib.sha256(contract["evidence"][0].encode("utf-8")).hexdigest(),
        }
    ]
    issue = {
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
        "acceptance_criteria": "",
    }
    contract["dependency_digest"] = mod.dependency_digest(issue)
    contract["source_digest"] = mod.source_digest(issue)
    issue["acceptance_criteria"] = mod.render(contract)
    return issue


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
    contract.pop("receipt")
    contract["source_digest"] = mod.source_digest(issue)
    issue["acceptance_criteria"] = "malformed contract must not render"
    assert "receipt must be an object" in mod.validate(issue)

    contract["receipt"] = {
        "kind": "live-operation",
        "requirement": "required",
        "bindings": sorted(mod._REQUIRED_RECEIPT_BINDINGS),
    }
    issue["acceptance_criteria"] = mod.render(contract)
    assert mod.validate(issue) == []


def test_live_operation_malformed_verification_is_reported_without_crashing() -> None:
    issue = _issue(kind="live_operation")
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["verification"] = None
    contract["source_digest"] = mod.source_digest(issue)

    errors = mod.validate(issue)

    assert "verification must be a non-empty list of strings" in errors
    assert "receipt" not in " ".join(errors)


def test_live_operation_requires_positive_receipt_clause() -> None:
    issue = _issue(kind="live_operation")
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["verification"] = ["Print a receipt."]
    issue["acceptance_criteria"] = mod.render(contract)

    assert mod.validate(issue) == []
    contract["receipt"]["bindings"] = ["operation"]
    assert "receipt.bindings must include each required live-operation dimension exactly once" in mod.validate(issue)


def test_closure_disposition_is_typed_and_rendered() -> None:
    issue = _issue()
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["closure"]["disposition"] = "partial"
    contract["source_digest"] = mod.source_digest(issue)

    errors = mod.validate(issue)

    assert "closure.disposition must be whole-or-explicit-partial" in errors


def test_partial_closure_requires_a_successor() -> None:
    issue = _issue()
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["closure"]["successor_required_for_partial"] = False

    assert "whole-or-explicit-partial requires successor_required_for_partial=true" in mod.validate(issue)


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


def test_title_and_design_changes_invalidate_scope_digest() -> None:
    issue = _issue()
    issue["title"] = "Changed title"
    assert "source_digest does not match the Bead source snapshot" in mod.validate(issue)

    issue = _issue()
    contract = issue["metadata"]["acceptance_contract_v1"]
    issue["design"] = "Changed design"
    assert "source_digest does not match the Bead source snapshot" in mod.validate(issue)
    assert contract["source_digest"] != mod.source_digest(issue)


def test_byte_identical_canonical_source_remains_valid() -> None:
    issue = _issue()
    clone = dict(issue)
    assert mod.source_digest(issue) == mod.source_digest(clone)
    assert mod.dependency_digest(issue) == mod.dependency_digest(clone)


def test_read_only_audit_contract_does_not_require_mutation_safety() -> None:
    issue = _issue(kind="audit", risk="read-only")
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["source_digest"] = mod.source_digest(issue)
    issue["acceptance_criteria"] = mod.render(contract)

    assert mod.validate(issue) == []
    assert "Safety:" not in issue["acceptance_criteria"]


def test_decision_contract_does_not_promote_example_mutation_to_typed_route() -> None:
    issue = _issue(kind="decision", risk="durable-mutation")
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["verification"] = ["Consider `polylogue ops maintenance blob-publications --abandon --yes` as an example."]
    issue["acceptance_criteria"] = mod.render(contract)

    assert mod.validate(issue) == []


def test_placeholder_is_rejected() -> None:
    issue = _issue()
    issue["metadata"]["acceptance_contract_v1"]["source_digest"] = mod.source_digest(issue)
    issue["metadata"]["acceptance_contract_v1"]["outcome"] = "Figure out the route ..."
    issue["acceptance_criteria"] = mod.render(issue["metadata"]["acceptance_contract_v1"])
    assert any("placeholder" in error for error in mod.validate(issue))


def test_lowercase_evidence_fragment_is_rejected() -> None:
    """Production dependency: acceptance policy -> validate; catches prose-only truncation authority."""
    issue = _issue()
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["evidence"] = [
        "Measured evidence remains reconciled with this recorded population: ue sets that should remain red."
    ]
    contract["source_digest"] = mod.source_digest(issue)

    assert "evidence_spans[0].range text does not match the evidence item" in mod.validate(issue)


def test_structured_incomplete_evidence_span_is_rejected() -> None:
    """Production dependency: acceptance policy -> validate; catches trusting a false complete flag."""
    issue = _issue()
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["evidence_spans"] = [{"complete": False}]

    assert "evidence_spans[0] fields must be exactly snapshot, snapshot_digest, range, and text_digest" in mod.validate(
        issue
    )


def test_truncated_snapshot_range_is_rejected() -> None:
    """Production dependency: acceptance policy -> validate; catches a range past truncated snapshot bytes."""
    issue = _issue()
    contract = issue["metadata"]["acceptance_contract_v1"]
    span = contract["evidence_spans"][0]
    span["snapshot"] = span["snapshot"][:-6]

    assert "evidence_spans[0].range exceeds the snapshot byte length" in mod.validate(issue)


def test_route_dispatch_must_match_contract_type() -> None:
    """Production dependency: acceptance policy -> validate; catches live work routed as a decision."""
    issue = _issue(kind="live_operation")
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["route_spec"]["dispatch"] = "decision"
    issue["acceptance_criteria"] = mod.render(contract)

    assert "route_spec.dispatch 'decision' is incompatible with contract_type 'live_operation'" in mod.validate(issue)


def test_route_authority_requires_a_structured_identifier() -> None:
    """Production dependency: acceptance policy -> validate; catches prose standing in for route identity."""
    issue = _issue()
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["route_spec"].pop("identifier")
    issue["acceptance_criteria"] = "untrusted prose"

    assert "route_spec.identifier must be a non-empty named identifier" in mod.validate(issue)


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


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("outcome", None),
        ("routes", True),
        ("evidence", 3),
        ("verification", {"route": "bad"}),
        ("anti_vacuity", ["ok", 7]),
        ("retained_scope", None),
        ("safety", {"backup": True}),
    ],
)
def test_malformed_shapes_return_validation_errors_without_render_tracebacks(key: str, value: object) -> None:
    issue = _issue()
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract[key] = value

    errors = mod.validate(issue)

    assert errors
    assert all(isinstance(error, str) for error in errors)


def test_optional_null_lists_use_the_empty_list_canonical_form() -> None:
    issue = _issue()
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["retained_scope"] = None

    assert "retained_scope must be a list of strings; use [] when empty" in mod.validate(issue)


def test_generic_route_placeholder_is_rejected_at_specification_time() -> None:
    issue = _issue()
    contract = issue["metadata"]["acceptance_contract_v1"]
    contract["routes"] = ["Exercise the named production route where applicable."]

    assert "routes contains a generic placeholder; use named route fields" in mod.validate(issue)


def test_manifest_loader_rejects_shrinkage_and_invalid_rows(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("polylogue-test\n", encoding="utf-8")

    try:
        mod.load_manifest(manifest)
    except SystemExit as exc:
        assert "manifest inventory is invalid" in str(exc)
    else:
        raise AssertionError("shrunk manifest must fail closed")


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

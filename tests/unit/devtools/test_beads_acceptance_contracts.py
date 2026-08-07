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
        "outcome": "The named behavior is observable through the production route.",
        "routes": ["Exercise the real production entry point."],
        "evidence": ["A red-before receipt records the defect."],
        "retained_scope": [],
        "verification": ["Run a focused production-route regression."],
        "anti_vacuity": ["Removing the guard makes the test fail."],
        "safety": ["Dry-run and backup are required."]
        if risk == "durable-mutation" or kind == "live_operation"
        else [],
        "closure": {
            "rule": "Close only with final-head evidence.",
            "disposition": "whole",
            "successor_required_for_partial": True,
        },
        "source_digest": "a" * 64,
    }
    return {
        "id": "polylogue-test",
        "metadata": {"acceptance_contract_v1": contract},
        "acceptance_criteria": mod.render(contract),
    }


def test_valid_contract_round_trips() -> None:
    assert mod.validate(_issue()) == []


def test_durable_mutation_requires_safety() -> None:
    issue = _issue(risk="durable-mutation")
    issue["metadata"]["acceptance_contract_v1"]["safety"] = []
    issue["acceptance_criteria"] = mod.render(issue["metadata"]["acceptance_contract_v1"])
    assert "durable-mutation requires safety clauses" in mod.validate(issue)


def test_placeholder_is_rejected() -> None:
    issue = _issue()
    issue["metadata"]["acceptance_contract_v1"]["outcome"] = "Figure out the route ..."
    issue["acceptance_criteria"] = mod.render(issue["metadata"]["acceptance_contract_v1"])
    assert any("placeholder" in error for error in mod.validate(issue))


def test_render_drift_is_rejected() -> None:
    issue = _issue()
    issue["acceptance_criteria"] = "weaker prose"
    assert "acceptance_criteria drifted from structured contract" in mod.validate(issue)

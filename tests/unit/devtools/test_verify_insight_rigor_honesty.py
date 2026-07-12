from __future__ import annotations

import json
from dataclasses import replace

import pytest

from devtools import verify_insight_rigor_honesty


def test_insight_rigor_honesty_passes_when_every_product_is_covered(capsys: pytest.CaptureFixture[str]) -> None:
    assert verify_insight_rigor_honesty.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["uncovered_insight_names"] == []
    assert payload["missing_numeric_field_coverage"] == []
    assert payload["missing_numeric_item_models"] == []


def test_insight_rigor_honesty_fails_when_a_contract_is_monkeypatched_out(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Removing a registered insight's contract must fail this policy check,
    not silently pass (9e5.28's core regression: a contract-less product used
    to vanish from the audit instead of showing as uncovered)."""
    from polylogue.insights import rigor as rigor_mod

    original = rigor_mod._RIGOR_MATRIX
    monkeypatch.setattr(
        rigor_mod,
        "_RIGOR_MATRIX",
        tuple(c for c in original if c.insight_name != "session_profiles"),
    )

    assert verify_insight_rigor_honesty.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "session_profiles" in payload["uncovered_insight_names"]


def test_insight_rigor_honesty_fails_when_a_registered_item_model_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The policy cannot inspect fields if a production descriptor omits its model."""

    from polylogue.insights.registry import INSIGHT_REGISTRY

    original = INSIGHT_REGISTRY["cost_rollups"]
    monkeypatch.setitem(INSIGHT_REGISTRY, "cost_rollups", replace(original, item_model=None))

    assert verify_insight_rigor_honesty.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["missing_numeric_item_models"] == ["cost_rollups"]

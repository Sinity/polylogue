"""Anti-vacuity tests for the disposable yeq.1 safety-case lab."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from polylogue.core.outcomes import OutcomeStatus


def test_safety_case_artifact_covers_all_durable_tiers_and_hazards() -> None:
    from devtools.safety_case_scenario import _artifact

    artifact = _artifact()
    tiers = cast(list[dict[str, object]], artifact["tiers"])
    hazards = cast(list[dict[str, object]], artifact["hazards"])
    assert artifact["version"] == 1
    assert {row["name"] for row in tiers} == {
        "source.db",
        "index.db",
        "embeddings.db",
        "user.db",
        "audit.db",
        "ops.db",
    }
    assert {row["id"] for row in hazards} == {
        "wrong-bytes",
        "writer-divergence",
        "destructive-loss",
        "restore-discontinuity",
        "false-readiness",
    }
    for hazard in hazards:
        assert all(hazard[field] for field in ("preventive_invariant", "detection", "recovery", "receipt", "owner"))


def test_cursor_model_uses_real_store_and_preserves_exclusion_on_reordered_retry() -> None:
    from devtools.safety_case_scenario import _cursor_model_sequence

    result = _cursor_model_sequence()
    assert result == {
        "states": ["retry_pending", "active", "excluded", "excluded"],
        "committed_evidence_retained": True,
    }


def test_safety_case_report_is_disposable_and_machine_readable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import devtools.safety_case_scenario as scenario

    monkeypatch.setattr(scenario, "_artifact", lambda: {"version": 1, "tiers": [1] * 6, "hazards": [1] * 5})
    monkeypatch.setattr(scenario, "_cursor_model_sequence", lambda: {"committed_evidence_retained": True})
    passed = SimpleNamespace(all_passed=True, failed_stages=(), diverging_tables=())
    monkeypatch.setattr(scenario, "run_storage_correctness", lambda report_dir=None: passed)
    monkeypatch.setattr(scenario, "run_rebuild_safety", lambda: passed)
    monkeypatch.setattr(scenario, "run_rebuild_differential", lambda: passed)

    result = scenario.run_safety_case(report_dir=tmp_path)
    payload = json.loads((tmp_path / "safety-case-v1.json").read_text(encoding="utf-8"))
    assert result.all_passed
    assert result.stage_statuses()["incremental_matches_full"] is OutcomeStatus.OK
    assert payload["artifact"] == "docs/safety-case-v1.json"
    assert payload["checks"]["full_rebuild_rerun"] is True

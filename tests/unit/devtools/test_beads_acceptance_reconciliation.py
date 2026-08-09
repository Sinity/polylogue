from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

MODULE_PATH = Path(__file__).parents[3] / "devtools" / "beads_acceptance_reconciliation.py"
spec = importlib.util.spec_from_file_location("beads_acceptance_reconciliation", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


@pytest.fixture(autouse=True)
def _test_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep synthetic tests on the same reconciliation entry point with a tiny ratchet."""
    monkeypatch.setattr(mod._contracts, "load_manifest", lambda path: ("polylogue-test",))
    monkeypatch.setattr(mod._contracts, "validate_route_registry", lambda ids: [])
    monkeypatch.setattr(
        mod._contracts,
        "resolve_route",
        lambda identifier: (
            {
                "bead_id": identifier.removeprefix("test/"),
                "class": "ImplementationRoute",
                "contract_type": "implementation",
                "dispatch": "production",
                "targets": ["Test production route."],
            }
            if isinstance(identifier, str) and identifier.startswith("test/")
            else None
        ),
    )


def _issue(*, bead_id: str = "polylogue-test", updated_at: str = "2026-08-07T00:00:00Z") -> dict[str, Any]:
    issue: dict[str, Any] = {
        "id": bead_id,
        "title": "A real contract source",
        "description": "A source description with observable scope.",
        "design": "The design is recorded.",
        "notes": "The notes are stable.",
        "status": "open",
        "priority": 1,
        "issue_type": "task",
        "updated_at": updated_at,
        "dependencies": [{"depends_on_id": "polylogue-parent", "type": "blocks"}],
        "comments": [{"id": "comment-1", "body": "Preserve this comment."}],
        "metadata": {"operator": "preserve-me"},
        "acceptance_criteria": None,
    }
    contract: dict[str, Any] = {
        "schema_version": 1,
        "bead_id": bead_id,
        "contract_type": "implementation",
        "risk": "ordinary",
        "confidence": "high",
        "outcome": "The behavior is observable through the production route.",
        "routes": ["Test production route."],
        "evidence": ["A source description with observable scope."],
        "retained_scope": [],
        "verification": ["Run a focused regression.", "Run `devtools verify` for the affected baseline."],
        "anti_vacuity": ["Removing the guard makes the regression fail."],
        "safety": [],
        "route_spec": {
            "mode": "named",
            "identifier": f"test/{bead_id}",
            "class": "ImplementationRoute",
            "dispatch": "production",
        },
        "verification_route": {"manager": "devtools", "focused": "devtools test", "default": "devtools verify"},
        "closure": {
            "rule": "Close only with final-head evidence.",
            "disposition": "whole-or-explicit-partial",
            "successor_required_for_partial": True,
        },
    }
    contract["source_digest"] = mod.source_digest(issue)
    contract["dependency_digest"] = mod._contracts.dependency_digest(issue)
    contract["evidence_spans"] = [
        {
            "source_field": "description",
            "snapshot": issue["description"],
            "snapshot_digest": hashlib.sha256(issue["description"].encode("utf-8")).hexdigest(),
            "range": {"start": 0, "end": len(contract["evidence"][0])},
            "text_digest": hashlib.sha256(contract["evidence"][0].encode("utf-8")).hexdigest(),
        }
    ]
    issue["metadata"]["acceptance_contract_v1"] = contract
    issue["acceptance_criteria"] = mod.render(contract)
    return issue


def _write_export(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_same_timestamp_different_is_reported_and_refused(tmp_path: Path) -> None:
    master = _issue()
    live = copy.deepcopy(master)
    live["description"] = "A changed live source description."
    live["acceptance_criteria"] = None
    live["metadata"] = {"operator": "preserve-me"}
    repository_path = tmp_path / "repository.jsonl"
    live_path = tmp_path / "live.jsonl"
    _write_export(repository_path, [master])
    _write_export(live_path, [live])

    report, wave = mod.reconcile(repository_path, live_path)

    assert report["ids"]["same_timestamp_different"] == ["polylogue-test"]
    assert report["ids"]["contract_refused"] == ["polylogue-test"]
    assert report["contract_refused_denominator"] == 1
    assert wave == []


def test_live_newer_source_equivalent_row_is_deferred_without_wave_record(tmp_path: Path) -> None:
    master = _issue()
    live = copy.deepcopy(master)
    live["updated_at"] = "2026-08-08T00:00:00Z"
    live["acceptance_criteria"] = None
    live["metadata"] = {"operator": "preserve-me"}
    repository_path = tmp_path / "repository.jsonl"
    live_path = tmp_path / "live.jsonl"
    _write_export(repository_path, [master])
    _write_export(live_path, [live])

    report, wave = mod.reconcile(repository_path, live_path)

    assert report["ids"]["live_newer"] == ["polylogue-test"]
    assert report["ids"]["contract_refused"] == []
    assert report["contract_deferred_denominator"] == 1
    assert report["contract_deferred_reasons"] == {
        "polylogue-test": "live-newer record is excluded from the targeted wave; coordinator adjudication is required"
    }
    assert wave == []


def test_changed_source_refusal_names_the_denominator_and_id(tmp_path: Path) -> None:
    master = _issue()
    live = copy.deepcopy(master)
    live["notes"] = "A live source change requires adjudication."
    live["acceptance_criteria"] = None
    live["metadata"] = {"operator": "preserve-me"}
    repository_path = tmp_path / "repository.jsonl"
    live_path = tmp_path / "live.jsonl"
    _write_export(repository_path, [master])
    _write_export(live_path, [live])

    report, wave = mod.reconcile(repository_path, live_path)

    assert report["contract_refused_denominator"] == 1
    assert report["counts"]["contract_refused"] == 1
    assert report["ids"]["contract_refused"] == ["polylogue-test"]
    assert "source digest mismatch" in report["contract_refused_reasons"]["polylogue-test"][0]
    assert wave == []


def test_malformed_live_dependency_scalars_are_structured_refusals(tmp_path: Path) -> None:
    master = _issue()
    live = copy.deepcopy(master)
    live["dependencies"] = [
        {"depends_on_id": "polylogue-parent", "type": "blocks"},
        {"depends_on_id": 7, "type": "blocks"},
    ]
    live["acceptance_criteria"] = None
    live["metadata"] = {}
    repository_path = tmp_path / "repository.jsonl"
    live_path = tmp_path / "live.jsonl"
    _write_export(repository_path, [master])
    _write_export(live_path, [live])

    report, wave = mod.reconcile(repository_path, live_path)

    assert report["ids"]["contract_refused"] == ["polylogue-test"]
    assert report["contract_refused_reasons"]["polylogue-test"] == [
        "dependencies[1].depends_on_id must be a string or null (got int)"
    ]
    assert wave == []


def test_guarded_wave_preserves_live_dependencies_comments_status_and_timestamp(tmp_path: Path) -> None:
    master = _issue(updated_at="2026-08-07T00:00:00Z")
    live = copy.deepcopy(master)
    live["updated_at"] = "2026-08-06T00:00:00Z"
    live["status"] = "in_progress"
    live["comments"] = [{"id": "comment-live", "body": "Later live comment."}]
    live["acceptance_criteria"] = None
    live["metadata"] = {"operator": "preserve-me", "other": {"value": 3}}
    repository_path = tmp_path / "repository.jsonl"
    live_path = tmp_path / "live.jsonl"
    _write_export(repository_path, [master])
    _write_export(live_path, [live])

    _, wave = mod.reconcile(repository_path, live_path)

    assert len(wave) == 1
    target = wave[0]
    for key in ("dependencies", "comments", "status", "updated_at"):
        assert target[key] == live[key]
    assert target["metadata"]["operator"] == live["metadata"]["operator"]
    assert target["metadata"]["other"] == live["metadata"]["other"]
    assert target["acceptance_criteria"] == master["acceptance_criteria"]
    assert target["metadata"]["acceptance_contract_v1"] == master["metadata"]["acceptance_contract_v1"]


def test_master_only_and_live_only_are_reported_without_wave_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod._contracts, "load_manifest", lambda path: ("polylogue-master-only", "polylogue-shared"))
    master = _issue(bead_id="polylogue-master-only")
    live = _issue(bead_id="polylogue-shared")
    live["acceptance_criteria"] = None
    live["metadata"] = {}
    live_only = {"id": "polylogue-live-only", "title": "Later live work", "updated_at": "2026-08-09T00:00:00Z"}
    repository_path = tmp_path / "repository.jsonl"
    live_path = tmp_path / "live.jsonl"
    _write_export(repository_path, [master, _issue(bead_id="polylogue-shared")])
    _write_export(live_path, [live, live_only])

    report, wave = mod.reconcile(repository_path, live_path)

    assert report["ids"]["master_only"] == ["polylogue-master-only"]
    assert report["ids"]["live_only"] == ["polylogue-live-only"]
    assert [row["id"] for row in wave] == ["polylogue-shared"]


def test_malformed_live_metadata_is_an_explicit_contract_refusal(tmp_path: Path) -> None:
    master = _issue()
    live = copy.deepcopy(master)
    live["metadata"] = "{malformed"
    live["acceptance_criteria"] = None
    repository_path = tmp_path / "repository.jsonl"
    live_path = tmp_path / "live.jsonl"
    _write_export(repository_path, [master])
    _write_export(live_path, [live])

    report, wave = mod.reconcile(repository_path, live_path)

    assert report["ids"]["contract_refused"] == ["polylogue-test"]
    assert report["contract_refused_reasons"]["polylogue-test"] == ["live metadata is not a JSON object"]
    assert wave == []


def test_malformed_timestamp_is_an_explicit_contract_refusal(tmp_path: Path) -> None:
    master = _issue()
    live = copy.deepcopy(master)
    live["updated_at"] = None
    live["acceptance_criteria"] = None
    live["metadata"] = {}
    repository_path = tmp_path / "repository.jsonl"
    live_path = tmp_path / "live.jsonl"
    _write_export(repository_path, [master])
    _write_export(live_path, [live])

    report, wave = mod.reconcile(repository_path, live_path)

    assert report["ids"]["contract_refused"] == ["polylogue-test"]
    assert report["contract_refused_reasons"]["polylogue-test"] == [
        "updated_at must be a string on both repository and live records"
    ]
    assert wave == []


def test_malformed_string_timestamp_cannot_authorize_a_wave(tmp_path: Path) -> None:
    """Production dependency: reconcile -> _classify_timestamp; catches lexicographic stale-wave authorization."""
    master = _issue()
    live = copy.deepcopy(master)
    live["updated_at"] = "z"
    live["acceptance_criteria"] = None
    live["metadata"] = {}
    repository_path = tmp_path / "repository.jsonl"
    live_path = tmp_path / "live.jsonl"
    _write_export(repository_path, [master])
    _write_export(live_path, [live])

    report, wave = mod.reconcile(repository_path, live_path)

    assert report["ids"]["contract_refused"] == ["polylogue-test"]
    assert report["contract_refused_reasons"]["polylogue-test"] == [
        "updated_at must be a valid canonical Beads timestamp"
    ]
    assert wave == []


def test_timestamp_comparison_preserves_fractional_precision(tmp_path: Path) -> None:
    """Production dependency: reconcile -> _classify_timestamp; catches precision loss in timestamp ordering."""
    master = _issue(updated_at="2026-08-07T00:00:00.1234567Z")
    live = copy.deepcopy(master)
    live["updated_at"] = "2026-08-07T00:00:00.1234568Z"
    live["acceptance_criteria"] = None
    live["metadata"] = {}
    repository_path = tmp_path / "repository.jsonl"
    live_path = tmp_path / "live.jsonl"
    _write_export(repository_path, [master])
    _write_export(live_path, [live])

    report, wave = mod.reconcile(repository_path, live_path)

    assert report["ids"]["live_newer"] == ["polylogue-test"]
    assert wave == []


def test_reconciliation_refuses_a_partial_manifest_before_wave_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production dependency: reconcile -> load_manifest; catches len(master_contracts) replacing the 218 denominator."""
    monkeypatch.setattr(mod._contracts, "load_manifest", lambda path: ("polylogue-test", "polylogue-required"))
    master = _issue()
    live = copy.deepcopy(master)
    live["acceptance_criteria"] = None
    live["metadata"] = {}
    repository_path = tmp_path / "repository.jsonl"
    live_path = tmp_path / "live.jsonl"
    _write_export(repository_path, [master])
    _write_export(live_path, [live])

    with pytest.raises(mod.ReconciliationError, match="manifest is incomplete"):
        mod.reconcile(repository_path, live_path)


def test_null_metadata_representation_is_preserved_by_non_contract_digest(tmp_path: Path) -> None:
    master = _issue()
    live = copy.deepcopy(master)
    live["metadata"] = None
    live["acceptance_criteria"] = None
    repository_path = tmp_path / "repository.jsonl"
    live_path = tmp_path / "live.jsonl"
    _write_export(repository_path, [master])
    _write_export(live_path, [live])

    _, wave = mod.reconcile(repository_path, live_path)

    assert len(wave) == 1
    assert mod.non_contract_equality_digest({live["id"]: live}, [live["id"]]) == mod.non_contract_equality_digest(
        {wave[0]["id"]: wave[0]}, [wave[0]["id"]]
    )
    assert wave[0]["metadata"] != live["metadata"]


def test_post_import_equality_digest_rejects_unrelated_mutation(tmp_path: Path) -> None:
    master = _issue()
    before = copy.deepcopy(master)
    before["acceptance_criteria"] = None
    before["metadata"] = {"operator": "preserve-me"}
    repository_path = tmp_path / "repository.jsonl"
    before_path = tmp_path / "before.jsonl"
    wave_path = tmp_path / "wave.jsonl"
    after_path = tmp_path / "after.jsonl"
    _write_export(repository_path, [master])
    _write_export(before_path, [before])
    _, wave = mod.reconcile(repository_path, before_path)
    _write_export(wave_path, wave)
    after = copy.deepcopy(wave[0])
    after["status"] = "closed"
    _write_export(after_path, [after])

    try:
        report, wave = mod.reconcile(repository_path, before_path)
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        mod.verify_post_import(
            repository=repository_path, before=before_path, after=after_path, wave=wave_path, report=report_path
        )
    except mod.ReconciliationError as exc:
        assert "non-contract fields" in str(exc)
    else:
        raise AssertionError("post-import equality must reject status mutation")


def test_post_import_equality_digest_rejects_new_live_record(tmp_path: Path) -> None:
    master = _issue()
    before = copy.deepcopy(master)
    before["acceptance_criteria"] = None
    before["metadata"] = {}
    repository_path = tmp_path / "repository.jsonl"
    before_path = tmp_path / "before.jsonl"
    wave_path = tmp_path / "wave.jsonl"
    after_path = tmp_path / "after.jsonl"
    _write_export(repository_path, [master])
    _write_export(before_path, [before])
    _, wave = mod.reconcile(repository_path, before_path)
    _write_export(wave_path, wave)
    after = copy.deepcopy(wave[0])
    after["id"] = "polylogue-live-added"
    _write_export(after_path, [wave[0], after])

    try:
        report, wave = mod.reconcile(repository_path, before_path)
        report_path = tmp_path / "report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        mod.verify_post_import(
            repository=repository_path, before=before_path, after=after_path, wave=wave_path, report=report_path
        )
    except mod.ReconciliationError as exc:
        assert "record universe" in str(exc)
    else:
        raise AssertionError("post-import equality must reject a newly added record")


def test_cli_returns_nonzero_when_canonical_contract_validation_fails(tmp_path: Path) -> None:
    master = _issue()
    master["metadata"]["acceptance_contract_v1"]["source_digest"] = "0" * 64
    live = copy.deepcopy(master)
    repository_path = tmp_path / "repository.jsonl"
    live_path = tmp_path / "live.jsonl"
    _write_export(repository_path, [master])
    _write_export(live_path, [live])

    assert mod.main(["--repository", str(repository_path), "--live", str(live_path), "--json"]) == 1


def test_post_import_rejects_a_modified_wave_contract(tmp_path: Path) -> None:
    master = _issue()
    before = copy.deepcopy(master)
    before["acceptance_criteria"] = None
    before["metadata"] = {}
    repository_path = tmp_path / "repository.jsonl"
    before_path = tmp_path / "before.jsonl"
    wave_path = tmp_path / "wave.jsonl"
    report_path = tmp_path / "report.json"
    after_path = tmp_path / "after.jsonl"
    _write_export(repository_path, [master])
    _write_export(before_path, [before])
    report, wave = mod.reconcile(repository_path, before_path)
    wave[0]["acceptance_criteria"] = "altered after reconciliation"
    _write_export(wave_path, wave)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    _write_export(after_path, wave)

    with pytest.raises(mod.ReconciliationError, match="targeted wave digest"):
        mod.verify_post_import(
            repository=repository_path,
            before=before_path,
            after=after_path,
            wave=wave_path,
            report=report_path,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda report: report["ids"]["live_only"].append("polylogue-unexpected"),
        lambda report: report["counts"].update(live_only=1),
        lambda report: report["contract_refused_reasons"].update({"polylogue-test": ["tampered refusal"]}),
        lambda report: report["contract_deferred_reasons"].update({"polylogue-test": "tampered deferral"}),
        lambda report: report["already_guarded_ids"].append("polylogue-test"),
    ],
    ids=["category", "count", "refusal-reasons", "deferred-reasons", "already-guarded"],
)
def test_post_import_rejects_tampered_complete_report(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], None]
) -> None:
    master = _issue()
    before = copy.deepcopy(master)
    before["acceptance_criteria"] = None
    before["metadata"] = {}
    repository_path = tmp_path / "repository.jsonl"
    before_path = tmp_path / "before.jsonl"
    wave_path = tmp_path / "wave.jsonl"
    report_path = tmp_path / "report.json"
    after_path = tmp_path / "after.jsonl"
    _write_export(repository_path, [master])
    _write_export(before_path, [before])
    report, wave = mod.reconcile(repository_path, before_path)
    _write_export(wave_path, wave)
    _write_export(after_path, wave)
    mutation(report)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(mod.ReconciliationError, match="canonical recomputation"):
        mod.verify_post_import(
            repository=repository_path,
            before=before_path,
            after=after_path,
            wave=wave_path,
            report=report_path,
        )


def test_post_import_revalidates_stale_before_source(tmp_path: Path) -> None:
    master = _issue()
    before = copy.deepcopy(master)
    before["acceptance_criteria"] = None
    before["metadata"] = {}
    repository_path = tmp_path / "repository.jsonl"
    before_path = tmp_path / "before.jsonl"
    wave_path = tmp_path / "wave.jsonl"
    report_path = tmp_path / "report.json"
    after_path = tmp_path / "after.jsonl"
    _write_export(repository_path, [master])
    _write_export(before_path, [before])
    report, wave = mod.reconcile(repository_path, before_path)
    _write_export(wave_path, wave)
    before["title"] = "stale live source"
    report["live_population_digest"] = mod.equality_digest({before["id"]: before})
    _write_export(before_path, [before])
    _write_export(after_path, wave)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(mod.ReconciliationError, match="stale source digest"):
        mod.verify_post_import(
            repository=repository_path,
            before=before_path,
            after=after_path,
            wave=wave_path,
            report=report_path,
        )


def test_post_import_rejects_reordered_wave_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mod._contracts, "load_manifest", lambda path: ("polylogue-a", "polylogue-b"))
    master_a = _issue(bead_id="polylogue-a")
    master_b = _issue(bead_id="polylogue-b")
    before_a = copy.deepcopy(master_a)
    before_b = copy.deepcopy(master_b)
    for row in (before_a, before_b):
        row["acceptance_criteria"] = None
        row["metadata"] = {}
    repository_path = tmp_path / "repository.jsonl"
    before_path = tmp_path / "before.jsonl"
    wave_path = tmp_path / "wave.jsonl"
    report_path = tmp_path / "report.json"
    after_path = tmp_path / "after.jsonl"
    _write_export(repository_path, [master_a, master_b])
    _write_export(before_path, [before_a, before_b])
    report, wave = mod.reconcile(repository_path, before_path)
    _write_export(wave_path, list(reversed(wave)))
    _write_export(after_path, wave)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(mod.ReconciliationError, match="order or population"):
        mod.verify_post_import(
            repository=repository_path,
            before=before_path,
            after=after_path,
            wave=wave_path,
            report=report_path,
        )

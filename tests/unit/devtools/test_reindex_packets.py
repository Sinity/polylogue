"""Miniature graphs for the reindex packet projection."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any, cast

import pytest

from devtools.reindex_packets import ROOT_ID, _record, _task_revision, main, validate

HEAD = "0123456789abcdef0123456789abcdef01234567"


class FakeReader:
    def __init__(self, beads: list[dict[str, Any]]) -> None:
        self.beads = tuple(beads)

    def read(self) -> tuple[dict[str, Any], ...]:
        return tuple(_record(bead) for bead in self.beads)


def dep(target: str, kind: str = "blocks") -> dict[str, str]:
    return {"depends_on_id": target, "type": kind}


def meta(**changes: Any) -> dict[str, Any]:
    value = {
        "campaign_id": "reindex-2026",
        "execution_shape": "leaf",
        "execution_wave": "reindex-prep-a",
        "execution_lane": "lane",
        "lane_packet": "1",
        "lane_order": "1",
        "affected_paths": "polylogue/example.py; tests/unit/test_example.py",
        "conflict_keys": "one; two",
        "write_scope": "free-form scope",
        "verification_commands": "devtools test",
        "model_policy": "provider-neutral-capability-v1",
        "worker_model_class": "cheap-capable",
        "review_model_class": "strong-review",
        "live_data_access": "synthetic",
        "execution_kind": "implementation",
        "decision_closure": "closed-by-spec",
        "necessity_class": "required",
        "judgment_class": "mechanical",
        "tdd_mode": "focused",
        "tdd_packet": "focused-v1",
        "packet_intent": "implement the example packet",
        "integration_intent": "one coherent example batch",
    }
    value.update(changes)
    return value


def bead(
    bead_id: str, metadata: dict[str, Any], dependencies: tuple[dict[str, str], ...] = (), status: str = "open"
) -> dict[str, Any]:
    return {
        "_type": "issue",
        "id": bead_id,
        "issue_type": "task",
        "status": status,
        "priority": 1,
        "title": f"Task {bead_id}",
        "description": f"Description for task {bead_id}",
        "design": "",
        "acceptance_criteria": "",
        "notes": "",
        "created_at": "2026-08-25T05:00:00Z",
        "updated_at": "2026-08-25T06:00:00Z",
        "created_by": "Sinity",
        "owner": "Sinity",
        "labels": ["campaign:reindex-2026"],
        "metadata": metadata,
        "dependencies": list(dependencies),
        "comment_count": 0,
        "dependency_count": len(dependencies),
        "dependent_count": 0,
    }


def operation_metadata(shape: str = "plan-rehearse-review-authorize-apply-verify", **changes: Any) -> dict[str, Any]:
    value = {
        "packet_execution_contract": "packet-v1",
        "deadline_policy": "bounded",
        "execution_kind": "authorized-prep-operation",
        "live_data_access": "explicit-operator-authorized-source-apply",
        "lane_mode": "serialized-writer",
        "operation_phase_contract": {"version": "prep-operation-phase-v2", "shape": shape},
        "initial_job_authority": {"mode": "read-only-plan-rehearsal"},
        "apply_authority": {"mode": "explicit-operator-authorized-apply"},
    }
    value.update(changes)
    return meta(**value)


def operation_evidence(
    *,
    plan_digest: str = "sha256:plan-1",
    rehearsal_state: str = "accepted",
    rehearsal_digest: str | None = None,
    authorization_state: str = "authorized",
    authorization_digest: str | None = None,
    review_state: str = "accepted",
    review_digest: str | None = None,
    expires_at: str = "2026-08-25T07:00:00+00:00",
) -> dict[str, Any]:
    return {
        "observed_at": "2026-08-25T06:00:00+00:00",
        "plan_digest": plan_digest,
        "rehearsal": {
            "evidence_id": "rehearsal-1",
            "state": rehearsal_state,
            "plan_digest": rehearsal_digest or plan_digest,
        },
        "authorization": {
            "evidence_id": "authorization-1",
            "state": authorization_state,
            "plan_digest": authorization_digest or plan_digest,
            "expires_at": expires_at,
        },
        "review": {"evidence_id": "review-1", "state": review_state, "plan_digest": review_digest or plan_digest},
    }


def graph(*, operation: bool = False) -> list[dict[str, Any]]:
    root = bead(ROOT_ID, {"campaign_id": "reindex-2026", "execution_shape": "gate"}, (dep("gate"),))
    gate = bead("gate", {"campaign_id": "reindex-2026", "execution_shape": "gate"}, tuple(dep(item) for item in "abc"))
    members = [
        bead(
            "a",
            operation_metadata()
            if operation
            else meta(packet_execution_contract="packet-v1", deadline_policy="bounded"),
        ),
        bead("b", meta(lane_order="2"), (dep("a"),)),
        bead("c", meta(lane_order="3"), (dep("b"),)),
    ]
    return [root, gate, *members]


def packet(report: dict[str, Any]) -> dict[str, Any]:
    return cast("dict[str, Any]", report["packets"][0])


def launch(report: dict[str, Any], phase: str) -> dict[str, Any]:
    return next(item for item in packet(report)["launches"] if item["selected_phase"] == phase)


def test_ordinary_packet_has_one_coherent_launch_projection() -> None:
    report = validate(FakeReader(graph()), integration_head=HEAD)
    item = packet(report)
    assert report["ok"] and item["ready"]
    assert item["selected_phase"] == "ordinary"
    assert [item["selected_phase"] for item in item["launches"]] == ["ordinary"]
    assert launch(report, "ordinary")["effective_authority"]["mode"] == "ordinary-launch"


def test_blocks_closure_and_gate_carriers_remain_diagnostic() -> None:
    beads = graph() + [bead("related", meta())]
    beads[1]["dependencies"] = [*beads[1]["dependencies"], dep("related", "relates-to")]
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert "related" not in report["blocks_only_closure"]
    assert "related" in report["mixed_relation_expansion"]
    assert report["differences"]["campaign_only_ids"] == ["related"]

    beads[1]["metadata"]["worker_model_class"] = "worker"
    assert (
        "gate: gate carries worker_model_class"
        in validate(FakeReader(beads), integration_head=HEAD)["structural_errors"]
    )


def test_packet_topology_conflicts_and_missing_launch_contract_stay_enforced() -> None:
    beads = graph()
    beads[2]["metadata"]["lane_order"] = "2"
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert any("internal blocker is not earlier" in error for error in report["structural_errors"])

    beads = graph()
    beads[3]["metadata"] = meta(execution_lane="other", lane_order="2", conflict_keys="one")
    beads[3]["dependencies"] = []
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert any("exact conflict-key overlap" in error for error in report["structural_errors"])
    assert all(not item["ready"] for item in report["packets"])

    beads = graph()
    beads[2]["metadata"].pop("packet_execution_contract")
    projection = launch(validate(FakeReader(beads), integration_head=HEAD), "ordinary")
    assert {failure["field"] for failure in projection["launch_failures"]} == {"packet_execution_contract"}


def test_closed_and_open_external_blockers_remain_packet_readiness() -> None:
    beads = graph()
    beads.append(bead("outside", {"campaign_id": "reindex-2026", "execution_shape": "gate"}, status="closed"))
    beads[2]["dependencies"] = [dep("outside")]
    assert packet(validate(FakeReader(beads), integration_head=HEAD))["ready"]

    beads[-1]["status"] = "open"
    projection = launch(validate(FakeReader(beads), integration_head=HEAD), "ordinary")
    assert {"kind": "blocks", "reason": "open", "bead_id": "outside"} in projection["launch_failures"]


def test_candidate_and_final_window_authority_keep_their_existing_serialization_rules() -> None:
    beads = graph()
    beads[2]["metadata"]["live_data_access"] = "isolated-candidate-write"
    assert (
        "a: candidate writer is not serialized"
        in validate(FakeReader(beads), integration_head=HEAD)["structural_errors"]
    )

    beads = graph()
    for item in beads[2:]:
        item["metadata"]["execution_wave"] = "reindex-window"
    beads[2]["metadata"].update(
        execution_kind="authorized-prep-operation", live_data_access="active-authorized", lane_mode="parallel"
    )
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert not any("requires operator-authorized live_data_access" in error for error in report["structural_errors"])
    assert any("operator-authorized writer is not serialized" in error for error in report["structural_errors"])


def test_legacy_readiness_census_remains_diagnostic_only() -> None:
    beads = graph()
    beads[2]["metadata"]["dispatch_readiness"] = "blocked-on-unknown; needs a note"
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert report["legacy_readiness_census"] == {
        "dispatch_readiness": {"count": 1, "record_ids": ["a"]},
        "program_dispatch_readiness": {"count": 0, "record_ids": []},
    }


def test_operation_initial_launch_breaks_the_static_evidence_catch_22() -> None:
    report = validate(FakeReader(graph(operation=True)), integration_head=HEAD)
    initial, apply = launch(report, "initial"), launch(report, "apply")
    assert packet(report)["ready"] and initial["ready"]
    assert initial["effective_authority"] == {
        "mode": "read-only-plan-rehearsal",
        "allowed_actions": ["read-only-plan", "isolated-rehearsal"],
        "live_authority": "none",
        "may_apply": False,
    }
    assert not apply["ready"]
    assert apply["launch_failures"] == [{"kind": "operation-evidence", "reason": "missing", "bead_id": "a"}]


def test_initial_phase_token_cannot_authorize_apply() -> None:
    report = validate(FakeReader(graph(operation=True)), integration_head=HEAD)
    initial, apply = launch(report, "initial"), launch(report, "apply")
    assert initial["context_digest"] != apply["context_digest"]
    assert initial["phase_token"] != apply["phase_token"]
    assert ":initial:" in initial["phase_token"] and ":apply:" in apply["phase_token"]
    assert initial["effective_authority"]["may_apply"] is False
    assert apply["effective_authority"]["may_apply"] is False


def test_apply_phase_token_binds_the_runtime_plan_digest() -> None:
    beads = graph(operation=True)
    first = validate(
        FakeReader(beads), integration_head=HEAD, selected_phase="apply", operation_evidence={"a": operation_evidence()}
    )
    second = validate(
        FakeReader(beads),
        integration_head=HEAD,
        selected_phase="apply",
        operation_evidence={"a": operation_evidence(plan_digest="sha256:plan-2")},
    )
    assert launch(first, "apply")["context_digest"] != launch(second, "apply")["context_digest"]
    assert launch(first, "apply")["operation_evidence"]["a"]["plan_digest"] == "sha256:plan-1"


def test_apply_requires_accepted_same_digest_rehearsal_authorization_and_review() -> None:
    beads = graph(operation=True)
    missing = validate(FakeReader(beads), integration_head=HEAD, selected_phase="apply")
    assert packet(missing)["ready"] is False
    report = validate(
        FakeReader(beads), integration_head=HEAD, selected_phase="apply", operation_evidence={"a": operation_evidence()}
    )
    assert packet(report)["selected_phase"] == "apply"
    assert packet(report)["ready"] and launch(report, "apply")["ready"]
    assert launch(report, "apply")["effective_authority"]["may_apply"] is True


@pytest.mark.parametrize(
    ("evidence", "reason"),
    [
        (operation_evidence(rehearsal_digest="sha256:other"), "mismatched-plan-digest"),
        (operation_evidence(authorization_digest="sha256:other"), "mismatched-plan-digest"),
        (operation_evidence(review_digest="sha256:other"), "mismatched-plan-digest"),
        (operation_evidence(rehearsal_state="stale"), "stale"),
        (operation_evidence(authorization_state="revoked"), "revoked"),
        (operation_evidence(expires_at="2026-08-25T05:00:00+00:00"), "expired"),
    ],
)
def test_apply_mismatched_or_stale_runtime_evidence_fails_closed(evidence: dict[str, Any], reason: str) -> None:
    report = validate(
        FakeReader(graph(operation=True)),
        integration_head=HEAD,
        selected_phase="apply",
        operation_evidence={"a": evidence},
    )
    apply = launch(report, "apply")
    assert not packet(report)["ready"]
    assert reason in {failure["reason"] for failure in apply["launch_failures"]}


def test_shape_declares_when_apply_needs_independent_review() -> None:
    beads = graph(operation=True)
    beads[2]["metadata"] = operation_metadata(shape="accepted-plan-rehearse-authorize-apply-verify")
    evidence = operation_evidence()
    evidence["review"] = None
    report = validate(
        FakeReader(beads), integration_head=HEAD, selected_phase="apply", operation_evidence={"a": evidence}
    )
    assert packet(report)["ready"]


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"operation_phase_contract": {"version": "prep-operation-phase-v2", "shape": "bad"}}, "invalid phase shape"),
        (
            {
                "operation_phase_contract": {
                    "version": "prep-operation-phase-v2",
                    "shape": "plan-rehearse-review-authorize-apply-verify",
                    "plan_digest": "sha256:x",
                }
            },
            "invalid operation_phase_contract",
        ),
        (
            {"initial_job_authority": {"mode": "read-only-plan-rehearsal", "job_id": "job-1"}},
            "invalid initial_job_authority",
        ),
        (
            {"apply_authority": {"mode": "explicit-operator-authorized-apply", "plan_digest": "sha256:x"}},
            "invalid apply_authority",
        ),
        ({"plan_digest": "sha256:x"}, "stores runtime policy field"),
    ],
)
def test_operation_policy_rejects_runtime_fields_in_static_contract(changes: dict[str, Any], expected: str) -> None:
    beads = graph(operation=True)
    beads[2]["metadata"].update(changes)
    assert any(expected in error for error in validate(FakeReader(beads), integration_head=HEAD)["structural_errors"])


def test_legacy_static_readiness_is_rejected_for_operations_without_mutating_beads() -> None:
    beads = graph(operation=True)
    beads[2]["metadata"]["readiness_contract"] = {"old": "runtime state"}
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert any("must not store runtime readiness_contract" in error for error in report["structural_errors"])
    assert beads[2]["metadata"]["readiness_contract"] == {"old": "runtime state"}


def test_task_revision_binds_unknown_semantic_metadata_but_not_operational_bookkeeping() -> None:
    beads = graph()
    baseline = launch(validate(FakeReader(beads), integration_head=HEAD), "ordinary")
    baseline_identity, baseline_context = baseline["task_identities"][0]["revision"], baseline["context_digest"]
    beads[2]["metadata"].update(
        active_job_id="job-1",
        active_backend="agentctl",
        active_model="x",
        review_state="pending",
        review_result="none",
        cancelled_job="job-0",
        correction_head="f" * 40,
    )
    bookkeeping = launch(validate(FakeReader(beads), integration_head=HEAD), "ordinary")
    assert bookkeeping["task_identities"][0]["revision"] == baseline_identity
    assert bookkeeping["context_digest"] == baseline_context
    beads[2]["metadata"]["unrecognized_semantic_policy"] = "changed"
    semantic = launch(validate(FakeReader(beads), integration_head=HEAD), "ordinary")
    assert semantic["task_identities"][0]["revision"] != baseline_identity
    assert semantic["context_digest"] != baseline_context


def test_real_bd_export_shape_is_parsed_and_exact_task_head_identity_is_projected() -> None:
    beads = graph()
    assert all("revision" not in record for record in beads)
    projection = launch(validate(FakeReader(beads), integration_head=HEAD), "ordinary")
    assert projection["integration_head"] == HEAD
    assert projection["task_identities"] == [
        {"bead_id": record["id"], "revision": _task_revision(_record(record))} for record in beads[2:]
    ]


def test_blocks_and_authority_writer_serialization_remain_launch_failures() -> None:
    beads = graph(operation=True)
    beads.append(bead("outside", {"campaign_id": "reindex-2026", "execution_shape": "gate"}, status="open"))
    beads[2]["dependencies"] = [dep("outside")]
    beads[2]["metadata"]["lane_mode"] = "parallel"
    report = validate(FakeReader(beads), integration_head=HEAD)
    reasons = {failure["reason"] for failure in launch(report, "initial")["launch_failures"]}
    assert any("must be serialized" in error for error in report["structural_errors"])
    assert "open" in reasons


@pytest.mark.parametrize(
    ("wave", "access"),
    [
        (wave, access)
        for wave in ("reindex-prep-a", "reindex-prep-b", "reindex-prep-c", "reindex-window")
        for access in (
            "active-authorized",
            "authorized-inactive-generation-writer",
            "explicit-operator-authorized-source-apply",
            "explicit-operator-authorized-blob-apply",
            "candidate-only",
        )
    ],
)
def test_all_authority_writers_remain_serialized_in_every_wave(wave: str, access: str) -> None:
    beads = graph()
    for item in beads[2:]:
        item["metadata"]["execution_wave"] = wave
    beads[2]["metadata"].update(live_data_access=access, lane_mode="parallel")
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert any("a:" in error and "not serialized" in error for error in report["structural_errors"])
    assert packet(report)["ready"] is False


def test_enforcement_selects_phase_while_diagnostic_remains_nonfailing() -> None:
    reader = FakeReader(graph(operation=True))
    assert main(["--enforce-readiness", "--integration-head", HEAD], reader=reader, stdout=StringIO()) == 0
    assert (
        main(["--enforce-readiness", "--phase", "apply", "--integration-head", HEAD], reader=reader, stdout=StringIO())
        == 1
    )
    assert main(["--diagnostic", "--phase", "apply", "--integration-head", HEAD], reader=reader, stdout=StringIO()) == 0


def test_reader_invocation_is_read_only_and_json_marks_it(monkeypatch: Any) -> None:
    calls: list[tuple[str, ...]] = []

    class Completed:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def run(command: list[str], **_: Any) -> Completed:
        calls.append(tuple(command))
        if command[0] == "git":
            return Completed("0" * 40 + "\n")
        return Completed(
            json.dumps(
                {
                    "id": ROOT_ID,
                    "status": "closed",
                    "labels": ["campaign:reindex-2026"],
                    "metadata": {"campaign_id": "reindex-2026", "execution_shape": "gate"},
                    "dependencies": [],
                }
            )
            + "\n"
        )

    monkeypatch.setattr("devtools.reindex_packets.subprocess.run", run)
    output = StringIO()
    assert main(["--json"], stdout=output) == 0
    assert calls == [
        ("git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "--verify", "HEAD"),
        ("bd", "--readonly", "export", "--all"),
    ]
    assert json.loads(output.getvalue())["read_only"] is True

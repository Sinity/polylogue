"""Miniature graphs for the reindex packet projection."""

from __future__ import annotations

import json
import os
import subprocess
from io import StringIO
from pathlib import Path
from typing import Any, cast

import pytest

import devtools.reindex_packets as reindex_packets
from devtools.reindex_packets import (
    APPLY_AUTHORITY_MODE,
    ROOT_ID,
    WAVES,
    ReindexPacketValidationError,
    _record,
    _task_revision,
    main,
    validate,
)


def checkout_head() -> str:
    return subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "--verify", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    ).stdout.strip()


HEAD = checkout_head()
GIT_ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}


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
        "operation_id": "a",
        "integration_head": HEAD,
        "packet_context_digest": "sha256:unbound",
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


def bound_operation_evidence(beads: list[dict[str, Any]], **changes: Any) -> dict[str, Any]:
    evidence = operation_evidence(**changes)
    return bind_diagnostic_evidence(beads, evidence)


def bind_diagnostic_evidence(beads: list[dict[str, Any]], evidence: dict[str, Any]) -> dict[str, Any]:
    evidence = json.loads(json.dumps(evidence))
    baseline = validate(FakeReader(beads), integration_head=HEAD)
    evidence["packet_context_digest"] = packet(baseline)["packet_context_digest"]
    return evidence


def graph(*, operation: bool = False) -> list[dict[str, Any]]:
    member_ids = "a" if operation else "abc"
    root = bead(ROOT_ID, {"campaign_id": "reindex-2026", "execution_shape": "gate"}, (dep("gate"),))
    gate = bead(
        "gate", {"campaign_id": "reindex-2026", "execution_shape": "gate"}, tuple(dep(item) for item in member_ids)
    )
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
    if operation:
        members = [
            bead(
                "a",
                operation_metadata(packet_size_exception="authorized-operation-singleton"),
            )
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
    assert any("requires operator-authorized live_data_access" in error for error in report["structural_errors"])
    assert any("requires a declared prep execution_wave" in error for error in report["structural_errors"])
    assert any("operator-authorized writer is not serialized" in error for error in report["structural_errors"])


def test_prep_operation_access_cannot_become_ordinary_by_wave_and_kind_edit() -> None:
    beads = graph()
    for item in beads[2:]:
        item["metadata"]["execution_wave"] = "reindex-window"
    beads[2]["metadata"].update(
        execution_kind="implementation",
        live_data_access="explicit-operator-authorized-source-apply",
        lane_mode="serialized-writer",
    )
    ordinary = launch(validate(FakeReader(beads), integration_head=HEAD), "ordinary")
    assert ordinary["launch_failures"] == [
        {
            "kind": "execution-kind",
            "reason": (
                "a: live_data_access explicit-operator-authorized-source-apply requires "
                "execution_kind authorized-prep-operation in a prep wave"
            ),
        }
    ]
    assert not ordinary["ready"]
    assert ordinary["effective_authority"]["live_authority"] == "none"


@pytest.mark.parametrize(
    "access",
    [
        "explicit-operator-authorized-apply",
        "explicit-operator-authorized--apply",
        " EXPLICIT-OPERATOR-AUTHORIZED-index-apply ",
        "explicit-operator-authorized-audit-apply",
        "explicit-operator-authorized-future-derived-tier-apply",
    ],
)
def test_every_reserved_apply_access_token_requires_the_prep_operation_regime(access: str) -> None:
    beads = graph()
    beads[2]["metadata"].update(execution_kind="implementation", live_data_access=access)
    ordinary = launch(validate(FakeReader(beads), integration_head=HEAD), "ordinary")
    assert not ordinary["ready"]
    assert any(failure["kind"] == "execution-kind" for failure in ordinary["launch_failures"])
    assert ordinary["effective_authority"]["live_authority"] == "none"

    beads = graph(operation=True)
    beads[2]["metadata"]["live_data_access"] = access
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert not report["structural_errors"]
    assert launch(report, "initial")["effective_authority"]["live_authority"] == "none"


@pytest.mark.parametrize(
    "access",
    [
        ["synthetic", "explicit-operator-authorized-source-apply"],
        {"access": "explicit-operator-authorized-source-apply"},
    ],
)
def test_non_string_live_authority_is_a_typed_packet_failure(access: object) -> None:
    beads = graph()
    beads[2]["metadata"]["live_data_access"] = access
    ordinary = launch(validate(FakeReader(beads), integration_head=HEAD), "ordinary")
    assert not ordinary["ready"]
    assert ordinary["effective_authority"]["live_authority"] == "none"
    assert {failure["kind"] for failure in ordinary["launch_failures"]} >= {"authority-shape"}


@pytest.mark.parametrize(
    ("membership", "structural_error", "reason"),
    [
        (
            "mixed",
            "operation packet mixes an authorized-prep-operation with ordinary members",
            "mixed-operation-membership",
        ),
        (
            "multiple",
            "operation packet has multiple authorized-prep-operation members",
            "multiple-operation-membership",
        ),
    ],
)
def test_invalid_operation_membership_is_in_each_launch_and_cli_diagnosis(
    membership: str, structural_error: str, reason: str
) -> None:
    beads = graph()
    if membership == "mixed":
        beads[3]["metadata"] = operation_metadata(lane_order="2")
        for field in ("packet_execution_contract", "deadline_policy"):
            beads[3]["metadata"].pop(field)
    else:
        beads[2]["metadata"] = operation_metadata(packet_size_exception="authorized-operation-singleton")
        beads[3]["metadata"] = operation_metadata(
            lane_order="2", packet_size_exception="authorized-operation-singleton"
        )
        beads[4]["metadata"] = operation_metadata(
            lane_order="3", packet_size_exception="authorized-operation-singleton"
        )
        for item in beads[3:]:
            for field in ("packet_execution_contract", "deadline_policy"):
                item["metadata"].pop(field)
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert any(structural_error in error for error in report["structural_errors"])
    expected_failure = {"kind": "operation-membership", "reason": reason}
    initial = launch(report, "initial")
    apply = launch(report, "apply")
    assert initial["launch_failures"] == [expected_failure]
    assert apply["launch_failures"][0] == expected_failure
    assert not initial["ready"] and not apply["ready"]
    assert initial["phase_token"] == f"{initial['version']}:initial:{initial['context_digest']}"
    assert apply["phase_token"] == f"{apply['version']}:apply:{apply['context_digest']}"

    output = StringIO()
    assert main(["--diagnostic", "--integration-head", HEAD], reader=FakeReader(beads), stdout=output) == 0
    assert f"NOT READY [initial] reindex-prep-a/lane/1: operation-membership:{reason}" in output.getvalue()
    assert f"ERROR: reindex-prep-a/lane/1: {structural_error}" in output.getvalue()


def test_integration_head_is_exact_current_checkout_head_and_is_typed(tmp_path: Path) -> None:
    assert validate(FakeReader(graph()), integration_head=HEAD)["packets"][0]["ready"]
    stale_ancestor = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD^"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
        env=GIT_ENV,
    ).stdout.strip()
    with pytest.raises(ReindexPacketValidationError, match="stale ancestor"):
        validate(FakeReader(graph()), integration_head=stale_ancestor)

    with pytest.raises(ReindexPacketValidationError, match="does not name an existing commit"):
        validate(FakeReader(graph()), integration_head="f" * 40)

    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True, timeout=5, env=GIT_ENV)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"], check=True, timeout=5, env=GIT_ENV
    )
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True, timeout=5, env=GIT_ENV)
    (tmp_path / "foreign.txt").write_text("foreign\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "foreign.txt"], check=True, timeout=5, env=GIT_ENV)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "--quiet", "-m", "foreign"], check=True, timeout=5, env=GIT_ENV
    )
    foreign_head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
        env=GIT_ENV,
    ).stdout.strip()
    with pytest.raises(ReindexPacketValidationError, match="does not name an existing commit"):
        validate(FakeReader(graph()), integration_head=foreign_head)

    with pytest.raises(ReindexPacketValidationError, match="40-character lowercase"):
        validate(FakeReader(graph()), integration_head="not-a-head")


class _GitResult:
    def __init__(self, stdout: str = "") -> None:
        self.stdout = stdout


def test_integration_head_rejects_descendant_and_divergent_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = "a" * 40

    def descendant(command: list[str], **_: object) -> _GitResult:
        if "rev-parse" in command and command[-1] == "HEAD":
            return _GitResult(HEAD + "\n")
        if "rev-parse" in command and command[-1] == f"{candidate}^{{commit}}":
            return _GitResult(candidate + "\n")
        if command[-2:] == [candidate, "HEAD"]:
            raise subprocess.CalledProcessError(1, command)
        assert command[-2:] == ["HEAD", candidate]
        return _GitResult()

    monkeypatch.setattr(subprocess, "run", descendant)
    with pytest.raises(ReindexPacketValidationError, match="is a descendant"):
        reindex_packets._validate_integration_head(candidate)

    def divergent(command: list[str], **_: object) -> _GitResult:
        if "rev-parse" in command and command[-1] == "HEAD":
            return _GitResult(HEAD + "\n")
        if "rev-parse" in command and command[-1] == f"{candidate}^{{commit}}":
            return _GitResult(candidate + "\n")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", divergent)
    with pytest.raises(ReindexPacketValidationError, match="does not equal exact checkout HEAD"):
        reindex_packets._validate_integration_head(candidate)


def test_integration_head_checkout_and_merge_base_failures_are_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = "a" * 40

    def checkout_failure(command: list[str], **_: object) -> _GitResult:
        raise OSError("git unavailable")

    monkeypatch.setattr(subprocess, "run", checkout_failure)
    with pytest.raises(ReindexPacketValidationError, match="unable to resolve exact checkout HEAD"):
        reindex_packets._validate_integration_head(candidate)

    def merge_base_timeout(command: list[str], **_: object) -> _GitResult:
        if "rev-parse" in command and command[-1] == "HEAD":
            return _GitResult(HEAD + "\n")
        if "rev-parse" in command and command[-1] == f"{candidate}^{{commit}}":
            return _GitResult(candidate + "\n")
        raise subprocess.TimeoutExpired(command, 5)

    monkeypatch.setattr(subprocess, "run", merge_base_timeout)
    with pytest.raises(ReindexPacketValidationError, match="unable to classify integration head"):
        reindex_packets._validate_integration_head(candidate)

    def descendant_classification_failure(command: list[str], **_: object) -> _GitResult:
        if "rev-parse" in command and command[-1] == "HEAD":
            return _GitResult(HEAD + "\n")
        if "rev-parse" in command and command[-1] == f"{candidate}^{{commit}}":
            return _GitResult(candidate + "\n")
        if command[-2:] == [candidate, "HEAD"]:
            raise subprocess.CalledProcessError(1, command)
        raise OSError("merge-base unavailable")

    monkeypatch.setattr(subprocess, "run", descendant_classification_failure)
    with pytest.raises(ReindexPacketValidationError, match="unable to classify integration head"):
        reindex_packets._validate_integration_head(candidate)


def test_wave_order_accepts_post_reindex_only_after_live_window_and_unknown_is_not_earliest() -> None:
    assert WAVES["post-reindex"] > WAVES["reindex-window"]
    beads = graph()
    for item in beads[2:]:
        item["metadata"]["packet_size_exception"] = "bounded"
    beads[2]["metadata"].update(execution_wave="reindex-window", lane_packet="1", lane_order="1")
    beads[3]["metadata"].update(execution_wave="post-reindex", lane_packet="2", lane_order="1")
    beads[4]["metadata"].update(execution_wave="post-reindex", lane_packet="2", lane_order="2")
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert not any("earlier wave blocks on later wave" in error for error in report["structural_errors"])

    beads[2]["metadata"]["execution_wave"] = "post-reindex"
    beads[3]["metadata"]["execution_wave"] = "reindex-window"
    beads[4]["metadata"]["execution_wave"] = "reindex-window"
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert any("earlier wave blocks on later wave" in error for error in report["structural_errors"])

    beads[2]["metadata"]["execution_wave"] = "unknown-wave"
    beads[3]["metadata"]["execution_wave"] = "post-reindex"
    beads[4]["metadata"]["execution_wave"] = "post-reindex"
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert any("unknown execution_wave" in error for error in report["structural_errors"])
    assert not any("earlier wave blocks on later wave" in error for error in report["structural_errors"])


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
    assert {(failure["kind"], failure["reason"]) for failure in apply["launch_failures"]} == {
        ("rehearsal", "missing"),
        ("operator-authorization", "missing"),
        ("independent-review", "missing"),
        ("apply-authority", "unsupported-evidence-adapter"),
    }


def test_initial_phase_token_cannot_authorize_apply() -> None:
    report = validate(FakeReader(graph(operation=True)), integration_head=HEAD)
    initial, apply = launch(report, "initial"), launch(report, "apply")
    assert initial["context_digest"] != apply["context_digest"]
    assert initial["phase_token"] != apply["phase_token"]
    assert ":initial:" in initial["phase_token"] and ":apply:" in apply["phase_token"]
    assert initial["effective_authority"]["may_apply"] is False
    assert apply["effective_authority"]["may_apply"] is False


def test_caller_evidence_cannot_change_any_initial_projection_field() -> None:
    beads = graph(operation=True)
    baseline = launch(validate(FakeReader(beads), integration_head=HEAD), "initial")
    valid_evidence = bound_operation_evidence(beads)
    for caller_evidence in (
        valid_evidence,
        {"arbitrary": "valid-json", "nested": [1, None, True]},
        None,
    ):
        report = validate(
            FakeReader(beads),
            integration_head=HEAD,
            operation_evidence={"a": caller_evidence},
        )
        assert launch(report, "initial") == baseline


def test_apply_phase_token_binds_the_runtime_plan_digest() -> None:
    beads = graph(operation=True)
    first_evidence = bound_operation_evidence(beads)
    second_evidence = bound_operation_evidence(beads, plan_digest="sha256:plan-2")
    first = validate(
        FakeReader(beads), integration_head=HEAD, selected_phase="apply", operation_evidence={"a": first_evidence}
    )
    second = validate(
        FakeReader(beads),
        integration_head=HEAD,
        selected_phase="apply",
        operation_evidence={"a": second_evidence},
    )
    assert launch(first, "apply")["context_digest"] != launch(second, "apply")["context_digest"]
    assert launch(first, "apply")["operation_evidence"]["a"]["plan_digest"] == "sha256:plan-1"


def test_diagnostic_evidence_cannot_authorize_apply() -> None:
    beads = graph(operation=True)
    missing = validate(FakeReader(beads), integration_head=HEAD, selected_phase="apply")
    assert packet(missing)["ready"] is False
    report = validate(
        FakeReader(beads),
        integration_head=HEAD,
        selected_phase="apply",
        operation_evidence={"a": bound_operation_evidence(beads)},
    )
    assert packet(report)["selected_phase"] == "apply"
    assert not packet(report)["ready"] and not launch(report, "apply")["ready"]
    assert launch(report, "apply")["effective_authority"]["may_apply"] is False
    assert {failure["reason"] for failure in launch(report, "apply")["launch_failures"]} == {
        "unsupported-evidence-adapter"
    }


@pytest.mark.parametrize(
    ("evidence", "reason"),
    [
        (operation_evidence(rehearsal_digest="sha256:other"), "mismatched-plan-digest"),
        (operation_evidence(authorization_digest="sha256:other"), "mismatched-plan-digest"),
        (operation_evidence(review_digest="sha256:other"), "mismatched-plan-digest"),
        (operation_evidence(rehearsal_state="stale"), "stale"),
        (operation_evidence(authorization_state="revoked"), "revoked"),
    ],
)
def test_apply_mismatched_or_stale_diagnostic_evidence_fails_closed(evidence: dict[str, Any], reason: str) -> None:
    report = validate(
        FakeReader(graph(operation=True)),
        integration_head=HEAD,
        selected_phase="apply",
        operation_evidence={"a": bind_diagnostic_evidence(graph(operation=True), evidence)},
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
        FakeReader(beads),
        integration_head=HEAD,
        selected_phase="apply",
        operation_evidence={"a": bind_diagnostic_evidence(beads, evidence)},
    )
    assert not any(failure["kind"] == "independent-review" for failure in launch(report, "apply")["launch_failures"])


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


@pytest.mark.parametrize("field", ["operation_phase_contract", "initial_job_authority", "apply_authority"])
def test_legacy_v1_string_operation_contracts_require_structured_v2(field: str) -> None:
    beads = graph(operation=True)
    beads[2]["metadata"][field] = "legacy-v1-value"
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert any(
        "structured-v2-required/legacy-field" in error and field in error for error in report["structural_errors"]
    )
    assert not any(field in error and "missing" in error for error in report["structural_errors"])


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
    call_options: list[dict[str, Any]] = []

    class Completed:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def run(command: list[str], **options: Any) -> Completed:
        calls.append(tuple(command))
        call_options.append(options)
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
    assert all(options["timeout"] == 5 for options in call_options)
    assert all(options["env"]["GIT_OPTIONAL_LOCKS"] == "0" for options in call_options)
    assert json.loads(output.getvalue())["read_only"] is True


def test_open_closure_shape_and_leaf_carriers_cannot_be_skipped() -> None:
    beads = graph()
    hidden = bead("hidden", {}, status="open")
    hidden["labels"] = []
    beads.append(hidden)
    beads[1]["dependencies"].append(dep("hidden"))
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert "open blocks-closure records have no campaign carrier: hidden" in report["structural_errors"]

    hidden["metadata"] = {"campaign_id": "reindex-2026", "execution_shape": "mystery"}
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert "open campaign closure records have no valid execution shape: hidden" in report["structural_errors"]

    beads = graph()
    beads[2]["metadata"].pop("tdd_mode")
    assert any(
        "a: missing leaf carrier(s)" in error
        for error in validate(FakeReader(beads), integration_head=HEAD)["structural_errors"]
    )


def test_declared_packet_keeps_invalid_order_member_in_its_digest_and_readiness() -> None:
    beads = graph()
    for item in beads[3:]:
        item["metadata"]["packet_size_exception"] = "reviewed-test-shape"
    beads[2]["metadata"]["lane_order"] = "not-a-number"
    for field in ("packet_execution_contract", "deadline_policy"):
        beads[2]["metadata"].pop(field)
        beads[3]["metadata"][field] = "packet-v1" if field == "packet_execution_contract" else "bounded"
    beads[3]["dependencies"] = []
    report = validate(FakeReader(beads), integration_head=HEAD)
    ordinary = launch(report, "ordinary")
    assert packet(report)["member_ids"] == ["b", "c", "a"]
    assert [identity["bead_id"] for identity in ordinary["task_identities"]] == ["b", "c", "a"]
    assert not packet(report)["ready"] and not ordinary["ready"]
    assert {failure["kind"] for failure in ordinary["launch_failures"]} >= {"packet-assignment"}


@pytest.mark.parametrize(
    ("change", "failure_kind"),
    [
        ({"tdd_mode": ""}, "missing-leaf-carrier"),
        ({"live_data_access": ""}, "missing-live-data-authority"),
        ({"model_policy": "provider-pinned-v1"}, "model-policy"),
    ],
)
def test_leaf_structural_contract_failures_are_consumed_by_packet_launchers(
    change: dict[str, object], failure_kind: str
) -> None:
    beads = graph()
    beads[2]["metadata"].update(change)
    report = validate(FakeReader(beads), integration_head=HEAD)
    ordinary = launch(report, "ordinary")
    assert not report["ok"] and not packet(report)["ready"] and not ordinary["ready"]
    assert failure_kind in {failure["kind"] for failure in ordinary["launch_failures"]}


def test_incomplete_packet_metadata_blocks_every_remaining_packet_with_typed_failure() -> None:
    beads = graph()
    beads[2]["metadata"]["execution_lane"] = ""
    report = validate(FakeReader(beads), integration_head=HEAD)
    ordinary = launch(report, "ordinary")
    assert not ordinary["ready"]
    assert {failure["kind"] for failure in ordinary["launch_failures"]} >= {
        "packet-assignment",
        "unassigned-structural",
    }
    assert report["global_launch_failures"]


def test_closed_noncampaign_closure_is_warning_and_non_task_records_are_filtered() -> None:
    beads = graph()
    history = bead("history", {}, status="closed")
    history["labels"] = []
    beads.append(history)
    beads[1]["dependencies"].append(dep("history"))
    beads.append({"_type": "memory", "content": "not a task"})
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert report["ok"]
    assert report["warnings"] == ["1 closed blocks-closure records have no campaign carrier"]
    assert "" not in report["blocks_only_closure"]


def test_packet_shape_order_leader_and_blocker_invariants_remain_enforced() -> None:
    beads = graph()
    beads[2]["metadata"]["lane_order"] = "2"
    beads[3]["metadata"].update(lane_order="2", packet_execution_contract="packet-v1")
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert any("duplicate packet order" in error for error in report["structural_errors"])
    assert any("internal blocker is not earlier" in error for error in report["structural_errors"])
    assert any("non-leader carries packet_execution_contract" in error for error in report["structural_errors"])

    beads = graph()
    beads[2]["metadata"].update(packet_size_exception="bounded", lane_packet="1")
    beads[3]["metadata"].update(packet_size_exception="bounded", lane_packet="2", lane_order="1")
    beads[3]["dependencies"] = []
    assert any(
        "packet blocker is in a later packet" in error
        for error in validate(FakeReader(beads), integration_head=HEAD)["structural_errors"]
    )

    beads = graph()
    beads[2]["metadata"]["execution_wave"] = "not-a-wave"
    assert any(
        "unknown execution_wave" in error
        for error in validate(FakeReader(beads), integration_head=HEAD)["structural_errors"]
    )

    beads = graph()
    beads[1]["dependencies"] = [dep("a")]
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert any("ordinary packet has 1 leaves" in error for error in report["structural_errors"])

    beads = graph()
    beads[2]["metadata"]["lane_order"] = "zero"
    assert any(
        "lane order is not numeric" in error
        for error in validate(FakeReader(beads), integration_head=HEAD)["structural_errors"]
    )

    beads = graph()
    beads[2]["metadata"]["lane_packet"] = ""
    assert any(
        "invalid packet assignment" in error
        for error in validate(FakeReader(beads), integration_head=HEAD)["structural_errors"]
    )


def test_operation_policy_is_wave_independent_and_cannot_be_bypassed_by_wave_edit() -> None:
    beads = graph(operation=True)
    beads[2]["metadata"].update(execution_wave="reindex-window", plan_id="forbidden")
    for field in ("operation_phase_contract", "initial_job_authority", "apply_authority"):
        beads[2]["metadata"].pop(field)
    report = validate(FakeReader(beads), integration_head=HEAD, selected_phase="apply")
    apply = launch(report, "apply")
    assert any("requires a declared prep execution_wave" in error for error in report["structural_errors"])
    assert any("missing operation_phase_contract" in error for error in report["structural_errors"])
    assert any("invalid initial_job_authority" in error for error in report["structural_errors"])
    assert any("invalid apply_authority" in error for error in report["structural_errors"])
    assert any("stores runtime policy field" in error for error in report["structural_errors"])
    assert not apply["ready"] and apply["effective_authority"]["may_apply"] is False


def test_operation_packet_membership_is_unambiguous_and_never_uses_an_ordinary_leader() -> None:
    beads = graph()
    beads[3]["metadata"] = operation_metadata(lane_order="2")
    report = validate(FakeReader(beads), integration_head=HEAD)
    initial = launch(report, "initial")
    assert any(
        "mixes an authorized-prep-operation with ordinary members" in error for error in report["structural_errors"]
    )
    assert initial["operation_member_id"] is None
    assert initial["effective_authority"]["mode"] == "invalid"
    assert not initial["ready"]

    beads = graph()
    beads[2]["metadata"] = operation_metadata(packet_size_exception="authorized-operation-singleton")
    beads[3]["metadata"] = operation_metadata(lane_order="2", packet_size_exception="authorized-operation-singleton")
    report = validate(FakeReader(beads), integration_head=HEAD)
    assert any("multiple authorized-prep-operation members" in error for error in report["structural_errors"])
    assert not launch(report, "initial")["ready"]


def test_invalid_initial_policy_is_never_ready_even_without_blockers() -> None:
    beads = graph(operation=True)
    beads[2]["metadata"]["initial_job_authority"] = {"mode": APPLY_AUTHORITY_MODE}
    initial = launch(validate(FakeReader(beads), integration_head=HEAD), "initial")
    assert initial["effective_authority"]["mode"] == "invalid"
    assert not initial["ready"]


def test_diagnostic_evidence_contract_binds_packet_head_and_distinct_records() -> None:
    beads = graph(operation=True)
    evidence = bound_operation_evidence(beads)
    evidence["review"]["evidence_id"] = evidence["rehearsal"]["evidence_id"]
    evidence["packet_context_digest"] = "sha256:other"
    evidence["extra"] = "must fail"
    apply = launch(
        validate(FakeReader(beads), integration_head=HEAD, selected_phase="apply", operation_evidence={"a": evidence}),
        "apply",
    )
    assert any(
        "diagnostic operation evidence fields must be" in failure["reason"] for failure in apply["launch_failures"]
    )
    assert apply["effective_authority"]["may_apply"] is False

    evidence = bound_operation_evidence(beads)
    evidence["review"]["evidence_id"] = evidence["rehearsal"]["evidence_id"]
    apply = launch(
        validate(FakeReader(beads), integration_head=HEAD, selected_phase="apply", operation_evidence={"a": evidence}),
        "apply",
    )
    assert "non-distinct-evidence-id" in {failure["reason"] for failure in apply["launch_failures"]}


def test_diagnostic_evidence_rejects_naive_expiry_and_mismatched_bindings() -> None:
    beads = graph(operation=True)
    evidence = bound_operation_evidence(beads)
    evidence["authorization"]["expires_at"] = "2026-08-25T07:00:00"
    apply = launch(
        validate(FakeReader(beads), integration_head=HEAD, selected_phase="apply", operation_evidence={"a": evidence}),
        "apply",
    )
    assert any("timezone-aware" in failure["reason"] for failure in apply["launch_failures"])

    evidence = bound_operation_evidence(beads)
    evidence["integration_head"] = "f" * 40
    evidence["operation_id"] = "other"
    apply = launch(
        validate(FakeReader(beads), integration_head=HEAD, selected_phase="apply", operation_evidence={"a": evidence}),
        "apply",
    )
    assert {failure["reason"] for failure in apply["launch_failures"]} >= {
        "mismatched-integration-head",
        "mismatched-operation-id",
        "unsupported-evidence-adapter",
    }

    evidence = bound_operation_evidence(beads)
    evidence["packet_context_digest"] = "sha256:other"
    apply = launch(
        validate(FakeReader(beads), integration_head=HEAD, selected_phase="apply", operation_evidence={"a": evidence}),
        "apply",
    )
    assert "mismatched-packet-context" in {failure["reason"] for failure in apply["launch_failures"]}


def test_task_revision_binds_top_level_semantics_and_has_no_self_oracle() -> None:
    record = _record(graph()[2])
    baseline = _task_revision(record)
    for field, value in (
        ("description", "materially changed"),
        ("status", "closed"),
        ("dependencies", ({"depends_on_id": "other", "type": "blocks"},)),
        ("new_future_semantic_field", {"meaning": "new"}),
    ):
        changed = dict(record)
        changed[field] = value
        assert _task_revision(changed) != baseline
    changed = dict(record)
    changed["updated_at"] = "2026-09-01T00:00:00Z"
    assert _task_revision(changed) == baseline


def test_invalid_integration_head_and_phase_explicit_text_report_fail_closed() -> None:
    with pytest.raises(ReindexPacketValidationError, match="40-character lowercase"):
        validate(FakeReader(graph()), integration_head="not-a-head")
    output = StringIO()
    assert (
        main(
            ["--diagnostic", "--phase", "apply", "--integration-head", HEAD],
            reader=FakeReader(graph(operation=True)),
            stdout=output,
        )
        == 0
    )
    assert "NOT READY [apply]" in output.getvalue()

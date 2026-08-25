"""Miniature graphs for the reindex packet projection."""

from __future__ import annotations

import json
from io import StringIO
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from devtools.reindex_packets import ROOT_ID, main, validate

HEAD = "0123456789abcdef"


class FakeReader:
    def __init__(self, beads: list[dict[str, Any]]) -> None:
        self.beads = tuple(beads)

    def read(self) -> tuple[dict[str, Any], ...]:
        return self.beads


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


def predicate(kind: str, evidence_id: str, state: str = "accepted", **changes: Any) -> dict[str, Any]:
    value = {"kind": kind, "evidence_id": evidence_id, "state": state}
    value.update(changes)
    return value


def readiness(
    *,
    members: tuple[str, ...] = ("a", "b", "c"),
    predicates: tuple[dict[str, Any], ...] = (),
    head: str = HEAD,
) -> dict[str, Any]:
    return {
        "version": "packet-readiness-v1",
        "integration_head": head,
        "evaluated_at": "2026-08-25T06:00:00+00:00",
        "members": [{"id": bead_id, "revision": f"{bead_id}-v1"} for bead_id in members],
        "predicates": list(predicates),
    }


def bead(
    bead_id: str, metadata: dict[str, Any], dependencies: tuple[dict[str, str], ...] = (), status: str = "open"
) -> dict[str, Any]:
    return {
        "id": bead_id,
        "revision": f"{bead_id}-v1",
        "status": status,
        "labels": ("campaign:reindex-2026",),
        "metadata": metadata,
        "dependencies": dependencies,
    }


def graph() -> list[dict[str, Any]]:
    root = bead(ROOT_ID, {"campaign_id": "reindex-2026", "execution_shape": "gate"}, (dep("gate"),))
    gate = bead("gate", {"campaign_id": "reindex-2026", "execution_shape": "gate"}, tuple(dep(item) for item in "abc"))
    leader = bead(
        "a",
        meta(
            packet_execution_contract="packet-v1",
            deadline_policy="bounded",
            readiness_contract=readiness(),
        ),
    )
    return [
        root,
        gate,
        leader,
        bead("b", meta(lane_order="2"), (dep("a"),)),
        bead("c", meta(lane_order="3"), (dep("b"),)),
    ]


def test_blocks_closure_excludes_mixed_and_reports_labelled_difference() -> None:
    beads = graph() + [bead("related", meta())]
    beads[1] = {**beads[1], "dependencies": (*beads[1]["dependencies"], dep("related", "relates-to"))}
    report = validate(FakeReader(beads))
    assert "related" not in report["blocks_only_closure"]
    assert "related" in report["mixed_relation_expansion"]
    assert report["differences"]["campaign_only_ids"] == ["related"]


def test_gate_and_leaf_carriers_are_separate_and_strings_are_valid() -> None:
    beads = graph()
    beads[1]["metadata"]["worker_model_class"] = "worker"
    report = validate(FakeReader(beads))
    assert "gate: gate carries worker_model_class" in report["structural_errors"]
    assert not any("a: missing leaf carrier" in error for error in report["structural_errors"])
    beads[2]["metadata"].pop("tdd_mode")
    assert any("a: missing leaf carrier" in error for error in validate(FakeReader(beads))["structural_errors"])


def test_open_closure_records_cannot_hide_behind_missing_campaign_or_shape() -> None:
    beads = graph()
    beads.append(bead("hidden", {}, status="open"))
    beads[-1]["labels"] = ()
    beads[1]["dependencies"] = (*beads[1]["dependencies"], dep("hidden"))
    report = validate(FakeReader(beads))
    assert "open blocks-closure records have no campaign carrier: hidden" in report["structural_errors"]

    beads[-1]["metadata"] = {"campaign_id": "reindex-2026", "execution_shape": "mystery"}
    report = validate(FakeReader(beads))
    assert "open campaign closure records have no valid execution shape: hidden" in report["structural_errors"]


def test_closed_historical_blocker_without_campaign_is_diagnostic_only() -> None:
    beads = graph()
    beads.append(bead("history", {}, status="closed"))
    beads[-1]["labels"] = ()
    beads[1]["dependencies"] = (*beads[1]["dependencies"], dep("history"))
    report = validate(FakeReader(beads))
    assert report["ok"]
    assert report["warnings"] == ["1 closed blocks-closure records have no campaign carrier"]


def test_gate_cannot_carry_packet_membership() -> None:
    beads = graph()
    beads[1]["metadata"].update(lane_packet="1", lane_order="1")
    report = validate(FakeReader(beads))
    assert "gate: gate carries lane_packet" in report["structural_errors"]
    assert "gate: gate carries lane_order" in report["structural_errors"]


def test_packet_order_and_leader_placement_fail_independently() -> None:
    beads = graph()
    beads[2]["metadata"]["lane_order"] = "2"
    beads[3]["metadata"] = meta(lane_order="1", readiness_contract=readiness())
    report = validate(FakeReader(beads))
    assert any("internal blocker is not earlier" in error for error in report["structural_errors"])
    assert any("non-leader carries readiness_contract" in error for error in report["structural_errors"])


def test_cross_packet_blocker_must_not_point_backwards() -> None:
    beads = graph()
    beads[2]["metadata"].update(packet_size_exception="bounded", lane_packet="1")
    beads[3]["metadata"].update(packet_size_exception="bounded", lane_packet="2", lane_order="1")
    beads[3]["dependencies"] = ()
    assert any(
        "packet blocker is in a later packet" in error for error in validate(FakeReader(beads))["structural_errors"]
    )


def test_exact_conflict_keys_ignore_write_scope_but_require_serialization() -> None:
    beads = graph()
    beads[3]["metadata"] = meta(
        execution_lane="other", lane_order="2", conflict_keys="one-long", write_scope="same prose"
    )
    beads[3]["dependencies"] = ()
    report = validate(FakeReader(beads))
    assert not any("exact conflict-key overlap" in error for error in report["structural_errors"])
    beads[3]["metadata"] = meta(
        execution_lane="other", lane_order="2", conflict_keys="one", write_scope="different prose"
    )
    assert any("exact conflict-key overlap" in error for error in validate(FakeReader(beads))["structural_errors"])


def operation_contract(shape: str = "plan-rehearse-review-authorize-apply-verify") -> dict[str, Any]:
    return {
        "version": "prep-operation-phase-v1",
        "shape": shape,
        "plan_id": "plan-1",
        "plan_digest": "sha256:plan-1",
        "rehearsal_id": "rehearsal-1",
    }


def operation_metadata(**changes: Any) -> dict[str, Any]:
    value = {
        "execution_kind": "authorized-prep-operation",
        "live_data_access": "explicit-operator-authorized-source-apply",
        "lane_mode": "serialized-writer",
        "operation_phase_contract": operation_contract(),
        "initial_job_authority": {"mode": "read-only-plan-rehearsal", "authority_id": "initial-1"},
        "apply_authority": {
            "mode": "explicit-operator-authorized-apply",
            "authorization_id": "authorization-1",
            "plan_digest": "sha256:plan-1",
        },
    }
    value.update(changes)
    return meta(**value)


def operation_predicates() -> tuple[dict[str, Any], ...]:
    return (
        predicate("rehearsal", "rehearsal-1", plan_digest="sha256:plan-1"),
        predicate(
            "operator-authorization",
            "authorization-1",
            "authorized",
            plan_digest="sha256:plan-1",
            expires_at="2026-08-25T07:00:00+00:00",
        ),
    )


def test_authorized_prep_operations_have_kind_phase_authority_and_serialization() -> None:
    beads = graph()
    beads[2]["metadata"]["live_data_access"] = "explicit-operator-authorized-source-apply"
    assert any(
        "reindex-prep-a/explicit-operator-authorized-source-apply requires execution_kind" in error
        for error in validate(FakeReader(beads))["structural_errors"]
    )
    beads[2]["metadata"] = operation_metadata()
    beads[2]["metadata"]["readiness_contract"] = readiness(predicates=operation_predicates())
    assert not any("authorized prep operation" in error for error in validate(FakeReader(beads))["structural_errors"])


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        (
            {"live_data_access": "read-only", "execution_kind": "authorized-prep-operation"},
            "requires operator-authorized live_data_access",
        ),
        ({"lane_mode": "parallel"}, "must be serialized"),
        ({"operation_phase_contract": None}, "missing operation_phase_contract"),
        ({"initial_job_authority": {"mode": "explicit-operator-authorized-apply"}}, "invalid initial_job_authority"),
        (
            {
                "apply_authority": {
                    "mode": "explicit-operator-authorized-apply",
                    "authorization_id": "authorization-1",
                    "plan_digest": "sha256:other",
                }
            },
            "apply authority plan digest",
        ),
    ],
)
def test_authorized_prep_operation_phase_and_apply_are_separate(changes: dict[str, Any], expected: str) -> None:
    beads = graph()
    beads[2]["metadata"] = operation_metadata(**changes)
    beads[2]["metadata"]["readiness_contract"] = readiness(predicates=operation_predicates())
    assert any(expected in error for error in validate(FakeReader(beads))["structural_errors"])


@pytest.mark.parametrize(
    ("access", "shape"),
    [
        ("explicit-operator-authorized-source-apply", "plan-rehearse-review-authorize-apply-verify"),
        ("explicit-operator-authorized-blob-apply", "accepted-plan-rehearse-authorize-apply-verify"),
    ],
)
def test_authorized_prep_operation_shapes_and_ordinary_prep_leaves_are_allowed(access: str, shape: str) -> None:
    beads = graph()
    beads[2]["metadata"] = operation_metadata(
        live_data_access=access,
        operation_phase_contract=operation_contract(shape),
    )
    beads[2]["metadata"]["readiness_contract"] = readiness(predicates=operation_predicates())
    beads[3]["metadata"].update(execution_kind="evidence", live_data_access="read-only")
    report = validate(FakeReader(beads))
    assert report["ok"]
    assert report["packets"][0]["ready"]


def test_candidate_writers_need_serialization() -> None:
    beads = graph()
    beads[2]["metadata"]["live_data_access"] = "isolated-candidate-write"
    assert "a: candidate writer is not serialized" in validate(FakeReader(beads))["structural_errors"]
    beads[2]["metadata"]["lane_mode"] = "serialized-writer"
    assert "a: candidate writer is not serialized" not in validate(FakeReader(beads))["structural_errors"]


def test_predicates_compile_to_typed_unsatisfied_reasons() -> None:
    beads = graph()
    beads[2]["metadata"]["readiness_contract"] = readiness(
        predicates=(predicate("calibration-receipt", "calibration-1", "stale"),)
    )
    report = validate(FakeReader(beads))
    assert report["ok"] and report["counts"]["non_ready_packets"] == 1
    projection = report["packets"][0]["launch_projection"]
    assert projection["unsatisfied_predicates"] == [
        {"evidence_id": "calibration-1", "kind": "calibration-receipt", "reason": "stale"}
    ]


def test_launch_projection_binds_identity_head_capability_and_deadline() -> None:
    projection = validate(FakeReader(graph()))["packets"][0]["launch_projection"]
    assert projection["version"] == "packet-launch-projection-v1"
    assert projection["integration_head"] == HEAD
    assert projection["task_identities"] == [
        {"bead_id": "a", "revision": "a-v1"},
        {"bead_id": "b", "revision": "b-v1"},
        {"bead_id": "c", "revision": "c-v1"},
    ]
    assert projection["launch_contract"] == {
        "packet_execution_contract": "packet-v1",
        "deadline_policy": "bounded",
        "model_policy": "provider-neutral-capability-v1",
        "worker_capability": "cheap-capable",
        "review_capability": "strong-review",
    }
    assert projection["context_digest"].startswith("sha256:")
    assert projection["projection_id"] == f"packet-launch-projection-v1:{projection['context_digest']}"


@given(st.sampled_from(("accepted", "pending", "rejected", "stale", "revoked")))
def test_closed_predicate_states_fail_closed_except_accepted(state: str) -> None:
    beads = graph()
    beads[2]["metadata"]["readiness_contract"] = readiness(
        predicates=(predicate("calibration-receipt", "calibration-1", state),)
    )
    projection = validate(FakeReader(beads))["packets"][0]["launch_projection"]
    assert projection["ready"] is (state == "accepted")


def test_missing_rejected_unknown_and_stale_evidence_fail_closed() -> None:
    beads = graph()
    beads[2]["metadata"]["readiness_contract"] = readiness(
        predicates=(predicate("exact-head-review", "review-1", "rejected", head=HEAD),)
    )
    assert not validate(FakeReader(beads))["packets"][0]["ready"]

    beads[2]["metadata"]["readiness_contract"] = readiness(
        predicates=(predicate("exact-head-review", "review-1", head="other-head"),)
    )
    stale = validate(FakeReader(beads))["packets"][0]["launch_projection"]
    assert stale["unsatisfied_predicates"][0]["reason"] == "stale"

    beads[2]["metadata"]["readiness_contract"] = readiness(predicates=(predicate("exact-head-review", "", head=HEAD),))
    assert any("missing evidence_id" in error for error in validate(FakeReader(beads))["structural_errors"])

    beads[2]["metadata"]["readiness_contract"] = readiness(predicates=(predicate("phrase-from-a-note", "note-1"),))
    assert any("unknown predicate kind" in error for error in validate(FakeReader(beads))["structural_errors"])


def test_authorization_expiry_and_revocation_fail_closed() -> None:
    beads = graph()
    for state, expires_at, reason in (
        ("authorized", "2026-08-25T05:00:00+00:00", "expired"),
        ("revoked", "2026-08-25T07:00:00+00:00", "revoked"),
    ):
        beads[2]["metadata"]["readiness_contract"] = readiness(
            predicates=(
                predicate(
                    "operator-authorization",
                    "authorization-1",
                    state,
                    plan_digest="sha256:plan-1",
                    expires_at=expires_at,
                ),
            )
        )
        projection = validate(FakeReader(beads))["packets"][0]["launch_projection"]
        assert projection["unsatisfied_predicates"][0]["reason"] == reason


def test_blocks_prerequisites_are_packet_unit_readiness() -> None:
    beads = graph()
    beads.append(bead("outside", {"campaign_id": "reindex-2026", "execution_shape": "gate"}, status="closed"))
    beads[2]["dependencies"] = (dep("outside"),)
    assert validate(FakeReader(beads))["packets"][0]["ready"]

    beads[-1]["status"] = "open"
    report = validate(FakeReader(beads))
    projection = report["packets"][0]["launch_projection"]
    assert not projection["ready"]
    assert projection["unsatisfied_predicates"] == [{"bead_id": "outside", "kind": "blocks", "reason": "open"}]


def test_contract_member_revision_and_head_drift_are_not_launchable() -> None:
    beads = graph()
    beads[2]["revision"] = "a-v2"
    report = validate(FakeReader(beads))
    assert not report["packets"][0]["ready"]
    assert report["packets"][0]["launch_projection"]["unsatisfied_predicates"] == [
        {"bead_id": "a", "kind": "task-identity", "reason": "stale"}
    ]

    beads = graph()
    report = validate(FakeReader(beads), integration_head="new-head")
    assert report["packets"][0]["launch_projection"]["unsatisfied_predicates"] == [
        {"kind": "integration-head", "reason": "stale"}
    ]


def test_legacy_readiness_is_censused_without_phrase_interpretation() -> None:
    beads = graph()
    beads[2]["metadata"]["dispatch_readiness"] = "blocked-on-unknown; needs a note"
    report = validate(FakeReader(beads))
    assert report["legacy_readiness_census"] == {
        "dispatch_readiness": {"count": 1, "record_ids": ["a"]},
        "program_dispatch_readiness": {"count": 0, "record_ids": []},
    }


def test_historical_only_damage_fields_are_not_a_validator_policy() -> None:
    beads = graph()
    beads[2]["metadata"].update(deletion_ledger="historical-only", existing_test_disposition="none")
    assert validate(FakeReader(beads))["ok"]


def test_reader_invocation_is_read_only_and_json_marks_it(monkeypatch: Any) -> None:
    calls: list[tuple[str, ...]] = []

    class Completed:
        stdout = (
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

    def run(command: list[str], **_: Any) -> Completed:
        calls.append(tuple(command))
        return Completed()

    monkeypatch.setattr("devtools.reindex_packets.subprocess.run", run)
    output = StringIO()
    assert main(["--json"], stdout=output) == 0
    assert calls == [("bd", "--readonly", "export", "--all")]
    assert json.loads(output.getvalue())["read_only"] is True

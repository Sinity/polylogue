"""Miniature graphs for the reindex packet projection."""

from __future__ import annotations

import json
from io import StringIO
from typing import Any

from devtools.reindex_packets import ROOT_ID, main, validate


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
        "id": bead_id,
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
            effort="small",
            expected_duration_evidence="receipt",
            deadline_policy="bounded",
            dispatch_readiness="ready",
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
    beads[3]["metadata"] = meta(lane_order="1", dispatch_readiness="ready")
    report = validate(FakeReader(beads))
    assert any("internal blocker is not earlier" in error for error in report["structural_errors"])
    assert any("non-leader carries dispatch_readiness" in error for error in report["structural_errors"])


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


def test_authorized_writers_need_serialization_and_serialized_writer_is_allowed() -> None:
    beads = graph()
    beads[2]["metadata"]["live_data_access"] = "explicit-operator-authorized-source-apply"
    assert any(
        "operator-authorized writer is not serialized" in error
        for error in validate(FakeReader(beads))["structural_errors"]
    )
    beads[2]["metadata"].update(
        live_data_access="explicit-operator-authorized-blob-apply", lane_mode="serialized-writer"
    )
    assert not any(
        "operator-authorized writer is not serialized" in error
        for error in validate(FakeReader(beads))["structural_errors"]
    )


def test_candidate_writers_need_serialization() -> None:
    beads = graph()
    beads[2]["metadata"]["live_data_access"] = "read-only-sources-and-isolated-candidate-write"
    assert "a: candidate writer is not serialized" in validate(FakeReader(beads))["structural_errors"]
    beads[2]["metadata"]["lane_mode"] = "serialized-writer"
    assert "a: candidate writer is not serialized" not in validate(FakeReader(beads))["structural_errors"]


def test_pending_calibration_is_non_ready_not_structural_error() -> None:
    beads = graph()
    beads[2]["metadata"].update(
        effort="calibration-pending", expected_duration_evidence="pending: receipt", dispatch_readiness="blocked"
    )
    report = validate(FakeReader(beads))
    assert report["ok"] and report["counts"]["non_ready_packets"] == 1
    assert "calibration pending" in report["packets"][0]["non_ready_reasons"]


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

"""Contract tests for the stateless reindex packet verifier."""

from __future__ import annotations

import json
from dataclasses import replace
from io import StringIO
from typing import Any, cast

import pytest

from devtools.reindex_packets import (
    ROOT_ID,
    Bead,
    BeadDependency,
    PacketReader,
    ValidationReport,
    main,
    validate,
)


class FakeReader(PacketReader):
    def __init__(self, beads: list[Bead]) -> None:
        self.beads = tuple(beads)
        self.read_count = 0

    def read(self) -> tuple[Bead, ...]:
        self.read_count += 1
        return self.beads


def _dep(source: str, target: str, kind: str = "blocks") -> BeadDependency:
    return BeadDependency(issue_id=source, depends_on_id=target, type=kind)


def _metadata(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "campaign_id": "reindex-2026",
        "campaign_role": "closure",
        "campaign_timing": "prep",
        "execution_shape": "leaf",
        "execution_wave": "wave1",
        "execution_lane": "lane1",
        "lane_packet": "1",
        "lane_order": "1",
        "conflict_keys": "tests/unit/devtools/test_reindex_packets.py",
        "write_scope": "tests/unit/devtools/test_reindex_packets.py",
        "necessity_class": "execution-assurance",
        "judgment_class": "mechanical",
        "model_policy": "provider-neutral-capability-v1",
        "worker_model_class": "cheap-capable",
        "review_model_class": "standard",
        "live_data_access": "forbidden",
        "decision_closure": {
            "resolved_decisions": [{"decision": "scope", "evidence": "bead:design"}],
            "remaining_decision_points": [],
            "escalation_owner": "operator",
        },
        "packet_execution_contract": {"outputs": ["report"], "verification": ["devtools test"]},
        "effort": "small",
        "expected_duration_evidence": {"source": "historical-run", "seconds": 600},
        "deadline_policy": {"kind": "wave-exit", "evidence_ref": "bead:deadline"},
        "dispatch_readiness": "ready",
        "verification_commands": "devtools test tests/unit/devtools/test_reindex_packets.py; devtools verify",
        "tdd_mode": "contract-first",
        "anti_vacuity": {"mutation": "remove conflict filtering", "red_test": "test_conflict"},
        "existing_test_disposition": {"status": "extend", "law": "packet contract"},
    }
    values.update(overrides)
    return values


def _bead(
    bead_id: str,
    *,
    metadata: dict[str, Any] | None = None,
    labels: tuple[str, ...] = ("campaign:reindex-2026",),
    status: str = "open",
    dependencies: tuple[BeadDependency, ...] = (),
) -> Bead:
    return Bead(
        id=bead_id,
        title=bead_id,
        description="description",
        design="design",
        acceptance_criteria="acceptance",
        notes="notes",
        status=status,
        issue_type="task",
        owner="owner",
        labels=labels,
        metadata=metadata or {},
        dependencies=dependencies,
    )


def _valid_graph() -> list[Bead]:
    root = _bead(
        ROOT_ID,
        metadata={"campaign_id": "reindex-2026", "campaign_role": "milestone", "execution_shape": "gate"},
        dependencies=(_dep(ROOT_ID, "gate"),),
    )
    gate = _bead(
        "gate",
        metadata={"campaign_id": "reindex-2026", "campaign_role": "closure-gate", "execution_shape": "gate"},
        dependencies=(_dep("gate", "a"), _dep("gate", "b"), _dep("gate", "c")),
    )
    leaves = [
        _bead("a", metadata=_metadata(execution_wave="wave1", lane_order="1")),
        _bead("b", metadata=_metadata(execution_wave="wave1", lane_order="2"), dependencies=(_dep("b", "a"),)),
        _bead("c", metadata=_metadata(execution_wave="wave1", lane_order="3"), dependencies=(_dep("c", "b"),)),
    ]
    return [root, gate, *leaves]


def _assert_error(report: ValidationReport, token: str) -> None:
    assert any(token in error for error in report.errors), report.errors


def test_blocks_only_scope_excludes_mixed_relation_expansion() -> None:
    graph = _valid_graph()
    graph.extend(
        [
            _bead("parent-child-only", metadata=_metadata(execution_shape="leaf"), dependencies=()),
            _bead("discovered-only", metadata=_metadata(execution_shape="leaf"), dependencies=()),
        ]
    )
    graph[1] = replace(
        graph[1],
        dependencies=graph[1].dependencies
        + (_dep("gate", "parent-child-only", "parent-child"), _dep("gate", "discovered-only", "discovered-from")),
    )

    report = validate(FakeReader(graph))

    assert report.blocks_only_ids == frozenset({ROOT_ID, "gate", "a", "b", "c"})
    assert report.mixed_relation_ids >= {"parent-child-only", "discovered-only"}
    closure_ids = cast(list[str], report.counts["closure_ids"])
    assert "parent-child-only" not in closure_ids
    assert report.differences["mixed_only_ids"] == ["discovered-only", "parent-child-only"]


def test_labelled_leaf_without_typed_root_path_is_an_error() -> None:
    graph = _valid_graph()
    graph.append(_bead("restoration", metadata=_metadata()))

    report = validate(FakeReader(graph))

    _assert_error(report, "restoration: campaign leaf is not blocks-reachable")


@pytest.mark.parametrize(
    ("field", "value", "token"),
    [
        ("execution_shape", None, "unshaped closure node"),
        ("execution_lane", None, "missing leaf assignment"),
        ("lane_order", "bad", "invalid lane order"),
        ("packet_size_exception", {"reason": "stale"}, "unjustified packet size exception"),
    ],
)
def test_typed_leaf_contract_rejects_malformed_carriers(field: str, value: Any, token: str) -> None:
    graph = _valid_graph()
    metadata = _metadata()
    if field == "execution_shape":
        metadata = {"campaign_id": "reindex-2026", "campaign_role": "closure"}
    elif value is None:
        metadata.pop(field)
    else:
        metadata[field] = value
    graph[2] = replace(graph[2], metadata=metadata)

    report = validate(FakeReader(graph))

    _assert_error(report, token)


def test_duplicate_assignment_and_packet_order_are_rejected() -> None:
    graph = _valid_graph()
    graph[2] = replace(
        graph[2],
        metadata=_metadata(
            execution_assignment={
                "execution_wave": "wave2",
                "execution_lane": "lane9",
                "lane_packet": "9",
                "lane_order": "9",
            }
        ),
    )
    graph[3] = replace(graph[3], metadata=_metadata(lane_order="1"), dependencies=(_dep("b", "a"),))
    graph[4] = replace(graph[4], metadata=_metadata(lane_order="1"), dependencies=(_dep("c", "b"),))

    report = validate(FakeReader(graph))

    _assert_error(report, "incompatible duplicate assignment")
    _assert_error(report, "duplicate packet order")


def test_concurrent_conflict_requires_blocks_serialization() -> None:
    graph = _valid_graph()
    graph[2] = replace(graph[2], metadata=_metadata(execution_lane="lane1", conflict_keys="storage/index_db"))
    graph[3] = replace(
        graph[3],
        metadata=_metadata(execution_lane="lane2", conflict_keys="storage-index.db", lane_order="2"),
        dependencies=(),
    )
    graph[4] = replace(graph[4], metadata=_metadata(execution_lane="lane3", conflict_keys="other", lane_order="3"))

    report = validate(FakeReader(graph))

    _assert_error(report, "concurrent conflict")


def test_carrier_capability_and_readiness_fields_are_typed() -> None:
    graph = _valid_graph()
    graph[2] = replace(
        graph[2],
        metadata=_metadata(
            model_policy="gpt-5.6",
            live_data_access=None,
            expected_duration_evidence="unknown",
            deadline_policy={"kind": "hard"},
            dispatch_readiness="ready",
            prerequisite_state="unmet",
        ),
    )

    report = validate(FakeReader(graph))

    _assert_error(report, "provider-specific model policy")
    _assert_error(report, "missing live-data authority")
    _assert_error(report, "unknown duration evidence")
    _assert_error(report, "deadline evidence")
    _assert_error(report, "prerequisite state is unmet")


def test_prep_lane_cannot_claim_window_mutation_and_temp_code_needs_owner() -> None:
    graph = _valid_graph()
    graph[2] = replace(
        graph[2],
        metadata=_metadata(
            live_data_access="live-mutation",
            temporary_code=True,
            deletion_ledger={"items": []},
        ),
    )

    report = validate(FakeReader(graph))

    _assert_error(report, "window mutation in prep lane")
    _assert_error(report, "temporary machinery without deletion owner")


def test_packet_leader_recomputes_after_closed_former_leader() -> None:
    graph = _valid_graph()
    graph[2] = replace(graph[2], status="closed", metadata={"execution_shape": "leaf"})

    report = validate(FakeReader(graph))

    packet = report.packet("wave1", "lane1", "1")
    assert packet.leader_id == "b"
    assert packet.dispatch_readiness == "ready"


def test_wave_order_regressions_and_duration_calibration_are_reported() -> None:
    graph = _valid_graph()
    graph[2] = replace(
        graph[2],
        metadata=_metadata(
            execution_wave="wave3",
            expected_duration_evidence={
                "source": "old",
                "seconds": 4000,
                "calibration_ref": "old",
                "calibration_status": "stale",
            },
            observed_duration_seconds=10000,
        ),
    )
    graph[3] = replace(
        graph[3], metadata=_metadata(execution_wave="wave3", lane_order="2"), dependencies=(_dep("b", "a"),)
    )

    report = validate(FakeReader(graph))

    _assert_error(report, "duration exceeds 3600s")
    _assert_error(report, "stale calibration reference")
    assert report.calibration_findings == ("a: observed duration overshot estimate by more than 50%",)


def test_read_only_reader_and_json_report_do_not_write_task_state(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    class Completed:
        stdout = json.dumps({"id": ROOT_ID, "status": "closed"}) + "\n"

    def fake_run(*args: Any, **kwargs: Any) -> Completed:
        calls.append((tuple(args[0]), kwargs))
        return Completed()

    monkeypatch.setattr("devtools.reindex_packets.subprocess.run", fake_run)
    output = StringIO()

    exit_code = main(["--json"], stdout=output)

    assert exit_code == 0
    assert calls and "--readonly" in calls[0][0]
    assert "note" not in calls[0][0]
    payload = json.loads(output.getvalue())
    assert payload["read_only"] is True


def test_four_wave_graph_recomputes_cleanly() -> None:
    graph = [
        _bead(
            ROOT_ID,
            metadata={"campaign_id": "reindex-2026", "campaign_role": "milestone", "execution_shape": "gate"},
            dependencies=(_dep(ROOT_ID, "gate"),),
        ),
        _bead(
            "gate",
            metadata={"campaign_id": "reindex-2026", "campaign_role": "closure-gate", "execution_shape": "gate"},
            dependencies=tuple(_dep("gate", f"w{wave}-{index}") for wave in range(1, 5) for index in range(3)),
        ),
    ]
    previous: str | None = None
    for wave in range(1, 5):
        for index in range(3):
            bead_id = f"w{wave}-{index}"
            dependencies = () if previous is None else (_dep(bead_id, previous),)
            graph.append(
                _bead(
                    bead_id,
                    metadata=_metadata(
                        execution_wave=f"wave{wave}",
                        execution_lane=f"lane{wave}",
                        lane_order=str(index + 1),
                        write_scope=f"scope/{wave}/{index}",
                        conflict_keys=f"scope/{wave}/{index}",
                    ),
                    dependencies=dependencies,
                )
            )
            previous = bead_id

    report = validate(FakeReader(graph))

    assert report.ok, report.errors
    assert report.counts["open_leaves"] == 12
    assert report.counts["packets"] == 4

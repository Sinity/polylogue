"""Deterministic causal fixtures exercise overlapping work and missing coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from devtools.fresh_build_bench.trace_ledger import interval_union, main, reduce_events


def _terminal(
    span: str,
    start: int,
    end: int,
    *,
    parent: str | None = None,
    tid: int = 1,
    cpu: tuple[int, int] | None = None,
    **fields: object,
) -> dict[str, object]:
    return {
        "event": f"{span}.ok",
        "span_id": span,
        "trace_id": "trace",
        "parent_span_id": parent,
        "pid": 7,
        "native_thread_id": tid,
        "monotonic_ns": end + 1,
        "span_start_ns": start,
        "span_end_ns": end,
        "span_start_native_thread_id": tid,
        "span_start_pid": 7,
        "outcome": "ok",
        "cpu_scope": "thread" if cpu else "unavailable",
        "thread_cpu_start_ns": cpu[0] if cpu else None,
        "thread_cpu_end_ns": cpu[1] if cpu else None,
        **fields,
    }


def test_union_never_double_counts_nested_and_overlapping_intervals() -> None:
    assert interval_union([(1, 10), (2, 3), (8, 12), (20, 21)]) == 12


@pytest.mark.parametrize("omitted", [1, -1, None, True, "1"])
def test_required_measurement_loss_marks_coverage_incomplete_without_discarding_surviving_timing(
    omitted: object,
) -> None:
    _, ledger = reduce_events([_terminal("body", 1, 10, required_measurement_fields_omitted=omitted)])
    assert ledger["measurement_coverage_complete"] is False
    assert ledger["coverage"]["measurement_loss_events"] == 1
    assert ledger["coverage"]["invalid_required_measurement_omission_counts"] == (
        0 if type(omitted) is int and omitted == 1 else 1
    )
    assert ledger["spans"][0]["chronology"] == "measured"


def test_optional_field_trimming_preserves_independent_measurement_coverage() -> None:
    _, ledger = reduce_events(
        [
            _terminal("body", 1, 10, omitted_fields=4),
            _terminal("second", 11, 20, omitted_fields=4, required_measurement_fields_omitted=0),
        ]
    )
    assert ledger["measurement_coverage_complete"] is True
    assert ledger["coverage"]["measurement_loss_events"] == 0


def test_global_cpu_unions_nested_siblings_and_reports_uncovered_clock_range() -> None:
    _, ledger = reduce_events(
        [
            _terminal("async_pass", 0, 100, tid=3),
            _terminal("compute.service", 0, 80, parent="async_pass", cpu=(10, 60)),
            # These stage spans are nested in the wrapper, but share its async cause.
            _terminal("stage_a", 10, 40, parent="async_pass", cpu=(15, 35)),
            _terminal("stage_b", 30, 50, parent="async_pass", cpu=(30, 40)),
            _terminal("later", 90, 100, cpu=(70, 80)),
            _terminal("parallel", 20, 50, tid=2, cpu=(10, 40)),
        ]
    )
    coverage = ledger["cpu_coverage"]
    assert coverage["measured_thread_cpu_union_ns"] == 90  # 60 on TID 1, 30 on TID 2.
    assert coverage["included_spans"] == 5
    assert coverage["excluded_work_spans"] == 1
    assert coverage["parent_local_exclusive_cpu_sum_ns_not_additive"] == 120
    first = coverage["threads"][0]
    assert first["pid"] == 7 and first["native_thread_id"] == 1
    assert first["clock_range_cpu_ns"] == 70
    assert first["measured_cpu_union_ns"] == 60
    assert first["uncovered_cpu_between_endpoints_ns"] == 10
    assert first["overlapping_inclusive_cpu_ns"] == 30
    assert coverage["threads"][1]["uncovered_cpu_between_endpoints_ns"] == 0
    assert next(s for s in ledger["spans"] if s["name"] == "compute.service")["exclusive_cpu_ns"] == 50


def test_exclusive_work_subtracts_union_of_same_thread_children_only() -> None:
    events = [
        _terminal("root", 0, 100, cpu=(0, 70)),
        _terminal("a", 10, 40, parent="root", cpu=(5, 20)),
        _terminal("b", 30, 60, parent="root", cpu=(15, 40)),
        _terminal("grandchild", 20, 25, parent="a", cpu=(10, 12)),
        _terminal("parallel", 0, 100, parent="root", tid=2, cpu=(0, 90)),
        _terminal("queue", 0, 100, parent="root", span_kind="wait", task_id="queued"),
    ]
    trace, ledger = reduce_events(events)
    by_name = {s["name"]: s for s in ledger["spans"]}
    assert by_name["root"]["exclusive_wall_ns"] == 50
    assert by_name["root"]["exclusive_cpu_ns"] == 35
    assert by_name["a"]["exclusive_wall_ns"] == 25
    assert by_name["a"]["exclusive_cpu_ns"] == 13
    assert by_name["queue"]["exclusive_wall_ns"] is None
    assert by_name["queue"]["inclusive_cpu_ns"] is None
    assert any(e["ph"] == "X" and e["tid"] >= (1 << 31) for e in trace["traceEvents"])
    assert sum(e["ph"] == "f" for e in trace["traceEvents"]) == 5


def test_cpu_is_unknown_for_async_or_changed_thread_and_missing_child_endpoints() -> None:
    _, ledger = reduce_events(
        [
            _terminal("parent", 0, 100, cpu=(0, 70)),
            _terminal("unknown", 10, 30, parent="parent"),
            _terminal("cross", 40, 60, tid=2, cpu=(2, 5), span_start_native_thread_id=1),
        ]
    )
    by_name = {s["name"]: s for s in ledger["spans"]}
    assert by_name["parent"]["exclusive_cpu_ns"] is None
    assert by_name["parent"]["cpu_coverage"] == "missing_child_thread_endpoints"
    assert by_name["cross"]["inclusive_cpu_ns"] is None
    assert by_name["cross"]["exclusive_wall_ns"] is None


def test_filtered_starts_are_reconstructed_and_legacy_chronology_is_unavailable() -> None:
    _, ledger = reduce_events(
        [
            _terminal("precise", 10, 20),
            {
                "event": "approx.error",
                "span_id": "approx",
                "pid": 7,
                "native_thread_id": 1,
                "monotonic_ns": 5_000_000,
                "duration_ms": 2,
                "outcome": "error",
            },
            {"event": "legacy.ok", "span_id": "legacy", "ts": "2026-01-01T00:00:00.000Z", "duration_ms": 5},
            {"event": "unfinished.start", "span_id": "unfinished", "pid": 7, "native_thread_id": 1, "monotonic_ns": 2},
            {"event": "matched.start", "span_id": "matched", "pid": 7, "native_thread_id": 1, "monotonic_ns": 3},
            {
                "event": "matched.error",
                "span_id": "matched",
                "pid": 7,
                "native_thread_id": 1,
                "monotonic_ns": 4,
                "duration_ms": 0,
                "outcome": "error",
            },
            {"event": "log.delivery", "dropped": 3},
            {"event": "log.field_rejected", "field": "span_end_ns"},
        ]
    )
    assert ledger["coverage"]["matched_spans"] == 1
    assert ledger["coverage"]["unmatched_starts"] == 1
    assert ledger["coverage"]["unmatched_terminals"] == 3
    assert ledger["coverage"]["inferred_spans"] == 1
    assert ledger["coverage"]["unavailable_spans"] == 1
    assert ledger["coverage"]["measured_spans"] == 2
    assert ledger["coverage"]["rejected_fields"] == 1
    assert ledger["loss_counters"] == {"dropped": 3}
    assert next(s for s in ledger["spans"] if s["name"] == "approx")["exclusive_wall_ns"] is None
    assert next(s for s in ledger["spans"] if s["name"] == "legacy")["start_ns"] is None


def test_cli_keeps_invalid_line_coverage_and_writes_perfetto_slices(tmp_path: Path) -> None:
    source = tmp_path / "events.jsonl"
    source.write_text(json.dumps(_terminal("body", 5, 15)) + "\ninvalid\n[]\n", encoding="utf-8")
    assert main([str(source), "--out", str(tmp_path / "out")]) == 0
    ledger = json.loads((tmp_path / "out" / "ledger.json").read_text())
    trace = json.loads((tmp_path / "out" / "trace.json").read_text())
    assert ledger["input_coverage"] == {"lines": 3, "malformed_json": 1, "non_object": 1}
    assert trace["traceEvents"][0]["dur"] == 0.01


def test_explicit_task_identity_links_queue_to_service_on_integer_logical_lane() -> None:
    trace, ledger = reduce_events(
        [
            _terminal("compute.queue", 1, 10, span_kind="wait", task_id="task"),
            _terminal("compute.service", 12, 25, tid=2, task_id="task", cpu=(1, 9)),
        ]
    )
    flows = [e for e in trace["traceEvents"] if e["name"] == "queue to service"]
    assert len(flows) == 2
    assert flows[0]["tid"] >= (1 << 31)
    assert flows[1]["tid"] == 2
    assert all(isinstance(e["tid"], int) for e in trace["traceEvents"])
    wait = next(s for s in ledger["spans"] if s["kind"] == "wait")
    assert wait["logical_tid"] != wait["tid"]
    assert wait["exclusive_wall_ns"] is None


def test_duration_event_inheriting_span_id_cannot_replace_real_causal_owner() -> None:
    trace, ledger = reduce_events(
        [
            _terminal("root", 0, 100_000_000, cpu=(0, 80_000_000)),
            _terminal("compute.service", 10_000_000, 90_000_000, parent="root", cpu=(5_000_000, 65_000_000)),
            _terminal("child", 20_000_000, 40_000_000, parent="compute.service", cpu=(10_000_000, 20_000_000)),
            {
                "event": "foo.ok",
                "span_id": "compute.service",
                "trace_id": "trace",
                "parent_span_id": "root",
                "pid": 7,
                "native_thread_id": 1,
                "monotonic_ns": 80_000_000,
                "duration_ms": 1,
                "outcome": "ok",
            },
        ]
    )
    by_name = {s["name"]: s for s in ledger["spans"]}
    assert by_name["root"]["exclusive_wall_ns"] == 20_000_000
    assert by_name["root"]["exclusive_cpu_ns"] == 20_000_000
    assert by_name["compute.service"]["exclusive_wall_ns"] == 60_000_000
    assert by_name["compute.service"]["exclusive_cpu_ns"] == 50_000_000
    assert by_name["compute.service"]["causal_owner"] is True
    assert by_name["foo"]["causal_owner"] is False
    assert by_name["foo"]["identity_source"] == "correlation_only"
    assert by_name["foo"]["parent_span_id"] is None
    assert by_name["foo"]["chronology"] == "inferred"
    assert by_name["foo"]["exclusive_wall_ns"] is None
    assert ledger["coverage"]["duplicate_span_ids"] == 1
    assert ledger["coverage"]["correlation_id_collisions"] == 1
    assert ledger["coverage"]["ambiguous_owner_ids"] == 0
    assert sum(e["ph"] == "f" for e in trace["traceEvents"]) == 2
    child_cause = next(e for e in trace["traceEvents"] if e["ph"] == "s" and e["id"].endswith(":child"))
    assert child_cause["ts"] == 20_000


def test_duplicate_explicit_owner_ids_refuse_causal_and_exclusive_claims() -> None:
    trace, ledger = reduce_events(
        [
            _terminal("root", 0, 100, cpu=(0, 80)),
            _terminal("first", 10, 50, parent="root", span_id="shared", cpu=(5, 30)),
            _terminal("second", 20, 70, parent="root", span_id="shared", cpu=(10, 50)),
            _terminal("child", 30, 40, parent="shared", cpu=(15, 20)),
        ]
    )
    by_name = {s["name"]: s for s in ledger["spans"]}
    assert by_name["first"]["causal_owner"] is False
    assert by_name["second"]["causal_owner"] is False
    assert by_name["root"]["exclusive_wall_ns"] is None
    assert by_name["root"]["exclusive_cpu_ns"] is None
    assert by_name["root"]["cpu_coverage"] == "ambiguous_child_identity"
    assert by_name["first"]["cpu_coverage"] == "ambiguous_span_identity"
    assert ledger["coverage"]["ambiguous_owner_ids"] == 1
    assert ledger["coverage"]["duplicate_span_ids"] == 1
    assert ledger["coverage"]["missing_parents"] == 1
    assert not any(e["ph"] in {"s", "f"} for e in trace["traceEvents"])

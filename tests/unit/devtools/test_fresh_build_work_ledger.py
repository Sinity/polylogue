"""Offline measurements preserve causality without granting execution authority."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from devtools.fresh_build_bench.work_ledger import export_work_ledger, reduce_event_ledger


def _span(
    name: str, start: int, end: int, *, parent: str | None = None, tid: int = 1, cpu: tuple[int, int] | None = None
) -> dict[str, object]:
    return {
        "event": f"daemon.{name}.ok",
        "span_id": name,
        "parent_span_id": parent,
        "trace_id": "trace",
        "pid": 7,
        "native_thread_id": tid,
        "monotonic_ns": end,
        "span_start_ns": start,
        "span_end_ns": end,
        "span_start_pid": 7,
        "span_start_native_thread_id": tid,
        "cpu_scope": "thread" if cpu else "unavailable",
        "thread_cpu_start_ns": cpu[0] if cpu else None,
        "thread_cpu_end_ns": cpu[1] if cpu else None,
        "outcome": "ok",
        "query": "PRIVATE SQL",
        "payload": "PRIVATE PAYLOAD",
    }


def test_nested_overlap_missing_cpu_and_private_fields() -> None:
    events = [
        _span("parent", 0, 100, cpu=(0, 70)),
        _span("child", 10, 40, parent="parent", cpu=(5, 20)),
        _span("sibling", 30, 60, parent="parent", cpu=(15, 40)),
        _span("async", 20, 90, parent="parent", tid=2),
        {"event": "daemon.writer.released", "monotonic_ns": 101, "hold_ms": 2, "wait_ms": 4},
    ]
    trace, ledger = reduce_event_ledger(events)
    rows = {row["name"]: row for row in ledger["spans"]}
    assert rows["daemon.parent"]["exclusive_wall_ns"] == 50
    assert rows["daemon.parent"]["exclusive_cpu_ns"] == 35
    assert rows["daemon.parent"]["observed_cross_thread_child_wall_union_ns"] == 70
    assert rows["daemon.async"]["inclusive_cpu_ns"] is None
    assert ledger["cpu_coverage"]["measured_thread_cpu_union_ns"] == 70
    assert ledger["qualified_body_thread_cpu_union_ns"] is None
    complete = [event for event in trace["traceEvents"] if event["ph"] == "X"]
    assert len({event["tid"] for event in complete}) == 4
    assert any(event["ph"] == "s" for event in trace["traceEvents"])
    assert ledger["writer_duration_observations"] == [
        {"monotonic_ns": 101, "hold_ms": 2, "wait_ms": 4, "outcome": None}
    ]
    serialized = json.dumps([trace, ledger])
    assert "PRIVATE" not in serialized
    assert '"trace"' not in serialized
    assert ledger["allocation_provider"]["status"] == "unavailable"
    assert reduce_event_ledger(events) == (trace, ledger)


def test_loss_withdraws_qualified_cpu_without_discarding_observed_endpoints() -> None:
    events: list[dict[str, object]] = []
    for name, seq, event in (("a", 1, "started"), ("a", 3, "ok")):
        row = _span("body", 0, 10, cpu=(0, 5))
        row.update(
            event=f"work.body.{event}",
            attempt_id=name,
            component="storage",
            operation="parse",
            process_generation_id="process",
            worker_generation_id="worker",
            producer_seq=seq,
            work_key="private/key",
        )
        events.append(row)
    _, ledger = reduce_event_ledger(events)
    assert ledger["known_capture_gap"] is True
    assert ledger["work_observations"]["operations"][0]["execution_amplification"] is None
    assert ledger["qualified_body_thread_cpu_union_ns"] is None
    assert ledger["work_observations"]["body_thread_cpu_union_ns"] == 5


@pytest.fixture
def external_tmp() -> Iterator[Path]:
    # Managed pytest's tmp_path is deliberately inside the checkout. This
    # exporter must exercise its production refusal rather than bypass it.
    parent = Path("/realm/tmp/work") if Path("/realm/tmp/work").is_dir() else Path("/var/tmp")
    with TemporaryDirectory(prefix="polylogue-ledger-test-", dir=parent) as directory:
        yield Path(directory)


def test_export_bound_hashes_no_overwrite_or_raw_ancestor(external_tmp: Path) -> None:
    tmp_path = external_tmp
    source = tmp_path / "raw" / "events.jsonl"
    source.parent.mkdir()
    source.write_text(json.dumps(_span("parent", 0, 100)) + "\nBAD JSON\n[]\n")
    destination = tmp_path / "export"
    manifest = export_work_ledger(source, destination)
    assert manifest["state"] == "complete"
    assert manifest["known_capture_gap"] is True
    assert len(manifest["input_sha256"]) == 64
    assert set(manifest["artifacts"]) == {"trace.json", "ledger.json"}
    assert json.loads((destination / "ledger.json").read_text())["input_loss"] == {
        "malformed_records": 1,
        "non_object_records": 1,
    }
    with pytest.raises(FileExistsError):
        export_work_ledger(source, destination)
    with pytest.raises(ValueError, match="raw input"):
        export_work_ledger(source, source.parent)


def test_export_refuses_any_checkout_ancestor(external_tmp: Path) -> None:
    source = external_tmp / "events.jsonl"
    source.write_text("{}\n")
    checkout = external_tmp / "other-checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: neutral\n")
    with pytest.raises(ValueError, match="checkout"):
        export_work_ledger(source, checkout / "export")
    assert not (checkout / "export").exists()

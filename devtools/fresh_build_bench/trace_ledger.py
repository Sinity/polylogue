"""Reduce structured JSONL into causal slices and a coverage-qualified ledger.

Run ``python -m devtools.fresh_build_bench.trace_ledger LOG --out DIR``.
Nanosecond clocks are process-host monotonic clocks. Queue slices occupy logical
lanes; they never count as the submitting thread's execution. Exclusive wall is
relative to observed same-thread work children. CPU uses thread-clock endpoints,
never process CPU or concurrent child wall. Uninstrumented work stays in the
parent's residual. Legacy duration-only records have no precise chronology.
An inherited span ID is correlation, not ownership: causal owners need explicit
terminal endpoints or an exactly matched start. Duplicate genuine owners make
causal and exclusive measurements unavailable.
Global measured CPU coverage unions comparable clock intervals per (PID, TID),
including nested siblings only once. Parent-local exclusive CPU is not additive.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_TERMINALS = frozenset({"ok", "empty", "skipped", "degraded", "refused", "unmeasured", "error"})


def _ns(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def interval_union(intervals: Iterable[tuple[int, int]]) -> int:
    """Measure the union, so nested and overlapping intervals count once."""
    total = 0
    right: int | None = None
    for start, end in sorted(intervals):
        if end <= start:
            continue
        total += max(0, end - max(start, right if right is not None else start))
        right = max(end, right if right is not None else end)
    return total


def _covered(start: int, end: int, intervals: Iterable[tuple[int, int]]) -> int:
    return interval_union((max(start, left), min(end, right)) for left, right in intervals)


@dataclass
class Slice:
    name: str
    span_id: str
    trace_id: str | None
    parent_span_id: str | None
    task_id: str | None
    pid: int | None
    tid: int | None
    end_tid: int | None
    kind: str
    outcome: str
    start_ns: int | None
    end_ns: int | None
    chronology: str
    inclusive_wall_ns: int | None
    exclusive_wall_ns: int | None = None
    thread_cpu_start_ns: int | None = None
    thread_cpu_end_ns: int | None = None
    inclusive_cpu_ns: int | None = None
    exclusive_cpu_ns: int | None = None
    cpu_coverage: str = "unavailable"
    logical_tid: int | None = None
    identity_source: str = "correlation_only"
    causal_owner: bool = False

    @property
    def lane(self) -> int:
        # Virtual integer TIDs keep Chrome/Perfetto lanes separate from Linux
        # native TIDs without giving a logical wait execution occupancy.
        return self.logical_tid if self.logical_tid is not None else self.tid or 0


def _cpu_coverage(slices: list[Slice]) -> dict[str, Any]:
    """Account for observed CPU without requiring a complete parent tree."""
    work = [s for s in slices if s.kind == "work"]
    threads: dict[tuple[int, int], list[Slice]] = defaultdict(list)
    for item in work:
        if (
            item.causal_owner
            and item.chronology == "measured"
            and item.pid is not None
            and item.tid is not None
            and item.tid == item.end_tid
            and item.thread_cpu_start_ns is not None
            and item.thread_cpu_end_ns is not None
        ):
            threads[(item.pid, item.tid)].append(item)
    by_thread = []
    for (pid, tid), items in sorted(threads.items()):
        intervals = [
            (s.thread_cpu_start_ns, s.thread_cpu_end_ns)
            for s in items
            if s.thread_cpu_start_ns is not None and s.thread_cpu_end_ns is not None
        ]
        left = min(start for start, _ in intervals)
        right = max(end for _, end in intervals)
        covered = interval_union(intervals)
        inclusive_sum = sum(end - start for start, end in intervals)
        by_thread.append(
            {
                "pid": pid,
                "native_thread_id": tid,
                "included_spans": len(items),
                "first_thread_cpu_ns": left,
                "last_thread_cpu_ns": right,
                "measured_cpu_union_ns": covered,
                "clock_range_cpu_ns": right - left,
                "uncovered_cpu_between_endpoints_ns": right - left - covered,
                "inclusive_cpu_sum_ns_not_additive": inclusive_sum,
                "overlapping_inclusive_cpu_ns": inclusive_sum - covered,
            }
        )
    included = sum(row["included_spans"] for row in by_thread)
    exclusive = [s.exclusive_cpu_ns for s in work if s.exclusive_cpu_ns is not None]
    return {
        "scope": "union of measured synchronous work clock intervals per (PID, native TID)",
        "measured_thread_cpu_union_ns": sum(row["measured_cpu_union_ns"] for row in by_thread) if by_thread else None,
        "included_spans": included,
        "excluded_work_spans": len(work) - included,
        "threads": by_thread,
        "parent_local_exclusive_cpu_sum_ns_not_additive": sum(exclusive) if exclusive else None,
        "parent_local_exclusive_cpu_known_spans": len(exclusive),
        "limitations": (
            "Clock ranges compare endpoints of one native thread in one process. "
            "Uncovered CPU between those endpoints is uninstrumented; CPU outside "
            "the ranges and on other threads is unavailable. Native TID reuse is "
            "not identified. Parent-local residuals can overlap across siblings "
            "and must not be summed as global coverage."
        ),
    }


def reduce_events(events: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (Chrome trace, ledger); missing data is counted, never guessed."""
    counts: Counter[str] = Counter()
    starts: dict[tuple[object, object, object, str], Mapping[str, Any]] = {}
    terminals: list[tuple[tuple[object, object, object, str], Mapping[str, Any]]] = []
    losses: dict[str, int] = {}
    work_events: list[Mapping[str, Any]] = []
    for event in events:
        if str(event.get("event", "")).startswith("work."):
            work_events.append(event)
        counts["events"] += 1
        if _ns(event.get("monotonic_ns")) is None:
            counts["missing_monotonic_timestamps"] += 1
        if _ns(event.get("native_thread_id")) is None:
            counts["missing_native_thread_ids"] += 1
        if event.get("event") == "log.field_rejected":
            counts["rejected_fields"] += 1
        if "required_measurement_fields_omitted" in event:
            omitted = _ns(event["required_measurement_fields_omitted"])
            if omitted is None:
                counts["invalid_required_measurement_omission_counts"] += 1
                counts["measurement_loss_events"] += 1
            elif omitted:
                counts["required_measurement_fields_omitted"] += omitted
                counts["measurement_loss_events"] += 1
        if str(event.get("event", "")).startswith("log."):
            for field in ("dropped", "undrained", "failures", "priority_evictions"):
                value = _ns(event.get(field))
                if value is not None:
                    losses[field] = max(losses.get(field, 0), value)
        name, _, suffix = str(event.get("event", "")).rpartition(".")
        if not event.get("span_id") or not name:
            continue
        key = (event.get("pid"), event.get("trace_id"), event["span_id"], name)
        if suffix == "start":
            if key in starts:
                counts["duplicate_starts"] += 1
            starts[key] = event
        elif suffix in _TERMINALS and ("span_end_ns" in event or "duration_ms" in event):
            terminals.append((key, event))
    slices: list[Slice] = []
    for key, terminal in terminals:
        start_event = starts.pop(key, None)
        counts["matched_spans" if start_event is not None else "unmatched_terminals"] += 1
        start = _ns(terminal.get("span_start_ns"))
        end = _ns(terminal.get("span_end_ns"))
        chronology = "measured"
        if start is None and start_event is not None:
            start = _ns(start_event.get("monotonic_ns"))
        if end is None:
            end = _ns(terminal.get("monotonic_ns"))
        if start is None and end is not None:
            duration = terminal.get("duration_ms")
            if (
                isinstance(duration, int | float)
                and not isinstance(duration, bool)
                and math.isfinite(duration)
                and duration >= 0
            ):
                start = end - round(duration * 1_000_000)
                chronology = "inferred"
        if start is None or end is None or start < 0 or end < start:
            start = end = None
            chronology = "unavailable"
        pid = _ns(terminal.get("span_start_pid", terminal.get("pid")))
        tid = _ns(terminal.get("span_start_native_thread_id"))
        if tid is None:
            tid = _ns((start_event or terminal).get("native_thread_id"))
        end_tid = _ns(terminal.get("native_thread_id"))
        cpu_start = _ns(terminal.get("thread_cpu_start_ns"))
        cpu_end = _ns(terminal.get("thread_cpu_end_ns"))
        valid_cpu = (
            chronology == "measured"
            and terminal.get("cpu_scope") == "thread"
            and tid is not None
            and tid == end_tid
            and pid == _ns(terminal.get("pid"))
            and cpu_start is not None
            and cpu_end is not None
            and cpu_end >= cpu_start
        )
        explicit_endpoints = (
            _ns(terminal.get("span_start_ns")) is not None and _ns(terminal.get("span_end_ns")) is not None
        )
        identity_source = (
            "terminal_endpoints"
            if explicit_endpoints and chronology == "measured"
            else "matched_start"
            if start_event is not None and chronology == "measured"
            else "correlation_only"
        )
        if identity_source == "correlation_only":
            counts["detached_correlated_events"] += 1
            valid_cpu = False
            if chronology == "measured":
                chronology = "inferred"
        counts[f"{chronology}_spans"] += 1
        item = Slice(
            name=key[3],
            identity_source=identity_source,
            span_id=str(terminal["span_id"]),
            trace_id=terminal.get("trace_id"),
            parent_span_id=terminal.get("parent_span_id") if identity_source != "correlation_only" else None,
            task_id=terminal.get("task_id"),
            pid=pid,
            tid=tid,
            end_tid=end_tid,
            kind=str(terminal.get("span_kind", "work")),
            outcome=str(terminal.get("outcome", "unknown")),
            start_ns=start,
            end_ns=end,
            chronology=chronology,
            inclusive_wall_ns=None if start is None or end is None else end - start,
            thread_cpu_start_ns=cpu_start if valid_cpu else None,
            thread_cpu_end_ns=cpu_end if valid_cpu else None,
            inclusive_cpu_ns=cpu_end - cpu_start
            if valid_cpu and cpu_end is not None and cpu_start is not None
            else None,
        )
        slices.append(item)
    counts["unmatched_starts"] = len(starts)
    children: dict[tuple[int | None, str | None, str], list[Slice]] = defaultdict(list)
    identities: dict[tuple[int | None, str | None, str], list[Slice]] = defaultdict(list)
    for item in slices:
        identities[(item.pid, item.trace_id, item.span_id)].append(item)
    owners: dict[tuple[int | None, str | None, str], Slice] = {}
    for identity, occurrences in identities.items():
        candidates = [s for s in occurrences if s.identity_source != "correlation_only"]
        if len(occurrences) > 1:
            counts["duplicate_span_ids"] += 1
        if len(candidates) == 1:
            candidates[0].causal_owner = True
            owners[identity] = candidates[0]
            if len(occurrences) > 1:
                counts["correlation_id_collisions"] += 1
        elif len(candidates) > 1:
            counts["ambiguous_owner_ids"] += 1

    for item in slices:
        if item.identity_source != "correlation_only" and item.parent_span_id:
            parent_key = (item.pid, item.trace_id, item.parent_span_id)
            children[parent_key].append(item)
            if parent_key not in owners:
                counts["missing_parents"] += 1
    trace_events: list[dict[str, Any]] = []
    origin = min((s.start_ns for s in slices if s.start_ns is not None), default=0)
    queue_tasks: dict[tuple[int | None, str | None, str], Slice] = {}
    next_logical_tid = 1 << 31
    for item in slices:
        if item.kind == "wait":
            item.logical_tid = next_logical_tid
            next_logical_tid += 1
            if item.causal_owner:
                queue_tasks[(item.pid, item.trace_id, item.task_id or item.span_id)] = item
            trace_events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": item.pid or 0,
                    "tid": item.lane,
                    "args": {"name": f"wait:{item.task_id or item.span_id}"},
                }
            )
    for item in slices:
        direct = children.get((item.pid, item.trace_id, item.span_id), [])
        same_thread = [c for c in direct if c.tid == item.tid and c.end_tid == item.tid and c.kind == "work"]
        wall_children = [
            (c.start_ns, c.end_ns)
            for c in same_thread
            if c.chronology == "measured" and c.start_ns is not None and c.end_ns is not None
        ]
        if (
            item.causal_owner
            and item.chronology == "measured"
            and item.start_ns is not None
            and item.end_ns is not None
            and item.tid is not None
            and item.tid == item.end_tid
            and item.kind == "work"
            and all(c.causal_owner and c.chronology == "measured" for c in same_thread)
        ):
            item.exclusive_wall_ns = item.end_ns - item.start_ns - _covered(item.start_ns, item.end_ns, wall_children)
        if (
            item.inclusive_cpu_ns is not None
            and item.thread_cpu_start_ns is not None
            and item.thread_cpu_end_ns is not None
        ):
            if not item.causal_owner:
                item.cpu_coverage = "ambiguous_span_identity"
            elif not all(c.causal_owner for c in same_thread):
                item.cpu_coverage = "ambiguous_child_identity"
            elif all(c.inclusive_cpu_ns is not None for c in same_thread):
                cpu_children = [
                    (c.thread_cpu_start_ns, c.thread_cpu_end_ns)
                    for c in same_thread
                    if c.thread_cpu_start_ns is not None and c.thread_cpu_end_ns is not None
                ]
                item.exclusive_cpu_ns = item.inclusive_cpu_ns - _covered(
                    item.thread_cpu_start_ns, item.thread_cpu_end_ns, cpu_children
                )
                item.cpu_coverage = "thread_endpoints_observed_children"
            else:
                item.cpu_coverage = "missing_child_thread_endpoints"
        if item.start_ns is None or item.end_ns is None:
            continue
        args = asdict(item)
        trace_events.append(
            {
                "name": item.name,
                "cat": item.kind,
                "ph": "X",
                "ts": (item.start_ns - origin) / 1000,
                "dur": (item.end_ns - item.start_ns) / 1000,
                "pid": item.pid or 0,
                "tid": item.lane,
                "args": args,
            }
        )
        parent = owners.get((item.pid, item.trace_id, item.parent_span_id or "")) if item.causal_owner else None
        if parent is not None and parent.start_ns is not None and parent.end_ns is not None:
            point = max(parent.start_ns, min(parent.end_ns, item.start_ns))
            flow_id = f"{item.pid}:{item.trace_id}:{item.span_id}"
            trace_events.extend(
                [
                    {
                        "name": "parent cause",
                        "cat": "cause",
                        "ph": "s",
                        "id": flow_id,
                        "ts": (point - origin) / 1000,
                        "pid": parent.pid or 0,
                        "tid": parent.lane,
                    },
                    {
                        "name": "parent cause",
                        "cat": "cause",
                        "ph": "f",
                        "bp": "e",
                        "id": flow_id,
                        "ts": (item.start_ns - origin) / 1000,
                        "pid": item.pid or 0,
                        "tid": item.lane,
                    },
                ]
            )
    for item in slices:
        queued = queue_tasks.get((item.pid, item.trace_id, item.task_id or ""))
        if (
            item.causal_owner
            and item.name == "compute.service"
            and queued is not None
            and queued.end_ns is not None
            and item.start_ns is not None
        ):
            flow_id = f"task:{item.pid}:{item.trace_id}:{item.task_id}"
            trace_events.extend(
                [
                    {
                        "name": "queue to service",
                        "cat": "cause",
                        "ph": "s",
                        "id": flow_id,
                        "ts": (queued.end_ns - origin) / 1000,
                        "pid": queued.pid or 0,
                        "tid": queued.lane,
                    },
                    {
                        "name": "queue to service",
                        "cat": "cause",
                        "ph": "f",
                        "bp": "e",
                        "id": flow_id,
                        "ts": (item.start_ns - origin) / 1000,
                        "pid": item.pid or 0,
                        "tid": item.lane,
                    },
                ]
            )
    # Counter keys exist even on legacy and empty input.
    for field in (
        "events",
        "matched_spans",
        "unmatched_terminals",
        "unmatched_starts",
        "inferred_spans",
        "measured_spans",
        "unavailable_spans",
        "missing_monotonic_timestamps",
        "missing_native_thread_ids",
        "rejected_fields",
        "missing_parents",
        "duplicate_span_ids",
        "correlation_id_collisions",
        "ambiguous_owner_ids",
        "detached_correlated_events",
        "required_measurement_fields_omitted",
        "invalid_required_measurement_omission_counts",
        "measurement_loss_events",
    ):
        counts.setdefault(field, 0)
    ledger = {
        "format": "polylogue.trace-ledger.v1",
        "coverage": dict(counts),
        "loss_counters": losses or None,
        "clock": "host_monotonic_ns",
        "exclusive_scope": "observed same-thread work children only",
        "cpu_coverage": _cpu_coverage(slices),
        "required_field_omissions_absent": not counts["measurement_loss_events"],
        "precise_chronology_available": bool(counts["measured_spans"]),
        "spans": [asdict(s) for s in slices],
    }
    from devtools.fresh_build_bench.work_ledger import mark_capture_gap, reduce_work_events

    ledger["work_observations"] = reduce_work_events(work_events)
    if any(losses.values()) or counts["measurement_loss_events"]:
        mark_capture_gap(ledger["work_observations"], "known_logging_capture_loss")
    return {"traceEvents": trace_events, "displayTimeUnit": "ms", "metadata": {"origin_monotonic_ns": origin}}, ledger


def read_events(path: Path, coverage: Counter[str]) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            coverage["lines"] += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                coverage["malformed_json"] += 1
                continue
            if isinstance(item, dict):
                yield item
            else:
                coverage["non_object"] += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    input_coverage: Counter[str] = Counter()
    trace, ledger = reduce_events(read_events(args.log, input_coverage))
    ledger["input_coverage"] = dict(input_coverage)
    if input_coverage["malformed_json"] or input_coverage["non_object"]:
        from devtools.fresh_build_bench.work_ledger import mark_capture_gap

        mark_capture_gap(ledger["work_observations"], "unreadable_input_records")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("trace.json", trace), ("ledger.json", ledger)):
        (args.out / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "coverage": ledger["coverage"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

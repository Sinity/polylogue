"""Reduce structured JSONL into causal slices and a coverage-qualified ledger.

Run ``python -m devtools.fresh_build_bench.trace_ledger LOG --out DIR``.
Nanosecond clocks are process-host monotonic clocks. Queue slices occupy logical
lanes; they never count as the submitting thread's execution. Exclusive wall is
relative to observed same-thread work children. CPU uses thread-clock endpoints,
never process CPU or concurrent child wall. Uninstrumented work stays in the
parent's residual. Legacy duration-only records have no precise chronology.
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

    @property
    def lane(self) -> int:
        # Virtual integer TIDs keep Chrome/Perfetto lanes separate from Linux
        # native TIDs without giving a logical wait execution occupancy.
        return self.logical_tid if self.logical_tid is not None else self.tid or 0


def reduce_events(events: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (Chrome trace, ledger); missing data is counted, never guessed."""
    counts: Counter[str] = Counter()
    starts: dict[tuple[object, object, object, str], Mapping[str, Any]] = {}
    terminals: list[tuple[tuple[object, object, object, str], Mapping[str, Any]]] = []
    losses: dict[str, int] = {}
    for event in events:
        counts["events"] += 1
        if _ns(event.get("monotonic_ns")) is None:
            counts["missing_monotonic_timestamps"] += 1
        if _ns(event.get("native_thread_id")) is None:
            counts["missing_native_thread_ids"] += 1
        if event.get("event") == "log.field_rejected":
            counts["rejected_fields"] += 1
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
        counts[f"{chronology}_spans"] += 1
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
        item = Slice(
            name=key[3],
            span_id=str(terminal["span_id"]),
            trace_id=terminal.get("trace_id"),
            parent_span_id=terminal.get("parent_span_id"),
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
    owners = {(s.pid, s.trace_id, s.span_id): s for s in slices}
    for item in slices:
        if item.parent_span_id:
            parent_key = (item.pid, item.trace_id, item.parent_span_id)
            children[parent_key].append(item)
            if parent_key not in owners:
                counts["missing_parents"] += 1
    trace_events: list[dict[str, Any]] = []
    origin = min((s.start_ns for s in slices if s.start_ns is not None), default=0)
    queue_tasks: dict[tuple[int | None, str | None, str], Slice] = {}
    for item in slices:
        if item.kind == "wait":
            item.logical_tid = (1 << 31) + len(queue_tasks)
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
            item.chronology == "measured"
            and item.start_ns is not None
            and item.end_ns is not None
            and item.tid is not None
            and item.tid == item.end_tid
            and item.kind == "work"
            and all(c.chronology == "measured" for c in same_thread)
        ):
            item.exclusive_wall_ns = item.end_ns - item.start_ns - _covered(item.start_ns, item.end_ns, wall_children)
        if (
            item.inclusive_cpu_ns is not None
            and item.thread_cpu_start_ns is not None
            and item.thread_cpu_end_ns is not None
        ):
            if all(c.inclusive_cpu_ns is not None for c in same_thread):
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
        parent = owners.get((item.pid, item.trace_id, item.parent_span_id or ""))
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
            item.name == "compute.service"
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
    ):
        counts.setdefault(field, 0)
    ledger = {
        "format": "polylogue.trace-ledger.v1",
        "coverage": dict(counts),
        "loss_counters": losses or None,
        "clock": "host_monotonic_ns",
        "exclusive_scope": "observed same-thread work children only",
        "precise_chronology_available": bool(counts["measured_spans"]),
        "spans": [asdict(s) for s in slices],
    }
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
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("trace.json", trace), ("ledger.json", ledger)):
        (args.out / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "coverage": ledger["coverage"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

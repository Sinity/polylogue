"""Reduce opt-in owner body records without deduplicating equal work keys.

Counts describe the observed stream only. Paired entries and no internal
sequence gaps cannot prove that capture began before all work or retained its
final records. Artifact links record decisions, not lifetime or avoided CPU.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from devtools.fresh_build_bench.trace_ledger import interval_union

_TERMINALS = frozenset({"work.body.ok", "work.body.empty", "work.body.error", "work.body.unmeasured"})
_EVENTS = _TERMINALS | {"work.body.started", "work.artifact.consume"}


def mark_capture_gap(ledger: dict[str, Any], reason: str) -> None:
    """Retain observed counts while withdrawing ratios after known capture loss."""
    ledger["coverage"][reason] = True
    for operation in ledger["operations"]:
        operation["execution_amplification"] = None
        operation["amplification_coverage"] = "unavailable"


def reduce_work_events(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep bodies, keys, artifacts, decisions and coverage as separate facts."""
    coverage: Counter[str] = Counter()
    records: list[Mapping[str, Any]] = []
    transports: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    sequences: dict[tuple[str, str], set[int]] = defaultdict(set)
    for event in events:
        if event.get("event") not in _EVENTS:
            continue
        coverage["records"] += 1
        if event.get("required_measurement_fields_omitted", 0):
            coverage["required_field_loss_records"] += 1
        identities = (event.get("process_generation_id"), event.get("worker_generation_id"))
        sequence = event.get("producer_seq")
        if (
            not all(isinstance(value, str) and value for value in identities)
            or not isinstance(sequence, int)
            or isinstance(sequence, bool)
            or sequence < 1
        ):
            coverage["unidentified_records"] += 1
            continue
        process, worker = identities
        assert isinstance(process, str) and isinstance(worker, str)
        key = process, worker, sequence
        if key in transports:
            coverage["transport_duplicates"] += 1
            if dict(transports[key]) != dict(event):
                coverage["conflicting_transport_records"] += 1
            continue
        transports[key] = event
        sequences[(process, worker)].add(sequence)
        if not all(
            isinstance(event.get(field), str) and event[field] for field in ("component", "operation", "attempt_id")
        ):
            coverage["missing_owner_identity_records"] += 1
            continue
        records.append(event)
    for seq in sequences.values():
        coverage["internal_sequence_gaps"] += max(seq) - min(seq) + 1 - len(seq)

    attempts: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    consumers: list[Mapping[str, Any]] = []
    for event in records:
        if event["event"] == "work.artifact.consume":
            consumers.append(event)
        else:
            attempts[(event["process_generation_id"], event["attempt_id"])].append(event)

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    products: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    cpu: dict[tuple[str, str, int, int], list[tuple[int, int]]] = defaultdict(list)
    body_rows: list[dict[str, Any]] = []
    for items in attempts.values():
        starts = [item for item in items if item["event"] == "work.body.started"]
        terminals = [item for item in items if item["event"] in _TERMINALS]
        representative = terminals[0] if terminals else starts[0]
        owner = representative["component"], representative["operation"]
        detail = {
            field: representative.get(field)
            for field in (
                "component",
                "operation",
                "attempt_id",
                "span_id",
                "task_id",
                "work_key",
                "work_key_reason",
                "artifact_ref",
                "process_generation_id",
                "worker_generation_id",
                "pid",
                "native_thread_id",
                "outcome",
                "span_start_ns",
                "span_end_ns",
                "thread_cpu_start_ns",
                "thread_cpu_end_ns",
                "cpu_scope",
            )
        }
        detail["paired"] = len(starts) == len(terminals) == 1
        body_rows.append(detail)
        group = groups.setdefault(
            owner,
            {
                "component": owner[0],
                "operation": owner[1],
                "started": 0,
                "completed": 0,
                "failed": 0,
                "incomplete": 0,
                "unknown_key_bodies": 0,
                "keys": set(),
                "artifact_products": 0,
            },
        )
        group["started"] += len(starts)
        if len(starts) != 1 or len(terminals) != 1:
            group["incomplete"] += 1
            coverage["unpaired_or_ambiguous_bodies"] += 1
            continue
        terminal = terminals[0]
        if any(
            (item["component"], item["operation"], item.get("span_id"), item["worker_generation_id"])
            != (
                representative["component"],
                representative["operation"],
                representative.get("span_id"),
                representative["worker_generation_id"],
            )
            for item in items
        ):
            detail["paired"] = False
            group["incomplete"] += 1
            coverage["conflicting_body_identity"] += 1
            continue
        complete = terminal["event"] in {"work.body.ok", "work.body.empty"}
        group["completed" if complete else "failed"] += 1
        work_key = terminal.get("work_key")
        if isinstance(work_key, str) and work_key and "<truncated>" not in work_key:
            group["keys"].add(work_key)
        else:
            group["unknown_key_bodies"] += 1
        artifact = terminal.get("artifact_ref")
        if complete and isinstance(artifact, str) and artifact:
            group["artifact_products"] += 1
            products[artifact].append(terminal)
        left, right = terminal.get("thread_cpu_start_ns"), terminal.get("thread_cpu_end_ns")
        pid, tid = terminal.get("pid"), terminal.get("native_thread_id")
        if (
            terminal.get("cpu_scope") == "thread"
            and pid == terminal.get("span_start_pid")
            and tid == terminal.get("span_start_native_thread_id")
            and isinstance(left, int)
            and isinstance(right, int)
            and isinstance(pid, int)
            and isinstance(tid, int)
            and not any(isinstance(value, bool) for value in (left, right, pid, tid))
            and 0 <= left <= right
        ):
            cpu[(terminal["process_generation_id"], terminal["worker_generation_id"], pid, tid)].append((left, right))
        else:
            coverage["body_cpu_unavailable"] += 1

    coverage["ambiguous_artifact_producers"] = sum(len(items) != 1 for items in products.values())
    links: Counter[str] = Counter()
    for consumer in consumers:
        effect = consumer.get("effect")
        links[effect if isinstance(effect, str) else "unknown_decision"] += 1
        artifact = consumer.get("artifact_ref")
        if not artifact or artifact not in products:
            coverage["consumer_producer_unavailable"] += 1
    record_gaps = any(
        coverage[name]
        for name in (
            "required_field_loss_records",
            "unidentified_records",
            "missing_owner_identity_records",
            "conflicting_transport_records",
            "internal_sequence_gaps",
            "unpaired_or_ambiguous_bodies",
            "conflicting_body_identity",
        )
    )
    rows = []
    for _owner, group in sorted(groups.items()):
        keys = group.pop("keys")
        bodies = group["completed"] + group["failed"]
        ratio_available = not record_gaps and not group["unknown_key_bodies"] and bool(keys)
        group.update(
            distinct_complete_work_keys=len(keys),
            execution_amplification=bodies / len(keys) if ratio_available else None,
            amplification_coverage="paired_observed_bodies" if ratio_available else "unavailable",
        )
        rows.append(group)
    return {
        "scope": "observed owner bodies; not a full-run work-population certificate",
        "coverage": dict(coverage),
        "operations": rows,
        "bodies": body_rows,
        "artifact_products": len(products),
        "consumer_decisions": dict(links),
        "consumers": [
            {
                field: consumer.get(field)
                for field in (
                    "component",
                    "operation",
                    "attempt_id",
                    "artifact_ref",
                    "work_key",
                    "effect",
                    "reason",
                    "process_generation_id",
                    "worker_generation_id",
                    "pid",
                    "native_thread_id",
                    "monotonic_ns",
                )
            }
            for consumer in consumers
        ],
        "body_thread_cpu_union_ns": sum(interval_union(intervals) for intervals in cpu.values()) if cpu else None,
        "cpu_scope": "union of synchronous body intervals per process and worker generation, PID and native TID",
        "limitations": [
            "Unobserved beginning/end and wholly absent producers cannot be recovered from these events.",
            "Equal keys do not establish avoidable work or permit artifact reuse.",
            "Consumer decisions do not prove artifact cleanup or physical lifetime settlement.",
            "Body CPU is inclusive of nested bodies before lifetime-qualified interval union; it is not additive by operation.",
        ],
    }

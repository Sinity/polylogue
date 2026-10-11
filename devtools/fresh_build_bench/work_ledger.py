"""Reduce opt-in owner body records without deduplicating equal work keys.

Counts describe the observed stream only. Paired entries and no internal
sequence gaps cannot prove that capture began before all work or retained its
final records. Artifact links record decisions, not lifetime or avoided CPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
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


# Export is deliberately narrower than the logging schema. Free text, paths,
# SQL, input values and arbitrary nested context never cross this boundary.
_NUMERIC_FIELDS = frozenset(
    {
        "pid",
        "native_thread_id",
        "monotonic_ns",
        "span_start_ns",
        "span_end_ns",
        "span_start_pid",
        "span_start_native_thread_id",
        "thread_cpu_start_ns",
        "thread_cpu_end_ns",
        "duration_ms",
        "wait_ms",
        "hold_ms",
        "budget_ms",
        "producer_seq",
        "required_measurement_fields_omitted",
        "dropped",
        "undrained",
        "failures",
        "priority_evictions",
        "private_input_bytes",
        "borrowed_context_bytes",
        "shared_context_bytes",
        "unique_backing_bytes",
        "unique_context_count",
    }
)
_TOKEN_FIELDS = frozenset({"event", "component", "operation", "span_kind", "cpu_scope", "outcome", "effect"})
_ID_FIELDS = frozenset(
    {
        "span_id",
        "parent_span_id",
        "trace_id",
        "task_id",
        "attempt_id",
        "work_key",
        "artifact_ref",
        "process_generation_id",
        "worker_generation_id",
    }
)
_TOKEN = re.compile(r"[a-zA-Z0-9_.:-]{1,160}\Z")
_PREFIXES = ("work.", "compute.", "daemon.", "log.", "storage.", "source.", "validation.")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sanitize(event: Mapping[str, Any]) -> dict[str, Any] | None:
    name = event.get("event")
    if not isinstance(name, str) or not name.startswith(_PREFIXES) or not _TOKEN.fullmatch(name):
        return None
    result: dict[str, Any] = {}
    for field in _NUMERIC_FIELDS:
        value = event.get(field)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            result[field] = value
    for field in _TOKEN_FIELDS:
        value = event.get(field)
        if isinstance(value, str) and _TOKEN.fullmatch(value):
            result[field] = value
    if "required_measurement_fields_omitted" in event and "required_measurement_fields_omitted" not in result:
        result["required_measurement_fields_omitted"] = None
    for field in _ID_FIELDS:
        value = event.get(field)
        if isinstance(value, str) and value and "<truncated>" not in value:
            namespace = "span_id" if field == "parent_span_id" else field
            result[field] = _digest((namespace + "\0" + value).encode())
    return result


def _category(name: str) -> str:
    if name.startswith("work.artifact."):
        return "artifact_decision"
    if name == "work.body" or name.startswith("work.body."):
        return "body"
    if name.startswith("compute.queue"):
        return "queue_admission"
    if name.startswith("compute.service"):
        return "compute"
    if name.startswith("daemon.writer."):
        return "writer"
    if "readiness" in name or "frontier" in name:
        return "readiness"
    if "debt" in name or "retry" in name:
        return "retry"
    return "other_observed"


def reduce_event_ledger(
    events: Iterable[Mapping[str, Any]], *, capture_metadata: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reduce existing measurements, without supplying missing clock evidence.

    Capture metadata accepts only numeric malformed/non-object counts. Runtime
    and allocation providers require an independently bound receipt; absence is
    recorded here rather than inferred from a package or this reducer process.
    """
    from devtools.fresh_build_bench.trace_ledger import reduce_events

    sanitized: list[dict[str, Any]] = []
    rejected = 0
    categories: Counter[str] = Counter()
    writer: list[dict[str, Any]] = []
    for event in events:
        clean = _sanitize(event)
        if clean is None:
            rejected += 1
            continue
        sanitized.append(clean)
        categories[_category(clean["event"])] += 1
        if clean["event"] == "daemon.writer.released":
            writer.append({field: clean.get(field) for field in ("monotonic_ns", "wait_ms", "hold_ms", "outcome")})
    _trace, ledger = reduce_events(sanitized)
    capture = capture_metadata or {}
    input_loss = {
        field: value
        for field in ("malformed_records", "non_object_records")
        if type(value := capture.get(field, 0)) is int and value >= 0
    }
    known_loss = bool(
        any(input_loss.values())
        or any((ledger["loss_counters"] or {}).values())
        or ledger["coverage"]["measurement_loss_events"]
        or any(
            ledger["work_observations"]["coverage"].get(field, 0)
            for field in (
                "internal_sequence_gaps",
                "conflicting_transport_records",
                "unidentified_records",
                "missing_owner_identity_records",
                "unpaired_or_ambiguous_bodies",
                "conflicting_body_identity",
            )
        )
    )
    if known_loss:
        mark_capture_gap(ledger["work_observations"], "known_capture_gap")
    spans = sorted(
        ledger["spans"],
        key=lambda row: (
            row["start_ns"] if row["start_ns"] is not None else -1,
            row["pid"] or 0,
            row["span_id"],
            row["name"],
            row["end_ns"] or 0,
        ),
    )
    origin = min((row["start_ns"] for row in spans if row["start_ns"] is not None), default=None)
    traces: list[dict[str, Any]] = []
    owners: dict[tuple[Any, Any, Any], tuple[dict[str, Any], int]] = {}
    for lane, row in enumerate(spans, start=1):
        # A logical lane per span permits arbitrary asynchronous overlap in
        # Chrome's X format, while native thread identity remains an operand.
        if row["causal_owner"]:
            owners[(row["pid"], row["trace_id"], row["span_id"])] = row, lane
        left, right = row["start_ns"], row["end_ns"]
        if left is not None and right is not None and origin is not None:
            traces.append(
                {
                    "name": row["name"],
                    "cat": _category(row["name"]),
                    "ph": "X",
                    "pid": row["pid"] or 0,
                    "tid": lane,
                    "ts": (left - origin) / 1000,
                    "dur": (right - left) / 1000,
                    "args": row.copy(),
                }
            )
    children: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in spans:
        children[(row["pid"], row["trace_id"], row["parent_span_id"])].append(row)
    for lane, row in enumerate(spans, start=1):
        parent = owners.get((row["pid"], row["trace_id"], row["parent_span_id"]))
        if parent and row["causal_owner"]:
            source, source_lane = parent
            if (
                origin is not None
                and source["start_ns"] is not None
                and source["end_ns"] is not None
                and row["start_ns"] is not None
            ):
                point = max(source["start_ns"], min(source["end_ns"], row["start_ns"]))
                flow = f"{lane}:{row['span_id']}"
                traces.extend(
                    [
                        {
                            "name": "parent cause",
                            "ph": "s",
                            "id": flow,
                            "pid": source["pid"] or 0,
                            "tid": source_lane,
                            "ts": (point - origin) / 1000,
                        },
                        {
                            "name": "parent cause",
                            "ph": "f",
                            "bp": "e",
                            "id": flow,
                            "pid": row["pid"] or 0,
                            "tid": lane,
                            "ts": (row["start_ns"] - origin) / 1000,
                        },
                    ]
                )
        cross_lane = [
            (child["start_ns"], child["end_ns"])
            for child in children.get((row["pid"], row["trace_id"], row["span_id"]), [])
            if child["causal_owner"]
            and child["tid"] != row["tid"]
            and child["chronology"] == "measured"
            and child["start_ns"] is not None
            and child["end_ns"] is not None
        ]
        row["observed_cross_thread_child_wall_union_ns"] = (
            interval_union(
                [
                    (max(row["start_ns"], a), min(row["end_ns"], b))
                    for a, b in cross_lane
                    if max(row["start_ns"], a) < min(row["end_ns"], b)
                ]
            )
            if row["chronology"] == "measured" and row["start_ns"] is not None and row["end_ns"] is not None
            else None
        )
    ledger.update(
        {
            "format": "polylogue.work-ledger.v2",
            "spans": spans,
            "clock_origin_monotonic_ns": origin,
            "categories": dict(sorted(categories.items())),
            "writer_duration_observations": writer,
            "input_loss": input_loss,
            "rejected_event_records": rejected,
            "known_capture_gap": known_loss,
            "qualified_body_thread_cpu_union_ns": None
            if known_loss
            else ledger["work_observations"]["body_thread_cpu_union_ns"],
            "runtime_evidence": {"status": "unavailable", "reason": "no independently bound runtime receipt"},
            "allocation_provider": {
                "status": "unavailable",
                "provider": None,
                "coverage_scope": None,
                "reason": "event capture contains no allocation-provider receipt",
            },
            "limitations": [
                "Observed same-thread exclusive residuals are parent-local, never an additive global CPU total.",
                "Native PID/TID-only CPU union does not identify thread lifetime reuse; body union uses producer generations.",
                "Async elapsed overlap is not CPU; missing or wholly absent producers and capture boundaries remain unknown.",
                "Writer hold/wait durations have no reconstructed start/end and are not placed on the causal timeline.",
                "Readiness and retry event counts do not certify current health or avoided work.",
            ],
        }
    )
    return {"traceEvents": traces, "displayTimeUnit": "ms", "metadata": {"origin_monotonic_ns": origin}}, ledger


def export_work_ledger(
    input_path: str | Path, output_directory: str | Path, *, capture_metadata: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Export once; a complete manifest is the last, exclusive publication.

    Raw capture remains with its owner. A failed export can leave an incomplete
    directory, which cannot be mistaken for a qualified manifest or overwritten.
    """
    source, destination = Path(input_path).resolve(), Path(output_directory).resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (
        destination == checkout
        or checkout in destination.parents
        or any(
            (ancestor / ".git").is_file() or (ancestor / ".git" / "HEAD").is_file()
            for ancestor in (destination, *destination.parents)
        )
    ):
        raise ValueError("ledger output must be outside the checkout")
    if destination == source.parent or destination in source.parents:
        raise ValueError("ledger output must not contain its raw input")
    metadata = dict(capture_metadata or {})
    input_digest = hashlib.sha256()

    def read_input() -> Iterable[Mapping[str, Any]]:
        # Unsanitized records are not accumulated. The causal reducer still
        # retains observed metadata; this is not a constant-memory trace claim.
        with source.open("rb") as handle:
            for line in handle:
                input_digest.update(line)
                try:
                    value = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    metadata["malformed_records"] = int(metadata.get("malformed_records", 0)) + 1
                    continue
                if not isinstance(value, dict):
                    metadata["non_object_records"] = int(metadata.get("non_object_records", 0)) + 1
                else:
                    yield value

    trace, ledger = reduce_event_ledger(read_input(), capture_metadata=metadata)
    input_hash = input_digest.hexdigest()
    with source.open("rb") as handle:
        final_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if final_hash != input_hash:
        raise ValueError("raw input changed during reduction")
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    artifacts: dict[str, str] = {}
    for name, value in (("trace.json", trace), ("ledger.json", ledger)):
        content = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        with (destination / name).open("xb") as handle:
            handle.write(content)
        artifacts[name] = _digest(content)
    manifest = {
        "format": "polylogue.work-ledger-export.v1",
        "state": "complete",
        "input_sha256": input_hash,
        "reducer_sha256": _digest(Path(__file__).read_bytes()),
        "causal_reducer_sha256": _digest(Path(__file__).with_name("trace_ledger.py").read_bytes()),
        "clock": "host_monotonic_ns",
        "clock_origin_monotonic_ns": ledger["clock_origin_monotonic_ns"],
        "artifacts": artifacts,
        "known_capture_gap": ledger["known_capture_gap"],
        "runtime_evidence": ledger["runtime_evidence"],
        "allocation_provider": ledger["allocation_provider"],
        "categories": ledger["categories"],
        "coverage": ledger["coverage"],
    }
    temporary_manifest = destination / ".manifest.pending"
    with temporary_manifest.open("x") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    # Hard-link publication is atomic and refuses any existing final name.
    os.link(temporary_manifest, destination / "manifest.json")
    temporary_manifest.unlink()
    return manifest

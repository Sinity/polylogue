"""Read-only execution-packet projection for the reindex Beads campaign."""

import argparse
import json
import subprocess
import sys
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import Any

ROOT_ID = "polylogue-reindex-2026"
CAMPAIGN_ID = "reindex-2026"
CORE = "execution_wave execution_lane lane_packet lane_order affected_paths conflict_keys write_scope verification_commands model_policy live_data_access decision_closure necessity_class judgment_class tdd_mode tdd_packet packet_intent integration_intent".split()  # noqa: SIM905
LAUNCH = "packet_execution_contract effort expected_duration_evidence deadline_policy dispatch_readiness".split()  # noqa: SIM905
WAVES = {"reindex-prep-a": 1, "reindex-prep-b": 2, "reindex-prep-c": 3, "reindex-window": 4}


def _metadata(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and isinstance(decoded := json.loads(value or "{}"), Mapping):
        return dict(decoded)
    raise ValueError("Bead metadata must be an object")


def _record(value: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(value)
    labels, dependencies = record.get("labels") or (), record.get("dependencies") or ()
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        raise ValueError("Bead labels must be an array")
    if isinstance(dependencies, (str, bytes)) or not isinstance(dependencies, Sequence):
        raise ValueError("Bead dependencies must be an array")
    record["id"] = str(record.get("id", ""))
    record["labels"] = tuple(map(str, labels))
    record["metadata"] = _metadata(record.get("metadata"))
    record["dependencies"] = tuple(dep for dep in dependencies if isinstance(dep, Mapping))
    return record


class BdExportReader:
    def __init__(self, executable: str = "bd") -> None:
        self.executable = executable

    def read(self) -> tuple[dict[str, Any], ...]:
        command = [self.executable, "--readonly"]
        command += ["export", "--all"]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return tuple(_record(json.loads(line)) for line in result.stdout.splitlines() if line.strip())


def _value(bead: Mapping[str, Any], name: str) -> object:
    return bead["metadata"].get(name)


def _present(value: object) -> bool:
    return bool(value.strip()) if isinstance(value, str) else bool(value)


def _label(bead: Mapping[str, Any], prefix: str) -> str | None:
    return next((value.removeprefix(prefix) for value in bead["labels"] if value.startswith(prefix)), None)


def _campaign(bead: Mapping[str, Any]) -> bool:
    return _value(bead, "campaign_id") == CAMPAIGN_ID or _label(bead, "campaign:") == CAMPAIGN_ID


def _deps(bead: Mapping[str, Any], kind: str | None = None) -> tuple[str, ...]:
    return tuple(
        str(dep.get("depends_on_id", "")) for dep in bead["dependencies"] if kind is None or dep.get("type") == kind
    )


def _walk(beads: Mapping[str, Mapping[str, Any]], root: str, blocks: bool) -> frozenset[str]:
    seen, queue = {root}, deque([root])
    while queue:
        for target in _deps(beads[queue.popleft()], "blocks" if blocks else None):
            if target in beads and target not in seen:
                seen.add(target)
                queue.append(target)
    return frozenset(seen)


def _path(graph: Mapping[str, tuple[str, ...]], start: str, target: str) -> bool:
    seen, queue = {start}, deque([start])
    while queue:
        current = queue.popleft()
        if current == target:
            return True
        for next_id in graph[current]:
            if next_id not in seen:
                seen.add(next_id)
                queue.append(next_id)
    return False


def _keys(value: object) -> frozenset[str]:
    values = value.split(";") if isinstance(value, str) else value if isinstance(value, Sequence) else ()
    return frozenset(str(item).strip() for item in values if str(item).strip())


def _serialized(bead: Mapping[str, Any]) -> bool:
    return str(_value(bead, "lane_mode") or "").startswith("serialized-")


def _operator_writer(bead: Mapping[str, Any]) -> bool:
    access = str(_value(bead, "live_data_access") or "").lower()
    return "explicit-operator-authorized" in access or access in {
        "active-authorized",
        "authorized-inactive-generation-writer",
    }


def _candidate_writer(bead: Mapping[str, Any]) -> bool:
    access = str(_value(bead, "live_data_access") or "").lower()
    return "candidate" in access and "write" in access


def validate(reader: Any, *, root_id: str = ROOT_ID) -> dict[str, Any]:
    beads = {bead["id"]: bead for bead in reader.read()}
    if root_id not in beads:
        raise ValueError(f"missing campaign root {root_id}")
    closure, mixed = _walk(beads, root_id, True), _walk(beads, root_id, False)
    labelled = frozenset(bead_id for bead_id, bead in beads.items() if _campaign(bead))
    selected = frozenset(
        bead_id
        for bead_id in closure & labelled
        if (_value(beads[bead_id], "execution_shape") or _label(beads[bead_id], "execution-shape:")) in {"gate", "leaf"}
    )
    errors: list[str] = []
    warnings = []
    open_closure = frozenset(bead_id for bead_id in closure if beads[bead_id].get("status") != "closed")
    open_without_campaign = open_closure - labelled
    if open_without_campaign:
        errors.append(
            "open blocks-closure records have no campaign carrier: " + ", ".join(sorted(open_without_campaign))
        )
    closed_without_campaign = (closure - open_closure) - labelled
    if closed_without_campaign:
        warnings.append(f"{len(closed_without_campaign)} closed blocks-closure records have no campaign carrier")
    if labelled - closure:
        warnings.append(f"{len(labelled - closure)} campaign-labelled records are outside the blocks closure")
    open_without_shape = frozenset(
        bead_id
        for bead_id in open_closure & labelled
        if (_value(beads[bead_id], "execution_shape") or _label(beads[bead_id], "execution-shape:"))
        not in {"gate", "leaf"}
    )
    if open_without_shape:
        errors.append(
            "open campaign closure records have no valid execution shape: " + ", ".join(sorted(open_without_shape))
        )
    leaves: list[Mapping[str, Any]] = []
    for bead_id in sorted(selected):
        bead = beads[bead_id]
        shape = _value(bead, "execution_shape") or _label(bead, "execution-shape:")
        if shape == "gate":
            for field in ("lane_packet", "lane_order", *LAUNCH, "worker_model_class", "worker_capability"):
                if _present(_value(bead, field)):
                    errors.append(f"{bead_id}: gate carries {field}")
        elif bead.get("status") != "closed":
            missing = [field for field in CORE if not _present(_value(bead, field))]
            if not any(_present(_value(bead, field)) for field in ("worker_model_class", "worker_capability")):
                missing.append("worker capability")
            if not any(_present(_value(bead, field)) for field in ("review_model_class", "review_capability")):
                missing.append("reviewer capability")
            if missing:
                errors.append(f"{bead_id}: missing leaf carrier(s): {', '.join(missing)}")
            if _present(_value(bead, "model_policy")) and "provider-neutral" not in str(_value(bead, "model_policy")):
                errors.append(f"{bead_id}: model policy is not provider-neutral")
            leaves.append(bead)

    assignments: dict[str, tuple[str, str, str, int]] = {}
    for bead in leaves:
        wave, lane, packet, order_text = (str(_value(bead, field) or "").strip() for field in CORE[:4])
        try:
            assignment = (wave, lane, packet, int(order_text))
        except ValueError:
            errors.append(f"{bead['id']}: lane order is not numeric")
            continue
        if not all((wave, lane, packet)) or assignment[3] < 1:
            errors.append(f"{bead['id']}: invalid packet assignment")
            continue
        assignments[bead["id"]] = assignment
        if _operator_writer(bead) and not _serialized(bead):
            errors.append(f"{bead['id']}: operator-authorized writer is not serialized")
        if _candidate_writer(bead) and not _serialized(bead):
            errors.append(f"{bead['id']}: candidate writer is not serialized")

    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for bead_id, assignment in assignments.items():
        groups[assignment[:3]].append(bead_id)
    graph = {bead_id: _deps(bead, "blocks") for bead_id, bead in beads.items()}
    packets = []
    for group, members in sorted(groups.items()):
        members.sort(key=lambda bead_id: (assignments[bead_id][3], bead_id))
        leader, reasons = members[0], []
        if (
            not any(_present(_value(beads[bead_id], "packet_size_exception")) for bead_id in members)
            and not 3 <= len(members) <= 5
        ):
            errors.append(f"{'/'.join(group)}: ordinary packet has {len(members)} leaves")
        if len({assignments[bead_id][3] for bead_id in members}) != len(members):
            errors.append(f"{'/'.join(group)}: duplicate packet order")
        for bead_id in members:
            for target in graph[bead_id]:
                if target in assignments:
                    current, predecessor = assignments[bead_id], assignments[target]
                    if current[:3] == predecessor[:3] and current[3] <= predecessor[3]:
                        errors.append(f"{bead_id}: internal blocker is not earlier")
                    if WAVES.get(predecessor[0], 0) > WAVES.get(current[0], 0):
                        errors.append(f"{bead_id}: earlier wave blocks on later wave")
        for field in LAUNCH:
            if not _present(_value(beads[leader], field)):
                reasons.append(f"missing {field}")
        if _value(beads[leader], "dispatch_readiness") != "ready":
            reasons.append(f"dispatch readiness is {_value(beads[leader], 'dispatch_readiness') or 'missing'}")
        if (
            str(_value(beads[leader], "expected_duration_evidence") or "").lower().startswith("pending")
            or _value(beads[leader], "effort") == "calibration-pending"
        ):
            reasons.append("calibration pending")
        if _value(beads[leader], "prerequisite_state") == "unmet":
            reasons.append("prerequisite unmet")
        for bead_id in members[1:]:
            for field in LAUNCH:
                if _present(_value(beads[bead_id], field)):
                    errors.append(f"{bead_id}: non-leader carries {field}")
        packets.append(
            {
                "wave": group[0],
                "lane": group[1],
                "packet": group[2],
                "member_ids": members,
                "leader_id": leader,
                "ready": not reasons,
                "non_ready_reasons": list(dict.fromkeys(reasons)),
            }
        )

    for left_id, left in assignments.items():
        for right_id, right in assignments.items():
            if left_id >= right_id or left[0] != right[0] or left[1] == right[1]:
                continue
            overlap = _keys(_value(beads[left_id], "conflict_keys")) & _keys(_value(beads[right_id], "conflict_keys"))
            if overlap and not (
                _path(graph, left_id, right_id)
                or _path(graph, right_id, left_id)
                or _serialized(beads[left_id])
                or _serialized(beads[right_id])
            ):
                errors.append(f"{left_id}/{right_id}: exact conflict-key overlap is not serialized")
    errors = list(dict.fromkeys(errors))
    counts = {
        "blocks_closure": len(closure),
        "mixed_relation_expansion": len(mixed),
        "campaign_labelled": len(labelled),
        "open_leaves": len(leaves),
        "open_gates": sum(
            _value(beads[bead_id], "execution_shape") == "gate" and beads[bead_id].get("status") != "closed"
            for bead_id in closure
        ),
        "packets": len(packets),
        "lanes": len({group[:2] for group in groups}),
        "structural_errors": len(errors),
        "non_ready_packets": sum(not packet["ready"] for packet in packets),
        "warnings": len(warnings),
    }
    return {
        "read_only": True,
        "ok": not errors,
        "counts": counts,
        "blocks_only_closure": sorted(closure),
        "mixed_relation_expansion": sorted(mixed),
        "differences": {
            "mixed_only_ids": sorted(mixed - closure),
            "campaign_only_ids": sorted(labelled - closure),
            "noncampaign_blocks_ids": sorted(closure - labelled),
        },
        "packets": packets,
        "structural_errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None, *, reader: Any = None, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-id", default=ROOT_ID)
    parser.add_argument("--json", action="store_true")
    args, output = parser.parse_args(argv), stdout or sys.stdout
    try:
        report = validate(reader or BdExportReader(), root_id=args.root_id)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"reindex-packets: unable to read external Beads: {exc}", file=output)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True), file=output)
    else:
        counts = report["counts"]
        print(
            f"blocks closure: {counts['blocks_closure']}; mixed expansion: {counts['mixed_relation_expansion']}; packets: {counts['packets']}",
            file=output,
        )
        print(
            f"structural errors: {counts['structural_errors']}; non-ready packets: {counts['non_ready_packets']}; warnings: {counts['warnings']}",
            file=output,
        )
        for packet in report["packets"]:
            if not packet["ready"]:
                print(
                    f"NOT READY {packet['wave']}/{packet['lane']}/{packet['packet']}: {'; '.join(packet['non_ready_reasons'])}",
                    file=output,
                )
        for error in report["structural_errors"]:
            print(f"ERROR: {error}", file=output)
        for warning in report["warnings"]:
            print(f"WARNING: {warning}", file=output)
        print("read-only: no Beads or campaign state was written", file=output)
    return 0 if report["ok"] else 1

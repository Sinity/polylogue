"""Validate the current reindex execution projection from external Beads.

The command is deliberately a read-only projection. Beads remains the source
of truth for issues and dependencies, and this module keeps no campaign state.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

ROOT_ID = "polylogue-reindex-2026"
CAMPAIGN_ID = "reindex-2026"
_OPEN_STATUSES = frozenset({"open", "in_progress", "blocked", "deferred", "pinned", "hooked"})
_PACKET_FIELDS = ("execution_wave", "execution_lane", "lane_packet", "lane_order")
_LEADER_FIELDS = (
    "packet_execution_contract",
    "effort",
    "expected_duration_evidence",
    "deadline_policy",
    "dispatch_readiness",
)
_STRUCTURED_LEAF_FIELDS = (
    "model_policy",
    "decision_closure",
    "necessity_class",
    "tdd_mode",
    "anti_vacuity",
    "existing_test_disposition",
)
_DEPENDENCY_TYPES = frozenset(
    {
        "blocks",
        "parent-child",
        "conditional-blocks",
        "waits-for",
        "related",
        "discovered-from",
        "replies-to",
        "relates-to",
        "duplicates",
        "supersedes",
        "authored-by",
        "assigned-to",
        "approved-by",
        "attests",
        "tracks",
        "until",
        "caused-by",
        "validates",
        "delegated-from",
    }
)


@dataclass(frozen=True, slots=True)
class BeadDependency:
    issue_id: str
    depends_on_id: str
    type: str


@dataclass(frozen=True, slots=True)
class Bead:
    id: str
    title: str
    description: str
    design: str
    acceptance_criteria: str
    notes: str
    status: str
    issue_type: str
    owner: str | None
    labels: tuple[str, ...]
    metadata: Mapping[str, Any]
    dependencies: tuple[BeadDependency, ...]


class PacketReader(Protocol):
    def read(self) -> tuple[Bead, ...]: ...


def _as_metadata(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        if not value.strip():
            return {}
        decoded = json.loads(value)
        if not isinstance(decoded, Mapping):
            raise ValueError("Bead metadata must decode to an object")
        return dict(decoded)
    raise ValueError(f"Bead metadata must be an object, got {type(value).__name__}")


def _as_dependency(value: object) -> BeadDependency:
    if not isinstance(value, Mapping):
        raise ValueError("Bead dependency must be an object")
    return BeadDependency(
        issue_id=str(value.get("issue_id", "")),
        depends_on_id=str(value.get("depends_on_id", "")),
        type=str(value.get("type", "")),
    )


def bead_from_export(value: Mapping[str, Any]) -> Bead:
    """Convert one ``bd export`` record into the verifier's typed carrier."""

    raw_labels = value.get("labels", ())
    if not isinstance(raw_labels, Sequence) or isinstance(raw_labels, (str, bytes, bytearray)):
        raise ValueError(f"Bead {value.get('id', '<unknown>')} labels must be an array")
    raw_dependencies = value.get("dependencies", ())
    if not isinstance(raw_dependencies, Sequence) or isinstance(raw_dependencies, (str, bytes, bytearray)):
        raise ValueError(f"Bead {value.get('id', '<unknown>')} dependencies must be an array")
    return Bead(
        id=str(value.get("id", "")),
        title=str(value.get("title", "")),
        description=str(value.get("description", "")),
        design=str(value.get("design", "")),
        acceptance_criteria=str(value.get("acceptance_criteria", "")),
        notes=str(value.get("notes", "")),
        status=str(value.get("status", "")),
        issue_type=str(value.get("issue_type", "")),
        owner=(str(value["owner"]) if value.get("owner") is not None else None),
        labels=tuple(str(label) for label in raw_labels),
        metadata=_as_metadata(value.get("metadata")),
        dependencies=tuple(_as_dependency(dep) for dep in raw_dependencies),
    )


@dataclass(frozen=True, slots=True)
class BdExportReader:
    """Read the configured Beads export without opening a write path."""

    directory: Path | None = None
    executable: str = "bd"

    def read(self) -> tuple[Bead, ...]:
        command = [self.executable, "--readonly"]
        if self.directory is not None:
            command.extend(("--directory", str(self.directory)))
        command.extend(("export", "--all"))
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        records: list[Bead] = []
        for line_number, line in enumerate(completed.stdout.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError("record is not an object")
                records.append(bead_from_export(value))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid bd export record at line {line_number}: {exc}") from exc
        return tuple(records)


@dataclass(frozen=True, slots=True)
class PacketReport:
    wave: str
    lane: str
    packet: str
    member_ids: tuple[str, ...]
    leader_id: str | None
    dispatch_readiness: str | None
    launch_ready: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "wave": self.wave,
            "lane": self.lane,
            "packet": self.packet,
            "member_ids": list(self.member_ids),
            "leader_id": self.leader_id,
            "dispatch_readiness": self.dispatch_readiness,
            "launch_ready": self.launch_ready,
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    blocks_only_ids: frozenset[str]
    mixed_relation_ids: frozenset[str]
    counts: Mapping[str, object]
    differences: Mapping[str, list[str]]
    packets: tuple[PacketReport, ...]
    errors: tuple[str, ...]
    review_obligations: tuple[str, ...]
    calibration_findings: tuple[str, ...] = ()
    read_only: bool = True

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def mixed_only_ids(self) -> frozenset[str]:
        return self.mixed_relation_ids - self.blocks_only_ids

    def packet(self, wave: str, lane: str, packet: str) -> PacketReport:
        for value in self.packets:
            if (value.wave, value.lane, value.packet) == (wave, lane, packet):
                return value
        raise KeyError((wave, lane, packet))

    def to_dict(self) -> dict[str, object]:
        return {
            "read_only": self.read_only,
            "ok": self.ok,
            "counts": dict(self.counts),
            "blocks_only_closure": sorted(self.blocks_only_ids),
            "mixed_relation_expansion": sorted(self.mixed_relation_ids),
            "differences": {key: list(value) for key, value in sorted(self.differences.items())},
            "packets": [packet.to_dict() for packet in self.packets],
            "errors": list(self.errors),
            "review_obligations": list(self.review_obligations),
            "calibration_findings": list(self.calibration_findings),
        }


def _is_open(bead: Bead) -> bool:
    return bead.status in _OPEN_STATUSES


def _label_value(bead: Bead, prefix: str) -> str | None:
    for label in bead.labels:
        if label.startswith(prefix):
            return label[len(prefix) :]
    return None


def _value(bead: Bead, key: str) -> object:
    return bead.metadata.get(key)


def _scalar(value: object) -> str | None:
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        text = str(value).strip()
        return text or None
    return None


def _typed_shape(bead: Bead) -> str | None:
    value = _scalar(_value(bead, "execution_shape"))
    return value or _label_value(bead, "execution-shape:")


def _adjacency(beads: Mapping[str, Bead], *, kinds: frozenset[str]) -> dict[str, tuple[str, ...]]:
    result: dict[str, list[str]] = defaultdict(list)
    for bead in beads.values():
        for dependency in bead.dependencies:
            if dependency.type in kinds and dependency.depends_on_id in beads:
                result[bead.id].append(dependency.depends_on_id)
    return {key: tuple(sorted(values)) for key, values in result.items()}


def _walk(adjacency: Mapping[str, Sequence[str]], root_id: str) -> frozenset[str]:
    seen: set[str] = set()
    pending = deque([root_id])
    while pending:
        current = pending.popleft()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(adjacency.get(current, ()))
    return frozenset(seen)


def _normalise_resources(value: object) -> frozenset[str]:
    if value is None:
        return frozenset()
    values = value if isinstance(value, (list, tuple, set, frozenset)) else re.split(r"[,;\n]+", str(value))
    result: set[str] = set()
    for item in values:
        text = str(item).strip().lower()
        if not text:
            continue
        text = re.sub(r"[^a-z0-9]+", "/", text).strip("/")
        if text:
            parts = text.split("/")
            result.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))
    return frozenset(result)


def _intersects(left: frozenset[str], right: frozenset[str]) -> bool:
    return bool(left & right)


def _has_path(adjacency: Mapping[str, Sequence[str]], source: str, target: str) -> bool:
    return target in _walk(adjacency, source) or source in _walk(adjacency, target)


def _wave_number(value: str) -> int | None:
    match = re.search(r"(?:wave|w)[^0-9]*(\d+)", value.lower())
    return int(match.group(1)) if match else None


def _structured(value: object, required: Sequence[str]) -> bool:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return False
    return isinstance(value, Mapping) and all(key in value for key in required)


def _mapping(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return None
        return decoded if isinstance(decoded, Mapping) else None
    return None


def _valid_exception(value: object) -> bool:
    value = _mapping(value)
    return (
        isinstance(value, Mapping)
        and bool(_scalar(value.get("reason")))
        and bool(_scalar(value.get("evidence_ref")) or _scalar(value.get("evidence")))
    )


def _serialised(bead: Bead) -> bool:
    for key in ("serialized_window_contract", "serialization_contract"):
        value = _value(bead, key)
        if isinstance(value, Mapping):
            return bool(value.get("wave") or value.get("reason") or value.get("evidence_ref"))
        if value not in (None, "", False):
            return True
    return False


def _assignment(bead: Bead, errors: list[str]) -> dict[str, str] | None:
    assignment = _value(bead, "execution_assignment")
    values: dict[str, str] = {}
    if assignment is not None:
        if not isinstance(assignment, Mapping):
            errors.append(f"{bead.id}: incompatible duplicate assignment carrier")
            return None
        for field_name in _PACKET_FIELDS:
            if field_name in assignment:
                scalar = _scalar(assignment[field_name])
                if scalar is None:
                    errors.append(f"{bead.id}: incompatible duplicate assignment carrier")
                    return None
                values[field_name] = scalar
    for field_name in _PACKET_FIELDS:
        scalar = _scalar(_value(bead, field_name))
        if scalar is None:
            if field_name in values:
                continue
            errors.append(f"{bead.id}: missing leaf assignment ({field_name})")
            return None
        if field_name in values and values[field_name] != scalar:
            errors.append(f"{bead.id}: incompatible duplicate assignment carrier ({field_name})")
            return None
        values[field_name] = scalar
    return values


def validate(reader: PacketReader, *, root_id: str = ROOT_ID) -> ValidationReport:
    beads = {bead.id: bead for bead in reader.read()}
    errors: list[str] = []
    reviews: list[str] = []
    calibration_findings: list[str] = []
    ledger_paths: dict[str, str] = {}
    if root_id not in beads:
        return ValidationReport(
            frozenset(),
            frozenset(),
            {"closure_ids": [], "open_leaves": 0, "open_gates": 0, "packets": 0},
            {"mixed_only_ids": []},
            (),
            (f"root bead not found: {root_id}",),
            (),
        )

    blocks = _adjacency(beads, kinds=frozenset({"blocks"}))
    mixed = _adjacency(beads, kinds=_DEPENDENCY_TYPES)
    blocks_ids = _walk(blocks, root_id)
    mixed_ids = _walk(mixed, root_id)
    campaign_ids = frozenset(bead.id for bead in beads.values() if f"campaign:{CAMPAIGN_ID}" in bead.labels)
    open_blocks = frozenset(bead_id for bead_id in blocks_ids if _is_open(beads[bead_id]))
    open_campaign = frozenset(bead_id for bead_id in campaign_ids if _is_open(beads[bead_id]))

    for bead_id in sorted(open_campaign - blocks_ids):
        bead = beads[bead_id]
        if _typed_shape(bead) in {"leaf", "gate"} or bead.metadata.get("campaign_role") in {"closure", "closure-gate"}:
            errors.append(f"{bead.id}: campaign leaf is not blocks-reachable")

    leaf_ids: list[str] = []
    assignments: dict[str, dict[str, str]] = {}
    for bead_id in sorted(open_blocks):
        bead = beads[bead_id]
        shape = _typed_shape(bead)
        if shape not in {"leaf", "gate"}:
            errors.append(f"{bead.id}: unshaped closure node")
            continue
        if bead.metadata.get("campaign_id") != CAMPAIGN_ID:
            errors.append(f"{bead.id}: incompatible campaign identity")
        if _scalar(bead.metadata.get("campaign_role")) not in {"closure", "closure-gate", "milestone"}:
            errors.append(f"{bead.id}: missing structured campaign role")
        if shape == "gate":
            if any(key in bead.metadata for key in _PACKET_FIELDS):
                errors.append(f"{bead.id}: gate carries executable assignment")
            continue
        leaf_ids.append(bead.id)
        assignment = _assignment(bead, errors)
        if assignment is not None:
            assignments[bead.id] = assignment
        if bead.metadata.get("broad_scope") is True:
            errors.append(f"{bead.id}: broad-parent scope import")

        for field_name in _STRUCTURED_LEAF_FIELDS:
            if _value(bead, field_name) is None:
                errors.append(f"{bead.id}: missing structured field ({field_name})")
        decision = _value(bead, "decision_closure")
        if decision is not None and not _structured(
            decision, ("resolved_decisions", "remaining_decision_points", "escalation_owner")
        ):
            errors.append(f"{bead.id}: unshaped decision closure")
        verification = _scalar(_value(bead, "verification_commands"))
        if not verification:
            errors.append(f"{bead.id}: missing managed verification")
        elif verification.strip() == "devtools verify --quick" or verification.strip() == "devtools verify --quick;":
            errors.append(f"{bead.id}: quick-only broad verification")
        for context_field in ("architecture_context", "non_goals", "expected_outputs"):
            if _value(bead, context_field) is None:
                reviews.append(f"{bead.id}: review obligation for missing {context_field}")

        model_policy = _scalar(_value(bead, "model_policy"))
        if model_policy and ("gpt" in model_policy.lower() or "claude" in model_policy.lower()):
            errors.append(f"{bead.id}: provider-specific model policy")
        if not model_policy:
            errors.append(f"{bead.id}: missing capability policy")
        if not _scalar(_value(bead, "worker_model_class")):
            errors.append(f"{bead.id}: missing capability class")
        for model_field in ("model", "model_name", "model_id", "vendor_model"):
            if _value(bead, model_field) is not None:
                errors.append(f"{bead.id}: vendor-specific model name")
        if _scalar(_value(bead, "judgment_class")) == "mechanical" and (
            _scalar(_value(bead, "worker_model_class")) or ""
        ).startswith("strong"):
            errors.append(f"{bead.id}: unnecessarily strong worker class for mechanical packet")
        if _value(bead, "live_data_access") is None:
            errors.append(f"{bead.id}: missing live-data authority")
        if _scalar(_value(bead, "campaign_timing")) == "prep" and str(_value(bead, "live_data_access")).lower() in {
            "live",
            "live-mutation",
            "mutation",
            "allowed",
        }:
            errors.append(f"{bead.id}: window mutation in prep lane")
        if bool(_value(bead, "destructive")) and not str(_value(bead, "review_model_class") or "").startswith(
            "strong-independent"
        ):
            errors.append(f"{bead.id}: destructive packet lacks independent review")

        duration = _value(bead, "expected_duration_evidence")
        if duration is None or (
            isinstance(duration, str) and duration.lower() in {"unknown", "pending", "calibration-pending"}
        ):
            errors.append(f"{bead.id}: unknown duration evidence")
        if isinstance(duration, Mapping):
            seconds = duration.get("seconds")
            if isinstance(seconds, (int, float)) and seconds > 3600 and not _value(bead, "split_or_checkpoint"):
                errors.append(f"{bead.id}: duration exceeds 3600s without split or checkpoint")
            calibration_status = str(duration.get("calibration_status", "")).lower()
            if duration.get("calibration_ref") and calibration_status in {"stale", "unknown", "expired"}:
                errors.append(f"{bead.id}: stale calibration reference")
            observed = _value(bead, "observed_duration_seconds")
            if isinstance(observed, (int, float)) and isinstance(seconds, (int, float)) and observed > seconds * 1.5:
                calibration_findings.append(f"{bead.id}: observed duration overshot estimate by more than 50%")
        deadline = _value(bead, "deadline_policy")
        if (
            isinstance(deadline, Mapping)
            and deadline.get("kind")
            and not (_scalar(deadline.get("evidence_ref")) or _scalar(deadline.get("evidence")))
        ):
            errors.append(f"{bead.id}: deadline evidence is missing")
        if _scalar(_value(bead, "dispatch_readiness")) == "ready" and _scalar(_value(bead, "prerequisite_state")) in {
            "unmet",
            "unknown",
            "blocked",
        }:
            errors.append(f"{bead.id}: prerequisite state is unmet")
        ledger = _mapping(_value(bead, "deletion_ledger"))
        if _value(bead, "temporary_code") and not (
            _scalar(_value(bead, "deletion_owner")) or (ledger is not None and _scalar(ledger.get("sunset_owner")))
        ):
            errors.append(f"{bead.id}: temporary machinery without deletion owner")
        judgment = _scalar(_value(bead, "judgment_class"))
        requires_ledger = _value(bead, "deletion_ledger") == "required" or judgment in {
            "selected-architecture",
            "cli-overhaul",
            "public-surface-consolidation",
            "transition-safety-and-debloat",
            "post-campaign-hygiene",
        }
        if requires_ledger and ledger is None:
            errors.append(f"{bead.id}: missing deletion ledger")
        if ledger is not None:
            items = ledger.get("items")
            if not isinstance(items, list) or ledger.get("capability_census") is None:
                errors.append(f"{bead.id}: malformed deletion ledger")
            if ledger.get("predecessor_reachable") is True:
                errors.append(f"{bead.id}: deletion ledger predecessor remains reachable")
            for item in items if isinstance(items, list) else ():
                if not isinstance(item, Mapping):
                    errors.append(f"{bead.id}: malformed deletion ledger item")
                    continue
                path = _scalar(item.get("path"))
                if path:
                    previous = ledger_paths.get(path)
                    if previous is not None and previous != bead.id:
                        errors.append(f"{bead.id}: deletion ledger duplicates {path} from {previous}")
                    ledger_paths[path] = bead.id
                    if str(item.get("action", "")).lower() == "deleted" and any(
                        token in path.lower() for token in ("generated", ".cache", "cache/", "relocated")
                    ):
                        errors.append(f"{bead.id}: deletion ledger counts generated/cache relocation as gross deletion")
                if str(item.get("kind", "")).lower() in {"test", "tests"} and not _scalar(item.get("owning_law")):
                    errors.append(f"{bead.id}: deleted test lacks owning law")

        anti_vacuity = _mapping(_value(bead, "anti_vacuity"))
        if anti_vacuity is not None:
            strategy = str(anti_vacuity.get("strategy", "")).lower()
            if strategy in {"helper-only", "expected-snapshot", "forbidden-spelling-scan"}:
                errors.append(f"{bead.id}: helper-only or snapshot-only anti-vacuity")
            if _scalar(anti_vacuity.get("mutation")) is None or _scalar(anti_vacuity.get("red_test")) is None:
                errors.append(f"{bead.id}: anti-vacuity lacks a controlled red test")
        disposition = _mapping(_value(bead, "existing_test_disposition"))
        if (
            disposition is not None
            and disposition.get("status") == "historical-only"
            and not _scalar(disposition.get("regression_case"))
        ):
            errors.append(f"{bead.id}: historical-only test disposition lacks permanent regression case")
        if _value(bead, "test_deletion") is True and disposition is not None and not _scalar(disposition.get("law")):
            errors.append(f"{bead.id}: test deletion has no transferred law")

    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for bead_id, assignment in assignments.items():
        groups[(assignment["execution_wave"], assignment["execution_lane"], assignment["lane_packet"])].append(bead_id)

    packets: list[PacketReport] = []
    for coordinate, member_ids in sorted(groups.items()):
        wave, lane, packet = coordinate
        member_ids.sort(
            key=lambda bead_id: (
                int(assignments[bead_id]["lane_order"]) if assignments[bead_id]["lane_order"].isdigit() else 10**9,
                bead_id,
            )
        )
        if not all(assignments[bead_id]["lane_order"].isdigit() for bead_id in member_ids):
            for bead_id in member_ids:
                if not assignments[bead_id]["lane_order"].isdigit():
                    errors.append(f"{bead_id}: invalid lane order")
        orders = [assignments[bead_id]["lane_order"] for bead_id in member_ids]
        if len(orders) != len(set(orders)):
            errors.append(f"{wave}/{lane}/{packet}: duplicate packet order")
        exception = _value(beads[member_ids[0]], "packet_size_exception") if member_ids else None
        if 3 <= len(member_ids) <= 5 and exception is not None:
            errors.append(f"{wave}/{lane}/{packet}: unjustified packet size exception for ordinary packet")
        elif not 3 <= len(member_ids) <= 5 and not _valid_exception(exception):
            errors.append(f"{wave}/{lane}/{packet}: unjustified packet size")
        member_set = set(member_ids)
        for bead_id in member_ids:
            order = int(assignments[bead_id]["lane_order"]) if assignments[bead_id]["lane_order"].isdigit() else 10**9
            for dependency in beads[bead_id].dependencies:
                if dependency.type != "blocks" or dependency.depends_on_id not in member_set:
                    continue
                predecessor_value = assignments[dependency.depends_on_id]["lane_order"]
                if not predecessor_value.isdigit():
                    continue
                predecessor_order = int(predecessor_value)
                if predecessor_order >= order:
                    errors.append(f"{bead_id}: predecessor order is not lower than dependent")
        leader_id = member_ids[0] if member_ids and assignments[member_ids[0]]["lane_order"].isdigit() else None
        if leader_id is not None:
            minimum = assignments[leader_id]["lane_order"]
            if sum(assignment["lane_order"] == minimum for assignment in (assignments[id] for id in member_ids)) != 1:
                errors.append(f"{wave}/{lane}/{packet}: duplicate minimum packet order")
        leader = beads[leader_id] if leader_id else None
        if leader is not None:
            for field_name in _LEADER_FIELDS:
                if _value(leader, field_name) is None:
                    errors.append(f"{leader.id}: missing packet carrier ({field_name})")
            if _mapping(_value(leader, "packet_execution_contract")) is None:
                errors.append(f"{leader.id}: malformed packet execution contract")
            if not _scalar(_value(leader, "effort")):
                errors.append(f"{leader.id}: missing effort carrier")
            if _mapping(_value(leader, "expected_duration_evidence")) is None:
                errors.append(f"{leader.id}: malformed duration evidence")
            if _mapping(_value(leader, "deadline_policy")) is None:
                errors.append(f"{leader.id}: malformed deadline policy")
            readiness = _scalar(_value(leader, "dispatch_readiness"))
            if readiness == "calibration-pending":
                errors.append(f"{leader.id}: calibration-pending launch")
            launch_ready = readiness == "ready" and not any(error.startswith(f"{leader.id}:") for error in errors)
        else:
            readiness = None
            launch_ready = False
        packets.append(PacketReport(wave, lane, packet, tuple(member_ids), leader_id, readiness, launch_ready))

    # A dependency on a later wave is a topology error, regardless of packet order.
    for bead_id in leaf_ids:
        if bead_id not in assignments:
            continue
        current_wave = _wave_number(assignments[bead_id]["execution_wave"])
        for dependency in beads[bead_id].dependencies:
            if dependency.type != "blocks" or dependency.depends_on_id not in assignments:
                continue
            dependency_wave = _wave_number(assignments[dependency.depends_on_id]["execution_wave"])
            if current_wave is not None and dependency_wave is not None and dependency_wave > current_wave:
                errors.append(f"{bead_id}: earlier wave cannot block on later wave")

    # Concurrent lanes must serialize resource intersections through topology.
    for left_index, left_id in enumerate(leaf_ids):
        if left_id not in assignments:
            continue
        for right_id in leaf_ids[left_index + 1 :]:
            if right_id not in assignments:
                continue
            left, right = assignments[left_id], assignments[right_id]
            if left["execution_wave"] != right["execution_wave"] or left["execution_lane"] == right["execution_lane"]:
                continue
            left_resources = _normalise_resources(_value(beads[left_id], "conflict_keys")) | _normalise_resources(
                _value(beads[left_id], "write_scope")
            )
            right_resources = _normalise_resources(_value(beads[right_id], "conflict_keys")) | _normalise_resources(
                _value(beads[right_id], "write_scope")
            )
            if _intersects(left_resources, right_resources) and not (
                _has_path(blocks, left_id, right_id) or _serialised(beads[left_id]) or _serialised(beads[right_id])
            ):
                errors.append(f"{left_id}/{right_id}: concurrent conflict requires serialization")

    for bead_id in sorted(open_blocks):
        bead = beads[bead_id]
        if _value(bead, "conflict_keys") is None:
            reviews.append(f"{bead.id}: review obligation for missing conflict-key carrier")
        if _value(bead, "owner") is None and bead.owner is None:
            reviews.append(f"{bead.id}: review obligation for missing ownership signal")

    return ValidationReport(
        blocks_only_ids=blocks_ids,
        mixed_relation_ids=mixed_ids,
        counts={
            "closure_ids": sorted(blocks_ids),
            "blocks_only": len(blocks_ids),
            "open_closure": len(open_blocks),
            "open_leaves": len(leaf_ids),
            "open_gates": len(open_blocks) - len(leaf_ids),
            "campaign_labelled": len(campaign_ids),
            "campaign_labelled_related": len(campaign_ids),
            "campaign_labelled_open": len(open_campaign),
            "mixed_expansion": len(mixed_ids),
            "mixed_relation": len(mixed_ids),
            "mixed_only": len(mixed_ids - blocks_ids),
            "campaign_only": len(campaign_ids - blocks_ids),
            "noncampaign_transitive": len(open_blocks - campaign_ids),
            "packets": len(packets),
            "lanes": len({(packet.wave, packet.lane) for packet in packets}),
        },
        differences={
            "mixed_only_ids": sorted(mixed_ids - blocks_ids),
            "blocks_only_ids": sorted(blocks_ids - mixed_ids),
            "campaign_only_ids": sorted(campaign_ids - blocks_ids),
        },
        packets=tuple(packets),
        errors=tuple(dict.fromkeys(errors)),
        review_obligations=tuple(dict.fromkeys(reviews)),
        calibration_findings=tuple(dict.fromkeys(calibration_findings)),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-id", default=ROOT_ID)
    parser.add_argument(
        "--directory", type=Path, default=None, help="Directory whose external Beads authority is read."
    )
    parser.add_argument("--json", action="store_true", help="Emit the complete report as JSON.")
    return parser


def main(argv: list[str] | None = None, *, reader: PacketReader | None = None, stdout: TextIO | None = None) -> int:
    args = _parser().parse_args(argv)
    output = stdout or sys.stdout
    try:
        report = validate(reader or BdExportReader(args.directory), root_id=args.root_id)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"reindex-packets: unable to read external Beads: {exc}", file=output)
        return 2
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True), file=output)
    else:
        print(f"blocks-only closure: {report.counts['open_closure']} open records", file=output)
        print(f"campaign-labelled related population: {report.counts['campaign_labelled']}", file=output)
        print(f"mixed-relation expansion: {report.counts['mixed_expansion']} records", file=output)
        print(f"packets: {report.counts['packets']} across {report.counts['lanes']} lanes", file=output)
        for error in report.errors:
            print(f"ERROR: {error}", file=output)
        for review in report.review_obligations:
            print(f"REVIEW: {review}", file=output)
        print("read-only: no Beads or campaign state was written", file=output)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

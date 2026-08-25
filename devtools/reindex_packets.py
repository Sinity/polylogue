"""Read-only execution-packet projection for the reindex Beads campaign."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_ID = "polylogue-reindex-2026"
CAMPAIGN_ID = "reindex-2026"
CORE = "execution_wave execution_lane lane_packet lane_order affected_paths conflict_keys write_scope verification_commands model_policy live_data_access decision_closure necessity_class judgment_class tdd_mode tdd_packet packet_intent integration_intent".split()  # noqa: SIM905
LAUNCH = "packet_execution_contract deadline_policy readiness_contract".split()  # noqa: SIM905
WAVES = {"reindex-prep-a": 1, "reindex-prep-b": 2, "reindex-prep-c": 3, "reindex-window": 4}
READINESS_VERSION = "packet-readiness-v1"
OPERATION_PHASE_VERSION = "prep-operation-phase-v1"
PREP_OPERATION_ACCESS = frozenset(
    {"explicit-operator-authorized-source-apply", "explicit-operator-authorized-blob-apply"}
)
FINAL_CANDIDATE_ACCESS = frozenset({"candidate-only"})
EXECUTION_KINDS = frozenset({"implementation", "evidence", "authorized-prep-operation"})
PREDICATE_KINDS = frozenset(
    {
        "exact-head-review",
        "calibration-receipt",
        "rehearsal",
        "operator-authorization",
        "external-gate",
        "source-carrier-receipt",
        "terminal-campaign-proof",
    }
)
PREDICATE_STATES = frozenset({"accepted", "authorized", "pending", "rejected", "stale", "revoked", "expired"})
PREDICATE_FIELDS = {
    "exact-head-review": frozenset({"kind", "evidence_id", "state", "head"}),
    "calibration-receipt": frozenset({"kind", "evidence_id", "state"}),
    "rehearsal": frozenset({"kind", "evidence_id", "state", "plan_digest"}),
    "operator-authorization": frozenset({"kind", "evidence_id", "state", "plan_digest", "expires_at"}),
    "external-gate": frozenset({"kind", "evidence_id", "state"}),
    "source-carrier-receipt": frozenset({"kind", "evidence_id", "state"}),
    "terminal-campaign-proof": frozenset({"kind", "evidence_id", "state"}),
}
ACCEPTED_STATES = dict.fromkeys(PREDICATE_KINDS, "accepted")
ACCEPTED_STATES["operator-authorization"] = "authorized"
INTEGRATION_HEAD_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class TaskIdentity:
    bead_id: str
    revision: str


@dataclass(frozen=True)
class EvidencePredicate:
    kind: str
    evidence_id: str
    state: str
    head: str | None = None
    plan_digest: str | None = None
    expires_at: str | None = None


@dataclass(frozen=True)
class ReadinessContract:
    integration_head: str
    evaluated_at: str
    members: tuple[TaskIdentity, ...]
    predicates: tuple[EvidencePredicate, ...]


class BdExportReader:
    def __init__(self, executable: str = "bd") -> None:
        self.executable = executable

    def read(self) -> tuple[dict[str, Any], ...]:
        result = subprocess.run(
            [self.executable, "--readonly", "export", "--all"], check=True, capture_output=True, text=True
        )
        return tuple(_record(json.loads(line)) for line in result.stdout.splitlines() if line.strip())


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


def _authority_writer(bead: Mapping[str, Any]) -> bool:
    access = str(_value(bead, "live_data_access") or "").lower()
    return (
        access in {"active-authorized", "authorized-inactive-generation-writer"}
        or "explicit-operator-authorized" in access
        or access in FINAL_CANDIDATE_ACCESS
        or ("candidate" in access and "write" in access)
    )


def _serialization_errors(bead: Mapping[str, Any], wave: str) -> list[str]:
    if not _authority_writer(bead) or _serialized(bead):
        return []
    access = str(_value(bead, "live_data_access") or "").lower()
    if wave.startswith("reindex-prep-") and _value(bead, "execution_kind") == "authorized-prep-operation":
        return [f"{bead['id']}: authorized prep operation must be serialized"]
    if (
        access in {"active-authorized", "authorized-inactive-generation-writer"}
        or "explicit-operator-authorized" in access
    ):
        return [f"{bead['id']}: operator-authorized writer is not serialized"]
    if access in FINAL_CANDIDATE_ACCESS or ("candidate" in access and "write" in access):
        return [f"{bead['id']}: candidate writer is not serialized"]
    return []


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _integration_head_argument(value: str) -> str:
    if not INTEGRATION_HEAD_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError("integration head must be a 40-character lowercase commit SHA")
    return value


def _checkout_integration_head() -> str:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "-C", str(repository_root), "rev-parse", "--verify", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return _integration_head_argument(result.stdout.strip())


def _parse_predicate(value: object) -> EvidencePredicate:
    if not isinstance(value, Mapping):
        raise ValueError("predicate must be an object")
    kind = value.get("kind")
    if kind not in PREDICATE_KINDS:
        raise ValueError(f"unknown predicate kind {kind!r}")
    if set(value) != PREDICATE_FIELDS[kind]:
        raise ValueError(f"{kind}: fields must be {sorted(PREDICATE_FIELDS[kind])}")
    evidence_id, state = value.get("evidence_id"), value.get("state")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError(f"{kind}: missing evidence_id")
    if state not in PREDICATE_STATES:
        raise ValueError(f"{kind}: unknown predicate state {state!r}")
    for field in PREDICATE_FIELDS[kind] - {"kind", "evidence_id", "state"}:
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"{kind}: missing {field}")
    if kind == "operator-authorization" and _parse_datetime(value["expires_at"]) is None:
        raise ValueError("operator-authorization: expires_at must be timezone-aware ISO-8601")
    return EvidencePredicate(
        kind, evidence_id, state, value.get("head"), value.get("plan_digest"), value.get("expires_at")
    )


def _parse_readiness_contract(value: object) -> ReadinessContract:
    if not isinstance(value, Mapping):
        raise ValueError("readiness_contract must be an object")
    fields = {"version", "integration_head", "evaluated_at", "members", "predicates"}
    if set(value) != fields:
        raise ValueError(f"readiness_contract fields must be {sorted(fields)}")
    if value.get("version") != READINESS_VERSION:
        raise ValueError(f"readiness_contract version must be {READINESS_VERSION}")
    integration_head, evaluated_at = value.get("integration_head"), value.get("evaluated_at")
    if not isinstance(integration_head, str) or not integration_head:
        raise ValueError("readiness_contract missing integration_head")
    if not isinstance(evaluated_at, str) or _parse_datetime(evaluated_at) is None:
        raise ValueError("readiness_contract evaluated_at must be timezone-aware ISO-8601")
    members, predicates = value.get("members"), value.get("predicates")
    if isinstance(members, (str, bytes)) or not isinstance(members, Sequence) or not members:
        raise ValueError("readiness_contract members must be a non-empty array")
    if isinstance(predicates, (str, bytes)) or not isinstance(predicates, Sequence):
        raise ValueError("readiness_contract predicates must be an array")
    identities = []
    for member in members:
        if not isinstance(member, Mapping) or set(member) != {"id", "revision"}:
            raise ValueError("readiness_contract member must have id and revision")
        bead_id, revision = member.get("id"), member.get("revision")
        if not isinstance(bead_id, str) or not bead_id or not isinstance(revision, str) or not revision:
            raise ValueError("readiness_contract member has invalid identity")
        identities.append(TaskIdentity(bead_id, revision))
    if len({item.bead_id for item in identities}) != len(identities):
        raise ValueError("readiness_contract members are ambiguous")
    parsed_predicates = tuple(_parse_predicate(predicate) for predicate in predicates)
    if len({(item.kind, item.evidence_id) for item in parsed_predicates}) != len(parsed_predicates):
        raise ValueError("readiness_contract predicates are ambiguous")
    return ReadinessContract(integration_head, evaluated_at, tuple(identities), parsed_predicates)


def _execution_kind_errors(bead: Mapping[str, Any], wave: str) -> list[str]:
    if not wave.startswith("reindex-prep-"):
        return []
    bead_id, access, kind = bead["id"], str(_value(bead, "live_data_access") or ""), _value(bead, "execution_kind")
    errors = []
    if kind is not None and kind not in EXECUTION_KINDS:
        errors.append(f"{bead_id}: unknown execution_kind {kind!r}")
    if access in PREP_OPERATION_ACCESS and kind != "authorized-prep-operation":
        errors.append(f"{bead_id}: {wave}/{access} requires execution_kind authorized-prep-operation")
    if kind == "authorized-prep-operation" and access not in PREP_OPERATION_ACCESS:
        errors.append(f"{bead_id}: authorized prep operation requires operator-authorized live_data_access")
    return errors


def _operation_phase_errors(bead: Mapping[str, Any], contract: ReadinessContract | None) -> list[str]:
    if _value(bead, "execution_kind") != "authorized-prep-operation":
        return []
    bead_id, phase = bead["id"], _value(bead, "operation_phase_contract")
    if not isinstance(phase, Mapping):
        return [f"{bead_id}: authorized prep operation missing operation_phase_contract"]
    fields = {"version", "shape", "plan_id", "plan_digest", "rehearsal_id"}
    if set(phase) != fields or phase.get("version") != OPERATION_PHASE_VERSION:
        return [f"{bead_id}: authorized prep operation has invalid operation_phase_contract"]
    if phase.get("shape") not in {
        "plan-rehearse-review-authorize-apply-verify",
        "accepted-plan-rehearse-authorize-apply-verify",
    }:
        return [f"{bead_id}: authorized prep operation has invalid phase shape"]
    if not all(
        isinstance(phase.get(field), str) and phase[field] for field in ("plan_id", "plan_digest", "rehearsal_id")
    ):
        return [f"{bead_id}: authorized prep operation has incomplete operation phase identity"]
    initial, apply = _value(bead, "initial_job_authority"), _value(bead, "apply_authority")
    errors = []
    if (
        not isinstance(initial, Mapping)
        or set(initial) != {"mode", "authority_id"}
        or initial.get("mode") not in {"read-only-plan-rehearsal", "accepted-plan-rehearsal"}
        or not _present(initial.get("authority_id"))
    ):
        errors.append(f"{bead_id}: authorized prep operation has invalid initial_job_authority")
    if (
        not isinstance(apply, Mapping)
        or set(apply) != {"mode", "authorization_id", "plan_digest"}
        or apply.get("mode") != "explicit-operator-authorized-apply"
        or not _present(apply.get("authorization_id"))
    ):
        errors.append(f"{bead_id}: authorized prep operation has invalid apply_authority")
    elif apply.get("plan_digest") != phase["plan_digest"]:
        errors.append(f"{bead_id}: authorized prep operation apply authority plan digest does not match phase plan")
    if contract is None:
        return errors
    rehearsal = any(
        predicate.kind == "rehearsal"
        and predicate.evidence_id == phase["rehearsal_id"]
        and predicate.plan_digest == phase["plan_digest"]
        for predicate in contract.predicates
    )
    apply_authorization_id = apply.get("authorization_id") if isinstance(apply, Mapping) else None
    authorization = any(
        predicate.kind == "operator-authorization"
        and predicate.evidence_id == apply_authorization_id
        and predicate.plan_digest == phase["plan_digest"]
        for predicate in contract.predicates
    )
    if not rehearsal:
        errors.append(f"{bead_id}: authorized prep operation missing rehearsal predicate")
    if not authorization:
        errors.append(f"{bead_id}: authorized prep operation missing operator authorization predicate")
    return errors


def _failure(kind: str, reason: str, **details: str) -> dict[str, str]:
    return {"kind": kind, "reason": reason, **details}


def _predicate_status(predicate: EvidencePredicate, contract: ReadinessContract) -> str | None:
    expires_at = _parse_datetime(predicate.expires_at)
    evaluated_at = _parse_datetime(contract.evaluated_at)
    if (
        predicate.kind == "operator-authorization"
        and expires_at is not None
        and evaluated_at is not None
        and expires_at <= evaluated_at
    ):
        return "expired"
    if predicate.state != ACCEPTED_STATES[predicate.kind]:
        return predicate.state
    if predicate.kind == "exact-head-review" and predicate.head != contract.integration_head:
        return "stale"
    return None


def _projection(
    group: tuple[str, str, str],
    members: list[str],
    contract: ReadinessContract | None,
    beads: Mapping[str, Mapping[str, Any]],
    integration_head: str | None,
    external_blockers: list[str],
    structural_failures: Sequence[dict[str, str]],
) -> dict[str, Any]:
    unsatisfied: list[dict[str, str]] = []
    if contract is not None:
        expected_members = {bead_id: str(beads[bead_id].get("revision") or "") for bead_id in members}
        contract_members = {identity.bead_id: identity.revision for identity in contract.members}
        for bead_id in sorted(set(expected_members) | set(contract_members)):
            if expected_members.get(bead_id) != contract_members.get(bead_id):
                unsatisfied.append({"kind": "task-identity", "bead_id": bead_id, "reason": "stale"})
        if integration_head is not None and contract.integration_head != integration_head:
            unsatisfied.append({"kind": "integration-head", "reason": "stale"})
        for predicate in contract.predicates:
            if reason := _predicate_status(predicate, contract):
                unsatisfied.append({"kind": predicate.kind, "evidence_id": predicate.evidence_id, "reason": reason})
    unsatisfied.extend({"kind": "blocks", "bead_id": bead_id, "reason": "open"} for bead_id in external_blockers)
    launch_failures = [*structural_failures, *unsatisfied]
    leader = beads[members[0]]
    projection: dict[str, Any] = {
        "version": "packet-launch-projection-v1",
        "wave": group[0],
        "lane": group[1],
        "packet": group[2],
        "member_ids": list(members),
        "integration_head": contract.integration_head if contract else None,
        "launch_contract": {
            "packet_execution_contract": _value(leader, "packet_execution_contract"),
            "deadline_policy": _value(leader, "deadline_policy"),
            "model_policy": _value(leader, "model_policy"),
            "worker_capability": _value(leader, "worker_capability") or _value(leader, "worker_model_class"),
            "review_capability": _value(leader, "review_capability") or _value(leader, "review_model_class"),
        },
        "task_identities": [asdict(identity) for identity in contract.members] if contract else [],
        "satisfied_predicate_ids": [
            {"kind": predicate.kind, "evidence_id": predicate.evidence_id}
            for predicate in contract.predicates
            if _predicate_status(predicate, contract) is None
        ]
        if contract
        else [],
        "unsatisfied_predicates": unsatisfied,
        "launch_failures": launch_failures,
    }
    projection["ready"] = not launch_failures
    payload = json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    projection["context_digest"] = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    projection["projection_id"] = f"{projection['version']}:{projection['context_digest']}"
    return projection


def validate(reader: Any, *, root_id: str = ROOT_ID, integration_head: str | None = None) -> dict[str, Any]:
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
    packet_structural_failures: dict[str, list[dict[str, str]]] = defaultdict(list)
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
        for serialization_error in _serialization_errors(bead, wave):
            errors.append(serialization_error)
            packet_structural_failures[bead["id"]].append(_failure("serialization", serialization_error))
        for execution_error in _execution_kind_errors(bead, wave):
            errors.append(execution_error)
            packet_structural_failures[bead["id"]].append(_failure("execution-kind", execution_error))
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for bead_id, assignment in assignments.items():
        groups[assignment[:3]].append(bead_id)
    graph = {bead_id: _deps(bead, "blocks") for bead_id, bead in beads.items()}
    conflict_failures: dict[str, list[dict[str, str]]] = defaultdict(list)
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
                error = f"{left_id}/{right_id}: exact conflict-key overlap is not serialized"
                errors.append(error)
                for bead_id in (left_id, right_id):
                    conflict_failures[bead_id].append(_failure("conflict-serialization", error, bead_id=bead_id))
    packets = []
    for group, members in sorted(groups.items()):
        members.sort(key=lambda bead_id: (assignments[bead_id][3], bead_id))
        leader = members[0]
        launch_failures = [
            failure
            for bead_id in members
            for failure in (*packet_structural_failures[bead_id], *conflict_failures[bead_id])
        ]
        if (
            not any(_present(_value(beads[bead_id], "packet_size_exception")) for bead_id in members)
            and not 3 <= len(members) <= 5
        ):
            error = f"{'/'.join(group)}: ordinary packet has {len(members)} leaves"
            errors.append(error)
            launch_failures.append(_failure("packet-shape", error))
        if len({assignments[bead_id][3] for bead_id in members}) != len(members):
            error = f"{'/'.join(group)}: duplicate packet order"
            errors.append(error)
            launch_failures.append(_failure("packet-order", error))
        external_blockers = []
        for bead_id in members:
            for target in graph[bead_id]:
                if target not in members:
                    if target in beads and beads[target].get("status") != "closed":
                        external_blockers.append(target)
                    if target in assignments:
                        current, predecessor = assignments[bead_id], assignments[target]
                        if current[:2] == predecessor[:2] and int(current[2]) < int(predecessor[2]):
                            error = f"{bead_id}: packet blocker is in a later packet"
                            errors.append(error)
                            launch_failures.append(_failure("blocker-order", error, bead_id=bead_id))
                        if WAVES.get(predecessor[0], 0) > WAVES.get(current[0], 0):
                            error = f"{bead_id}: earlier wave blocks on later wave"
                            errors.append(error)
                            launch_failures.append(_failure("blocker-order", error, bead_id=bead_id))
                    continue
                current, predecessor = assignments[bead_id], assignments[target]
                if current[3] <= predecessor[3]:
                    error = f"{bead_id}: internal blocker is not earlier"
                    errors.append(error)
                    launch_failures.append(_failure("blocker-order", error, bead_id=bead_id))
        for field in LAUNCH:
            if not _present(_value(beads[leader], field)):
                reason = f"missing {field}"
                launch_failures.append(_failure("missing-launch-field", reason, field=field))
        contract = None
        if _present(_value(beads[leader], "readiness_contract")):
            try:
                contract = _parse_readiness_contract(_value(beads[leader], "readiness_contract"))
            except ValueError as exc:
                contract_error = f"{leader}: {exc}"
                errors.append(contract_error)
                launch_failures.append(_failure("readiness-contract", contract_error))
        if group[0].startswith("reindex-prep-"):
            for bead_id in members:
                for phase_error in _operation_phase_errors(beads[bead_id], contract):
                    errors.append(phase_error)
                    launch_failures.append(_failure("operation-phase", phase_error, bead_id=bead_id))
        for bead_id in members[1:]:
            for field in LAUNCH:
                if _present(_value(beads[bead_id], field)):
                    error = f"{bead_id}: non-leader carries {field}"
                    errors.append(error)
                    launch_failures.append(_failure("non-leader-launch-field", error, bead_id=bead_id, field=field))
        projection = _projection(
            group,
            members,
            contract,
            beads,
            integration_head,
            sorted(set(external_blockers)),
            launch_failures,
        )
        packets.append(
            {
                "wave": group[0],
                "lane": group[1],
                "packet": group[2],
                "member_ids": members,
                "leader_id": leader,
                "ready": projection["ready"],
                "non_ready_reasons": list(dict.fromkeys(item["reason"] for item in projection["launch_failures"])),
                "launch_projection": projection,
            }
        )
    errors = list(dict.fromkeys(errors))
    legacy_readiness_census = {
        field: {
            "count": len(ids := sorted(bead_id for bead_id, bead in beads.items() if _present(_value(bead, field)))),
            "record_ids": ids,
        }
        for field in ("dispatch_readiness", "program_dispatch_readiness")
    }
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
        "legacy_readiness_census": legacy_readiness_census,
        "structural_errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None, *, reader: Any = None, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-id", default=ROOT_ID)
    parser.add_argument("--integration-head", type=_integration_head_argument)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--enforce-readiness",
        "--enforce",
        dest="enforce_readiness",
        action="store_true",
        help="fail when structural errors or any packet is not launch-ready (the default)",
    )
    mode.add_argument(
        "--diagnostic",
        dest="diagnostic",
        action="store_true",
        help="print the read-only projection without failing on structural or readiness findings",
    )
    parser.add_argument("--json", action="store_true")
    args, output = parser.parse_args(argv), stdout or sys.stdout
    try:
        report = validate(
            reader or BdExportReader(),
            root_id=args.root_id,
            integration_head=args.integration_head or _checkout_integration_head(),
        )
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
    if args.diagnostic:
        return 0
    return 0 if report["ok"] and all(packet["ready"] for packet in report["packets"]) else 1

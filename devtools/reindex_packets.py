"""Read-only execution-packet projection for the reindex Beads campaign."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_ID = "polylogue-reindex-2026"
CAMPAIGN_ID = "reindex-2026"
CORE = "execution_wave execution_lane lane_packet lane_order ownership_resources verification_specs evidence_obligations live_data_access decision_stage necessity_class judgment_class tdd_mode tdd_packet packet_intent integration_intent".split()  # noqa: SIM905
LAUNCH = "packet_execution_contract deadline_policy".split()  # noqa: SIM905
LEGACY_METADATA_FIELDS = frozenset(
    {
        "affected_paths",
        "write_scope",
        "verification_commands",
        "worker_model_class",
        "review_model_class",
        "residual_decision_load",
        "dispatch_readiness",
        "program_dispatch_readiness",
    }
)
WORKER_CAPABILITIES = frozenset({"routine", "strong"})
REVIEW_CAPABILITIES = frozenset({"none", "standard-independent", "strong-independent"})
DECISION_STAGES = frozenset(
    {"decision-closed", "bounded-experiment", "evidence-adjudication", "authorized-live-judgment"}
)
RESOURCE_KINDS = frozenset(
    {"path-prefix", "generated-surface", "schema-slot", "fixture-authority", "database-tier", "runtime-resource"}
)
WAVES = {
    "reindex-prep-a": 1,
    "reindex-prep-b": 2,
    "reindex-prep-c": 3,
    "reindex-window": 4,
    "post-reindex": 5,
}
TASK_REVISION_PREFIX = "sha256:"
TASK_REVISION_BOOKKEEPING_FIELDS = frozenset(
    {"created_at", "updated_at", "created_by", "comment_count", "dependency_count", "dependent_count"}
)
OPERATION_PHASE_VERSION = "prep-operation-phase-v2"
FINAL_CANDIDATE_ACCESS = frozenset({"candidate-only"})
EXECUTION_KINDS = frozenset({"implementation", "evidence", "authorized-prep-operation"})
PHASES = frozenset({"initial", "apply"})
PREP_OPERATION_WAVES = frozenset(wave for wave in WAVES if wave.startswith("reindex-prep-"))
INITIAL_AUTHORITY_MODES = frozenset({"read-only-plan-rehearsal", "accepted-plan-rehearsal"})
APPLY_AUTHORITY_MODE = "explicit-operator-authorized-apply"
OPERATION_SHAPES = frozenset(
    {"plan-rehearse-review-authorize-apply-verify", "accepted-plan-rehearse-authorize-apply-verify"}
)
RUNTIME_BOOKKEEPING_METADATA_FIELDS = frozenset(
    {
        "active_job_id",
        "active_backend",
        "active_model",
        "review_state",
        "review_result",
        "cancelled_job",
        "cancelled_job_id",
        "correction_head",
        "readiness_contract",
    }
)
DIAGNOSTIC_EVIDENCE_FIELDS = frozenset(
    {"operation_id", "integration_head", "packet_context_digest", "plan_digest", "rehearsal", "authorization", "review"}
)
REHEARSAL_EVIDENCE_FIELDS = frozenset({"evidence_id", "state", "plan_digest"})
AUTHORIZATION_EVIDENCE_FIELDS = frozenset({"evidence_id", "state", "plan_digest", "expires_at"})
REVIEW_EVIDENCE_FIELDS = frozenset({"evidence_id", "state", "plan_digest"})
PROHIBITED_OPERATION_POLICY_FIELDS = frozenset(
    {"plan_id", "plan_digest", "rehearsal_id", "authorization_id", "job_id", "phase_state"}
)
INTEGRATION_HEAD_PATTERN = re.compile(r"^[0-9a-f]{40}$")
PREP_OPERATION_ACCESS_PATTERN = re.compile(r"^explicit-operator-authorized-(?:.*-)?apply$")
SUBPROCESS_TIMEOUT_SECONDS = 5


class ReindexPacketValidationError(ValueError):
    """Typed validation failure for an unlaunchable reindex packet."""


@dataclass(frozen=True)
class TaskIdentity:
    bead_id: str
    revision: str


@dataclass(frozen=True)
class PhaseEvidence:
    evidence_id: str
    state: str
    plan_digest: str
    expires_at: str | None = None


@dataclass(frozen=True)
class OperationEvidence:
    operation_id: str
    integration_head: str
    packet_context_digest: str
    plan_digest: str
    rehearsal: PhaseEvidence | None
    authorization: PhaseEvidence | None
    review: PhaseEvidence | None


class BdExportReader:
    def __init__(self, executable: str = "bd") -> None:
        self.executable = executable

    def read(self) -> tuple[dict[str, Any], ...]:
        result = subprocess.run(
            [self.executable, "--readonly", "export", "--all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
        return tuple(
            _record(record)
            for line in result.stdout.splitlines()
            if line.strip() and _is_task(record := json.loads(line))
        )


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
    metadata = _metadata(record.get("metadata"))
    if isinstance(metadata.get("live_data_access"), str):
        metadata["live_data_access"] = metadata["live_data_access"].strip().casefold()
    record["metadata"] = metadata
    record["dependencies"] = tuple(dep for dep in dependencies if isinstance(dep, Mapping))
    return record


def _is_task(value: object) -> bool:
    return isinstance(value, Mapping) and value.get("_type", "issue") == "issue" and bool(value.get("id"))


def _canonical_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


def _task_revision(bead: Mapping[str, Any]) -> str:
    """Return the stable task-policy token bound by launch projections.

    The export has no top-level revision. This digest covers task content and
    routing state, omits export timestamps and counters, and excludes only
    named operational bookkeeping. Unknown metadata remains semantic and
    therefore invalidates a prior task identity.
    """
    metadata = _metadata(bead.get("metadata"))
    semantic_metadata = {
        key: value for key, value in metadata.items() if key not in RUNTIME_BOOKKEEPING_METADATA_FIELDS
    }
    dependencies = tuple(
        {
            "depends_on_id": dependency.get("depends_on_id"),
            "type": dependency.get("type"),
            "metadata": _metadata(dependency.get("metadata")),
        }
        for dependency in bead.get("dependencies", ())
        if isinstance(dependency, Mapping)
    )
    payload = {
        key: value
        for key, value in bead.items()
        if key not in TASK_REVISION_BOOKKEEPING_FIELDS and key not in {"labels", "metadata", "dependencies"}
    }
    payload.update(
        labels=sorted(map(str, bead.get("labels", ()) or ())),
        metadata=semantic_metadata,
        dependencies=sorted(
            dependencies,
            key=lambda dependency: json.dumps(_canonical_value(dependency), sort_keys=True, separators=(",", ":")),
        ),
    )
    encoded = json.dumps(
        _canonical_value({"domain": "polylogue.reindex.task-revision.v1", "task": payload}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return f"{TASK_REVISION_PREFIX}{hashlib.sha256(encoded).hexdigest()}"


def _value(bead: Mapping[str, Any], name: str) -> object:
    return bead["metadata"].get(name)


def _effective_policy(beads: Mapping[str, Mapping[str, Any]], bead: Mapping[str, Any]) -> object:
    """Resolve the one campaign policy default, with an explicit leaf override."""
    policy = _value(bead, "model_policy")
    return policy if policy is not None else _value(beads[ROOT_ID], "model_policy")


def _typed_resources(value: object) -> tuple[tuple[str, str], ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    result: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"kind", "key"}:
            return None
        kind, key = item.get("kind"), item.get("key")
        if not isinstance(kind, str) or kind not in RESOURCE_KINDS or not isinstance(key, str) or not key.strip():
            return None
        normalized = key.strip().replace("\\", "/").removeprefix("./")
        if kind == "path-prefix" and (not normalized or normalized.startswith("/") or ".." in normalized.split("/")):
            return None
        result.append((kind, normalized.rstrip("/")))
    return tuple(sorted(set(result)))


def _valid_verification_specs(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    return all(
        isinstance(item, Mapping)
        and set(item) == {"argv"}
        and isinstance(item["argv"], Sequence)
        and not isinstance(item["argv"], (str, bytes))
        and all(isinstance(token, str) and token for token in item["argv"])
        for item in value
    )


def _valid_evidence_obligations(value: object) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    return all(
        isinstance(item, Mapping)
        and set(item) == {"kind", "producer", "binding"}
        and all(isinstance(item[key], str) and item[key].strip() for key in ("kind", "producer", "binding"))
        for item in value
    )


def _resources_overlap(left: set[tuple[str, str]], right: set[tuple[str, str]]) -> bool:
    for left_kind, left_key in left:
        for right_kind, right_key in right:
            if left_kind != right_kind:
                continue
            if left_kind == "path-prefix":
                if (
                    left_key == right_key
                    or left_key.startswith(right_key + "/")
                    or right_key.startswith(left_key + "/")
                ):
                    return True
            elif left_key == right_key:
                return True
    return False


def _trimmed_string(bead: Mapping[str, Any], name: str) -> str:
    value = _value(bead, name)
    return value.strip() if isinstance(value, str) else ""


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


def _is_access(value: object, *allowed: str) -> bool:
    return isinstance(value, str) and value in allowed


def _is_prep_operation_access(value: object) -> bool:
    return isinstance(value, str) and PREP_OPERATION_ACCESS_PATTERN.fullmatch(value) is not None


def _authority_writer(bead: Mapping[str, Any]) -> bool:
    access = _value(bead, "live_data_access")
    return (
        _is_access(access, "active-authorized", "authorized-inactive-generation-writer")
        or _is_prep_operation_access(access)
        or _is_access(access, *FINAL_CANDIDATE_ACCESS)
        or isinstance(access, str)
        and "candidate" in access
        and "write" in access
    )


def _serialization_errors(bead: Mapping[str, Any], wave: str) -> list[str]:
    if not _authority_writer(bead) or _serialized(bead):
        return []
    access = _value(bead, "live_data_access")
    if wave.startswith("reindex-prep-") and _value(bead, "execution_kind") == "authorized-prep-operation":
        return [f"{bead['id']}: authorized prep operation must be serialized"]
    if _is_access(access, "active-authorized", "authorized-inactive-generation-writer") or _is_prep_operation_access(
        access
    ):
        return [f"{bead['id']}: operator-authorized writer is not serialized"]
    if _is_access(access, *FINAL_CANDIDATE_ACCESS) or (
        isinstance(access, str) and "candidate" in access and "write" in access
    ):
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


def _validated_integration_head_syntax(value: object) -> str:
    if not isinstance(value, str) or INTEGRATION_HEAD_PATTERN.fullmatch(value) is None:
        raise ReindexPacketValidationError("integration head must be a 40-character lowercase commit SHA")
    return value


def _integration_head_argument(value: str) -> str:
    try:
        return _validated_integration_head_syntax(value)
    except ReindexPacketValidationError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _checkout_integration_head() -> str:
    repository_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ReindexPacketValidationError("unable to resolve exact checkout HEAD") from exc
    return _validated_integration_head_syntax(result.stdout.strip())


def _validate_integration_head(value: str | None) -> str:
    """Require an exact commit identity for this checkout's current HEAD."""
    repository_root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    checkout_head = _checkout_integration_head()
    if value is None:
        return checkout_head
    head = _validated_integration_head_syntax(value)
    if head == checkout_head:
        return head
    try:
        resolved = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "--verify", f"{head}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            env=environment,
        )
        if resolved.stdout.strip() != head:
            raise ReindexPacketValidationError(
                f"integration head {head} does not name an actual commit in this checkout"
            )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ReindexPacketValidationError(
            f"integration head {head} does not name an existing commit in this checkout"
        ) from exc
    try:
        subprocess.run(
            ["git", "-C", str(repository_root), "merge-base", "--is-ancestor", head, "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            env=environment,
        )
    except subprocess.CalledProcessError as exc:
        try:
            subprocess.run(
                ["git", "-C", str(repository_root), "merge-base", "--is-ancestor", "HEAD", head],
                check=True,
                capture_output=True,
                text=True,
                timeout=SUBPROCESS_TIMEOUT_SECONDS,
                env=environment,
            )
        except subprocess.CalledProcessError:
            raise ReindexPacketValidationError(
                f"integration head {head} does not equal exact checkout HEAD {checkout_head}"
            ) from exc
        except (OSError, subprocess.TimeoutExpired) as descendant_exc:
            raise ReindexPacketValidationError(
                f"unable to classify integration head {head} against exact checkout HEAD {checkout_head}"
            ) from descendant_exc
        raise ReindexPacketValidationError(
            f"integration head {head} is a descendant of exact checkout HEAD {checkout_head}"
        ) from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ReindexPacketValidationError(
            f"unable to classify integration head {head} against exact checkout HEAD {checkout_head}"
        ) from exc
    raise ReindexPacketValidationError(
        f"integration head {head} is a stale ancestor of exact checkout HEAD {checkout_head}"
    )


def _parse_phase_evidence(value: object, fields: frozenset[str], *, name: str) -> PhaseEvidence | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} evidence fields must be {sorted(fields)}")
    evidence_id, state, plan_digest = value.get("evidence_id"), value.get("state"), value.get("plan_digest")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError(f"{name} evidence has incomplete identity")
    if not isinstance(state, str) or not state:
        raise ValueError(f"{name} evidence has incomplete identity")
    if not isinstance(plan_digest, str) or not plan_digest:
        raise ValueError(f"{name} evidence has incomplete identity")
    expires_at = value.get("expires_at")
    if "expires_at" in fields and (not isinstance(expires_at, str) or _parse_datetime(expires_at) is None):
        raise ValueError("authorization evidence expires_at must be timezone-aware ISO-8601")
    return PhaseEvidence(evidence_id, state, plan_digest, expires_at)


def _parse_operation_evidence(value: object) -> OperationEvidence:
    """Parse untrusted diagnostic evidence without granting authority from it."""
    if not isinstance(value, Mapping) or set(value) != DIAGNOSTIC_EVIDENCE_FIELDS:
        raise ValueError(f"diagnostic operation evidence fields must be {sorted(DIAGNOSTIC_EVIDENCE_FIELDS)}")
    operation_id = value.get("operation_id")
    integration_head = value.get("integration_head")
    packet_context_digest = value.get("packet_context_digest")
    plan_digest = value.get("plan_digest")
    if not isinstance(operation_id, str) or not operation_id:
        raise ValueError("diagnostic operation evidence missing operation_id")
    if not isinstance(integration_head, str) or not INTEGRATION_HEAD_PATTERN.fullmatch(integration_head):
        raise ValueError("diagnostic operation evidence has invalid integration_head")
    if not isinstance(packet_context_digest, str) or not packet_context_digest.startswith(TASK_REVISION_PREFIX):
        raise ValueError("diagnostic operation evidence has invalid packet_context_digest")
    if not isinstance(plan_digest, str) or not plan_digest:
        raise ValueError("diagnostic operation evidence missing plan_digest")
    return OperationEvidence(
        operation_id,
        integration_head,
        packet_context_digest,
        plan_digest,
        _parse_phase_evidence(value.get("rehearsal"), REHEARSAL_EVIDENCE_FIELDS, name="rehearsal"),
        _parse_phase_evidence(value.get("authorization"), AUTHORIZATION_EVIDENCE_FIELDS, name="authorization"),
        _parse_phase_evidence(value.get("review"), REVIEW_EVIDENCE_FIELDS, name="review"),
    )


def _execution_kind_errors(bead: Mapping[str, Any], wave: str) -> list[str]:
    bead_id, access, kind = bead["id"], _value(bead, "live_data_access"), _value(bead, "execution_kind")
    errors = []
    if wave not in WAVES:
        errors.append(f"{bead_id}: unknown execution_wave {wave!r}")
    if kind is not None and kind not in EXECUTION_KINDS:
        errors.append(f"{bead_id}: unknown execution_kind {kind!r}")
    if _is_prep_operation_access(access) and kind != "authorized-prep-operation":
        if wave in PREP_OPERATION_WAVES:
            errors.append(f"{bead_id}: {wave}/{access} requires execution_kind authorized-prep-operation")
        else:
            errors.append(
                f"{bead_id}: live_data_access {access} requires execution_kind authorized-prep-operation in a prep wave"
            )
    if kind == "authorized-prep-operation":
        if wave not in PREP_OPERATION_WAVES:
            errors.append(f"{bead_id}: authorized prep operation requires a declared prep execution_wave")
        if not _is_prep_operation_access(access):
            errors.append(f"{bead_id}: authorized prep operation requires operator-authorized live_data_access")
    return errors


def _operation_phase_errors(bead: Mapping[str, Any]) -> list[str]:
    if _value(bead, "execution_kind") != "authorized-prep-operation":
        return []
    bead_id, phase = bead["id"], _value(bead, "operation_phase_contract")
    errors = []
    if isinstance(phase, str) and phase.strip():
        errors.append(f"{bead_id}: operation_phase_contract requires structured-v2-required/legacy-field")
    elif not isinstance(phase, Mapping):
        errors.append(f"{bead_id}: authorized prep operation missing operation_phase_contract")
    else:
        fields = {"version", "shape"}
        if set(phase) != fields or phase.get("version") != OPERATION_PHASE_VERSION:
            errors.append(f"{bead_id}: authorized prep operation has invalid operation_phase_contract")
        elif phase.get("shape") not in OPERATION_SHAPES:
            errors.append(f"{bead_id}: authorized prep operation has invalid phase shape")
    initial, apply = _value(bead, "initial_job_authority"), _value(bead, "apply_authority")
    if isinstance(initial, str) and initial.strip():
        errors.append(f"{bead_id}: initial_job_authority requires structured-v2-required/legacy-field")
    elif (
        not isinstance(initial, Mapping)
        or set(initial) != {"mode"}
        or initial.get("mode") not in INITIAL_AUTHORITY_MODES
    ):
        errors.append(f"{bead_id}: authorized prep operation has invalid initial_job_authority")
    if isinstance(apply, str) and apply.strip():
        errors.append(f"{bead_id}: apply_authority requires structured-v2-required/legacy-field")
    elif not isinstance(apply, Mapping) or set(apply) != {"mode"} or apply.get("mode") != APPLY_AUTHORITY_MODE:
        errors.append(f"{bead_id}: authorized prep operation has invalid apply_authority")
    if _present(_value(bead, "readiness_contract")):
        errors.append(f"{bead_id}: authorized prep operation must not store runtime readiness_contract")
    forbidden = sorted(field for field in PROHIBITED_OPERATION_POLICY_FIELDS if _present(_value(bead, field)))
    if forbidden:
        errors.append(f"{bead_id}: authorized prep operation stores runtime policy field(s): {', '.join(forbidden)}")
    return errors


def _failure(kind: str, reason: str, **details: str) -> dict[str, str]:
    return {"kind": kind, "reason": reason, **details}


def _packet_context_digest(
    group: tuple[str, str, str], members: Sequence[str], beads: Mapping[str, Mapping[str, Any]], integration_head: str
) -> str:
    payload = {
        "domain": "polylogue.reindex.packet-context.v1",
        "wave": group[0],
        "lane": group[1],
        "packet": group[2],
        "member_ids": list(members),
        "integration_head": integration_head,
        "task_identities": [asdict(TaskIdentity(bead_id, _task_revision(beads[bead_id]))) for bead_id in members],
    }
    encoded = json.dumps(_canonical_value(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return f"{TASK_REVISION_PREFIX}{hashlib.sha256(encoded).hexdigest()}"


def _operation_apply_failures(
    bead: Mapping[str, Any], evidence: object | None, *, integration_head: str, packet_context_digest: str
) -> list[dict[str, str]]:
    bead_id, phase = bead["id"], _value(bead, "operation_phase_contract")
    evidence_by_kind: list[tuple[str, PhaseEvidence | None, str]] = []
    if evidence is None:
        evidence_by_kind = [("rehearsal", None, "accepted"), ("operator-authorization", None, "authorized")]
        if isinstance(phase, Mapping) and phase.get("shape") == "plan-rehearse-review-authorize-apply-verify":
            evidence_by_kind.append(("independent-review", None, "accepted"))
        return [_failure(kind, "missing", bead_id=bead_id) for kind, _, _ in evidence_by_kind] + [
            _failure("apply-authority", "unsupported-evidence-adapter", bead_id=bead_id)
        ]
    try:
        parsed = _parse_operation_evidence(evidence)
    except ValueError as exc:
        return [
            _failure("diagnostic-operation-evidence", str(exc), bead_id=bead_id),
            _failure("apply-authority", "unsupported-evidence-adapter", bead_id=bead_id),
        ]
    failures: list[dict[str, str]] = []
    if parsed.operation_id != bead_id:
        failures.append(_failure("diagnostic-operation-evidence", "mismatched-operation-id", bead_id=bead_id))
    if parsed.integration_head != integration_head:
        failures.append(_failure("diagnostic-operation-evidence", "mismatched-integration-head", bead_id=bead_id))
    if parsed.packet_context_digest != packet_context_digest:
        failures.append(_failure("diagnostic-operation-evidence", "mismatched-packet-context", bead_id=bead_id))
    evidence_by_kind = [
        ("rehearsal", parsed.rehearsal, "accepted"),
        ("operator-authorization", parsed.authorization, "authorized"),
    ]
    if isinstance(phase, Mapping) and phase.get("shape") == "plan-rehearse-review-authorize-apply-verify":
        evidence_by_kind.append(("independent-review", parsed.review, "accepted"))
    for kind, item, accepted_state in evidence_by_kind:
        if item is None:
            failures.append(_failure(kind, "missing", bead_id=bead_id))
            continue
        if item.state != accepted_state:
            failures.append(_failure(kind, item.state, bead_id=bead_id, evidence_id=item.evidence_id))
        elif item.plan_digest != parsed.plan_digest:
            failures.append(_failure(kind, "mismatched-plan-digest", bead_id=bead_id, evidence_id=item.evidence_id))
    evidence_ids = [item.evidence_id for _, item, _ in evidence_by_kind if item is not None]
    if len(evidence_ids) != len(set(evidence_ids)):
        failures.append(_failure("diagnostic-operation-evidence", "non-distinct-evidence-id", bead_id=bead_id))
    # Caller JSON has no trusted clock or attestation. It can diagnose malformed
    # evidence, but it cannot make an apply projection ready or grant authority.
    failures.append(_failure("apply-authority", "unsupported-evidence-adapter", bead_id=bead_id))
    return failures


def _projection(
    group: tuple[str, str, str],
    members: list[str],
    beads: Mapping[str, Mapping[str, Any]],
    integration_head: str,
    phase: str,
    launch_failures: Sequence[dict[str, str]],
    operation_evidence: Mapping[str, object] | None = None,
    operation_member: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    leader = beads[members[0]]
    if phase == "initial":
        initial_authority = _value(operation_member, "initial_job_authority") if operation_member else None
        initial_mode = initial_authority.get("mode") if isinstance(initial_authority, Mapping) else None
        initial_valid = initial_mode in INITIAL_AUTHORITY_MODES
        authority = {
            "mode": initial_mode if operation_member and initial_valid else "invalid",
            "allowed_actions": ["read-only-plan", "isolated-rehearsal"] if operation_member else ["packet-execution"],
            "live_authority": "none",
            "may_apply": False,
        }
    elif phase == "apply":
        authority = {
            "mode": APPLY_AUTHORITY_MODE,
            "allowed_actions": [],
            "live_authority": "none",
            "may_apply": False,
        }
    else:
        live_data_access = _value(leader, "live_data_access")
        authority = {
            "mode": "ordinary-launch",
            "allowed_actions": ["packet-execution"],
            "live_authority": (
                "none"
                if _is_prep_operation_access(live_data_access) or not isinstance(live_data_access, str)
                else live_data_access
            ),
            "may_apply": False,
        }
    projection: dict[str, Any] = {
        "version": "packet-launch-projection-v2",
        "wave": group[0],
        "lane": group[1],
        "packet": group[2],
        "selected_phase": phase,
        "operation_member_id": operation_member["id"] if operation_member else None,
        "effective_authority": authority,
        "member_ids": list(members),
        "integration_head": integration_head,
        "launch_contract": {
            "packet_execution_contract": _value(leader, "packet_execution_contract"),
            "deadline_policy": _value(leader, "deadline_policy"),
            "model_policy": _effective_policy(beads, leader),
            "worker_capability": _value(leader, "worker_capability"),
            "review_capability": _value(leader, "review_capability"),
            "ownership_resources": _value(leader, "ownership_resources"),
            "verification_specs": _value(leader, "verification_specs"),
            "evidence_obligations": _value(leader, "evidence_obligations"),
        },
        "task_identities": [asdict(TaskIdentity(bead_id, _task_revision(beads[bead_id]))) for bead_id in members],
        "operation_evidence": dict(operation_evidence or {}),
        "launch_failures": list(launch_failures),
    }
    projection["ready"] = not projection["launch_failures"] and authority["mode"] != "invalid"
    payload = json.dumps(
        {"domain": "polylogue.reindex.launch-projection.v3", "projection": projection},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    projection["context_digest"] = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    projection["phase_token"] = f"{projection['version']}:{phase}:{projection['context_digest']}"
    return projection


def _operation_evidence_argument(value: str) -> Mapping[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"operation evidence must be JSON: {exc.msg}") from exc
    if not isinstance(parsed, Mapping):
        raise argparse.ArgumentTypeError("operation evidence must map operation IDs to evidence")
    return parsed


def validate(
    reader: Any,
    *,
    root_id: str = ROOT_ID,
    integration_head: str | None = None,
    operation_evidence: Mapping[str, object] | None = None,
    selected_phase: str = "initial",
) -> dict[str, Any]:
    if selected_phase not in PHASES:
        raise ValueError(f"selected phase must be one of {sorted(PHASES)}")
    effective_head = _validate_integration_head(integration_head)
    evidence_by_operation = dict(operation_evidence or {})
    beads = {bead["id"]: bead for bead in reader.read() if _is_task(bead)}
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
    root_policy = _value(beads[root_id], "model_policy")
    if not isinstance(root_policy, str) or not root_policy.startswith("provider-neutral-"):
        errors.append(f"{root_id}: campaign root must define a provider-neutral model policy")
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
    leaf_structural_failures: dict[str, list[dict[str, str]]] = defaultdict(list)

    def leaf_failure(bead: Mapping[str, Any], kind: str, reason: str, **details: str) -> None:
        errors.append(reason)
        leaf_structural_failures[bead["id"]].append(_failure(kind, reason, bead_id=bead["id"], **details))

    for bead_id in sorted(selected):
        bead = beads[bead_id]
        shape = _value(bead, "execution_shape") or _label(bead, "execution-shape:")
        legacy = sorted(field for field in LEGACY_METADATA_FIELDS if _present(_value(bead, field)))
        if legacy:
            leaf_failure(
                bead,
                "legacy-metadata",
                f"{bead_id}: retired metadata field(s): {', '.join(legacy)}",
                field=",".join(legacy),
            )
        if shape == "gate":
            for field in (
                "lane_packet",
                "lane_order",
                *LAUNCH,
                "worker_capability",
                "review_capability",
                "ownership_resources",
                "verification_specs",
                "evidence_obligations",
            ):
                if _present(_value(bead, field)):
                    errors.append(f"{bead_id}: gate carries {field}")
        elif bead.get("status") != "closed":
            missing = [field for field in CORE if not _present(_value(bead, field))]
            if _value(bead, "worker_capability") not in WORKER_CAPABILITIES:
                leaf_failure(bead, "worker-capability", f"{bead_id}: worker_capability is not a closed class")
            if _value(bead, "review_capability") not in REVIEW_CAPABILITIES:
                leaf_failure(bead, "review-capability", f"{bead_id}: review_capability is not a closed class")
            decision_stage = _value(bead, "decision_stage")
            if decision_stage not in DECISION_STAGES:
                leaf_failure(bead, "decision-stage", f"{bead_id}: decision_stage is not a closed class")
            resources = _typed_resources(_value(bead, "ownership_resources"))
            if resources is None:
                leaf_failure(bead, "ownership-resources", f"{bead_id}: ownership_resources must be typed resources")
            else:
                repo_root = Path(__file__).resolve().parents[1]
                for kind, key in resources:
                    if kind == "path-prefix" and not (repo_root / key).exists():
                        leaf_failure(bead, "ownership-path", f"{bead_id}: ownership path does not exist: {key}")
            if not _valid_verification_specs(_value(bead, "verification_specs")):
                leaf_failure(bead, "verification-spec", f"{bead_id}: verification_specs must contain argv arrays")
            if not _valid_evidence_obligations(_value(bead, "evidence_obligations")):
                leaf_failure(bead, "evidence-obligation", f"{bead_id}: evidence_obligations must be typed obligations")
            if missing:
                leaf_failure(
                    bead,
                    "missing-leaf-carrier",
                    f"{bead_id}: missing leaf carrier(s): {', '.join(missing)}",
                    field=",".join(missing),
                )
            policy = _value(bead, "model_policy")
            if policy is not None and policy == root_policy:
                leaf_failure(bead, "duplicate-model-policy", f"{bead_id}: repeated root model policy must be omitted")
            if policy is not None and (not isinstance(policy, str) or not policy.startswith("provider-neutral-")):
                leaf_failure(bead, "model-policy", f"{bead_id}: model policy is not provider-neutral")
            access = _value(bead, "live_data_access")
            if not isinstance(access, str):
                leaf_failure(bead, "authority-shape", f"{bead_id}: live_data_access must be a string")
            elif not access:
                leaf_failure(bead, "missing-live-data-authority", f"{bead_id}: missing live_data_access")
            leaves.append(bead)
    assignments: dict[str, tuple[str, str, str, int]] = {}
    packet_structural_failures: dict[str, list[dict[str, str]]] = defaultdict(list)
    groups: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    unassigned_structural_failures: list[dict[str, str]] = []
    for bead in leaves:
        wave, lane, packet, order_text = (_trimmed_string(bead, field) for field in CORE[:4])
        group = (wave, lane, packet)
        declared_packet = all(group)
        if declared_packet:
            groups[group].append(bead["id"])
        for serialization_error in _serialization_errors(bead, wave):
            errors.append(serialization_error)
            packet_structural_failures[bead["id"]].append(_failure("serialization", serialization_error))
        if wave:
            for execution_error in _execution_kind_errors(bead, wave):
                errors.append(execution_error)
                packet_structural_failures[bead["id"]].append(_failure("execution-kind", execution_error))
        for phase_error in _operation_phase_errors(bead):
            errors.append(phase_error)
            packet_structural_failures[bead["id"]].append(_failure("operation-phase", phase_error, bead_id=bead["id"]))
        try:
            assignment = (wave, lane, packet, int(order_text))
        except ValueError:
            reason = f"{bead['id']}: lane order is not numeric"
            leaf_failure(bead, "packet-assignment", reason)
            continue
        if not all((wave, lane, packet)) or assignment[3] < 1:
            reason = f"{bead['id']}: invalid packet assignment"
            leaf_failure(bead, "packet-assignment", reason)
            continue
        assignments[bead["id"]] = assignment
    assigned_to_declared_packet = {bead_id for members in groups.values() for bead_id in members}
    for bead in leaves:
        if bead["id"] in assigned_to_declared_packet:
            continue
        unassigned_structural_failures.extend(leaf_structural_failures[bead["id"]])
        unassigned_structural_failures.extend(packet_structural_failures[bead["id"]])
        unassigned_structural_failures.append(
            _failure(
                "unassigned-structural",
                f"{bead['id']}: cannot identify declared packet for structural failure",
                bead_id=bead["id"],
            )
        )
    graph = {bead_id: _deps(bead, "blocks") for bead_id, bead in beads.items()}
    conflict_failures: dict[str, list[dict[str, str]]] = defaultdict(list)
    for left_id, left in assignments.items():
        for right_id, right in assignments.items():
            if left_id >= right_id or left[0] != right[0] or left[1] == right[1]:
                continue
            left_resources = set(_typed_resources(_value(beads[left_id], "ownership_resources")) or ())
            right_resources = set(_typed_resources(_value(beads[right_id], "ownership_resources")) or ())
            overlap = _resources_overlap(left_resources, right_resources)
            if overlap and not (
                _path(graph, left_id, right_id)
                or _path(graph, right_id, left_id)
                or _serialized(beads[left_id])
                or _serialized(beads[right_id])
            ):
                error = f"{left_id}/{right_id}: ownership-resource overlap is not serialized"
                errors.append(error)
                for bead_id in (left_id, right_id):
                    conflict_failures[bead_id].append(_failure("conflict-serialization", error, bead_id=bead_id))
    packets = []
    for group, members in sorted(groups.items()):
        members.sort(
            key=lambda bead_id: (assignments[bead_id][3], bead_id) if bead_id in assignments else (sys.maxsize, bead_id)
        )
        leader = members[0]
        launch_failures = [
            failure
            for bead_id in members
            for failure in (
                *leaf_structural_failures[bead_id],
                *packet_structural_failures[bead_id],
                *conflict_failures[bead_id],
            )
        ]
        launch_failures.extend(unassigned_structural_failures)
        if (
            not any(_present(_value(beads[bead_id], "packet_size_exception")) for bead_id in members)
            and not 3 <= len(members) <= 5
        ):
            error = f"{'/'.join(group)}: ordinary packet has {len(members)} leaves"
            errors.append(error)
            launch_failures.append(_failure("packet-shape", error))
        assigned_members = [bead_id for bead_id in members if bead_id in assignments]
        if len(assigned_members) == len(members) and (
            len({assignments[bead_id][3] for bead_id in members}) != len(members)
        ):
            error = f"{'/'.join(group)}: duplicate packet order"
            errors.append(error)
            launch_failures.append(_failure("packet-order", error))
        external_blockers = []
        for bead_id in members:
            for target in graph[bead_id]:
                if target not in members:
                    if target in beads and beads[target].get("status") != "closed":
                        external_blockers.append(target)
                    if bead_id in assignments and target in assignments:
                        current, predecessor = assignments[bead_id], assignments[target]
                        if current[:2] == predecessor[:2] and int(current[2]) < int(predecessor[2]):
                            error = f"{bead_id}: packet blocker is in a later packet"
                            errors.append(error)
                            launch_failures.append(_failure("blocker-order", error, bead_id=bead_id))
                        if (
                            predecessor[0] in WAVES
                            and current[0] in WAVES
                            and WAVES[predecessor[0]] > WAVES[current[0]]
                        ):
                            error = f"{bead_id}: earlier wave blocks on later wave"
                            errors.append(error)
                            launch_failures.append(_failure("blocker-order", error, bead_id=bead_id))
                    continue
                if bead_id not in assignments or target not in assignments:
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
        for bead_id in members[1:]:
            for field in LAUNCH:
                if _present(_value(beads[bead_id], field)):
                    error = f"{bead_id}: non-leader carries {field}"
                    errors.append(error)
                    launch_failures.append(_failure("non-leader-launch-field", error, bead_id=bead_id, field=field))
        operation_members = [
            bead_id for bead_id in members if _value(beads[bead_id], "execution_kind") == "authorized-prep-operation"
        ]
        operation_membership_failures: list[dict[str, str]] = []
        operation_member: Mapping[str, Any] | None = None
        if operation_members:
            if len(operation_members) != 1:
                error = f"{'/'.join(group)}: operation packet has multiple authorized-prep-operation members"
                errors.append(error)
                operation_membership_failures.append(_failure("operation-membership", "multiple-operation-membership"))
            elif len(members) != 1:
                error = f"{'/'.join(group)}: operation packet mixes an authorized-prep-operation with ordinary members"
                errors.append(error)
                operation_membership_failures.append(_failure("operation-membership", "mixed-operation-membership"))
            else:
                operation_member = beads[operation_members[0]]
        ordinary_failures = [
            *launch_failures,
            *operation_membership_failures,
            *(_failure("blocks", "open", bead_id=bead_id) for bead_id in sorted(set(external_blockers))),
        ]
        phases = ("initial", "apply") if operation_members else ("ordinary",)
        packet_context_digest = _packet_context_digest(group, members, beads, effective_head)
        launches = []
        for phase in phases:
            phase_failures = list(ordinary_failures)
            if phase == "apply":
                phase_failures.extend(
                    failure
                    for bead_id in operation_members
                    for failure in _operation_apply_failures(
                        beads[bead_id],
                        evidence_by_operation.get(bead_id),
                        integration_head=effective_head,
                        packet_context_digest=packet_context_digest,
                    )
                )
            evidence_binding = None
            if phase == "apply":
                evidence_binding = {}
                for bead_id in operation_members:
                    with suppress(ValueError):
                        evidence_binding[bead_id] = asdict(
                            _parse_operation_evidence(evidence_by_operation.get(bead_id))
                        )
            launches.append(
                _projection(
                    group,
                    members,
                    beads,
                    effective_head,
                    phase,
                    phase_failures,
                    evidence_binding,
                    operation_member,
                )
            )
        selected_launch = next(
            (launch for launch in launches if launch["selected_phase"] == selected_phase), launches[0]
        )
        packets.append(
            {
                "wave": group[0],
                "lane": group[1],
                "packet": group[2],
                "member_ids": members,
                "leader_id": leader,
                "packet_context_digest": packet_context_digest,
                "selected_phase": selected_launch["selected_phase"],
                "ready": selected_launch["ready"],
                "non_ready_reasons": list(
                    dict.fromkeys(f"{item['kind']}:{item['reason']}" for item in selected_launch["launch_failures"])
                ),
                "launches": launches,
            }
        )
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
        "global_launch_failures": unassigned_structural_failures,
        "structural_errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None, *, reader: Any = None, stdout: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-id", default=ROOT_ID)
    parser.add_argument("--integration-head", type=_integration_head_argument)
    parser.add_argument("--phase", choices=sorted(PHASES), default="initial")
    parser.add_argument(
        "--operation-evidence-json",
        type=_operation_evidence_argument,
        help=(
            "untrusted diagnostic evidence keyed by operation Bead ID; it is never read from or written to Beads "
            "and cannot grant apply authority"
        ),
    )
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
            integration_head=args.integration_head,
            operation_evidence=args.operation_evidence_json,
            selected_phase=args.phase,
        )
    except ReindexPacketValidationError as exc:
        print(f"reindex-packets: validation failed: {exc}", file=output)
        return 2
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
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
                    f"NOT READY [{packet['selected_phase']}] "
                    f"{packet['wave']}/{packet['lane']}/{packet['packet']}: {'; '.join(packet['non_ready_reasons'])}",
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

"""Validate structural integrity of the Beads dependency graph.

The gate inspects typed dependency records: endpoint existence, duplicate
edges, parent cardinality, and cycles.  When the native reindex campaign marker
is present, it also projects campaign bindings and WIP rules from native Beads
fields; it never reads a roster, snapshot, prose ledger, or persisted queue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEPENDENCY_KINDS = frozenset({"blocks", "parent-child", "relates-to", "discovered-from", "supersedes"})


@dataclass(frozen=True, slots=True)
class Finding:
    kind: str
    bead_id: str
    detail: str


CAMPAIGN_ID = "reindex-2026"
CAMPAIGN_LABEL = f"campaign:{CAMPAIGN_ID}"
CAMPAIGN_ROOT = "polylogue-reindex-2026"
CAMPAIGN_SCHEMA = "polylogue.campaign-control.v1"
WORKSTREAMS = frozenset("abcdefgh")
BATCH_STAGES = frozenset({"prepare", "implement", "verify", "merge", "dispositions"})
BATCH_FORMULA = "mol-polylogue-thematic-batch"
BATCH_FORMULA_VERSION = 1
BATCH_BINDING_PREFIX = "NATIVE_BATCH_BINDING_V1 "
BATCH_LABELS = frozenset({CAMPAIGN_LABEL, "campaign-role:batch", "timing:execution"})
# The migration commit is durable source evidence, not a roster embedded in this
# validator.  The source loader below reads it through git without writing state.
CAMPAIGN_SOURCE_REF = "317b59f41f938884d289d48737cfe87ec00bd769:.beads/issues.jsonl"
CAMPAIGN_SOURCE_VERSION = "reindex-native-v1"
CAMPAIGN_NATIVE_CONTROL_ID = "polylogue-reindex-native-control-plane"
CAMPAIGN_ADAPTER_ID = "polylogue-agentctl-adapter"
CAMPAIGN_ADAPTER_PROVENANCE = frozenset(
    {
        ("staged-adapter", "agentctl:staged-native-adapter"),
        ("production-adapter", "agentctl:production-native-adapter"),
    }
)
CAMPAIGN_GENESIS_PATH = "devtools/campaign_genesis/reindex-2026.json"
CAMPAIGN_GENESIS_SCHEMA = "polylogue.campaign-genesis/v1"
CAMPAIGN_ACCEPTANCE_SCHEMA = "polylogue.campaign-acceptance/v1"
CAMPAIGN_POUR_RECEIPT_SCHEMA = "polylogue.campaign-pour-receipt/v1"
CAMPAIGN_POUR_RECEIPT_DIRECTORY = "devtools/campaign_pour_receipts/reindex-2026"
MERGE_READY_STATUSES = frozenset({"open", "in_progress"})
KNOWN_STATUSES = frozenset({"open", "in_progress", "blocked", "deferred", "closed"})
WIP_LIMITS = {"implementation_lane_wip": 6, "merge_train_wip": 1, "workstream_active_batch_wip": 1}
BEAD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")
BATCH_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _source_evidence() -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    """Read the immutable migration export used for historical census.

    This is deliberately a read-only git lookup.  A missing/unparseable source
    is a finding rather than permission to trust the live graph.  The returned
    rows are evidence for identity/edge comparison, never an operational queue.
    """
    try:
        result = subprocess.run(["git", "show", CAMPAIGN_SOURCE_REF], capture_output=True, text=True, check=False)
    except OSError as exc:
        return None, [], f"unable to read migration source: {exc}"
    if result.returncode != 0:
        return None, [], f"unable to read migration source {CAMPAIGN_SOURCE_REF!r}"
    try:
        rows = _validated_issues(
            [json.loads(line) for line in result.stdout.splitlines() if line.strip()], source=CAMPAIGN_SOURCE_REF
        )
    except (RuntimeError, json.JSONDecodeError) as exc:
        return None, [], f"migration source is malformed: {exc}"
    root = next((row for row in rows if str(row.get("id")) == CAMPAIGN_ROOT), None)
    return root, rows, None


def _labels(issue: dict[str, Any], findings: list[Finding] | None = None) -> list[str]:
    value = issue.get("labels")
    if value is None:
        return []
    if not isinstance(value, list):
        if findings is not None:
            findings.append(Finding("malformed-labels", str(issue.get("id", "<unknown>")), "labels must be a list"))
        return []
    labels = [label for label in value if isinstance(label, str)]
    if findings is not None and len(labels) != len(value):
        findings.append(Finding("malformed-label", str(issue.get("id", "<unknown>")), "labels must contain strings"))
    return labels


def _metadata(issue: dict[str, Any], findings: list[Finding] | None = None) -> dict[str, Any]:
    value = issue.get("metadata")
    if value is None:
        return {}
    if not isinstance(value, dict):
        if findings is not None:
            findings.append(
                Finding("malformed-metadata", str(issue.get("id", "<unknown>")), "metadata must be an object")
            )
        return {}
    return value


def _labels_with_prefix(issue: dict[str, Any], prefix: str, findings: list[Finding] | None = None) -> set[str]:
    return {label for label in _labels(issue, findings) if label.startswith(prefix)}


def _suffixes(issue: dict[str, Any], prefix: str, findings: list[Finding] | None = None) -> set[str]:
    return {label.removeprefix(prefix) for label in _labels_with_prefix(issue, prefix, findings)}


def _metadata_text(issue: dict[str, Any], key: str, findings: list[Finding] | None = None) -> str | None:
    """Return a campaign metadata scalar only when it is a string.

    Campaign records come from untrusted JSONL.  Keeping the type guard at the
    access boundary prevents list/dict values from reaching ``.lower`` or set
    membership while retaining a structured diagnostic for the caller.
    """
    value = _metadata(issue, findings).get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        if findings is not None:
            findings.append(
                Finding(
                    "malformed-metadata-field",
                    str(issue.get("id", "<unknown>")),
                    f"metadata.{key} must be a string",
                )
            )
        return None
    return value


def _metadata_integer(issue: dict[str, Any], key: str, findings: list[Finding] | None = None) -> int | None:
    value = _metadata(issue, findings).get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        if findings is not None:
            findings.append(
                Finding(
                    "malformed-metadata-field",
                    str(issue.get("id", "<unknown>")),
                    f"metadata.{key} must be an integer",
                )
            )
        return None
    return value


def _status(issue: dict[str, Any], findings: list[Finding] | None = None) -> str | None:
    value = issue.get("status")
    if value is None:
        return None
    if not isinstance(value, str):
        if findings is not None:
            findings.append(Finding("malformed-status", str(issue.get("id", "<unknown>")), "status must be a string"))
        return None
    return value


def _issue_type(issue: dict[str, Any], findings: list[Finding] | None = None) -> str | None:
    value = issue.get("issue_type")
    if value is None:
        return None
    if not isinstance(value, str):
        if findings is not None:
            findings.append(
                Finding("malformed-issue-type", str(issue.get("id", "<unknown>")), "issue_type must be a string")
            )
        return None
    return value


def _metadata_set(issue: dict[str, Any], key: str, findings: list[Finding] | None = None) -> set[str]:
    value = _metadata_text(issue, key, findings)
    if value is None:
        return set()
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def _binding_fields(issue: dict[str, Any]) -> dict[str, str]:
    """Read formula variables from the supported description substitution path."""
    description = issue.get("description")
    if not isinstance(description, str):
        return {}
    fields: dict[str, str] = {}
    for line in description.splitlines():
        if not line.startswith(BATCH_BINDING_PREFIX):
            continue
        key, separator, value = line[len(BATCH_BINDING_PREFIX) :].partition("=")
        if separator and key and key not in fields:
            fields[key] = value
    return fields


def _parent_targets(issue: dict[str, Any]) -> list[str]:
    dependencies = issue.get("dependencies")
    if not isinstance(dependencies, list):
        return []
    return [
        target
        for dependency in dependencies
        if isinstance(dependency, dict)
        and dependency.get("type") == "parent-child"
        and isinstance((target := dependency.get("depends_on_id")), str)
        and target
    ]


def _parse_authoritative_beads(value: Any) -> tuple[set[str], str | None]:
    if not isinstance(value, str) or not value.strip():
        return set(), "authoritative_beads is required"
    members = [part.strip() for part in value.split(",")]
    if any(not part or not BEAD_ID_RE.fullmatch(part) for part in members):
        return set(), "authoritative_beads contains an invalid Bead id"
    parsed = set(members)
    if len(parsed) != len(members):
        return set(), "authoritative_beads contains duplicate Bead ids"
    if not 3 <= len(parsed) <= 5:
        return parsed, "authoritative_beads must contain 3-5 Bead ids"
    return parsed, None


def _wip_limit(metadata: dict[str, Any], key: str, findings: list[Finding], root_id: str) -> int:
    value = metadata.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    findings.append(Finding("campaign-wip-metadata", root_id, f"{key} must be a non-negative integer"))
    return WIP_LIMITS[key]


def _staged_adapter_findings(
    issue: dict[str, Any],
    *,
    by_id: dict[str, dict[str, Any]],
    root_id: str,
    epic_id: str,
) -> list[Finding]:
    """Verify AgentCTL staging from native records and graph edges.

    Metadata names the claimed controls, but never authenticates them.  The
    control records, their campaign attachment, and the exact adapter edges
    are the authority; this makes copied metadata insufficient to pass.
    """
    bead_id = str(issue["id"])
    findings: list[Finding] = []
    metadata = _metadata(issue, findings)
    declared = metadata.get("native_control_ids")
    if not isinstance(declared, list) or any(not isinstance(item, str) or not item for item in declared):
        return [Finding("campaign-agentctl-provenance", bead_id, "native_control_ids must be a non-empty string list")]
    if len(set(declared)) != len(declared):
        findings.append(Finding("campaign-agentctl-provenance", bead_id, "native_control_ids contains duplicates"))

    root = by_id.get(root_id)
    epic = by_id.get(epic_id)
    root_labels = set(_labels(root or {}))
    root_attachment = _metadata_text(root or {}, "source_control_plane_sha256", findings)
    source_root, _source_rows, source_error = _source_evidence()
    source_attachment = _metadata_text(source_root or {}, "source_control_plane_sha256")
    native = by_id.get(CAMPAIGN_NATIVE_CONTROL_ID)
    native_id: str | None = CAMPAIGN_NATIVE_CONTROL_ID if native is not None else None
    if source_error is not None or root_attachment != source_attachment:
        findings.append(
            Finding(
                "campaign-source-anchor",
                root_id,
                "root source_control_plane_sha256 does not match the trusted immutable attachment",
            )
        )
    if native is None:
        findings.append(
            Finding(
                "campaign-agentctl-provenance",
                bead_id,
                f"trusted native control {CAMPAIGN_NATIVE_CONTROL_ID!r} is absent",
            )
        )
    else:
        native_attachment = _metadata_text(native, "source_attachment_sha256", findings)
        native_campaign_id = _metadata_text(native, "campaign_id", findings)
        native_workstream = _metadata_text(native, "workstream", findings)
        if native_attachment != source_attachment:
            findings.append(
                Finding(
                    "campaign-agentctl-provenance",
                    bead_id,
                    "native control is not attached to the trusted source digest",
                )
            )
        if native_campaign_id != CAMPAIGN_ID or native_workstream is None or native_workstream.lower() != "e":
            findings.append(
                Finding("campaign-agentctl-provenance", bead_id, "trusted native control identity is mismatched")
            )
    clones = [
        candidate
        for candidate in by_id.values()
        if str(candidate.get("id")) != CAMPAIGN_NATIVE_CONTROL_ID
        and (
            _metadata_text(candidate, "campaign_membership_source", findings) == "native-control-plane"
            or _metadata_text(candidate, "source_attachment_sha256", findings) == source_attachment
        )
    ]
    if clones:
        findings.append(
            Finding(
                "campaign-agentctl-provenance",
                bead_id,
                f"native control clone(s) claimed: {sorted(str(item['id']) for item in clones)}",
            )
        )

    expected = {root_id, epic_id} | ({native_id} if native_id is not None else set())
    if set(declared) != expected:
        findings.append(
            Finding(
                "campaign-agentctl-provenance",
                bead_id,
                f"declared native_control_ids={sorted(declared)} expected graph controls={sorted(expected)}",
            )
        )

    def has_edge(source: dict[str, Any] | None, target: str) -> bool:
        return source is not None and target in _campaign_dependency_targets(source)

    if not has_edge(epic, bead_id):
        findings.append(Finding("campaign-agentctl-edge", bead_id, f"{epic_id} must block staged adapter"))
    if native_id is not None and not has_edge(issue, native_id):
        findings.append(Finding("campaign-agentctl-edge", bead_id, f"staged adapter must block {native_id}"))
    if not has_edge(root, epic_id):
        findings.append(Finding("campaign-agentctl-edge", bead_id, f"{root_id} must block {epic_id}"))
    if native_id is not None and not has_edge(root, native_id):
        findings.append(Finding("campaign-agentctl-edge", bead_id, f"{root_id} must block {native_id}"))
    if (
        root is None
        or CAMPAIGN_LABEL not in root_labels
        or _metadata_text(root or {}, "campaign_schema", findings) != CAMPAIGN_SCHEMA
    ):
        findings.append(Finding("campaign-agentctl-provenance", bead_id, "root control record is not campaign-bound"))
    if (
        epic is None
        or CAMPAIGN_LABEL not in set(_labels(epic))
        or (
            (epic_workstream := _metadata_text(epic, "workstream", findings)) is None
            or f"workstream:{epic_workstream.lower()}" not in set(_labels(epic, findings))
        )
    ):
        findings.append(
            Finding("campaign-agentctl-provenance", bead_id, "workstream control record is not campaign-bound")
        )
    return findings


def _campaign_marker_present(issues: list[dict[str, Any]], *, campaign_id: str, root_id: str) -> bool:
    label = f"campaign:{campaign_id}"
    root = next((issue for issue in issues if str(issue.get("id")) == root_id), None)
    return any(
        label in set(_labels(issue)) or _metadata_text(issue, "campaign_id") == campaign_id for issue in issues
    ) or (root is not None and _metadata_text(root, "campaign_schema") is not None)


def _campaign_dependency_targets(issue: dict[str, Any], *, dependency_type: str = "blocks") -> set[str]:
    return {
        target
        for dependency in _dependency_records(issue)
        if dependency.get("type") == dependency_type
        and isinstance((target := dependency.get("depends_on_id")), str)
        and target
    }


def _batch_dag_findings(
    by_id: dict[str, dict[str, Any]],
) -> tuple[list[Finding], dict[str, list[dict[str, Any]]]]:
    """Validate every child and edge of each native poured batch molecule."""
    findings: list[Finding] = []
    groups: dict[str, list[dict[str, Any]]] = {}
    roots = [
        issue
        for issue in by_id.values()
        if _issue_type(issue, findings) == "molecule" and issue.get("title") == BATCH_FORMULA
    ]
    required_predecessors = {
        "implement": "prepare",
        "verify": "implement",
        "merge": "verify",
        "dispositions": "merge",
    }
    for root in roots:
        root_id = str(root["id"])
        # bd mol pour roots intentionally carry only issue_type/title; formula
        # identity is proven by the typed child metadata and exact child shape.
        children = [issue for issue in by_id.values() if root_id in _parent_targets(issue)]
        stage_children = [
            issue
            for issue in children
            if _issue_type(issue, findings) == "task" and _metadata_text(issue, "stage", findings) in BATCH_STAGES
        ]
        forged_stage_children = [
            issue
            for issue in children
            if _metadata_text(issue, "stage", findings) in BATCH_STAGES and _issue_type(issue, findings) != "task"
        ]
        for forged in forged_stage_children:
            findings.append(Finding("campaign-batch-child", root_id, f"stage child {forged['id']!r} is not a task"))
        stages = [_metadata_text(issue, "stage", findings) or "<missing>" for issue in stage_children]
        if len(stage_children) != len(set(stages)) or set(stages) != BATCH_STAGES:
            findings.append(
                Finding(
                    "campaign-batch-binding",
                    root_id,
                    f"native molecule children must contain exactly {sorted(BATCH_STAGES)}; got {sorted(stages)}",
                )
            )
        gates = [issue for issue in children if _issue_type(issue, findings) == "gate"]
        valid_gates = [
            gate
            for gate in gates
            if gate.get("await_type") == "human" and gate.get("await_id") == "operator-merge-authorization"
        ]
        if len(valid_gates) != 1:
            findings.append(
                Finding(
                    "campaign-batch-gate",
                    root_id,
                    "native molecule must have exactly one human merge authorization gate",
                )
            )
        allowed_children = {str(issue["id"]) for issue in stage_children} | {str(issue["id"]) for issue in valid_gates}
        for child in children:
            if str(child["id"]) not in allowed_children:
                findings.append(
                    Finding("campaign-batch-child", root_id, f"unvalidated native molecule child {child['id']!r}")
                )
        for stage_issue in stage_children:
            if (
                set(_labels(stage_issue, findings)) != BATCH_LABELS
                or _metadata_text(stage_issue, "campaign_id", findings) != CAMPAIGN_ID
                or _metadata_text(stage_issue, "formula", findings) != BATCH_FORMULA
                or _metadata_integer(stage_issue, "formula_version", findings) != BATCH_FORMULA_VERSION
                or _metadata_text(stage_issue, "molecule_type", findings) != "workflow"
                or _metadata_text(stage_issue, "pour_origin", findings) != "native-formula"
                or _metadata_text(stage_issue, "campaign_role", findings) != "batch"
                or _metadata_text(stage_issue, "campaign_timing", findings) != "execution"
            ):
                findings.append(
                    Finding(
                        "campaign-batch-binding", root_id, f"stage {stage_issue['id']!r} lacks native pour metadata"
                    )
                )
        by_stage = {
            stage: issue for issue in stage_children if (stage := _metadata_text(issue, "stage", findings)) is not None
        }
        for stage, predecessor in required_predecessors.items():
            current = by_stage.get(stage)
            prior = by_stage.get(predecessor)
            if current is None or prior is None or prior["id"] not in _campaign_dependency_targets(current):
                findings.append(Finding("campaign-batch-edge", root_id, f"{stage} must block on {predecessor}"))
        merge = by_stage.get("merge")
        if merge is not None and valid_gates and valid_gates[0]["id"] not in _campaign_dependency_targets(merge):
            findings.append(
                Finding("campaign-batch-gate", root_id, "merge stage must block on its human authorization gate")
            )
        groups[root_id] = stage_children
    return findings, groups


def collect_campaign_findings(
    issues: list[dict[str, Any]],
    *,
    campaign_id: str = CAMPAIGN_ID,
    root_id: str = CAMPAIGN_ROOT,
) -> list[Finding]:
    """Project campaign bindings from native graph records only.

    Work molecules are grouped by their native parent-child edges.  Formula
    variables are deliberately read from structured description markers: bd
    substitutes titles/descriptions, but does not substitute labels or
    metadata.  The complete stage DAG and identical authority set are checked
    for every group, so copied fields cannot manufacture a valid batch.
    """
    campaign_label = f"campaign:{campaign_id}"
    by_id = {str(issue["id"]): issue for issue in issues}
    findings: list[Finding] = []
    campaign_rows = [issue for issue in issues if campaign_label in set(_labels(issue, findings))]
    for issue in issues:
        labels = set(_labels(issue, findings))
        metadata = _metadata(issue, findings)
        if (
            metadata.get("campaign_id") == campaign_id
            or metadata.get("campaign_role") == "batch"
            or any(label.startswith("batch:") for label in labels)
        ) and campaign_label not in labels:
            findings.append(
                Finding("campaign-missing-binding", str(issue["id"]), "staged campaign row lacks campaign label")
            )
    root = by_id.get(root_id)
    if root is None:
        return [Finding("campaign-missing-root", root_id, f"campaign root {root_id!r} is absent")]
    root_metadata = _metadata(root, findings)
    root_labels = set(_labels(root, findings))
    if _metadata_text(root, "campaign_id", findings) != campaign_id:
        findings.append(
            Finding("campaign-missing-binding", root_id, "root metadata campaign_id is missing or inconsistent")
        )
    if _metadata_text(root, "campaign_schema", findings) != CAMPAIGN_SCHEMA:
        findings.append(Finding("campaign-schema", root_id, "root campaign_schema is missing or inconsistent"))
    if campaign_label not in root_labels or "campaign-role:milestone" not in root_labels:
        findings.append(Finding("campaign-missing-binding", root_id, "root campaign/milestone labels are missing"))

    root_targets = _campaign_dependency_targets(root)
    expected_epics = {f"polylogue-reindex-ws-{workstream}" for workstream in WORKSTREAMS}
    for epic_id in sorted(expected_epics - root_targets):
        findings.append(
            Finding("campaign-workstream-edge", epic_id, "workstream gate is not directly blocked by the root")
        )

    epics: dict[str, str] = {}
    for workstream in sorted(WORKSTREAMS):
        epic_id = f"polylogue-reindex-ws-{workstream}"
        epic = by_id.get(epic_id)
        if epic is None:
            findings.append(Finding("campaign-missing-workstream", epic_id, "workstream closure gate is absent"))
            continue
        epics[workstream] = epic_id
        labels = set(_labels(epic, findings))
        metadata = _metadata(epic, findings)
        if (
            campaign_label not in labels
            or f"workstream:{workstream}" not in labels
            or "campaign-role:closure-gate" not in labels
        ):
            findings.append(
                Finding("campaign-missing-binding", epic_id, "workstream gate lacks campaign/workstream/role labels")
            )
        epic_workstream = _metadata_text(epic, "workstream", findings)
        if (
            _metadata_text(epic, "campaign_id", findings) != campaign_id
            or epic_workstream is None
            or epic_workstream.lower() != workstream
            or _metadata_text(epic, "epic_semantics", findings) != "closure-gate-not-executable"
        ):
            findings.append(
                Finding("campaign-missing-binding", epic_id, "workstream gate metadata is missing or inconsistent")
            )

    owned: dict[str, set[str]] = defaultdict(set)
    for workstream, epic_id in epics.items():
        epic = by_id[epic_id]
        for target in _campaign_dependency_targets(epic):
            if target in by_id:
                owned[target].add(workstream)

    anchor = _metadata_text(root, "source_control_plane_sha256", findings)
    source_root, source_rows, source_error = _source_evidence()
    source_anchor = _metadata_text(source_root or {}, "source_control_plane_sha256")
    trusted_source = isinstance(anchor, str) and bool(anchor) and anchor == source_anchor and source_error is None
    if not isinstance(anchor, str) or not anchor.strip():
        findings.append(
            Finding("campaign-source-anchor", root_id, "source_control_plane_sha256 is missing or malformed")
        )
    elif source_error is not None:
        findings.append(Finding("campaign-source-anchor", root_id, source_error))
    elif anchor != source_anchor:
        findings.append(
            Finding("campaign-source-anchor", root_id, "root anchor disagrees with durable migration source")
        )
    if trusted_source:
        source_by_id = {str(row["id"]): row for row in source_rows}
        source_epics = {workstream: f"polylogue-reindex-ws-{workstream}" for workstream in WORKSTREAMS}

        def historical(row: dict[str, Any]) -> bool:
            bead_id = str(row.get("id"))
            return (
                CAMPAIGN_LABEL in set(_labels(row))
                and bead_id
                not in {CAMPAIGN_ROOT, *source_epics.values(), CAMPAIGN_NATIVE_CONTROL_ID, CAMPAIGN_ADAPTER_ID}
                and _metadata_text(row, "campaign_membership_kind") != "staged-adapter"
            )

        expected_members = {str(row["id"]) for row in source_rows if historical(row)}
        observed_members = {
            bead_id
            for bead_id, workstreams in owned.items()
            if bead_id not in {CAMPAIGN_NATIVE_CONTROL_ID, CAMPAIGN_ADAPTER_ID}
        }
        if observed_members != expected_members:
            missing = sorted(expected_members - observed_members)
            extra = sorted(observed_members - expected_members)
            findings.append(
                Finding(
                    "campaign-source-census",
                    root_id,
                    f"historical member identity differs: missing={missing}, extra={extra}",
                )
            )
        for workstream, source_epic_id in source_epics.items():
            expected_edges = {
                target
                for target in _campaign_dependency_targets(source_by_id.get(source_epic_id, {}))
                if target in expected_members
            }
            observed_edges = {
                target
                for target in _campaign_dependency_targets(by_id.get(epics.get(workstream, ""), {}))
                if target in observed_members
            }
            if observed_edges != expected_edges:
                findings.append(
                    Finding(
                        "campaign-source-census",
                        root_id,
                        f"historical {workstream} edge identity differs: missing={sorted(expected_edges - observed_edges)}, extra={sorted(observed_edges - expected_edges)}",
                    )
                )

        adapter = by_id.get(CAMPAIGN_ADAPTER_ID)
        if adapter is None:
            findings.append(
                Finding(
                    "campaign-agentctl-provenance", CAMPAIGN_ADAPTER_ID, "trusted singleton AgentCTL adapter is absent"
                )
            )
        else:
            adapter_labels = set(_labels(adapter, findings))
            adapter_provenance = (
                _metadata_text(adapter, "campaign_membership_kind", findings),
                _metadata_text(adapter, "campaign_membership_source", findings),
            )
            if adapter_provenance not in CAMPAIGN_ADAPTER_PROVENANCE:
                findings.append(
                    Finding(
                        "campaign-agentctl-provenance",
                        CAMPAIGN_ADAPTER_ID,
                        "trusted singleton adapter provenance is mismatched",
                    )
                )
            if CAMPAIGN_LABEL not in adapter_labels or "workstream:e" not in adapter_labels:
                findings.append(
                    Finding(
                        "campaign-agentctl-provenance",
                        CAMPAIGN_ADAPTER_ID,
                        "trusted singleton adapter labels are mismatched",
                    )
                )
            if CAMPAIGN_ADAPTER_ID not in owned:
                findings.append(
                    Finding(
                        "campaign-agentctl-edge", CAMPAIGN_ADAPTER_ID, "trusted singleton adapter is not graph-owned"
                    )
                )

        claimed_adapters = [
            issue
            for issue in issues
            if (
                (
                    _metadata_text(issue, "campaign_membership_kind", findings),
                    _metadata_text(issue, "campaign_membership_source", findings),
                )
                in CAMPAIGN_ADAPTER_PROVENANCE
                or _metadata(issue).get("native_control_ids") is not None
            )
            and str(issue.get("id")) != CAMPAIGN_ADAPTER_ID
        ]
        for clone in claimed_adapters:
            findings.append(
                Finding(
                    "campaign-agentctl-provenance",
                    str(clone["id"]),
                    "AgentCTL provenance claims a non-singleton adapter identity",
                )
            )
    controls = {root_id, *epics.values()}
    for target in sorted(root_targets - expected_epics):
        target_issue = by_id.get(target)
        target_labels = set(_labels(target_issue, findings)) if target_issue else set()
        campaign_bound = (
            target_issue is not None
            and campaign_label in target_labels
            and _metadata_text(target_issue, "campaign_id", findings) == campaign_id
        )
        if target not in owned or not campaign_bound:
            findings.append(
                Finding("campaign-root-edge", target, "root blocker is not campaign-bound and workstream-owned")
            )

    # Ordinary campaign members are still validated from their graph ownership.
    for issue in campaign_rows:
        bead_id = str(issue["id"])
        if bead_id in controls:
            continue
        labels = set(_labels(issue, findings))
        metadata = _metadata(issue, findings)
        workstreams = _suffixes(issue, "workstream:")
        roles = _suffixes(issue, "campaign-role:")
        timings = _suffixes(issue, "timing:")
        is_batch_claim = _metadata_text(issue, "campaign_role", findings) == "batch" or "campaign-role:batch" in labels
        if _metadata_text(issue, "campaign_id", findings) != campaign_id:
            findings.append(
                Finding(
                    "campaign-missing-binding", bead_id, "campaign row metadata campaign_id is missing or inconsistent"
                )
            )
        if not is_batch_claim and (not workstreams or not workstreams <= WORKSTREAMS):
            findings.append(
                Finding("campaign-missing-binding", bead_id, "campaign row has no valid workstream binding")
            )

        if is_batch_claim:
            parent_ids = _parent_targets(issue)
            if len(parent_ids) != 1:
                findings.append(
                    Finding(
                        "campaign-batch-binding", bead_id, "native batch step must have exactly one parent-child edge"
                    )
                )
            parent = by_id.get(parent_ids[0]) if len(parent_ids) == 1 else None
            if parent is None or parent.get("issue_type") != "molecule" or parent.get("title") != BATCH_FORMULA:
                findings.append(
                    Finding(
                        "campaign-batch-binding", bead_id, "batch step is not attached to a native poured formula root"
                    )
                )
            if set(labels) != BATCH_LABELS:
                findings.append(
                    Finding(
                        "campaign-batch-binding",
                        bead_id,
                        "batch labels must be fixed native labels; dynamic substitutions are unsupported",
                    )
                )
            if roles != {"batch"} or timings != {"execution"}:
                findings.append(Finding("campaign-batch-binding", bead_id, "batch role/timing labels are invalid"))
            if (
                _metadata_text(issue, "campaign_role", findings) != "batch"
                or _metadata_text(issue, "campaign_timing", findings) != "execution"
            ):
                findings.append(Finding("campaign-batch-binding", bead_id, "batch role/timing metadata is invalid"))
            if (
                _metadata_text(issue, "formula", findings) != BATCH_FORMULA
                or _metadata_integer(issue, "formula_version", findings) != BATCH_FORMULA_VERSION
                or _metadata_text(issue, "molecule_type", findings) != "workflow"
                or _metadata_text(issue, "pour_origin", findings) != "native-formula"
            ):
                findings.append(
                    Finding("campaign-batch-binding", bead_id, "batch lacks typed native formula provenance")
                )
        elif bead_id not in owned:
            findings.append(
                Finding("campaign-extra-unowned", bead_id, "campaign-labelled row is not owned by a workstream edge")
            )
        else:
            expected = owned[bead_id]
            if workstreams != expected:
                findings.append(
                    Finding(
                        "campaign-workstream-edge", bead_id, f"labels={sorted(workstreams)} edges={sorted(expected)}"
                    )
                )
            membership_source = _metadata_text(issue, "campaign_membership_source", findings)
            source_attachment = _metadata_text(issue, "source_attachment_sha256", findings)
            root_attachment = _metadata_text(root, "source_control_plane_sha256", findings)
            # A missing/empty anchor is never an attachment.  In particular,
            # ``None == None`` must not promote a row to native control when
            # both the candidate and root have lost their provenance.
            is_native_attachment = (
                isinstance(source_attachment, str)
                and bool(source_attachment.strip())
                and isinstance(root_attachment, str)
                and bool(root_attachment.strip())
                and source_attachment == root_attachment
            )
            is_native_control = membership_source == "native-control-plane" or (
                is_native_attachment and roles == {"implementation"} and timings == {"prep"}
            )
            if not is_native_control:
                if len(roles) != 1 or len(timings) < 1:
                    findings.append(
                        Finding("campaign-role-timing", bead_id, "member role/timing labels are missing or ambiguous")
                    )
                else:
                    if _metadata_text(issue, "campaign_role", findings) != next(iter(roles)):
                        findings.append(
                            Finding("campaign-role-timing", bead_id, "campaign_role metadata disagrees with labels")
                        )
                    if _metadata_set(issue, "campaign_timing", findings) != timings:
                        findings.append(
                            Finding("campaign-role-timing", bead_id, "campaign_timing metadata disagrees with labels")
                        )
                if _metadata_set(issue, "campaign_workstreams", findings) != workstreams:
                    findings.append(
                        Finding("campaign-role-timing", bead_id, "campaign_workstreams metadata disagrees with labels")
                    )
            if (not isinstance(membership_source, str) or not membership_source) and not is_native_attachment:
                findings.append(
                    Finding("campaign-role-timing", bead_id, "campaign_membership_source metadata is missing")
                )
            if is_native_control:
                if roles != {"implementation"} or timings != {"prep"}:
                    findings.append(
                        Finding(
                            "campaign-role-timing", bead_id, "native control labels must identify implementation prep"
                        )
                    )
                continue
            membership_kind = _metadata_text(issue, "campaign_membership_kind", findings)
            claims_agentctl = (
                membership_source
                in {
                    "agentctl:staged-native-adapter",
                    "agentctl:production-native-adapter",
                }
                or metadata.get("native_control_ids") is not None
            )
            if membership_kind is not None and membership_kind not in {
                "historical-roster",
                "staged-adapter",
                "production-adapter",
            }:
                findings.append(Finding("campaign-role-timing", bead_id, "campaign_membership_kind is invalid"))
            if claims_agentctl and (membership_kind, membership_source) not in CAMPAIGN_ADAPTER_PROVENANCE:
                findings.append(
                    Finding(
                        "campaign-agentctl-provenance",
                        bead_id,
                        "AgentCTL provenance kind and source must describe the same lifecycle stage",
                    )
                )
            if (membership_kind, membership_source) in CAMPAIGN_ADAPTER_PROVENANCE:
                workstream = next(iter(workstreams), "")
                findings.extend(
                    _staged_adapter_findings(issue, by_id=by_id, root_id=root_id, epic_id=epics.get(workstream, ""))
                )

    # Every graph-owned member must carry exactly the workstream labels and campaign binding.
    for bead_id, workstreams in sorted(owned.items()):
        issue = by_id[bead_id]
        labels = _suffixes(issue, "workstream:", findings)
        if labels != workstreams:
            findings.append(
                Finding("campaign-workstream-edge", bead_id, f"labels={sorted(labels)} edges={sorted(workstreams)}")
            )
        if campaign_label not in set(_labels(issue, findings)):
            findings.append(Finding("campaign-missing-binding", bead_id, "workstream-owned row lacks campaign label"))

    # Group every native formula molecule by its actual parent edge, then require
    # one complete stage set, one identical authority set, and a complete DAG.
    dag_findings, groups = _batch_dag_findings(by_id)
    findings.extend(dag_findings)
    complete_groups: set[str] = set()
    for molecule_id, group in sorted(groups.items()):
        bindings = [_binding_fields(issue) for issue in group]
        required = {"batch", "workstream", "authoritative_beads"}
        if any(set(binding) < required for binding in bindings):
            findings.append(
                Finding("campaign-batch-binding", molecule_id, "every stage needs complete native binding markers")
            )
            continue
        signatures = {(binding["batch"], binding["workstream"], binding["authoritative_beads"]) for binding in bindings}
        if len(signatures) != 1:
            findings.append(
                Finding("campaign-batch-binding", molecule_id, "batch stages do not share one identical binding")
            )
            continue
        batch_name, workstream, authorities = next(iter(signatures))
        if not BATCH_NAME_RE.fullmatch(batch_name):
            findings.append(
                Finding(
                    "campaign-batch-binding",
                    molecule_id,
                    "batch marker must match the native formula batch variable pattern",
                )
            )
        members, error = _parse_authoritative_beads(authorities)
        if error:
            findings.append(Finding("campaign-batch-binding", molecule_id, error))
        elif workstream not in WORKSTREAMS:
            findings.append(Finding("campaign-batch-binding", molecule_id, "batch workstream marker is invalid"))
        elif not members <= set(owned) or any(not (owned[member] & {workstream}) for member in members):
            findings.append(
                Finding(
                    "campaign-batch-binding", molecule_id, "authority set is outside graph-owned workstream membership"
                )
            )
        elif not any(
            finding.bead_id == molecule_id and finding.kind.startswith("campaign-batch") for finding in dag_findings
        ):
            complete_groups.add(molecule_id)

    active_groups = [
        molecule_id
        for molecule_id in complete_groups
        if any(_status(issue, findings) in {"open", "in_progress"} for issue in groups[molecule_id])
    ]
    active_by_workstream: dict[str, set[str]] = defaultdict(set)
    implementation_lanes = 0
    merge_ready = 0
    for molecule_id in active_groups:
        merge_stage = next(
            (issue for issue in groups[molecule_id] if _metadata_text(issue, "stage", findings) == "merge"), None
        )
        for issue in groups[molecule_id]:
            workstream = _binding_fields(issue).get("workstream", "")
            if workstream:
                active_by_workstream[workstream].add(molecule_id)
            if _status(issue, findings) == "in_progress" and _metadata_text(issue, "stage", findings) == "implement":
                implementation_lanes += 1
        verify_stage = next(
            (issue for issue in groups[molecule_id] if _metadata_text(issue, "stage", findings) == "verify"), None
        )
        gate = next(
            (
                issue
                for issue in by_id.values()
                if _issue_type(issue, findings) == "gate" and molecule_id in _parent_targets(issue)
            ),
            None,
        )
        if (
            merge_stage is not None
            and _status(merge_stage, findings) in MERGE_READY_STATUSES
            and verify_stage is not None
            and _status(verify_stage, findings) == "closed"
            and gate is not None
            and _status(gate, findings) == "closed"
        ):
            merge_ready += 1
    implementation_limit = _wip_limit(root_metadata, "implementation_lane_wip", findings, root_id)
    merge_limit = _wip_limit(root_metadata, "merge_train_wip", findings, root_id)
    batch_limit = _wip_limit(root_metadata, "workstream_active_batch_wip", findings, root_id)
    for workstream, batches in sorted(active_by_workstream.items()):
        if len(batches) > batch_limit:
            findings.append(Finding("campaign-active-batch-wip", workstream, f"active batches={sorted(batches)}"))
    if implementation_lanes > implementation_limit:
        findings.append(
            Finding("campaign-implementation-lane-wip", root_id, f"active implementation lanes={implementation_lanes}")
        )
    if merge_ready > merge_limit:
        findings.append(Finding("campaign-merge-train-wip", root_id, f"merge-ready trains={merge_ready}"))
    return findings


def _validated_issues(payload: object, *, source: str) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise RuntimeError(f"{source} returned {type(payload).__name__}, expected list")
    issues: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, issue in enumerate(payload):
        if not isinstance(issue, dict):
            raise RuntimeError(f"{source} record {index} is {type(issue).__name__}, expected object")
        bead_id = issue.get("id")
        if not isinstance(bead_id, str) or not bead_id:
            raise RuntimeError(f"{source} record {index} has no non-empty string id")
        if bead_id in seen:
            raise RuntimeError(f"{source} contains duplicate issue id {bead_id!r}")
        seen.add(bead_id)
        issues.append(issue)
    return issues


def _run_bd_dep_cycles() -> tuple[bool, str]:
    result = subprocess.run(["bd", "dep", "cycles"], capture_output=True, text=True, check=False)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output.strip()


def _run_bd_list_all() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["bd", "list", "--all", "--include-gates", "-n", "0", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    return _validated_issues(json.loads(result.stdout), source="bd list")


def _load_export(path: Path) -> list[dict[str, Any]]:
    records: list[object] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
    return _validated_issues(records, source=str(path))


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_revision(revision: str) -> str:
    if not revision or revision.startswith("-"):
        raise RuntimeError("revision must be a non-option Git revision")
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"unable to resolve revision {revision!r}")
    return result.stdout.strip()


def _git_blob(revision: str, path: str) -> bytes:
    result = subprocess.run(["git", "show", f"{revision}:{path}"], capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"unable to read {path!r} at revision {revision!r}")
    return result.stdout


def _parse_jsonl_bytes(payload: bytes, *, source: str) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"{source} is not UTF-8") from exc
    records: list[object] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{source}:{line_number}: invalid JSON: {exc.msg}") from exc
    return _validated_issues(records, source=source)


def _load_revision_export(revision: str) -> tuple[str, bytes, list[dict[str, Any]]]:
    pinned_revision = _git_revision(revision)
    payload = _git_blob(pinned_revision, ".beads/issues.jsonl")
    return pinned_revision, payload, _parse_jsonl_bytes(payload, source=f"{pinned_revision}:.beads/issues.jsonl")


def _load_campaign_genesis(revision: str) -> dict[str, Any]:
    payload = _git_blob(revision, CAMPAIGN_GENESIS_PATH)
    try:
        genesis = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"campaign genesis is invalid JSON: {exc.msg}") from exc
    if not isinstance(genesis, dict):
        raise RuntimeError("campaign genesis must be an object")
    expected = {"schema", "campaign_id", "input_snapshot", "migration_snapshot", "formula_snapshot"}
    if set(genesis) != expected:
        raise RuntimeError("campaign genesis has an unexpected schema")
    if genesis.get("schema") != CAMPAIGN_GENESIS_SCHEMA or genesis.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError("campaign genesis identity is invalid")
    for key in ("input_snapshot", "migration_snapshot", "formula_snapshot"):
        snapshot = genesis.get(key)
        if not isinstance(snapshot, dict) or set(snapshot) != {"revision", "path", "sha256"}:
            raise RuntimeError(f"campaign genesis {key} must name revision, path, and sha256")
        if not all(isinstance(snapshot.get(field), str) and snapshot[field] for field in snapshot):
            raise RuntimeError(f"campaign genesis {key} contains an invalid value")
        if not SHA256_RE.fullmatch(snapshot["sha256"]):
            raise RuntimeError(f"campaign genesis {key} sha256 is invalid")
    return genesis


def _snapshot_from_genesis(genesis: dict[str, Any], key: str) -> tuple[str, str, bytes]:
    snapshot = genesis[key]
    assert isinstance(snapshot, dict)
    revision = _git_revision(str(snapshot["revision"]))
    path = str(snapshot["path"])
    payload = _git_blob(revision, path)
    digest = _sha256_bytes(payload)
    if digest != snapshot["sha256"]:
        raise RuntimeError(f"campaign genesis {key} digest does not match its pinned Git object")
    return revision, path, payload


def _record_digest(record: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(record))


def _derive_campaign_migration(genesis: dict[str, Any]) -> dict[str, Any]:
    """Derive the reviewed native migration from its two immutable inputs."""
    input_revision, input_path, input_payload = _snapshot_from_genesis(genesis, "input_snapshot")
    migration_revision, migration_path, migration_payload = _snapshot_from_genesis(genesis, "migration_snapshot")
    formula_revision, formula_path, formula_payload = _snapshot_from_genesis(genesis, "formula_snapshot")
    input_rows = _parse_jsonl_bytes(input_payload, source=f"{input_revision}:{input_path}")
    migration_rows = _parse_jsonl_bytes(migration_payload, source=f"{migration_revision}:{migration_path}")
    input_by_id = {str(row["id"]): row for row in input_rows}
    migration_by_id = {str(row["id"]): row for row in migration_rows}
    changed = [
        {
            "id": bead_id,
            "before_sha256": _record_digest(input_by_id[bead_id]) if bead_id in input_by_id else None,
            "after_sha256": _record_digest(migration_by_id[bead_id]) if bead_id in migration_by_id else None,
        }
        for bead_id in sorted(set(input_by_id) | set(migration_by_id))
        if input_by_id.get(bead_id) != migration_by_id.get(bead_id)
    ]
    return {
        "schema": "polylogue.campaign-derivation/v1",
        "campaign_id": CAMPAIGN_ID,
        "input": {"revision": input_revision, "path": input_path, "sha256": _sha256_bytes(input_payload)},
        "migration": {
            "revision": migration_revision,
            "path": migration_path,
            "sha256": _sha256_bytes(migration_payload),
        },
        "formula": {"revision": formula_revision, "path": formula_path, "sha256": _sha256_bytes(formula_payload)},
        "changed_record_count": len(changed),
        "changed_records_sha256": _sha256_bytes(_canonical_json_bytes(changed)),
    }


def _molecule_records(issues: list[dict[str, Any]], molecule_id: str) -> list[dict[str, Any]]:
    return [issue for issue in issues if str(issue["id"]) == molecule_id or molecule_id in _parent_targets(issue)]


def _validate_pour_receipts(revision: str, issues: list[dict[str, Any]]) -> list[Finding]:
    """Bind every formula molecule to one immutable native-pour receipt."""
    findings: list[Finding] = []
    molecules = [
        issue for issue in issues if _issue_type(issue, findings) == "molecule" and issue.get("title") == BATCH_FORMULA
    ]
    for molecule in molecules:
        molecule_id = str(molecule["id"])
        receipt_path = f"{CAMPAIGN_POUR_RECEIPT_DIRECTORY}/{molecule_id}.json"
        try:
            receipt_payload = _git_blob(revision, receipt_path)
            receipt = json.loads(receipt_payload)
        except (RuntimeError, json.JSONDecodeError):
            findings.append(Finding("campaign-pour-receipt", molecule_id, "immutable native-pour receipt is absent"))
            continue
        if not isinstance(receipt, dict):
            findings.append(Finding("campaign-pour-receipt", molecule_id, "native-pour receipt must be an object"))
            continue
        expected = {
            "schema",
            "campaign_id",
            "molecule_id",
            "formula",
            "native_pour_output_sha256",
            "poured_records_sha256",
        }
        if set(receipt) != expected or receipt.get("schema") != CAMPAIGN_POUR_RECEIPT_SCHEMA:
            findings.append(Finding("campaign-pour-receipt", molecule_id, "native-pour receipt schema is invalid"))
            continue
        formula = receipt.get("formula")
        if (
            receipt.get("campaign_id") != CAMPAIGN_ID
            or receipt.get("molecule_id") != molecule_id
            or not isinstance(formula, dict)
        ):
            findings.append(Finding("campaign-pour-receipt", molecule_id, "native-pour receipt identity is invalid"))
            continue
        if set(formula) != {"revision", "path", "sha256"} or not all(
            isinstance(formula.get(field), str) for field in formula
        ):
            findings.append(Finding("campaign-pour-receipt", molecule_id, "native-pour formula evidence is invalid"))
            continue
        if not all(
            isinstance(receipt.get(field), str) and SHA256_RE.fullmatch(receipt[field])
            for field in ("native_pour_output_sha256", "poured_records_sha256")
        ):
            findings.append(Finding("campaign-pour-receipt", molecule_id, "native-pour receipt hashes are invalid"))
            continue
        try:
            formula_payload = _git_blob(_git_revision(formula["revision"]), formula["path"])
        except RuntimeError:
            findings.append(Finding("campaign-pour-receipt", molecule_id, "receipt formula object is unreadable"))
            continue
        if _sha256_bytes(formula_payload) != formula["sha256"]:
            findings.append(Finding("campaign-pour-receipt", molecule_id, "receipt formula digest is mismatched"))
            continue
        records = sorted(_molecule_records(issues, molecule_id), key=lambda record: str(record["id"]))
        if _sha256_bytes(_canonical_json_bytes(records)) != receipt["poured_records_sha256"]:
            findings.append(
                Finding("campaign-pour-receipt", molecule_id, "receipt does not bind the accepted molecule")
            )
    return findings


def _acceptance_digest(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("acceptance_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(unsigned))


def _build_campaign_acceptance(
    revision: str, snapshot_payload: bytes, issues: list[dict[str, Any]], report: dict[str, Any]
) -> dict[str, Any]:
    genesis = _load_campaign_genesis(revision)
    derivation = _derive_campaign_migration(genesis)
    receipt_findings = _validate_pour_receipts(revision, issues)
    payload: dict[str, Any] = {
        "schema": CAMPAIGN_ACCEPTANCE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "snapshot": {
            "revision": revision,
            "path": ".beads/issues.jsonl",
            "sha256": _sha256_bytes(snapshot_payload),
        },
        "genesis_sha256": _sha256_bytes(_canonical_json_bytes(genesis)),
        "derivation": derivation,
        # The CLI attaches this witness to ``report`` after it is finalized.
        # Serialize a detached graph snapshot so that attachment cannot make
        # the witness self-referential.
        "bead_graph": json.loads(json.dumps(report, sort_keys=True)),
        "findings": [{"kind": item.kind, "id": item.bead_id, "detail": item.detail} for item in receipt_findings],
    }
    payload["acceptance_sha256"] = _acceptance_digest(payload)
    return payload


def _write_acceptance_output(path: Path, payload: dict[str, Any]) -> None:
    digest = payload.get("acceptance_sha256")
    expected_name = f"{CAMPAIGN_ID}-acceptance-{digest}.json"
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest) or path.name != expected_name:
        raise RuntimeError(f"acceptance output must be named {expected_name!r}")
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"existing acceptance output is unreadable: {path}") from exc
        if existing != payload:
            raise RuntimeError(f"refusing to overwrite immutable acceptance output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_acceptance_output(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"acceptance output is unreadable: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != CAMPAIGN_ACCEPTANCE_SCHEMA:
        raise RuntimeError("acceptance output schema is invalid")
    digest = payload.get("acceptance_sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest) or _acceptance_digest(payload) != digest:
        raise RuntimeError("acceptance output digest is invalid")
    if path.name != f"{CAMPAIGN_ID}-acceptance-{digest}.json":
        raise RuntimeError("acceptance output filename does not bind its digest")


def _dependency_records(issue: dict[str, Any]) -> list[dict[str, Any]]:
    dependencies = issue.get("dependencies")
    return (
        [dependency for dependency in dependencies if isinstance(dependency, dict)]
        if isinstance(dependencies, list)
        else []
    )


def canonical_parent_map(issues: list[dict[str, Any]]) -> dict[str, str | None]:
    parents: dict[str, str | None] = {}
    for issue in issues:
        targets = _parent_targets(issue)
        parents[str(issue["id"])] = targets[0] if len(targets) == 1 else None
    return parents


def _cycle_findings(edges: dict[str, set[str]], *, kind: str, label: str) -> list[Finding]:
    findings: list[Finding] = []
    state: dict[str, int] = {}
    path: list[str] = []
    positions: dict[str, int] = {}
    reported: set[frozenset[str]] = set()

    def visit(node: str) -> None:
        state[node] = 1
        positions[node] = len(path)
        path.append(node)
        for target in sorted(edges.get(node, set())):
            if state.get(target, 0) == 0:
                visit(target)
            elif state.get(target) == 1:
                cycle = path[positions[target] :] + [target]
                identity = frozenset(cycle)
                if identity not in reported:
                    findings.append(Finding(kind, target, f"{label}: " + " -> ".join(cycle)))
                    reported.add(identity)
        path.pop()
        positions.pop(node)
        state[node] = 2

    for start in sorted(edges):
        if state.get(start, 0) == 0:
            visit(start)
    return findings


def collect_findings(issues: list[dict[str, Any]]) -> list[Finding]:
    by_id = {str(issue["id"]): issue for issue in issues}
    findings: list[Finding] = []
    for issue in issues:
        bead_id = str(issue.get("id", "<unknown>"))
        _labels(issue, findings)
        _metadata(issue, findings)
        status = issue.get("status")
        if status is not None and (not isinstance(status, str) or status not in KNOWN_STATUSES):
            findings.append(Finding("malformed-status", bead_id, "status must be a known Beads status string"))
        for field in ("issue_type", "title", "description"):
            value = issue.get(field)
            if value is not None and not isinstance(value, str):
                findings.append(Finding(f"malformed-{field.replace('_', '-')}", bead_id, f"{field} must be a string"))
    parent_edges: dict[str, set[str]] = {}
    block_edges: dict[str, set[str]] = defaultdict(set)

    for bead_id, issue in sorted(by_id.items()):
        raw_dependencies = issue.get("dependencies")
        if raw_dependencies is not None and not isinstance(raw_dependencies, list):
            findings.append(Finding("malformed-dependencies", bead_id, "dependencies must be a list"))
            continue
        seen_edges: set[tuple[str, str]] = set()
        parents: list[str] = []
        for index, dependency in enumerate(raw_dependencies or []):
            if not isinstance(dependency, dict):
                findings.append(Finding("malformed-dependency", bead_id, f"dependency {index} is not an object"))
                continue
            dep_type = dependency.get("type")
            target = dependency.get("depends_on_id")
            if not isinstance(dep_type, str) or not dep_type or not isinstance(target, str) or not target:
                findings.append(Finding("malformed-dependency", bead_id, f"dependency {index} lacks type or target"))
                continue
            edge = (dep_type, target)
            if edge in seen_edges:
                findings.append(Finding("duplicate-dependency", bead_id, f"duplicate {dep_type} edge to {target}"))
            seen_edges.add(edge)
            if dep_type not in DEPENDENCY_KINDS:
                findings.append(Finding("unknown-dependency-kind", bead_id, f"unknown dependency kind {dep_type!r}"))
            if target not in by_id:
                findings.append(
                    Finding("missing-dependency-target", bead_id, f"{dep_type} target {target!r} does not exist")
                )
            if target == bead_id:
                findings.append(Finding("self-dependency", bead_id, f"{dep_type} edge targets itself"))
            if dep_type == "parent-child":
                parents.append(target)
            elif dep_type == "blocks":
                block_edges[bead_id].add(target)
        if len(parents) > 1:
            findings.append(Finding("multiple-parents", bead_id, f"parent-child targets={sorted(parents)}"))
        elif parents:
            parent_edges[bead_id] = {parents[0]}

    findings.extend(_cycle_findings(parent_edges, kind="parent-cycle", label="parent-child cycle"))
    findings.extend(_cycle_findings(block_edges, kind="blocks-cycle", label="blocks cycle"))
    return sorted(findings, key=lambda finding: (finding.kind, finding.bead_id, finding.detail))


def _graph_digest(issues: list[dict[str, Any]]) -> str:
    records = [
        {
            "id": issue["id"],
            "status": issue.get("status"),
            # Preserve malformed records in the digest without asking Python
            # to order values of unrelated JSON types (for example None and
            # str). Structural findings below remain the authority on their
            # validity.
            "dependencies": sorted(
                json.dumps(dependency, sort_keys=True, separators=(",", ":"))
                for dependency in _dependency_records(issue)
            ),
        }
        for issue in sorted(issues, key=lambda item: str(item["id"]))
    ]
    return hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _forcing_report(issues: list[dict[str, Any]], root: str, *, graph_sha256: str) -> dict[str, Any]:
    by_id = {str(issue["id"]): issue for issue in issues}
    if root not in by_id:
        raise RuntimeError(f"forcing root {root!r} does not exist")
    pending = [root]
    blockers: set[str] = set()
    while pending:
        bead_id = pending.pop()
        for dependency in _dependency_records(by_id[bead_id]):
            if dependency.get("type") != "blocks":
                continue
            target = dependency.get("depends_on_id")
            if isinstance(target, str) and target and target in by_id and target not in blockers:
                blockers.add(target)
                pending.append(target)
    blocker_ids = sorted(blockers)
    statuses = {bead_id: str(by_id[bead_id].get("status", "unknown")) for bead_id in blocker_ids}
    status_counts: dict[str, int] = defaultdict(int)
    for status in statuses.values():
        status_counts[status] += 1
    unresolved_ids = sorted(bead_id for bead_id, status in statuses.items() if status != "closed")
    forcing_payload = {
        "root_bead_id": root,
        "graph_sha256": graph_sha256,
        "blocker_ids": blocker_ids,
        "statuses": statuses,
    }
    return {
        "root_bead_id": root,
        "forcing_sha256": hashlib.sha256(
            json.dumps(forcing_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "blocker_ids": blocker_ids,
        "status_counts": dict(sorted(status_counts.items())),
        "unresolved_ids": unresolved_ids,
        "resolved": not unresolved_ids,
    }


def _registry_findings(issues: list[dict[str, Any]]) -> list[Finding]:
    """Cross-check existing runtime registries against one immutable Beads population."""
    from polylogue.maintenance.archive_verification import (
        ARCHIVE_VERIFICATION_CHECKS,
        validate_archive_verification_registry,
    )
    from polylogue.maintenance.live_proof import LIVE_PROOF_SPECS, validate_live_proof_registry

    status_by_bead = {str(issue["id"]): str(issue.get("status", "unknown")) for issue in issues}
    findings: list[Finding] = []
    try:
        validate_archive_verification_registry(waiver_bead_statuses=status_by_bead)
    except ValueError as exc:
        findings.append(Finding("archive-verification-registry", "registry", str(exc)))
    try:
        validate_live_proof_registry()
    except ValueError as exc:
        findings.append(Finding("live-proof-registry", "registry", str(exc)))
    for archive_spec in ARCHIVE_VERIFICATION_CHECKS:
        if archive_spec.incident is not None and archive_spec.incident.bead_id not in status_by_bead:
            findings.append(Finding("unknown-incident-bead", archive_spec.name, archive_spec.incident.bead_id))
    for proof_spec in LIVE_PROOF_SPECS:
        if proof_spec.bead_id not in status_by_bead:
            findings.append(Finding("unknown-live-proof-bead", proof_spec.proof_id.value, proof_spec.bead_id))
    return findings


def build_report(
    issues: list[dict[str, Any],],
    *,
    cycles_ok: bool,
    cycles_output: str,
    forcing_roots: list[str] | None = None,
) -> dict[str, Any]:
    findings = collect_findings(issues)
    if _campaign_marker_present(issues, campaign_id=CAMPAIGN_ID, root_id=CAMPAIGN_ROOT):
        findings.extend(collect_campaign_findings(issues))
    findings.extend(_registry_findings(issues))
    findings.sort(key=lambda finding: (finding.kind, finding.bead_id, finding.detail))
    structured_cycles_ok = not any(finding.kind in {"parent-cycle", "blocks-cycle"} for finding in findings)
    counts: dict[str, int] = defaultdict(int)
    for finding in findings:
        counts[finding.kind] += 1
    graph_sha256 = _graph_digest(issues)
    forcing = [_forcing_report(issues, root, graph_sha256=graph_sha256) for root in sorted(set(forcing_roots or []))]
    return {
        "report_version": 3,
        "cycles": {"ok": cycles_ok and structured_cycles_ok, "output": cycles_output},
        "issues_scanned": len(issues),
        "graph_sha256": graph_sha256,
        "dependency_kind_counts": {
            kind: sum(
                1 for issue in issues for dependency in _dependency_records(issue) if dependency.get("type") == kind
            )
            for kind in sorted(DEPENDENCY_KINDS)
        },
        "forcing": forcing,
        "findings": [{"kind": f.kind, "id": f.bead_id, "detail": f.detail} for f in findings],
        "counts": dict(sorted(counts.items())),
    }


def _format_report(report: dict[str, Any]) -> str:
    lines = [str(report["cycles"]["output"])] if report["cycles"]["output"] else []
    lines.extend(f"{item['kind']}: {item['id']} {item['detail']}" for item in report["findings"])
    lines.append(f"bead-graph: {report['issues_scanned']} issues, {len(report['findings'])} structural violations")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a machine-readable structural report")
    parser.add_argument(
        "--export", type=Path, help="validate a JSONL export without touching the shared live Beads database"
    )
    parser.add_argument("--revision", help="validate the exact Git revision of .beads/issues.jsonl without invoking bd")
    parser.add_argument(
        "--acceptance-output",
        type=Path,
        help="write one immutable digest-named campaign acceptance witness for --revision",
    )
    parser.add_argument(
        "--verify-acceptance-output", type=Path, help="verify an existing digest-named campaign acceptance witness"
    )
    parser.add_argument(
        "--forcing-root", action="append", default=[], help="Bead ID whose transitive blocks closure to report"
    )
    parser.add_argument(
        "--require-resolved", action="store_true", help="fail when a requested forcing closure has non-closed blockers"
    )
    args = parser.parse_args(argv)

    try:
        if args.verify_acceptance_output is not None:
            if args.export is not None or args.revision is not None or args.acceptance_output is not None:
                raise RuntimeError("--verify-acceptance-output cannot be combined with snapshot inputs")
            _verify_acceptance_output(args.verify_acceptance_output)
            report = {
                "report_version": 3,
                "acceptance_output": str(args.verify_acceptance_output),
                "verified": True,
                "findings": [],
                "forcing": [],
            }
        elif args.export is not None:
            if args.revision is not None or args.acceptance_output is not None:
                raise RuntimeError("--export cannot be combined with revision-pinned acceptance")
            issues = _load_export(args.export)
            cycles_ok, cycles_output = True, ""
            report = build_report(
                issues, cycles_ok=cycles_ok, cycles_output=cycles_output, forcing_roots=args.forcing_root
            )
        elif args.revision is not None:
            revision, snapshot_payload, issues = _load_revision_export(args.revision)
            cycles_ok, cycles_output = True, ""
            report = build_report(
                issues, cycles_ok=cycles_ok, cycles_output=cycles_output, forcing_roots=args.forcing_root
            )
            if args.acceptance_output is not None:
                acceptance = _build_campaign_acceptance(revision, snapshot_payload, issues, report)
                _write_acceptance_output(args.acceptance_output, acceptance)
                report["campaign_acceptance"] = acceptance
                graph_findings = report.get("findings")
                acceptance_findings = acceptance.get("findings")
                if not isinstance(graph_findings, list) or not isinstance(acceptance_findings, list):
                    raise RuntimeError("campaign acceptance findings are malformed")
                merged_findings = sorted(
                    [*graph_findings, *acceptance_findings],
                    key=lambda item: (item["kind"], item["id"], item["detail"]),
                )
                report["findings"] = merged_findings
                counts: dict[str, int] = defaultdict(int)
                for item in merged_findings:
                    counts[item["kind"]] += 1
                report["counts"] = dict(sorted(counts.items()))
        else:
            if args.acceptance_output is not None:
                raise RuntimeError("--acceptance-output requires --revision")
            cycles_ok, cycles_output = _run_bd_dep_cycles()
            if not cycles_ok:
                raise RuntimeError(f"dependency cycle check failed: {cycles_output}")
            issues = _run_bd_list_all()
            report = build_report(
                issues, cycles_ok=cycles_ok, cycles_output=cycles_output, forcing_roots=args.forcing_root
            )
    except (OSError, subprocess.CalledProcessError, RuntimeError, json.JSONDecodeError) as exc:
        payload = {"report_version": 3, "error": str(exc)}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"bead-graph: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.verify_acceptance_output is not None:
        print(f"campaign acceptance output verified: {args.verify_acceptance_output}")
    else:
        print(_format_report(report))
    forcing = report.get("forcing")
    if not isinstance(forcing, list):
        raise RuntimeError("bead-graph report forcing payload is malformed")
    unresolved = [str(item["root_bead_id"]) for item in forcing if not item["resolved"]]
    if args.require_resolved and unresolved:
        print(f"bead-graph: unresolved forcing blockers for {', '.join(unresolved)}", file=sys.stderr)
        return 1
    return 0 if not report["findings"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

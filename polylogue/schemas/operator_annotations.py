"""Schema-annotation summarization helpers for operator workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from polylogue.lib.json import JSONDocument, JSONValue, json_document
from polylogue.schemas.operator_models import (
    JSONDocument as OperatorJSONDocument,
)
from polylogue.schemas.operator_models import (
    JSONDocumentList as OperatorJSONDocumentList,
)
from polylogue.schemas.operator_models import (
    SchemaAnnotationSummary,
    SchemaCoverageSummary,
    SchemaReviewProof,
    SchemaRoleAssignment,
    SchemaRoleProofEntry,
)


@dataclass(frozen=True, slots=True)
class _RoleCandidate:
    path: str
    score: float
    evidence: OperatorJSONDocument

    def to_payload(self) -> OperatorJSONDocument:
        return {
            "path": self.path,
            "score": self.score,
            "evidence": self.evidence,
        }


def _schema_node(value: JSONValue | object) -> JSONDocument | None:
    return value if isinstance(value, dict) else None


def _node_evidence(node: Mapping[str, object]) -> OperatorJSONDocument:
    return cast(OperatorJSONDocument, json_document(node.get("x-polylogue-evidence")))


def _score_value(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, str)]


def collect_annotation_summary(schema: Mapping[str, object]) -> SchemaAnnotationSummary:
    """Collect format/value/semantic coverage from a schema document."""
    schema_node = json_document(schema)
    format_count = 0
    values_count = 0
    total_enum_values = 0
    role_by_key: dict[tuple[str, str], SchemaRoleAssignment] = {}
    total_fields = 0
    with_format = 0
    with_values = 0
    with_role = 0

    def visit(node: JSONDocument, *, path: str) -> None:
        nonlocal format_count, values_count, total_enum_values
        nonlocal total_fields, with_format, with_values, with_role
        role = node.get("x-polylogue-semantic-role")
        if role:
            assignment = SchemaRoleAssignment(
                path=path,
                role=str(role),
                confidence=_score_value(node.get("x-polylogue-score", 0.0)),
                evidence=_node_evidence(node),
            )
            key = (assignment.path, assignment.role)
            current = role_by_key.get(key)
            if current is None or assignment.confidence >= current.confidence:
                role_by_key[key] = assignment
        if "x-polylogue-format" in node:
            format_count += 1
        enum_values = node.get("x-polylogue-values")
        enum_list = _string_list(enum_values)
        if enum_list:
            values_count += 1
            total_enum_values += len(enum_list)

        properties = _schema_node(node.get("properties"))
        if properties is not None:
            for name, child in properties.items():
                child_node = _schema_node(child)
                if child_node is None:
                    continue
                total_fields += 1
                if "x-polylogue-format" in child_node:
                    with_format += 1
                if "x-polylogue-values" in child_node:
                    with_values += 1
                if "x-polylogue-semantic-role" in child_node:
                    with_role += 1
                visit(child_node, path=f"{path}.{name}")
        items = _schema_node(node.get("items"))
        if items is not None:
            visit(items, path=f"{path}[*]")
        additional_properties = _schema_node(node.get("additionalProperties"))
        if additional_properties is not None:
            visit(additional_properties, path=f"{path}.*")
        for keyword in ("anyOf", "oneOf", "allOf"):
            keyword_items = node.get(keyword)
            if isinstance(keyword_items, Sequence) and not isinstance(keyword_items, (str, bytes, bytearray)):
                for child in keyword_items:
                    child_node = _schema_node(child)
                    if child_node is not None:
                        visit(child_node, path=path)

    visit(schema_node, path="$")
    roles = sorted(role_by_key.values(), key=lambda item: (-item.confidence, item.path, item.role))
    return SchemaAnnotationSummary(
        semantic_count=len(roles),
        format_count=format_count,
        values_count=values_count,
        total_enum_values=total_enum_values,
        roles=roles,
        coverage=SchemaCoverageSummary(
            total_fields=total_fields,
            with_format=with_format,
            with_values=with_values,
            with_role=with_role,
        ),
    )


def build_review_proof(schema: Mapping[str, object]) -> SchemaReviewProof:
    """Build a proof surface from a schema's semantic annotations and field stats.

    Re-runs inference from the schema's own samples metadata to produce
    full candidate lists, competing paths, and abstention details.
    """
    from polylogue.schemas.semantic_inference_models import SEMANTIC_ROLES
    from polylogue.schemas.semantic_inference_runtime import (
        RECORD_STREAM_ELIGIBLE_ROLES,
        RECORD_STREAM_KINDS,
    )

    schema_node = json_document(schema)
    artifact_kind_value = schema_node.get("x-polylogue-artifact-kind")
    artifact_kind = artifact_kind_value if isinstance(artifact_kind_value, str) else None
    is_record_stream = artifact_kind in RECORD_STREAM_KINDS

    eligible_roles = list(SEMANTIC_ROLES)
    ineligible_roles: list[str] = []
    if is_record_stream:
        eligible_roles = [r for r in SEMANTIC_ROLES if r in RECORD_STREAM_ELIGIBLE_ROLES]
        ineligible_roles = [r for r in SEMANTIC_ROLES if r not in RECORD_STREAM_ELIGIBLE_ROLES]

    # Collect all role assignments from the schema itself
    role_entries: dict[str, list[_RoleCandidate]] = {}
    _collect_role_candidates_from_schema(schema_node, "$", role_entries)

    # Build proof entries for each semantic role
    proof_entries: list[SchemaRoleProofEntry] = []
    for role in SEMANTIC_ROLES:
        if role in ineligible_roles:
            proof_entries.append(
                SchemaRoleProofEntry(
                    role=role,
                    chosen_path=None,
                    chosen_score=0.0,
                    competing=[],
                    evidence={},
                    abstained=True,
                    abstain_reason=f"artifact_kind={artifact_kind} excludes {role}",
                )
            )
            continue

        candidates = role_entries.get(role, [])
        if not candidates:
            proof_entries.append(
                SchemaRoleProofEntry(
                    role=role,
                    chosen_path=None,
                    chosen_score=0.0,
                    competing=[],
                    evidence={},
                    abstained=True,
                    abstain_reason="no candidates scored above threshold",
                )
            )
            continue

        chosen = candidates[0]
        competing: OperatorJSONDocumentList = [candidate.to_payload() for candidate in candidates[1:]]
        proof_entries.append(
            SchemaRoleProofEntry(
                role=role,
                chosen_path=chosen.path,
                chosen_score=chosen.score,
                competing=competing,
                evidence=chosen.evidence,
                abstained=False,
            )
        )

    return SchemaReviewProof(
        roles=proof_entries,
        artifact_kind=artifact_kind,
        eligible_roles=eligible_roles,
        ineligible_roles=ineligible_roles,
    )


def _collect_role_candidates_from_schema(
    node: JSONDocument,
    path: str,
    role_entries: dict[str, list[_RoleCandidate]],
) -> None:
    """Walk a schema and collect all semantic role annotations with scores."""
    role = node.get("x-polylogue-semantic-role")
    if role:
        role_entries.setdefault(str(role), []).append(
            _RoleCandidate(
                path=path,
                score=_score_value(node.get("x-polylogue-score", 0.0)),
                evidence=_node_evidence(node),
            )
        )

    properties = _schema_node(node.get("properties"))
    if properties is not None:
        for name, child in properties.items():
            child_node = _schema_node(child)
            if child_node is not None:
                _collect_role_candidates_from_schema(child_node, f"{path}.{name}", role_entries)
    items = _schema_node(node.get("items"))
    if items is not None:
        _collect_role_candidates_from_schema(items, f"{path}[*]", role_entries)
    additional_properties = _schema_node(node.get("additionalProperties"))
    if additional_properties is not None:
        _collect_role_candidates_from_schema(
            additional_properties,
            f"{path}.*",
            role_entries,
        )
    for keyword in ("anyOf", "oneOf", "allOf"):
        keyword_items = node.get(keyword)
        if isinstance(keyword_items, Sequence) and not isinstance(keyword_items, (str, bytes, bytearray)):
            for child in keyword_items:
                child_node = _schema_node(child)
                if child_node is not None:
                    _collect_role_candidates_from_schema(child_node, path, role_entries)


__all__ = ["build_review_proof", "collect_annotation_summary"]

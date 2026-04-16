"""Schema-annotation summarization helpers for operator workflows."""

from __future__ import annotations

from polylogue.schemas.operator_models import (
    SchemaAnnotationSummary,
    SchemaCoverageSummary,
    SchemaReviewProof,
    SchemaRoleAssignment,
    SchemaRoleProofEntry,
)


def collect_annotation_summary(schema: dict) -> SchemaAnnotationSummary:
    """Collect format/value/semantic coverage from a schema document."""
    semantic_count = 0
    format_count = 0
    values_count = 0
    total_enum_values = 0
    roles: list[SchemaRoleAssignment] = []
    total_fields = 0
    with_format = 0
    with_values = 0
    with_role = 0

    def visit(node: dict, *, path: str) -> None:
        nonlocal semantic_count, format_count, values_count, total_enum_values
        nonlocal total_fields, with_format, with_values, with_role
        if not isinstance(node, dict):
            return
        role = node.get("x-polylogue-semantic-role")
        if role:
            semantic_count += 1
            roles.append(
                SchemaRoleAssignment(
                    path=path,
                    role=str(role),
                    confidence=float(node.get("x-polylogue-score", 0.0) or 0.0),
                    evidence=dict(node.get("x-polylogue-evidence", {})),
                )
            )
        if "x-polylogue-format" in node:
            format_count += 1
        if "x-polylogue-values" in node:
            values_count += 1
            total_enum_values += len(node["x-polylogue-values"])

        for name, child in node.get("properties", {}).items():
            if isinstance(child, dict):
                total_fields += 1
                if "x-polylogue-format" in child:
                    with_format += 1
                if "x-polylogue-values" in child:
                    with_values += 1
                if "x-polylogue-semantic-role" in child:
                    with_role += 1
                visit(child, path=f"{path}.{name}")
        if isinstance(node.get("items"), dict):
            visit(node["items"], path=f"{path}[*]")
        if isinstance(node.get("additionalProperties"), dict):
            visit(node["additionalProperties"], path=f"{path}.*")
        for keyword in ("anyOf", "oneOf", "allOf"):
            for child in node.get(keyword, []):
                if isinstance(child, dict):
                    visit(child, path=path)

    visit(schema, path="$")
    return SchemaAnnotationSummary(
        semantic_count=semantic_count,
        format_count=format_count,
        values_count=values_count,
        total_enum_values=total_enum_values,
        roles=sorted(roles, key=lambda item: (-item.confidence, item.path, item.role)),
        coverage=SchemaCoverageSummary(
            total_fields=total_fields,
            with_format=with_format,
            with_values=with_values,
            with_role=with_role,
        ),
    )


def build_review_proof(schema: dict) -> SchemaReviewProof:
    """Build a proof surface from a schema's semantic annotations and field stats.

    Re-runs inference from the schema's own samples metadata to produce
    full candidate lists, competing paths, and abstention details.
    """
    from polylogue.schemas.semantic_inference_models import SEMANTIC_ROLES
    from polylogue.schemas.semantic_inference_runtime import (
        RECORD_STREAM_ELIGIBLE_ROLES,
        RECORD_STREAM_KINDS,
    )

    artifact_kind = schema.get("x-polylogue-artifact-kind")
    is_record_stream = artifact_kind in RECORD_STREAM_KINDS

    eligible_roles = list(SEMANTIC_ROLES)
    ineligible_roles: list[str] = []
    if is_record_stream:
        eligible_roles = [r for r in SEMANTIC_ROLES if r in RECORD_STREAM_ELIGIBLE_ROLES]
        ineligible_roles = [r for r in SEMANTIC_ROLES if r not in RECORD_STREAM_ELIGIBLE_ROLES]

    # Collect all role assignments from the schema itself
    role_entries: dict[str, list[dict]] = {}  # role -> list of {path, score, evidence}
    _collect_role_candidates_from_schema(schema, "$", role_entries)

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
        competing = [{"path": c["path"], "score": c["score"], "evidence": c["evidence"]} for c in candidates[1:]]
        proof_entries.append(
            SchemaRoleProofEntry(
                role=role,
                chosen_path=chosen["path"],
                chosen_score=chosen["score"],
                competing=competing,
                evidence=chosen["evidence"],
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
    node: dict,
    path: str,
    role_entries: dict[str, list[dict]],
) -> None:
    """Walk a schema and collect all semantic role annotations with scores."""
    if not isinstance(node, dict):
        return
    role = node.get("x-polylogue-semantic-role")
    if role:
        score = float(node.get("x-polylogue-score", 0.0) or 0.0)
        evidence = dict(node.get("x-polylogue-evidence", {}))
        role_entries.setdefault(role, []).append(
            {
                "path": path,
                "score": score,
                "evidence": evidence,
            }
        )

    for name, child in node.get("properties", {}).items():
        if isinstance(child, dict):
            _collect_role_candidates_from_schema(child, f"{path}.{name}", role_entries)
    if isinstance(node.get("items"), dict):
        _collect_role_candidates_from_schema(node["items"], f"{path}[*]", role_entries)
    if isinstance(node.get("additionalProperties"), dict):
        _collect_role_candidates_from_schema(
            node["additionalProperties"],
            f"{path}.*",
            role_entries,
        )
    for keyword in ("anyOf", "oneOf", "allOf"):
        for child in node.get(keyword, []):
            if isinstance(child, dict):
                _collect_role_candidates_from_schema(child, path, role_entries)


__all__ = ["build_review_proof", "collect_annotation_summary"]

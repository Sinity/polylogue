"""Explain rendering for schema CLI commands."""

from __future__ import annotations

import json
from typing import Any

import click

from polylogue.cli.machine_errors import emit_success


def render_schema_explain_result(*, result: Any, json_output: bool, verbose: bool) -> None:
    """Render schema explain output in JSON or human-readable form."""
    if json_output:
        emit_success(result.to_dict())
        return

    if result.review_proof is not None:
        _render_proof_surface(result)
        return

    props = result.schema.get("properties", {})
    resolved_element = result.element_kind or (
        result.package.default_element_kind
        if result.package is not None
        else result.schema.get("x-polylogue-element-kind", "?")
    )
    click.echo(f"Schema: {result.provider} {result.version} [{resolved_element}]")
    if result.package is not None:
        click.echo(
            f"  Package anchor={result.package.anchor_kind}, "
            f"anchor-family={result.package.anchor_profile_family_id or 'n/a'}, "
            f"profiles={len(result.package.profile_family_ids)}, scopes={result.package.bundle_scope_count}, "
            f"window={result.package.first_seen} -> {result.package.last_seen}"
        )
    click.echo(
        f"  {len(props)} properties, {result.schema.get('x-polylogue-sample-count', '?')} samples, "
        f"{result.annotations.semantic_count} semantic roles, {result.annotations.format_count} format annotations"
    )
    click.echo(
        f"  Privacy: standard ({result.annotations.values_count} fields with enums, "
        f"{result.annotations.total_enum_values} values included)"
    )
    click.echo()
    click.echo(f"  $id: {result.schema.get('$id', 'N/A')}")
    click.echo(f"  Title: {result.schema.get('title', 'N/A')}")
    click.echo(f"  Description: {result.schema.get('description', 'N/A')}")
    click.echo()

    click.echo("  Metadata:")
    for key, label in [
        ("x-polylogue-version", "Version"),
        ("x-polylogue-generated-at", "Generated"),
        ("x-polylogue-registered-at", "Registered"),
        ("x-polylogue-promoted-at", "Promoted"),
        ("x-polylogue-sample-count", "Samples"),
        ("x-polylogue-sample-granularity", "Granularity"),
        ("x-polylogue-anchor-profile-family-id", "Anchor profile family"),
        ("x-polylogue-profile-family-ids", "Element profile families"),
        ("x-polylogue-package-profile-family-ids", "Package profile families"),
        ("x-polylogue-observed-artifact-count", "Observed artifacts"),
        ("x-polylogue-evidence-confidence", "Evidence confidence"),
    ]:
        value = result.schema.get(key)
        if value is not None:
            click.echo(f"    {label}: {value}")

    click.echo(f"\n  Properties ({len(props)}):")
    for name, prop_schema in sorted(props.items()):
        type_str = prop_schema.get("type", "?")
        annotations: list[str] = []
        if prop_schema.get("x-polylogue-semantic-role"):
            annotations.append(f"role={prop_schema['x-polylogue-semantic-role']}")
        if prop_schema.get("x-polylogue-format"):
            annotations.append(f"fmt={prop_schema['x-polylogue-format']}")
        if prop_schema.get("x-polylogue-frequency") is not None:
            annotations.append(f"freq={prop_schema['x-polylogue-frequency']}")
        if prop_schema.get("x-polylogue-values"):
            values = prop_schema["x-polylogue-values"]
            preview = ", ".join(str(value) for value in values[:5])
            if len(values) > 5:
                preview += f" (+{len(values) - 5} more)"
            annotations.append(f"values=[{preview}]")
        ann_str = f" ({', '.join(annotations)})" if annotations else ""
        click.echo(f"    {name}: {type_str}{ann_str}")

    for key in ("x-polylogue-foreign-keys", "x-polylogue-time-deltas", "x-polylogue-mutually-exclusive"):
        value = result.schema.get(key)
        if value:
            click.echo(f"\n  {key}:")
            for entry in value:
                click.echo(f"    {json.dumps(entry)}")

    if verbose:
        render_explain_verbose(result)


def render_explain_verbose(result: Any) -> None:
    """Render verbose schema explain coverage and role evidence."""
    click.echo()
    roles = result.annotations.roles
    if roles:
        click.echo("  Semantic Roles:")
        for role in roles:
            evidence_str = ", ".join(f"{key}={value}" for key, value in role.evidence.items())
            click.echo(f"    {role.role} -> {role.path} (score={role.confidence:.3f})")
            if evidence_str:
                click.echo(f"      evidence: {evidence_str}")

    coverage = result.annotations.coverage
    if coverage.total_fields:
        click.echo(f"\n  Annotation Coverage ({coverage.total_fields} fields):")
        click.echo(
            f"    Format:        {coverage.with_format}/{coverage.total_fields} "
            f"({coverage.with_format / coverage.total_fields * 100:.0f}%)"
        )
        click.echo(
            f"    Enum values:   {coverage.with_values}/{coverage.total_fields} "
            f"({coverage.with_values / coverage.total_fields * 100:.0f}%)"
        )
        click.echo(
            f"    Semantic role: {coverage.with_role}/{coverage.total_fields} "
            f"({coverage.with_role / coverage.total_fields * 100:.0f}%)"
        )


def _render_proof_surface(result: Any) -> None:
    """Render the proof surface for schema role assignment decisions."""
    proof = result.review_proof
    click.echo(f"Schema Review Proof: {result.provider} {result.version}")
    click.echo(f"  Artifact kind: {proof.artifact_kind or 'unknown'}")
    click.echo(f"  Eligible roles: {', '.join(proof.eligible_roles)}")
    if proof.ineligible_roles:
        click.echo(f"  Ineligible roles: {', '.join(proof.ineligible_roles)}")
    click.echo()

    for entry in proof.roles:
        if entry.abstained:
            click.echo(f"  {entry.role}: ABSTAINED")
            click.echo(f"    reason: {entry.abstain_reason}")
        else:
            click.echo(f"  {entry.role}: {entry.chosen_path} (score={entry.chosen_score:.3f})")
            if entry.evidence:
                evidence_str = ", ".join(f"{k}={v}" for k, v in entry.evidence.items())
                click.echo(f"    evidence: {evidence_str}")
            if entry.competing:
                click.echo(f"    competing ({len(entry.competing)}):")
                for comp in entry.competing[:5]:
                    click.echo(f"      {comp['path']} (score={comp['score']:.3f})")
        click.echo()


__all__ = ["render_explain_verbose", "render_schema_explain_result"]

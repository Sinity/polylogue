"""Atomic schema audit checks."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Mapping
from pathlib import Path

from polylogue.core.json import json_document, json_document_list
from polylogue.core.outcomes import OutcomeCheck as CheckResult
from polylogue.core.outcomes import OutcomeStatus
from polylogue.schemas.audit.walkers import _HEX_RE, _UUID_RE, SchemaNode, _walk_semantic_roles, _walk_values
from polylogue.schemas.privacy import (
    PUBLISHABLE_VOCABULARY_ROLES,
    _is_safe_enum_value,
    _looks_high_entropy_token,
)


def _redacted_value(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"sha256:{digest};length={len(value)}"


def check_privacy_guards(schema: Mapping[str, object] | SchemaNode) -> CheckResult:
    """Check that no UUIDs, hashes, or PII leak through enum values."""
    violations: list[str] = []
    root = json_document(schema)

    for path, values in _walk_values(root):
        for v in values:
            redacted = _redacted_value(v)
            if _UUID_RE.match(v):
                violations.append(f"{path}: UUID leak {redacted}")
            elif _HEX_RE.match(v):
                violations.append(f"{path}: hex-id leak {redacted}")
            elif _looks_high_entropy_token(v):
                violations.append(f"{path}: high-entropy token {redacted}")
            elif not _is_safe_enum_value(v, path=path):
                violations.append(f"{path}: unsafe value {redacted}")

    if violations:
        return CheckResult(
            name="privacy_guards",
            status=OutcomeStatus.ERROR,
            summary=f"{len(violations)} unsafe enum value(s) found",
            details=violations[:20],
        )
    return CheckResult(
        name="privacy_guards",
        status=OutcomeStatus.OK,
        summary="All enum values pass privacy checks",
    )


def _schema_node_at_path(schema: Mapping[str, object] | SchemaNode, path: str) -> SchemaNode:
    current = json_document(schema)
    for part in path.split(".")[1:]:
        if part == "*":
            current = json_document(current.get("additionalProperties"))
            continue
        if part.endswith("[*]"):
            name = part[:-3]
            if name:
                current = json_document(json_document(current.get("properties")).get(name))
            current = json_document(current.get("items"))
            continue
        current = json_document(json_document(current.get("properties")).get(part))
    return current


def check_semantic_roles(schema: Mapping[str, object] | SchemaNode) -> CheckResult:
    """Check semantic role assignments for sanity."""
    issues: list[str] = []
    root = json_document(schema)
    roles = _walk_semantic_roles(root)

    for path, role, _confidence in roles:
        if role == "session_title":
            current = _schema_node_at_path(root, path)
            fmt = current.get("x-polylogue-format")

            if fmt in ("uuid4", "uuid", "hex-id"):
                issues.append(f"{path}: {fmt}-format field assigned as {role}")

            terminal = path.rsplit(".", 1)[-1].lower()
            if terminal.endswith(("id", "_id", "uuid")):
                issues.append(f"{path}: ID-like field assigned as {role}")

    if issues:
        return CheckResult(
            name="semantic_roles",
            status=OutcomeStatus.ERROR,
            summary=f"{len(issues)} misclassified semantic role(s)",
            details=issues[:10],
        )

    if not roles:
        return CheckResult(
            name="semantic_roles",
            status=OutcomeStatus.WARNING,
            summary="No semantic roles detected",
        )

    return CheckResult(
        name="semantic_roles",
        status=OutcomeStatus.OK,
        summary=f"{len(roles)} role(s) assigned correctly",
    )


def check_annotation_coverage(schema: Mapping[str, object] | SchemaNode) -> CheckResult:
    """Check that schema has adequate annotation coverage."""
    total_fields = 0
    annotated_fields = 0
    annotation_keys = {
        "x-polylogue-format",
        "x-polylogue-values",
        "x-polylogue-semantic-role",
        "x-polylogue-frequency",
        "x-polylogue-range",
        "x-polylogue-multiline",
        "x-polylogue-high-cardinality-keys",
    }

    def _count(node: SchemaNode) -> None:
        nonlocal total_fields, annotated_fields
        properties = json_document(node.get("properties"))
        for prop in properties.values():
            child = json_document(prop)
            if not child:
                continue
            total_fields += 1
            if any(key in child for key in annotation_keys):
                annotated_fields += 1
            if child.get("x-polylogue-high-cardinality-keys") is True:
                continue
            _count(child)
        items = json_document(node.get("items"))
        if items:
            _count(items)
        additional_properties = json_document(node.get("additionalProperties"))
        if additional_properties:
            _count(additional_properties)
        for keyword in ("anyOf", "oneOf", "allOf"):
            for child in json_document_list(node.get(keyword)):
                _count(child)

    _count(json_document(schema))

    if total_fields == 0:
        return CheckResult(
            name="annotation_coverage",
            status=OutcomeStatus.WARNING,
            summary="No properties found in schema",
        )

    pct = (annotated_fields / total_fields) * 100
    if total_fields > 500:
        pass_pct, warn_pct = 5, 2
    else:
        pass_pct, warn_pct = 30, 10
    if pct >= pass_pct:
        status = OutcomeStatus.OK
    elif pct >= warn_pct:
        status = OutcomeStatus.WARNING
    else:
        status = OutcomeStatus.ERROR

    return CheckResult(
        name="annotation_coverage",
        status=status,
        summary=f"{annotated_fields}/{total_fields} fields annotated ({pct:.0f}%)",
    )


def check_cross_provider_consistency(
    schemas: Mapping[str, Mapping[str, object] | SchemaNode],
    *,
    sample_counts: Mapping[str, int | None] | None = None,
) -> CheckResult:
    """Check consistency across all provider schemas.

    Sample counts come from the package manifest, not from the element
    schema document: committed element schemas are content-only projections
    (see :func:`check_schema_staleness`), so the generation route publishes
    the denominator in ``package.json`` instead.
    """
    issues: list[str] = []
    counts = sample_counts or {}

    for provider, schema in schemas.items():
        root = json_document(schema)
        roles = _walk_semantic_roles(root)
        if not roles:
            issues.append(f"{provider}: no semantic roles")

        if not counts.get(provider):
            issues.append(f"{provider}: package manifest records no sample count")

    if issues:
        return CheckResult(
            name="cross_provider_consistency",
            status=OutcomeStatus.WARNING,
            summary=f"{len(issues)} consistency issue(s)",
            details=issues,
        )

    return CheckResult(
        name="cross_provider_consistency",
        status=OutcomeStatus.OK,
        summary=f"All {len(schemas)} provider schemas consistent",
    )


def check_schema_staleness(observed_at: str | None) -> CheckResult:
    """Report when the committed package was last observed against its source.

    ``observed_at`` is the package manifest's ``last_seen``. The committed
    element schema document deliberately carries no ``x-polylogue-generated-at``
    field: ``SchemaRegistry.write_package`` gzips each element with ``mtime=0``
    and sorted keys so an unchanged structure regenerates to byte-identical
    output, and that byte equality is the evidence a regeneration run uses to
    prove a package unchanged. A wall-clock timestamp inside the document would
    make every regeneration diff against itself, so the generation route
    (``polylogue.schemas.generation.workflow``) keeps observation time in the
    manifest instead. Do not re-add the field to published elements.

    Age is reported, never failed. A package does not stop describing its
    source because the calendar advanced, and no repository edit can clear a
    date-triggered warning, so an age threshold is not a gate invariant. A
    manifest with no usable observation time is a real defect and warns.
    """
    from datetime import UTC, datetime

    if not observed_at or not isinstance(observed_at, str):
        return CheckResult(
            name="schema_staleness",
            status=OutcomeStatus.WARNING,
            summary="Package manifest records no observation time",
        )
    try:
        observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
        age_days = (datetime.now(UTC) - observed).days
    except ValueError:
        return CheckResult(
            name="schema_staleness",
            status=OutcomeStatus.WARNING,
            summary=f"Unparseable manifest observation time: {observed_at!r}",
        )
    suffix = " — refresh from current source material when a generation run is authorized" if age_days > 90 else ""
    return CheckResult(
        name="schema_staleness",
        status=OutcomeStatus.OK,
        summary=f"Package last observed {age_days} days ago{suffix}",
    )


def check_schema_drift(
    schema: Mapping[str, object] | SchemaNode,
    *,
    db_path: Path | None = None,
    provider: str = "",
    max_samples: int = 50,
) -> CheckResult:
    """Check whether committed schema has drifted from live archive data.

    Loads samples from the database and runs detect_drift() against the
    committed schema.  Reports fields present in live data but missing from
    the schema.
    """
    from polylogue.core.json import json_document
    from polylogue.schemas.sampling import load_samples_from_db
    from polylogue.schemas.validator import detect_drift

    if db_path is None or not Path(db_path).exists():
        return CheckResult(
            name="schema_drift",
            status=OutcomeStatus.SKIP,
            summary="No database available for drift detection",
        )

    try:
        samples = load_samples_from_db(provider, db_path=Path(db_path), max_samples=max_samples)
    except Exception as exc:
        return CheckResult(
            name="schema_drift",
            status=OutcomeStatus.WARNING,
            summary=f"Failed to load samples for drift detection: {exc}",
        )

    if not samples:
        return CheckResult(
            name="schema_drift",
            status=OutcomeStatus.SKIP,
            summary="No samples available for drift detection",
        )

    root = json_document(schema)
    all_drift: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        for warning in detect_drift(sample, root, ""):
            if warning not in seen:
                seen.add(warning)
                all_drift.append(warning)

    if not all_drift:
        return CheckResult(
            name="schema_drift",
            status=OutcomeStatus.OK,
            summary=f"No drift detected across {len(samples)} sample(s)",
            count=len(samples),
        )

    unique_drift = list(dict.fromkeys(all_drift))
    return CheckResult(
        name="schema_drift",
        status=OutcomeStatus.WARNING,
        summary=f"Schema drift detected: {len(unique_drift)} unexpected field(s) in live data",
        details=unique_drift[:50],
        count=len(samples),
    )


__all__ = [
    "CheckResult",
    "check_annotation_coverage",
    "check_cross_provider_consistency",
    "check_privacy_guards",
    "check_schema_drift",
    "check_schema_staleness",
    "check_semantic_roles",
]


_EXEMPT_STRUCTURAL_KEYS = frozenset(
    {
        "$id",
        "$ref",
        "$schema",
        "hash_algorithm",
        "x-polylogue-anchor-profile-family-id",
        "x-polylogue-exact-structure-ids",
        "x-polylogue-package-profile-family-ids",
        "x-polylogue-profile-family-ids",
        "x-polylogue-ref",
    }
)

_HOME_PATH_RE = re.compile(r"(?:^|[\s\"'=(:])(?:~/|/home/|/Users/|/realm/|/etc/|/var/|/opt/|[A-Za-z]:\\)")
_ABSOLUTE_PATH_RE = re.compile(r"^/[A-Za-z0-9._-]")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_SCHEME_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://")


def _leak_kind(value: str) -> str | None:
    """Classify a published string that must never reach a public package."""

    if _EMAIL_RE.search(value):
        return "email address"
    if _URL_SCHEME_RE.search(value):
        return "absolute URL"
    if _HOME_PATH_RE.search(value) or _ABSOLUTE_PATH_RE.match(value):
        return "filesystem path"
    return None


def _iter_published_vocabularies(node: SchemaNode, path: str = "$") -> Iterator[tuple[str, list[str], object]]:
    """Yield every published member list with its declaring node's semantic role.

    The walk is structural rather than keyed on ``properties``: a committed
    element embeds provider payload shapes whose own property names include
    ``enum`` and ``const``, so a keyword-position walk would both miss nested
    containers and mistake wire data for schema keywords.
    """

    values = node.get("x-polylogue-values")
    if isinstance(values, list):
        members = [item for item in values if isinstance(item, str)]
        if members:
            yield path, members, node.get("x-polylogue-semantic-role")
    for key, child in node.items():
        if key == "x-polylogue-values":
            continue
        if isinstance(child, dict):
            yield from _iter_published_vocabularies(child, f"{path}.{key}")
        elif isinstance(child, list):
            for index, item in enumerate(child):
                if isinstance(item, dict):
                    yield from _iter_published_vocabularies(item, f"{path}.{key}[{index}]")


def check_published_vocabulary(schema: Mapping[str, object] | SchemaNode) -> CheckResult:
    """Refuse a closed vocabulary published from an undeclared slot.

    ``x-polylogue-values`` is the only annotation carrying *observed member
    values* into a committed package, and a committed package is public. Value
    shape cannot separate a provider protocol constant from a recurring private
    token, so publication is an allowlist over declared semantic roles
    (:data:`polylogue.schemas.privacy.PUBLISHABLE_VOCABULARY_ROLES`) and every
    other slot publishes structure without members.
    """

    violations: list[str] = []
    for path, members, role in _iter_published_vocabularies(json_document(schema)):
        if isinstance(role, str) and role in PUBLISHABLE_VOCABULARY_ROLES:
            continue
        declared = role if isinstance(role, str) else "no declared semantic role"
        violations.append(f"{path}: unjustified closed vocabulary ({declared}, {len(members)} member(s))")

    if violations:
        return CheckResult(
            name="published_vocabulary",
            status=OutcomeStatus.ERROR,
            summary=f"{len(violations)} unjustified published vocabulary site(s)",
            count=len(violations),
            details=violations[:20],
        )
    return CheckResult(
        name="published_vocabulary",
        status=OutcomeStatus.OK,
        summary="Every published vocabulary sits on a declared protocol slot",
    )


def _iter_published_strings(node: object, path: str = "$", key: str | None = None) -> Iterator[tuple[str, str]]:
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            for name in properties:
                if isinstance(name, str):
                    yield f"{path}.properties", name
        for child_key, child in node.items():
            if child_key in _EXEMPT_STRUCTURAL_KEYS:
                continue
            yield from _iter_published_strings(child, f"{path}.{child_key}", child_key)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _iter_published_strings(item, f"{path}[{index}]", key)
    elif isinstance(node, str):
        yield path, node


def check_published_paths(schema: Mapping[str, object] | SchemaNode) -> CheckResult:
    """Refuse a filesystem path, mail address, or URL anywhere in a bundle.

    Dynamic-key collapse and vocabulary publication both harvest strings out of
    acquired material, so the leak can arrive as a property *name* as easily as
    a value. This check reads the whole decompressed element rather than one
    keyword, and reports a digest instead of the offending text.
    """

    violations: list[str] = []
    for path, value in _iter_published_strings(json_document(schema)):
        kind = _leak_kind(value)
        if kind is not None:
            violations.append(f"{path}: {kind} {_redacted_value(value)}")

    if violations:
        return CheckResult(
            name="published_paths",
            status=OutcomeStatus.ERROR,
            summary=f"{len(violations)} published path/address leak(s)",
            count=len(violations),
            details=violations[:20],
        )
    return CheckResult(
        name="published_paths",
        status=OutcomeStatus.OK,
        summary="No filesystem path, mail address, or URL is published",
    )

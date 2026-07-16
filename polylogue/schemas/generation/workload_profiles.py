"""Privacy-safe workload-profile artifacts derived from schema evidence."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from polylogue.core.json import JSONDocument, JSONValue, json_document
from polylogue.schemas.field_stats.distributions import CategoricalSketch, DistributionSketch
from polylogue.schemas.generation.models import _PackageAccumulator, _UnitMembership
from polylogue.schemas.generation.packages import _package_bundle_scope_count
from polylogue.schemas.generation.replay import (
    membership_sample_count,
    metadata_memberships,
    select_artifact_memberships,
)

WORKLOAD_PROFILE_VERSION = 1
_STRUCTURAL_VARIANT_CAP = 256
_TOOL_IDS_PER_SCOPE_CAP = 4_096


def _normalize_profile_strings(value: object) -> object:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, child in value.items():
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise ValueError(f"Workload profile keys collide after NFC normalization: {normalized_key!r}")
            normalized[normalized_key] = _normalize_profile_strings(child)
        return normalized
    if isinstance(value, list):
        return [_normalize_profile_strings(child) for child in value]
    return value


def workload_profile_identity(profile: Mapping[str, object]) -> str:
    """Return a deterministic identity for a content-only workload profile."""
    payload = {key: value for key, value in profile.items() if key != "profile_id"}
    encoded = json.dumps(
        _normalize_profile_strings(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


_PARENT_FIELD_NAMES = frozenset(
    {
        "parentUuid",
        "parent_id",
        "parentId",
        "parentSessionId",
        "parent_session_id",
        "branch_from_id",
    }
)


def _schema_node(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _json_values(values: Iterable[str]) -> list[JSONValue]:
    return list(values)


def _walk_mappings(value: object) -> Iterable[Mapping[str, object]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _text_token(value: object) -> str:
    return value.lower() if isinstance(value, str) else ""


def _first_text(record: Mapping[str, object], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value:
            return value
    return None


@dataclass
class _ToolScope:
    calls: dict[str, list[int]] = field(default_factory=dict)
    results: dict[str, list[int]] = field(default_factory=dict)
    call_without_id: int = 0
    result_without_id: int = 0
    error_results: int = 0
    functions_exec_calls: int = 0
    dropped_ids: int = 0
    known_ids: set[str] = field(default_factory=set)

    def _record_id(self, target: dict[str, list[int]], tool_id: str, position: int) -> None:
        if tool_id in self.known_ids or len(self.known_ids) < _TOOL_IDS_PER_SCOPE_CAP:
            self.known_ids.add(tool_id)
            target.setdefault(tool_id, []).append(position)
        else:
            self.dropped_ids += 1

    def observe(self, record: Mapping[str, object], position: int) -> None:
        kind = _text_token(record.get("type")) or _text_token(record.get("kind"))
        name = _first_text(record, ("name", "recipient_name", "tool_name"))
        is_exec = name == "functions.exec"
        is_result = kind in {"tool_result", "function_call_output", "tool_call_output", "function_result"}
        is_call = is_exec or kind in {"tool_use", "function_call", "tool_call", "function"}
        if not (is_call or is_result):
            return
        if is_exec:
            self.functions_exec_calls += 1
        if is_result:
            tool_id = _first_text(record, ("tool_use_id", "call_id", "tool_call_id", "id"))
            if tool_id is None:
                self.result_without_id += 1
            else:
                self._record_id(self.results, tool_id, position)
            exit_code = record.get("exit_code")
            if (
                record.get("is_error") is True
                or record.get("tool_result_is_error") is True
                or _text_token(record.get("status")) in {"error", "failed", "failure"}
                or (isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0)
            ):
                self.error_results += 1
            return
        tool_id = _first_text(record, ("id", "call_id", "tool_call_id"))
        if tool_id is None:
            self.call_without_id += 1
        else:
            self._record_id(self.calls, tool_id, position)


@dataclass
class _RelationshipAccumulator:
    paired: int = 0
    missing_results: int = 0
    orphan_results: int = 0
    duplicate_calls: int = 0
    duplicate_results: int = 0
    out_of_order_results: int = 0
    call_without_id: int = 0
    result_without_id: int = 0
    error_results: int = 0
    functions_exec_calls: int = 0
    dropped_ids: int = 0
    scope_record_counts: DistributionSketch = field(default_factory=DistributionSketch)
    result_distance: DistributionSketch = field(default_factory=DistributionSketch)
    parent_reference_count: int = 0
    root_parent_count: int = 0
    sidechain_true_count: int = 0
    subagent_marker_count: int = 0

    def merge_scope(self, scope: _ToolScope, record_count: int) -> None:
        self.scope_record_counts.observe(record_count)
        self.call_without_id += scope.call_without_id
        self.result_without_id += scope.result_without_id
        self.error_results += scope.error_results
        self.functions_exec_calls += scope.functions_exec_calls
        self.dropped_ids += scope.dropped_ids
        for tool_id in set(scope.calls) | set(scope.results):
            calls = scope.calls.get(tool_id, [])
            results = scope.results.get(tool_id, [])
            paired = min(len(calls), len(results))
            self.paired += paired
            self.missing_results += max(0, len(calls) - len(results))
            self.orphan_results += max(0, len(results) - len(calls))
            self.duplicate_calls += max(0, len(calls) - 1)
            self.duplicate_results += max(0, len(results) - 1)
            for call_position, result_position in zip(calls, results, strict=False):
                distance = result_position - call_position
                self.result_distance.observe(distance)
                if distance < 0:
                    self.out_of_order_results += 1

    def observe_lineage(self, record: Mapping[str, object]) -> None:
        for name in _PARENT_FIELD_NAMES:
            if name not in record:
                continue
            if record[name] is None or record[name] == "":
                self.root_parent_count += 1
            else:
                self.parent_reference_count += 1
        if record.get("isSidechain") is True or record.get("is_sidechain") is True:
            self.sidechain_true_count += 1
        kind = _text_token(record.get("type")) or _text_token(record.get("kind"))
        if "subagent" in kind or ("agent" in kind and "spawn" in kind):
            self.subagent_marker_count += 1

    def to_payload(self) -> JSONDocument:
        return json_document(
            {
                "tool_results": {
                    "paired": self.paired,
                    "missing": self.missing_results,
                    "orphan": self.orphan_results,
                    "duplicate_calls": self.duplicate_calls,
                    "duplicate_results": self.duplicate_results,
                    "out_of_order": self.out_of_order_results,
                    "call_without_id": self.call_without_id,
                    "result_without_id": self.result_without_id,
                    "error_results": self.error_results,
                    "functions_exec_calls": self.functions_exec_calls,
                    "result_record_distance": self.result_distance.to_payload(),
                },
                "lineage": {
                    "parent_references": self.parent_reference_count,
                    "root_parent_values": self.root_parent_count,
                    "sidechain_true": self.sidechain_true_count,
                    "subagent_markers": self.subagent_marker_count,
                },
                "scope_record_count": self.scope_record_counts.to_payload(),
                "loss_inventory": {"tool_ids_over_capacity": self.dropped_ids},
            }
        )


def _scope_key(membership: _UnitMembership) -> str:
    unit = membership.unit
    return unit.bundle_scope or unit.raw_id or unit.source_path or membership.profile_family_id


def _relationship_profile(memberships: Sequence[_UnitMembership]) -> JSONDocument:
    aggregate = _RelationshipAccumulator()
    current_scope: str | None = None
    tool_scope = _ToolScope()
    position = 0
    record_count = 0
    for membership in memberships:
        scope = _scope_key(membership)
        if current_scope is not None and scope != current_scope:
            aggregate.merge_scope(tool_scope, record_count)
            tool_scope = _ToolScope()
            position = 0
            record_count = 0
        current_scope = scope
        for sample in membership.unit.schema_samples:
            record_count += 1
            for record in _walk_mappings(sample):
                aggregate.observe_lineage(record)
                tool_scope.observe(record, position)
                position += 1
    if current_scope is not None:
        aggregate.merge_scope(tool_scope, record_count)
    return aggregate.to_payload()


def _field_profiles(schema: Mapping[str, object], path: str = "$") -> dict[str, JSONDocument]:
    profiles: dict[str, JSONDocument] = {}
    distribution = _schema_node(schema.get("x-polylogue-observed-distribution"))
    if distribution is not None:
        payload = json_document(dict(distribution))
        values = schema.get("x-polylogue-values")
        if isinstance(values, list):
            payload["privacy_approved_values"] = list(values)
        fmt = schema.get("x-polylogue-format")
        if isinstance(fmt, str):
            payload["format"] = fmt
        profiles[path] = payload

    properties = _schema_node(schema.get("properties"))
    if properties is not None:
        for name, child in properties.items():
            child_node = _schema_node(child)
            if child_node is not None:
                profiles.update(_field_profiles(child_node, f"{path}.{name}"))
    additional = _schema_node(schema.get("additionalProperties"))
    if additional is not None:
        profiles.update(_field_profiles(additional, f"{path}.*"))
    items = _schema_node(schema.get("items"))
    if items is not None:
        profiles.update(_field_profiles(items, f"{path}[*]"))
    return profiles


def _misra_gries_candidates(
    memberships: Iterable[_UnitMembership],
    *,
    element_kind: str,
) -> tuple[set[tuple[str, ...]], int]:
    candidates: Counter[tuple[str, ...]] = Counter()
    reductions = 0
    for membership in memberships:
        if membership.unit.artifact_kind != element_kind:
            continue
        variant = tuple(membership.unit.profile_tokens)
        if variant in candidates or len(candidates) < _STRUCTURAL_VARIANT_CAP:
            candidates[variant] += 1
            continue
        reductions += 1
        for key in list(candidates):
            candidates[key] -= 1
            if candidates[key] <= 0:
                del candidates[key]
    return set(candidates), reductions


def _structural_variants(
    memberships: Sequence[_UnitMembership],
    *,
    element_kind: str,
) -> tuple[list[JSONDocument], JSONDocument]:
    candidates, reductions = _misra_gries_candidates(memberships, element_kind=element_kind)
    exact_counts: Counter[tuple[str, ...]] = Counter()
    all_variants = CategoricalSketch()
    total = 0
    for membership in memberships:
        if membership.unit.artifact_kind != element_kind:
            continue
        total += 1
        variant = tuple(membership.unit.profile_tokens)
        all_variants.observe("\x1f".join(variant))
        if variant in candidates:
            exact_counts[variant] += 1

    variants: list[JSONDocument] = []
    retained = 0
    for tokens, count in sorted(exact_counts.items(), key=lambda item: (-item[1], item[0])):
        retained += count
        variants.append(
            {
                "count": count,
                "frequency": count / total if total else 0.0,
                "tokens": _json_values(tokens),
            }
        )
    loss: JSONDocument = {
        "variant_capacity": _STRUCTURAL_VARIANT_CAP,
        "candidate_reductions": reductions,
        "unrepresented_observations": max(0, total - retained),
        "all_observations": all_variants.to_payload(),
    }
    return variants, loss


def build_package_workload_profile(
    *,
    provider: str,
    version: str,
    package: _PackageAccumulator,
    element_schemas: Mapping[str, JSONDocument],
    privacy_policy: str,
    observation_outcomes: JSONDocument | None = None,
) -> JSONDocument:
    """Build a bounded package profile without retaining corpus content."""
    package_metadata = metadata_memberships(package.memberships)
    elements: JSONDocument = {}
    for element_kind, schema in sorted(element_schemas.items()):
        variants, variant_loss = _structural_variants(package_metadata, element_kind=element_kind)
        element_memberships = select_artifact_memberships(package.memberships, element_kind)
        element_metadata = select_artifact_memberships(package_metadata, element_kind)
        sample_count = membership_sample_count(element_memberships)
        elements[element_kind] = json_document(
            {
                "artifact_count": len(element_metadata),
                "sample_count": sample_count,
                "field_profiles": _field_profiles(schema),
                "structural_variants": variants,
                "loss_inventory": variant_loss,
            }
        )

    profile: JSONDocument = {
        "profile_version": WORKLOAD_PROFILE_VERSION,
        "profile_kind": "provider-package",
        "provider": provider,
        "package_version": version,
        "inference_version": "field-distributions-v1",
        "privacy_policy": privacy_policy,
        "privacy_classification": "aggregate-structural-no-raw-content",
        "provenance": {
            "first_seen": package.first_seen,
            "last_seen": package.last_seen,
            "bundle_scope_count": _package_bundle_scope_count(package),
            "sample_count": membership_sample_count(package.memberships),
            "observation_window": {
                "start": package.first_seen,
                "end": package.last_seen,
            },
        },
        "relationships": _relationship_profile(package.memberships),
        "elements": elements,
    }
    if observation_outcomes:
        provenance = profile["provenance"]
        if not isinstance(provenance, dict):
            raise TypeError("Workload profile provenance must be an object")
        provenance["observation_outcomes"] = observation_outcomes
    profile["profile_id"] = workload_profile_identity(profile)
    return profile


__all__ = ["WORKLOAD_PROFILE_VERSION", "build_package_workload_profile", "workload_profile_identity"]

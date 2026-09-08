"""Semantic contracts for reusable source-inference evidence.

Bump the affected revision when decoding, identity, structure, or statistics
change meaning. Implementation fingerprints record provenance independently;
performance and presentation edits do not change these contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from polylogue.core.hashing import hash_payload
from polylogue.core.json import JSONDocument, JSONValue
from polylogue.schemas.field_stats.detection import is_dynamic_key, key_policy_parameters
from polylogue.schemas.generation.evidence import SCHEMA_EVIDENCE_VERSION

EvidencePhase = Literal["structure", "statistics"]


@dataclass(frozen=True)
class SourceEvidenceRecipe:
    admission_revision: int = 1
    identity_revision: int = 2
    zip_member_revision: int = 2
    structure_revision: int = 1
    statistics_revision: int = 1

    def contract(self, phase: EvidencePhase) -> JSONDocument:
        return {
            "evidence_format": SCHEMA_EVIDENCE_VERSION,
            "admission_revision": self.admission_revision,
            "identity_revision": self.identity_revision,
            "zip_member_revision": self.zip_member_revision,
            "phase": phase,
            "reduction_revision": self.structure_revision if phase == "structure" else self.statistics_revision,
            "key_policy": key_policy_parameters(),
        }

    def fingerprint(self, phase: EvidencePhase) -> str:
        return hash_payload(self.contract(phase))

    def provenance(self, implementation_fingerprint: str) -> JSONDocument:
        return {
            "structure": self.contract("structure"),
            "statistics": self.contract("statistics"),
            "structure_fingerprint": self.fingerprint("structure"),
            "statistics_fingerprint": self.fingerprint("statistics"),
            "implementation_fingerprint": implementation_fingerprint,
        }


def contracts_match_except_key_limit(previous: JSONDocument, current: JSONDocument) -> bool:
    """A key-limit change can be checked against each contribution's evidence."""

    def without_limit(contract: JSONDocument) -> JSONDocument:
        policy = contract.get("key_policy")
        if not isinstance(policy, dict):
            return contract
        return {**contract, "key_policy": {key: value for key, value in policy.items() if key != "cardinality_limit"}}

    previous_without_limit = without_limit(previous)
    current_without_limit = without_limit(current)
    for revision_key in ("identity_revision", "zip_member_revision"):
        previous_revision = previous_without_limit.get(revision_key, 1)
        current_revision = current_without_limit.get(revision_key, 1)
        if previous_revision == current_revision:
            continue
        if previous_revision == 1 and current_revision == 2:
            previous_without_limit = {
                key: value for key, value in previous_without_limit.items() if key != revision_key
            }
            current_without_limit = {key: value for key, value in current_without_limit.items() if key != revision_key}
            continue
        return False
    return previous_without_limit == current_without_limit


def has_collapsed_names(schema: JSONValue) -> bool:
    """Collapsed names cannot be recovered from a structural summary."""
    if not isinstance(schema, dict):
        return False
    if schema.get("x-polylogue-high-cardinality-keys") is True:
        return True
    properties = schema.get("properties")
    if isinstance(properties, dict) and any(has_collapsed_names(value) for value in properties.values()):
        return True
    return any(has_collapsed_names(schema.get(key)) for key in ("items", "additionalProperties"))


def relevant_normalization_paths(paths: tuple[str, ...], schema: JSONDocument) -> tuple[str, ...]:
    """Ignore normalization changes in object paths absent from this source."""

    selected = set(paths)
    relevant: set[str] = set()

    def walk(node: JSONValue, path: str) -> None:
        if not isinstance(node, dict):
            return
        node_type = node.get("type")
        if (node_type == "object" or isinstance(node_type, list) and "object" in node_type) and path in selected:
            relevant.add(path)
        properties = node.get("properties")
        if isinstance(properties, dict):
            for name, child in properties.items():
                key = "*" if path in selected or is_dynamic_key(name) else name
                walk(child, f"{path}.{key}")
        walk(node.get("additionalProperties"), f"{path}.*")
        walk(node.get("items"), f"{path}[*]")

    walk(schema, "$")
    return tuple(sorted(relevant))

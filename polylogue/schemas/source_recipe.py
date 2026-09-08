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
from polylogue.schemas.field_stats.detection import key_policy_parameters
from polylogue.schemas.generation.evidence import SCHEMA_EVIDENCE_VERSION

EvidencePhase = Literal["structure", "statistics"]


@dataclass(frozen=True)
class SourceEvidenceRecipe:
    admission_revision: int = 1
    structure_revision: int = 1
    statistics_revision: int = 1

    def contract(self, phase: EvidencePhase) -> JSONDocument:
        return {
            "evidence_format": SCHEMA_EVIDENCE_VERSION,
            "admission_revision": self.admission_revision,
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

    return without_limit(previous) == without_limit(current)


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

    def possible(node: JSONValue, steps: tuple[str, ...]) -> bool:
        if not isinstance(node, dict):
            return False
        if not steps:
            node_type = node.get("type")
            return node_type == "object" or isinstance(node_type, list) and "object" in node_type
        step, *rest = steps
        remaining = tuple(rest)
        if step == "[*]":
            return possible(node.get("items"), remaining)
        properties = node.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        if step == "*":
            return any(possible(child, remaining) for child in properties.values()) or possible(
                node.get("additionalProperties"), remaining
            )
        if step in properties:
            return possible(properties[step], remaining)
        return possible(node.get("additionalProperties"), remaining)

    def steps(path: str) -> tuple[str, ...]:
        return tuple(part for part in path.removeprefix("$").replace("[*]", ".[*]").split(".") if part)

    return tuple(sorted(path for path in paths if possible(schema, steps(path))))

"""Canonical durable reference fields used by candidate transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ReferenceCardinality = Literal["scalar", "json-list"]
ReferenceGrammar = Literal["public", "public-or-opaque"]
ReferenceTransition = Literal["rewrite", "successor", "sealed"]
DurableTier = Literal["user", "audit"]


@dataclass(frozen=True, slots=True)
class DurableReferenceField:
    """One declared reference-bearing column in a durable relation."""

    column: str
    cardinality: ReferenceCardinality
    grammar: ReferenceGrammar


@dataclass(frozen=True, slots=True)
class DurableReferenceRelation:
    """One durable relation whose public references survive reconstruction."""

    tier: DurableTier
    table: str
    identity_columns: tuple[str, ...]
    fields: tuple[DurableReferenceField, ...]
    transition: ReferenceTransition = "sealed"


def _scalar(column: str, *, grammar: ReferenceGrammar = "public") -> DurableReferenceField:
    return DurableReferenceField(column, "scalar", grammar)


def _list(column: str, *, grammar: ReferenceGrammar = "public-or-opaque") -> DurableReferenceField:
    return DurableReferenceField(column, "json-list", grammar)


_USER_RELATIONS = (
    DurableReferenceRelation(
        "user",
        "assertions",
        ("assertion_id",),
        (
            _scalar("scope_ref"),
            _scalar("target_ref"),
            _scalar("author_ref"),
            _list("evidence_refs_json"),
            _list("supersedes_json"),
        ),
        "rewrite",
    ),
    DurableReferenceRelation("user", "query_excision_ledger", ("ledger_id",), (_scalar("actor_ref"),)),
    DurableReferenceRelation(
        "user",
        "result_set_members",
        ("result_set_id", "rank"),
        (_scalar("member_ref"),),
        "successor",
    ),
    DurableReferenceRelation(
        "user",
        "query_evaluation_receipts",
        ("receipt_id",),
        (_scalar("runtime_build_ref", grammar="public-or-opaque"), _list("model_refs_json")),
    ),
    DurableReferenceRelation("user", "holdout_access_receipts", ("receipt_id",), (_scalar("accessor_ref"),)),
    DurableReferenceRelation(
        "user",
        "annotation_batches",
        ("batch_id",),
        (
            _scalar("target_ref"),
            _scalar("source_result_ref"),
            _scalar("actor_ref"),
            _scalar("model_ref"),
            _scalar("prompt_ref"),
            _list("assertion_refs_json"),
        ),
    ),
    DurableReferenceRelation("user", "user_settings", ("setting_key",), (_scalar("author_ref"),)),
    DurableReferenceRelation(
        "user",
        "context_deliveries",
        ("snapshot_ref",),
        (
            _scalar("snapshot_ref"),
            _scalar("recipient_ref"),
            _scalar("run_ref", grammar="public-or-opaque"),
            _list("segment_refs_json"),
            _list("evidence_refs_json"),
            _list("assertion_refs_json"),
            _scalar("delivered_by_ref"),
        ),
    ),
)

_AUDIT_RELATIONS = (
    DurableReferenceRelation("audit", "operation_previews", ("preview_id",), (_scalar("principal_actor_ref"),)),
    DurableReferenceRelation("audit", "operation_preview_targets", ("preview_id", "ordinal"), (_scalar("target_ref"),)),
    DurableReferenceRelation("audit", "operation_authorizations", ("authorization_id",), (_scalar("actor_ref"),)),
    DurableReferenceRelation(
        "audit",
        "operation_runs",
        ("operation_id",),
        (_scalar("actor_ref"), _scalar("domain_receipt_ref", grammar="public-or-opaque")),
    ),
    DurableReferenceRelation(
        "audit",
        "operation_targets",
        ("operation_id", "ordinal"),
        (_scalar("target_ref"), _scalar("domain_receipt_ref", grammar="public-or-opaque")),
    ),
    DurableReferenceRelation("audit", "operation_events", ("operation_id", "sequence"), (_scalar("actor_ref"),)),
)

DURABLE_REFERENCE_RELATIONS = _USER_RELATIONS + _AUDIT_RELATIONS


def durable_reference_relations(tier: DurableTier) -> tuple[DurableReferenceRelation, ...]:
    """Return the canonical declarations for one durable tier."""

    return tuple(relation for relation in DURABLE_REFERENCE_RELATIONS if relation.tier == tier)


__all__ = [
    "DURABLE_REFERENCE_RELATIONS",
    "DurableReferenceField",
    "DurableReferenceRelation",
    "ReferenceCardinality",
    "ReferenceGrammar",
    "ReferenceTransition",
    "durable_reference_relations",
]

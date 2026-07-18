"""Single-row annotation write path: schema-validated labels as assertions.

This is the atomic operation the batch/JSONL import surface loops over.
It accepts durable ``annotation-batch:`` provenance but deliberately does not
attempt the CLI/MCP surface here -- it validates one label against a declared
:class:`~polylogue.annotations.schema.AnnotationSchema` and writes it through
the existing single assertion-write chokepoint
(:func:`polylogue.storage.sqlite.archive_tiers.user_write.upsert_assertion`),
so it inherits that function's agent-authored candidate-coercion invariant
(polylogue-37t.15) for free: any ``author_kind`` other than ``"user"`` lands
``status=candidate`` with a non-injected context policy, regardless of what
this helper requests.

Scope note: this validates the row's *shape* (schema conformance) and that
evidence refs are well-formed object/evidence refs (enforced transitively by
``upsert_assertion``'s own ref normalization). It does not check that
``target_ref``/``evidence_refs`` resolve to rows that actually exist in the
live archive -- that referential-integrity check belongs to the batch import
surface, which has the archive handle this single-connection helper does not.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, cast

from polylogue.annotations.schema import (
    ANNOTATION_SCHEMA_REGISTRY,
    AnnotationSchema,
    AnnotationSchemaRegistry,
    validate_annotation_row,
)
from polylogue.core.enums import AssertionKind, AssertionStatus, AssertionVisibility
from polylogue.core.json import JSONDocument, JSONValue, require_json_document
from polylogue.core.refs import ObjectRef, normalize_object_ref_text, normalize_public_ref_text
from polylogue.storage.sqlite.archive_tiers.user_write import (
    ArchiveAssertionEnvelope,
    ArchiveAssertionJudgmentEnvelope,
    judge_assertion_candidate,
    read_assertion_envelope,
    upsert_assertion,
)

if TYPE_CHECKING:
    from polylogue.storage.sqlite.archive_tiers.user_annotations import DurableAnnotationSchema

_SCHEMA_PROVENANCE_KEY = "_schema"
_BATCH_PROVENANCE_KEY = "_batch"
ONTOLOGY_CANDIDATE_FORMAT = "polylogue.ontology-candidate/v1"
ONTOLOGY_GOVERNANCE_FORMAT = "polylogue.ontology-governance/v1"

OntologyBootstrapView = Literal["content", "action_pattern", "temporal_cost", "outcome"]
OntologyGovernanceDecision = Literal["accept", "rename", "split", "reject"]


class AnnotationValidationError(ValueError):
    """Raised when a candidate annotation row fails schema/shape validation."""

    def __init__(self, *, schema_id: str, target_ref: str, errors: Sequence[str]) -> None:
        self.schema_id = schema_id
        self.target_ref = target_ref
        self.errors = tuple(errors)
        message = f"annotation row for schema {schema_id!r} target {target_ref!r} failed validation: " + "; ".join(
            self.errors
        )
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class OntologyViewProposal:
    """One independently-declared bootstrap view and its candidate label."""

    view: OntologyBootstrapView
    label: str
    confidence: float
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.view not in {"content", "action_pattern", "temporal_cost", "outcome"}:
            raise ValueError(f"unsupported ontology bootstrap view: {self.view!r}")
        normalized_label = self.label.strip()
        if not normalized_label:
            raise ValueError("ontology view label cannot be empty")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("ontology view confidence must be finite and between zero and one")
        object.__setattr__(self, "label", normalized_label)
        normalized_evidence_refs = tuple(dict.fromkeys(normalize_public_ref_text(ref) for ref in self.evidence_refs))
        if not normalized_evidence_refs:
            raise ValueError("ontology view proposal requires at least one evidence ref")
        object.__setattr__(self, "evidence_refs", normalized_evidence_refs)

    def as_json_document(self) -> JSONDocument:
        return {
            "view": self.view,
            "label": self.label,
            "confidence": self.confidence,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True, slots=True)
class OntologyCandidateNomination:
    """Governed candidate-schema nomination produced by archive-local analysis.

    Every source axis is retained separately. In particular, ``source_tag_refs``
    and ``affinity`` are nomination evidence only; this request never writes an
    annotation row or an active ``annotation_schemas`` definition.
    """

    candidate_schema: AnnotationSchema
    target_ref: str
    affinity: float
    confidence: float
    classifier_ref: str
    classifier_definition: JSONDocument
    version_crosswalk: JSONDocument
    frame_ref: str
    epoch_ref: str
    privacy_policy_ref: str
    author_ref: str
    view_proposals: tuple[OntologyViewProposal, ...]
    source_tag_refs: tuple[str, ...] = ()
    residue_refs: tuple[str, ...] = ()
    rare_sample_refs: tuple[str, ...] = ()
    privacy_excluded_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    _canonical_nomination: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.candidate_schema.status != "draft":
            raise ValueError("ontology candidate schema must have status='draft'")
        if not math.isfinite(self.affinity) or not 0.0 <= self.affinity <= 1.0:
            raise ValueError("ontology candidate affinity must be finite and between zero and one")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("ontology candidate confidence must be finite and between zero and one")
        proposals = tuple(self.view_proposals)
        if not proposals:
            raise ValueError("ontology candidate requires at least one declared bootstrap view")
        if not all(isinstance(proposal, OntologyViewProposal) for proposal in proposals):
            raise TypeError("ontology candidate view_proposals must contain OntologyViewProposal values")
        views = [proposal.view for proposal in proposals]
        if len(views) != len(set(views)):
            raise ValueError("ontology candidate bootstrap views must be unique")
        object.__setattr__(self, "view_proposals", proposals)

        object.__setattr__(self, "target_ref", normalize_object_ref_text(self.target_ref))
        object.__setattr__(self, "classifier_ref", normalize_object_ref_text(self.classifier_ref))
        object.__setattr__(self, "frame_ref", normalize_object_ref_text(self.frame_ref))
        object.__setattr__(self, "epoch_ref", normalize_object_ref_text(self.epoch_ref))
        object.__setattr__(self, "privacy_policy_ref", normalize_object_ref_text(self.privacy_policy_ref))
        object.__setattr__(self, "author_ref", normalize_object_ref_text(self.author_ref))
        if ObjectRef.parse(self.author_ref).kind != "agent":
            raise ValueError("ontology candidate author_ref must identify an agent")
        object.__setattr__(
            self,
            "classifier_definition",
            _detached_json_document(self.classifier_definition, context="ontology classifier definition"),
        )
        object.__setattr__(
            self,
            "version_crosswalk",
            _detached_json_document(self.version_crosswalk, context="ontology version crosswalk"),
        )
        for field_name in (
            "source_tag_refs",
            "residue_refs",
            "rare_sample_refs",
            "privacy_excluded_refs",
            "evidence_refs",
        ):
            refs = cast(tuple[str, ...], getattr(self, field_name))
            object.__setattr__(
                self,
                field_name,
                tuple(dict.fromkeys(normalize_public_ref_text(ref) for ref in refs)),
            )
        object.__setattr__(self, "_canonical_nomination", _canonical_json_bytes(self._document_from_fields()))

    @property
    def cross_view_state(self) -> Literal["agreement", "disagreement", "insufficient"]:
        if len(self.view_proposals) < 2:
            return "insufficient"
        return "agreement" if len({proposal.label for proposal in self.view_proposals}) == 1 else "disagreement"

    def _document_from_fields(self) -> JSONDocument:
        return {
            "format": ONTOLOGY_CANDIDATE_FORMAT,
            "candidate_schema": cast(JSONValue, self.candidate_schema.definition_document()),
            "source_tag_refs": list(self.source_tag_refs),
            "affinity": self.affinity,
            "confidence": self.confidence,
            "classifier_ref": self.classifier_ref,
            "classifier_definition": dict(self.classifier_definition),
            "version_crosswalk": dict(self.version_crosswalk),
            "frame_ref": self.frame_ref,
            "epoch_ref": self.epoch_ref,
            "view_proposals": [proposal.as_json_document() for proposal in self.view_proposals],
            "cross_view_state": self.cross_view_state,
            "residue_refs": list(self.residue_refs),
            "rare_sample_refs": list(self.rare_sample_refs),
            "privacy_policy_ref": self.privacy_policy_ref,
            "privacy_excluded_refs": list(self.privacy_excluded_refs),
        }

    def as_json_document(self) -> JSONDocument:
        """Return a detached copy of the immutable nomination provenance."""

        return require_json_document(json.loads(self._canonical_nomination), context="ontology nomination")


@dataclass(frozen=True, slots=True)
class OntologyCandidateGovernance:
    """One operator decision over an ontology candidate schema."""

    candidate_ref: str
    decision: OntologyGovernanceDecision
    actor_ref: str
    reason: str | None = None
    active_schemas: tuple[AnnotationSchema, ...] = ()

    def __post_init__(self) -> None:
        parsed = ObjectRef.parse(normalize_object_ref_text(self.candidate_ref))
        if parsed.kind != "assertion" or parsed.qualifiers:
            raise ValueError("ontology governance candidate_ref must be an assertion ref")
        if self.decision not in {"accept", "rename", "split", "reject"}:
            raise ValueError(f"unsupported ontology governance decision: {self.decision!r}")
        object.__setattr__(self, "candidate_ref", parsed.format())
        object.__setattr__(self, "actor_ref", normalize_object_ref_text(self.actor_ref))
        if ObjectRef.parse(self.actor_ref).kind != "user":
            raise ValueError("ontology governance actor_ref must identify an operator user")
        if self.reason is not None and not self.reason.strip():
            raise ValueError("ontology governance reason cannot be blank")
        if self.reason is not None:
            object.__setattr__(self, "reason", self.reason.strip())
        active_schemas = tuple(self.active_schemas)
        if not all(isinstance(schema, AnnotationSchema) for schema in active_schemas):
            raise TypeError("ontology governance outputs must be AnnotationSchema values")
        if any(schema.status != "active" for schema in active_schemas):
            raise ValueError("ontology governance outputs must have status='active'")
        object.__setattr__(self, "active_schemas", active_schemas)
        identities = [schema.qualified_id for schema in active_schemas]
        if len(identities) != len(set(identities)):
            raise ValueError("ontology governance outputs must have unique schema identities")
        if self.decision in {"accept", "rename"} and len(self.active_schemas) != 1:
            raise ValueError(f"ontology governance decision {self.decision!r} requires exactly one active schema")
        if self.decision == "split" and len(self.active_schemas) < 2:
            raise ValueError("ontology governance decision 'split' requires at least two active schemas")
        if self.decision == "reject" and self.active_schemas:
            raise ValueError("ontology governance decision 'reject' cannot register active schemas")


@dataclass(frozen=True, slots=True)
class OntologyGovernanceResult:
    """Atomic judgment, registry, and governance-receipt result."""

    judgment: ArchiveAssertionJudgmentEnvelope
    governance_receipt: ArchiveAssertionEnvelope
    active_schemas: tuple[DurableAnnotationSchema, ...]


def _nfc_json_value(value: object) -> object:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_nfc_json_value(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise ValueError(f"NFC-normalized JSON keys collide at {normalized_key!r}")
            normalized[normalized_key] = _nfc_json_value(item)
        return normalized
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _nfc_json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _detached_json_document(value: object, *, context: str) -> JSONDocument:
    try:
        document = require_json_document(value, context=context)
        return require_json_document(json.loads(_canonical_json_bytes(document)), context=context)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError(f"{context} must be finite canonical JSON") from exc


@contextmanager
def _immediate_user_write(conn: sqlite3.Connection) -> Iterator[None]:
    """Own one complete preserve/read/write decision under ``BEGIN IMMEDIATE``.

    The caller must supply an idle connection. Accepting an already-open
    deferred transaction would reintroduce the judgment-preservation TOCTOU
    described by polylogue-41ow, so this path fails closed instead.
    """

    if conn.in_transaction:
        raise RuntimeError("governed ontology writes require an idle connection so they can BEGIN IMMEDIATE")
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.rollback()
        raise
    else:
        conn.commit()


def _normalized_annotation_confidence(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError("confidence must be a finite float or null")
    return 0.0 if value == 0.0 else value


def _immutable_annotation_inputs(
    *,
    scope_ref: str | None,
    target_ref: str,
    key: str | None,
    value: object,
    body_text: str | None,
    author_ref: str,
    author_kind: str,
    evidence_refs: Sequence[str],
    confidence: float | None,
) -> dict[str, object]:
    return {
        "author_kind": author_kind,
        "author_ref": author_ref,
        "body_text": body_text,
        "confidence": confidence,
        "evidence_refs": list(evidence_refs),
        "key": key,
        "kind": AssertionKind.ANNOTATION.value,
        "scope_ref": scope_ref,
        "target_ref": target_ref,
        "value": value,
    }


def assertion_id_for_schema_annotation(
    *,
    schema_qualified_id: str,
    target_ref: str,
    author_ref: str,
    row_key: str,
    batch_ref: str | None = None,
) -> str:
    """Return a deterministic assertion id for one schema-validated annotation row.

    Namespaced separately from :func:`polylogue.storage.sqlite.archive_tiers.
    user_write.assertion_id_for_annotation` (the freeform user-note helper) so
    the two annotation concepts can never collide on identity even though
    both currently write ``kind=AssertionKind.ANNOTATION`` rows.
    """

    digest = hashlib.sha256()
    identity_parts = [schema_qualified_id, target_ref, author_ref, row_key]
    if batch_ref is not None:
        identity_parts.append(normalize_object_ref_text(batch_ref))
    for part in identity_parts:
        digest.update(part.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return f"assertion-annotation-schema:{digest.hexdigest()}"


def _nomination_evidence_refs(nomination: OntologyCandidateNomination) -> tuple[str, ...]:
    refs = [
        nomination.classifier_ref,
        nomination.frame_ref,
        nomination.epoch_ref,
        nomination.privacy_policy_ref,
        *nomination.source_tag_refs,
        *nomination.residue_refs,
        *nomination.rare_sample_refs,
        *nomination.privacy_excluded_refs,
        *nomination.evidence_refs,
    ]
    for proposal in nomination.view_proposals:
        refs.extend(proposal.evidence_refs)
    return tuple(dict.fromkeys(refs))


def assertion_id_for_ontology_candidate(nomination: OntologyCandidateNomination) -> str:
    """Return the content-addressed identity for one complete nomination."""

    identity = {
        "target_ref": nomination.target_ref,
        "author_ref": nomination.author_ref,
        "value": nomination.as_json_document(),
        "evidence_refs": list(_nomination_evidence_refs(nomination)),
    }
    return f"assertion-ontology-candidate:{hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()}"


def nominate_ontology_candidate(
    conn: sqlite3.Connection,
    nomination: OntologyCandidateNomination,
    *,
    now_ms: int | None = None,
) -> ArchiveAssertionEnvelope:
    """Write one archive-specific schema nomination, never an active construct.

    The entire terminal-status read/preserve/write decision runs under
    ``BEGIN IMMEDIATE``. An identical retry returns the existing candidate or
    judged lifecycle row unchanged; it cannot resurrect a rejected/accepted
    candidate or mutate any source tag assertion.
    """

    from polylogue.storage.sqlite.archive_tiers.user_annotations import read_durable_annotation_schema

    assertion_id = assertion_id_for_ontology_candidate(nomination)
    value = nomination.as_json_document()
    evidence_refs = _nomination_evidence_refs(nomination)
    with _immediate_user_write(conn):
        existing = read_assertion_envelope(conn, assertion_id)
        if existing is not None:
            if (
                existing.kind != AssertionKind.ONTOLOGY_CANDIDATE
                or existing.target_ref != nomination.target_ref
                or existing.scope_ref != nomination.frame_ref
                or existing.author_ref != nomination.author_ref
                or existing.value != value
                or tuple(existing.evidence_refs) != evidence_refs
            ):
                raise RuntimeError(f"ontology candidate assertion identity collision: {assertion_id}")
            return existing
        durable = read_durable_annotation_schema(
            conn,
            nomination.candidate_schema.schema_id,
            nomination.candidate_schema.version,
        )
        if durable is not None:
            raise ValueError(
                f"ontology candidate {nomination.candidate_schema.qualified_id!r} already has a durable schema row; "
                "propose a new version"
            )
        return upsert_assertion(
            conn,
            assertion_id=assertion_id,
            scope_ref=nomination.frame_ref,
            target_ref=nomination.target_ref,
            key=nomination.candidate_schema.qualified_id,
            kind=AssertionKind.ONTOLOGY_CANDIDATE,
            value=value,
            body_text=nomination.candidate_schema.title,
            author_ref=nomination.author_ref,
            author_kind="agent",
            evidence_refs=evidence_refs,
            status=AssertionStatus.CANDIDATE,
            visibility=AssertionVisibility.PRIVATE,
            confidence=nomination.confidence,
            context_policy={"inject": False, "promotion_required": True},
            now_ms=now_ms,
        )


def assertion_id_for_ontology_governance(
    *,
    candidate_ref: str,
    decision: OntologyGovernanceDecision,
    active_schemas: Sequence[AnnotationSchema],
) -> str:
    """Return a deterministic receipt id for one candidate decision/output set."""

    identity = {
        "candidate_ref": normalize_object_ref_text(candidate_ref),
        "decision": decision,
        "active_schema_fingerprints": [schema.definition_fingerprint for schema in active_schemas],
    }
    return f"assertion-ontology-governance:{hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()}"


def _ontology_candidate_schema(candidate: ArchiveAssertionEnvelope) -> AnnotationSchema:
    if candidate.kind != AssertionKind.ONTOLOGY_CANDIDATE:
        raise ValueError(f"assertion:{candidate.assertion_id} is not an ontology candidate")
    if not isinstance(candidate.value, dict) or candidate.value.get("format") != ONTOLOGY_CANDIDATE_FORMAT:
        raise ValueError(f"assertion:{candidate.assertion_id} has malformed ontology candidate provenance")
    definition = candidate.value.get("candidate_schema")
    schema = AnnotationSchema.from_definition_document(definition)
    if schema.status != "draft":
        raise ValueError(f"assertion:{candidate.assertion_id} candidate schema is not draft")
    return schema


def _governance_replacement_value(
    candidate: ArchiveAssertionEnvelope,
    governance: OntologyCandidateGovernance,
) -> JSONDocument:
    value = require_json_document(candidate.value, context="ontology candidate value")
    replacement_value = dict(value)
    replacement_value["governance_decision"] = governance.decision
    replacement_value["governance_output_schemas"] = [
        cast(JSONValue, schema.definition_document()) for schema in governance.active_schemas
    ]
    return replacement_value


def govern_ontology_candidate(
    conn: sqlite3.Connection,
    governance: OntologyCandidateGovernance,
    *,
    now_ms: int | None = None,
) -> OntologyGovernanceResult:
    """Judge a schema candidate and atomically register governed active versions.

    ``accept`` preserves the draft definition exactly except for its status.
    ``rename`` and ``split`` record explicit supersession outputs. ``reject``
    leaves every source tag and evidence row intact. All decisions reuse the
    ordinary candidate judgment lifecycle and add a deterministic typed
    governance receipt; no formal annotation membership is written here.
    """

    from polylogue.storage.sqlite.archive_tiers.user_annotations import persist_annotation_schema

    timestamp = now_ms if now_ms is not None else int(time.time() * 1000)
    candidate_id = ObjectRef.parse(governance.candidate_ref).object_id
    with _immediate_user_write(conn):
        candidate = read_assertion_envelope(conn, candidate_id)
        if candidate is None:
            raise ValueError(f"ontology candidate not found: {governance.candidate_ref}")
        draft_schema = _ontology_candidate_schema(candidate)
        expected_active = replace(draft_schema, status="active")
        if governance.decision == "accept":
            if governance.active_schemas[0].canonical_definition_json() != expected_active.canonical_definition_json():
                raise ValueError(
                    "ontology accept must preserve the candidate schema definition exactly except for status='active'"
                )
        elif (
            governance.decision == "rename"
            and governance.active_schemas[0].canonical_definition_json() == expected_active.canonical_definition_json()
        ):
            raise ValueError("ontology rename must change the accepted schema definition")

        judgment_decision = "reject" if governance.decision == "reject" else "accept"
        replacement_value: JSONDocument | None = None
        if governance.decision in {"rename", "split"}:
            judgment_decision = "supersede"
            replacement_value = _governance_replacement_value(candidate, governance)

        judgment = judge_assertion_candidate(
            conn,
            candidate_ref=governance.candidate_ref,
            decision=judgment_decision,
            reason=governance.reason,
            actor_ref=governance.actor_ref,
            inject=False,
            replacement_value=replacement_value,
            now_ms=timestamp,
        )
        persisted = tuple(
            persist_annotation_schema(conn, schema, registered_at_ms=timestamp) for schema in governance.active_schemas
        )
        judgment_ref = f"assertion:{judgment.judgment.assertion_id}"
        resulting_ref = (
            None if judgment.resulting_assertion is None else f"assertion:{judgment.resulting_assertion.assertion_id}"
        )
        candidate_value = require_json_document(candidate.value, context="ontology candidate value")
        source_tag_refs_raw = candidate_value.get("source_tag_refs", [])
        source_tag_refs = (
            [str(ref) for ref in source_tag_refs_raw]
            if isinstance(source_tag_refs_raw, list) and all(isinstance(ref, str) for ref in source_tag_refs_raw)
            else []
        )
        receipt_value: JSONDocument = {
            "format": ONTOLOGY_GOVERNANCE_FORMAT,
            "decision": governance.decision,
            "candidate_ref": governance.candidate_ref,
            "judgment_ref": judgment_ref,
            "resulting_candidate_ref": resulting_ref,
            "active_schemas": [
                {
                    "qualified_id": durable.schema.qualified_id,
                    "definition_sha256": durable.definition_sha256,
                }
                for durable in persisted
            ],
            "source_tag_refs": cast("list[JSONValue]", source_tag_refs),
            "affinity": candidate_value.get("affinity"),
            "confidence": candidate_value.get("confidence"),
            "classifier_ref": candidate_value.get("classifier_ref"),
            "version_crosswalk": candidate_value.get("version_crosswalk"),
            "frame_ref": candidate_value.get("frame_ref"),
            "epoch_ref": candidate_value.get("epoch_ref"),
            "cross_view_state": candidate_value.get("cross_view_state"),
            "privacy_policy_ref": candidate_value.get("privacy_policy_ref"),
            "annotation_batch_required": governance.decision != "reject",
        }
        receipt_id = assertion_id_for_ontology_governance(
            candidate_ref=governance.candidate_ref,
            decision=governance.decision,
            active_schemas=governance.active_schemas,
        )
        receipt_evidence = list(
            dict.fromkeys(
                (
                    *candidate.evidence_refs,
                    governance.candidate_ref,
                    judgment_ref,
                    *((resulting_ref,) if resulting_ref is not None else ()),
                )
            )
        )
        existing_receipt = read_assertion_envelope(conn, receipt_id)
        if existing_receipt is not None:
            if (
                existing_receipt.kind != AssertionKind.ONTOLOGY_GOVERNANCE
                or existing_receipt.value != receipt_value
                or existing_receipt.author_ref != governance.actor_ref
                or existing_receipt.evidence_refs != receipt_evidence
            ):
                raise RuntimeError(f"ontology governance receipt identity collision: {receipt_id}")
            receipt = existing_receipt
        else:
            receipt = upsert_assertion(
                conn,
                assertion_id=receipt_id,
                scope_ref=judgment_ref,
                target_ref=governance.candidate_ref,
                key=f"{governance.decision}/{draft_schema.qualified_id}",
                kind=AssertionKind.ONTOLOGY_GOVERNANCE,
                value=receipt_value,
                body_text=governance.reason,
                author_ref=governance.actor_ref,
                author_kind="user",
                evidence_refs=receipt_evidence,
                status=AssertionStatus.ACTIVE,
                visibility=AssertionVisibility.PRIVATE,
                context_policy={"inject": False},
                now_ms=timestamp,
            )
        return OntologyGovernanceResult(
            judgment=judgment,
            governance_receipt=receipt,
            active_schemas=persisted,
        )


def upsert_annotation_assertion(
    conn: sqlite3.Connection,
    *,
    schema: AnnotationSchema,
    registry: AnnotationSchemaRegistry = ANNOTATION_SCHEMA_REGISTRY,
    target_ref: str,
    value: Mapping[str, object],
    row_key: str,
    evidence_refs: Sequence[str] = (),
    author_ref: str,
    author_kind: str = "agent",
    confidence: float | None = None,
    body_text: str | None = None,
    batch_ref: str | None = None,
    now_ms: int | None = None,
) -> ArchiveAssertionEnvelope:
    """Validate one label against *schema* and upsert it as a candidate assertion.

    The schema must match an active entry in *registry*. Raises
    :class:`~polylogue.annotations.schema.AnnotationSchemaError` for missing,
    drifted, draft, or deprecated registrations, and
    :class:`AnnotationValidationError` (writing nothing) when the row fails
    target-grain, field-shape, or evidence-policy validation.
    On success, delegates to ``upsert_assertion`` with
    ``kind=AssertionKind.ANNOTATION``; the resulting row's status/context
    policy is decided by that function's own author-kind chokepoint, not by
    this caller. When ``batch_ref`` is supplied, it must resolve to durable
    provenance for the same schema, target, actor, and declared assertion id.
    """

    registered_schema = registry.require_active(schema)
    errors = validate_annotation_row(
        registered_schema,
        target_ref=target_ref,
        value=value,
        evidence_refs=evidence_refs,
    )
    if errors:
        raise AnnotationValidationError(
            schema_id=registered_schema.qualified_id,
            target_ref=target_ref,
            errors=errors,
        )

    normalized_target_ref = normalize_object_ref_text(target_ref)
    normalized_author_ref = normalize_object_ref_text(author_ref)
    normalized_evidence_refs = tuple(normalize_public_ref_text(ref) for ref in evidence_refs)
    try:
        normalized_confidence = _normalized_annotation_confidence(confidence)
    except ValueError as exc:
        raise AnnotationValidationError(
            schema_id=registered_schema.qualified_id,
            target_ref=normalized_target_ref,
            errors=(str(exc),),
        ) from exc
    normalized_batch_ref: str | None = None
    durable_batch = None
    if batch_ref is not None:
        from polylogue.storage.sqlite.archive_tiers.user_annotations import (
            read_annotation_batch,
            read_durable_annotation_schema,
        )

        batch_errors: list[str] = []
        try:
            normalized_batch_ref = normalize_object_ref_text(batch_ref)
            parsed_batch_ref = ObjectRef.parse(normalized_batch_ref)
        except ValueError:
            parsed_batch_ref = None
            batch_errors.append("batch_ref must be a valid ObjectRef")
        if parsed_batch_ref is not None and parsed_batch_ref.kind != "annotation-batch":
            batch_errors.append("batch_ref must use the 'annotation-batch' ObjectRef kind")
        durable_batch = (
            read_annotation_batch(conn, parsed_batch_ref.object_id)
            if parsed_batch_ref is not None and not batch_errors
            else None
        )
        if not batch_errors and durable_batch is None:
            batch_errors.append(f"batch_ref {normalized_batch_ref!r} does not resolve to durable batch provenance")
        if durable_batch is not None and durable_batch.qualified_schema_id != registered_schema.qualified_id:
            batch_errors.append(
                f"batch_ref {normalized_batch_ref!r} uses schema {durable_batch.qualified_schema_id!r}, "
                f"not {registered_schema.qualified_id!r}"
            )
        if durable_batch is not None and durable_batch.target_ref != normalized_target_ref:
            batch_errors.append(
                f"batch_ref {normalized_batch_ref!r} targets {durable_batch.target_ref!r}, "
                f"not {normalized_target_ref!r}"
            )
        if durable_batch is not None and durable_batch.actor_ref != normalized_author_ref:
            batch_errors.append(
                f"batch_ref {normalized_batch_ref!r} records actor {durable_batch.actor_ref!r}, "
                f"not {normalized_author_ref!r}"
            )
        durable_schema = (
            read_durable_annotation_schema(conn, durable_batch.schema_id, durable_batch.schema_version)
            if durable_batch is not None
            else None
        )
        if durable_batch is not None and durable_schema is None:
            batch_errors.append(f"batch_ref {normalized_batch_ref!r} does not resolve its durable schema definition")
        if (
            durable_schema is not None
            and durable_schema.definition_json != registered_schema.canonical_definition_json()
        ):
            batch_errors.append(
                f"batch_ref {normalized_batch_ref!r} resolves a durable schema definition that differs from the writer"
            )
        if batch_errors:
            raise AnnotationValidationError(
                schema_id=registered_schema.qualified_id,
                target_ref=normalized_target_ref,
                errors=batch_errors,
            )
    assertion_id = assertion_id_for_schema_annotation(
        schema_qualified_id=registered_schema.qualified_id,
        target_ref=normalized_target_ref,
        author_ref=normalized_author_ref,
        row_key=row_key,
        batch_ref=normalized_batch_ref,
    )
    if durable_batch is not None and f"assertion:{assertion_id}" not in durable_batch.assertion_refs:
        raise AnnotationValidationError(
            schema_id=registered_schema.qualified_id,
            target_ref=normalized_target_ref,
            errors=(f"batch_ref {normalized_batch_ref!r} does not declare assertion:{assertion_id}",),
        )
    stamped_value: dict[str, object] = {
        _SCHEMA_PROVENANCE_KEY: registered_schema.qualified_id,
        **dict(value),
    }
    if normalized_batch_ref is not None:
        stamped_value[_BATCH_PROVENANCE_KEY] = normalized_batch_ref

    candidate_inputs = _immutable_annotation_inputs(
        scope_ref=normalized_batch_ref,
        target_ref=normalized_target_ref,
        key=row_key,
        value=stamped_value,
        body_text=body_text,
        author_ref=normalized_author_ref,
        author_kind=author_kind,
        evidence_refs=normalized_evidence_refs,
        confidence=normalized_confidence,
    )
    try:
        _canonical_json_bytes(candidate_inputs)
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise AnnotationValidationError(
            schema_id=registered_schema.qualified_id,
            target_ref=normalized_target_ref,
            errors=("annotation immutable inputs must be finite canonical JSON",),
        ) from exc

    existing = read_assertion_envelope(conn, assertion_id)
    if existing is not None and normalized_batch_ref is not None:
        existing_inputs = _immutable_annotation_inputs(
            scope_ref=existing.scope_ref,
            target_ref=existing.target_ref,
            key=existing.key,
            value=existing.value,
            body_text=existing.body_text,
            author_ref=existing.author_ref or "",
            author_kind=existing.author_kind or "",
            evidence_refs=existing.evidence_refs,
            confidence=existing.confidence,
        )
        try:
            drifted_fields = [
                field
                for field in candidate_inputs
                if _canonical_json_bytes(candidate_inputs[field]) != _canonical_json_bytes(existing_inputs[field])
            ]
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            raise AnnotationValidationError(
                schema_id=registered_schema.qualified_id,
                target_ref=normalized_target_ref,
                errors=(f"assertion_id {assertion_id!r} has non-canonical stored immutable inputs",),
            ) from exc
        if drifted_fields:
            raise AnnotationValidationError(
                schema_id=registered_schema.qualified_id,
                target_ref=normalized_target_ref,
                errors=(f"assertion_id {assertion_id!r} immutable input drift: {sorted(drifted_fields)}",),
            )
        return existing

    return upsert_assertion(
        conn,
        assertion_id=assertion_id,
        target_ref=normalized_target_ref,
        kind=AssertionKind.ANNOTATION,
        scope_ref=normalized_batch_ref,
        key=row_key,
        value=stamped_value,
        body_text=body_text,
        author_ref=normalized_author_ref,
        author_kind=author_kind,
        evidence_refs=normalized_evidence_refs,
        confidence=normalized_confidence,
        now_ms=now_ms,
    )


__all__ = [
    "ONTOLOGY_CANDIDATE_FORMAT",
    "ONTOLOGY_GOVERNANCE_FORMAT",
    "AnnotationValidationError",
    "OntologyBootstrapView",
    "OntologyCandidateGovernance",
    "OntologyCandidateNomination",
    "OntologyGovernanceDecision",
    "OntologyGovernanceResult",
    "OntologyViewProposal",
    "assertion_id_for_ontology_candidate",
    "assertion_id_for_ontology_governance",
    "assertion_id_for_schema_annotation",
    "govern_ontology_candidate",
    "nominate_ontology_candidate",
    "upsert_annotation_assertion",
]

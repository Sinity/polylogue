"""Typed durable context images and their canonical content identity.

The compiler and delivery boundary share these exact carriers. Their stored
shape does not depend on a renderer or on the query implementation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from polylogue.analysis.archive_models import ArchiveInsightModel
from polylogue.core.assertions import AssertionContextTrustClass
from polylogue.core.digest import REFERENCE, canonical_bytes, digest
from polylogue.core.refs import EvidenceRef, ExecutionContextRef, ObjectRef
from polylogue.surfaces.projection_spec import QueryProjectionSpec

ContextPurpose = Literal["continue", "review", "handoff", "debug", "export"]
ContextSegmentKind = Literal["read_view", "query_unit", "assertion", "caveat"]
ContextSegmentProfile = Literal["default", "prose_with_refs"]
ContextOmissionReason = Literal["budget", "unsupported", "not_found", "policy", "redacted", "missing_evidence"]
DEFAULT_CONTEXT_IMAGE_MAX_MESSAGES_PER_SESSION = 24
DEFAULT_CONTEXT_IMAGE_MAX_CHARS_PER_MESSAGE = 1_800


class ContextSegment(ArchiveInsightModel):
    """One bounded section of a compiled context image."""

    segment_id: str
    kind: ContextSegmentKind
    title: str
    markdown: str | None = None
    payload_kind: str | None = None
    object_refs: tuple[ObjectRef, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    assertion_refs: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()
    token_estimate: int = 0
    lossiness: str | None = None
    # Only populated for kind="assertion" segments (polylogue-x2y9). Assertion
    # prose flows in from providers, gets judged/summarised by agents, and can
    # flow back out into another agent's compiled context -- this label is
    # what lets a consumer distinguish injected evidence from surrounding
    # instruction-grade text instead of trusting it by convention.
    trust_class: AssertionContextTrustClass | None = None


class ContextOmission(ArchiveInsightModel):
    """A requested context input that was omitted deliberately."""

    ref: str | None = None
    query: str | None = None
    view: str | None = None
    reason: ContextOmissionReason
    detail: str


class ContextSpec(ArchiveInsightModel):
    """Declarative request for a compiled context image.

    The spec is pure input. It does not imply delivery, durable recording, or
    automatic context injection.
    """

    purpose: ContextPurpose = "continue"
    seed_query: str | None = None
    seed_query_limit: int = Field(default=5, ge=1, le=50)
    seed_project_path: str | None = None
    seed_project_repo: str | None = None
    seed_since: str | None = None
    seed_until: str | None = None
    seed_origin: str | None = None
    seed_refs: tuple[str, ...] = ()
    read_views: tuple[str, ...] = ("messages",)
    unit_queries: tuple[str, ...] = ()
    unit_query_limit: int = Field(default=20, ge=1, le=200)
    max_tokens: int | None = Field(default=None, ge=1)
    max_messages_per_session: int | None = Field(default=None, ge=1, le=500)
    max_chars_per_message: int | None = Field(default=None, ge=1, le=20_000)
    include_assertions: bool = True
    include_candidates: bool = False
    redaction_policy: Literal["default", "raw-opt-in"] = "default"
    segment_profile: ContextSegmentProfile = "default"

    @model_validator(mode="after")
    def _requires_seed(self) -> ContextSpec:
        if self.seed_query is None and not self.seed_refs and not self.unit_queries and not _has_seed_filters(self):
            raise ValueError("ContextSpec requires seed_query, seed_refs, or unit_queries")
        return self


def _has_seed_filters(spec: ContextSpec) -> bool:
    return any(
        (
            spec.seed_project_path,
            spec.seed_project_repo,
            spec.seed_since,
            spec.seed_until,
            spec.seed_origin,
        )
    )


class ContextImage(ArchiveInsightModel):
    """Compiled, storage-free context payload over archive refs and views."""

    spec: ContextSpec
    projection_spec: QueryProjectionSpec | None = None
    selection_strategy: str = "context_spec_v1"
    redaction_policy: str = "default"
    segments: tuple[ContextSegment, ...]
    object_refs: tuple[ObjectRef, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    assertion_refs: tuple[str, ...] = ()
    omitted: tuple[ContextOmission, ...] = ()
    caveats: tuple[str, ...] = ()
    token_estimate: int = 0
    size_estimate: dict[str, int] = Field(default_factory=dict)
    execution_context_ref: ExecutionContextRef | None = None
    ledger: tuple[object, ...] = ()
    build_ref: str | None = None


class ContextSnapshotRecord(ArchiveInsightModel):
    """Evidence record for context that was actually delivered."""

    snapshot_ref: str
    run_ref: str | None = None
    boundary: str
    inheritance_mode: str = "explicit"
    segment_refs: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    metadata: dict[str, str] = Field(default_factory=dict)


def canonical_context_image_json(image: ContextImage) -> str:
    """Serialize one compiled image deterministically for delivery evidence."""

    return canonical_bytes(image.model_dump(mode="json"), REFERENCE).decode("utf-8")


def context_image_sha256(image: ContextImage) -> str:
    """Return the content identity of the complete compiled image."""

    return digest(image.model_dump(mode="json"), REFERENCE)


__all__ = [
    "ContextImage",
    "ContextOmission",
    "ContextOmissionReason",
    "ContextPurpose",
    "ContextSegment",
    "ContextSegmentKind",
    "ContextSegmentProfile",
    "ContextSnapshotRecord",
    "ContextSpec",
    "DEFAULT_CONTEXT_IMAGE_MAX_CHARS_PER_MESSAGE",
    "DEFAULT_CONTEXT_IMAGE_MAX_MESSAGES_PER_SESSION",
    "canonical_context_image_json",
    "context_image_sha256",
]

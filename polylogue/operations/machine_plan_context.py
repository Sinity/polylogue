"""Closed replay semantics for declared machine mutation families.

Only operation meaning enters this versioned record. Transport envelopes,
credentials, arbitrary plan extensions and runtime objects are not replay data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _DeleteContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    session_ids: list[str]


class _IdentityResetContext(_DeleteContext):
    present_in_index: list[str]
    reason: str

    @model_validator(mode="after")
    def exact_presence(self) -> _IdentityResetContext:
        if len(set(self.session_ids)) != len(self.session_ids):
            raise ValueError("reset replay repeats a suppression target")
        present = set(self.present_in_index)
        if self.present_in_index != [value for value in self.session_ids if value in present]:
            raise ValueError("reset replay presence is not an ordered target subset")
        return self


class _BulkContext(_DeleteContext):
    requested_session_ids: list[str]
    unresolved_session_ids: list[str]
    requested_session_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_request_evidence(self) -> _BulkContext:
        if self.requested_session_count != len(self.requested_session_ids):
            raise ValueError("bulk replay request count must match its original IDs")
        requested = list(dict.fromkeys(self.requested_session_ids))
        unresolved = set(self.unresolved_session_ids)
        if self.unresolved_session_ids != [value for value in requested if value in unresolved]:
            raise ValueError("bulk replay gaps must be distinct ordered original request IDs")
        if self.session_ids != [value for value in requested if value not in unresolved]:
            raise ValueError("bulk replay effect targets must be the exact request partition")
        return self


class _TagContext(_BulkContext):
    tags: list[str]
    author_ref: str | None
    author_kind: str | None


class _MetadataContext(_BulkContext):
    pairs: list[list[object]]

    @model_validator(mode="after")
    def validate_pairs(self) -> _MetadataContext:
        for pair in self.pairs:
            if len(pair) != 2 or not isinstance(pair[0], str) or not pair[0]:
                raise ValueError("metadata replay requires exact key/value pairs")
            # Existing metadata operations accept JSON values. This field is
            # their intended durable value, not a generic operation payload.
            json.dumps(pair[1], allow_nan=False)
        return self


class _MarkContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    target_type: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    mark_type: str = Field(min_length=1)
    owner_session_id: str | None


class _AnnotationContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    annotation_id: str = Field(min_length=1)
    target_type: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    note_text: str = Field(min_length=1)
    owner_session_id: str | None


class _TagRemoveContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    session_id: str = Field(min_length=1)
    tag: str = Field(min_length=1)


class _InsightTargetContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    target_ref: str = Field(min_length=9)
    disposition: Literal["required", "excess"]

    @model_validator(mode="after")
    def validate_target(self) -> _InsightTargetContext:
        if not self.target_ref.startswith("session:"):
            raise ValueError("insight replay target must be an exact session reference")
        return self


class _InsightContext(BaseModel):
    """Only frozen insight acceptance facts survive source-WAL replay."""

    model_config = ConfigDict(extra="forbid", strict=True)

    scope_kind: Literal["explicit", "full"]
    index_generation: str = Field(min_length=1)
    recipe_version: str = Field(min_length=1)
    page_ordinal: int = Field(ge=0)
    page_count: int = Field(ge=1)
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    previous_preview_ref: str | None = None
    targets: list[_InsightTargetContext] = Field(max_length=256)

    @model_validator(mode="after")
    def validate_page(self) -> _InsightContext:
        if not self.index_generation.startswith("index-generation:"):
            raise ValueError("insight replay generation must be the pinned index-generation token")
        if self.page_ordinal >= self.page_count:
            raise ValueError("insight replay page ordinal is outside its count")
        if (self.page_ordinal == 0) != (self.previous_preview_ref is None):
            raise ValueError("insight replay predecessor does not match page ordinal")
        return self


class _IngestContext(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_generation_id: str = Field(min_length=1)
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    custody_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    enumeration_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_count: int = Field(ge=1)
    recipe_version: str = Field(min_length=1)
    source_name: str | None = Field(default=None, min_length=1, max_length=255)


class _ExcisionContext(BaseModel):
    """Frozen closure coordinates survive Source-first removal and rebuild."""

    model_config = ConfigDict(extra="forbid", strict=True)

    session_id: str
    actor: str
    found: bool
    reason: str
    cascade_lineage: bool
    lineage_dependent_session_ids: list[str]
    source_marker_inputs_pending: int = Field(ge=0)
    source_marker_inputs_accepted: int = Field(ge=0)
    marker_input_digests: list[str]
    targets: list[dict[str, object]]
    user_frame_epoch: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_targets(self) -> _ExcisionContext:
        from polylogue.security.excision import excision_target_from_replay, excision_target_replay

        targets = [excision_target_from_replay(value) for value in self.targets]
        expected = [*self.lineage_dependent_session_ids, self.session_id]
        if [target.session_id for target in targets] != expected or len(set(expected)) != len(expected):
            raise ValueError("excision replay requires the exact ordered original cascade")
        if not self.cascade_lineage and self.lineage_dependent_session_ids:
            raise ValueError("excision replay cannot widen a noncascade plan")
        if targets[-1].found != self.found:
            raise ValueError("excision replay existence must match its frozen target")
        if any(target.session_exists and target.session_content_hash is None for target in targets):
            raise ValueError("excision replay requires the original session content identity")
        self.targets = [excision_target_replay(target) for target in targets]
        return self


_CONTEXT_MODELS: dict[str, type[BaseModel]] = {
    "mutate-delete-session": _DeleteContext,
    "mutate-identity-reset": _IdentityResetContext,
    "mutate-session-excision": _ExcisionContext,
    "mutate-add-mark": _MarkContext,
    "mutate-remove-mark": _MarkContext,
    "mutate-save-annotation": _AnnotationContext,
    "mutate-remove-tag": _TagRemoveContext,
    "mutate-bulk-tag-sessions": _TagContext,
    "mutate-bulk-set-metadata": _MetadataContext,
    "mutate-rebuild-insights": _InsightContext,
    "ingest-archive-runtime": _IngestContext,
}
_FORMAT = "polylogue.machine-plan-context/v1"


def replay_context(operation: str, context: Mapping[str, object]) -> dict[str, object] | None:
    """Encode only a declared context, refusing unregistered extension fields."""
    model = _CONTEXT_MODELS.get(operation)
    if model is None:
        return None
    decoded = model.model_validate(dict(context)).model_dump(mode="json")
    if operation == "ingest-archive-runtime":
        for field in ("custody_digest", "source_name"):
            if decoded.get(field) is None:
                decoded.pop(field)
    return {
        "format": _FORMAT,
        "operation": operation,
        "context": decoded,
    }


def context_from_replay(operation: str, value: object) -> dict[str, object]:
    """Decode without guessing missing or previously redacted operation meaning."""
    if (
        not isinstance(value, dict)
        or set(value) != {"format", "operation", "context"}
        or value["format"] != _FORMAT
        or value["operation"] != operation
        or operation not in _CONTEXT_MODELS
        or not isinstance(value["context"], dict)
    ):
        raise ValueError("unsupported machine plan replay context")
    decoded = _CONTEXT_MODELS[operation].model_validate(value["context"]).model_dump(mode="json")
    if operation == "ingest-archive-runtime":
        for field in ("custody_digest", "source_name"):
            if decoded.get(field) is None:
                decoded.pop(field)
    return decoded

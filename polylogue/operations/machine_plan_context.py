"""Closed replay semantics for the three declared machine mutation families.

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

    session_ids: list[str] = Field(max_length=256)


class _TagContext(_DeleteContext):
    tags: list[str]
    requested_session_count: int = Field(ge=0, le=10_000)


class _MetadataContext(_DeleteContext):
    pairs: list[list[object]]
    requested_session_count: int = Field(ge=0, le=10_000)

    @model_validator(mode="after")
    def validate_pairs(self) -> _MetadataContext:
        for pair in self.pairs:
            if len(pair) != 2 or not isinstance(pair[0], str) or not pair[0]:
                raise ValueError("metadata replay requires exact key/value pairs")
            # Existing metadata operations accept JSON values. This field is
            # their intended durable value, not a generic operation payload.
            json.dumps(pair[1], allow_nan=False)
        return self


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
    page_ordinal: int = Field(ge=0, le=4095)
    page_count: int = Field(ge=1, le=4096)
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


_CONTEXT_MODELS: dict[str, type[BaseModel]] = {
    "mutate-delete-session": _DeleteContext,
    "mutate-bulk-tag-sessions": _TagContext,
    "mutate-bulk-set-metadata": _MetadataContext,
    "mutate-rebuild-insights": _InsightContext,
}
_FORMAT = "polylogue.machine-plan-context/v1"


def replay_context(operation: str, context: Mapping[str, object]) -> dict[str, object] | None:
    """Encode only a declared context, refusing unregistered extension fields."""
    model = _CONTEXT_MODELS.get(operation)
    if model is None:
        return None
    return {
        "format": _FORMAT,
        "operation": operation,
        "context": model.model_validate(dict(context)).model_dump(mode="json"),
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
    return _CONTEXT_MODELS[operation].model_validate(value["context"]).model_dump(mode="json")

"""Closed replay semantics for the three declared machine mutation families.

Only operation meaning enters this versioned record. Transport envelopes,
credentials, arbitrary plan extensions and runtime objects are not replay data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

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


_CONTEXT_MODELS: dict[str, type[_DeleteContext]] = {
    "mutate-delete-session": _DeleteContext,
    "mutate-bulk-tag-sessions": _TagContext,
    "mutate-bulk-set-metadata": _MetadataContext,
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

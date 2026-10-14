"""Shared immutable JSON payload models and serialization."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from polylogue.core.json import JSONDocument, require_json_document


def serialize_surface_payload(payload: BaseModel, *, exclude_none: bool = False) -> str:
    """Serialize a surface payload model with stable JSON formatting."""
    return payload.model_dump_json(indent=2, exclude_none=exclude_none)


def model_json_document(payload: BaseModel, *, exclude_none: bool = False) -> JSONDocument:
    """Return a model dump constrained to the shared JSON document type."""
    return require_json_document(
        payload.model_dump(mode="json", exclude_none=exclude_none),
        context=f"{payload.__class__.__name__} JSON payload",
    )


class SurfacePayloadModel(BaseModel):
    """Shared base for immutable JSON payload models exposed by surfaces."""

    # Surface fields such as ``model_name`` and ``model_ref`` are payload data,
    # not pydantic API; generated projection models inherit this setting.
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    def to_json(self, *, exclude_none: bool = False) -> str:
        return serialize_surface_payload(self, exclude_none=exclude_none)

    @classmethod
    def _from_row_generic(cls, row: object, **field_overrides: object) -> Self:
        """Generic from_row implementation: copy matching fields from row, allow overrides.

        This implements the pattern used across 70+ simple field assignments in payloads:
        - For each model field, try to get the value from row if it exists
        - Allow explicit overrides for fields that need transformations or renames
        - Ignore row fields that don't exist on the model

        Args:
            row: A row object (dataclass, namedtuple, or similar) with field attributes
            **field_overrides: Explicit field assignments to override defaults

        Returns:
            Instance of cls with fields populated from row + overrides

        Example:
            return cls._from_row_generic(row, kind=AssertionKind.from_string(row.kind))
        """
        # Collect all fields from row that match model fields
        payload_fields = {}
        for field_name in cls.model_fields:
            if hasattr(row, field_name):
                payload_fields[field_name] = getattr(row, field_name)
        # Apply overrides
        payload_fields.update(field_overrides)
        return cls(**payload_fields)

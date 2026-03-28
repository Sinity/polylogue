"""Schema-driven synthetic conversation generator."""

from __future__ import annotations

from typing import Any

from polylogue.schemas.runtime_registry import SchemaRegistry, canonical_schema_provider
from polylogue.schemas.synthetic import selection as _selection
from polylogue.schemas.synthetic.builders import (
    _ensure_wire_chatgpt,
    _ensure_wire_claude_ai,
    _ensure_wire_claude_code,
    _ensure_wire_codex,
    _ensure_wire_format,
    _ensure_wire_gemini,
    _generate_conversation,
    _generate_jsonl_records,
    _generate_linear_json,
    _generate_tree_json,
    _role_cycle,
    generate_batch,
)
from polylogue.schemas.synthetic.models import (
    SyntheticGenerationBatch,
    SyntheticGenerationReport,
    SyntheticSchemaSelection,
)
from polylogue.schemas.synthetic.relations import RelationConstraintSolver
from polylogue.schemas.synthetic.runtime import (
    _generate_array,
    _generate_from_schema,
    _generate_number,
    _generate_object,
    _generate_string,
    _serialize,
)
from polylogue.schemas.synthetic.selection import available_synthetic_providers, select_synthetic_schema
from polylogue.schemas.synthetic.wire_formats import WireFormat


def _sync_selection_patch_surfaces() -> None:
    _selection.SchemaRegistry = SchemaRegistry
    _selection.canonical_schema_provider = canonical_schema_provider


class SyntheticCorpus:
    """Generate synthetic provider data from annotated schemas."""

    def __init__(
        self,
        schema: dict[str, Any],
        wire_format: WireFormat,
        provider: str,
        *,
        package_version: str = "default",
        element_kind: str | None = None,
    ):
        self.schema = schema
        self.wire_format = wire_format
        self.provider = provider
        self.package_version = package_version
        self.element_kind = element_kind
        self._relation_solver = RelationConstraintSolver(schema)

    @classmethod
    def for_provider(
        cls,
        provider: str,
        *,
        version: str = "default",
        element_kind: str | None = None,
    ) -> SyntheticCorpus:
        _sync_selection_patch_surfaces()
        selection = select_synthetic_schema(
            provider,
            version=version,
            element_kind=element_kind,
        )
        return cls.from_selection(selection)

    @classmethod
    def from_selection(cls, selection: SyntheticSchemaSelection) -> SyntheticCorpus:
        return cls(
            selection.schema,
            selection.wire_format,
            selection.provider,
            package_version=selection.package_version,
            element_kind=selection.element_kind,
        )

    @classmethod
    def available_providers(cls) -> list[str]:
        _sync_selection_patch_surfaces()
        return available_synthetic_providers()

    def generate_batch(
        self,
        count: int = 5,
        messages_per_conversation: range = range(3, 15),
        seed: int | None = None,
        style: str = "default",
    ) -> SyntheticGenerationBatch:
        return generate_batch(
            self,
            count=count,
            messages_per_conversation=messages_per_conversation,
            seed=seed,
            style=style,
        )

    def generate(
        self,
        count: int = 5,
        messages_per_conversation: range = range(3, 15),
        seed: int | None = None,
        style: str = "default",
    ) -> list[bytes]:
        return self.generate_batch(
            count=count,
            messages_per_conversation=messages_per_conversation,
            seed=seed,
            style=style,
        ).raw_items

    _ensure_wire_chatgpt = _ensure_wire_chatgpt
    _ensure_wire_claude_ai = _ensure_wire_claude_ai
    _ensure_wire_claude_code = _ensure_wire_claude_code
    _ensure_wire_codex = _ensure_wire_codex
    _ensure_wire_format = _ensure_wire_format
    _ensure_wire_gemini = _ensure_wire_gemini
    _generate_array = _generate_array
    _generate_conversation = _generate_conversation
    _generate_from_schema = _generate_from_schema
    _generate_jsonl_records = _generate_jsonl_records
    _generate_linear_json = _generate_linear_json
    _generate_number = _generate_number
    _generate_object = _generate_object
    _generate_string = _generate_string
    _generate_tree_json = _generate_tree_json
    _role_cycle = _role_cycle
    _serialize = _serialize


__all__ = [
    "SyntheticCorpus",
    "SyntheticGenerationBatch",
    "SyntheticGenerationReport",
]

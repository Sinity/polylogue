"""Schema-driven synthetic conversation generator."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from polylogue.scenarios import CorpusSpec
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
    SyntheticWrittenBatch,
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
    def from_spec(cls, spec: CorpusSpec) -> SyntheticCorpus:
        return cls.for_provider(
            spec.provider,
            version=spec.package_version,
            element_kind=spec.element_kind,
        )

    @classmethod
    def available_providers(cls) -> list[str]:
        _sync_selection_patch_surfaces()
        return available_synthetic_providers()

    @classmethod
    def generate_batch_for_spec(cls, spec: CorpusSpec) -> SyntheticGenerationBatch:
        corpus = cls.from_spec(spec)
        return corpus.generate_batch(
            count=spec.count,
            messages_per_conversation=spec.messages_per_conversation,
            seed=spec.seed,
            style=spec.style,
        )

    @classmethod
    def generate_for_spec(cls, spec: CorpusSpec) -> list[bytes]:
        return cls.generate_batch_for_spec(spec).raw_items

    @classmethod
    def write_spec_artifacts(
        cls,
        spec: CorpusSpec,
        output_dir: Path,
        *,
        prefix: str,
        index_width: int = 2,
    ) -> SyntheticWrittenBatch:
        corpus = cls.from_spec(spec)
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = ".json" if corpus.wire_format.encoding == "json" else ".jsonl"
        batch = cls.generate_batch_for_spec(spec)
        written_files: list[Path] = []
        for idx, artifact in enumerate(batch.artifacts):
            file_path = output_dir / f"{prefix}-{idx:0{index_width}d}{ext}"
            file_path.write_bytes(artifact.raw_bytes)
            written_files.append(file_path)
        return SyntheticWrittenBatch(batch=batch, files=tuple(written_files))

    @classmethod
    def write_specs_artifacts(
        cls,
        corpus_specs: tuple[CorpusSpec, ...],
        output_root: Path,
        *,
        prefix: str,
        index_width: int = 2,
    ) -> tuple[SyntheticWrittenBatch, ...]:
        provider_counts = Counter(spec.provider for spec in corpus_specs)
        return tuple(
            cls.write_spec_artifacts(
                spec,
                output_root / spec.provider,
                prefix=prefix if provider_counts[spec.provider] == 1 else f"{prefix}-{spec.scope_label}",
                index_width=index_width,
            )
            for spec in corpus_specs
        )

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

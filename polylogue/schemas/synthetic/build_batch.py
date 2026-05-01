"""Batch orchestration helpers for synthetic corpora."""

from __future__ import annotations

import random
from typing import Protocol

from polylogue.core.json import JSONValue, is_json_value
from polylogue.schemas.synthetic.models import (
    SchemaRecord,
    SyntheticArtifact,
    SyntheticGenerationBatch,
    SyntheticGenerationReport,
    SyntheticStyle,
)
from polylogue.schemas.synthetic.relations import RelationConstraintSolver
from polylogue.schemas.synthetic.semantic_values import SemanticValueGenerator
from polylogue.schemas.synthetic.showcase import _SHOWCASE_THEMES, ConversationTheme
from polylogue.schemas.synthetic.wire_formats import WireFormat


class _SyntheticBatchContext(Protocol):
    schema: SchemaRecord
    wire_format: WireFormat
    provider: str
    package_version: str
    element_kind: str | None
    _relation_solver: RelationConstraintSolver
    _semantic_gen: SemanticValueGenerator | None

    def _generate_jsonl_records(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: ConversationTheme | None = None,
    ) -> list[dict[str, JSONValue]]: ...

    def _generate_tree_json(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: ConversationTheme | None = None,
    ) -> dict[str, JSONValue]: ...

    def _generate_linear_json(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: ConversationTheme | None = None,
    ) -> dict[str, JSONValue]: ...

    def _generate_from_schema(self, schema: SchemaRecord, rng: random.Random) -> JSONValue: ...

    def _serialize(self, data: JSONValue) -> bytes: ...


def _resolve_style(style: str) -> SyntheticStyle:
    if style == "default":
        return "default"
    if style == "showcase":
        return "showcase"
    raise ValueError(f"Unknown synthetic style: {style}")


def _as_json_value(value: object) -> JSONValue:
    if is_json_value(value):
        return value
    raise TypeError(f"Generated payload is not JSON-compatible: {type(value).__name__}")


def generate_batch(
    self: _SyntheticBatchContext,
    count: int = 5,
    messages_per_conversation: range = range(3, 15),
    seed: int | None = None,
    style: str = "default",
) -> SyntheticGenerationBatch:
    resolved_style = _resolve_style(style)

    rng = random.Random(seed)
    artifacts: list[SyntheticArtifact] = []
    for _ in range(count):
        self._relation_solver = RelationConstraintSolver(self.schema)
        self._semantic_gen = None
        n_messages = rng.choice(messages_per_conversation)
        theme = rng.choice(_SHOWCASE_THEMES) if resolved_style == "showcase" else None
        data = _generate_conversation(self, n_messages, rng, theme=theme)
        artifacts.append(
            SyntheticArtifact(
                raw_bytes=self._serialize(data),
                message_count=n_messages,
                style=resolved_style,
            )
        )

    report = SyntheticGenerationReport(
        provider=self.provider,
        package_version=self.package_version,
        element_kind=self.element_kind,
        wire_encoding=self.wire_format.encoding,
        requested_count=count,
        generated_count=len(artifacts),
        style=resolved_style,
        seed=seed,
    )
    return SyntheticGenerationBatch(artifacts=artifacts, report=report)


def _generate_conversation(
    self: _SyntheticBatchContext,
    n_messages: int,
    rng: random.Random,
    *,
    theme: ConversationTheme | None = None,
) -> JSONValue:
    wf = self.wire_format
    if wf.encoding == "jsonl":
        return _as_json_value(self._generate_jsonl_records(n_messages, rng, theme=theme))
    if wf.tree and wf.tree.container_path:
        return _as_json_value(self._generate_tree_json(n_messages, rng, theme=theme))
    if wf.messages_path:
        return _as_json_value(self._generate_linear_json(n_messages, rng, theme=theme))
    return _as_json_value(self._generate_from_schema(self.schema, rng))


def _role_cycle(self: _SyntheticBatchContext) -> list[str]:
    match self.provider:
        case "chatgpt":
            return ["user", "assistant"]
        case "claude-ai":
            return ["human", "assistant"]
        case "claude-code":
            return ["user", "assistant"]
        case "codex":
            return ["user", "assistant"]
        case "gemini":
            return ["user", "model"]
        case _:
            return ["user", "assistant"]


__all__ = [
    "_generate_conversation",
    "_role_cycle",
    "generate_batch",
]

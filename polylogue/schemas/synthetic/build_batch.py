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
from polylogue.schemas.synthetic.showcase import _SHOWCASE_THEMES, SessionTheme
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
        theme: SessionTheme | None = None,
    ) -> list[dict[str, JSONValue]]: ...

    def _generate_tree_json(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: SessionTheme | None = None,
    ) -> dict[str, JSONValue]: ...

    def _generate_linear_json(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: SessionTheme | None = None,
    ) -> dict[str, JSONValue]: ...

    def _generate_from_schema(self, schema: SchemaRecord, rng: random.Random) -> JSONValue: ...

    def _role_cycle(self) -> list[str]: ...

    def _serialize(self, data: JSONValue) -> bytes: ...


def _resolve_style(style: str) -> SyntheticStyle:
    if style == "default":
        return "default"
    if style == "showcase":
        return "showcase"
    if style == "tool-heavy":
        return "tool-heavy"
    raise ValueError(f"Unknown synthetic style: {style}")


def _as_json_value(value: object) -> JSONValue:
    if is_json_value(value):
        return value
    raise TypeError(f"Generated payload is not JSON-compatible: {type(value).__name__}")


def generate_batch(
    self: _SyntheticBatchContext,
    count: int = 5,
    messages_per_session: range = range(3, 15),
    seed: int | None = None,
    style: str = "default",
) -> SyntheticGenerationBatch:
    resolved_style = _resolve_style(style)

    rng = random.Random(seed)
    artifacts: list[SyntheticArtifact] = []
    for _ in range(count):
        self._relation_solver = RelationConstraintSolver(self.schema)
        self._semantic_gen = None
        n_messages = rng.choice(messages_per_session)
        theme = rng.choice(_SHOWCASE_THEMES) if resolved_style == "showcase" else None
        data = _generate_session(self, n_messages, rng, theme=theme)
        if resolved_style == "tool-heavy":
            data = _add_tool_heavy_blocks(self.provider, data)
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


def _tool_use_blocks(provider: str, index: int) -> list[JSONValue]:
    read_id = f"{provider}-read-{index:04d}"
    bash_id = f"{provider}-bash-{index:04d}"
    return [
        {"type": "text", "text": f"Inspecting generated workload record {index}."},
        {
            "type": "tool_use",
            "id": read_id,
            "name": "Read",
            "input": {"file_path": f"fixtures/generated-{index:04d}.jsonl"},
        },
        {
            "type": "tool_use",
            "id": bash_id,
            "name": "Bash",
            "input": {"command": f"python -m pytest tests/generated/test_{index:04d}.py -q"},
        },
        {"type": "tool_result", "tool_use_id": read_id, "content": "read 4096 bytes"},
        {"type": "tool_result", "tool_use_id": bash_id, "content": "1 passed"},
    ]


def _add_tool_heavy_blocks(provider: str, data: JSONValue) -> JSONValue:
    if provider not in {"claude-code", "codex"} or not isinstance(data, list):
        return data
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            continue
        role = _record_role(provider, record)
        if role not in {"assistant", "model"}:
            continue
        blocks = _tool_use_blocks(provider, index)
        if provider == "claude-code":
            message = record.get("message")
            if not isinstance(message, dict):
                message = {}
                record["message"] = message
            message["content"] = blocks
            message["role"] = "assistant"
            record["type"] = "assistant"
            continue
        record["content"] = blocks
        record["role"] = "assistant"
        record["type"] = "message"
    return data


def _record_role(provider: str, record: dict[str, JSONValue]) -> str | None:
    if provider == "claude-code":
        message = record.get("message")
        if isinstance(message, dict):
            role = message.get("role")
            if isinstance(role, str):
                return role
        role = record.get("type")
        return role if isinstance(role, str) else None
    role = record.get("role")
    return role if isinstance(role, str) else None


def _generate_session(
    self: _SyntheticBatchContext,
    n_messages: int,
    rng: random.Random,
    *,
    theme: SessionTheme | None = None,
) -> JSONValue:
    wf = self.wire_format
    if wf.encoding == "jsonl":
        return _as_json_value(self._generate_jsonl_records(n_messages, rng, theme=theme))
    if wf.tree and wf.tree.container_path:
        return _as_json_value(self._generate_tree_json(n_messages, rng, theme=theme))
    if wf.messages_path:
        return _as_json_value(self._generate_linear_json(n_messages, rng, theme=theme))
    roles = self._role_cycle()
    self._semantic_gen = SemanticValueGenerator(
        rng,
        theme=theme,
        base_ts=rng.uniform(1670000000, 1760000000),
        role_cycle=roles,
    )
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
    "_generate_session",
    "_role_cycle",
    "generate_batch",
]

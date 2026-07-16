"""Batch orchestration helpers for synthetic corpora."""

from __future__ import annotations

import base64
import random
from typing import Protocol

from polylogue.core.json import JSONValue, is_json_value
from polylogue.schemas.synthetic.demo_themes import _DEMO_THEMES, SessionTheme
from polylogue.schemas.synthetic.models import (
    SchemaRecord,
    SyntheticArtifact,
    SyntheticGenerationBatch,
    SyntheticGenerationReport,
    SyntheticStyle,
)
from polylogue.schemas.synthetic.relations import RelationConstraintSolver
from polylogue.schemas.synthetic.semantic_values import SemanticValueGenerator
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
    if style == "demo":
        return "demo"
    if style == "tool-heavy":
        return "tool-heavy"
    if style == "demo-tool-heavy":
        return "demo-tool-heavy"
    if style == "demo-attachments":
        return "demo-attachments"
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
    session_native_ids: tuple[str, ...] = (),
) -> SyntheticGenerationBatch:
    resolved_style = _resolve_style(style)
    if session_native_ids and len(session_native_ids) != count:
        raise ValueError("session_native_ids must contain exactly one id per generated session")

    rng = random.Random(seed)
    artifacts: list[SyntheticArtifact] = []
    for index in range(count):
        self._relation_solver = RelationConstraintSolver(self.schema)
        self._semantic_gen = None
        n_messages = rng.choice(messages_per_session)
        theme = rng.choice(_DEMO_THEMES) if resolved_style == "demo" else None
        data = _generate_session(self, n_messages, rng, theme=theme)
        if resolved_style in {"tool-heavy", "demo-tool-heavy"}:
            data = _add_tool_heavy_blocks(
                self.provider,
                data,
                include_failure_followups=resolved_style == "demo-tool-heavy",
            )
        if resolved_style == "demo-attachments":
            data = _add_demo_attachment_blocks(self.provider, data)
        if session_native_ids:
            data = _pin_session_native_id(self.provider, data, session_native_ids[index])
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


def _pin_session_native_id(provider: str, data: JSONValue, native_id: str) -> JSONValue:
    """Apply an authored fixture identity after stochastic schema generation.

    Scenario identity is a workload contract, not an emergent consequence of
    random-call order. Provider-specific wire shaping keeps the override on the
    same production field each parser treats as authoritative.
    """
    if not native_id:
        raise ValueError("session native ids must not be empty")
    if provider == "chatgpt" and isinstance(data, dict):
        data["id"] = native_id
        if "conversation_id" in data:
            data["conversation_id"] = native_id
        return data
    if provider == "claude-ai" and isinstance(data, dict):
        data["uuid"] = native_id
        return data
    if provider == "gemini" and isinstance(data, dict):
        data["id"] = native_id
        return data
    if provider == "claude-code" and isinstance(data, list):
        for record in data:
            if isinstance(record, dict):
                record["sessionId"] = native_id
        return data
    if provider == "codex" and isinstance(data, list):
        data.insert(0, {"type": "session_meta", "payload": {"id": native_id}})
        return data
    raise ValueError(f"Provider {provider!r} does not support pinned synthetic session identities")


def _tool_use_blocks(provider: str, index: int, *, include_failures: bool = False) -> list[JSONValue]:
    read_id = f"{provider}-read-{index:04d}"
    bash_id = f"{provider}-bash-{index:04d}"
    bash_failed = include_failures and index in {1, 3}
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
        {
            "type": "tool_result",
            "tool_use_id": bash_id,
            "content": "AssertionError: generated fixture mismatch",
            "is_error": True,
        }
        if bash_failed
        else {"type": "tool_result", "tool_use_id": bash_id, "content": "1 passed"},
    ]


def _add_tool_heavy_blocks(
    provider: str,
    data: JSONValue,
    *,
    include_failure_followups: bool = False,
) -> JSONValue:
    if provider not in {"claude-code", "codex"} or not isinstance(data, list):
        return data
    followups: list[tuple[int, dict[str, JSONValue]]] = []
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            continue
        role = _record_role(provider, record)
        if role not in {"assistant", "model"}:
            continue
        blocks = _tool_use_blocks(provider, index, include_failures=include_failure_followups)
        if provider == "claude-code":
            message = record.get("message")
            if not isinstance(message, dict):
                message = {}
                record["message"] = message
            message["content"] = blocks
            message["role"] = "assistant"
            record["type"] = "assistant"
            if include_failure_followups and index in {1, 3}:
                followups.append((index + 1, _demo_failure_followup(provider, record, index)))
            continue
        record["content"] = blocks
        record["role"] = "assistant"
        record["type"] = "message"
        if include_failure_followups and index in {1, 3}:
            followups.append((index + 1, _demo_failure_followup(provider, record, index)))
    for insert_at, followup in reversed(followups):
        data.insert(insert_at, followup)
    return data


def _add_demo_attachment_blocks(provider: str, data: JSONValue) -> JSONValue:
    """Add one inline attachment family to demo Gemini chunks."""

    if provider != "gemini" or not isinstance(data, dict):
        return data
    chunked_prompt = data.get("chunkedPrompt")
    if not isinstance(chunked_prompt, dict):
        return data
    chunks = chunked_prompt.get("chunks")
    if not isinstance(chunks, list):
        return data
    for chunk in chunks:
        if not isinstance(chunk, dict) or chunk.get("role") != "user":
            continue
        payload = b"demo attachment bytes for Polylogue fixture coverage\n"
        chunk["inlineFile"] = {
            "mimeType": "text/plain",
            "data": base64.b64encode(payload).decode("ascii"),
        }
        chunk["text"] = "Please inspect the attached fixture note."
        return data
    return data


def _demo_failure_followup(provider: str, record: dict[str, JSONValue], index: int) -> dict[str, JSONValue]:
    text = (
        "The pytest command failed with exit code 1; I will inspect the assertion before retrying."
        if index == 1
        else "I will inspect the generated fixture and adjust the next command."
    )
    if provider == "claude-code":
        followup = dict(record)
        previous_uuid = record.get("uuid")
        followup["uuid"] = f"demo-{provider}-failure-followup-{index:04d}"
        if isinstance(previous_uuid, str):
            followup["parentUuid"] = previous_uuid
        followup["type"] = "assistant"
        followup["message"] = {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        }
        return followup
    followup = dict(record)
    followup["id"] = f"demo-{provider}-failure-followup-{index:04d}"
    followup["type"] = "message"
    followup["role"] = "assistant"
    followup["content"] = [{"type": "text", "text": text}]
    return followup


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

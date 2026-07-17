"""Batch orchestration helpers for synthetic corpora."""

from __future__ import annotations

import base64
import copy
import hashlib
import random
from collections import Counter
from collections.abc import Mapping
from typing import Protocol

from polylogue.core.json import JSONValue, is_json_value
from polylogue.schemas.synthetic.demo_themes import _DEMO_THEMES, SessionTheme
from polylogue.schemas.synthetic.models import (
    SchemaRecord,
    SyntheticArtifact,
    SyntheticArtifactFacts,
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
    workload_profile: SchemaRecord | None
    _active_profile_tokens: tuple[str, ...]
    _active_record_bucket: tuple[str, str] | None
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

    def _select_structural_variant(self, rng: random.Random) -> str | None: ...


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


def _expected_session_id(provider: str, native_session_id: str | None) -> str | None:
    if native_session_id is None:
        return None
    prefixes = {
        "chatgpt": "chatgpt-export",
        "claude-ai": "claude-ai-export",
        "claude-code": "claude-code-session",
        "codex": "codex-session",
        "gemini": "aistudio-drive",
        "gemini-cli": "gemini-cli-session",
        "hermes": "hermes-session",
        "antigravity": "antigravity-session",
    }
    prefix = prefixes.get(provider)
    return f"{prefix}:{native_session_id}" if prefix is not None else None


def _planted_artifact_facts(
    provider: str,
    data: JSONValue,
    *,
    raw_bytes: bytes,
    native_session_id: str | None,
    message_count: int,
) -> SyntheticArtifactFacts:
    """Record wire-level facts without consulting production normalization."""
    tool_uses: list[str] = []
    tool_results: list[str] = []

    def visit(value: JSONValue) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, Mapping):
            return
        block_type = value.get("type")
        tool_use_id = value.get("id")
        tool_result_id = value.get("tool_use_id")
        if block_type == "tool_use" and isinstance(tool_use_id, str):
            tool_uses.append(tool_use_id)
        if block_type == "tool_result" and isinstance(tool_result_id, str):
            tool_results.append(tool_result_id)
        for child in value.values():
            if is_json_value(child):
                visit(child)

    visit(data)
    return SyntheticArtifactFacts(
        provider=provider,
        native_session_id=native_session_id,
        expected_session_id=_expected_session_id(provider, native_session_id),
        message_count=message_count,
        tool_use_ids=tuple(tool_uses),
        tool_result_ids=tuple(tool_results),
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def _profile_mapping(profile: SchemaRecord | None, *path: str) -> dict[str, JSONValue]:
    value: object = profile
    for name in path:
        if not isinstance(value, dict):
            return {}
        value = value.get(name)
    return value if isinstance(value, dict) else {}


def _positive_profile_count(profile: dict[str, JSONValue], name: str) -> int:
    value = profile.get(name)
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _tool_relationship_variants(profile: SchemaRecord | None) -> tuple[tuple[str, int], ...]:
    relationships = _profile_mapping(profile, "relationships", "tool_results")
    names = (
        "paired",
        "missing",
        "orphan",
        "duplicate_calls",
        "duplicate_results",
        "out_of_order",
        "call_without_id",
        "result_without_id",
    )
    return tuple((name, count) for name in names if (count := _positive_profile_count(relationships, name)))


def _relationship_variant_sequence(
    variants: tuple[tuple[str, int], ...],
    *,
    count: int,
    rng: random.Random,
) -> list[str]:
    """Choose weighted states while retaining every positive class when possible."""
    if count <= 0 or not variants:
        return []
    selected = [name for name, _weight in variants] if count >= len(variants) else []
    selected.extend(
        rng.choices(
            [item[0] for item in variants],
            weights=[item[1] for item in variants],
            k=count - len(selected),
        )
    )
    rng.shuffle(selected)
    return selected


def _profile_result_distance(profile: SchemaRecord | None) -> int:
    relationships = _profile_mapping(profile, "relationships", "tool_results")
    distance = relationships.get("result_record_distance")
    if not isinstance(distance, dict):
        return 1
    quantiles = distance.get("quantiles")
    candidate: object = quantiles.get("p99") if isinstance(quantiles, dict) else None
    if not isinstance(candidate, (int, float)) or isinstance(candidate, bool):
        candidate = distance.get("max")
    if not isinstance(candidate, (int, float)) or isinstance(candidate, bool):
        return 1
    return max(1, int(round(candidate)))


def _tool_result_payload(call_id: str | None, *, failed: bool) -> dict[str, JSONValue]:
    payload: dict[str, JSONValue] = {
        "type": "custom_tool_call_output",
        "output": {
            "output": "synthetic command failed" if failed else "synthetic command succeeded",
            "metadata": {"exit_code": 1 if failed else 0},
        },
    }
    if call_id is not None:
        payload["call_id"] = call_id
    return payload


def _codex_tool_records(
    variant: str,
    *,
    call_id: str,
    functions_exec: bool,
    failed: bool,
    result_distance: int,
) -> list[dict[str, JSONValue]]:
    call_payload: dict[str, JSONValue] = {
        "type": "custom_tool_call",
        "name": "exec" if functions_exec else "synthetic_tool",
        "input": "printf synthetic",
    }
    if variant != "call_without_id":
        call_payload["call_id"] = call_id
    call: dict[str, JSONValue] = {"type": "response_item", "payload": call_payload}
    result_id = None if variant == "result_without_id" else call_id
    result: dict[str, JSONValue] = {
        "type": "response_item",
        "payload": _tool_result_payload(result_id, failed=failed),
    }
    fillers: list[dict[str, JSONValue]] = [
        {
            "type": "event_msg",
            "payload": {"type": "agent_message", "message": f"synthetic intervening event {index}"},
        }
        for index in range(max(0, result_distance - 1))
    ]

    if variant in {"missing", "call_without_id"}:
        return [call]
    if variant in {"orphan", "result_without_id"}:
        return [result]
    if variant == "duplicate_calls":
        return [call, copy.deepcopy(call), *fillers, result]
    if variant == "duplicate_results":
        return [call, *fillers, result, copy.deepcopy(result)]
    if variant == "out_of_order":
        return [result, call]
    return [call, *fillers, result]


def _claude_code_tool_records(
    variant: str,
    *,
    call_id: str,
    functions_exec: bool,
    failed: bool,
    session_id: str,
    rng: random.Random,
    result_distance: int,
) -> list[dict[str, JSONValue]]:
    call_block: dict[str, JSONValue] = {
        "type": "tool_use",
        "name": "Bash" if functions_exec else "SyntheticTool",
        "input": {"command": "printf synthetic"},
    }
    if variant != "call_without_id":
        call_block["id"] = call_id
    result_block: dict[str, JSONValue] = {
        "type": "tool_result",
        "content": "synthetic command failed" if failed else "synthetic command succeeded",
        "is_error": failed,
    }
    if variant != "result_without_id":
        result_block["tool_use_id"] = call_id

    def record(role: str, block: dict[str, JSONValue]) -> dict[str, JSONValue]:
        return {
            "type": role,
            "sessionId": session_id,
            "uuid": str(rng.getrandbits(128)),
            "parentUuid": None,
            "message": {"role": role, "content": [block]},
        }

    call = record("assistant", call_block)
    result = record("user", result_block)
    fillers = [
        record(
            "assistant",
            {"type": "text", "text": f"synthetic intervening message {index}"},
        )
        for index in range(max(0, result_distance - 1))
    ]
    if variant in {"missing", "call_without_id"}:
        return [call]
    if variant in {"orphan", "result_without_id"}:
        return [result]
    if variant == "duplicate_calls":
        return [call, copy.deepcopy(call), *fillers, result]
    if variant == "duplicate_results":
        return [call, *fillers, result, copy.deepcopy(result)]
    if variant == "out_of_order":
        return [result, call]
    return [call, *fillers, result]


def _session_native_id(provider: str, data: JSONValue) -> str | None:
    if not isinstance(data, list):
        return None
    if provider == "codex":
        for record in data:
            if not isinstance(record, dict) or record.get("type") != "session_meta":
                continue
            payload = record.get("payload")
            if isinstance(payload, dict):
                native_id = payload.get("id")
                if isinstance(native_id, str):
                    return native_id
    if provider == "claude-code":
        for record in data:
            if isinstance(record, dict):
                native_id = record.get("sessionId")
                if isinstance(native_id, str):
                    return native_id
    return None


def _apply_tool_relationship(
    provider: str,
    data: JSONValue,
    *,
    variant: str,
    profile: SchemaRecord | None,
    rng: random.Random,
) -> tuple[bool, bool]:
    if not isinstance(data, list):
        return False, False
    relationships = _profile_mapping(profile, "relationships", "tool_results")
    call_count = max(_positive_profile_count(relationships, "calls"), 1)
    functions_exec = rng.randrange(call_count) < _positive_profile_count(relationships, "functions_exec_calls")
    result_count = max(_positive_profile_count(relationships, "results"), 1)
    failed = rng.randrange(result_count) < _positive_profile_count(relationships, "error_results")
    result_distance = _profile_result_distance(profile)
    call_id = f"synthetic-call-{rng.getrandbits(64):016x}"
    if provider == "codex":
        data.extend(
            _codex_tool_records(
                variant,
                call_id=call_id,
                functions_exec=functions_exec,
                failed=failed,
                result_distance=result_distance,
            )
        )
        return functions_exec, failed
    if provider == "claude-code":
        session_id = _session_native_id(provider, data)
        if session_id is None:
            return False, False
        data.extend(
            _claude_code_tool_records(
                variant,
                call_id=call_id,
                functions_exec=functions_exec,
                failed=failed,
                session_id=session_id,
                rng=rng,
                result_distance=result_distance,
            )
        )
        return functions_exec, failed
    return False, False


def _codex_message_records(data: JSONValue) -> list[dict[str, JSONValue]]:
    if not isinstance(data, list):
        return []
    records: list[dict[str, JSONValue]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "response_item":
            payload = item.get("payload")
            if isinstance(payload, dict) and payload.get("type") == "message":
                records.append(copy.deepcopy(item))
        elif item.get("type") == "message":
            records.append(copy.deepcopy(item))
    return records


def _apply_profile_lineage(provider: str, data_items: list[JSONValue], profile: SchemaRecord | None) -> int:
    if provider != "codex" or len(data_items) < 2:
        return 0
    lineage = _profile_mapping(profile, "relationships", "lineage")
    if _positive_profile_count(lineage, "parent_references") == 0:
        return 0
    edge_count = 0
    for index in range(1, len(data_items)):
        parent = data_items[index - 1]
        child = data_items[index]
        parent_id = _session_native_id(provider, parent)
        if parent_id is None or not isinstance(child, list):
            continue
        prefix = _codex_message_records(parent)
        if not prefix:
            continue
        meta_index = next(
            (
                record_index
                for record_index, record in enumerate(child)
                if isinstance(record, dict) and record.get("type") == "session_meta"
            ),
            None,
        )
        if meta_index is None:
            continue
        meta = child[meta_index]
        if not isinstance(meta, dict):
            continue
        payload = meta.get("payload")
        if not isinstance(payload, dict):
            payload = {}
            meta["payload"] = payload
        payload["forked_from_id"] = parent_id
        child[meta_index + 1 : meta_index + 1] = prefix
        edge_count += 1
    return edge_count


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
    profile_id = (
        str(self.workload_profile.get("profile_id"))
        if isinstance(self.workload_profile, dict) and self.workload_profile.get("profile_id") is not None
        else "no-profile"
    )
    profile_rng = random.Random(f"{seed!r}\x1f{self.provider}\x1f{self.package_version}\x1f{profile_id}")
    generated_data: list[JSONValue] = []
    generated_message_counts: list[int] = []
    selected_variants: Counter[str] = Counter()
    for index in range(count):
        self._relation_solver = RelationConstraintSolver(self.schema)
        self._semantic_gen = None
        selected_variant = self._select_structural_variant(profile_rng)
        if selected_variant is not None:
            selected_variants[selected_variant] += 1
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
        generated_data.append(data)
        generated_message_counts.append(n_messages)

    relationship_variants = _tool_relationship_variants(self.workload_profile)
    selected_relationships: Counter[str] = Counter()
    if relationship_variants:
        variants = _relationship_variant_sequence(
            relationship_variants,
            count=len(generated_data),
            rng=profile_rng,
        )
        result_distance = _profile_result_distance(self.workload_profile)
        for data, variant in zip(generated_data, variants, strict=True):
            functions_exec, failed = _apply_tool_relationship(
                self.provider,
                data,
                variant=variant,
                profile=self.workload_profile,
                rng=profile_rng,
            )
            selected_relationships[variant] += 1
            if functions_exec:
                selected_relationships["functions_exec"] += 1
            if failed:
                selected_relationships["error_result"] += 1
            if result_distance > 1 and variant not in {
                "missing",
                "orphan",
                "out_of_order",
                "call_without_id",
                "result_without_id",
            }:
                selected_relationships["late_result"] += 1

    lineage_edge_count = _apply_profile_lineage(self.provider, generated_data, self.workload_profile)
    artifacts: list[SyntheticArtifact] = []
    for index, (data, message_count) in enumerate(zip(generated_data, generated_message_counts, strict=True)):
        raw_bytes = self._serialize(data)
        artifacts.append(
            SyntheticArtifact(
                raw_bytes=raw_bytes,
                message_count=message_count,
                style=resolved_style,
                facts=_planted_artifact_facts(
                    self.provider,
                    data,
                    raw_bytes=raw_bytes,
                    native_session_id=session_native_ids[index] if session_native_ids else None,
                    message_count=message_count,
                ),
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
        workload_profile_id=None if profile_id == "no-profile" else profile_id,
        structural_variant_counts=dict(sorted(selected_variants.items())),
        relationship_variant_counts=dict(sorted(selected_relationships.items())),
        lineage_edge_count=lineage_edge_count,
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
        for record in data:
            if not isinstance(record, dict) or record.get("type") != "session_meta":
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict):
                payload = {}
                record["payload"] = payload
            payload["id"] = native_id
            return data
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

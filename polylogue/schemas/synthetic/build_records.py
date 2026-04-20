"""Record-shaping helpers for synthetic corpora."""

from __future__ import annotations

import random
import uuid
from collections.abc import Sequence
from typing import Protocol, TypeAlias

from polylogue.lib.raw_payload_decode import JSONValue
from polylogue.schemas.synthetic.models import SchemaRecord, SchemaValue
from polylogue.schemas.synthetic.semantic_values import SemanticValueGenerator
from polylogue.schemas.synthetic.showcase import ConversationTheme
from polylogue.schemas.synthetic.wire_formats import WireFormat

SyntheticRecord: TypeAlias = dict[str, JSONValue]


class _WireFormatContext(Protocol):
    provider: str
    schema: SchemaRecord
    wire_format: WireFormat
    _semantic_gen: SemanticValueGenerator | None

    def _generate_from_schema(
        self,
        schema: SchemaRecord,
        rng: random.Random,
        *,
        skip_keys: set[str] | None = ...,
        depth: int = ...,
        max_depth: int = ...,
        path: str = ...,
    ) -> JSONValue: ...

    def _ensure_wire_format(
        self,
        data: SyntheticRecord,
        role: str,
        rng: random.Random,
        index: int,
        base_ts: float,
        theme: ConversationTheme | None = ...,
    ) -> None: ...

    def _role_cycle(self) -> list[str]: ...


def _coerce_record(value: JSONValue) -> SyntheticRecord:
    return value if isinstance(value, dict) else {}


def _coerce_schema(value: SchemaValue | object) -> SchemaRecord:
    return value if isinstance(value, dict) else {}


def _child_id_list(record: SyntheticRecord, field_name: str) -> list[JSONValue]:
    existing = record.get(field_name)
    if isinstance(existing, list) and all(isinstance(item, str) for item in existing):
        return existing
    children: list[JSONValue] = []
    record[field_name] = children
    return children


def _message_payloads(records: list[SyntheticRecord]) -> list[JSONValue]:
    return list(records)


def _reset_semantic_generator(
    self: _WireFormatContext,
    *,
    rng: random.Random,
    theme: ConversationTheme | None,
    base_ts: float,
    roles: Sequence[str],
) -> None:
    self._semantic_gen = SemanticValueGenerator(rng, theme=theme, base_ts=base_ts, role_cycle=list(roles))


def _has_messages_path(parts: Sequence[str], schema: SchemaRecord) -> tuple[bool, SchemaRecord]:
    cursor = schema
    for part in parts:
        props = _coerce_schema(cursor.get("properties"))
        next_node = _coerce_schema(props.get(part))
        if not next_node:
            return False, {}
        cursor = next_node
    return True, cursor


def _advance_semantic_turn(self: _WireFormatContext) -> None:
    semantic_generator = self._semantic_gen
    if semantic_generator is not None:
        semantic_generator.advance_turn()


def _generate_tree_json(
    self: _WireFormatContext, n_messages: int, rng: random.Random, *, theme: ConversationTheme | None = None
) -> SyntheticRecord:
    tree_cfg = self.wire_format.tree
    assert tree_cfg is not None and tree_cfg.container_path is not None

    roles = self._role_cycle()
    base_ts = rng.uniform(1670000000, 1760000000)
    _reset_semantic_generator(self, rng=rng, theme=theme, base_ts=base_ts, roles=roles)

    top = self._generate_from_schema(self.schema, rng, skip_keys={tree_cfg.container_path})
    if not isinstance(top, dict):
        top = {}
    top_record = _coerce_record(top)
    properties = _coerce_schema(self.schema.get("properties"))

    container_schema = _coerce_schema(properties.get(tree_cfg.container_path))
    node_schema = _coerce_schema(container_schema.get("additionalProperties"))
    nodes: list[SyntheticRecord] = []

    for i in range(n_messages):
        node = _coerce_record(self._generate_from_schema(node_schema, rng))

        node_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))
        node[tree_cfg.key_field] = node_id

        if i == 0:
            node[tree_cfg.parent_field] = None
            if tree_cfg.children_field:
                node[tree_cfg.children_field] = []
        else:
            parent_idx = rng.randint(max(0, i - 3), i - 1)
            parent = nodes[parent_idx]
            node[tree_cfg.parent_field] = parent[tree_cfg.key_field]
            if tree_cfg.children_field:
                _child_id_list(parent, tree_cfg.children_field).append(node_id)
                node[tree_cfg.children_field] = []

        role = roles[i % len(roles)]
        self._ensure_wire_format(node, role, rng, i, base_ts=base_ts, theme=theme)
        _advance_semantic_turn(self)
        nodes.append(node)

    children_by_id: SyntheticRecord = {}
    for node in nodes:
        node_id_value = node.get(tree_cfg.key_field)
        if isinstance(node_id_value, str):
            children_by_id[node_id_value] = node
    top_record[tree_cfg.container_path] = children_by_id
    if "current_node" in properties:
        top_record["current_node"] = nodes[-1][tree_cfg.key_field]
    if self.provider == "chatgpt":
        top_record["id"] = str(uuid.UUID(int=rng.getrandbits(128), version=4))
        top_record.setdefault("create_time", base_ts)
        top_record.setdefault("update_time", base_ts + max(0, n_messages - 1) * 60)
    if theme is not None and "title" in _coerce_schema(self.schema.get("properties")):
        top_record["title"] = theme.title
    return top_record


def _generate_linear_json(
    self: _WireFormatContext, n_messages: int, rng: random.Random, *, theme: ConversationTheme | None = None
) -> SyntheticRecord:
    msgs_path = self.wire_format.messages_path
    assert msgs_path is not None
    msgs_parts = msgs_path.split(".")

    roles = self._role_cycle()
    base_ts = rng.uniform(1670000000, 1760000000)
    _reset_semantic_generator(self, rng=rng, theme=theme, base_ts=base_ts, roles=roles)

    schema_has_path, msgs_schema = _has_messages_path(msgs_parts, self.schema)

    if schema_has_path:
        item_schema = _coerce_schema(msgs_schema.get("items"))
        top = self._generate_from_schema(self.schema, rng, skip_keys={msgs_parts[0]})
    else:
        item_schema = self.schema
        top = {}

    if not isinstance(top, dict):
        top = {}
    top_record = _coerce_record(top)

    messages: list[SyntheticRecord] = []
    for i in range(n_messages):
        msg = _coerce_record(self._generate_from_schema(item_schema, rng))
        role = roles[i % len(roles)]
        self._ensure_wire_format(msg, role, rng, i, base_ts=base_ts, theme=theme)
        _advance_semantic_turn(self)
        messages.append(msg)

    target: SyntheticRecord = top_record
    for index, part in enumerate(msgs_parts):
        if index == len(msgs_parts) - 1:
            target[part] = _message_payloads(messages)
        else:
            nested = _coerce_record(target.get(part))
            target[part] = nested
            target = nested

    if theme is not None and self.provider == "claude-ai" and "name" in _coerce_schema(self.schema.get("properties")):
        top_record["name"] = theme.title
    return top_record


def _generate_jsonl_records(
    self: _WireFormatContext, n_messages: int, rng: random.Random, *, theme: ConversationTheme | None = None
) -> list[SyntheticRecord]:
    tree_cfg = self.wire_format.tree
    records: list[SyntheticRecord] = []
    roles = self._role_cycle()
    session_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))
    base_ts = rng.uniform(1670000000, 1760000000)
    _reset_semantic_generator(self, rng=rng, theme=theme, base_ts=base_ts, roles=roles)

    for i in range(n_messages):
        record = _coerce_record(self._generate_from_schema(self.schema, rng))

        role = roles[i % len(roles)]
        self._ensure_wire_format(record, role, rng, i, base_ts=base_ts, theme=theme)
        _advance_semantic_turn(self)

        if tree_cfg:
            node_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))
            record[tree_cfg.key_field] = node_id
            if i == 0:
                record[tree_cfg.parent_field] = None
            else:
                record[tree_cfg.parent_field] = records[i - 1].get(tree_cfg.key_field)
            if tree_cfg.session_field:
                record[tree_cfg.session_field] = session_id

        if theme is not None:
            payload = record.get("payload")
            if isinstance(payload, dict) and "instructions" in payload:
                payload["instructions"] = theme.instructions
            if "instructions" in record:
                record["instructions"] = theme.instructions

        records.append(record)

    return records


__all__ = [
    "_generate_jsonl_records",
    "_generate_linear_json",
    "_generate_tree_json",
]

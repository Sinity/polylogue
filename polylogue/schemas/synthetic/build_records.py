"""Record-shaping helpers for synthetic corpora."""

from __future__ import annotations

import random
import uuid
from typing import Any

from polylogue.schemas.synthetic.semantic_values import SemanticValueGenerator
from polylogue.schemas.synthetic.showcase import ConversationTheme


def _generate_tree_json(
    self: Any, n_messages: int, rng: random.Random, *, theme: ConversationTheme | None = None
) -> dict[str, Any]:
    tree_cfg = self.wire_format.tree
    assert tree_cfg is not None and tree_cfg.container_path is not None

    roles = self._role_cycle()
    base_ts = rng.uniform(1670000000, 1760000000)
    self._semantic_gen = SemanticValueGenerator(rng, theme=theme, base_ts=base_ts, role_cycle=roles)

    top = self._generate_from_schema(self.schema, rng, skip_keys={tree_cfg.container_path})
    if not isinstance(top, dict):
        top = {}

    container_schema = self.schema.get("properties", {}).get(tree_cfg.container_path, {})
    node_schema = container_schema.get("additionalProperties", {})
    nodes: list[dict[str, Any]] = []

    for i in range(n_messages):
        node = self._generate_from_schema(node_schema, rng)
        if not isinstance(node, dict):
            node = {}

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
                parent.setdefault(tree_cfg.children_field, []).append(node_id)
                node[tree_cfg.children_field] = []

        role = roles[i % len(roles)]
        self._ensure_wire_format(node, role, rng, i, base_ts=base_ts, theme=theme)
        self._semantic_gen.advance_turn()
        nodes.append(node)

    top[tree_cfg.container_path] = {node[tree_cfg.key_field]: node for node in nodes}
    if "current_node" in self.schema.get("properties", {}):
        top["current_node"] = nodes[-1][tree_cfg.key_field]
    if self.provider == "chatgpt":
        top["id"] = str(uuid.UUID(int=rng.getrandbits(128), version=4))
        top.setdefault("create_time", base_ts)
        top.setdefault("update_time", base_ts + max(0, n_messages - 1) * 60)
    if theme is not None and "title" in self.schema.get("properties", {}):
        top["title"] = theme.title
    return top


def _generate_linear_json(
    self: Any, n_messages: int, rng: random.Random, *, theme: ConversationTheme | None = None
) -> dict[str, Any]:
    msgs_path = self.wire_format.messages_path
    assert msgs_path is not None

    parts = msgs_path.split(".")
    roles = self._role_cycle()
    base_ts = rng.uniform(1670000000, 1760000000)
    self._semantic_gen = SemanticValueGenerator(rng, theme=theme, base_ts=base_ts, role_cycle=roles)

    msgs_schema = self.schema
    schema_has_path = True
    for part in parts:
        if "properties" in msgs_schema and part in msgs_schema["properties"]:
            msgs_schema = msgs_schema["properties"][part]
        else:
            schema_has_path = False
            break

    if schema_has_path:
        item_schema = msgs_schema.get("items", {})
        top = self._generate_from_schema(self.schema, rng, skip_keys={parts[0]})
    else:
        item_schema = self.schema
        top = {}

    if not isinstance(top, dict):
        top = {}

    messages = []
    for i in range(n_messages):
        msg = self._generate_from_schema(item_schema, rng)
        if not isinstance(msg, dict):
            msg = {}
        role = roles[i % len(roles)]
        self._ensure_wire_format(msg, role, rng, i, base_ts=base_ts, theme=theme)
        self._semantic_gen.advance_turn()
        messages.append(msg)

    target = top
    for index, part in enumerate(parts):
        if index == len(parts) - 1:
            target[part] = messages
        else:
            target.setdefault(part, {})
            target = target[part]

    if theme is not None and self.provider == "claude-ai" and "name" in self.schema.get("properties", {}):
        top["name"] = theme.title
    return top


def _generate_jsonl_records(
    self: Any, n_messages: int, rng: random.Random, *, theme: ConversationTheme | None = None
) -> list[dict[str, Any]]:
    tree_cfg = self.wire_format.tree
    records: list[dict[str, Any]] = []
    roles = self._role_cycle()
    session_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))
    base_ts = rng.uniform(1670000000, 1760000000)
    self._semantic_gen = SemanticValueGenerator(rng, theme=theme, base_ts=base_ts, role_cycle=roles)

    for i in range(n_messages):
        record = self._generate_from_schema(self.schema, rng)
        if not isinstance(record, dict):
            record = {}

        role = roles[i % len(roles)]
        self._ensure_wire_format(record, role, rng, i, base_ts=base_ts, theme=theme)
        self._semantic_gen.advance_turn()

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

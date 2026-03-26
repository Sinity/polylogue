"""Conversation builders and wire-format shaping for synthetic corpora."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone

from polylogue.schemas.synthetic.models import (
    SyntheticArtifact,
    SyntheticGenerationBatch,
    SyntheticGenerationReport,
)
from polylogue.schemas.synthetic.relations import RelationConstraintSolver
from polylogue.schemas.synthetic.semantic_values import SemanticValueGenerator, _text_for_role
from polylogue.schemas.synthetic.showcase import _SHOWCASE_THEMES, ConversationTheme


def generate_batch(
    self,
    count: int = 5,
    messages_per_conversation: range = range(3, 15),
    seed: int | None = None,
    style: str = "default",
) -> SyntheticGenerationBatch:
    if style not in {"default", "showcase"}:
        raise ValueError(f"Unknown synthetic style: {style}")

    rng = random.Random(seed)
    artifacts: list[SyntheticArtifact] = []
    for _ in range(count):
        self._relation_solver = RelationConstraintSolver(self.schema)
        self._semantic_gen = None
        n_messages = rng.choice(messages_per_conversation)
        theme = rng.choice(_SHOWCASE_THEMES) if style == "showcase" else None
        data = self._generate_conversation(n_messages, rng, theme=theme)
        artifacts.append(
            SyntheticArtifact(
                raw_bytes=self._serialize(data),
                message_count=n_messages,
                style=style,
            )
        )

    report = SyntheticGenerationReport(
        provider=self.provider,
        package_version=self.package_version,
        element_kind=self.element_kind,
        wire_encoding=self.wire_format.encoding,
        requested_count=count,
        generated_count=len(artifacts),
        style=style,
        seed=seed,
    )
    return SyntheticGenerationBatch(artifacts=artifacts, report=report)


def _generate_conversation(
    self,
    n_messages: int,
    rng: random.Random,
    *,
    theme: ConversationTheme | None = None,
):
    wf = self.wire_format
    if wf.encoding == "jsonl":
        return self._generate_jsonl_records(n_messages, rng, theme=theme)
    if wf.tree and wf.tree.container_path:
        return self._generate_tree_json(n_messages, rng, theme=theme)
    if wf.messages_path:
        return self._generate_linear_json(n_messages, rng, theme=theme)
    return self._generate_from_schema(self.schema, rng)


def _generate_tree_json(self, n_messages: int, rng: random.Random, *, theme: ConversationTheme | None = None) -> dict:
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
    nodes: list[dict] = []

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
    if theme is not None and "title" in self.schema.get("properties", {}):
        top["title"] = theme.title
    return top


def _generate_linear_json(self, n_messages: int, rng: random.Random, *, theme: ConversationTheme | None = None) -> dict:
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


def _generate_jsonl_records(self, n_messages: int, rng: random.Random, *, theme: ConversationTheme | None = None) -> list[dict]:
    tree_cfg = self.wire_format.tree
    records: list[dict] = []
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


def _role_cycle(self) -> list[str]:
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


def _ensure_wire_format(
    self,
    data: dict,
    role: str,
    rng: random.Random,
    index: int,
    base_ts: float = 1700000000.0,
    theme: ConversationTheme | None = None,
) -> None:
    ts = base_ts + index * 60
    match self.provider:
        case "chatgpt":
            self._ensure_wire_chatgpt(data, role, rng, ts, index=index, theme=theme)
        case "claude-ai":
            self._ensure_wire_claude_ai(data, role, rng, ts, index=index, theme=theme)
        case "claude-code":
            self._ensure_wire_claude_code(data, role, rng, ts, index=index, theme=theme)
        case "codex":
            self._ensure_wire_codex(data, role, rng, ts, index=index, theme=theme)
        case "gemini":
            self._ensure_wire_gemini(data, role, rng, index=index, theme=theme)


def _ensure_wire_chatgpt(self, data: dict, role: str, rng: random.Random, ts: float, *, index: int, theme: ConversationTheme | None) -> None:
    msg = data.get("message")
    if not isinstance(msg, dict):
        msg = {"id": str(uuid.UUID(int=rng.getrandbits(128), version=4))}
        data["message"] = msg
    author = msg.setdefault("author", {})
    if isinstance(author, dict):
        author.setdefault("role", role)
    content = msg.setdefault("content", {})
    if isinstance(content, dict):
        if "parts" not in content or not content["parts"]:
            content["parts"] = [_text_for_role(rng, role, turn_index=index, theme=theme)]
        content.setdefault("content_type", "text")
    msg.setdefault("create_time", ts)
    msg.setdefault("id", str(uuid.UUID(int=rng.getrandbits(128), version=4)))


def _ensure_wire_claude_ai(self, data: dict, role: str, rng: random.Random, ts: float, *, index: int, theme: ConversationTheme | None) -> None:
    data.setdefault("uuid", str(uuid.UUID(int=rng.getrandbits(128), version=4)))
    data.setdefault("sender", role)
    if not data.get("text"):
        data["text"] = _text_for_role(rng, role, turn_index=index, theme=theme)
    if "created_at" not in data:
        data["created_at"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _ensure_wire_claude_code(self, data: dict, role: str, rng: random.Random, ts: float, *, index: int, theme: ConversationTheme | None) -> None:
    data.setdefault("type", role)
    if not isinstance(data.get("message"), dict):
        data["message"] = {}
    msg = data["message"]
    msg.setdefault("role", role)
    if "content" not in msg:
        msg["content"] = [{"type": "text", "text": _text_for_role(rng, role, turn_index=index, theme=theme)}]
    if "timestamp" not in data:
        data["timestamp"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _ensure_wire_codex(self, data: dict, role: str, rng: random.Random, ts: float, *, index: int, theme: ConversationTheme | None) -> None:
    data["type"] = "message"
    data.setdefault("role", role)
    if "content" not in data:
        content_type = "input_text" if role == "user" else "output_text"
        data["content"] = [{"type": content_type, "text": _text_for_role(rng, role, turn_index=index, theme=theme)}]
    data.setdefault("id", str(uuid.UUID(int=rng.getrandbits(128), version=4)))
    if "timestamp" not in data:
        data["timestamp"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    data.pop("payload", None)


def _ensure_wire_gemini(self, data: dict, role: str, rng: random.Random, *, index: int, theme: ConversationTheme | None) -> None:
    data.setdefault("role", role)
    if not data.get("text"):
        data["text"] = _text_for_role(rng, role, turn_index=index, theme=theme)


__all__ = [
    "_ensure_wire_chatgpt",
    "_ensure_wire_claude_ai",
    "_ensure_wire_claude_code",
    "_ensure_wire_codex",
    "_ensure_wire_format",
    "_ensure_wire_gemini",
    "_generate_conversation",
    "_generate_jsonl_records",
    "_generate_linear_json",
    "_generate_tree_json",
    "_role_cycle",
    "generate_batch",
]

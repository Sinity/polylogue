"""Schema-driven synthetic conversation generator.

Generates realistic provider-format test data from annotated JSON schemas,
eliminating the need for manual fixtures, hardcoded builders, and real exports.

The annotated schemas (with ``x-polylogue-*`` extensions) capture enum values,
format patterns, numeric ranges, and field frequencies from real data. This
module uses those annotations to generate structurally valid synthetic data
that round-trips through all provider parsers.

Usage::

    corpus = SyntheticCorpus.for_provider("chatgpt")
    raw_bytes = corpus.generate(count=5, seed=42)
    # Each element is valid wire-format bytes that parsers can consume

    # All providers at once:
    for provider in SyntheticCorpus.available_providers():
        corpus = SyntheticCorpus.for_provider(provider)
        for raw in corpus.generate(count=3, seed=0):
            parser.parse(raw)  # works
"""

from __future__ import annotations

import gzip
import json
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Schema directory (in-package baseline schemas)
SCHEMA_DIR = Path(__file__).parent / "providers"


# =============================================================================
# Wire Format Configuration
# =============================================================================


@dataclass(frozen=True)
class TreeConfig:
    """Configuration for tree-structured message formats."""

    container_path: str | None = None  # Top-level key containing the tree dict
    key_field: str = "id"
    parent_field: str = "parent"
    children_field: str | None = None
    session_field: str | None = None


@dataclass(frozen=True)
class WireFormat:
    """Wire format configuration for a provider's export format."""

    encoding: str  # "json" | "jsonl"
    tree: TreeConfig | None = None
    messages_path: str | None = None  # Dot-path to messages array


# Per-provider wire format configs — the only manual piece (~50 lines).
# Describes HOW the format is structured, not WHAT conversations say.
PROVIDER_WIRE_FORMATS: dict[str, WireFormat] = {
    "chatgpt": WireFormat(
        encoding="json",
        tree=TreeConfig(
            container_path="mapping",
            key_field="id",
            parent_field="parent",
            children_field="children",
        ),
    ),
    "claude-code": WireFormat(
        encoding="jsonl",
        tree=TreeConfig(
            key_field="uuid",
            parent_field="parentUuid",
            session_field="sessionId",
        ),
    ),
    "claude-ai": WireFormat(
        encoding="json",
        messages_path="chat_messages",
    ),
    "codex": WireFormat(
        encoding="jsonl",
    ),
    "gemini": WireFormat(
        encoding="json",
        messages_path="chunkedPrompt.chunks",
    ),
}


# =============================================================================
# Placeholder Text
# =============================================================================

# Minimal plausible text per role — parsers don't validate content, but
# having non-empty text ensures the parser doesn't skip the message.
_ROLE_TEXTS: dict[str, list[str]] = {
    "user": [
        "Can you help me debug this issue?",
        "I need to implement a function that processes this data.",
        "What's the best approach for handling errors here?",
        "Could you review this code for potential issues?",
    ],
    "assistant": [
        "I'll analyze the issue. Looking at the code structure...",
        "Here's an implementation:\n\n```python\ndef process(data):\n    return [x for x in data if x]\n```",
        "After reviewing, I found several areas for improvement.",
        "The module structure looks good. A few suggestions:",
    ],
    "system": ["You are a helpful programming assistant."],
    "human": ["Can you explain how this works?", "I'm trying to understand the architecture."],
    "model": ["I'll explain step by step.", "Here's a breakdown of the architecture."],
    "tool": ["Function executed successfully.", "Error: resource not found."],
}


def _text_for_role(rng: random.Random, role: str) -> str:
    """Generate plausible text content for a given role."""
    texts = _ROLE_TEXTS.get(role, _ROLE_TEXTS["user"])
    return rng.choice(texts)


# =============================================================================
# Core Generator
# =============================================================================


class SyntheticCorpus:
    """Generate synthetic provider data from annotated schemas.

    The generator works in two layers:
    1. **Schema-driven**: Recursively generates data matching the JSON schema
       structure, using ``x-polylogue-*`` annotations for realistic values
       (enum selection, UUID generation, timestamp ranges, etc.)
    2. **Wire format fixup**: Ensures parser-essential fields have known-good
       values (correct roles, valid tree linkage, proper timestamps).
    """

    def __init__(self, schema: dict[str, Any], wire_format: WireFormat, provider: str):
        self.schema = schema
        self.wire_format = wire_format
        self.provider = provider

    @classmethod
    def for_provider(cls, provider: str) -> SyntheticCorpus:
        """Create a corpus generator for a specific provider."""
        schema_path = SCHEMA_DIR / f"{provider}.schema.json.gz"
        if not schema_path.exists():
            raise FileNotFoundError(f"No schema for provider {provider}: {schema_path}")

        schema = json.loads(gzip.decompress(schema_path.read_bytes()))
        wire_format = PROVIDER_WIRE_FORMATS.get(provider)
        if not wire_format:
            raise ValueError(f"No wire format config for provider: {provider}")

        return cls(schema, wire_format, provider)

    @classmethod
    def available_providers(cls) -> list[str]:
        """List providers that have both schemas and wire format configs."""
        return [
            p for p in PROVIDER_WIRE_FORMATS
            if (SCHEMA_DIR / f"{p}.schema.json.gz").exists()
        ]

    def generate(
        self,
        count: int = 5,
        messages_per_conversation: range = range(3, 15),
        seed: int | None = None,
    ) -> list[bytes]:
        """Generate ``count`` conversations as wire-format bytes.

        Args:
            count: Number of conversations to generate.
            messages_per_conversation: Range of message counts per conversation.
            seed: Random seed for reproducibility.

        Returns:
            List of raw bytes, each a valid wire-format conversation.
        """
        rng = random.Random(seed)
        results = []
        for _ in range(count):
            n_messages = rng.choice(messages_per_conversation)
            data = self._generate_conversation(n_messages, rng)
            raw = self._serialize(data)
            results.append(raw)
        return results

    # ── Conversation-level dispatch ──────────────────────────────────────

    def _generate_conversation(self, n_messages: int, rng: random.Random) -> Any:
        """Generate one conversation's data structure."""
        wf = self.wire_format

        if wf.encoding == "jsonl":
            return self._generate_jsonl_records(n_messages, rng)

        if wf.tree and wf.tree.container_path:
            return self._generate_tree_json(n_messages, rng)

        if wf.messages_path:
            return self._generate_linear_json(n_messages, rng)

        # Fallback: pure schema-driven
        return self._generate_from_schema(self.schema, rng)

    # ── JSON with tree structure (ChatGPT) ──────────────────────────────

    def _generate_tree_json(self, n_messages: int, rng: random.Random) -> dict:
        tree_cfg = self.wire_format.tree
        assert tree_cfg is not None and tree_cfg.container_path is not None

        # Generate top-level structure, skipping the tree container
        top = self._generate_from_schema(
            self.schema, rng, skip_keys={tree_cfg.container_path}
        )
        if not isinstance(top, dict):
            top = {}

        # Get the node schema from additionalProperties
        container_schema = self.schema.get("properties", {}).get(
            tree_cfg.container_path, {}
        )
        node_schema = container_schema.get("additionalProperties", {})

        # Generate tree nodes
        nodes: list[dict] = []
        roles = self._role_cycle()
        base_ts = rng.uniform(1670000000, 1760000000)

        for i in range(n_messages):
            node = self._generate_from_schema(node_schema, rng)
            if not isinstance(node, dict):
                node = {}

            node_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))
            node[tree_cfg.key_field] = node_id

            # Tree linkage
            if i == 0:
                node[tree_cfg.parent_field] = None
                if tree_cfg.children_field:
                    node[tree_cfg.children_field] = []
            else:
                # Parent is a recent ancestor (creates realistic linear chains
                # with occasional branches)
                parent_idx = rng.randint(max(0, i - 3), i - 1)
                parent = nodes[parent_idx]
                node[tree_cfg.parent_field] = parent[tree_cfg.key_field]
                if tree_cfg.children_field:
                    parent.setdefault(tree_cfg.children_field, []).append(node_id)
                    node[tree_cfg.children_field] = []

            role = roles[i % len(roles)]
            self._fix_message_fields(node, role, rng, i, base_ts=base_ts)
            nodes.append(node)

        # Build mapping dict (UUID → node)
        mapping = {n[tree_cfg.key_field]: n for n in nodes}
        top[tree_cfg.container_path] = mapping

        # Set current_node to last message
        if "current_node" in self.schema.get("properties", {}):
            top["current_node"] = nodes[-1][tree_cfg.key_field]

        return top

    # ── JSON with linear messages (Claude AI, Gemini) ───────────────────

    def _generate_linear_json(self, n_messages: int, rng: random.Random) -> dict:
        msgs_path = self.wire_format.messages_path
        assert msgs_path is not None

        parts = msgs_path.split(".")

        # Navigate schema to find the item schema for messages.
        # Two cases:
        #   1. Schema describes full file (Claude AI) — navigate to items schema
        #   2. Schema describes individual items (Gemini) — use schema directly
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
            # Generate top-level from full schema, skipping the messages root key
            top = self._generate_from_schema(self.schema, rng, skip_keys={parts[0]})
        else:
            # Schema describes individual items — generate bare wrapper
            item_schema = self.schema
            top = {}

        if not isinstance(top, dict):
            top = {}

        # Generate messages
        roles = self._role_cycle()
        messages = []
        base_ts = rng.uniform(1670000000, 1760000000)

        for i in range(n_messages):
            msg = self._generate_from_schema(item_schema, rng)
            if not isinstance(msg, dict):
                msg = {}
            role = roles[i % len(roles)]
            self._fix_message_fields(msg, role, rng, i, base_ts=base_ts)
            messages.append(msg)

        # Set messages at the correct nested path
        target = top
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                target[part] = messages
            else:
                target.setdefault(part, {})
                target = target[part]

        return top

    # ── JSONL records (Claude Code, Codex) ──────────────────────────────

    def _generate_jsonl_records(self, n_messages: int, rng: random.Random) -> list[dict]:
        tree_cfg = self.wire_format.tree

        records: list[dict] = []
        roles = self._role_cycle()
        session_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))
        base_ts = rng.uniform(1670000000, 1760000000)

        for i in range(n_messages):
            record = self._generate_from_schema(self.schema, rng)
            if not isinstance(record, dict):
                record = {}

            role = roles[i % len(roles)]
            self._fix_message_fields(record, role, rng, i, base_ts=base_ts)

            if tree_cfg:
                # UUID-based tree linkage
                node_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))
                record[tree_cfg.key_field] = node_id

                if i == 0:
                    record[tree_cfg.parent_field] = None
                else:
                    record[tree_cfg.parent_field] = records[i - 1].get(
                        tree_cfg.key_field
                    )

                if tree_cfg.session_field:
                    record[tree_cfg.session_field] = session_id

            records.append(record)

        return records

    # ── Provider role patterns ──────────────────────────────────────────

    def _role_cycle(self) -> list[str]:
        """Role alternation pattern for this provider."""
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

    # ── Parser-essential field fixup ────────────────────────────────────

    def _fix_message_fields(
        self,
        data: dict,
        role: str,
        rng: random.Random,
        index: int,
        base_ts: float = 1700000000.0,
    ) -> None:
        """Ensure parser-essential fields have valid values.

        The schema-driven generation produces structurally valid data, but
        parsers have semantic requirements (correct role values, non-empty
        text content, valid timestamps). This method overrides the critical
        fields while leaving schema-generated fields intact.
        """
        ts = base_ts + index * 60  # 1-minute intervals

        match self.provider:
            case "chatgpt":
                self._fix_chatgpt(data, role, rng, ts)
            case "claude-ai":
                self._fix_claude_ai(data, role, rng, ts)
            case "claude-code":
                self._fix_claude_code(data, role, rng, ts)
            case "codex":
                self._fix_codex(data, role, rng, ts)
            case "gemini":
                self._fix_gemini(data, role, rng)

    def _fix_chatgpt(self, data: dict, role: str, rng: random.Random, ts: float) -> None:
        msg = data.get("message")
        if not isinstance(msg, dict):
            # Schema might generate null message — force a valid one
            msg = {"id": str(uuid.UUID(int=rng.getrandbits(128), version=4))}
            data["message"] = msg

        author = msg.setdefault("author", {})
        if isinstance(author, dict):
            author["role"] = role

        content = msg.setdefault("content", {})
        if isinstance(content, dict):
            if "parts" not in content or not content["parts"]:
                content["parts"] = [_text_for_role(rng, role)]
            content.setdefault("content_type", "text")

        msg.setdefault("create_time", ts)
        msg.setdefault("id", str(uuid.UUID(int=rng.getrandbits(128), version=4)))

    def _fix_claude_ai(self, data: dict, role: str, rng: random.Random, ts: float) -> None:
        data.setdefault("uuid", str(uuid.UUID(int=rng.getrandbits(128), version=4)))
        data["sender"] = role
        if not data.get("text"):
            data["text"] = _text_for_role(rng, role)
        if "created_at" not in data:
            data["created_at"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    def _fix_claude_code(self, data: dict, role: str, rng: random.Random, ts: float) -> None:
        data["type"] = role
        if not isinstance(data.get("message"), dict):
            data["message"] = {}
        msg = data["message"]
        # The extraction layer reads message.role; the parser reads top-level type.
        # Real Claude Code data has message.role set; ensure synthetic does too.
        msg["role"] = role
        if "content" not in msg:
            msg["content"] = [{"type": "text", "text": _text_for_role(rng, role)}]
        if "timestamp" not in data:
            data["timestamp"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    def _fix_codex(self, data: dict, role: str, rng: random.Random, ts: float) -> None:
        data["type"] = "message"
        data["role"] = role
        if "content" not in data:
            content_type = "input_text" if role == "user" else "output_text"
            data["content"] = [{"type": content_type, "text": _text_for_role(rng, role)}]
        data.setdefault("id", str(uuid.UUID(int=rng.getrandbits(128), version=4)))
        if "timestamp" not in data:
            data["timestamp"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        # Remove payload key — its presence triggers "envelope" format detection,
        # which expects the message inside payload rather than at top level
        data.pop("payload", None)

    def _fix_gemini(self, data: dict, role: str, rng: random.Random) -> None:
        data["role"] = role
        if not data.get("text"):
            data["text"] = _text_for_role(rng, role)

    # ── Recursive schema-to-data generation ─────────────────────────────

    def _generate_from_schema(
        self,
        schema: dict[str, Any],
        rng: random.Random,
        *,
        skip_keys: set[str] | None = None,
        depth: int = 0,
        max_depth: int = 6,
    ) -> Any:
        """Recursively generate data matching a JSON schema node.

        Uses ``x-polylogue-*`` annotations for realistic values when available,
        falls back to sensible defaults otherwise. Depth-limited to prevent
        unbounded recursion in deeply nested schemas.
        """
        if depth > max_depth or not isinstance(schema, dict):
            return None

        # Handle polymorphic types (anyOf/oneOf)
        for keyword in ("anyOf", "oneOf"):
            if keyword in schema:
                variants = schema[keyword]
                # Prefer non-null, non-empty variants for parser coverage
                non_null = [
                    v for v in variants
                    if v.get("type") != "null" and v.get("type") is not None
                ]
                chosen = rng.choice(non_null) if non_null else rng.choice(variants)
                return self._generate_from_schema(
                    chosen, rng, skip_keys=skip_keys, depth=depth, max_depth=max_depth
                )

        schema_type = schema.get("type")

        # Handle type arrays like ["null", "string"]
        if isinstance(schema_type, list):
            non_null = [t for t in schema_type if t != "null"]
            schema_type = rng.choice(non_null) if non_null else "null"

        # Skip low-frequency optional fields (at depth > 0)
        freq = schema.get("x-polylogue-frequency", 1.0)
        if depth > 0 and freq < 1.0 and rng.random() > freq:
            return None

        match schema_type:
            case "object":
                return self._generate_object(
                    schema, rng, skip_keys=skip_keys, depth=depth, max_depth=max_depth
                )
            case "string":
                return self._generate_string(schema, rng)
            case "number":
                return self._generate_number(schema, rng, is_int=False)
            case "integer":
                return self._generate_number(schema, rng, is_int=True)
            case "array":
                return self._generate_array(schema, rng, depth=depth, max_depth=max_depth)
            case "boolean":
                return rng.choice([True, False])
            case "null":
                return None
            case _:
                # No type but has properties → implicit object
                if "properties" in schema:
                    return self._generate_object(
                        schema, rng, skip_keys=skip_keys, depth=depth, max_depth=max_depth
                    )
                return None

    def _generate_object(
        self,
        schema: dict,
        rng: random.Random,
        *,
        skip_keys: set[str] | None = None,
        depth: int = 0,
        max_depth: int = 6,
    ) -> dict:
        """Generate an object from schema properties."""
        obj: dict[str, Any] = {}
        for prop_name, prop_schema in schema.get("properties", {}).items():
            if skip_keys and prop_name in skip_keys:
                continue

            # Skip low-frequency fields probabilistically
            freq = prop_schema.get("x-polylogue-frequency", 1.0)
            if freq < 1.0 and rng.random() > freq:
                continue

            value = self._generate_from_schema(
                prop_schema, rng, depth=depth + 1, max_depth=max_depth
            )
            if value is not None:
                obj[prop_name] = value

        # Don't generate additionalProperties — tree/container generation
        # handles dynamic keys separately
        return obj

    def _generate_string(self, schema: dict, rng: random.Random) -> str:
        """Generate a string value using schema annotations."""
        # Observed enum values (most reliable source)
        if values := schema.get("x-polylogue-values"):
            return rng.choice(values)

        # Format-based generation
        match schema.get("x-polylogue-format"):
            case "uuid4" | "uuid":
                return str(uuid.UUID(int=rng.getrandbits(128), version=4))
            case "hex-id":
                return rng.randbytes(12).hex()
            case "iso8601":
                ts = rng.uniform(1670000000, 1760000000)
                return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            case "unix-epoch" | "unix-epoch-str":
                return str(rng.uniform(1670000000, 1760000000))
            case "url":
                return f"https://example.com/{rng.randint(1000, 9999)}"
            case "email":
                return f"user{rng.randint(1, 999)}@example.com"
            case "mime-type":
                return rng.choice(["text/plain", "application/json", "text/html"])
            case "base64":
                return rng.randbytes(24).hex()

        # Multiline content
        if schema.get("x-polylogue-multiline"):
            return _text_for_role(rng, "assistant")

        return f"synthetic-{rng.randint(0, 99999)}"

    def _generate_number(
        self, schema: dict, rng: random.Random, *, is_int: bool = False
    ) -> float | int:
        """Generate a numeric value using schema annotations."""
        if r := schema.get("x-polylogue-range"):
            lo, hi = r
            val = rng.uniform(lo, hi)
        elif schema.get("x-polylogue-format") == "unix-epoch":
            val = rng.uniform(1670000000, 1760000000)
        else:
            val = rng.uniform(0, 1000)

        return int(val) if is_int else val

    def _generate_array(
        self,
        schema: dict,
        rng: random.Random,
        *,
        depth: int = 0,
        max_depth: int = 6,
    ) -> list:
        """Generate an array from schema items definition."""
        item_schema = schema.get("items", {})

        # Use array length annotation (clamp to reasonable output size)
        if lengths := schema.get("x-polylogue-array-lengths"):
            lo, hi = lengths
            clamped_lo = min(max(0, lo), 5)
            clamped_hi = min(max(hi, clamped_lo), 5)
            n = rng.randint(clamped_lo, clamped_hi)
        else:
            n = rng.randint(1, 3)

        return [
            self._generate_from_schema(item_schema, rng, depth=depth + 1, max_depth=max_depth)
            for _ in range(n)
        ]

    # ── Wire format serialization ───────────────────────────────────────

    def _serialize(self, data: Any) -> bytes:
        """Serialize generated data to wire-format bytes."""
        if self.wire_format.encoding == "jsonl":
            # data is a list of records
            lines = [json.dumps(record, separators=(",", ":")) for record in data]
            return ("\n".join(lines) + "\n").encode("utf-8")
        else:
            return json.dumps(data, indent=2).encode("utf-8")

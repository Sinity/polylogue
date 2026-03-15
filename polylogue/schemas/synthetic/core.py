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

import json
import random
import uuid
from datetime import datetime, timezone
from typing import Any

from polylogue.schemas.registry import SchemaRegistry, canonical_schema_provider
from polylogue.schemas.synthetic.relations import RelationConstraintSolver
from polylogue.schemas.synthetic.semantic_values import (
    SemanticValueGenerator,
    _text_for_role,
)
from polylogue.schemas.synthetic.showcase import ConversationTheme, _SHOWCASE_THEMES
from polylogue.schemas.synthetic.wire_formats import (
    PROVIDER_WIRE_FORMATS,
    TreeConfig,
    WireFormat,
)


class SyntheticCorpus:
    """Generate synthetic provider data from annotated schemas.

    The generator works in three layers:
    1. **Semantic-role-driven**: Fields with ``x-polylogue-semantic-role``
       annotations are generated using contextually appropriate values.
    2. **Schema-driven**: Recursively generates data matching the JSON schema
       structure, using ``x-polylogue-*`` annotations for realistic values
       (enum selection, UUID generation, timestamp ranges, etc.)
    3. **Wire format fixup**: Ensures parser-essential fields have known-good
       values (correct roles, valid tree linkage, proper timestamps).

    Relational constraints (``x-polylogue-foreign-keys``,
    ``x-polylogue-mutually-exclusive``, ``x-polylogue-string-lengths``)
    are enforced across the generated data for structural consistency.
    """

    def __init__(self, schema: dict[str, Any], wire_format: WireFormat, provider: str):
        self.schema = schema
        self.wire_format = wire_format
        self.provider = provider
        self._relation_solver = RelationConstraintSolver(schema)

    @classmethod
    def for_provider(cls, provider: str) -> SyntheticCorpus:
        """Create a corpus generator for a specific provider."""
        canonical_provider = canonical_schema_provider(provider)
        schema = SchemaRegistry().get_schema(canonical_provider, version="latest")
        if schema is None:
            raise FileNotFoundError(f"No schema for provider {provider} (canonical: {canonical_provider})")

        wire_format = PROVIDER_WIRE_FORMATS.get(canonical_provider)
        if not wire_format:
            raise ValueError(f"No wire format config for provider: {canonical_provider}")

        return cls(schema, wire_format, canonical_provider)

    @classmethod
    def available_providers(cls) -> list[str]:
        """List providers that have both schemas and wire format configs."""
        schema_providers = set(SchemaRegistry().list_providers())
        return [
            p for p in PROVIDER_WIRE_FORMATS
            if p in schema_providers
        ]

    def generate(
        self,
        count: int = 5,
        messages_per_conversation: range = range(3, 15),
        seed: int | None = None,
        style: str = "default",
    ) -> list[bytes]:
        """Generate ``count`` conversations as wire-format bytes.

        Args:
            count: Number of conversations to generate.
            messages_per_conversation: Range of message counts per conversation.
            seed: Random seed for reproducibility.
            style: Generation style. ``default`` preserves baseline random text;
                ``showcase`` generates coherent narrative turns for human-readable
                demo/showcase corpora.

        Returns:
            List of raw bytes, each a valid wire-format conversation.
        """
        if style not in {"default", "showcase"}:
            raise ValueError(f"Unknown synthetic style: {style}")

        rng = random.Random(seed)
        results = []
        for _ in range(count):
            n_messages = rng.choice(messages_per_conversation)
            theme = rng.choice(_SHOWCASE_THEMES) if style == "showcase" else None
            data = self._generate_conversation(n_messages, rng, theme=theme)
            raw = self._serialize(data)
            results.append(raw)
        return results

    # -- Conversation-level dispatch ---------------------------------------

    def _generate_conversation(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: ConversationTheme | None = None,
    ) -> Any:
        """Generate one conversation's data structure."""
        wf = self.wire_format

        if wf.encoding == "jsonl":
            return self._generate_jsonl_records(n_messages, rng, theme=theme)

        if wf.tree and wf.tree.container_path:
            return self._generate_tree_json(n_messages, rng, theme=theme)

        if wf.messages_path:
            return self._generate_linear_json(n_messages, rng, theme=theme)

        # Fallback: pure schema-driven
        return self._generate_from_schema(self.schema, rng)

    # -- JSON with tree structure (ChatGPT) --------------------------------

    def _generate_tree_json(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: ConversationTheme | None = None,
    ) -> dict:
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
        self._semantic_gen = SemanticValueGenerator(
            rng, theme=theme, base_ts=base_ts, role_cycle=roles,
        )

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
            self._ensure_wire_format(node, role, rng, i, base_ts=base_ts, theme=theme)
            self._semantic_gen.advance_turn()
            nodes.append(node)

        # Build mapping dict (UUID -> node)
        mapping = {n[tree_cfg.key_field]: n for n in nodes}
        top[tree_cfg.container_path] = mapping

        # Set current_node to last message
        if "current_node" in self.schema.get("properties", {}):
            top["current_node"] = nodes[-1][tree_cfg.key_field]

        if theme is not None and "title" in self.schema.get("properties", {}):
            top["title"] = theme.title

        return top

    # -- JSON with linear messages (Claude AI, Gemini) ---------------------

    def _generate_linear_json(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: ConversationTheme | None = None,
    ) -> dict:
        msgs_path = self.wire_format.messages_path
        assert msgs_path is not None

        parts = msgs_path.split(".")

        # Navigate schema to find the item schema for messages.
        # Two cases:
        #   1. Schema describes full file (Claude AI) -- navigate to items schema
        #   2. Schema describes individual items (Gemini) -- use schema directly
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
            # Schema describes individual items -- generate bare wrapper
            item_schema = self.schema
            top = {}

        if not isinstance(top, dict):
            top = {}

        # Generate messages
        roles = self._role_cycle()
        messages = []
        base_ts = rng.uniform(1670000000, 1760000000)
        self._semantic_gen = SemanticValueGenerator(
            rng, theme=theme, base_ts=base_ts, role_cycle=roles,
        )

        for i in range(n_messages):
            msg = self._generate_from_schema(item_schema, rng)
            if not isinstance(msg, dict):
                msg = {}
            role = roles[i % len(roles)]
            self._ensure_wire_format(msg, role, rng, i, base_ts=base_ts, theme=theme)
            self._semantic_gen.advance_turn()
            messages.append(msg)

        # Set messages at the correct nested path
        target = top
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                target[part] = messages
            else:
                target.setdefault(part, {})
                target = target[part]

        if theme is not None and self.provider == "claude-ai" and "name" in self.schema.get("properties", {}):
            top["name"] = theme.title

        return top

    # -- JSONL records (Claude Code, Codex) --------------------------------

    def _generate_jsonl_records(
        self,
        n_messages: int,
        rng: random.Random,
        *,
        theme: ConversationTheme | None = None,
    ) -> list[dict]:
        tree_cfg = self.wire_format.tree

        records: list[dict] = []
        roles = self._role_cycle()
        session_id = str(uuid.UUID(int=rng.getrandbits(128), version=4))
        base_ts = rng.uniform(1670000000, 1760000000)
        self._semantic_gen = SemanticValueGenerator(
            rng, theme=theme, base_ts=base_ts, role_cycle=roles,
        )

        for i in range(n_messages):
            record = self._generate_from_schema(self.schema, rng)
            if not isinstance(record, dict):
                record = {}

            role = roles[i % len(roles)]
            self._ensure_wire_format(record, role, rng, i, base_ts=base_ts, theme=theme)
            self._semantic_gen.advance_turn()

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

            if theme is not None:
                payload = record.get("payload")
                if isinstance(payload, dict) and "instructions" in payload:
                    payload["instructions"] = theme.instructions
                if "instructions" in record:
                    record["instructions"] = theme.instructions

            records.append(record)

        return records

    # -- Provider role patterns --------------------------------------------

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

    # -- Wire-format fixup (structural only) ---------------------------------
    # Semantic content (roles, text bodies, timestamps) is now generated by
    # SemanticValueGenerator via schema annotations. These methods only ensure
    # the structural envelope is valid, using setdefault for semantic fields
    # so the annotation-driven values are preserved.

    def _ensure_wire_format(
        self,
        data: dict,
        role: str,
        rng: random.Random,
        index: int,
        base_ts: float = 1700000000.0,
        theme: ConversationTheme | None = None,
    ) -> None:
        """Ensure wire-format envelope is valid for the provider's parser.

        Semantic fields (role, text, timestamps) use ``setdefault`` so
        annotation-driven values from ``SemanticValueGenerator`` are preserved.
        Structural fields (dict shapes, UUIDs, envelope markers) are
        unconditionally set.
        """
        ts = base_ts + index * 60  # 1-minute intervals

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

    def _ensure_wire_chatgpt(
        self,
        data: dict,
        role: str,
        rng: random.Random,
        ts: float,
        *,
        index: int,
        theme: ConversationTheme | None,
    ) -> None:
        # Structural: ensure message dict exists
        msg = data.get("message")
        if not isinstance(msg, dict):
            msg = {"id": str(uuid.UUID(int=rng.getrandbits(128), version=4))}
            data["message"] = msg

        # Structural: ensure author dict exists
        author = msg.setdefault("author", {})
        # Semantic (fallback): role
        if isinstance(author, dict):
            author.setdefault("role", role)

        # Structural: ensure content dict exists with content_type
        content = msg.setdefault("content", {})
        if isinstance(content, dict):
            # Semantic (fallback): parts text
            if "parts" not in content or not content["parts"]:
                content["parts"] = [_text_for_role(rng, role, turn_index=index, theme=theme)]
            content.setdefault("content_type", "text")

        # Semantic (fallback): timestamp and ID
        msg.setdefault("create_time", ts)
        msg.setdefault("id", str(uuid.UUID(int=rng.getrandbits(128), version=4)))

    def _ensure_wire_claude_ai(
        self,
        data: dict,
        role: str,
        rng: random.Random,
        ts: float,
        *,
        index: int,
        theme: ConversationTheme | None,
    ) -> None:
        # Structural: UUID
        data.setdefault("uuid", str(uuid.UUID(int=rng.getrandbits(128), version=4)))
        # Semantic (fallback): role, text, timestamp
        data.setdefault("sender", role)
        if not data.get("text"):
            data["text"] = _text_for_role(rng, role, turn_index=index, theme=theme)
        if "created_at" not in data:
            data["created_at"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    def _ensure_wire_claude_code(
        self,
        data: dict,
        role: str,
        rng: random.Random,
        ts: float,
        *,
        index: int,
        theme: ConversationTheme | None,
    ) -> None:
        # Semantic (fallback): type field IS the role for claude-code
        data.setdefault("type", role)
        # Structural: ensure message dict exists
        if not isinstance(data.get("message"), dict):
            data["message"] = {}
        msg = data["message"]
        # Semantic (fallback): message.role mirrors top-level type
        msg.setdefault("role", role)
        # Semantic (fallback): content blocks
        if "content" not in msg:
            msg["content"] = [{
                "type": "text",
                "text": _text_for_role(rng, role, turn_index=index, theme=theme),
            }]
        # Semantic (fallback): timestamp
        if "timestamp" not in data:
            data["timestamp"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    def _ensure_wire_codex(
        self,
        data: dict,
        role: str,
        rng: random.Random,
        ts: float,
        *,
        index: int,
        theme: ConversationTheme | None,
    ) -> None:
        # Structural: envelope marker (always "message", not a role value)
        data["type"] = "message"
        # Semantic (fallback): role
        data.setdefault("role", role)
        # Semantic (fallback): content blocks
        if "content" not in data:
            content_type = "input_text" if role == "user" else "output_text"
            data["content"] = [{
                "type": content_type,
                "text": _text_for_role(rng, role, turn_index=index, theme=theme),
            }]
        # Structural: ID
        data.setdefault("id", str(uuid.UUID(int=rng.getrandbits(128), version=4)))
        # Semantic (fallback): timestamp
        if "timestamp" not in data:
            data["timestamp"] = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        # Structural: remove payload key (triggers wrong format detection)
        data.pop("payload", None)

    def _ensure_wire_gemini(
        self,
        data: dict,
        role: str,
        rng: random.Random,
        *,
        index: int,
        theme: ConversationTheme | None,
    ) -> None:
        # Semantic (fallback): role and text
        data.setdefault("role", role)
        if not data.get("text"):
            data["text"] = _text_for_role(rng, role, turn_index=index, theme=theme)

    # -- Recursive schema-to-data generation -------------------------------

    def _generate_from_schema(
        self,
        schema: dict[str, Any],
        rng: random.Random,
        *,
        skip_keys: set[str] | None = None,
        depth: int = 0,
        max_depth: int = 6,
        path: str = "$",
    ) -> Any:
        """Recursively generate data matching a JSON schema node.

        Uses ``x-polylogue-*`` annotations for realistic values when available,
        falls back to sensible defaults otherwise. Depth-limited to prevent
        unbounded recursion in deeply nested schemas.

        The ``path`` parameter tracks the current position in the schema tree
        for matching against relational constraint paths.
        """
        if depth > max_depth or not isinstance(schema, dict):
            return None

        # Check for semantic role annotation first — SemanticValueGenerator
        # is initialized in each generation path before _generate_from_schema
        semantic_gen = getattr(self, '_semantic_gen', None)
        if schema.get("x-polylogue-semantic-role") and semantic_gen is not None:
            handled, value = semantic_gen.try_generate(schema)
            if handled:
                return value

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
                    chosen, rng, skip_keys=skip_keys, depth=depth, max_depth=max_depth,
                    path=path,
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
                    schema, rng, skip_keys=skip_keys, depth=depth, max_depth=max_depth,
                    path=path,
                )
            case "string":
                value = self._generate_string(schema, rng)
                # Apply string length constraints from relational annotations
                value = self._relation_solver.generate_string_with_length(path, rng, value)
                # Register as potential FK target
                fmt = schema.get("x-polylogue-format")
                if fmt in {"uuid4", "uuid", "hex-id"}:
                    self._relation_solver.register_generated_id(path, value)
                return value
            case "number":
                return self._generate_number(schema, rng, is_int=False)
            case "integer":
                return self._generate_number(schema, rng, is_int=True)
            case "array":
                return self._generate_array(schema, rng, depth=depth, max_depth=max_depth,
                                            path=path)
            case "boolean":
                return rng.choice([True, False])
            case "null":
                return None
            case _:
                # No type but has properties -> implicit object
                if "properties" in schema:
                    return self._generate_object(
                        schema, rng, skip_keys=skip_keys, depth=depth, max_depth=max_depth,
                        path=path,
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
        path: str = "$",
    ) -> dict:
        """Generate an object from schema properties."""
        obj: dict[str, Any] = {}
        properties = schema.get("properties", {})

        # Determine which fields to generate
        candidate_keys = set(properties.keys())
        if skip_keys:
            candidate_keys -= skip_keys

        # Apply mutual exclusion constraints
        if self._relation_solver.mutual_exclusions:
            candidate_keys = self._relation_solver.filter_mutually_exclusive(
                path, candidate_keys, rng,
            )

        for prop_name in properties:
            if prop_name not in candidate_keys:
                continue

            prop_schema = properties[prop_name]

            # Skip low-frequency fields probabilistically
            freq = prop_schema.get("x-polylogue-frequency", 1.0)
            if freq < 1.0 and rng.random() > freq:
                continue

            child_path = f"{path}.{prop_name}"

            # Try foreign-key resolution first
            ref = self._relation_solver.resolve_foreign_key(child_path, rng)
            if ref is not None:
                obj[prop_name] = ref
                continue

            value = self._generate_from_schema(
                prop_schema, rng, depth=depth + 1, max_depth=max_depth,
                path=child_path,
            )
            if value is not None:
                obj[prop_name] = value

        # Don't generate additionalProperties -- tree/container generation
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
        path: str = "$",
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
            self._generate_from_schema(
                item_schema, rng, depth=depth + 1, max_depth=max_depth,
                path=f"{path}[*]",
            )
            for _ in range(n)
        ]

    # -- Wire format serialization -----------------------------------------

    def _serialize(self, data: Any) -> bytes:
        """Serialize generated data to wire-format bytes."""
        if self.wire_format.encoding == "jsonl":
            # data is a list of records
            lines = [json.dumps(record, separators=(",", ":")) for record in data]
            return ("\n".join(lines) + "\n").encode("utf-8")
        else:
            return json.dumps(data, indent=2).encode("utf-8")


__all__ = [
    "SyntheticCorpus",
]

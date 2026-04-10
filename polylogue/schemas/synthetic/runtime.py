"""Recursive schema-to-data generation helpers for synthetic corpora."""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timezone
from typing import Any

from polylogue.schemas.synthetic.semantic_values import _text_for_role


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
    if depth > max_depth or not isinstance(schema, dict):
        return None

    semantic_gen = getattr(self, "_semantic_gen", None)
    if schema.get("x-polylogue-semantic-role") and semantic_gen is not None:
        handled, value = semantic_gen.try_generate(schema)
        if handled:
            return value

    for keyword in ("anyOf", "oneOf"):
        if keyword in schema:
            variants = schema[keyword]
            non_null = [
                variant for variant in variants if variant.get("type") != "null" and variant.get("type") is not None
            ]
            chosen = rng.choice(non_null) if non_null else rng.choice(variants)
            return self._generate_from_schema(
                chosen,
                rng,
                skip_keys=skip_keys,
                depth=depth,
                max_depth=max_depth,
                path=path,
            )

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        non_null = [item for item in schema_type if item != "null"]
        schema_type = rng.choice(non_null) if non_null else "null"

    freq = schema.get("x-polylogue-frequency", 1.0)
    if depth > 0 and freq < 1.0 and rng.random() > freq:
        return None

    match schema_type:
        case "object":
            return self._generate_object(
                schema,
                rng,
                skip_keys=skip_keys,
                depth=depth,
                max_depth=max_depth,
                path=path,
            )
        case "string":
            value = self._generate_string(schema, rng)
            value = self._relation_solver.generate_string_with_length(path, rng, value)
            fmt = schema.get("x-polylogue-format")
            if fmt in {"uuid4", "uuid", "hex-id"}:
                self._relation_solver.register_generated_id(path, value)
            return value
        case "number":
            return self._generate_number(schema, rng, is_int=False)
        case "integer":
            return self._generate_number(schema, rng, is_int=True)
        case "array":
            return self._generate_array(schema, rng, depth=depth, max_depth=max_depth, path=path)
        case "boolean":
            return rng.choice([True, False])
        case "null":
            return None
        case _:
            if "properties" in schema:
                return self._generate_object(
                    schema,
                    rng,
                    skip_keys=skip_keys,
                    depth=depth,
                    max_depth=max_depth,
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
    obj: dict[str, Any] = {}
    properties = schema.get("properties", {})
    candidate_keys = set(properties.keys())
    if skip_keys:
        candidate_keys -= skip_keys

    if self._relation_solver.mutual_exclusions:
        candidate_keys = self._relation_solver.filter_mutually_exclusive(path, candidate_keys, rng)

    for prop_name in properties:
        if prop_name not in candidate_keys:
            continue

        prop_schema = properties[prop_name]
        freq = prop_schema.get("x-polylogue-frequency", 1.0)
        if freq < 1.0 and rng.random() > freq:
            continue

        child_path = f"{path}.{prop_name}"
        ref = self._relation_solver.resolve_foreign_key(child_path, rng)
        if ref is not None:
            obj[prop_name] = ref
            continue

        value = self._generate_from_schema(
            prop_schema,
            rng,
            depth=depth + 1,
            max_depth=max_depth,
            path=child_path,
        )
        if value is not None:
            obj[prop_name] = value

    return obj


def _generate_string(self, schema: dict, rng: random.Random) -> str:
    if values := schema.get("x-polylogue-values"):
        return rng.choice(values)

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

    if schema.get("x-polylogue-multiline"):
        return _text_for_role(rng, "assistant")

    return f"synthetic-{rng.randint(0, 99999)}"


def _generate_number(self, schema: dict, rng: random.Random, *, is_int: bool = False) -> float | int:
    if value_range := schema.get("x-polylogue-range"):
        lo, hi = value_range
        value = rng.uniform(lo, hi)
    elif schema.get("x-polylogue-format") == "unix-epoch":
        value = rng.uniform(1670000000, 1760000000)
    else:
        value = rng.uniform(0, 1000)
    return int(value) if is_int else value


def _generate_array(
    self,
    schema: dict,
    rng: random.Random,
    *,
    depth: int = 0,
    max_depth: int = 6,
    path: str = "$",
) -> list:
    item_schema = schema.get("items", {})
    if lengths := schema.get("x-polylogue-array-lengths"):
        lo, hi = lengths
        clamped_lo = min(max(0, lo), 5)
        clamped_hi = min(max(hi, clamped_lo), 5)
        n_items = rng.randint(clamped_lo, clamped_hi)
    else:
        n_items = rng.randint(1, 3)

    item_allows_null = (
        item_schema.get("type") == "null"
        or (isinstance(item_schema.get("type"), list) and "null" in item_schema["type"])
        or any(variant.get("type") == "null" for variant in item_schema.get("anyOf", []))
        or any(variant.get("type") == "null" for variant in item_schema.get("oneOf", []))
    )
    items = [
        self._generate_from_schema(
            item_schema,
            rng,
            depth=depth + 1,
            max_depth=max_depth,
            path=f"{path}[*]",
        )
        for _ in range(n_items)
    ]
    if not item_allows_null:
        items = [value for value in items if value is not None]
    return items


def _serialize(self, data: Any) -> bytes:
    if self.wire_format.encoding == "jsonl":
        lines = [json.dumps(record, separators=(",", ":")) for record in data]
        return ("\n".join(lines) + "\n").encode("utf-8")
    return json.dumps(data, indent=2).encode("utf-8")


__all__ = [
    "_generate_array",
    "_generate_from_schema",
    "_generate_number",
    "_generate_object",
    "_generate_string",
    "_serialize",
]

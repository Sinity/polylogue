"""Recursive schema-to-data generation helpers for synthetic corpora."""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, TypeAlias

from polylogue.lib.raw_payload.decode import JSONValue
from polylogue.schemas.synthetic.models import SchemaRecord, SchemaValue
from polylogue.schemas.synthetic.semantic_values import SemanticValueGenerator, _text_for_role
from polylogue.schemas.synthetic.wire_formats import WireFormat

if TYPE_CHECKING:
    from polylogue.schemas.synthetic.relations import RelationConstraintSolver

SyntheticRecord: TypeAlias = dict[str, JSONValue]


class _SyntheticRuntimeContext(Protocol):
    wire_format: WireFormat
    _semantic_gen: SemanticValueGenerator | None
    _relation_solver: RelationConstraintSolver

    def _generate_from_schema(
        self,
        schema: SchemaRecord,
        rng: random.Random,
        *,
        skip_keys: set[str] | None = None,
        depth: int = 0,
        max_depth: int = 6,
        path: str = "$",
    ) -> JSONValue: ...

    def _generate_object(
        self,
        schema: SchemaRecord,
        rng: random.Random,
        *,
        skip_keys: set[str] | None = None,
        depth: int = 0,
        max_depth: int = 6,
        path: str = "$",
    ) -> SyntheticRecord: ...

    def _generate_array(
        self,
        schema: SchemaRecord,
        rng: random.Random,
        *,
        depth: int = 0,
        max_depth: int = 6,
        path: str = "$",
    ) -> list[JSONValue]: ...

    def _generate_string(self, schema: SchemaRecord, rng: random.Random) -> str: ...

    def _generate_number(
        self,
        schema: SchemaRecord,
        rng: random.Random,
        *,
        is_int: bool = False,
    ) -> float | int: ...


def _schema_record(value: SchemaValue | object) -> SchemaRecord:
    return value if isinstance(value, dict) else {}


def _schema_records(value: SchemaValue | object) -> list[SchemaRecord]:
    if not isinstance(value, list):
        return []
    return [record for item in value if (record := _schema_record(item))]


def _schema_type(schema: SchemaRecord, rng: random.Random) -> str | None:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return schema_type
    if isinstance(schema_type, list):
        non_null = [item for item in schema_type if isinstance(item, str) and item != "null"]
        return rng.choice(non_null) if non_null else "null"
    return None


def _coerce_float(value: SchemaValue, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return default


def _coerce_int(value: SchemaValue, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return default


def _generate_from_schema(
    self: _SyntheticRuntimeContext,
    schema: SchemaRecord,
    rng: random.Random,
    *,
    skip_keys: set[str] | None = None,
    depth: int = 0,
    max_depth: int = 6,
    path: str = "$",
) -> JSONValue:
    if depth > max_depth or not schema:
        return None

    semantic_role = schema.get("x-polylogue-semantic-role")
    if isinstance(semantic_role, str) and self._semantic_gen is not None:
        handled, value = self._semantic_gen.try_generate(schema)
        if handled:
            return value

    for keyword in ("anyOf", "oneOf"):
        variants = _schema_records(schema.get(keyword))
        if variants:
            non_null = [variant for variant in variants if variant.get("type") != "null"]
            chosen = rng.choice(non_null) if non_null else rng.choice(variants)
            return self._generate_from_schema(
                chosen,
                rng,
                skip_keys=skip_keys,
                depth=depth,
                max_depth=max_depth,
                path=path,
            )

    schema_type = _schema_type(schema, rng)
    freq_value = schema.get("x-polylogue-frequency")
    freq = float(freq_value) if isinstance(freq_value, (int, float)) else 1.0
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
    self: _SyntheticRuntimeContext,
    schema: SchemaRecord,
    rng: random.Random,
    *,
    skip_keys: set[str] | None = None,
    depth: int = 0,
    max_depth: int = 6,
    path: str = "$",
) -> SyntheticRecord:
    obj: SyntheticRecord = {}
    properties = _schema_record(schema.get("properties"))
    candidate_keys = set(properties.keys())
    if skip_keys:
        candidate_keys -= skip_keys

    if self._relation_solver.mutual_exclusions:
        candidate_keys = self._relation_solver.filter_mutually_exclusive(path, candidate_keys, rng)

    for prop_name, prop_value in properties.items():
        if prop_name not in candidate_keys:
            continue
        prop_schema = _schema_record(prop_value)
        if not prop_schema:
            continue

        freq_value = prop_schema.get("x-polylogue-frequency")
        freq = float(freq_value) if isinstance(freq_value, (int, float)) else 1.0
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


def _generate_string(self: _SyntheticRuntimeContext, schema: SchemaRecord, rng: random.Random) -> str:
    values = schema.get("x-polylogue-values")
    if isinstance(values, list) and values:
        return str(rng.choice(values))

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


def _generate_number(
    self: _SyntheticRuntimeContext,
    schema: SchemaRecord,
    rng: random.Random,
    *,
    is_int: bool = False,
) -> float | int:
    range_value = schema.get("x-polylogue-range")
    if isinstance(range_value, list) and len(range_value) >= 2:
        lo = _coerce_float(range_value[0])
        hi = _coerce_float(range_value[1])
        value = rng.uniform(lo, hi)
    elif schema.get("x-polylogue-format") == "unix-epoch":
        value = rng.uniform(1670000000, 1760000000)
    else:
        value = rng.uniform(0, 1000)
    return int(value) if is_int else value


def _generate_array(
    self: _SyntheticRuntimeContext,
    schema: SchemaRecord,
    rng: random.Random,
    *,
    depth: int = 0,
    max_depth: int = 6,
    path: str = "$",
) -> list[JSONValue]:
    item_schema = _schema_record(schema.get("items"))
    lengths = schema.get("x-polylogue-array-lengths")
    if isinstance(lengths, list) and len(lengths) >= 2:
        lo = _coerce_int(lengths[0])
        hi = _coerce_int(lengths[1])
        clamped_lo = min(max(0, lo), 5)
        clamped_hi = min(max(hi, clamped_lo), 5)
        n_items = rng.randint(clamped_lo, clamped_hi)
    else:
        n_items = rng.randint(1, 3)

    item_type = item_schema.get("type")
    item_allows_null = item_type == "null" or (
        isinstance(item_type, list) and any(item == "null" for item in item_type if isinstance(item, str))
    )
    if not item_allows_null:
        item_allows_null = any(variant.get("type") == "null" for variant in _schema_records(item_schema.get("anyOf")))
    if not item_allows_null:
        item_allows_null = any(variant.get("type") == "null" for variant in _schema_records(item_schema.get("oneOf")))

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


def _serialize(self: _SyntheticRuntimeContext, data: JSONValue) -> bytes:
    if self.wire_format.encoding == "jsonl":
        if not isinstance(data, list):
            raise ValueError("JSONL wire format requires a list payload")
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

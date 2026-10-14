"""Record-shaping helpers for synthetic corpora."""

from __future__ import annotations

import random
import uuid
from collections.abc import Iterator, Mapping, Sequence
from typing import Protocol, TypeAlias

from polylogue.schemas.synthetic.array_value import GeneratedArray
from polylogue.schemas.synthetic.build_wire_formats import _wire_field, _wire_record, _wire_schema
from polylogue.schemas.synthetic.classification import _schema_types
from polylogue.schemas.synthetic.demo_themes import SessionTheme
from polylogue.schemas.synthetic.models import GeneratedRecord, SchemaRecord, SchemaValue, SyntheticGenerationState
from polylogue.schemas.synthetic.models import GeneratedValue as JSONValue
from polylogue.schemas.synthetic.record_profiles import PublicRecordPlan
from polylogue.schemas.synthetic.semantic_values import SemanticValueGenerator
from polylogue.schemas.synthetic.wire_formats import WireFormat

SyntheticRecord: TypeAlias = dict[str, JSONValue] | GeneratedRecord


class _WireFormatContext(Protocol):
    provider: str
    schema: SchemaRecord
    workload_profile: Mapping[str, object] | None
    wire_format: WireFormat
    _semantic_gen: SemanticValueGenerator | None
    _public_record_plan: PublicRecordPlan | None
    _coverage_witness_mode: bool
    _session_native_id: str | None
    _generation_state: SyntheticGenerationState

    def _generate_from_schema(
        self,
        schema: SchemaRecord,
        rng: random.Random,
        *,
        skip_keys: set[str] | None = ...,
        path: str = ...,
    ) -> JSONValue: ...

    def _ensure_wire_format(
        self,
        data: SyntheticRecord,
        role: str,
        rng: random.Random,
        index: int,
        base_ts: float,
        theme: SessionTheme | None = ...,
    ) -> None: ...

    def _role_cycle(self) -> list[str]: ...

    def _start_public_record(self, rng: random.Random, schema: SchemaRecord) -> None: ...

    def _finish_public_record(self, record: SyntheticRecord) -> None: ...


def _coerce_record(value: JSONValue) -> SyntheticRecord:
    if not isinstance(value, dict | GeneratedRecord):
        raise ValueError("selected record carrier must be an object")
    return value


def _coerce_schema(value: SchemaValue | object) -> SchemaRecord:
    return value if isinstance(value, dict) else {}


def _wire_link(self: _WireFormatContext, record: SyntheticRecord, field: str, value: JSONValue) -> None:
    """The graph owns links, conditioned on declared types and vocabulary."""
    schema = _wire_schema(self, record, (field,))
    if field in record and record[field] is None and value is not None:
        raise ValueError("selected null conflicts with a required graph link")
    pending: list[Iterator[tuple[SchemaRecord, JSONValue]]] = [iter(((schema, value),))]
    while pending:
        try:
            current, chosen = next(pending[-1])
        except StopIteration:
            pending.pop()
            continue
        kinds = _schema_types(current)
        kind = (
            "null"
            if chosen is None
            else "boolean"
            if isinstance(chosen, bool)
            else "string"
            if isinstance(chosen, str)
            else "array"
            if isinstance(chosen, list | GeneratedArray)
            else "object"
            if isinstance(chosen, dict | GeneratedRecord)
            else "integer"
            if isinstance(chosen, int)
            else "number"
        )
        if kinds and kind not in kinds and not (kind == "integer" and "number" in kinds):
            raise ValueError("graph link conflicts with the declared wire type")
        labels = current.get("x-polylogue-values")
        if isinstance(labels, list) and labels and chosen not in labels:
            raise ValueError("graph link conflicts with the declared vocabulary")
        if isinstance(chosen, list | GeneratedArray):
            items = _coerce_schema(current.get("items"))
            pending.append((items, item) for item in chosen)
    record[field] = value


def _child_id_list(record: SyntheticRecord, field_name: str) -> list[JSONValue]:
    existing = record.get(field_name)
    if isinstance(existing, list) and all(isinstance(item, str) for item in existing):
        return existing
    raise ValueError("selected graph children must be a string list")


def _reset_semantic_generator(
    self: _WireFormatContext,
    *,
    rng: random.Random,
    theme: SessionTheme | None,
    base_ts: float,
    roles: Sequence[str],
) -> None:
    self._semantic_gen = SemanticValueGenerator(rng, theme=theme, base_ts=base_ts, role_cycle=list(roles))


def _declared_numeric(schema: SchemaValue | object) -> str | None:
    """``"number"`` or ``"integer"`` when the schema admits that type, else None.

    A schema admitting any non-integral number reports ``"number"``; one
    admitting integers only reports ``"integer"``.
    """
    node = _coerce_schema(schema)
    declared: list[object] = []
    raw_type = node.get("type")
    declared.extend(raw_type if isinstance(raw_type, list) else [raw_type])
    for key in ("anyOf", "oneOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            for variant in variants:
                variant_type = _coerce_schema(variant).get("type")
                declared.extend(variant_type if isinstance(variant_type, list) else [variant_type])
    kinds = {kind for kind in declared if isinstance(kind, str)}
    if "number" in kinds:
        return "number"
    return "integer" if "integer" in kinds else None


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
    self: _WireFormatContext, n_messages: int, rng: random.Random, *, theme: SessionTheme | None = None
) -> SyntheticRecord:
    tree_cfg = self.wire_format.tree
    assert tree_cfg is not None and tree_cfg.container_path is not None

    roles = self._role_cycle()
    base_ts = rng.uniform(1670000000, 1760000000)
    _reset_semantic_generator(self, rng=rng, theme=theme, base_ts=base_ts, roles=roles)

    properties = _coerce_schema(self.schema.get("properties"))
    original_identity = self._session_native_id
    identity_field = self.wire_format.document_session_field
    identity_fields = (
        {identity_field}
        if original_identity is not None and identity_field is not None and identity_field in properties
        else set()
    )
    if original_identity is not None and not identity_fields:
        raise ValueError("selected original document declares no native session identity")
    top = self._generate_from_schema(
        self.schema, rng, skip_keys={tree_cfg.container_path, "current_node", *identity_fields}
    )
    top_record = _coerce_record(top)
    for field in identity_fields:
        _wire_link(self, top_record, field, original_identity)

    container_schema = _coerce_schema(properties.get(tree_cfg.container_path))
    node_schema = _coerce_schema(container_schema.get("additionalProperties"))
    owner = self._generation_state.relation_solver.fk_graph
    nodes = owner.new_array()
    graph_fields = {tree_cfg.parent_field}
    if tree_cfg.children_field:
        graph_fields.add(tree_cfg.children_field)

    for i in range(n_messages):
        from polylogue.schemas.synthetic.authored_selection import authored_record_schema

        original_node_schema = authored_record_schema(self.workload_profile, node_schema, i)
        self._start_public_record(rng, original_node_schema)
        node_path = "$.properties." + tree_cfg.container_path + ".additionalProperties.*"
        node = _coerce_record(
            self._generate_from_schema(
                original_node_schema,
                rng,
                skip_keys=graph_fields,
                path=node_path if self._coverage_witness_mode else "$",
            )
        )

        node_id = _wire_field(self, node, tree_cfg.key_field, rng, str(uuid.UUID(int=rng.getrandbits(128), version=4)))
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("tree node identity must be a unique nonempty string")
        owner.claim_record_id(nodes, node_id)

        if i == 0:
            _wire_link(self, node, tree_cfg.parent_field, None)
            if tree_cfg.children_field:
                _wire_link(self, node, tree_cfg.children_field, [])
        else:
            parent_idx = rng.randint(max(0, i - 3), i - 1)
            parent = _coerce_record(nodes[parent_idx])
            _wire_link(self, node, tree_cfg.parent_field, parent[tree_cfg.key_field])
            if tree_cfg.children_field:
                _wire_link(
                    self, parent, tree_cfg.children_field, [*_child_id_list(parent, tree_cfg.children_field), node_id]
                )
                _wire_link(self, node, tree_cfg.children_field, [])

        role = roles[i % len(roles)]
        self._ensure_wire_format(node, role, rng, i, base_ts=base_ts, theme=theme)
        _advance_semantic_turn(self)
        nodes.append(node)

    self._public_record_plan = None
    children_by_id: SyntheticRecord = owner.new_record({})
    for raw_node in nodes:
        node = _coerce_record(raw_node)
        node_id_value = node.get(tree_cfg.key_field)
        if isinstance(node_id_value, str):
            children_by_id[node_id_value] = node
    top_record[tree_cfg.container_path] = children_by_id
    if "current_node" in properties:
        _wire_link(self, top_record, "current_node", _coerce_record(nodes[-1])[tree_cfg.key_field])
    if self.provider == "chatgpt":
        _wire_field(self, top_record, "id", rng, str(uuid.UUID(int=rng.getrandbits(128), version=4)))
        # Conversation timestamps are epoch floats, but only where the selected
        # element declares a numeric field; other ChatGPT elements (the export
        # asset index) type these fields differently, and a default must not
        # contradict the element's own schema.
        defaults = {"create_time": base_ts, "update_time": base_ts + max(0, n_messages - 1) * 60}
        for field_name, value in defaults.items():
            numeric = None if field_name in top_record else _declared_numeric(properties.get(field_name))
            if numeric is not None:
                _wire_field(
                    self,
                    top_record,
                    field_name,
                    rng,
                    int(value) if numeric == "integer" else value,
                    semantic_role="message_timestamp",
                )
    return top_record


def _generate_linear_json(
    self: _WireFormatContext, n_messages: int, rng: random.Random, *, theme: SessionTheme | None = None
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
        identity_field = self.wire_format.document_session_field
        identity_fields = (
            {identity_field} if self._session_native_id is not None and identity_field is not None else set()
        )
        properties = _coerce_schema(self.schema.get("properties"))
        if identity_fields and identity_field not in properties:
            raise ValueError("selected original document declares no native session identity")
        top = self._generate_from_schema(self.schema, rng, skip_keys={msgs_parts[0], *identity_fields})
    else:
        item_schema = self.schema
        top = {}

    top_record = _coerce_record(top)
    if self._session_native_id is not None:
        identity_field = self.wire_format.document_session_field
        if not schema_has_path or identity_field is None:
            raise ValueError("selected original document declares no native session identity")
        _wire_link(self, top_record, identity_field, self._session_native_id)

    messages = self._generation_state.relation_solver.fk_graph.new_array()
    for i in range(n_messages):
        from polylogue.schemas.synthetic.authored_selection import authored_record_schema

        original_item_schema = authored_record_schema(self.workload_profile, item_schema, i)
        self._start_public_record(rng, original_item_schema)
        item_path = (
            "$.properties." + ".properties.".join(msgs_parts) + ".items[*]" if self._coverage_witness_mode else "$"
        )
        msg = _coerce_record(
            self._generate_from_schema(
                original_item_schema,
                rng,
                path=item_path,
            )
        )
        role = roles[i % len(roles)]
        self._ensure_wire_format(msg, role, rng, i, base_ts=base_ts, theme=theme)
        _advance_semantic_turn(self)
        messages.append(msg)

    self._public_record_plan = None
    target: SyntheticRecord = top_record
    for index, part in enumerate(msgs_parts):
        if index == len(msgs_parts) - 1:
            target[part] = messages
        else:
            target = _wire_record(self, top_record, tuple(msgs_parts[: index + 1]), rng)

    return top_record


def _generate_jsonl_records(
    self: _WireFormatContext, n_messages: int, rng: random.Random, *, theme: SessionTheme | None = None
) -> GeneratedArray:
    tree_cfg = self.wire_format.tree
    owner = self._generation_state.relation_solver.fk_graph
    records = owner.new_array()
    graph_fields = {tree_cfg.parent_field} if tree_cfg else set()
    if tree_cfg and tree_cfg.session_field:
        graph_fields.add(tree_cfg.session_field)
    roles = self._role_cycle()
    session_id = self._session_native_id or str(uuid.UUID(int=rng.getrandbits(128), version=4))
    base_ts = rng.uniform(1670000000, 1760000000)
    _reset_semantic_generator(self, rng=rng, theme=theme, base_ts=base_ts, roles=roles)

    for i in range(n_messages):
        from polylogue.schemas.synthetic.authored_selection import authored_record_schema

        record_schema = authored_record_schema(self.workload_profile, self.schema, i)
        self._start_public_record(rng, record_schema)
        record = _coerce_record(self._generate_from_schema(record_schema, rng, skip_keys=graph_fields))

        role = roles[i % len(roles)]
        self._ensure_wire_format(record, role, rng, i, base_ts=base_ts, theme=theme)
        _advance_semantic_turn(self)

        if tree_cfg:
            node_id = _wire_field(
                self, record, tree_cfg.key_field, rng, str(uuid.UUID(int=rng.getrandbits(128), version=4))
            )
            if not isinstance(node_id, str) or not node_id:
                raise ValueError("record node identity must be a unique nonempty string")
            owner.claim_record_id(records, node_id)
            if i == 0:
                _wire_link(self, record, tree_cfg.parent_field, None)
            else:
                _wire_link(self, record, tree_cfg.parent_field, _coerce_record(records[i - 1]).get(tree_cfg.key_field))
            if tree_cfg.session_field:
                _wire_link(self, record, tree_cfg.session_field, session_id)

        self._finish_public_record(record)
        records.append(record)

    self._public_record_plan = None
    return records


__all__ = [
    "_generate_jsonl_records",
    "_generate_linear_json",
    "_generate_tree_json",
]

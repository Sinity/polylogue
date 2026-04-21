"""Hypothesis strategies for schema inference and validator laws."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from hypothesis import strategies as st

from polylogue.lib.json import JSONDocument
from tests.infra.strategies.sources import json_document_strategy

JSONRecord: TypeAlias = dict[str, object]
_STATIC_KEY_ALPHABET = "abcdefghijklmnopqrstuvwxyz_"
_RECORD_VARIANTS: tuple[tuple[str, str], ...] = (
    ("type", "session_meta"),
    ("type", "response_item"),
    ("record_type", "state"),
    ("payload.type", "message"),
)


@dataclass(frozen=True)
class SessionJsonlFileSpec:
    """One generated JSONL session file and the dict records it should contribute."""

    relative_path: str
    text: str
    expected_documents: tuple[JSONDocument, ...]


@st.composite
def dynamic_key_strategy(draw: st.DrawFn) -> str:
    """Generate keys that should be treated as dynamic identifiers."""
    flavor = draw(st.sampled_from(("uuid", "hex", "prefixed", "digits")))
    if flavor == "uuid":
        return str(draw(st.uuids()))
    if flavor == "hex":
        return draw(
            st.text(
                alphabet="0123456789abcdef",
                min_size=24,
                max_size=40,
            )
        )
    if flavor == "prefixed":
        prefix = draw(st.sampled_from(("msg", "node", "conv", "item", "att")))
        suffix = draw(st.text(alphabet="0123456789abcdef-", min_size=6, max_size=12))
        return f"{prefix}-{suffix}"
    return draw(
        st.text(
            alphabet="0123456789abcdef",
            min_size=24,
            max_size=24,
        )
    )


@st.composite
def static_key_strategy(draw: st.DrawFn) -> str:
    """Generate ordinary field names that should not collapse as dynamic identifiers."""
    return draw(
        st.text(
            alphabet=_STATIC_KEY_ALPHABET,
            min_size=2,
            max_size=12,
        ).filter(lambda value: value not in {"id", "msg", "node", "conv"})
    )


@st.composite
def nested_required_schema_strategy(draw: st.DrawFn, *, max_depth: int = 3) -> JSONRecord:
    """Generate a JSON-schema-like tree with required fields at every object node."""

    def _node(depth: int) -> JSONRecord:
        if depth <= 0:
            return {"type": draw(st.sampled_from(("string", "integer", "boolean")))}

        kind = draw(st.sampled_from(("object", "array", "anyOf", "oneOf", "allOf", "scalar")))
        if kind == "scalar":
            return {"type": draw(st.sampled_from(("string", "integer", "boolean")))}
        if kind == "array":
            return {"type": "array", "items": _node(depth - 1)}
        if kind in {"anyOf", "oneOf", "allOf"}:
            return {kind: [_node(depth - 1) for _ in range(draw(st.integers(min_value=1, max_value=3)))]}

        property_count = draw(st.integers(min_value=1, max_value=3))
        keys = [f"field_{index}" for index in range(property_count)]
        properties = {key: _node(depth - 1) for key in keys}
        return {
            "type": "object",
            "required": keys,
            "properties": properties,
        }

    root = _node(max_depth)
    if root.get("type") != "object":
        root = {
            "type": "object",
            "required": ["root"],
            "properties": {"root": root},
        }
    return root


@st.composite
def record_payload_strategy(
    draw: st.DrawFn,
    *,
    min_variants: int = 1,
    max_variants: int = 4,
    min_records_per_variant: int = 1,
    max_records_per_variant: int = 6,
) -> list[JSONRecord]:
    """Generate heterogeneous record payloads for record-granularity validation laws."""
    variants = draw(
        st.lists(
            st.sampled_from(_RECORD_VARIANTS),
            min_size=min_variants,
            max_size=max_variants,
            unique=True,
        )
    )
    payload: list[JSONRecord] = []
    for key, value in variants:
        count = draw(st.integers(min_value=min_records_per_variant, max_value=max_records_per_variant))
        for index in range(count):
            record: JSONRecord = (
                {"payload": {"type": value}, "idx": index} if key == "payload.type" else {key: value, "idx": index}
            )
            payload.append(record)
    return payload


@st.composite
def session_jsonl_tree_strategy(
    draw: st.DrawFn,
    *,
    min_files: int = 1,
    max_files: int = 4,
    max_documents_per_file: int = 4,
) -> tuple[SessionJsonlFileSpec, ...]:
    """Generate a recursive JSONL session tree with valid, blank, and malformed lines."""
    file_count = draw(st.integers(min_value=min_files, max_value=max_files))
    files: list[SessionJsonlFileSpec] = []

    for index in range(file_count):
        documents = tuple(draw(st.lists(json_document_strategy(), min_size=1, max_size=max_documents_per_file)))
        gap_counts = draw(
            st.lists(
                st.integers(min_value=0, max_value=2),
                min_size=len(documents),
                max_size=len(documents),
            )
        )
        nested = draw(st.booleans())
        prefix_invalid = draw(st.booleans())
        suffix_invalid = draw(st.booleans())
        prefix_blank_lines = draw(st.integers(min_value=0, max_value=2))
        suffix_blank_lines = draw(st.integers(min_value=0, max_value=2))

        lines: list[str] = [""] * prefix_blank_lines
        if prefix_invalid:
            lines.append("not json")
        for document, blank_count in zip(documents, gap_counts, strict=True):
            lines.append(json.dumps(document))
            if draw(st.booleans()):
                lines.append("{broken")
            lines.extend([""] * blank_count)
        if suffix_invalid:
            lines.append("[invalid")
        lines.extend([""] * suffix_blank_lines)

        relative_path = f"subdir_{index}/session_{index}.jsonl" if nested else f"session_{index}.jsonl"
        files.append(
            SessionJsonlFileSpec(
                relative_path=relative_path,
                text="\n".join(lines) + "\n",
                expected_documents=documents,
            )
        )

    return tuple(files)


def record_variant_signature(record: Mapping[str, object]) -> str:
    """Return the same stratification signature family used by record sampling."""
    for key in ("type", "record_type"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return f"{key}:{value}"
    payload = record.get("payload")
    if isinstance(payload, dict):
        value = payload.get("type")
        if isinstance(value, str) and value:
            return f"payload.type:{value}"
    return "unknown"


def expected_session_documents(files: tuple[SessionJsonlFileSpec, ...]) -> list[JSONDocument]:
    """Return session-loader output order when file mtimes follow tuple order."""
    documents: list[JSONDocument] = []
    for file_spec in reversed(files):
        documents.extend(file_spec.expected_documents)
    return documents

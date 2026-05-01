"""Shared JSON assertion contracts for CLI, MCP, and devtools tests."""

from __future__ import annotations

from collections.abc import Sequence

from polylogue.core.json import JSONDocument, JSONValue, loads, require_json_document, require_json_value

JSONArray = list[JSONValue]


def json_object(value: object, *, context: str = "JSON object") -> JSONDocument:
    """Return a JSON object or fail with test-oriented context."""
    try:
        return require_json_document(value, context=context)
    except TypeError as exc:
        raise AssertionError(str(exc)) from exc


def json_array(value: object, *, context: str = "JSON array") -> JSONArray:
    """Return a JSON array whose members satisfy the JSONValue contract."""
    if not isinstance(value, list):
        raise AssertionError(f"{context} is not a JSON array")
    return [require_json_value(item, context=f"{context}[{index}]") for index, item in enumerate(value)]


def json_object_list(value: object, *, context: str = "JSON object list") -> list[JSONDocument]:
    """Return a JSON array narrowed to object entries."""
    return [
        json_object(item, context=f"{context}[{index}]")
        for index, item in enumerate(json_array(value, context=context))
    ]


def json_bool(value: object, *, context: str = "JSON boolean") -> bool:
    if not isinstance(value, bool):
        raise AssertionError(f"{context} is not a JSON boolean")
    return value


def json_int(value: object, *, context: str = "JSON integer") -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise AssertionError(f"{context} is not a JSON integer")
    return value


def json_number(value: object, *, context: str = "JSON number") -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AssertionError(f"{context} is not a JSON number")
    return float(value)


def json_object_field(payload: JSONDocument, key: str, *, context: str = "JSON object") -> JSONDocument:
    return json_object(payload.get(key), context=f"{context}.{key}")


def json_array_field(payload: JSONDocument, key: str, *, context: str = "JSON object") -> JSONArray:
    return json_array(payload.get(key), context=f"{context}.{key}")


def json_array_item(items: Sequence[JSONValue], index: int, *, context: str = "JSON array") -> JSONDocument:
    return json_object(items[index], context=f"{context}[{index}]")


def parse_json_object(text: str, *, context: str = "JSON text") -> JSONDocument:
    return json_object(loads(text), context=context)


def extract_json_object(output: str, *, context: str = "CLI output") -> JSONDocument:
    """Extract the first JSON object from output that may include banners/logs."""
    lines = output.strip().splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("{"):
            return parse_json_object("\n".join(lines[index:]), context=context)
    raise AssertionError(f"No JSON object found in {context}:\n{output}")


def envelope_result(payload: JSONDocument, *, context: str = "JSON envelope") -> JSONDocument:
    if "result" not in payload:
        raise AssertionError(f"{context} is missing result")
    return json_object(payload["result"], context=f"{context}.result")


def unwrap_success_result(payload: JSONDocument, *, context: str = "JSON envelope") -> JSONDocument:
    if payload.get("status") == "ok" and "result" in payload:
        return envelope_result(payload, context=context)
    return payload


def extract_json_result(output: str, *, context: str = "CLI output") -> JSONDocument:
    return unwrap_success_result(extract_json_object(output, context=context), context=context)


__all__ = [
    "JSONArray",
    "envelope_result",
    "extract_json_object",
    "extract_json_result",
    "json_array",
    "json_array_field",
    "json_array_item",
    "json_bool",
    "json_int",
    "json_number",
    "json_object",
    "json_object_field",
    "json_object_list",
    "parse_json_object",
    "unwrap_success_result",
]

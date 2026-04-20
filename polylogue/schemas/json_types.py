"""Shared JSON-shaped aliases for schema and tooling payloads."""

from __future__ import annotations

from typing import TypeAlias, cast

JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
JSONDocument: TypeAlias = dict[str, JSONValue]
JSONDocumentList: TypeAlias = list[JSONDocument]


def json_document(value: object) -> JSONDocument:
    return cast(JSONDocument, value) if isinstance(value, dict) else {}


def json_document_list(value: object) -> JSONDocumentList:
    if not isinstance(value, list):
        return []
    return [document for item in value if (document := json_document(item))]


__all__ = [
    "JSONDocument",
    "JSONDocumentList",
    "JSONScalar",
    "JSONValue",
    "json_document",
    "json_document_list",
]

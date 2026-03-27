"""Normalization helpers for typed query specs."""

from __future__ import annotations

from datetime import datetime

from polylogue.lib.dates import parse_date
from polylogue.lib.viewports import ToolCategory

from .query_spec_errors import QuerySpecError

QUERY_ACTION_TYPES = tuple(category.value for category in ToolCategory) + ("none",)
QUERY_SEQUENCE_ACTION_TYPES = tuple(category.value for category in ToolCategory)
QUERY_RETRIEVAL_LANES = ("auto", "dialogue", "actions", "hybrid")


def split_csv(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def parse_query_date(field: str, value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = parse_date(value)
    if parsed is None:
        raise QuerySpecError(field, value)
    return parsed


def normalize_tool_terms(value: object) -> tuple[str, ...]:
    normalized: list[str] = []
    for term in as_tuple(value):
        candidate = str(term).strip().lower()
        if candidate:
            normalized.append(candidate)
    return tuple(normalized)


def normalize_action_terms(field: str, value: object) -> tuple[str, ...]:
    normalized: list[str] = []
    for term in as_tuple(value):
        candidate = str(term).strip().lower()
        if candidate not in QUERY_ACTION_TYPES:
            raise QuerySpecError(field, term)
        normalized.append(candidate)
    return tuple(normalized)


def normalize_action_sequence(field: str, value: object) -> tuple[str, ...]:
    normalized: list[str] = []
    for term in split_csv(value):
        candidate = str(term).strip().lower()
        if candidate not in QUERY_SEQUENCE_ACTION_TYPES:
            raise QuerySpecError(field, term)
        normalized.append(candidate)
    return tuple(normalized)


__all__ = [
    "QUERY_ACTION_TYPES",
    "QUERY_RETRIEVAL_LANES",
    "QUERY_SEQUENCE_ACTION_TYPES",
    "as_tuple",
    "normalize_action_sequence",
    "normalize_action_terms",
    "normalize_tool_terms",
    "parse_query_date",
    "split_csv",
]

"""Canonical shared utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from polylogue.core.json import json_document

# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------


def chunked(items: Sequence[str], *, size: int) -> Iterable[Sequence[str]]:
    """Yield successive chunks from a sequence of items."""
    for index in range(0, len(items), size):
        yield items[index : index + size]


def json_object(value: object) -> dict[str, object]:
    """Convert a JSON-compatible value to a plain dict of str->object.

    Used by publication and run record mappers to convert manifest/plan
    documents into dict form without retaining orjson-specific types.
    """
    document = json_document(value)
    result: dict[str, object] = {}
    for key, item in document.items():
        result[key] = item
    return result


def format_malformed_jsonl_error(*, malformed_lines: int, malformed_detail: str | None) -> str:
    """Format a human-readable error for malformed JSONL input."""
    message = f"Malformed JSONL lines: {malformed_lines}"
    if malformed_detail:
        return f"{message} (first bad {malformed_detail})"
    return message


__all__ = [
    "chunked",
    "format_malformed_jsonl_error",
    "json_object",
]

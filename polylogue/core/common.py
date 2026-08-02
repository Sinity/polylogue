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
    documents into dict form without retaining the active JSON backend's
    own specific types.
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


def peek_truthy(iterable: Iterable[object]) -> bool:
    """Answer truthiness from a bounded single-item peek, not full iteration.

    A lazy ``Sequence``/``Collection`` whose ``__len__`` is an O(n) replay
    (re-decoding a backing file, re-filtering a source, ...) still gets
    Python's default ``Sequence`` truthiness protocol unless it defines its
    own ``__bool__``: ``bool(x)``/``if x:`` falls back to ``len(x) > 0``,
    forcing that full rescan just to answer "is there at least one item?".
    A bounded peek at the first item answers the same question without
    consuming (or replaying) the rest of the source. Same defect class as
    PR #3546 (`ReplayableRecordSamples.__bool__`).
    """
    sentinel = object()
    return next(iter(iterable), sentinel) is not sentinel


def forward_bounded_slice(index: slice) -> tuple[int, int | None, int] | None:
    """Resolve a slice's ``(start, stop, step)`` without needing full length.

    ``slice.indices(length)`` is the normal way to resolve a slice, but it
    requires ``length`` up front -- for a lazy replay-backed sequence that
    means forcing a full rescan just to slice a small bounded prefix (e.g.
    ``samples[:5]``). When ``start``/``stop`` are already non-negative (or
    absent) and ``step`` is positive, the slice is answerable directly
    against an iterator via ``itertools.islice`` with no length lookup at
    all. Returns ``None`` when the slice genuinely needs the sequence's true
    length (a negative bound, or a non-positive step) so the caller can fall
    back to its normal ``len()``-based resolution.
    """
    step = index.step if index.step is not None else 1
    if step <= 0:
        return None
    if index.start is not None and index.start < 0:
        return None
    if index.stop is not None and index.stop < 0:
        return None
    start = index.start if index.start is not None else 0
    return start, index.stop, step


__all__ = [
    "chunked",
    "forward_bounded_slice",
    "format_malformed_jsonl_error",
    "json_object",
    "peek_truthy",
]

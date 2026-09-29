"""Synthetic original-byte fixtures and bounded-search drain assertions."""

from __future__ import annotations

from pathlib import Path

from polylogue.operations.raw_sessions.sessions import SessionSource
from polylogue.operations.session_contracts import RawMemorySearch, RawObservation, RawSearch
from polylogue.operations.session_reads import raw_operation


def write_raw(root: Path, payload: bytes) -> tuple[tuple[SessionSource, ...], Path]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "sample.jsonl"
    path.write_bytes(payload)
    return (SessionSource("codex", root),), path


def drain_raw_search(
    sources: tuple[SessionSource, ...],
    query: str,
    budgets: tuple[int, ...],
    limit: int,
    *,
    max_pages: int,
    memory: bool = False,
) -> list[RawObservation]:
    rows: list[RawObservation] = []
    continuation = None
    source_cursors = None
    seen = set()
    for page_number in range(max_pages):
        budget = budgets[page_number % len(budgets)]
        request: RawSearch | RawMemorySearch
        if memory:
            request = RawMemorySearch(
                origins=["codex-session"],
                query=query,
                scan_bytes=budget,
                limit=limit,
                source_cursors=source_cursors,
            )
        else:
            request = RawSearch(
                origin="codex-session",
                query=query,
                scan_bytes=budget,
                limit=limit,
                continuation=continuation,
            )
        page = raw_operation(request, sources=sources)
        assert page.coverage.gaps == []
        assert page.coverage.scanned_bytes <= budget
        rows.extend(page.items)
        source_cursors = page.source_cursors
        continuation = (source_cursors or {}).get("codex-session") if memory else page.continuation
        if continuation is None:
            assert page.coverage.complete
            return rows
        assert continuation not in seen
        seen.add(continuation)
    raise AssertionError("search did not terminate within its synthetic input's progress bound")

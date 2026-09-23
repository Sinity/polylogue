"""The session-list owner enters through the shared read request contract."""

from __future__ import annotations

import pytest

from polylogue.operations import session_reads
from polylogue.operations.session_contracts import SessionList
from polylogue.surfaces.read_contract import ReadRequest


@pytest.mark.asyncio
async def test_session_query_lowers_selection_through_read_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session query cannot bypass Selection × Projection × Render normalization.

    The transaction is stubbed so this is a route-shape test, not an archive
    fixture test.  Removing the ``ReadRequest.normalize`` call makes the
    assertion fail while keeping the rest of the operation executable.
    """

    calls: list[tuple[dict[str, object], str | None]] = []
    normalize = ReadRequest.normalize

    def recording_normalize(params: dict[str, object], *, preset: str | None = None) -> ReadRequest:
        calls.append((params, preset))
        return normalize(params, preset=preset)

    monkeypatch.setattr(ReadRequest, "normalize", staticmethod(recording_normalize))

    class StubTransaction:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def run(self, _work: object) -> object:
            return "stub-page"

    monkeypatch.setattr(session_reads, "QueryTransaction", StubTransaction)

    result = await session_reads.session_query(
        "/tmp/archive",  # transaction is stubbed; no filesystem access occurs
        SessionList(repo="polylogue", limit=3),
    )

    assert result == "stub-page"
    assert len(calls) == 1
    params, preset = calls[0]
    assert preset == "summary"
    assert params["repo"] == "polylogue"
    assert params["limit"] == 3

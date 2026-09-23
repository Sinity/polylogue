"""Contract coverage for the bounded attachment-library facade seam."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from polylogue import Polylogue


class _Repository:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def get_attachment_library_page(self, **kwargs: object) -> list[tuple[object, str, str | None]]:
        self.calls.append(kwargs)
        return []

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"attachment facade must use the declared page read, not {name}")


@pytest.mark.asyncio
async def test_attachment_library_page_delegates_one_bounded_read() -> None:
    """The product facade carries page bounds and filters to the repository."""

    repository = _Repository()
    poly = cast(Polylogue, Polylogue.__new__(Polylogue))
    poly._services = SimpleNamespace(get_repository=lambda: repository)

    result = await poly._get_attachment_library_page(
        limit=21,
        offset=40,
        mime_filter="image/",
        session_filter="session-7",
        state_filter="available",
    )

    assert result == []
    assert repository.calls == [
        {
            "limit": 21,
            "offset": 40,
            "mime_filter": "image/",
            "session_filter": "session-7",
            "state_filter": "available",
        }
    ]

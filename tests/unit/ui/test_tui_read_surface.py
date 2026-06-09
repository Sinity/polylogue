"""Direct TUI read-surface adapter checks.

These tests pin the TUI read-surface adapter to the same typed
envelopes the web reader, CLI JSON, MCP, and Python API consume (ref
issues #848 and #859).  Broader cross-surface envelope parity lives in
``tests/unit/api/test_read_surface_contracts.py``; this module exercises
the TUI-specific construction shape and confirms the rendered fields
screens depend on are present on the payload rows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polylogue.api import Polylogue
from polylogue.api.contracts.read_surface import (
    SessionListSurface,
    SessionSearchSurface,
    SessionStatsSurface,
    SessionTagsSurface,
)
from polylogue.api.contracts.tui_surface import TUIReadSurface
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.surfaces.payloads import (
    SessionListResponse,
    SessionListRowPayload,
)
from tests.infra.storage_records import SessionBuilder, db_setup


def test_tui_surface_satisfies_read_protocols() -> None:
    """TUIReadSurface conforms to the shared read-surface protocol family."""
    for protocol in (
        SessionListSurface,
        SessionSearchSurface,
        SessionStatsSurface,
        SessionTagsSurface,
    ):
        assert issubclass(TUIReadSurface, protocol), f"TUIReadSurface does not implement {protocol.__name__}"


@pytest.mark.asyncio
async def test_tui_surface_list_returns_typed_envelope(workspace_env: dict[str, Path]) -> None:
    """TUIReadSurface.list_sessions returns SessionListResponse with typed rows."""
    db_path = db_setup(workspace_env)
    await SessionBuilder(db_path, "conv-1").provider("chatgpt").title("First").add_message("m1", text="hi").build()
    await SessionBuilder(db_path, "conv-2").provider("claude-ai").title("Second").add_message("m2", text="yo").build()

    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as facade:
        surface = TUIReadSurface(facade)
        envelope = await surface.list_sessions(SessionQuerySpec(limit=10))

        assert isinstance(envelope, SessionListResponse)
        assert envelope.total == 2
        assert len(envelope.items) == 2
        for row in envelope.items:
            assert isinstance(row, SessionListRowPayload)
            # Fields the TUI browser/search screens render from
            assert row.id
            assert row.origin
            assert row.title


@pytest.mark.asyncio
async def test_tui_surface_search_shares_envelope(workspace_env: dict[str, Path]) -> None:
    """search_sessions returns the same envelope shape as list."""
    db_path = db_setup(workspace_env)
    await SessionBuilder(db_path, "conv-1").add_message("m1", text="UniqueTokenABCXYZ").build()

    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as facade:
        surface = TUIReadSurface(facade)
        list_env = await surface.list_sessions(SessionQuerySpec(limit=10))
        search_env = await surface.search_sessions(SessionQuerySpec(limit=10))

        assert isinstance(search_env, SessionListResponse)
        assert type(search_env) is type(list_env)


@pytest.mark.asyncio
async def test_tui_surface_empty_archive_returns_empty_envelope(workspace_env: dict[str, Path]) -> None:
    """Empty archive yields total=0, items=(), no exception."""
    db_path = db_setup(workspace_env)

    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as facade:
        surface = TUIReadSurface(facade)
        envelope = await surface.list_sessions(SessionQuerySpec(limit=10))

        assert envelope.total == 0
        assert envelope.items == ()
        assert envelope.offset == 0

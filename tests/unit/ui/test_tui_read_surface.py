"""Direct TUI read-surface adapter checks.

These tests pin the TUI's read-surface adapter to the same typed
envelopes the web reader, CLI JSON, MCP, and Python API consume (ref
issues #848 and #859).  Broader cross-surface envelope parity lives in
``tests/unit/api/test_read_surface_contracts.py``; this module exercises
the TUI-specific construction shape (``ArchiveOperations`` injection)
and confirms the rendered fields screens depend on are present on the
payload rows.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAlias

import pytest

from polylogue.api.contracts.read_surface import (
    ConversationListSurface,
    ConversationSearchSurface,
    ConversationStatsSurface,
    ConversationTagsSurface,
)
from polylogue.api.contracts.tui_surface import TUIReadSurface
from polylogue.archive.query.spec import ConversationQuerySpec
from polylogue.operations import ArchiveOperations
from polylogue.surfaces.payloads import (
    ConversationListResponse,
    ConversationListRowPayload,
)

if TYPE_CHECKING:
    from polylogue.storage.repository import ConversationRepository
    from tests.infra.storage_records import ConversationBuilder


ConversationBuilderFactory: TypeAlias = Callable[[str], "ConversationBuilder"]


def _make_surface(repo: ConversationRepository) -> TUIReadSurface:
    operations = ArchiveOperations(repository=repo)
    return TUIReadSurface(operations)


def test_tui_surface_satisfies_read_protocols() -> None:
    """TUIReadSurface conforms to the shared read-surface protocol family."""
    for protocol in (
        ConversationListSurface,
        ConversationSearchSurface,
        ConversationStatsSurface,
        ConversationTagsSurface,
    ):
        assert issubclass(TUIReadSurface, protocol), f"TUIReadSurface does not implement {protocol.__name__}"


@pytest.mark.asyncio
async def test_tui_surface_list_returns_typed_envelope(
    storage_repository: ConversationRepository,
    conversation_builder: ConversationBuilderFactory,
) -> None:
    """TUIReadSurface.list_conversations returns ConversationListResponse with typed rows."""
    conversation_builder("conv-1").provider("chatgpt").title("First").add_message("m1", text="hi").save()
    conversation_builder("conv-2").provider("claude-ai").title("Second").add_message("m2", text="yo").save()

    surface = _make_surface(storage_repository)
    envelope = await surface.list_conversations(ConversationQuerySpec(limit=10))

    assert isinstance(envelope, ConversationListResponse)
    assert envelope.total == 2
    assert len(envelope.items) == 2
    for row in envelope.items:
        assert isinstance(row, ConversationListRowPayload)
        # Fields the TUI browser/search screens render from
        assert row.id
        assert row.provider
        assert row.title


@pytest.mark.asyncio
async def test_tui_surface_search_shares_envelope(
    storage_repository: ConversationRepository,
    conversation_builder: ConversationBuilderFactory,
) -> None:
    """search_conversations returns the same envelope shape as list."""
    conversation_builder("conv-1").add_message("m1", text="UniqueTokenABCXYZ").save()

    surface = _make_surface(storage_repository)
    list_env = await surface.list_conversations(ConversationQuerySpec(limit=10))
    search_env = await surface.search_conversations(ConversationQuerySpec(limit=10))

    assert isinstance(search_env, ConversationListResponse)
    assert type(search_env) is type(list_env)


@pytest.mark.asyncio
async def test_tui_surface_empty_archive_returns_empty_envelope(
    storage_repository: ConversationRepository,
) -> None:
    """Empty archive yields total=0, items=(), no exception."""
    surface = _make_surface(storage_repository)
    envelope = await surface.list_conversations(ConversationQuerySpec(limit=10))

    assert envelope.total == 0
    assert envelope.items == ()
    assert envelope.offset == 0

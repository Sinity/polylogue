"""Contract tests for the TUI read-surface adapter.

The Python API, CLI, and MCP do not go through dedicated read-surface
adapters — they call into operations / repository / services directly and
hand-roll their own envelopes (see ``daemon/http.py:_do_list`` and the
MCP server tools). The only adapter that has a real production consumer
is :class:`TUIReadSurface`, used by ``polylogue/ui/tui/screens/``.

These tests pin the conformance and runtime behavior of that adapter
against the canonical :class:`ConversationListResponse` envelope.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.api.contracts import (
    ConversationListSurface,
    ConversationSearchSurface,
    ConversationStatsSurface,
    ConversationTagsSurface,
    ReadSurface,
)
from polylogue.api.contracts.tui_surface import TUIReadSurface
from polylogue.archive.query.spec import ConversationQuerySpec
from polylogue.operations import ArchiveOperations
from polylogue.services import build_runtime_services
from polylogue.surfaces.payloads import ConversationListResponse
from polylogue.types import Provider
from tests.infra.storage_records import ConversationBuilder, db_setup

_PROTOCOL_FAMILY: tuple[type, ...] = (
    ConversationListSurface,
    ConversationSearchSurface,
    ConversationStatsSurface,
    ConversationTagsSurface,
)


def test_tui_adapter_conforms_to_protocol_family() -> None:
    """The TUI adapter structurally satisfies every read-surface protocol."""
    for protocol in _PROTOCOL_FAMILY:
        assert issubclass(TUIReadSurface, protocol), (
            f"TUIReadSurface does not implement {protocol.__name__}; "
            "the TUI relies on the canonical async read contract."
        )
    assert issubclass(TUIReadSurface, ReadSurface)


async def _seed_archive(db_path: Path) -> None:
    """Seed two conversations under different providers."""
    await (
        ConversationBuilder(db_path, "conv-alpha")
        .provider(Provider.CLAUDE_AI.value)
        .title("Alpha")
        .add_message(text="alpha message body")
        .build()
    )
    await (
        ConversationBuilder(db_path, "conv-beta")
        .provider(Provider.CHATGPT.value)
        .title("Beta")
        .add_message(text="beta message body")
        .build()
    )


async def test_tui_list_returns_canonical_envelope(workspace_env: dict[str, Path]) -> None:
    """TUI list returns a typed :class:`ConversationListResponse` envelope."""
    db_path = db_setup(workspace_env)
    await _seed_archive(db_path)

    services = build_runtime_services(db_path=db_path)
    try:
        surface = TUIReadSurface(ArchiveOperations.from_services(services))
        envelope = await surface.list_conversations(ConversationQuerySpec(limit=10))

        assert isinstance(envelope, ConversationListResponse)
        assert envelope.total == 2
        assert len(envelope.items) == 2
        ids = {row.id for row in envelope.items}
        assert ids == {"conv-alpha", "conv-beta"}
    finally:
        await services.close()


async def test_tui_provider_filter(workspace_env: dict[str, Path]) -> None:
    """Provider filtering narrows the TUI list envelope."""
    db_path = db_setup(workspace_env)
    await _seed_archive(db_path)

    services = build_runtime_services(db_path=db_path)
    try:
        surface = TUIReadSurface(ArchiveOperations.from_services(services))
        spec = ConversationQuerySpec(providers=(Provider.CLAUDE_AI,), limit=10)
        envelope = await surface.list_conversations(spec)

        ids = {row.id for row in envelope.items}
        assert ids == {"conv-alpha"}
    finally:
        await services.close()


async def test_tui_empty_archive_envelope(workspace_env: dict[str, Path]) -> None:
    """The TUI surface returns total=0/items=() on an empty archive."""
    db_path = db_setup(workspace_env)

    services = build_runtime_services(db_path=db_path)
    try:
        surface = TUIReadSurface(ArchiveOperations.from_services(services))
        envelope = await surface.list_conversations(ConversationQuerySpec(limit=10))

        assert envelope.total == 0
        assert envelope.items == ()
        assert envelope.offset == 0
    finally:
        await services.close()


async def test_tui_stats_envelope(workspace_env: dict[str, Path]) -> None:
    """TUI ``archive_stats()`` reports the expected counts."""
    db_path = db_setup(workspace_env)
    await _seed_archive(db_path)

    services = build_runtime_services(db_path=db_path)
    try:
        surface = TUIReadSurface(ArchiveOperations.from_services(services))
        stats = await surface.archive_stats()

        assert stats.conversation_count == 2
        assert stats.message_count >= 1
    finally:
        await services.close()

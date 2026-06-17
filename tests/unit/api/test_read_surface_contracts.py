"""Contract tests for the TUI read-surface adapter.

The Python API, CLI, and MCP do not go through dedicated read-surface
adapters — they call into operations / repository / services directly and
hand-roll their own envelopes (see ``daemon/http.py:_do_list`` and the
MCP server tools). The only adapter that has a real production consumer
is :class:`TUIReadSurface`, used by ``polylogue/ui/tui/screens/``.

These tests pin the conformance and runtime behavior of that adapter
against the canonical :class:`SessionListResponse` envelope.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.api import Polylogue
from polylogue.api.contracts import (
    ReadSurface,
    SessionListSurface,
    SessionSearchSurface,
    SessionStatsSurface,
    SessionTagsSurface,
)
from polylogue.api.contracts.tui_surface import TUIReadSurface
from polylogue.archive.query.spec import SessionQuerySpec
from polylogue.core.enums import Provider
from polylogue.surfaces.payloads import SessionListResponse
from tests.infra.archive_scenarios import native_session_id_for
from tests.infra.storage_records import SessionBuilder, db_setup

# Archive session ids the builder seeds under (``<origin>:ext-<conv_id>``).
_ALPHA_ID = native_session_id_for("claude-ai", "conv-alpha")
_BETA_ID = native_session_id_for("chatgpt", "conv-beta")

_PROTOCOL_FAMILY: tuple[type, ...] = (
    SessionListSurface,
    SessionSearchSurface,
    SessionStatsSurface,
    SessionTagsSurface,
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
    """Seed two sessions under different providers."""
    await (
        SessionBuilder(db_path, "conv-alpha")
        .provider(Provider.CLAUDE_AI.value)
        .title("Alpha")
        .add_message(text="alpha message body")
        .build()
    )
    await (
        SessionBuilder(db_path, "conv-beta")
        .provider(Provider.CHATGPT.value)
        .title("Beta")
        .add_message(text="beta message body")
        .build()
    )


async def test_tui_list_returns_canonical_envelope(workspace_env: dict[str, Path]) -> None:
    """TUI list returns a typed :class:`SessionListResponse` envelope."""
    db_path = db_setup(workspace_env)
    await _seed_archive(db_path)

    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as facade:
        surface = TUIReadSurface(facade)
        envelope = await surface.list_sessions(SessionQuerySpec(limit=10))

        assert isinstance(envelope, SessionListResponse)
        assert envelope.total == 2
        assert len(envelope.items) == 2
        ids = {row.id for row in envelope.items}
        assert ids == {_ALPHA_ID, _BETA_ID}


async def test_tui_origin_filter(workspace_env: dict[str, Path]) -> None:
    """Origin filtering narrows the TUI list envelope."""
    db_path = db_setup(workspace_env)
    await _seed_archive(db_path)

    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as facade:
        surface = TUIReadSurface(facade)
        spec = SessionQuerySpec(origins=("claude-ai-export",), limit=10)
        envelope = await surface.list_sessions(spec)

        ids = {row.id for row in envelope.items}
        assert ids == {_ALPHA_ID}


async def test_tui_empty_archive_envelope(workspace_env: dict[str, Path]) -> None:
    """The TUI surface returns total=0/items=() on an empty archive."""
    db_path = db_setup(workspace_env)

    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as facade:
        surface = TUIReadSurface(facade)
        envelope = await surface.list_sessions(SessionQuerySpec(limit=10))

        assert envelope.total == 0
        assert envelope.items == ()
        assert envelope.offset == 0


async def test_tui_stats_envelope(workspace_env: dict[str, Path]) -> None:
    """TUI ``archive_stats()`` reports the expected counts."""
    db_path = db_setup(workspace_env)
    await _seed_archive(db_path)

    async with Polylogue(archive_root=db_path.parent, db_path=db_path) as facade:
        surface = TUIReadSurface(facade)
        stats = await surface.archive_stats()

        assert stats.session_count == 2
        assert stats.message_count >= 1

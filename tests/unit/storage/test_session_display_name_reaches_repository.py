"""``display_name`` (the Claude Code ``slug`` wire field) reaches Session/SessionSummary.

polylogue-cgfy: the parser has captured ``slug`` (1,500 sampled occurrences)
into ``ParsedSession.display_name`` and the writer persists it into the
``sessions.display_name`` column since polylogue-2qx.4, but neither
``Session`` nor ``SessionSummary`` carried a ``display_name`` field at all --
the value was written durably and then dropped on every read, so a session
whose only title-worthy evidence was its slug (the common subagent case,
"<uuid-prefix>:agent-<suffix>" instead of a human name) still rendered as a
raw id/UUID everywhere. This test proves the real writer -> real async
repository -> domain-model ``display_title`` chain now surfaces it, not the
storage column in isolation.
"""

from __future__ import annotations

from pathlib import Path

from polylogue.core.enums import BlockType, Provider, Role
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.repository import SessionRepository
from polylogue.storage.sqlite.async_sqlite import SQLiteBackend
from tests.infra.live_ingest import ingest_session


async def test_display_name_reaches_session_display_title_when_title_absent(tmp_path: Path) -> None:
    """A session with no title-worthy evidence falls back to its slug, not a raw id."""
    backend = SQLiteBackend(db_path=tmp_path / "display-name.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="slug-only-session",
                title=None,
                display_name="greedy-squishing-hamming",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.ASSISTANT,
                        position=0,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hi")],
                    ),
                ],
            ),
            backend=backend,
        )
        session = await repo.get(session_id)
        assert session is not None
        summary = await repo.get_summary(session_id)
    finally:
        await repo.close()

    assert session.display_name == "greedy-squishing-hamming"
    assert session.display_title == "greedy-squishing-hamming"

    assert summary is not None
    assert summary.display_name == "greedy-squishing-hamming"
    assert summary.display_title == "greedy-squishing-hamming"


async def test_display_name_does_not_override_a_real_title(tmp_path: Path) -> None:
    """A real provider title still wins over the slug (title > display_name precedence)."""
    backend = SQLiteBackend(db_path=tmp_path / "display-name-title-wins.db")
    repo = SessionRepository(backend=backend)
    try:
        session_id = await ingest_session(
            ParsedSession(
                source_name=Provider.CLAUDE_CODE,
                provider_session_id="titled-session",
                title="Recover what was lost",
                display_name="greedy-squishing-hamming",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.ASSISTANT,
                        position=0,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text="hi")],
                    ),
                ],
            ),
            backend=backend,
        )
        session = await repo.get(session_id)
    finally:
        await repo.close()

    assert session is not None
    assert session.display_name == "greedy-squishing-hamming"
    assert session.display_title == "Recover what was lost"

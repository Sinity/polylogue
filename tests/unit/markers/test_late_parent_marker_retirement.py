"""A child ingested before its parent does not leave duplicate prefix markers.

The child is stored whole while its parent is absent, so its accepted marker
carrier seals candidates for the replayed prefix under the child's message
ids. When the parent arrives, re-extraction hands those blocks to the parent,
whose own carrier seals the same markers under the parent's ids.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import asdict
from pathlib import Path

from polylogue.core.enums import BlockType, Role
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage
from polylogue.storage.derived.session.marker_domain import SessionMarkerDerivation
from tests.infra.durable_tier_fixtures import initialize_runtime_source_fixture

_PREFIX_MARKER = "::note: shared prefix lesson"
_TAIL_MARKER = "::note: child tail lesson"


def _message(provider_id: str, role: Role, text: str, position: int) -> ParsedMessage:
    return ParsedMessage(
        provider_message_id=provider_id,
        role=role,
        text=text,
        position=position,
        variant_index=0,
        is_active_path=True,
        is_active_leaf=False,
        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
    )


def _marker_rows(user_db: Path) -> dict[tuple[str, str], str]:
    with sqlite3.connect(user_db) as user:
        rows = user.execute(
            "SELECT body_text, target_ref, status FROM assertions WHERE author_kind = 'agent' ORDER BY target_ref"
        ).fetchall()
    return {(str(body), str(target)): str(status) for body, target, status in rows}


def test_retirement_delivered_before_the_child_carrier_still_holds(tmp_path: Path) -> None:
    """Anti-vacuity: a retirement that only supersedes existing rows is lost when the parent's carrier lands first.

    Source finalization does not order a batch's raws, and an interrupted
    child can finalize after its parent. The later child carrier must not
    lower the retired candidate live.
    """
    import aiosqlite

    from polylogue.markers import candidates_for_block
    from polylogue.markers.lowering import assertion_id_for_marker
    from polylogue.storage.accepted_marker_inputs import append_accepted_marker_input, prepare_accepted_marker_input
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    initialize_runtime_source_fixture(tmp_path / "source.db")
    initialize_archive_database(tmp_path / "user.db", ArchiveTier.USER)
    child_candidate = candidates_for_block("codex-session:child:c1", "codex-session:child:c1:0", _PREFIX_MARKER)[0]
    child_record = asdict(child_candidate)
    child_record["assertion_kind"] = child_candidate.assertion_kind.value if child_candidate.assertion_kind else None
    retired_id = assertion_id_for_marker(child_candidate)
    assert retired_id is not None
    parent_carrier = prepare_accepted_marker_input(
        "parent-raw",
        [{"session_id": "codex-session:parent", "candidates": [], "retired_assertions": [retired_id]}],
    )
    child_carrier = prepare_accepted_marker_input(
        "child-raw", [{"session_id": "codex-session:child", "candidates": [child_record]}]
    )

    async def append() -> None:
        async with aiosqlite.connect(tmp_path / "source.db") as conn:
            await append_accepted_marker_input(conn, parent_carrier)
            await append_accepted_marker_input(conn, child_carrier)
            await conn.commit()

    asyncio.run(append())
    adapter = SessionMarkerDerivation(
        lambda: sqlite3.connect(f"file:{tmp_path / 'source.db'}?mode=ro", uri=True),
        lambda: sqlite3.connect(f"file:{tmp_path / 'user.db'}?mode=ro", uri=True),
        lambda: sqlite3.connect(tmp_path / "user.db"),
    )
    frame = object()
    for _ in range(2):
        keys, _next = adapter.required_page(frame, cursor=None, limit=1)
        assert adapter.publish(frame, adapter.compute(frame, keys[0])) is True

    with sqlite3.connect(tmp_path / "user.db") as user:
        live = user.execute(
            "SELECT COUNT(*) FROM assertions WHERE assertion_id = ? AND status = 'candidate'", (retired_id,)
        ).fetchone()
    assert live == (0,)

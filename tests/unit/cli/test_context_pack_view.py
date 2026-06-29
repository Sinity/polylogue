"""Tests for the collapsed ``context-pack`` capability.

The standalone context-pack assembler (``polylogue.context.pack``) and its
``ContextPack*`` DTO family were removed: the capability is now a thin lens over
the shared ``compile_context`` engine. ``Polylogue.context_pack_payload`` selects
sessions through the query algebra and returns the ``ContextImage`` payload that
the CLI ``read`` modifiers and the MCP ``build_context_pack`` tool both emit.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue import Polylogue
from polylogue.archive.message.roles import Role
from polylogue.context.compiler import ContextImage, ContextSpec
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.parsers.base import ParsedContentBlock, ParsedMessage, ParsedSession
from polylogue.storage.sqlite.archive_tiers.archive import ArchiveStore
from polylogue.storage.sqlite.archive_tiers.user_write import AssertionKind, upsert_assertion


def _seed(archive_root: Path, *, provider_session_id: str, text: str) -> None:
    with ArchiveStore(archive_root) as archive:
        archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=provider_session_id,
                title="Archive context pack",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:01:00+00:00",
                messages=[
                    ParsedMessage(
                        provider_message_id="m1",
                        role=Role.USER,
                        text=text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                    )
                ],
            )
        )


@pytest.mark.asyncio
async def test_context_pack_payload_compiles_context_image(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _seed(archive_root, provider_session_id="context-pack-v1", text="hello archive pack")

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        image = await poly.context_pack_payload(
            query="hello archive",
            project_path="/realm/project/polylogue",
            max_sessions=1,
        )

    assert isinstance(image, ContextImage)
    # Selection runs through the query algebra and seeds compile_context; the
    # message transcript is the compiled segment.
    message_segments = [segment for segment in image.segments if segment.payload_kind == "messages"]
    assert message_segments, "expected a messages segment for the selected session"
    assert "hello archive pack" in (message_segments[0].markdown or "")
    assert image.spec.read_views == ("messages",)


@pytest.mark.asyncio
async def test_max_tokens_bounds_output_with_omission_accounting(tmp_path: Path) -> None:
    """A tiny --max-tokens budget drops over-budget segments as explicit omissions."""
    archive_root = tmp_path / "archive"
    _seed(archive_root, provider_session_id="budget-a", text="alpha budget body that has several words")
    _seed(archive_root, provider_session_id="budget-b", text="beta budget body that also has several words")

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        unbounded = await poly.compile_context(
            ContextSpec(
                seed_refs=("session:codex-session:budget-a", "session:codex-session:budget-b"),
                read_views=("messages",),
            )
        )
        bounded = await poly.compile_context(
            ContextSpec(
                seed_refs=("session:codex-session:budget-a", "session:codex-session:budget-b"),
                read_views=("messages",),
                max_tokens=1,
            )
        )

    assert len(unbounded.segments) == 2
    # The budget bounds accumulation: fewer segments survive, and every drop is
    # reported as a budget omission rather than silently truncated.
    assert len(bounded.segments) < len(unbounded.segments)
    assert bounded.token_estimate <= unbounded.token_estimate
    budget_omissions = [omission for omission in bounded.omitted if omission.reason == "budget"]
    assert budget_omissions
    assert len(bounded.segments) + len(budget_omissions) == len(unbounded.segments)


@pytest.mark.asyncio
async def test_context_pack_payload_includes_injectable_assertions_via_recovery(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _seed(archive_root, provider_session_id="context-pack-assertions", text="assertion context pack")
    with ArchiveStore(archive_root) as archive:
        with sqlite3.connect(archive.user_db_path) as conn:
            upsert_assertion(
                conn,
                assertion_id="inject-decision",
                target_ref="session:codex-session:context-pack-assertions",
                scope_ref="repo:polylogue",
                kind=AssertionKind.DECISION,
                body_text="Use the shared assertion facade in context surfaces.",
                status="active",
                context_policy={"inject": True},
                now_ms=1_700_000_000_000,
            )
            upsert_assertion(
                conn,
                assertion_id="private-caveat",
                target_ref="session:codex-session:context-pack-assertions",
                scope_ref="repo:polylogue",
                kind=AssertionKind.CAVEAT,
                body_text="This private caveat should stay out of context.",
                status="active",
                context_policy={"inject": False},
                now_ms=1_700_000_000_100,
            )
            conn.commit()

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        image = await poly.context_pack_payload(
            query="assertion",
            max_sessions=1,
            include_messages=False,
            include_assertions=True,
        )

    rendered = "\n".join(segment.markdown or "" for segment in image.segments)
    assert "Use the shared assertion facade in context surfaces." in rendered
    assert "This private caveat should stay out of context." not in rendered

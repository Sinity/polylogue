"""Tests for the collapsed ``context-image`` capability.

The standalone context-image assembler (``polylogue.context.pack``) and its
``ContextImage*`` DTO family were removed: the capability is now a thin lens over
the shared ``compile_context`` engine. ``Polylogue.context_image_payload`` selects
sessions through the query algebra and returns the ``ContextImage`` payload that
the CLI ``read`` modifiers and the MCP ``build_context_image`` tool both emit.
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
    _seed_messages(archive_root, provider_session_id=provider_session_id, texts=(text,))


def _seed_messages(archive_root: Path, *, provider_session_id: str, texts: tuple[str, ...]) -> None:
    with ArchiveStore(archive_root) as archive:
        archive.write_parsed(
            ParsedSession(
                source_name=Provider.CODEX,
                provider_session_id=provider_session_id,
                title="Archive context image",
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:01:00+00:00",
                messages=[
                    ParsedMessage(
                        provider_message_id=f"m{index}",
                        role=Role.USER,
                        text=text,
                        blocks=[ParsedContentBlock(type=BlockType.TEXT, text=text)],
                    )
                    for index, text in enumerate(texts, start=1)
                ],
            )
        )


@pytest.mark.asyncio
async def test_context_image_payload_compiles_context_image(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _seed(archive_root, provider_session_id="context-image-v1", text="hello archive pack")

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        image = await poly.context_image_payload(
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
    assert image.spec.seed_project_path == "/realm/project/polylogue"


@pytest.mark.asyncio
async def test_compile_context_accepts_filtered_seed_selection(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _seed(archive_root, provider_session_id="context-filter-v1", text="filtered archive context")

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        image = await poly.compile_context(
            ContextSpec(
                purpose="handoff",
                seed_query="filtered archive",
                seed_project_path="/realm/project/polylogue",
                seed_query_limit=1,
                read_views=("messages",),
            )
        )

    assert image.spec.seed_project_path == "/realm/project/polylogue"
    assert image.segments
    assert "filtered archive context" in (image.segments[0].markdown or "")


@pytest.mark.asyncio
async def test_max_tokens_bounds_output_with_omission_accounting(tmp_path: Path) -> None:
    """A tiny --max-tokens budget clips within resolved message segments."""
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
    assert len(bounded.segments) == len(unbounded.segments)
    assert bounded.omitted == ()
    assert all(segment.caveats for segment in bounded.segments)
    assert all("omitted from this message" in (segment.markdown or "") for segment in bounded.segments)


@pytest.mark.asyncio
async def test_compile_context_can_slice_query_matched_message_window(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _seed_messages(
        archive_root,
        provider_session_id="windowed",
        texts=(
            "alpha before one",
            "alpha before two",
            "alpha before three",
            "needle matched text " + ("x" * 80),
            "alpha after one",
            "alpha after two",
        ),
    )

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        image = await poly.compile_context(
            ContextSpec(
                seed_query="needle",
                read_views=("messages",),
                max_tokens=100,
                max_messages_per_session=3,
                max_chars_per_message=32,
            )
        )

    assert not image.omitted
    segment = image.segments[0]
    rendered = segment.markdown or ""
    assert "needle matched text" in rendered
    assert "chars omitted from this message" in rendered
    assert "earlier messages omitted from this window" in rendered
    assert "later messages omitted from this window" in rendered
    assert segment.lossiness == "bounded_message_window"


@pytest.mark.asyncio
async def test_context_image_payload_uses_bounded_message_defaults(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _seed(archive_root, provider_session_id="long-context-image", text="long body " + ("x" * 20_000))

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        image = await poly.context_image_payload(
            query="long body",
            max_sessions=1,
            max_tokens=5_000,
        )

    assert image.segments
    assert not image.omitted
    assert image.spec.max_messages_per_session == 24
    assert image.spec.max_chars_per_message == 1800
    rendered = image.segments[0].markdown or ""
    assert "chars omitted from this message" in rendered
    assert image.segments[0].lossiness == "bounded_message_window"


@pytest.mark.asyncio
async def test_context_image_query_preserves_matched_message_window(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _seed_messages(
        archive_root,
        provider_session_id="anchored-context",
        texts=(
            "alpha first",
            "alpha second",
            "alpha third",
            "needle centered",
            "alpha fifth",
            "alpha sixth",
        ),
    )

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        image = await poly.context_image_payload(
            query="needle",
            max_sessions=1,
            max_tokens=100,
            max_messages_per_session=3,
        )

    assert image.spec.seed_query == "needle"
    assert not image.omitted
    rendered = image.segments[0].markdown or ""
    assert "needle centered" in rendered
    assert "alpha first" not in rendered
    assert "earlier messages omitted from this window" in rendered
    assert "later messages omitted from this window" in rendered


@pytest.mark.asyncio
async def test_context_image_payload_includes_injectable_assertions_via_context_image(tmp_path: Path) -> None:
    archive_root = tmp_path / "archive"
    _seed(archive_root, provider_session_id="context-image-assertions", text="assertion context image")
    with ArchiveStore(archive_root) as archive:
        with sqlite3.connect(archive.user_db_path) as conn:
            upsert_assertion(
                conn,
                assertion_id="inject-decision",
                target_ref="session:codex-session:context-image-assertions",
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
                target_ref="session:codex-session:context-image-assertions",
                scope_ref="repo:polylogue",
                kind=AssertionKind.CAVEAT,
                body_text="This private caveat should stay out of context.",
                status="active",
                context_policy={"inject": False},
                now_ms=1_700_000_000_100,
            )
            conn.commit()

    async with Polylogue(archive_root=archive_root, db_path=archive_root / "index.db") as poly:
        image = await poly.context_image_payload(
            query="assertion",
            max_sessions=1,
            include_messages=False,
            include_assertions=True,
        )

    rendered = "\n".join(segment.markdown or "" for segment in image.segments)
    assert "Use the shared assertion facade in context surfaces." in rendered
    assert "This private caveat should stay out of context." not in rendered

from __future__ import annotations

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider, TitleSource
from polylogue.pipeline.ids import session_revision_projection
from polylogue.sources.parsers.antigravity import AntigravitySessionSummary, _mark_active_leaf, parse_markdown_export
from polylogue.sources.parsers.base import ParsedMessage


def test_parse_markdown_export_splits_known_sections() -> None:
    markdown = """# Chat Session

Note: _This is purely the output of the chat session._

### User Input

Run pytest.

### Planner Response

The focused checks passed.
"""
    summary = AntigravitySessionSummary(
        cascade_id="cascade-1",
        title="Focused checks",
        workspace_name="polylogue",
        snippet="Run pytest.",
        last_modified_time="2026-03-05T04:21:34Z",
    )

    session = parse_markdown_export(markdown, summary)

    assert session.source_name is Provider.ANTIGRAVITY
    assert session.provider_session_id == "cascade-1"
    assert session.title == "Focused checks"
    assert session.title_source is TitleSource.ORIGIN
    assert session.updated_at == "2026-03-05T04:21:34Z"
    assert [message.role for message in session.messages] == [Role.USER, Role.ASSISTANT]
    assert [message.text for message in session.messages] == [
        "Run pytest.",
        "The focused checks passed.",
    ]
    assert session.messages[0].blocks[0].type == BlockType.TEXT
    assert session.messages[0].provider_message_id.startswith("synthetic-")
    assert session.messages[1].provider_message_id.startswith("synthetic-")
    assert session.messages[0].provider_message_id != session.messages[1].provider_message_id
    assert [message.position for message in session.messages] == [0, 1]
    assert [message.variant_index for message in session.messages] == [0, 0]
    assert [message.is_active_path for message in session.messages] == [True, True]
    assert [message.is_active_leaf for message in session.messages] == [False, True]
    assert session.active_leaf_message_provider_id == session.messages[1].provider_message_id


def test_parse_markdown_export_reordering_keeps_synthetic_revision_identity() -> None:
    summary = AntigravitySessionSummary(cascade_id="cascade-order")
    forward = parse_markdown_export(
        "### User Input\n\nQuestion\n\n### Planner Response\n\nAnswer\n",
        summary,
    )
    reordered = parse_markdown_export(
        "### Planner Response\n\nAnswer\n\n### User Input\n\nQuestion\n",
        summary,
    )

    assert (
        session_revision_projection(forward).message_contents == session_revision_projection(reordered).message_contents
    )


def test_mark_active_leaf_flags_exactly_one_message_with_duplicate_ids() -> None:
    """bd polylogue-2hwl: a duplicate ``provider_message_id`` (a retried or
    regenerated section reusing the same id) must not produce more than one
    ``is_active_leaf=True`` message -- the pre-fix comparison
    (``message.provider_message_id == active_leaf_message_provider_id``)
    flagged every position sharing that id, not just the true final one.
    """
    messages = [
        ParsedMessage(provider_message_id="dup", role=Role.USER, text="first", position=0, is_active_path=True),
        ParsedMessage(provider_message_id="other", role=Role.ASSISTANT, text="middle", position=1, is_active_path=True),
        ParsedMessage(provider_message_id="dup", role=Role.ASSISTANT, text="final", position=2, is_active_path=True),
    ]

    marked = _mark_active_leaf(messages)

    leaves = [message for message in marked if message.is_active_leaf]
    assert len(leaves) == 1
    assert leaves[0].text == "final"


def test_parse_markdown_export_falls_back_to_single_export_message() -> None:
    markdown = """# Chat Session

Note: generated export

Unstructured transcript body.
"""
    summary = AntigravitySessionSummary(cascade_id="cascade-2")

    session = parse_markdown_export(markdown, summary)

    assert [message.role for message in session.messages] == [Role.ASSISTANT]
    assert session.messages[0].provider_message_id.startswith("synthetic-")
    assert session.messages[0].text == "Unstructured transcript body."
    assert session.messages[0].position == 0
    assert session.messages[0].is_active_leaf is True
    assert session.active_leaf_message_provider_id == session.messages[0].provider_message_id
    assert session.title_source is None


def test_parse_markdown_export_has_no_degraded_flag() -> None:
    """Language-server export sessions are whole transcripts — not fragmented.

    Markdown export sessions must NOT carry the brain-metadata fragment flag;
    they represent complete work sessions and must not be excluded from primary
    views by the same filter that suppresses brain-metadata fragments.
    """
    markdown = "### User Input\n\nRun checks.\n\n### Planner Response\n\nDone.\n"
    summary = AntigravitySessionSummary(
        cascade_id="cascade-clean",
        title="Clean session",
        last_modified_time="2026-04-01T12:00:00Z",
    )

    session = parse_markdown_export(markdown, summary)

    assert session.ingest_flags == []

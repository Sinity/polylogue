from __future__ import annotations

from pathlib import Path

from polylogue.archive.message.roles import Role
from polylogue.core.json import JSONDocument
from polylogue.sources.parsers.antigravity import (
    BRAIN_METADATA_FRAGMENT_FLAG,
    AntigravitySessionSummary,
    looks_like_brain_metadata,
    parse_brain_metadata,
    parse_markdown_export,
)
from polylogue.types import BlockType, Provider


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
    assert session.updated_at == "2026-03-05T04:21:34Z"
    assert [message.role for message in session.messages] == [Role.USER, Role.ASSISTANT]
    assert [message.text for message in session.messages] == [
        "Run pytest.",
        "The focused checks passed.",
    ]
    assert session.messages[0].content_blocks[0].type == BlockType.TEXT
    assert session.messages[0].provider_message_id == "cascade-1:0:user_input"
    assert session.messages[1].provider_message_id == "cascade-1:1:planner_response"
    assert [message.position for message in session.messages] == [0, 1]
    assert [message.variant_index for message in session.messages] == [0, 0]
    assert [message.is_active_path for message in session.messages] == [True, True]
    assert [message.is_active_leaf for message in session.messages] == [False, True]
    assert session.active_leaf_message_provider_id == "cascade-1:1:planner_response"


def test_parse_markdown_export_falls_back_to_single_export_message() -> None:
    markdown = """# Chat Session

Note: generated export

Unstructured transcript body.
"""
    summary = AntigravitySessionSummary(cascade_id="cascade-2")

    session = parse_markdown_export(markdown, summary)

    assert [message.role for message in session.messages] == [Role.ASSISTANT]
    assert session.messages[0].provider_message_id == "cascade-2:0:export"
    assert session.messages[0].text == "Unstructured transcript body."
    assert session.messages[0].position == 0
    assert session.messages[0].is_active_leaf is True
    assert session.active_leaf_message_provider_id == "cascade-2:0:export"


def test_parse_brain_metadata_reads_adjacent_artifact(tmp_path: Path) -> None:
    session_dir = tmp_path / "brain" / "session-1"
    session_dir.mkdir(parents=True)
    artifact_path = session_dir / "implementation_plan.md"
    artifact_path.write_text("# Implementation Plan\n\nDo the work.\n", encoding="utf-8")
    metadata_path = session_dir / "implementation_plan.md.metadata.json"
    payload: JSONDocument = {
        "artifactType": "ARTIFACT_TYPE_OTHER",
        "summary": "Implementation Plan",
        "updatedAt": "2026-01-07T19:08:15.216541610Z",
    }

    assert looks_like_brain_metadata(payload, metadata_path) is True

    session = parse_brain_metadata(payload, metadata_path, "fallback")

    assert session.source_name is Provider.ANTIGRAVITY
    assert session.provider_session_id == "session-1:implementation_plan.md"
    assert session.title == "Implementation Plan"
    assert session.updated_at == "2026-01-07T19:08:15.216541610Z"
    assert session.messages[0].role is Role.ASSISTANT
    assert session.messages[0].text == "# Implementation Plan\n\nDo the work.\n"
    assert session.messages[0].position == 0
    assert session.messages[0].is_active_leaf is True
    assert session.active_leaf_message_provider_id == "session-1:implementation_plan.md:artifact"
    assert session.messages[0].content_blocks[0].type == BlockType.TEXT


def test_parse_brain_metadata_marks_missing_artifact(tmp_path: Path) -> None:
    metadata_path = tmp_path / "brain" / "session-2" / "missing.md.metadata.json"
    metadata_path.parent.mkdir(parents=True)
    payload: JSONDocument = {
        "artifactType": "ARTIFACT_TYPE_OTHER",
        "summary": "Missing body",
        "updatedAt": "2026-01-07T19:08:15.216541610Z",
    }

    session = parse_brain_metadata(payload, metadata_path, "fallback")

    assert session.provider_session_id == "session-2:missing.md"
    assert session.messages[0].text == "Missing body"


# ---------------------------------------------------------------------------
# Regression: brain-metadata fragmentation degraded flag (#1764)
# ---------------------------------------------------------------------------


def test_parse_brain_metadata_sets_degraded_ingest_flag(tmp_path: Path) -> None:
    """Brain-metadata sessions are flagged degraded:brain-metadata-fragment.

    Each *.metadata.json artifact becomes its own single-message session,
    fragmenting one work session into N sessions.  The ingest_flags field
    carries the degraded marker so it is written as an auto-tag at storage
    time and can be excluded from primary session counts and insight rollups.
    """
    session_dir = tmp_path / "brain" / "session-abc"
    session_dir.mkdir(parents=True)
    artifact_path = session_dir / "output.md"
    artifact_path.write_text("# Output\n\nSome result.\n", encoding="utf-8")
    metadata_path = session_dir / "output.md.metadata.json"
    payload: JSONDocument = {
        "artifactType": "ARTIFACT_TYPE_OTHER",
        "summary": "Output",
        "updatedAt": "2026-04-01T10:00:00Z",
    }

    session = parse_brain_metadata(payload, metadata_path, "fallback")

    assert BRAIN_METADATA_FRAGMENT_FLAG in session.ingest_flags, (
        f"Expected {BRAIN_METADATA_FRAGMENT_FLAG!r} in ingest_flags, got {session.ingest_flags!r}"
    )


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

    assert BRAIN_METADATA_FRAGMENT_FLAG not in session.ingest_flags, (
        f"Markdown export session must not carry {BRAIN_METADATA_FRAGMENT_FLAG!r}, "
        f"got ingest_flags={session.ingest_flags!r}"
    )

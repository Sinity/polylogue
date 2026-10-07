"""Antigravity trajectory steps: timing, tool outcomes, edits and parent aliases."""

from pathlib import Path

import pytest

from polylogue.archive.message.roles import Role
from polylogue.core.enums import BlockType, Provider
from polylogue.sources.dispatch import admit_parsed_sessions_for_publication
from polylogue.sources.parsers.base import AdmissionDisposition, AdmissionRefusalReason
from polylogue.sources.sqlite_inspection import inspect_sqlite_source
from tests.infra.antigravity_parser import parse_trajectory_db
from tests.infra.source_parser_cases import trajectory_db


@pytest.mark.parametrize("name", ["nested", "malformed_patch", "parent", "row_exit"])
def test_streamed_bound_trajectory_preview_matches_parser_evidence(tmp_path: Path, name: str) -> None:
    source = trajectory_db(tmp_path / "source.sqlite", name)
    sessions = list(parse_trajectory_db(source, source.stem))
    preview = inspect_sqlite_source(source)
    assert preview.produced == {
        "sessions": len(sessions),
        "messages": sum(len(session.messages) for session in sessions),
        "blocks": sum(len(message.blocks) for session in sessions for message in session.messages),
        "actions": sum(
            block.type is BlockType.TOOL_USE
            for session in sessions
            for message in session.messages
            for block in message.blocks
        ),
        "raw_records": len(sessions),
        "session_refs": [f"session:antigravity:{session.provider_session_id}" for session in sessions],
    }
    assert preview.degraded == any(session.ingest_flags for session in sessions)
    preflight = inspect_sqlite_source(source, preflight=True)
    positive = admit_parsed_sessions_for_publication(sessions, provider=Provider.ANTIGRAVITY, source_path=str(source))
    assert preflight.admitted == len(positive)
    assert preflight.produced == {**preview.produced, "session_refs": []}
    assert preflight.degraded == (preview.degraded or len(positive) != len(sessions))


def test_millisecond_step_timestamp(tmp_path: Path) -> None:
    """Treating milliseconds as seconds refuses the otherwise valid step."""
    [session] = list(parse_trajectory_db(trajectory_db(tmp_path / "one.db", "timestamp")))
    assert session.messages[0].occurred_at_ms == 1700000000123


def test_structural_row_exit_code(tmp_path: Path) -> None:
    """Ignoring the SQLite exit_code column leaves a failed result unknown."""
    [session] = list(parse_trajectory_db(trajectory_db(tmp_path / "one.db", "row_exit")))
    [block] = session.messages[0].blocks
    assert block.exit_code == 7
    assert block.is_error is True


@pytest.mark.parametrize("name", ["error_text", "error_object"])
def test_nonempty_error_is_failure(tmp_path: Path, name: str) -> None:
    """Boolean-only error handling loses string/object error verdicts."""
    [session] = list(parse_trajectory_db(trajectory_db(tmp_path / "one.db", name)))
    assert session.messages[0].blocks[0].is_error is True


def test_nested_payload_keeps_outer_role_and_clock(tmp_path: Path) -> None:
    """Replacing the outer envelope loses its user role and timestamp."""
    [session] = list(parse_trajectory_db(trajectory_db(tmp_path / "one.db", "nested")))
    assert session.messages[0].role is Role.USER
    assert session.messages[0].occurred_at_ms == 1700000000123
    assert session.messages[0].text == "nested text"


def test_bad_patch_refuses_only_its_step(tmp_path: Path) -> None:
    """An invalid structured-patch member formerly aborted every sibling."""
    [session] = list(parse_trajectory_db(trajectory_db(tmp_path / "one.db", "malformed_patch")))
    assert [message.text for message in session.messages] == ["keep this"]
    assert session.unit_accounting is not None
    refusals = [
        row for row in session.unit_accounting.outcomes if row.disposition is AdmissionDisposition.TYPED_REFUSAL
    ]
    assert len(refusals) == 1
    assert refusals[0].ordinal == 1
    assert refusals[0].reason == AdmissionRefusalReason.MALFORMED


def test_camel_case_edit_evidence(tmp_path: Path) -> None:
    """CamelCase-only native edit fields formerly produced no edit facet."""
    [session] = list(parse_trajectory_db(trajectory_db(tmp_path / "one.db", "camel_edit")))
    edit = session.messages[0].blocks[0].file_edit
    assert edit is not None
    assert (edit.file_path, edit.old_string, edit.new_string, edit.original_file) == (
        "note.txt",
        "before",
        "after",
        "before",
    )
    assert edit.replace_all is True
    assert edit.user_modified is False


@pytest.mark.parametrize("ambiguous", [False, True])
def test_parent_aliases_canonicalize_before_ambiguity(tmp_path: Path, ambiguous: bool) -> None:
    """Two names for one parent are not two parents; true conflicts stay open."""
    sessions = list(parse_trajectory_db(trajectory_db(tmp_path / "one.db", "parent", ambiguous_parent=ambiguous)))
    child = next(session for session in sessions if session.provider_session_id == "child")
    assert child.parent_session_provider_id == (None if ambiguous else "parent")


@pytest.mark.parametrize(
    "name,reason", [("absent_outcome", "not_reported"), ("unknown_outcome", "unsupported_construct")]
)
def test_absent_outcome_is_not_reported(tmp_path: Path, name: str, reason: str) -> None:
    """An absent verdict formerly claimed an unsupported producer construct."""
    [session] = list(parse_trajectory_db(trajectory_db(tmp_path / "one.db", name)))
    block = session.messages[0].blocks[0]
    assert block.is_error is None
    assert block.outcome_unknown_reason == reason

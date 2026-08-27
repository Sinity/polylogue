from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.markers import (
    MARKER_REGISTRY,
    MarkerKindSpec,
    MarkerRegistry,
    MarkerStreamParser,
    candidates_for_block,
    lower_markers,
    parse_markers,
)


def test_line_inline_escape_markdown_and_malformed_are_observable() -> None:
    found = parse_markers(
        "::goal(owner=agent): ship it\n"
        "text [[finding: suspicious branch]]\n"
        r"\::note: escaped"
        "\n::not-registered: still evidence\n"
        "```\n::note: code\n[[note: code]]\n```\n"
    )
    assert [(item.kind, item.body) for item in found] == [
        ("goal", "ship it"),
        ("finding", "suspicious branch"),
        ("malformed", "still evidence"),
    ]
    assert found[-1].malformed


def test_registry_covers_declared_authoring_kinds_and_unknown_inline_is_evidence() -> None:
    expected = {"note", "claim", "lesson", "decision", "predict", "handoff", "anchor", "bead", "eval"}
    assert expected <= {spec.kind for spec in MARKER_REGISTRY}
    found = parse_markers("[[future-kind: retain this evidence]]\n[[note: keep this]]\n")
    assert [(item.kind, item.body, item.malformed) for item in found] == [
        ("malformed", "retain this evidence", True),
        ("note", "keep this", False),
    ]


def test_unterminated_inline_marker_is_malformed_evidence() -> None:
    found = parse_markers("before [[note: split at end\n")
    assert len(found) == 1
    assert found[0].kind == "malformed"
    assert found[0].raw_text == "[[note: split at end"


def test_streaming_split_marker_is_parsed_after_newline() -> None:
    stream = MarkerStreamParser()
    assert stream.feed("prefix\n::fin") == ()
    assert stream.feed("ding: body\n")[0].body == "body"
    assert stream.finish() == ()


def test_new_kind_is_registry_data_not_parser_control_flow() -> None:
    registry = MarkerRegistry((MarkerKindSpec("lesson", "text", AssertionKind.LESSON, "lesson"),))
    match = parse_markers("::lesson: remember\n", registry=registry)[0]
    assert match.kind == "lesson"
    assert (
        candidates_for_block("m-1", "b-2", "::lesson: remember\n", registry=registry)[0].assertion_kind
        == AssertionKind.LESSON
    )


def test_candidate_lowering_uses_existing_assertion_service_and_exact_refs(tmp_path: Path) -> None:
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    user_db = tmp_path / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    conn = sqlite3.connect(user_db)
    candidates = candidates_for_block("message-1", "block-2", "::finding: bad path\n")
    ids = lower_markers(conn, candidates, now_ms=123)
    row = conn.execute("SELECT * FROM assertions WHERE assertion_id = ?", ids).fetchone()
    assert row is not None
    assert row[4] == AssertionKind.FINDING.value
    assert row[10] == AssertionStatus.CANDIDATE.value
    assert row[8] == "agent"
    assert "message:message-1" in row[9] and "block:block-2" in row[9]
    conn.close()


def test_ownerless_declaration_fails_actionably() -> None:
    with pytest.raises(ValueError, match="lowering_target"):
        MarkerRegistry((MarkerKindSpec("orphan", "text", None, "bad"),))

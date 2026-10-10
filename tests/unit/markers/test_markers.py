from __future__ import annotations

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
from polylogue.storage.io_phase_metrics import connect_measured
from tests.infra.identity import archive_block_id, fixture_block_content_identity


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
    expected = {"note", "claim", "lesson", "decision", "predict", "blocker", "handoff", "anchor", "bead", "eval"}
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


def test_fence_closes_only_with_compatible_delimiter_and_length() -> None:
    """Anti-vacuity: toggling on every fence-looking line parses example text as a finding."""
    found = parse_markers(
        "````md\n~~~\n::note: hidden short fence\n```\n::note: hidden mixed fence\n````\n::note: visible\n"
    )
    assert [item.body for item in found] == ["visible"]


def test_unregistered_inline_marker_is_malformed_evidence() -> None:
    """Anti-vacuity: dropping future inline kinds makes an audit marker disappear."""
    found = parse_markers("[[future-kind: preserve me]]")
    assert len(found) == 1
    assert found[0].kind == "malformed" and found[0].malformed
    assert found[0].arguments == {"unregistered_kind": "future-kind"}


def test_trailing_unterminated_inline_marker_survives_after_valid_marker() -> None:
    """Anti-vacuity: the earlier close must not hide the final broken declaration."""
    found = parse_markers("[[note: good]] then [[note: broken")
    assert [(item.kind, item.body, item.malformed) for item in found] == [
        ("note", "good", False),
        ("malformed", "broken", True),
    ]


def test_nested_inline_opener_inside_accepted_span_is_not_also_malformed() -> None:
    """Anti-vacuity: flagging the outer opener as unterminated emits overlapping valid and malformed markers."""
    found = parse_markers("[[note: first [[note: second]]")
    spans = [(item.start, item.end) for item in found]
    assert all(
        not (a_start < b_end and b_start < a_end)
        for index, (a_start, a_end) in enumerate(spans)
        for b_start, b_end in spans[index + 1 :]
    )
    assert [item.malformed for item in found] == [False]


def test_stream_offsets_include_previously_consumed_chunks() -> None:
    """Anti-vacuity: resetting offsets for each feed points at the wrong source text."""
    stream = MarkerStreamParser()
    assert stream.feed("prefix\n") == ()
    found = stream.feed("::note: body\n")
    assert (found[0].start, found[0].end) == (7, 19)


def test_stream_finish_offsets_include_completed_chunks() -> None:
    """Anti-vacuity: finish must keep source coordinates after prior feed calls."""
    stream = MarkerStreamParser()
    assert stream.feed("prefix\n") == ()
    assert stream.feed("suffix\n") == ()
    found = stream.feed("::note: tail")
    assert found == ()
    finished = stream.finish()
    assert (finished[0].start, finished[0].end) == (14, 26)


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
    conn = connect_measured(user_db)
    block_id = archive_block_id("message-1", content_identity=fixture_block_content_identity("block-2"))
    candidates = candidates_for_block("message-1", block_id, "::finding: bad path\n")
    ids = lower_markers(conn, candidates, now_ms=123)
    row = conn.execute("SELECT * FROM assertions WHERE assertion_id = ?", ids).fetchone()
    assert row is not None
    assert row[4] == AssertionKind.FINDING.value
    assert row[10] == AssertionStatus.CANDIDATE.value
    assert row[8] == "agent"
    assert "message:message-1" in row[9] and f"block:{block_id}" in row[9]
    conn.close()


def test_ownerless_declaration_fails_actionably() -> None:
    with pytest.raises(ValueError, match="lowering_target"):
        MarkerRegistry((MarkerKindSpec("orphan", "text", None, "bad"),))


def test_objective_posture_assertion_kinds_are_all_agent_authorable(tmp_path: Path) -> None:
    """Anti-vacuity: drop a MarkerKindSpec whose lowering target objective posture
    reads -- or drop a kind from ``ASSERTION_TIER_KINDS`` -- and this goes red.

    ``analysis/objective_posture.py`` declares the assertion kinds that decide a
    session's posture. Before polylogue-jwqj, BLOCKER was readable there but had
    no direct authoring affordance at all, so an agent could state "I am handed
    off" and not "I am blocked".
    """

    from polylogue.analysis.objective_posture import ASSERTION_TIER_KINDS
    from polylogue.storage.sqlite.archive_tiers.bootstrap import initialize_archive_database
    from polylogue.storage.sqlite.archive_tiers.types import ArchiveTier

    authorable = {spec.lowering_target for spec in MARKER_REGISTRY}
    assert set(ASSERTION_TIER_KINDS) <= authorable

    user_db = tmp_path / "user.db"
    initialize_archive_database(user_db, ArchiveTier.USER)
    conn = connect_measured(user_db)
    try:
        block_id = archive_block_id("message-9", content_identity=fixture_block_content_identity("block-9"))
        candidates = candidates_for_block("message-9", block_id, "::blocker: waiting on a credential\n")
        assert [candidate.assertion_kind for candidate in candidates] == [AssertionKind.BLOCKER]
        ids = lower_markers(conn, candidates, now_ms=456)
        kinds = [
            row[0] for row in conn.execute("SELECT kind FROM assertions WHERE assertion_id IN (?)", ids).fetchall()
        ]
        assert kinds == [AssertionKind.BLOCKER.value]
    finally:
        conn.close()


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize(
    "body",
    [
        "::note: before\n```md\n::note: hidden\n[[finding: hidden]]\n```\n::note: after",
        "````md\n~~~\n::note: mixed hidden\n```\n::note: short hidden\n```` trailing\n::note: still hidden\n`````\n[[note: visible]]",
        "   ~~~~md\n::note: hidden\n~~~\n[[note: hidden]]\n~~~~\n::note: visible\n",
        "λ before [[note: visible]]\n\\::note: escaped\n::future-kind: malformed\n[[note: unterminated",
        "::note: before\n```\n::note: unclosed fence hides finish",
    ],
)
def test_stream_matches_batch_at_every_cut_and_line_chunk(body: str, newline: str) -> None:
    """Fence state and source offsets survive every split, including CRLF halves."""
    text = body.replace("\n", newline)
    expected = parse_markers(text)
    for match in expected:
        assert text[match.start : match.end] == match.raw_text
    for cut in range(len(text) + 1):
        stream = MarkerStreamParser()
        found = stream.feed(text[:cut]) + stream.feed(text[cut:]) + stream.finish()
        assert found == expected, cut
        assert stream.finish() == ()
    stream = MarkerStreamParser()
    found = tuple(match for line in text.splitlines(keepends=True) for match in stream.feed(line)) + stream.finish()
    assert found == expected
    stream = MarkerStreamParser()
    found = tuple(match for character in text for match in stream.feed(character)) + stream.finish()
    assert found == expected


def test_stream_fence_state_uses_declared_registry() -> None:
    registry = MarkerRegistry((MarkerKindSpec("lesson", "text", AssertionKind.LESSON, "lesson"),))
    text = "~~~\n::lesson: hidden\n~~~\n::lesson: visible\n::note: unknown"
    stream = MarkerStreamParser(registry=registry)
    found = tuple(match for line in text.splitlines(keepends=True) for match in stream.feed(line)) + stream.finish()
    assert found == parse_markers(text, registry=registry)
    assert [(match.kind, match.body) for match in found] == [("lesson", "visible"), ("malformed", "unknown")]

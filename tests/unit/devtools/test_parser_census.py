"""The parser census is an oracle, so every test here is about it going red.

Anti-vacuity, stated once for the file: each red is produced by breaking
exactly one thing a real parser change breaks -- a parser that stops returning
sessions for an origin, a denominator that stops naming a member, a parse that
changes with identical bytes and an unchanged parser fingerprint -- and the
assertions name the origin and the count. A census that always reported
success, or one whose diff only counted rows, would fail every one of them.
The corpus is the repository's own neutral synthetic fixtures; nothing here
reads operator material.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from devtools.parser_census import (
    OUTCOME_NO_SESSIONS,
    OUTCOME_PARSED,
    Census,
    MemberCensus,
    ParserCensusError,
    build_census,
    census_dir,
    diff_censuses,
    load_census,
    main,
    source_denominator,
    write_census,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CLAUDE_CODE_FIXTURES = REPO_ROOT / "tests" / "fixtures" / "claude-code"
CHATGPT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "chatgpt" / "native-conversation-v1.json"


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A two-origin demo corpus built from the repository's own fixtures."""
    root = tmp_path / "corpus"
    (root / "claude-code").mkdir(parents=True)
    (root / "chatgpt").mkdir(parents=True)
    for source in sorted(CLAUDE_CODE_FIXTURES.glob("*.jsonl")):
        shutil.copy(source, root / "claude-code" / source.name)
    shutil.copy(CHATGPT_FIXTURE, root / "chatgpt" / CHATGPT_FIXTURE.name)
    return root


def _sources(corpus: Path) -> list[tuple[str, Path]]:
    return [("claude-code", corpus / "claude-code"), ("chatgpt", corpus / "chatgpt")]


def _census(corpus: Path) -> Census:
    members, denominator = source_denominator(_sources(corpus))
    return build_census(members, denominator, workers=1)


def test_census_classifies_every_member_of_the_declared_denominator(corpus: Path) -> None:
    census = _census(corpus)

    assert len(census.members) == len(list((corpus / "claude-code").iterdir())) + 1
    assert census.totals["members"] == len(census.members)
    origins = {member.origin for member in census.members}
    assert "claude-code-session" in origins
    assert "chatgpt-export" in origins or "chatgpt" in str(origins)
    parsed = [member for member in census.members if member.outcome == OUTCOME_PARSED]
    assert parsed, "the demo corpus must parse, or the oracle proves nothing"
    assert all(member.session_digests for member in parsed)
    assert all(member.digest for member in census.members)
    assert census.parser_fingerprint


def test_a_parser_that_stops_returning_sessions_makes_every_member_of_that_origin_a_new_failure(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance criterion 4, demonstrated red."""
    before = _census(corpus)
    claude_members = [m for m in before.members if m.origin == "claude-code-session"]
    assert claude_members and all(m.outcome == OUTCOME_PARSED for m in claude_members)

    from polylogue.sources import dispatch

    monkeypatch.setattr(dispatch, "parse_stream_payload", lambda *args, **kwargs: [])
    after = _census(corpus)

    diff = diff_censuses(before, after)
    assert not diff.ok
    rows = {row.origin: row for row in diff.origins}
    assert rows["claude-code-session"].new_failures == len(claude_members)
    for row in rows.values():
        if row.origin != "claude-code-session":
            assert row.new_failures == 0
    failures = [f for f in diff.errors if f.kind == "parse_failure_new"]
    assert len(failures) == len(claude_members)
    assert all(f.origin == "claude-code-session" for f in failures)
    assert all(
        member.outcome == OUTCOME_NO_SESSIONS for member in after.members if member.origin == "claude-code-session"
    )


def test_a_denominator_that_stops_naming_a_member_is_an_error(corpus: Path) -> None:
    """Silent narrowing -- how tu1f's stale package survived -- is red."""
    before = _census(corpus)
    victim = next(member for member in before.members if member.relative.endswith(".jsonl"))
    (Path(victim.root) / victim.relative).unlink()

    after = _census(corpus)
    diff = diff_censuses(before, after)

    assert not diff.ok
    absent = [f for f in diff.errors if f.kind == "member_absent"]
    assert [f.member for f in absent] == [victim.key]
    assert {row.origin: row.removed for row in diff.origins}[victim.origin] == 1


def test_a_changed_parse_with_identical_bytes_and_fingerprint_is_nondeterminism(corpus: Path) -> None:
    before = _census(corpus)
    victim = before.members[0]
    after = replace(before, members=(replace(victim, messages=victim.messages + 1),) + before.members[1:])

    diff = diff_censuses(before, after)

    assert not diff.ok
    assert [f.kind for f in diff.errors] == ["digest_changed_nondeterministic"]
    assert diff.errors[0].member == victim.key


def test_the_same_change_under_a_moved_parser_fingerprint_is_a_notice(corpus: Path) -> None:
    before = _census(corpus)
    victim = before.members[0]
    after = replace(
        before,
        parser_fingerprint="a-different-parser",
        members=(replace(victim, messages=victim.messages + 1),) + before.members[1:],
    )

    diff = diff_censuses(before, after)

    assert diff.ok
    assert diff.parser_moved
    assert [f.kind for f in diff.notices] == ["digest_changed_parser_moved"]
    assert {row.origin: row.changed_digest for row in diff.origins}[victim.origin] == 1


def test_an_origin_that_stops_classifying_is_an_error(corpus: Path) -> None:
    before = _census(corpus)
    victim = before.members[0]
    after = replace(before, members=(replace(victim, origin="unknown"),) + before.members[1:])

    diff = diff_censuses(before, after)

    assert not diff.ok
    kinds = {f.kind for f in diff.errors}
    assert "origin_changed" in kinds


def test_a_bounded_census_is_never_comparable_evidence(corpus: Path) -> None:
    """A --limit run omits members; that is not evidence they disappeared."""
    before = _census(corpus)
    partial = replace(before, members=before.members[:1], partial=True)

    diff = diff_censuses(before, partial)

    assert diff.compared is False
    assert diff.findings == ()
    assert diff.skipped_reason is not None


def test_changed_source_bytes_are_a_notice_and_suppress_the_digest_verdict(corpus: Path) -> None:
    before = _census(corpus)
    victim = before.members[0]
    after = replace(
        before,
        members=(replace(victim, sha256="0" * 64, messages=victim.messages + 1),) + before.members[1:],
    )

    diff = diff_censuses(before, after)

    assert diff.ok
    assert [f.kind for f in diff.notices] == ["input_changed"]


def test_round_trips_through_its_document(corpus: Path, tmp_path: Path) -> None:
    census = _census(corpus)
    path = write_census(census, tmp_path / "census-1.json")

    reloaded = load_census(path)

    assert reloaded.to_payload() == census.to_payload()
    assert {m.key: m.digest for m in reloaded.members} == {m.key: m.digest for m in census.members}


def test_a_foreign_document_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "census-1.json"
    path.write_text(json.dumps({"schema": "something.else"}), encoding="utf-8")

    with pytest.raises(ParserCensusError):
        load_census(path)


def test_the_command_exits_nonzero_on_a_red_and_writes_no_archive(
    corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The census resolves an archive root it never opens (polylogue-pv8xp)."""
    absent_archive = tmp_path / "no-such-archive"
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(absent_archive))
    census_home = tmp_path / "census"
    monkeypatch.setenv("POLYLOGUE_PARSER_CENSUS_DIR", str(census_home))
    argv = [
        "--source",
        f"claude-code={corpus / 'claude-code'}",
        "--source",
        f"chatgpt={corpus / 'chatgpt'}",
        "--workers",
        "1",
    ]

    assert main(argv) == 0
    assert not absent_archive.exists()
    recorded = sorted(census_home.glob("census-*.json"))
    assert len(recorded) == 1

    from polylogue.sources import dispatch

    monkeypatch.setattr(dispatch, "parse_stream_payload", lambda *args, **kwargs: [])
    assert main(argv) == 1
    assert len(sorted(census_home.glob("census-*.json"))) == 2


def test_the_census_directory_is_never_inside_the_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYLOGUE_PARSER_CENSUS_DIR", raising=False)
    monkeypatch.setenv("POLYLOGUE_ARCHIVE_ROOT", str(tmp_path / "archive"))

    directory = census_dir(None)

    assert not directory.is_relative_to(tmp_path / "archive")
    assert directory.parent == tmp_path


def test_a_member_outside_the_byte_bound_is_recorded_not_omitted(corpus: Path) -> None:
    members, denominator = source_denominator(_sources(corpus))
    census = build_census(members, denominator, workers=1, max_member_bytes=1)

    assert len(census.members) == len(members)
    assert {member.outcome for member in census.members} == {"oversized"}
    assert all("over the declared" in (member.reason or "") for member in census.members)


def test_the_bound_crossing_is_visible_in_the_diff(corpus: Path) -> None:
    members, denominator = source_denominator(_sources(corpus))
    before = build_census(members, denominator, workers=1)
    after = build_census(members, denominator, workers=1, max_member_bytes=1)

    diff = diff_censuses(before, after)

    assert not diff.ok
    assert all(row.changed_digest == row.members for row in diff.origins)


def test_workers_do_not_change_the_result(corpus: Path) -> None:
    members, denominator = source_denominator(_sources(corpus))
    serial = build_census(members, denominator, workers=1)
    pooled = build_census(members, denominator, workers=2)

    assert {m.key: m.digest for m in serial.members} == {m.key: m.digest for m in pooled.members}


def test_a_member_class_the_direct_route_does_not_own_still_gets_a_row(corpus: Path, tmp_path: Path) -> None:
    """A zip or database member is censused through the explain route."""
    (corpus / "claude-code" / "notes.md").write_text("not a session", encoding="utf-8")

    census = _census(corpus)

    extra = next(member for member in census.members if member.relative.endswith(".md"))
    assert extra.depth == "explain"
    assert extra.outcome in {"not_session_artifact", "no_sessions", "decode_failure", "parse_failure"}


def test_an_unrecorded_denominator_member_is_unavailable_not_missing(corpus: Path) -> None:
    members, denominator = source_denominator(_sources(corpus))
    ghost = replace(members[0], relative="ghost.jsonl")

    census = build_census((*members, ghost), denominator, workers=1)

    row = next(member for member in census.members if member.relative == "ghost.jsonl")
    assert row.outcome == "unavailable"
    assert row.failed


def test_member_payloads_round_trip_without_loss(corpus: Path) -> None:
    census = _census(corpus)

    for member in census.members:
        assert MemberCensus.from_payload(member.to_payload()) == member

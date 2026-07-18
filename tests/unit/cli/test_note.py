"""Behavioral proof for the terminal candidate-capture command."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from pathlib import Path
from typing import cast

from click.testing import CliRunner

from polylogue.api import Polylogue
from polylogue.cli import cli
from polylogue.cli.commands.note import MAX_NOTE_STDIN_BYTES
from polylogue.core.enums import AssertionKind, AssertionStatus
from polylogue.surfaces.payloads import AssertionCandidateReviewListPayload, PublicRefResolutionPayload
from tests.infra.storage_records import SessionBuilder


def _capture(cli_workspace: dict[str, Path], args: list[str], *, input: str | bytes | None = None) -> dict[str, object]:
    result = CliRunner().invoke(
        cli,
        ["--plain", "note", *args, "--format", "json"],
        input=input,
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    return cast(dict[str, object], json.loads(result.output))


def test_terminal_note_writes_one_unanchored_candidate_to_user_tier(cli_workspace: dict[str, Path]) -> None:
    payload = _capture(cli_workspace, ["WAL contention was the real cause"])

    assert payload["kind"] == "note"
    assert payload["status"] == "candidate"
    assert payload["context_policy"] == {"inject": False, "promotion_required": True}
    assert payload["value"] == {
        "capture_surface": "terminal",
        "scope_refs": [],
        "unanchored": True,
    }


def test_terminal_note_preserves_anchor_scope_and_kind_choices(cli_workspace: dict[str, Path]) -> None:
    session = (
        SessionBuilder(cli_workspace["db_path"], "terminal-note-anchor")
        .provider("codex")
        .working_directories([str(Path.cwd())])
    )
    session.save()
    session_ref = f"session:{session.native_session_id()}"
    payload = _capture(
        cli_workspace,
        [
            "capture this lesson",
            "--ref",
            session_ref,
            "--repo",
            "polylogue",
            "--topic",
            "sqlite",
            "--kind",
            "lesson",
        ],
    )

    assert payload["kind"] == "lesson"
    assert payload["target_ref"] == session_ref
    assert payload["scope_ref"] == "repo:polylogue"
    assert payload["evidence_refs"] == [
        session_ref,
        "repo:polylogue",
        "insight:sqlite",
    ]
    assert payload["value"] == {
        "capture_surface": "terminal",
        "scope_refs": ["repo:polylogue", "insight:sqlite"],
        "unanchored": False,
    }


def test_terminal_note_kind_options_all_land_as_candidates(cli_workspace: dict[str, Path]) -> None:
    expected = {
        "note": AssertionKind.NOTE,
        "claim": AssertionKind.DECISION,
        "correction": AssertionKind.CORRECTION,
        "lesson": AssertionKind.LESSON,
    }
    for option, assertion_kind in expected.items():
        payload = _capture(cli_workspace, [f"{option} capture", "--kind", option])
        assert payload["kind"] == assertion_kind.value
        assert payload["status"] == AssertionStatus.CANDIDATE.value


def test_terminal_note_reads_bounded_stdin(cli_workspace: dict[str, Path]) -> None:
    payload = _capture(cli_workspace, ["--stdin", "--kind", "lesson"], input="lesson from a diff")
    assert payload["body_text"] == "lesson from a diff"
    assert payload["kind"] == "lesson"

    oversized = CliRunner().invoke(cli, ["--plain", "note", "--stdin"], input=b"x" * (MAX_NOTE_STDIN_BYTES + 1))
    assert oversized.exit_code == 2
    assert "stdin note exceeds" in oversized.output


def test_terminal_note_last_resolves_the_latest_session_for_current_cwd(cli_workspace: dict[str, Path]) -> None:
    repo_root = cli_workspace["archive_root"].parent / "repo"
    nested_cwd = repo_root / "nested" / "terminal"
    nested_cwd.mkdir(parents=True)
    (repo_root / ".git").mkdir()
    session = (
        SessionBuilder(cli_workspace["db_path"], "terminal-note-last")
        .provider("codex")
        .working_directories([str(repo_root)])
    )
    session.save()

    previous_cwd = Path.cwd()
    try:
        os.chdir(nested_cwd)
        payload = _capture(cli_workspace, ["anchor to last", "--ref", "last"])
    finally:
        os.chdir(previous_cwd)
    assert payload["target_ref"] == f"session:{session.native_session_id()}"
    assert payload["evidence_refs"] == [f"session:{session.native_session_id()}"]


def test_terminal_note_rejects_non_session_refs(cli_workspace: dict[str, Path]) -> None:
    result = CliRunner().invoke(
        cli,
        ["--plain", "note", "not an anchor", "--ref", "repo:polylogue"],
        catch_exceptions=False,
    )
    assert result.exit_code == 1
    assert "--ref must be a session" in result.output


def test_terminal_note_reports_a_missing_session_anchor_without_a_traceback(cli_workspace: dict[str, Path]) -> None:
    """A real root invocation turns storage lookup failure into CLI feedback."""

    result = CliRunner().invoke(
        cli,
        ["--plain", "note", "not an anchor", "--ref", "session:missing"],
        catch_exceptions=False,
    )
    assert result.exit_code == 1
    assert "session ref not found: missing" in result.output
    assert "Traceback" not in result.output


def test_terminal_note_is_visible_to_the_real_pending_candidate_reader(cli_workspace: dict[str, Path]) -> None:
    payload = _capture(cli_workspace, ["candidate stays pending"])

    async def read() -> tuple[AssertionCandidateReviewListPayload, list[str], list[str]]:
        async with Polylogue(archive_root=cli_workspace["archive_root"]) as poly:
            reviews = await poly.list_assertion_candidate_reviews()
            before_judgment = await poly.list_blackboard_notes()
            await poly.judge_assertion_candidate(
                candidate_ref=f"assertion:{payload['assertion_id']}",
                decision="accept",
                reason="operator approved terminal capture",
            )
            after_judgment = await poly.list_blackboard_notes()
            return (
                reviews,
                [note.note_id for note in before_judgment],
                [note.note_id for note in after_judgment],
            )

    reviews, blackboard_before_judgment, blackboard_after_judgment = asyncio.run(read())
    assert [review.candidate.assertion_id for review in reviews.items] == [payload["assertion_id"]]
    assert reviews.items[0].candidate.status is AssertionStatus.CANDIDATE
    assert blackboard_before_judgment == []
    assert blackboard_after_judgment == [payload["assertion_id"]]


def test_terminal_note_idempotency_survives_root_judgment_canary_route(
    cli_workspace: dict[str, Path],
) -> None:
    """Production CLI capture -> root judge -> replay preserves one lifecycle.

    Anti-vacuity: random capture ids, overwriting a terminal candidate on
    replay, bypassing root ``judge``, or omitting the promotion evidence and
    context policy makes this test fail.
    """

    session = SessionBuilder(cli_workspace["db_path"], "terminal-note-canary").provider("codex")
    session.save()
    session_ref = f"session:{session.native_session_id()}"
    key = "mrxt-operator-canary"
    capture_args = [
        "Operator observed WAL reservation preserving judgment state",
        "--kind",
        "lesson",
        "--ref",
        session_ref,
        "--idempotency-key",
        key,
    ]

    first = _capture(cli_workspace, capture_args)
    before_judgment_retry = _capture(cli_workspace, capture_args)
    assert before_judgment_retry["assertion_id"] == first["assertion_id"]
    candidate_ref = f"assertion:{first['assertion_id']}"

    judged = CliRunner().invoke(
        cli,
        [
            "--plain",
            "judge",
            "--accept",
            candidate_ref,
            "--reason",
            "genuine operator canary acceptance",
            "--actor-ref",
            "user:operator",
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert judged.exit_code == 0, judged.output
    judgment_payload = json.loads(judged.output)
    assert judgment_payload["applied_count"] == 1
    item = judgment_payload["items"][0]
    assert item["outcome"] == "applied"
    resulting_ref = item["result"]["judgment"]["resulting_assertion_ref"]
    assert resulting_ref.startswith("assertion:")

    after_judgment_retry = _capture(cli_workspace, capture_args)
    assert after_judgment_retry["assertion_id"] == first["assertion_id"]
    assert after_judgment_retry["status"] == "accepted"

    changed = CliRunner().invoke(
        cli,
        [
            "--plain",
            "note",
            "Changed content must conflict",
            "--kind",
            "lesson",
            "--ref",
            session_ref,
            "--idempotency-key",
            key,
            "--format",
            "json",
        ],
        catch_exceptions=False,
    )
    assert changed.exit_code == 1
    assert "idempotency_key conflicts" in changed.output

    async def verify_surfaces() -> tuple[
        AssertionCandidateReviewListPayload, PublicRefResolutionPayload, PublicRefResolutionPayload
    ]:
        async with Polylogue(archive_root=cli_workspace["archive_root"]) as poly:
            reviews = await poly.list_assertion_candidate_reviews(
                target_ref=session_ref,
                statuses=("accepted",),
            )
            candidate_resolution = await poly.resolve_ref(candidate_ref)
            result_resolution = await poly.resolve_ref(resulting_ref)
            return reviews, candidate_resolution, result_resolution

    reviews, candidate_resolution, result_resolution = asyncio.run(verify_surfaces())
    assert [review.candidate_ref for review in reviews.items] == [candidate_ref]
    review = reviews.items[0]
    assert review.latest_judgment is not None
    assert review.latest_judgment.decision == "accept"
    assert review.latest_judgment.resulting_assertion_ref == resulting_ref
    assert candidate_resolution.resolved is True
    assert result_resolution.resolved is True

    with sqlite3.connect(cli_workspace["archive_root"] / "user.db") as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT assertion_id, kind, status, evidence_refs_json,
                   context_policy_json, supersedes_json, value_json
            FROM assertions
            WHERE assertion_id = ?
               OR target_ref = ?
               OR assertion_id = ?
            ORDER BY kind, assertion_id
            """,
            (first["assertion_id"], candidate_ref, resulting_ref.removeprefix("assertion:")),
        ).fetchall()

    candidate_rows = [row for row in rows if row["assertion_id"] == first["assertion_id"]]
    judgment_rows = [row for row in rows if row["kind"] == "judgment"]
    result_rows = [row for row in rows if row["assertion_id"] == resulting_ref.removeprefix("assertion:")]
    assert len(candidate_rows) == 1
    assert candidate_rows[0]["status"] == "accepted"
    assert len(judgment_rows) == 1
    assert len(result_rows) == 1
    result = result_rows[0]
    assert result["kind"] == "lesson"
    assert result["status"] == "active"
    assert json.loads(result["context_policy_json"]) == {"inject": False}
    assert json.loads(result["supersedes_json"]) == [candidate_ref]
    assert json.loads(result["evidence_refs_json"]) == [session_ref, candidate_ref]
    judgment_value = json.loads(judgment_rows[0]["value_json"])
    assert judgment_value["candidate_ref"] == candidate_ref
    assert judgment_value["resulting_assertion_ref"] == resulting_ref
    assert judgment_value["inject_authorized"] is False

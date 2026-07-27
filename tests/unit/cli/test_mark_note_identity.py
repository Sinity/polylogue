"""Regression coverage for `mark --note` update-in-place identity (polylogue-tilk).

Prior behavior minted a fresh random/content-hash id on every `mark --note`
call (`uuid.uuid4()`, later a `sha256(session_id, note_text)` digest), so two
notes on the same session -- even byte-identical repeats, and definitely an
edited note -- produced two distinct rows instead of one row being updated in
place, despite the command's own help text promising "Add or update a note
annotation". See `.agent/scratch/dogfood-2/investigations/write-path-correctness.md`
(finding F-028) for the original empirical evidence.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from polylogue.cli import cli
from tests.infra.storage_records import SessionBuilder

WorkspacePaths = dict[str, Path]


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


def _annotation_rows(user_db: Path) -> list[tuple[str, str, str]]:
    with sqlite3.connect(user_db) as conn:
        rows = conn.execute(
            "SELECT assertion_id, target_ref, body_text FROM assertions WHERE kind = 'annotation'"
        ).fetchall()
    return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]


def test_mark_note_twice_with_same_text_is_a_true_no_op(cli_workspace: WorkspacePaths, cli_runner: CliRunner) -> None:
    db_path = cli_workspace["db_path"]
    builder = SessionBuilder(db_path, "conv-note-repeat").provider("claude-code")
    builder.save()
    session_id = builder.native_session_id()

    for _ in range(2):
        result = cli_runner.invoke(
            cli,
            ["--plain", "find", f"id:{session_id}", "then", "mark", "--note", "key insight"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

    rows = _annotation_rows(cli_workspace["archive_root"] / "user.db")
    assert len(rows) == 1
    assert rows[0][2] == "key insight"


def test_mark_note_edit_updates_the_same_row_instead_of_forking(
    cli_workspace: WorkspacePaths, cli_runner: CliRunner
) -> None:
    """The core regression: editing note text must update in place, not insert."""
    db_path = cli_workspace["db_path"]
    builder = SessionBuilder(db_path, "conv-note-edit").provider("claude-code")
    builder.save()
    session_id = builder.native_session_id()

    first = cli_runner.invoke(
        cli,
        ["--plain", "find", f"id:{session_id}", "then", "mark", "--note", "first draft"],
        catch_exceptions=False,
    )
    assert first.exit_code == 0, first.output

    user_db = cli_workspace["archive_root"] / "user.db"
    rows_after_first = _annotation_rows(user_db)
    assert len(rows_after_first) == 1
    assert rows_after_first[0][2] == "first draft"
    first_assertion_id = rows_after_first[0][0]

    second = cli_runner.invoke(
        cli,
        ["--plain", "find", f"id:{session_id}", "then", "mark", "--note", "revised, corrected wording"],
        catch_exceptions=False,
    )
    assert second.exit_code == 0, second.output

    rows_after_second = _annotation_rows(user_db)
    # Anti-vacuity: a naive content-hash-default fix (or no fix at all) makes
    # this len(rows_after_second) == 2 -- two distinct annotation rows for one
    # session's "note" concept -- fail here.
    assert len(rows_after_second) == 1
    assert rows_after_second[0][0] == first_assertion_id
    assert rows_after_second[0][2] == "revised, corrected wording"
    assert rows_after_second[0][1] == rows_after_first[0][1]
